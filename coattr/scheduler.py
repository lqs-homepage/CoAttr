"""Decoupled Mask-Trie + Full-Width Pure-Suffix Queue Scheduler (P-18 Stage 3.1).

Stage 2.4 reanalysis of Stage 2.2 B200 runs uncovered why naive Horton-Strahler
``trie`` scheduling was 1.7x-2.85x slower in wall-clock time than ``prefix``
on multi-question documents: ``trie`` put q0 and q1..q_{J-1} of the same mask
into adjacent levels, mixing ~3,600-token uncached document prefills with
~16-token cached suffix-only requests in 87-103 mixed batches.

``mask_trie_fanout_batches`` fixes this root cause while staying 100% fair and
design-agnostic (shared equally by CUBE, ordered cosets, LOO, and chains):
1. Builds the zero-duplication Horton-Strahler prefix tree strictly over the
   shared document prefixes of distinct masks.
2. Dispatches homogeneous document-prefill waves (width-bounded tiles of q0).
3. Enqueues remaining question+answer suffixes of warmed masks into a pure
   suffix queue, flushing full-width suffix-only batches immediately (zero
   mixed heavy+suffix batches, bounded KV cache working set).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterator, Mapping, Sequence


RequestRow = tuple[Sequence[int], int, int]  # (prompt_tokens, mask, question_idx)


def _lcp_len(a: Sequence[int], b: Sequence[int]) -> int:
    limit = min(len(a), len(b))
    for i in range(limit):
        if a[i] != b[i]:
            return i
    return limit


def _shared_doc_prefix(reqs: Sequence[RequestRow]) -> tuple[int, ...]:
    """Longest common token prefix (excluding final token) across questions of a mask."""
    p0 = reqs[0][0][:-1]
    plen = len(p0)
    for req in reqs[1:]:
        pr = req[0][:-1]
        limit = min(plen, len(pr))
        idx = 0
        while idx < limit and p0[idx] == pr[idx]:
            idx += 1
        plen = idx
    return tuple(p0[:plen])


def mask_trie_fanout_batches(
    by_mask: Mapping[int, Sequence[RequestRow]],
    width: int = 32,
    *,
    max_live_masks: int | None = None,
) -> Iterator[list[RequestRow]]:
    """Zero-duplication mask-trie prefill waves decoupled from full-width suffix waves.

    If ``max_live_masks`` is set, pending suffixes are flushed early whenever the
    number of distinct masks with un-drained suffixes reaches ``max_live_masks``,
    preventing RadixCache eviction under tight GPU memory budgets.
    """
    if width < 1:
        raise ValueError("batch width must be positive")
    if not by_mask:
        return

    items: list[tuple[tuple[int, ...], int, list[RequestRow]]] = []
    for mask, reqs in by_mask.items():
        if not reqs:
            continue
        reqs_sorted = sorted(reqs, key=lambda r: r[2])
        items.append((_shared_doc_prefix(reqs_sorted), mask, reqs_sorted))
    if not items:
        return

    items.sort(key=lambda x: (x[0], x[1]))
    lcp = [_lcp_len(items[i][0], items[i + 1][0]) for i in range(len(items) - 1)]

    def build(lo: int, hi: int):
        if hi == lo + 1:
            return (1, 1, lo, ())
        depth = min(lcp[lo : hi - 1])
        bounds = [lo, *[i + 1 for i in range(lo, hi - 1) if lcp[i] == depth], hi]
        children = [build(a, b) for a, b in zip(bounds, bounds[1:])]
        children.sort(key=lambda x: (-x[0], -x[1], x[2]))
        height = max(children[0][0], 1 + children[1][0])
        return (height, hi - lo, lo, children)

    rounds: dict[int, list[tuple[int, int]]] = defaultdict(list)

    def assign(node, level: int, parent_depth: int) -> None:
        _, _, idx, children = node
        if not children:
            # Estimated uncached extension length for homogeneous prefill packing
            est_ext = len(items[idx][2][0][0]) - parent_depth
            rounds[level].append((est_ext, idx))
            return
        # Node's shared prefix depth is the min LCP across its child boundaries
        child_indices = [c[2] for c in children]
        lo_c, hi_c = min(child_indices), max(child_indices)
        node_depth = min(lcp[lo_c:hi_c]) if hi_c > lo_c else parent_depth
        for i, child in enumerate(children):
            assign(child, level + int(i != 0), parent_depth if i == 0 else node_depth)

    root_bounds = [0, *[j + 1 for j, v in enumerate(lcp) if v == 0], len(items)]
    for lo, hi in zip(root_bounds, root_bounds[1:]):
        assign(build(lo, hi), 0, 0)

    suffix_queue: list[RequestRow] = []
    pending_masks: set[int] = set()

    def drain_suffix_queue(*, flush_all: bool = False) -> Iterator[list[RequestRow]]:
        while len(suffix_queue) >= width or (flush_all and suffix_queue):
            batch = suffix_queue[:width]
            del suffix_queue[:width]
            yield batch
        if not suffix_queue:
            pending_masks.clear()

    for level in sorted(rounds):
        # Pack masks with similar uncached extension lengths into the same prefill wave
        ordered = [idx for _, idx in sorted(rounds[level], key=lambda pair: (-pair[0], pair[1]))]
        for pos in range(0, len(ordered), width):
            tile_items = [items[i] for i in ordered[pos : pos + width]]
            # 1. Homogeneous prefill wave (q0 for every mask in this trie-level tile)
            yield [reqs[0] for _, _, reqs in tile_items]
            # 2. Enqueue remaining question suffixes in question-major order for locality
            max_q = max(len(reqs) for _, _, reqs in tile_items)
            for q_idx in range(1, max_q):
                for _, mask, reqs in tile_items:
                    if q_idx < len(reqs):
                        suffix_queue.append(reqs[q_idx])
                        pending_masks.add(mask)
            # 3. Flush full-width pure-suffix batches (or all if live-mask cap reached)
            force_flush = max_live_masks is not None and len(pending_masks) >= max_live_masks
            yield from drain_suffix_queue(flush_all=force_flush)

    yield from drain_suffix_queue(flush_all=True)


def _grouping_cut(masks: Sequence[int], n: int) -> int:
    target = max(1, len(masks) // 2)
    best = None
    for c in range(1, n + 1):
        prefix = (1 << c) - 1
        count = len({m & prefix for m in masks})
        key = (abs(count - target), -c)
        if best is None or key < best[0]:
            best = (key, c)
    return best[1]


def _legacy_trie_batches(by_mask: Mapping[int, Sequence[RequestRow]], width: int) -> Iterator[list[RequestRow]]:
    items = [(tuple(row[0][:-1]), row) for reqs in by_mask.values() for row in reqs]
    if not items:
        return
    items.sort(key=lambda x: x[0])
    lcp = [_lcp_len(items[i][0], items[i + 1][0]) for i in range(len(items) - 1)]

    def build(lo: int, hi: int):
        if hi == lo + 1:
            return (1, 1, lo, ())
        depth = min(lcp[lo : hi - 1])
        bounds = [lo, *[i + 1 for i in range(lo, hi - 1) if lcp[i] == depth], hi]
        children = [build(a, b) for a, b in zip(bounds, bounds[1:])]
        children.sort(key=lambda x: (-x[0], -x[1], x[2]))
        return (max(children[0][0], 1 + children[1][0]), hi - lo, lo, children)

    rounds: dict[int, list[int]] = defaultdict(list)

    def assign(node, level: int) -> None:
        _, _, idx, children = node
        if not children:
            rounds[level].append(idx)
            return
        for i, child in enumerate(children):
            assign(child, level + int(i != 0))

    bounds = [0, *[j + 1 for j, v in enumerate(lcp) if v == 0], len(items)]
    for lo, hi in zip(bounds, bounds[1:]):
        assign(build(lo, hi), 0)
    for level in sorted(rounds):
        ordered = sorted(rounds[level])
        for pos in range(0, len(ordered), width):
            yield [items[i][1] for i in ordered[pos : pos + width]]


def schedule_batches(
    by_mask: Mapping[int, Sequence[RequestRow]],
    layout: str,
    n: int,
    width: int = 32,
    *,
    max_live_masks: int | None = None,
) -> Iterator[list[RequestRow]]:
    """Unified fair scheduler dispatcher supporting all Stage 2 & Stage 3 layouts."""
    if layout == "mask_trie_fanout":
        yield from mask_trie_fanout_batches(by_mask, width, max_live_masks=max_live_masks)
        return
    if layout == "trie":
        yield from _legacy_trie_batches(by_mask, width)
        return
    if layout == "lex":
        order = sorted((x for batch in by_mask.values() for x in batch), key=lambda x: tuple(x[0]))
        for pos in range(0, len(order), width):
            yield order[pos : pos + width]
        return
    if layout not in ("questions", "prefix"):
        raise ValueError(f"unsupported layout: {layout}")
    cut = n if layout == "questions" else _grouping_cut(list(by_mask), n)
    mask_prefix = (1 << cut) - 1
    classes: dict[int, list[int]] = defaultdict(list)
    for m in by_mask:
        classes[m & mask_prefix].append(m)
    groups: list[list[RequestRow]] = []
    for cls in classes.values():
        cls.sort(key=lambda m: tuple(by_mask[m][0][0]))
        for pos in range(0, len(cls), 2):
            group = [row for m in cls[pos : pos + 2] for row in by_mask[m]]
            groups.append(group)
    groups.sort(key=lambda group: tuple(group[0][0]))
    for pos in range(0, len(groups), width):
        tile = groups[pos : pos + width]
        for wave in range(max(len(g) for g in tile)):
            yield [g[wave] for g in tile if wave < len(g)]


@dataclass(frozen=True)
class ScheduleCostProfile:
    """Deterministic pre-GPU physical profile of a scheduled request batch stream."""

    layout: str
    requests: int
    batches: int
    heavy_batches: int
    mixed_batches: int
    mixed_prefill_suffix_batches: int
    pure_suffix_batches: int
    total_input_tokens: int
    total_uncached_tokens: int
    total_cached_tokens: int
    sum_max_uncached_per_batch: int
    distinct_prefix_nodes: int


def simulate_schedule_cost(
    by_mask: Mapping[int, Sequence[RequestRow]],
    layout: str,
    n: int,
    width: int = 32,
    *,
    suffix_token_threshold: int = 128,
    heavy_token_threshold: int = 512,
    max_live_masks: int | None = None,
) -> ScheduleCostProfile:
    """Simulate exact cross-round prefix KV visibility for any schedule layout."""
    batches = list(schedule_batches(by_mask, layout, n, width, max_live_masks=max_live_masks))
    # Trie over token prefixes stored as dict[node_id -> dict[token -> child_id]]
    children: list[dict[int, int]] = [{}]
    total_in = total_unc = total_cached = sum_max_unc = 0
    heavy = mixed = mixed_ps = suffix = 0
    total_reqs = 0
    seen_masks: set[int] = set()

    for batch in batches:
        total_reqs += len(batch)
        batch_uncs: list[int] = []
        to_insert: list[Sequence[int]] = []
        has_new_mask = False
        has_cached_mask_suffix = False
        batch_masks: set[int] = set()
        for toks, mask, _ in batch:
            total_in += len(toks)
            if mask in seen_masks:
                has_cached_mask_suffix = True
            else:
                has_new_mask = True
            batch_masks.add(mask)
            node = 0
            hit = 0
            for tk in toks:
                nxt = children[node].get(tk)
                if nxt is None:
                    break
                node = nxt
                hit += 1
            unc = len(toks) - hit
            batch_uncs.append(unc)
            total_cached += hit
            total_unc += unc
            if unc > 0:
                to_insert.append(toks)
        if has_new_mask and has_cached_mask_suffix:
            mixed_ps += 1
        seen_masks.update(batch_masks)
        # Commit prefixes at round boundary (cross-round KV visibility)
        for toks in to_insert:
            node = 0
            for tk in toks:
                nxt = children[node].get(tk)
                if nxt is None:
                    nxt = len(children)
                    children[node][tk] = nxt
                    children.append({})
                node = nxt
        if batch_uncs:
            mx = max(batch_uncs)
            mn = min(batch_uncs)
            sum_max_unc += mx
            if mx <= suffix_token_threshold:
                suffix += 1
            elif mn <= suffix_token_threshold and mx >= heavy_token_threshold:
                mixed += 1
                heavy += 1
            else:
                heavy += 1

    return ScheduleCostProfile(
        layout=layout,
        requests=total_reqs,
        batches=len(batches),
        heavy_batches=heavy,
        mixed_batches=mixed,
        mixed_prefill_suffix_batches=mixed_ps,
        pure_suffix_batches=suffix,
        total_input_tokens=total_in,
        total_uncached_tokens=total_unc,
        total_cached_tokens=total_cached,
        sum_max_uncached_per_batch=sum_max_unc,
        distinct_prefix_nodes=len(children) - 1,
    )
