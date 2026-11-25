# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy==2.2.6"]
# ///
"""HBM-Capacity-Aware Subtree-Tiled Mask-Trie Scheduler (Stage 3.2 Discovery).

Live Qwen3-32B B200 experiments (Job 152, max_total_num_tokens = 273,472) revealed
a sharp physical phase transition absent on Qwen3-4B (875,648 KV tokens):
- When the total unique prefix working set fits in HBM KV cache (all Qwen3-4B
  docs and Qwen3-32B B=128 docs), global width-aligned pioneer peeling
  (``merged_mask_trie``) beats ``prefix`` (17/18 on 4B, and 8.80s vs 9.15s on
  32B QuALITY-20027).
- When the 256-mask unique prefix working set (~580K tokens) EXCEEDS HBM KV
  capacity (Qwen3-32B 273K), global pioneer peeling spreads Wave-0 pioneers across
  all 256 masks, so Wave-0 pioneers get LRU-evicted from RadixCache before
  Wave-4..7 followers arrive (+14% uncached-token rebound vs ``prefix``).

``capacity_aware_merged_mask_trie_batches`` solves this automatically from the
engine's reported ``max_total_num_tokens`` (zero manual tuning):
1. Estimates peak unique prefix tokens per mask directly from prompt lengths.
2. Computes safe contiguous prefix-subtree tile size ``T_tile`` (a multiple of
   batch width ``W``, >= ``2*W``) whose working set stays below
   ``headroom_ratio * kv_token_capacity``.
3. Sorts distinct masks by shared document prefix and partitions them into
   contiguous prefix-tree subtrees of size ``T_tile``.
4. Runs width-aligned pioneer peeling + immediate pure-suffix draining INSIDE
   each subtree before advancing to the next subtree (zero cross-subtree KV
   eviction, exact hardware-minimum ``ceil(M/W)`` heavy batches, zero mixed
   prefill+suffix batches).
"""

from __future__ import annotations

import gzip
import json
import math
import random
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import designs  # noqa: E402
from integration import draw_design_rows  # noqa: E402
from scheduler import RequestRow, schedule_batches  # noqa: E402


def _pioneer_peel_indices(n_items: int, width: int) -> list[int]:
    """Bottom-up alternating pioneer peeling order over a contiguous prefix-sorted slice."""
    cur = list(range(n_items))
    layers: list[list[int]] = []
    while len(cur) > width:
        layers.append(cur[1::2])
        cur = cur[0::2]
    layers.append(cur)
    layers.reverse()
    return [idx for layer in layers for idx in layer]


def capacity_aware_merged_mask_trie_batches(
    by_mask: Mapping[int, Sequence[RequestRow]],
    width: int = 32,
    *,
    kv_token_capacity: int | None = None,
    headroom_ratio: float = 0.85,
    page_size: int = 64,
) -> tuple[list[list[RequestRow]], int]:
    """Capacity-aware subtree-tiled pioneer-peeling mask-trie scheduler.

    Returns ``(batches, subtree_tile_size)``.
    When ``kv_token_capacity is None`` or the entire mask set fits within
    ``headroom_ratio * kv_token_capacity``, ``subtree_tile_size == len(by_mask)``
    (identical to global ``merged_mask_trie``).
    """
    if width < 1:
        raise ValueError("width must be positive")
    items = []
    for mask, reqs in by_mask.items():
        if not reqs:
            continue
        reqs_sorted = sorted(reqs, key=lambda r: r[2])
        items.append((reqs_sorted[0][0], mask, reqs_sorted))
    if not items:
        return [], 0
    items.sort(key=lambda x: (x[0], x[1]))
    n_masks = len(items)

    # Exact unique paged prefix-token footprint across consecutive prefix-sorted masks:
    # For each mask i, marginal new prefix pages = ceil((len(prompt_i) - lcp(i-1, i)) / page_size) * page_size.
    def _lcp_pages(a: Sequence[int], b: Sequence[int]) -> int:
        lim = min(len(a), len(b))
        idx = 0
        while idx < lim and a[idx] == b[idx]:
            idx += 1
        return (idx // page_size) * page_size

    marg_toks = [math.ceil(len(items[0][0]) / page_size) * page_size]
    for i in range(1, n_masks):
        shared = _lcp_pages(items[i - 1][0], items[i][0])
        marg_toks.append(max(page_size, math.ceil((len(items[i][0]) - shared) / page_size) * page_size))

    # Partition sorted masks into contiguous width-aligned subtrees whose EXACT unique prefix
    # footprint stays within ``headroom_ratio * kv_token_capacity`` (min 2*width for pioneer+sibling).
    batches: list[list[RequestRow]] = []
    tile_sizes: list[int] = []
    safe_budget = (headroom_ratio * kv_token_capacity) if (kv_token_capacity and kv_token_capacity > 0) else float("inf")

    sub_start = 0
    while sub_start < n_masks:
        # Greedily grow subtree in multiples of `width` (minimum 2*width if available)
        # while exact unique prefix footprint fits in safe_budget.
        best_end = min(n_masks, sub_start + 2 * width)
        # First mask in subtree pays full prompt length; subsequent pay exact marginal delta
        cum = math.ceil(len(items[sub_start][0]) / page_size) * page_size
        pos = sub_start + 1
        while pos < n_masks:
            cum += marg_toks[pos]
            pos += 1
            if (pos - sub_start) >= 2 * width and ((pos - sub_start) % width == 0 or pos == n_masks):
                if cum <= safe_budget:
                    best_end = pos
                else:
                    break
        # Prefer power-of-two waves (2W, 4W, 8W...) when smaller than remaining masks for balanced peeling
        span = best_end - sub_start
        if best_end < n_masks and span > 2 * width:
            k_waves = span // width
            pow2_waves = 1 << (k_waves.bit_length() - 1)
            best_end = sub_start + pow2_waves * width
        sub_items = items[sub_start:best_end]
        tile_sizes.append(len(sub_items))
        order = _pioneer_peel_indices(len(sub_items), width)
        suffix_queue: list[RequestRow] = []

        def drain(*, flush_all: bool = False) -> None:
            while len(suffix_queue) >= width or (flush_all and suffix_queue):
                batches.append(suffix_queue[:width])
                del suffix_queue[:width]

        for p_idx in range(0, len(order), width):
            tile = [sub_items[i] for i in order[p_idx : p_idx + width]]
            batches.append([reqs[0] for _, _, reqs in tile])
            max_q = max(len(reqs) for _, _, reqs in tile)
            for q_idx in range(1, max_q):
                for _, _, reqs in tile:
                    if q_idx < len(reqs):
                        suffix_queue.append(reqs[q_idx])
            drain(flush_all=False)
        drain(flush_all=True)
        sub_start = best_end

    return batches, (max(tile_sizes) if tile_sizes else 0)


def simulate_paged_lru_radix_cache(
    batches: Sequence[Sequence[RequestRow]],
    *,
    kv_token_capacity: int,
    page_size: int = 64,
) -> dict[str, int]:
    """Simulate SGLang paged RadixCache (page_size=64 for trtllm_mha) with exact LRU
    leaf-page eviction under finite HBM KV token budget ``kv_token_capacity``."""
    max_pages = max(1, kv_token_capacity // page_size)
    # Represent each cached page by its full cumulative prefix hash tuple (parent_id, page_tokens)
    # Track LRU order over cached page_ids; a parent page is refreshed whenever a child hits.
    page_lru: OrderedDict[int, int] = OrderedDict()  # page_id -> parent_id (0 = root)
    children_map: dict[tuple[int, tuple[int, ...]], int] = {}
    children_count: defaultdict[int, int] = defaultdict(int)
    node_key: dict[int, tuple[int, tuple[int, ...]]] = {}
    next_id = 1

    def evict_if_needed(protected: set[int]) -> None:
        while len(page_lru) > max_pages:
            # Evict oldest leaf page not in current active batch
            victim = None
            for pid in page_lru:
                if children_count[pid] == 0 and pid not in protected:
                    victim = pid
                    break
            if victim is None:
                # Fallback: evict oldest leaf even if protected (oversized single batch)
                for pid in page_lru:
                    if children_count[pid] == 0:
                        victim = pid
                        break
            if victim is None:
                break
            parent = page_lru.pop(victim)
            if parent != 0:
                children_count[parent] -= 1
            key = node_key.pop(victim)
            children_map.pop(key, None)

    total_in = 0
    total_cached = 0
    total_unc = 0
    evictions = 0

    for batch in batches:
        protected: set[int] = set()
        to_insert: list[list[tuple[int, ...]]] = []
        for toks, _, _ in batch:
            total_in += len(toks)
            n_full_pages = (len(toks) - 1) // page_size
            pages = [tuple(toks[p * page_size : (p + 1) * page_size]) for p in range(n_full_pages)]
            cur = 0
            hit_pages = 0
            for pg in pages:
                nxt = children_map.get((cur, pg))
                if nxt is None:
                    break
                cur = nxt
                hit_pages += 1
                page_lru.move_to_end(cur)
                protected.add(cur)
            hit_toks = hit_pages * page_size
            total_cached += hit_toks
            total_unc += len(toks) - hit_toks
            to_insert.append(pages)

        # Commit newly computed pages at round boundary with LRU eviction
        for pages in to_insert:
            cur = 0
            for pg in pages:
                key = (cur, pg)
                nxt = children_map.get(key)
                if nxt is None:
                    if len(page_lru) >= max_pages:
                        before = len(page_lru)
                        evict_if_needed(protected)
                        evictions += max(0, before - len(page_lru))
                    nxt = next_id
                    next_id += 1
                    children_map[key] = nxt
                    node_key[nxt] = key
                    page_lru[nxt] = cur
                    if cur != 0:
                        children_count[cur] += 1
                else:
                    page_lru.move_to_end(nxt)
                protected.add(nxt)
                cur = nxt

    return {
        "batches": len(batches),
        "input_tokens": total_in,
        "cached_tokens": total_cached,
        "uncached_tokens": total_unc,
        "page_evictions": evictions,
    }


def main() -> None:
    """Reproduce Qwen3-32B (273,472 KV tokens, page_size=64) eviction on QASPER-1810
    and QuALITY-22967 and verify capacity_aware_merged_mask_trie eliminates thrashing."""
    tok = json.loads(gzip.decompress((S22 / "tokenized.json.gz").read_bytes()))
    kv_32b = 273472
    kv_4b = 875648

    print("=== Finite-KV Paged RadixCache Simulation (Qwen3-32B: 273,472 tok, page=64) ===")
    for doc, nq in [("QuALITY-20027", 2), ("QASPER-1810.05241", 2), ("QASPER-1810.05241", 6), ("QuALITY-22967", 2)]:
        dt = tok[doc]
        parts = dt["parts"]
        n = len(parts)
        m = designs.budget(n) // 2
        head = dt["heads"][0]
        tails = [dt["tails"][0] + [200000 + q] + [210000 + j for j in range(30)] for q in range(nq)]

        for des, tg, g in [
            ("cube", None, 1),
            ("half", tuple(0 if i < n // 2 else 1 for i in range(n)), 2),
            ("four", tuple(j for j in range(4) for _ in range((j + 1) * n // 4 - j * n // 4)), 4),
        ]:
            rows, _, _ = draw_design_rows(n, m, random.Random(17), tags=tg, groups=g)
            bm: dict[int, list[RequestRow]] = defaultdict(list)
            for msk in dict.fromkeys(rows):
                body = [tk for i in range(n) if (msk >> i) & 1 for tk in parts[i]]
                for q in range(nq):
                    bm[msk].append((head + body + tails[q], msk, q))

            b_pre = list(schedule_batches(bm, "prefix", n, 32))
            b_glob, _ = capacity_aware_merged_mask_trie_batches(bm, 32, kv_token_capacity=None)
            b_cap32, tile32 = capacity_aware_merged_mask_trie_batches(bm, 32, kv_token_capacity=kv_32b)
            b_cap4, tile4 = capacity_aware_merged_mask_trie_batches(bm, 32, kv_token_capacity=kv_4b)

            s_pre = simulate_paged_lru_radix_cache(b_pre, kv_token_capacity=kv_32b, page_size=64)
            s_glob = simulate_paged_lru_radix_cache(b_glob, kv_token_capacity=kv_32b, page_size=64)
            s_cap = simulate_paged_lru_radix_cache(b_cap32, kv_token_capacity=kv_32b, page_size=64)
            print(
                f"  {doc:18s} nq={nq} {des:4s} (tile32={tile32:3d}, tile4={tile4:3d}) | "
                f"32B unc: pre={s_pre['uncached_tokens']:7d} glob_mrg={s_glob['uncached_tokens']:7d} "
                f"cap_mrg={s_cap['uncached_tokens']:7d} (cap/pre={s_cap['uncached_tokens']/s_pre['uncached_tokens']:.4f}x, "
                f"cap/glob={s_cap['uncached_tokens']/s_glob['uncached_tokens']:.4f}x)"
            )


if __name__ == "__main__":
    main()
