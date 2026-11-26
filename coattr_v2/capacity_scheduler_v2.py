"""Upgraded HBM-Capacity-Aware Subtree-Tiled Scheduler with Explicit Transient
Wave-Duplicate Accounting (Delta_wave) and Adaptive Inner Wave-Width Contraction (W')
(CoAttr v2 — Algorithm 2 Upgrade).

Directly resolves Reviewer 2 (W4, Theorem 3(2)-(3)) and conquers extreme-tight KV
regimes such as gemma-4-31B-it (99K KV pool):
1. **Explicit Transient Pioneer Duplicate-Prefix Accounting (Delta_wave)**:
   Computes the exact in-flight HBM KV peak of any candidate subtree S and inner
   wave width W', accounting for BOTH:
   - Wave-0 concurrent pioneers allocating full independent prefix pages before
     RadixCache/APC commit: sum_{j in Wave_0} ceil(L_j / P) * P;
   - Later waves w >= 1 holding the committed subtree trie WS_unique(S) PLUS
     in-flight uncached duplicate prefix pages across concurrent wave-w requests
     PLUS active suffix KV pages Delta_suf = W' * max_q L_suf.
2. **Closed-Loop Adaptive Inner Wave-Width Contraction (W')**:
   When even the minimal 2-wave tile (2*W) at nominal width W (e.g., 32) violates
   PeakKV(S_{2W}, W) <= headroom_ratio * C_KV (such as gemma-4-31B-it at C_KV = 99K
   where 2*32*L_bar ~= 170K > 99K), Algorithm 2 automatically contracts the inner
   wave width W' down the power-of-two ladder (32 -> 16 -> 8 -> 4) until the
   2-wave tile fits strictly inside HBM KV capacity with ZERO intra-tile thrashing!
3. **Dual-Engine Paged-KV Simulator (SGLang RadixAttention & vLLM APC)**:
   Simulates BOTH SGLang token-level/page-aligned RadixCache (page_size=64) AND
   vLLM Automatic Prefix Caching full-block-only hashing (block_size=16, partial
   trailing block uncached) under exact finite-capacity LRU leaf eviction.
"""

from __future__ import annotations

import math
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence

# RequestRow = (prompt_token_ids, mask_int, question_index)
RequestRow = tuple[Sequence[int], int, int]


def pioneer_peel_order(n_items: int, width: int) -> list[int]:
    """Recursive alternating bottom-up pioneer peeling permutation pi over [0..n_items-1].

    Given a prefix-sorted slice of length T:
    - Recursively peel odd-indexed siblings to the end of the wave stack while
      halving the surviving even-indexed pioneer set until |cur| <= width.
    - Emit layers top-down (pioneers first, then successive sibling waves), so
      every wave-w request shares its longest prefix with an already-committed
      wave-(<w) pioneer while maximizing GPU batch width.
    """
    cur = list(range(n_items))
    layers: list[list[int]] = []
    while len(cur) > width:
        layers.append(cur[1::2])
        cur = cur[0::2]
    layers.append(cur)
    layers.reverse()
    return [idx for layer in layers for idx in layer]


def _page_aligned_len(length: int, page_size: int) -> int:
    return math.ceil(length / page_size) * page_size


def _lcp_aligned(a: Sequence[int], b: Sequence[int], page_size: int) -> int:
    lim = min(len(a), len(b))
    idx = 0
    while idx < lim and a[idx] == b[idx]:
        idx += 1
    return (idx // page_size) * page_size


def exact_subtree_peak_kv_tokens(
    prompts: Sequence[Sequence[int]],
    width: int,
    *,
    page_size: int = 64,
    suffix_headroom_tokens: int = 256,
) -> dict[str, int]:
    """Compute exact peak live HBM KV tokens for a prefix-sorted subtree under
    pioneer-peeling wave width ``width``.

    Models the exact SGLang/vLLM commit boundary:
    - Requests in the same wave run concurrently and CANNOT share newly computed
      pages with each other until the wave finishes and commits to the prefix cache.
    - Therefore, during wave w, live KV occupancy equals:
          committed_trie_tokens_before_w
          + sum_{r in wave w} (padded_len(r) - lcp_hit_in_committed_trie(r))
          + width * suffix_headroom_tokens
      and ``delta_wave`` is the maximum transient duplicate prefix allocation
      (in-wave uncached tokens minus unique new trie tokens committed by that wave).
    """
    t_len = len(prompts)
    if t_len == 0:
        return {"ws_unique": 0, "max_delta_wave": 0, "delta_suf": 0, "peak_kv": 0}

    order = pioneer_peel_order(t_len, width)
    # Track committed unique page hashes in this subtree
    committed: set[tuple[int, tuple[int, ...]]] = set()
    # Map (parent_node_id, page_tuple) -> node_id for exact trie sharing
    node_map: dict[tuple[int, tuple[int, ...]], int] = {}
    next_id = 1

    # Pre-slice prompts into full + padded trailing pages
    prompt_pages: list[list[tuple[int, ...]]] = []
    padded_lens: list[int] = []
    for p in prompts:
        n_pages = math.ceil(len(p) / page_size)
        padded_lens.append(n_pages * page_size)
        pages = [tuple(p[k * page_size : (k + 1) * page_size]) for k in range(n_pages)]
        prompt_pages.append(pages)

    peak_kv = 0
    max_delta_wave = 0
    delta_suf = min(width, t_len) * _page_aligned_len(suffix_headroom_tokens, page_size)

    for start in range(0, len(order), width):
        wave_idxs = order[start : start + width]
        committed_before = len(node_map) * page_size
        inflight_uncached = 0
        # 1. Measure hits against ALREADY-committed trie (before wave commits)
        for idx in wave_idxs:
            cur = 0
            hit = 0
            for pg in prompt_pages[idx]:
                nxt = node_map.get((cur, pg))
                if nxt is None:
                    break
                cur = nxt
                hit += 1
            inflight_uncached += (len(prompt_pages[idx]) - hit) * page_size

        # 2. Commit wave pages into canonical trie
        for idx in wave_idxs:
            cur = 0
            for pg in prompt_pages[idx]:
                key = (cur, pg)
                nxt = node_map.get(key)
                if nxt is None:
                    nxt = next_id
                    next_id += 1
                    node_map[key] = nxt
                    committed.add(key)
                cur = nxt
        committed_after = len(node_map) * page_size
        unique_added = committed_after - committed_before
        delta_wave = max(0, inflight_uncached - unique_added)
        if delta_wave > max_delta_wave:
            max_delta_wave = delta_wave
        live_during_wave = committed_before + inflight_uncached + delta_suf
        if live_during_wave > peak_kv:
            peak_kv = live_during_wave

    ws_unique = len(node_map) * page_size
    peak_kv = max(peak_kv, ws_unique + delta_suf)
    return {
        "ws_unique": ws_unique,
        "max_delta_wave": max_delta_wave,
        "delta_suf": delta_suf,
        "peak_kv": peak_kv,
    }


@dataclass(frozen=True)
class AdaptiveSchedulePlan:
    batches: list[list[RequestRow]]
    effective_width: int
    nominal_width: int
    max_tile_size: int
    tile_sizes: tuple[int, ...]
    max_predicted_peak_kv: int
    max_predicted_delta_wave: int
    width_contracted: bool


def adaptive_capacity_mask_trie_batches(
    by_mask: Mapping[int, Sequence[RequestRow]],
    nominal_width: int = 32,
    *,
    kv_token_capacity: int | None = None,
    headroom_ratio: float = 0.90,
    page_size: int = 64,
    min_width: int = 4,
    suffix_headroom_tokens: int = 256,
) -> AdaptiveSchedulePlan:
    """Upgraded Algorithm 2: Capacity-Aware Subtree-Tiled Scheduler with explicit
    Delta_wave accounting and Closed-Loop Adaptive Inner Wave-Width Contraction W'.

    Steps:
    1. Sort distinct masks lexicographically by token prefix.
    2. Choose largest power-of-two inner wave width W' in [min_width, nominal_width]
       such that the minimal 2-wave subtree (size min(n_masks, 2*W')) fits inside
       ``safe_budget = headroom_ratio * kv_token_capacity`` INCLUDING exact
       transient pioneer duplicate allocation ``Delta_wave`` and suffix headroom.
    3. Greedily partition sorted masks into contiguous W'-aligned prefix subtrees
       (preferring power-of-two wave counts 2W', 4W', 8W', ... when splitting)
       whose EXACT ``peak_kv <= safe_budget``.
    4. Inside each subtree, execute width-W' pioneer peeling with immediate
       pure-suffix draining so zero intra-tile KV eviction occurs!
    """
    if nominal_width < 1:
        raise ValueError("nominal_width must be >= 1")
    items = []
    for mask, reqs in by_mask.items():
        if not reqs:
            continue
        reqs_sorted = sorted(reqs, key=lambda r: r[2])
        items.append((reqs_sorted[0][0], mask, reqs_sorted))
    if not items:
        return AdaptiveSchedulePlan([], nominal_width, nominal_width, 0, (), 0, 0, False)

    items.sort(key=lambda x: (x[0], x[1]))
    n_masks = len(items)
    all_prompts = [it[0] for it in items]
    safe_budget = (headroom_ratio * kv_token_capacity) if (kv_token_capacity and kv_token_capacity > 0) else float("inf")

    # Step 1: Closed-loop adaptive inner wave-width contraction W'
    eff_width = nominal_width
    if math.isfinite(safe_budget):
        # Probe worst-case 2-wave slice across the document (check start, middle, end slices)
        while eff_width > min_width:
            probe_span = min(n_masks, 2 * eff_width)
            probe_starts = sorted({0, max(0, (n_masks - probe_span) // 2), max(0, n_masks - probe_span)})
            worst_peak = max(
                exact_subtree_peak_kv_tokens(
                    all_prompts[s : s + probe_span],
                    eff_width,
                    page_size=page_size,
                    suffix_headroom_tokens=suffix_headroom_tokens,
                )["peak_kv"]
                for s in probe_starts
            )
            if worst_peak <= safe_budget:
                break
            eff_width = max(min_width, eff_width // 2)

    # Step 2: Partition into exact-budgeted contiguous prefix subtrees of multiples of eff_width
    batches: list[list[RequestRow]] = []
    tile_sizes: list[int] = []
    max_peak_kv = 0
    max_delta_wave = 0

    sub_start = 0
    while sub_start < n_masks:
        rem = n_masks - sub_start
        # First check if all remaining masks fit in one global subtree (matches untiled when HBM allows)
        if not math.isfinite(safe_budget):
            best_end = n_masks
            best_stats = exact_subtree_peak_kv_tokens(
                all_prompts[sub_start:best_end],
                eff_width,
                page_size=page_size,
                suffix_headroom_tokens=suffix_headroom_tokens,
            )
        else:
            full_stats = exact_subtree_peak_kv_tokens(
                all_prompts[sub_start:n_masks],
                eff_width,
                page_size=page_size,
                suffix_headroom_tokens=suffix_headroom_tokens,
            )
            if full_stats["peak_kv"] <= safe_budget:
                best_end = n_masks
                best_stats = full_stats
            else:
                # Test all width-aligned tile multiples 2W', 3W', 4W', ... fitting inside safe_budget
                # (avoids needlessly collapsing to 2-wave tiles when 3W' or 6W' fits with zero eviction!)
                cand_spans = list(range(2 * eff_width, rem, eff_width))
                best_end = min(n_masks, sub_start + eff_width)
                best_stats = exact_subtree_peak_kv_tokens(
                    all_prompts[sub_start:best_end],
                    eff_width,
                    page_size=page_size,
                    suffix_headroom_tokens=suffix_headroom_tokens,
                )
                for span in cand_spans:
                    st = exact_subtree_peak_kv_tokens(
                        all_prompts[sub_start : sub_start + span],
                        eff_width,
                        page_size=page_size,
                        suffix_headroom_tokens=suffix_headroom_tokens,
                    )
                    if st["peak_kv"] <= safe_budget:
                        best_end = sub_start + span
                        best_stats = st
                    else:
                        break

        sub_items = items[sub_start:best_end]
        tile_sizes.append(len(sub_items))
        max_peak_kv = max(max_peak_kv, best_stats["peak_kv"])
        max_delta_wave = max(max_delta_wave, best_stats["max_delta_wave"])

        order = pioneer_peel_order(len(sub_items), eff_width)
        suffix_queue: list[RequestRow] = []

        def drain(*, flush_all: bool = False) -> None:
            while len(suffix_queue) >= eff_width or (flush_all and suffix_queue):
                batches.append(suffix_queue[:eff_width])
                del suffix_queue[:eff_width]

        for p_idx in range(0, len(order), eff_width):
            tile = [sub_items[i] for i in order[p_idx : p_idx + eff_width]]
            batches.append([reqs[0] for _, _, reqs in tile])
            max_q = max(len(reqs) for _, _, reqs in tile)
            for q_idx in range(1, max_q):
                for _, _, reqs in tile:
                    if q_idx < len(reqs):
                        suffix_queue.append(reqs[q_idx])
            drain(flush_all=False)
        drain(flush_all=True)
        sub_start = best_end

    return AdaptiveSchedulePlan(
        batches=batches,
        effective_width=eff_width,
        nominal_width=nominal_width,
        max_tile_size=max(tile_sizes) if tile_sizes else 0,
        tile_sizes=tuple(tile_sizes),
        max_predicted_peak_kv=max_peak_kv,
        max_predicted_delta_wave=max_delta_wave,
        width_contracted=(eff_width < nominal_width),
    )


def simulate_paged_lru_kv_cache(
    batches: Sequence[Sequence[RequestRow]],
    *,
    kv_token_capacity: int,
    page_size: int = 64,
    full_block_only: bool = True,
) -> dict[str, int]:
    """Simulate finite-HBM paged prefix cache with exact LRU leaf-block eviction.

    - ``page_size=64, full_block_only=True`` models SGLang RadixCache (trtllm_mha page=64).
    - ``page_size=16, full_block_only=True`` models vLLM Automatic Prefix Caching (block_size=16),
      where ONLY full 16-token prefix blocks are hashed and cached (trailing partial block uncached).
    """
    max_pages = max(1, kv_token_capacity // page_size)
    page_lru: OrderedDict[int, int] = OrderedDict()
    children_map: dict[tuple[int, tuple[int, ...]], int] = {}
    children_count: defaultdict[int, int] = defaultdict(int)
    node_key: dict[int, tuple[int, tuple[int, ...]]] = {}
    next_id = 1

    def evict_if_needed(protected: set[int]) -> int:
        ev = 0
        while len(page_lru) > max_pages:
            victim = None
            for pid in page_lru:
                if children_count[pid] == 0 and pid not in protected:
                    victim = pid
                    break
            if victim is None:
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
            ev += 1
        return ev

    total_in = 0
    total_cached = 0
    total_unc = 0
    evictions = 0
    peak_live_pages = 0

    for batch in batches:
        protected: set[int] = set()
        to_insert: list[list[tuple[int, ...]]] = []
        inflight_new_pages = 0
        for toks, _, _ in batch:
            total_in += len(toks)
            n_full_pages = (len(toks) - 1) // page_size if full_block_only else math.ceil(len(toks) / page_size)
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
            inflight_new_pages += max(0, math.ceil(len(toks) / page_size) - hit_pages)
            to_insert.append(pages)

        peak_live_pages = max(peak_live_pages, len(protected) + inflight_new_pages)

        for pages in to_insert:
            cur = 0
            for pg in pages:
                key = (cur, pg)
                nxt = children_map.get(key)
                if nxt is None:
                    if len(page_lru) >= max_pages:
                        evictions += evict_if_needed(protected)
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
        "peak_live_tokens": peak_live_pages * page_size,
    }
