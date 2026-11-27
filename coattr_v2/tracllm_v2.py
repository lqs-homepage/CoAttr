"""Faithful TracLLM (USENIX Security'25) Hierarchical Perturbation Search Baseline.

Implements Wang et al.'s informed top-down binary/k-ary perturbation search tree
with top-K beam expansion and denoised LOO + Single-Text Contribution (STC) score
ensembling, strictly over the same black-box teacher-forced evaluation oracle
f(mask) as CoAttr and BLIP-Pruning.

Because TracLLM's tree search is inherently adaptive and sequential across levels
(level l+1 cannot be constructed until level l scores identify the top-K branches),
it naturally exposes both its strengths (logarithmic O(K log n) forward passes for
single-source long-context traceback) and its genuine structural limitations:
  - Under 2-copy / 3-copy redundant poisoning split across sibling subtrees,
    dropping one branch leaves the other copy active so LOO sees near-zero drop;
  - Per-question adaptive search trees cannot be amortized across multi-question
    batches cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class TracLLMResult:
    """Output of TracLLM hierarchical traceback for a single query."""

    scores: tuple[float, ...]  # length n, final chunk attribution scores
    top_chunks: tuple[int, ...]  # top-k ranked chunks
    total_evals: int  # total oracle evaluations used
    tree_depth: int  # number of sequential refinement rounds


def _chunks_to_mask(chunks: Sequence[int]) -> int:
    m = 0
    for c in chunks:
        m |= (1 << int(c))
    return m


def run_tracllm_single_query(
    eval_batch: Callable[[Sequence[int]], list[float]],
    n: int,
    *,
    top_k: int = 5,
    branch_factor: int = 2,
    use_stc_ensemble: bool = True,
    alpha_stc: float = 0.25,
    full_score: float | None = None,
    empty_score: float | None = None,
) -> TracLLMResult:
    """Run faithful TracLLM informed hierarchical search tree for one query.

    At each level:
      1. Split every active surviving node (size > 1) into ``branch_factor`` contiguous
         children.
      2. Batch-evaluate all child leave-one-out masks (Omega \\ child) — and when
         ``use_stc_ensemble=True``, also single-text-contribution masks (child alone) —
         in ONE batch per tree level (giving TracLLM maximum fair intra-level batching!).
      3. Score each child by denoised ensemble:
           score(child) = max(0, f(Omega) - f(Omega \\ child))
                          + alpha_stc * max(0, f(child) - f(empty))
         (following TracLLM's LOO + STC contribution ensembling).
      4. Retain only the top ``top_k`` highest-scoring nodes (plus any already-singleton
         nodes in the top-K pool) for the next refinement level until all surviving
         top-K nodes are individual chunks.
    """
    cache: dict[int, float] = {}
    total_evals = 0

    def query_many(masks: Sequence[int]) -> list[float]:
        nonlocal total_evals
        missing = list(dict.fromkeys(m for m in masks if m not in cache))
        if missing:
            vals = eval_batch(missing)
            for m, v in zip(missing, vals, strict=True):
                cache[m] = float(v)
            total_evals += len(missing)
        return [cache[m] for m in masks]

    full_mask = (1 << n) - 1
    init_need: list[int] = []
    if full_score is None:
        init_need.append(full_mask)
    else:
        cache[full_mask] = float(full_score)
    if use_stc_ensemble:
        if empty_score is None:
            init_need.append(0)
        else:
            cache[0] = float(empty_score)
    if init_need:
        query_many(init_need)

    f_full = cache[full_mask]
    f_empty = cache.get(0, 0.0)

    # Start from root node covering all n chunks
    active_nodes: list[tuple[int, ...]] = [tuple(range(n))]
    node_scores: dict[tuple[int, ...], float] = {active_nodes[0]: 0.0}
    chunk_scores = np.zeros(n, dtype=np.float64)
    depth = 0

    while any(len(nd) > 1 for nd in active_nodes):
        depth += 1
        children_to_eval: list[tuple[int, ...]] = []
        retained_singletons: list[tuple[int, ...]] = []

        for nd in active_nodes:
            if len(nd) == 1:
                retained_singletons.append(nd)
                continue
            b = min(branch_factor, len(nd))
            for j in range(b):
                lo = (j * len(nd)) // b
                hi = ((j + 1) * len(nd)) // b
                if hi > lo:
                    children_to_eval.append(nd[lo:hi])

        # Build all masks for this tree level and evaluate in one batch
        loo_masks = [full_mask ^ _chunks_to_mask(ch) for ch in children_to_eval]
        stc_masks = [_chunks_to_mask(ch) for ch in children_to_eval] if use_stc_ensemble else []
        query_many(loo_masks + stc_masks)

        for ch, m_loo in zip(children_to_eval, loo_masks, strict=True):
            loo_drop = f_full - cache[m_loo]
            if use_stc_ensemble:
                stc_gain = max(0.0, cache[_chunks_to_mask(ch)] - f_empty)
                sc = max(0.0, loo_drop) + alpha_stc * stc_gain
            else:
                sc = loo_drop
            node_scores[ch] = sc
            # Propagate normalized score estimate to constituent chunks (overwritten if refined)
            for idx in ch:
                chunk_scores[idx] = sc / len(ch)

        pool = retained_singletons + children_to_eval
        pool.sort(key=lambda nd: (-node_scores.get(nd, -1e18), len(nd), nd[0]))
        active_nodes = pool[: max(top_k, 1)]

        # Update exact leaf scores for any singletons in pool
        for nd in pool:
            if len(nd) == 1:
                chunk_scores[nd[0]] = node_scores[nd]

    top_order = tuple(int(i) for i in np.argsort(-chunk_scores)[:top_k])
    return TracLLMResult(
        scores=tuple(float(x) for x in chunk_scores),
        top_chunks=top_order,
        total_evals=total_evals,
        tree_depth=depth,
    )
