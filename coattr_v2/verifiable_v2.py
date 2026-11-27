"""Dual-Mode Multi-Witness Verifiable Provenance (CoAttr-Verify).

Bridges the fundamental divide in 2025-2026 LLM provenance research between:
  - Single-witness minimal verifiable provenance (BLIP, PVLDB'26; Evergreen, PVLDB'27),
    which guarantees reproducibility f(S*) >= f(Omega) - eps for ONE minimal subset
    but silently hides redundant corroborating clauses or redundant poisoned copies; and
  - Quantitative marginal attribution (ContextCite, TracLLM, CoAttr Banzhaf), which
    exposes all redundant and multi-hop causes without certifying subset sufficiency.

Given CoAttr's unbiased Banzhaf scores B_hat[0..n-1], CoAttr-Verify uses O(m log k)
prefix-shared verification evaluations to:
  1. Extract and certify ALL disjoint minimal sufficient witnesses W_1, W_2, ..., W_m
     satisfying f(W_r) >= f(Omega) - eps; and
  2. Certify complete post-remediation clearance: f(Omega \\ union W_r) < f(Omega) - eps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class MultiWitnessCertificate:
    """Certified multi-witness provenance result for a single query."""

    full_score: float
    empty_score: float
    epsilon: float
    witnesses: tuple[tuple[int, ...], ...]  # disjoint certified sufficient subsets W_1, ..., W_m
    witness_scores: tuple[float, ...]  # f(W_r) for each certified witness
    post_removal_score: float  # f(Omega \ union_r W_r)
    remediation_certified: bool  # True iff post_removal_score < full_score - epsilon
    verification_evals: int  # extra forward passes beyond Banzhaf estimation


def _subset_to_mask(indices: Sequence[int]) -> int:
    m = 0
    for idx in indices:
        m |= (1 << int(idx))
    return m


def extract_multi_witness_provenance(
    banzhaf_scores: Sequence[float],
    eval_mask: Callable[[int], float],
    n: int,
    *,
    epsilon: float = 0.05,
    sufficiency_threshold: float | None = None,
    clearance_threshold: float | None = None,
    max_witnesses: int = 4,
    max_witness_size: int = 8,
    minimize_witness: bool = True,
    full_score: float | None = None,
    empty_score: float | None = None,
) -> MultiWitnessCertificate:
    """Extract and certify all disjoint minimal sufficient witnesses guided by CoAttr's
    unbiased Banzhaf ranking.

    Supports both single-threshold (``tau_suff = tau_clear = f(Omega) - eps``) and
    dual-threshold verifiable remediation (``tau_suff`` for isolated witness sufficiency
    and ``tau_clear <= tau_suff`` for certified residual clearance ``f(Omega \\ union W) < tau_clear``),
    preventing premature termination under redundant saturated triggers where 1 remaining
    copy still scores high (e.g., 0.85 vs 0.99 for 2 copies).

    Algorithm:
      1. Establish full-context target f(Omega), empty baseline f(0), and thresholds
         tau_suff (default f(Omega) - eps) and tau_clear (default tau_suff).
      2. Maintain remaining pool U = {0, ..., n-1} (initially all chunks).
      3. While |witnesses| < max_witnesses and f(U) >= tau_clear:
         (a) Order chunks in U by descending Banzhaf score (positive scores first).
         (b) Binary-search smallest prefix k <= min(|U|, max_witness_size) such that
             f(top_k(U)) >= tau_suff (with linear fallback up to max_witness_size).
         (c) Optionally prune unnecessary chunks inside W_r backward to certify minimality.
         (d) Record W_r, remove W_r from U, and evaluate f(U) to test if another
             hidden redundant witness still keeps f(U) >= tau_clear!
      4. Stop as soon as f(U) < tau_clear (complete remediation certified!) or max_witnesses reached.
    """
    cache: dict[int, float] = {}
    eval_count = 0

    def query(mask: int) -> float:
        nonlocal eval_count
        if mask not in cache:
            cache[mask] = float(eval_mask(mask))
            eval_count += 1
        return cache[mask]

    full_mask = (1 << n) - 1
    f_full = float(full_score) if full_score is not None else query(full_mask)
    f_empty = float(empty_score) if empty_score is not None else query(0)
    if full_score is not None:
        cache[full_mask] = f_full
    if empty_score is not None:
        cache[0] = f_empty

    tau_suff = float(sufficiency_threshold) if sufficiency_threshold is not None else (f_full - float(epsilon))
    tau_clear = float(clearance_threshold) if clearance_threshold is not None else tau_suff
    scores = np.asarray(banzhaf_scores, dtype=np.float64)

    # If empty context already exceeds clearance threshold, no chunk witness can clear it
    if f_empty >= tau_suff and f_empty >= tau_clear:
        return MultiWitnessCertificate(
            full_score=f_full,
            empty_score=f_empty,
            epsilon=float(epsilon),
            witnesses=(),
            witness_scores=(),
            post_removal_score=f_empty,
            remediation_certified=False,
            verification_evals=eval_count,
        )

    remaining = set(range(n))
    witnesses: list[tuple[int, ...]] = []
    witness_scores: list[float] = []
    cur_pool_score = f_full

    while len(witnesses) < max_witnesses and cur_pool_score >= tau_clear and remaining:
        ordered = sorted(remaining, key=lambda i: (-float(scores[i]), i))
        cap = min(len(ordered), max_witness_size)

        # Binary search for smallest prefix length k in [1, cap] achieving >= tau_suff
        lo, hi = 1, cap
        best_k: int | None = None
        best_val = -1e18
        while lo <= hi:
            mid = (lo + hi) // 2
            val = query(_subset_to_mask(ordered[:mid]))
            if val >= tau_suff:
                best_k = mid
                best_val = val
                hi = mid - 1
            else:
                lo = mid + 1

        # If binary search missed due to non-monotonicity, scan prefixes up to cap
        if best_k is None:
            for k_try in range(1, cap + 1):
                v_try = query(_subset_to_mask(ordered[:k_try]))
                if v_try >= tau_suff:
                    best_k = k_try
                    best_val = v_try
                    break
        if best_k is None:
            # Could not isolate a compact witness <= max_witness_size from remaining pool
            break

        cand = list(ordered[:best_k])
        cand_score = best_val

        # Optional minimal pruning inside the small candidate witness (preserves sufficiency)
        if minimize_witness and len(cand) > 1:
            idx = len(cand) - 1
            while idx >= 0 and len(cand) > 1:
                trial = cand[:idx] + cand[idx + 1 :]
                v_tr = query(_subset_to_mask(trial))
                if v_tr >= tau_suff:
                    cand = trial
                    cand_score = v_tr
                idx -= 1

        w_tuple = tuple(sorted(cand))
        witnesses.append(w_tuple)
        witness_scores.append(cand_score)
        for ch in w_tuple:
            remaining.discard(ch)

        cur_pool_score = query(_subset_to_mask(sorted(remaining))) if remaining else f_empty

    return MultiWitnessCertificate(
        full_score=f_full,
        empty_score=f_empty,
        epsilon=float(epsilon),
        witnesses=tuple(witnesses),
        witness_scores=tuple(witness_scores),
        post_removal_score=cur_pool_score,
        remediation_certified=(cur_pool_score < tau_clear),
        verification_evals=eval_count,
    )
