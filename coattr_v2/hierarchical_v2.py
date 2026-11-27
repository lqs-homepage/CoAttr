"""Hierarchical Random-Polarity Kronecker Paley-Hadamard Attribution (Hier-CoAttr, Theorem 5).

Scales CoAttr cleanly from n <= 90 to ultra-long contexts (n = 200..512+ chunks,
24K..64K+ tokens) while preserving:
  1. Exact unbiasedness for the flat n-chunk uniform Banzhaf value for ANY
     Boolean utility f: {+-1}^n -> R via random-polarity background super-chunks
     (Theorem 5(1)):
       - All linear & pairwise cross-background interactions cancel IDENTICALLY
         even conditional on a single polarity draw s!
       - All higher-order even-subset co-located background interactions have
         EXACT ZERO expectation over s ~ Unif({+-1}^{n - |A|}), and vanish
         identically on every single draw whenever background chunks are null!
  2. Elimination of Level-1 intra-super-chunk sign-cancellation blindness via
     random-polarity L2-energy detection: E_s[B_{G_u}(s)^2] = sum_{j in G_u} B_j^2 + ...
  3. 100% compatibility with Kronecker Paley-Hadamard early prefix capping and
     Algorithm 2's Delta_wave-aware adaptive-width physical scheduler (every
     Level-1 and Level-2 row maps directly to an n-bit document mask).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .designs_v2 import (
    DesignBlock,
    draw_dyadic_block,
    draw_kronecker_paley_block,
    dyadic_half_budget,
    select_best_paley_configs,
)
from .safe_selector_v2 import vacant_channel_variance


@dataclass(frozen=True)
class SuperChunkPartition:
    """Contiguous partition of n ordered document chunks into G super-chunks."""

    n: int
    groups: tuple[tuple[int, ...], ...]  # G tuples of 0-indexed fine chunk indices

    @property
    def num_groups(self) -> int:
        return len(self.groups)


@dataclass(frozen=True)
class HierarchicalBlockPlan:
    """One level of a hierarchical orthogonal design mapped to full n-bit masks."""

    level: int  # 1 (coarse) or 2 (refined)
    n_full: int
    n_vars: int  # number of active macro/fine variables in the orthogonal matrix
    m_half: int
    g_cosets: int
    l_order: int
    design_family: str  # "kronecker_paley" or "dyadic_coset"
    polarity_mode: str  # "locked" or "random"
    var_to_chunks: tuple[tuple[int, ...], ...]  # length n_vars; fine vars have len 1, super-chunks len >= 1
    chunk_polarities: tuple[int, ...]  # length n_full, each in {+1, -1}; +1 means active when macro=+1
    fine_chunk_to_var: dict[int, int]  # for Level 2: maps expanded fine chunk idx -> var idx
    base_block: DesignBlock
    full_masks: tuple[int, ...]  # 2 * m_half integer n-bit masks in complementary pairs (2t, 2t+1)


def partition_super_chunks(n: int, target_groups: int = 36) -> SuperChunkPartition:
    """Partition n ordered chunks into G contiguous, balanced super-chunks.

    When n <= target_groups, each chunk is its own singleton super-chunk (degenerates
    cleanly to flat 1-level CoAttr).
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    g = min(n, max(2, int(target_groups)))
    groups = tuple(
        tuple(range((j * n) // g, ((j + 1) * n) // g))
        for j in range(g)
    )
    return SuperChunkPartition(n=n, groups=groups)


def _draw_compact_strength3_block(
    n_vars: int,
    rng: random.Random,
    *,
    prefer_paley: bool = True,
) -> tuple[DesignBlock, str]:
    """Select the smallest valid strength-3 Kronecker Paley-Hadamard (L === 4 mod 8)
    or dyadic ordered coset block covering ``n_vars`` variables."""
    if prefer_paley:
        cfgs = select_best_paley_configs(
            n_vars,
            groups=(2, 4),
            mod8_eq4_only=True,
            strictly_smaller_than_dyadic=True,
        )
        if cfgs:
            top = cfgs[0]
            blk = draw_kronecker_paley_block(n_vars, int(top["g"]), int(top["L"]), rng)
            return blk, "kronecker_paley"
    m_d = dyadic_half_budget(n_vars)
    g = 4 if m_d >= 64 else (2 if m_d >= 32 else 1)
    # Check coset 0 capacity
    if g > 1:
        sz = [(j + 1) * n_vars // g - j * n_vars // g for j in range(g)]
        if any(s >= m_d // g for s in sz):
            g = 2 if g == 4 else 1
    blk = draw_dyadic_block(n_vars, m_d, rng, g=g)
    return blk, "dyadic_coset"


def _lift_var_rows_to_full_masks(
    var_rows: Sequence[int],
    var_to_chunks: Sequence[Sequence[int]],
    chunk_polarities: Sequence[int],
) -> tuple[int, ...]:
    """Deterministically expand each n_vars-bit mask row into a full n-bit document mask.

    For each chunk j belonging to variable v with polarity s_j in {+1, -1}:
      - if s_j == +1: chunk j is INCLUDED iff bit v of var_row is 1 (y_v = +1).
      - if s_j == -1: chunk j is INCLUDED iff bit v of var_row is 0 (y_v = -1),
        so z_j = y_v * s_j in {+-1} holds identically on every row AND its fold-over!
    """
    full: list[int] = []
    for row in var_rows:
        m_full = 0
        for v_idx, chunks in enumerate(var_to_chunks):
            bit_v = (row >> v_idx) & 1
            for ch in chunks:
                inc = bit_v if chunk_polarities[ch] == 1 else (1 - bit_v)
                if inc:
                    m_full |= (1 << ch)
        full.append(m_full)
    return tuple(full)


def build_level1_coarse_plan(
    partition: SuperChunkPartition,
    rng: random.Random,
    *,
    polarity_mode: str = "locked",
    prefer_paley: bool = True,
) -> HierarchicalBlockPlan:
    """Build a Level-1 coarse strength-3 Paley/dyadic block over G super-chunks.

    ``polarity_mode``:
      - ``"locked"``: s_j = +1 for all chunks (preserves full contiguous super-chunk
        semantics; strongest signal when chunks in a super-chunk reinforce each other).
      - ``"random"``: independent s_j ~ Unif({+1, -1}) per chunk inside each super-chunk
        (unbiased L2-energy detector E[B_{G_u}^2] = sum B_j^2 + ..., immune to exact
        opposite-sign cancellation inside a super-chunk).
    """
    if polarity_mode not in ("locked", "random"):
        raise ValueError(f"unsupported polarity_mode: {polarity_mode}")
    n_full = partition.n
    g_cnt = partition.num_groups
    if polarity_mode == "locked":
        pol = tuple(1 for _ in range(n_full))
    else:
        pol = tuple(1 if rng.random() < 0.5 else -1 for _ in range(n_full))

    blk, fam = _draw_compact_strength3_block(g_cnt, rng, prefer_paley=prefer_paley)
    full_masks = _lift_var_rows_to_full_masks(blk.rows, partition.groups, pol)
    return HierarchicalBlockPlan(
        level=1,
        n_full=n_full,
        n_vars=g_cnt,
        m_half=blk.m,
        g_cosets=blk.g,
        l_order=blk.L,
        design_family=fam,
        polarity_mode=polarity_mode,
        var_to_chunks=partition.groups,
        chunk_polarities=pol,
        fine_chunk_to_var={},
        base_block=blk,
        full_masks=full_masks,
    )


def select_active_super_chunks(
    level1_scores_per_q: Sequence[Sequence[float]],
    *,
    top_k_per_q: int = 3,
    max_total_groups: int = 6,
    min_energy_frac: float = 0.05,
    extra_level1_scores_per_q: Sequence[Sequence[float]] | None = None,
) -> list[int]:
    """Select active super-chunks to refine at Level 2 across a batch of n_q questions.

    Aggregates per-question absolute super-chunk scores (and optionally a second
    random-polarity Level-1 block via root-mean-square energy) so multi-hop and
    redundant super-chunks for EVERY question in the batch are preserved while
    keeping total expanded groups bounded by ``max_total_groups``.
    """
    arr = np.abs(np.asarray(level1_scores_per_q, dtype=np.float64))  # (nq, G)
    if arr.ndim == 1:
        arr = arr[None, :]
    if extra_level1_scores_per_q is not None:
        arr2 = np.abs(np.asarray(extra_level1_scores_per_q, dtype=np.float64))
        if arr2.ndim == 1:
            arr2 = arr2[None, :]
        score_mat = np.sqrt(0.5 * (arr ** 2 + arr2 ** 2))
    else:
        score_mat = arr

    nq, g_cnt = score_mat.shape
    selected: set[int] = set()
    # 1. Guarantee each question's top_k_per_q strongest super-chunks (subject to cap)
    rank_votes = np.zeros(g_cnt, dtype=np.float64)
    for q in range(nq):
        row = score_mat[q]
        mx = float(np.max(row)) if g_cnt > 0 else 0.0
        order = np.argsort(-row)
        for rank, gid in enumerate(order[: min(top_k_per_q, g_cnt)]):
            if mx <= 1e-12 or row[gid] >= min_energy_frac * mx or rank == 0:
                rank_votes[int(gid)] += (top_k_per_q - rank) + (row[gid] / max(mx, 1e-12))
                selected.add(int(gid))

    if len(selected) <= max_total_groups:
        return sorted(selected)

    # Cap to max_total_groups by cross-question normalized peak importance
    peak_norm = np.max(score_mat / np.maximum(np.max(score_mat, axis=1, keepdims=True), 1e-12), axis=0)
    combined = rank_votes + 2.0 * peak_norm
    top_g = sorted(np.argsort(-combined)[:max_total_groups].tolist())
    return top_g


def build_level2_refined_plan(
    partition: SuperChunkPartition,
    active_group_ids: Sequence[int],
    rng: random.Random,
    *,
    background_polarity: str = "random",
    prefer_paley: bool = True,
) -> HierarchicalBlockPlan:
    """Build a Level-2 fine-expansion strength-3 block over expanded active chunks
    + random-polarity unexpanded background super-chunks (Theorem 5).

    Variables are ordered strictly in document order (so ordered cosets preserve
    maximal contiguous token-prefix sharing!):
      - For g_id in active_group_ids: each individual chunk ch in groups[g_id]
        gets its own fine variable (polarity = +1).
      - For g_id NOT in active_group_ids: the entire super-chunk groups[g_id]
        occupies 1 macro variable with independent random polarity s_j ~ Unif({+-1})
        (or locked +1 for ablation), making every expanded chunk's Banzhaf estimator
        EXACTLY UNBIASED for the flat n-chunk uniform Banzhaf value (Theorem 5(1)).
    """
    if background_polarity not in ("random", "locked"):
        raise ValueError(f"unsupported background_polarity: {background_polarity}")
    active_set = set(int(g) for g in active_group_ids)
    n_full = partition.n

    var_to_chunks: list[tuple[int, ...]] = []
    fine_chunk_to_var: dict[int, int] = {}
    pol = [1] * n_full

    for g_id, grp in enumerate(partition.groups):
        if g_id in active_set:
            for ch in grp:
                fine_chunk_to_var[ch] = len(var_to_chunks)
                var_to_chunks.append((ch,))
                pol[ch] = 1
        else:
            var_to_chunks.append(tuple(grp))
            if background_polarity == "random":
                for ch in grp:
                    pol[ch] = 1 if rng.random() < 0.5 else -1

    n_vars = len(var_to_chunks)
    blk, fam = _draw_compact_strength3_block(n_vars, rng, prefer_paley=prefer_paley)
    full_masks = _lift_var_rows_to_full_masks(blk.rows, var_to_chunks, pol)
    return HierarchicalBlockPlan(
        level=2,
        n_full=n_full,
        n_vars=n_vars,
        m_half=blk.m,
        g_cosets=blk.g,
        l_order=blk.L,
        design_family=fam,
        polarity_mode=background_polarity,
        var_to_chunks=tuple(var_to_chunks),
        chunk_polarities=tuple(pol),
        fine_chunk_to_var=fine_chunk_to_var,
        base_block=blk,
        full_masks=full_masks,
    )


def decode_hierarchical_block(
    plan: HierarchicalBlockPlan,
    mask_scores_per_q: Sequence[dict[int, float]],
) -> tuple[np.ndarray, list[float]]:
    """Decode exact unbiased Banzhaf scores & vacant variances for all ``plan.n_vars``
    variables from complementary fold-over pairs (2t, 2t+1) in ``plan.full_masks``.

    Returns:
      - ``var_effects``: shape (nq, n_vars)
      - ``vacant_vars``: list of length nq (Parseval vacant-channel variance V_r)
    """
    m = plan.m_half
    n_vars = plan.n_vars
    z_mat = np.array(
        [[1.0 if ((plan.base_block.rows[2 * t] >> v) & 1) else -1.0 for v in range(n_vars)] for t in range(m)],
        dtype=np.float64,
    )
    nq = len(mask_scores_per_q)
    effects = np.zeros((nq, n_vars), dtype=np.float64)
    vac_vars: list[float] = []
    for q in range(nq):
        sc = mask_scores_per_q[q]
        odd_v = np.array(
            [0.5 * (sc[plan.full_masks[2 * t]] - sc[plan.full_masks[2 * t + 1]]) for t in range(m)],
            dtype=np.float64,
        )
        eff_q = 2.0 * (z_mat.T @ odd_v) / m
        effects[q] = eff_q
        vac_vars.append(vacant_channel_variance(odd_v.tolist(), eff_q.tolist(), m) if m > n_vars else 0.0)
    return effects, vac_vars


def assemble_full_n_scores(
    plan_l2: HierarchicalBlockPlan,
    l2_var_effects: np.ndarray,
    *,
    partition: SuperChunkPartition,
    l1_group_effects: np.ndarray | None = None,
    l1_chunk_polarities: Sequence[int] | None = None,
) -> np.ndarray:
    """Assemble full (nq, n_full) chunk-level Banzhaf attribution vector from Level 2.

    - Expanded fine chunks in A receive their exact unbiased Level-2 fine estimates.
    - Unexpanded background chunks in G_u receive a small tie-breaking residual
      (polarity-aligned l1_group_effect / |G_u| if provided, else 0.0) so all n
      chunks have well-defined scores while expanded chunks dominate cleanly.
    """
    nq = l2_var_effects.shape[0]
    out = np.zeros((nq, plan_l2.n_full), dtype=np.float64)
    for ch, v_idx in plan_l2.fine_chunk_to_var.items():
        out[:, ch] = l2_var_effects[:, v_idx]
    if l1_group_effects is not None:
        expanded_chunks = set(plan_l2.fine_chunk_to_var.keys())
        for g_id, grp in enumerate(partition.groups):
            if grp[0] in expanded_chunks:
                continue
            share = l1_group_effects[:, g_id] / max(1, len(grp))
            for ch in grp:
                sgn = float(l1_chunk_polarities[ch]) if l1_chunk_polarities is not None else 1.0
                out[:, ch] = sgn * share
    return out
