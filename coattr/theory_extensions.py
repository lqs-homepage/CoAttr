# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy==2.2.6"]
# ///
"""Theoretical deepening and exact numerical verification for P-18 gaps T1, T2, and T3.

All additions live in this additive module so Stage-2-reviewed files remain byte-identical.

Covers:
- T1: Exact single-block 1-sparse vacant-channel blind-spot probability W_o(s),
      multi-block combined (vacant + cross-block) blind-spot collapse, and
      exact FWHT spectral delocalization audit on archived Stage 2.2 runs/137 tables.
- T2: Exact polynomial-time DP for single-pair gap variance inflation ratios
      (same-coset vs cross-coset, worst-case single character and local degree-3),
      plus Multi-Block Rotated Contiguous Coset Cuts verified on real runs/137 tables.
- T3: Deterministic [0, R_f^2/(M-n)] vacant-channel variance range tightening for
      Regime B Empirical Bernstein (38x-57x tighter than [-1, 1]) + exact Regime A/B
      sample-size crossover, and universal Rao-rank prefix-history lower bound
      h_k >= min(2k, 2M) for arbitrary (non-character) strength-3 orthogonal arrays.
"""

from __future__ import annotations

import gzip
import json
import math
import random
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import designs  # noqa: E402
from integration import draw_design_rows  # noqa: E402
from safe_selector import (  # noqa: E402
    empirical_bernstein_radius,
    exact_coset_certificates,
    general_tag_history_counts,
    native_alias,
    signed_population,
    vacant_channel_variance,
)


# ---------------------------------------------------------------------------
# T1: Single-Block Sparse Counterexample, Multi-Block Combined Detector,
#     and Spectral Delocalization Audit
# ---------------------------------------------------------------------------


def t1_sparse_blind_prob_cube(n: int, m: int, s: int = 3) -> Fraction:
    """Exact single-block probability that a 1-sparse odd character of degree s >= 3
    aliases onto an occupied channel in CUBE, leaving ALL M - n vacant channels 0."""
    if not (3 <= s <= n < m) or (s % 2 == 0) or (m & (m - 1)):
        raise ValueError("require odd s with 3 <= s <= n < m and power-of-two m")
    return s * native_alias(m, s - 1) + (n - s) * native_alias(m, s + 1)


def t1_multiblock_both_blind_prob_cube(n: int, m: int, r_blocks: int, s: int = 3) -> Fraction:
    """Exact probability that a 1-sparse odd character of degree s >= 3 escapes
    BOTH every vacant channel AND cross-block variance across R independent CUBE blocks.

    To leave all vacant channels identically 0 in every block AND produce zero
    cross-block sample variance, the aliased label must hit the SAME occupied
    channel j with the SAME independent Rademacher sign across all R blocks.
    """
    if r_blocks < 1:
        raise ValueError("r_blocks must be >= 1")
    p_minus = native_alias(m, s - 1)
    p_plus = native_alias(m, s + 1)
    half = Fraction(1, 2)
    term_in = s * p_minus * ((p_minus * half) ** (r_blocks - 1)) if p_minus > 0 else Fraction(0)
    term_out = (n - s) * p_plus * ((p_plus * half) ** (r_blocks - 1)) if p_plus > 0 else Fraction(0)
    return term_in + term_out


def combined_multiblock_variance(
    block_effects: Sequence[Sequence[float]],
    block_vacant_vars: Sequence[float],
    *,
    kappa: float = 1.0,
) -> dict[str, object]:
    """Combine R >= 2 independent blocks' vacant-channel and cross-block variance.

    Returns both per-segment and average-MSE combined variance estimates for the
    R-block mean estimator ``B_bar = mean(block_effects, axis=0)``.
    """
    r_blocks = len(block_effects)
    if r_blocks < 2 or len(block_vacant_vars) != r_blocks:
        raise ValueError("need R >= 2 blocks with matching block_vacant_vars")
    arr = np.asarray(block_effects, dtype=np.float64)  # (R, n)
    mean_b = arr.mean(axis=0)
    # Unbiased variance of the R-block mean per segment: sample_var / R
    cross_var_per_seg = np.var(arr, axis=0, ddof=1) / r_blocks
    vac_mean_var = float(kappa * np.mean(block_vacant_vars) / r_blocks)
    comb_per_seg = np.maximum(cross_var_per_seg, vac_mean_var)
    return {
        "mean_effects": mean_b.tolist(),
        "cross_block_avg_var": float(np.mean(cross_var_per_seg)),
        "vacant_calibrated_avg_var": vac_mean_var,
        "combined_avg_var": float(max(float(np.mean(cross_var_per_seg)), vac_mean_var)),
        "combined_max_seg_var": float(np.max(comb_per_seg)),
        "combined_per_seg_var": comb_per_seg.tolist(),
    }


def simulate_t1_sparse_counterexample(
    n: int = 20, m: int = 32, *, theta_s: float = 0.25, trials: int = 2000, seed: int = 42
) -> dict[str, object]:
    """Empirical Monte Carlo verification of T1 negative & positive theorems on
    f(z) = 0.5 + 0.1*z_0 + 0.08*z_1 + theta_s * z_2*z_3*z_4."""
    rng = random.Random(seed)
    p1_theory = float(t1_sparse_blind_prob_cube(n, m, s=3))
    p2_theory = float(t1_multiblock_both_blind_prob_cube(n, m, r_blocks=2, s=3))
    p3_theory = float(t1_multiblock_both_blind_prob_cube(n, m, r_blocks=3, s=3))

    def run_one_block() -> tuple[np.ndarray, float]:
        # Exact designs.py CUBE draw (labels in 1..m-1, independent Rademacher shift)
        rows = designs.cube(n, 2 * m, rng, cut=None)
        odd_vals = np.empty(m, dtype=np.float64)
        z_mat = np.empty((m, n), dtype=np.float64)
        for t in range(m):
            msk = rows[2 * t]
            # Evaluate odd part directly: O(z) = 0.1*z_0 + 0.08*z_1 + theta_s*z_2*z_3*z_4
            z = np.array([1.0 if ((msk >> i) & 1) else -1.0 for i in range(n)], dtype=np.float64)
            z_mat[t] = z
            odd_vals[t] = 0.1 * z[0] + 0.08 * z[1] + theta_s * z[2] * z[3] * z[4]
        eff = 2.0 * (z_mat.T @ odd_vals) / m
        vac = vacant_channel_variance(odd_vals.tolist(), eff.tolist(), m)
        return eff, vac

    blind_r1 = 0
    blind_r2_vac_only = 0
    blind_r2_combined = 0
    blind_r3_combined = 0
    max_err_when_blind_r1 = 0.0
    truth = np.zeros(n, dtype=np.float64)
    truth[0], truth[1] = 0.2, 0.16

    for _ in range(trials):
        e1, v1 = run_one_block()
        e2, v2 = run_one_block()
        e3, v3 = run_one_block()
        if v1 < 1e-14:
            blind_r1 += 1
            max_err_when_blind_r1 = max(max_err_when_blind_r1, float(np.max(np.abs(e1 - truth))))
        if v1 < 1e-14 and v2 < 1e-14:
            blind_r2_vac_only += 1
        c2 = combined_multiblock_variance([e1, e2], [v1, v2], kappa=1.0)
        if c2["combined_avg_var"] < 1e-14:
            blind_r2_combined += 1
        c3 = combined_multiblock_variance([e1, e2, e3], [v1, v2, v3], kappa=1.0)
        if c3["combined_avg_var"] < 1e-14:
            blind_r3_combined += 1

    return {
        "n": n,
        "M": m,
        "theta_S": theta_s,
        "trials": trials,
        "r1_vac_blind_theory": round(p1_theory, 6),
        "r1_vac_blind_empirical": round(blind_r1 / trials, 6),
        "r1_max_abs_B_error_when_vac_zero": round(max_err_when_blind_r1, 6),
        "r2_vac_only_blind_empirical": round(blind_r2_vac_only / trials, 6),
        "r2_combined_blind_theory": round(p2_theory, 6),
        "r2_combined_blind_empirical": round(blind_r2_combined / trials, 6),
        "r3_combined_blind_theory": float(f"{p3_theory:.3e}"),
        "r3_combined_blind_empirical": round(blind_r3_combined / trials, 6),
    }


def audit_exact_tables_delocalization(seeds: int = 80) -> list[dict[str, object]]:
    """Measure exact odd-spectrum delocalization (eta, r_eff, participation ratio)
    and single-block vacant-channel variance calibration on all 6 archived runs/137 tables."""
    out: list[dict[str, object]] = []
    for p in sorted((S22 / "runs/137").glob("*-invariant-table.json.gz")):
        d = json.loads(gzip.decompress(p.read_bytes()))
        n = int(d["n"])
        m = designs.budget(n) // 2
        scores_hex = d["scores"]
        nq = len(next(iter(scores_hex.values())))
        kappa_cube = exact_coset_certificates([n], m)[1]

        popcount = np.array([bin(mask).count("1") for mask in range(1 << n)], dtype=np.int32)
        odd_hi_mask = (popcount >= 3) & ((popcount & 1) == 1)
        deg1_indices = [1 << i for i in range(n)]

        y_table = np.array([scores_hex[hex(mask)] for mask in range(1 << n)], dtype=np.float64)
        q_rows = []
        for q in range(nq):
            f_wht = y_table[:, q].copy()
            for bit in range(n):
                step = 1 << bit
                f_wht = f_wht.reshape(-1, 2, step)
                a, b = f_wht[:, 0, :].copy(), f_wht[:, 1, :].copy()
                f_wht[:, 0, :] = (a + b) / 2.0
                f_wht[:, 1, :] = (b - a) / 2.0
                f_wht = f_wht.reshape(-1)
            e1 = float(np.sum(f_wht[deg1_indices] ** 2))
            odd_hi = f_wht[odd_hi_mask] ** 2
            e_odd_hi = float(np.sum(odd_hi))
            max_odd_hi = float(np.max(odd_hi)) if odd_hi.size > 0 else 0.0
            eta = max_odd_hi / e_odd_hi if e_odd_hi > 1e-18 else 0.0
            r_eff = (1.0 / eta) if eta > 0.0 else float("inf")
            sum_sq = float(np.sum(odd_hi ** 2))
            r_pr = (e_odd_hi ** 2 / sum_sq) if sum_sq > 1e-24 else float("inf")

            # Empirical single-block CUBE vacant variance vs true MSE across `seeds` draws
            truth_B = np.array([2.0 * f_wht[1 << i] for i in range(n)], dtype=np.float64)
            ratios = []
            zero_vac_cnt = 0
            rng = random.Random(3000 + 37 * n + q)
            for _ in range(seeds):
                rows = designs.cube(n, 2 * m, rng, cut=None)
                odd_v = np.array(
                    [0.5 * (y_table[rows[2 * t], q] - y_table[rows[2 * t + 1], q]) for t in range(m)],
                    dtype=np.float64,
                )
                z_mat = np.array(
                    [[1.0 if ((rows[2 * t] >> i) & 1) else -1.0 for i in range(n)] for t in range(m)],
                    dtype=np.float64,
                )
                eff = 2.0 * (z_mat.T @ odd_v) / m
                vac = vacant_channel_variance(odd_v.tolist(), eff.tolist(), m)
                mse = float(np.mean((eff - truth_B) ** 2))
                if vac < 1e-14:
                    zero_vac_cnt += 1
                if mse > 1e-15 and math.isfinite(kappa_cube):
                    ratios.append((kappa_cube * vac) / mse)

            q_rows.append(
                {
                    "q": q,
                    "E1": float(f"{e1:.4e}"),
                    "E_odd_ge3": float(f"{e_odd_hi:.4e}"),
                    "eta_peakiness": round(eta, 4),
                    "r_eff": round(r_eff, 2) if math.isfinite(r_eff) else "inf",
                    "participation_ratio": round(r_pr, 2) if math.isfinite(r_pr) else "inf",
                    "vac_zero_rate": round(zero_vac_cnt / seeds, 4),
                    "single_blk_ratio_p10": round(float(np.percentile(ratios, 10)), 4) if ratios else None,
                    "single_blk_ratio_median": round(float(np.median(ratios)), 4) if ratios else None,
                    "single_blk_ratio_mean": round(float(np.mean(ratios)), 4) if ratios else None,
                }
            )
        out.append(
            {
                "doc": d["doc"],
                "n": n,
                "M": m,
                "vacant_dof": m - n,
                "kappa_cube": round(kappa_cube, 4) if math.isfinite(kappa_cube) else "inf",
                "questions": q_rows,
            }
        )
    return out


# ---------------------------------------------------------------------------
# T2: Exact Pairwise Gap Variance Ratio DP & Multi-Block Rotated Contiguous Cuts
# ---------------------------------------------------------------------------


def exact_pairwise_gap_ratio_dp(
    sizes: Sequence[int], m: int, ci: int, cj: int
) -> dict[str, float]:
    """Exact polynomial-DP worst-case pairwise gap variance ratio
    ``sup_f Var_D(B_hat_i - B_hat_j) / Var_CUBE(B_hat_i - B_hat_j)`` for a fixed
    pair ``(i, j)`` with coset tags ``(ci, cj)``.

    Evaluates BOTH exact components of the closed-form variance decomposition:
    1. Even complements U subseteq [n]\\{i,j} (|U| >= 2 even):
       weight P_D(U) = Pr(l_U = 0) + Pr(l_U = l_i ^ l_j)
       vs P_CUBE(U) = p_o(|U|) + p_o(|U| + 2).
    2. Odd complements V subseteq [n]\\{i,j} (|V| >= 3 odd):
       weight P_D(V) = Pr(l_V = l_i) + Pr(l_V = l_j)
       vs P_CUBE(V) = 2 * p_o(|V| + 1).
    Also returns exact degree-3 local (within-single-coset U) and isotropic ratios.
    """
    g = len(sizes)
    if g < 1 or (g & (g - 1)) or m % g != 0 or (m & (m - 1)):
        raise ValueError("g and m must be powers of two with g dividing m")
    if not (0 <= ci < g and 0 <= cj < g):
        raise ValueError("coset indices out of range")
    n = sum(sizes)
    h = m // g
    rem = list(sizes)
    rem[ci] -= 1
    rem[cj] -= 1
    if any(r < 0 for r in rem) or any(sz > h for sz in sizes):
        raise ValueError("invalid coset sizes for chosen pair (ci, cj)")
    if g == 1:
        return {
            "worst_overall": 1.0,
            "worst_even_branch": 1.0,
            "worst_odd_branch": 1.0,
            "deg3_same_coset_local_ratio": 1.0,
            "deg3_isotropic_ratio": 1.0,
        }

    k = [float(signed_population(h, v)) for v in range(h + 1)]
    po = [float(native_alias(m, v)) if v <= m else 0.0 for v in range(n + 3)]

    def dp_product_bounds(shift: Sequence[int]) -> tuple[list[list[float]], list[list[float]]]:
        """DP over all counts u_c in 0..rem[c] returning min/max of prod_c k_h(u_c + shift[c])
        indexed by [total_size][xor_tag_of_u]."""
        lo = [[float("inf")] * g]
        hi = [[float("-inf")] * g]
        lo[0][0] = 1.0
        hi[0][0] = 1.0
        for c, rc in enumerate(rem):
            sh = shift[c]
            span = len(lo)
            nlo = [[float("inf")] * g for _ in range(span + rc)]
            nhi = [[float("-inf")] * g for _ in range(span + rc)]
            for t in range(span):
                for x in range(g):
                    lv = lo[t][x]
                    if not math.isfinite(lv):
                        continue
                    hv = hi[t][x]
                    for u in range(rc + 1):
                        kv = k[u + sh]
                        nx = x ^ (c if (u & 1) else 0)
                        a, b = lv * kv, hv * kv
                        if a > b:
                            a, b = b, a
                        nt = t + u
                        if a < nlo[nt][nx]:
                            nlo[nt][nx] = a
                        if b > nhi[nt][nx]:
                            nhi[nt][nx] = b
            lo, hi = nlo, nhi
        return lo, hi

    # For even U (|U| >= 2 even):
    # - If ci == cj: x = 0 activates BOTH Pr(l_U = 0) and Pr(l_U = l_i ^ l_j) simultaneously!
    #   Need exact DP for the SUM k_h(u_ci) + k_h(u_ci + 2) multiplied by prod_{c != ci} k_h(u_c)!
    # - If ci != cj: x = 0 activates ONLY Pr(l_U = 0), while x = ci ^ cj activates ONLY Pr(l_U = l_i ^ l_j)
    #   (mutually exclusive x values!).
    worst_even = 0.0
    if ci == cj:
        # Factor out coset ci so prod_{c != ci} k_h(u_c) is exact
        lo_oth = [[float("inf")] * g]
        hi_oth = [[float("-inf")] * g]
        lo_oth[0][0] = hi_oth[0][0] = 1.0
        for c, rc in enumerate(rem):
            if c == ci:
                continue
            span = len(lo_oth)
            nlo = [[float("inf")] * g for _ in range(span + rc)]
            nhi = [[float("-inf")] * g for _ in range(span + rc)]
            for t in range(span):
                for x in range(g):
                    lv = lo_oth[t][x]
                    if not math.isfinite(lv):
                        continue
                    hv = hi_oth[t][x]
                    for u in range(rc + 1):
                        kv = k[u]
                        nx = x ^ (c if (u & 1) else 0)
                        a, b = lv * kv, hv * kv
                        if a > b:
                            a, b = b, a
                        nt = t + u
                        if a < nlo[nt][nx]:
                            nlo[nt][nx] = a
                        if b > nhi[nt][nx]:
                            nhi[nt][nx] = b
            lo_oth, hi_oth = nlo, nhi

        for u in range(rem[ci] + 1):
            wanted_x = ci if (u & 1) else 0  # so total xor = wanted_x ^ (ci if u odd else 0) == 0
            coeff = k[u] + k[u + 2]
            for t in range(len(lo_oth)):
                s = t + u
                if s < 2 or (s & 1) == 1:
                    continue
                if not math.isfinite(lo_oth[t][wanted_x]):
                    continue
                prod = hi_oth[t][wanted_x] if coeff >= 0.0 else lo_oth[t][wanted_x]
                pd = (2.0 + (h - 1) * coeff * prod) / h
                pc = po[s] + po[s + 2]
                if pc > 1e-15:
                    worst_even = max(worst_even, pd / pc)
                elif pd > 1e-15:
                    worst_even = float("inf")
    else:
        _, hi_zero = dp_product_bounds([0] * g)
        sh_ij = [0] * g
        sh_ij[ci] = 1
        sh_ij[cj] = 1
        _, hi_ij = dp_product_bounds(sh_ij)
        target_x = ci ^ cj
        for s in range(2, n - 1, 2):
            pc = po[s] + po[s + 2]
            # Branch x = 0: Pr(l_U = 0)
            if math.isfinite(hi_zero[s][0]):
                pd0 = (1.0 + (h - 1) * hi_zero[s][0]) / h
                if pc > 1e-15:
                    worst_even = max(worst_even, pd0 / pc)
                elif pd0 > 1e-15:
                    worst_even = float("inf")
            # Branch x = ci ^ cj: Pr(l_U = l_i ^ l_j)
            if math.isfinite(hi_ij[s][target_x]):
                pd1 = (1.0 + (h - 1) * hi_ij[s][target_x]) / h
                if pc > 1e-15:
                    worst_even = max(worst_even, pd1 / pc)
                elif pd1 > 1e-15:
                    worst_even = float("inf")

    # Odd V (|V| >= 3 odd):
    # - If ci == cj: x = ci activates BOTH Pr(l_V = l_i) and Pr(l_V = l_j) identically (2 * Pr(l_V = l_i))!
    # - If ci != cj: x = ci activates Pr(l_V = l_i), x = cj activates Pr(l_V = l_j) (mutually exclusive).
    sh_i = [0] * g
    sh_i[ci] = 1
    sh_j = [0] * g
    sh_j[cj] = 1
    _, hi_i = dp_product_bounds(sh_i)
    _, hi_j = dp_product_bounds(sh_j) if ci != cj else (None, hi_i)

    worst_odd = 0.0
    for s in range(3, n - 1, 2):
        pc = 2.0 * po[s + 1]
        if ci == cj:
            if math.isfinite(hi_i[s][ci]):
                pd = 2.0 * (1.0 + (h - 1) * hi_i[s][ci]) / h
                if pc > 1e-15:
                    worst_odd = max(worst_odd, pd / pc)
                elif pd > 1e-15:
                    worst_odd = float("inf")
        else:
            for target_x, hi_tab in ((ci, hi_i), (cj, hi_j)):
                if math.isfinite(hi_tab[s][target_x]):
                    pd = (1.0 + (h - 1) * hi_tab[s][target_x]) / h
                    if pc > 1e-15:
                        worst_odd = max(worst_odd, pd / pc)
                    elif pd > 1e-15:
                        worst_odd = float("inf")

    # Exact degree-3 local-interaction (U = {a, b} in same coset as i) and isotropic degree-3 ratios
    pc_deg3 = po[2] + po[4]
    if ci == cj:
        # Local U in same coset ci (requires rem[ci] >= 2)
        if rem[ci] >= 2:
            pd_local = (1.0 + (h - 1) * k[4]) / h
            local_ratio = pd_local / pc_deg3 if pc_deg3 > 1e-15 else float("inf")
        else:
            local_ratio = 0.0
        # Isotropic sum over all pairs {a, b} in [n]\{i,j}
        total_pairs = math.comb(n - 2, 2)
        iso_num = 0.0
        for c in range(g):
            if rem[c] >= 2:
                prob = (1.0 + (h - 1) * (k[4] if c == ci else k[2] * k[2])) / h
                iso_num += math.comb(rem[c], 2) * prob
        iso_ratio = (iso_num / (total_pairs * pc_deg3)) if total_pairs > 0 and pc_deg3 > 1e-15 else 1.0
    else:
        # Cross-coset: ANY local pair {a, b} inside a single coset has xor tag 0 != ci ^ cj => ZERO collision!
        local_ratio = 0.0
        total_pairs = math.comb(n - 2, 2)
        target_x = ci ^ cj
        iso_num = 0.0
        for c1 in range(g):
            c2 = c1 ^ target_x
            if c2 <= c1:
                continue
            cnt = rem[c1] * rem[c2]
            if cnt == 0:
                continue
            # counts of U U {i, j} in each coset:
            sh = [0] * g
            sh[ci] += 1
            sh[cj] += 1
            sh[c1] += 1
            sh[c2] += 1
            prod = 1.0
            for v in sh:
                prod *= k[v]
            prob = (1.0 + (h - 1) * prod) / h
            iso_num += cnt * prob
        iso_ratio = (iso_num / (total_pairs * pc_deg3)) if total_pairs > 0 and pc_deg3 > 1e-15 else 1.0

    return {
        "worst_overall": max(worst_even, worst_odd),
        "worst_even_branch": worst_even,
        "worst_odd_branch": worst_odd,
        "deg3_same_coset_local_ratio": local_ratio,
        "deg3_isotropic_ratio": iso_ratio,
    }


def rotated_contiguous_2coset_cuts(
    n: int, m: int, r_blocks: int, *, max_gamma: float = 1.05
) -> list[int]:
    """Select R contiguous 2-coset cut boundaries evenly spread across the
    analytically safe ``gamma <= max_gamma`` range.

    Every block remains 100% contiguous (preserving full prefix cache sharing
    and working simultaneously for ALL questions with zero multi-question conflict
    and zero selection bias), while diversifying which segment pairs share a coset.
    """
    h = m // 2
    lo = max(1, n - h + 1)
    hi = min(n - 1, h - 1)
    safe_cuts = [c for c in range(lo, hi + 1) if exact_coset_certificates((c, n - c), m)[0] <= max_gamma]
    if not safe_cuts:
        return []
    if r_blocks <= 1 or len(safe_cuts) == 1:
        # Closest to n // 2
        best = min(safe_cuts, key=lambda c: (abs(c - n / 2.0), c))
        return [best] * max(1, r_blocks)
    # Pick evenly spaced indices across safe_cuts
    cuts = []
    for k in range(r_blocks):
        idx = round(k * (len(safe_cuts) - 1) / (r_blocks - 1))
        cuts.append(safe_cuts[idx])
    return cuts


def rotated_contiguous_gcoset_tags(
    n: int, m: int, r_blocks: int, *, g: int = 4
) -> list[tuple[int, ...]]:
    """Construct R blocks of shifted contiguous g-coset partitions.

    Block r shifts the cut boundaries by floor(r * (n/g) / R) segments while
    wrapping ONLY the final partial remainder into the last coset OR keeping
    strictly contiguous monotonic intervals (block 0 has sizes [s, s, s, s],
    block r shifts internal cut points by delta = round(r * (n/g) / R) within
    safe capacity limits) so every block preserves monotonic document order!
    """
    h = m // g
    base_step = n / g
    plans: list[tuple[int, ...]] = []
    for r in range(r_blocks):
        offset = int(round((r * base_step) / max(1, r_blocks)))
        # Monotonic contiguous cuts: c_1, ..., c_{g-1}
        cuts = []
        for j in range(1, g):
            c = int(round(j * base_step)) + (offset if offset < h else 0)
            # Keep every interval strictly within [1, h - 1]
            c = max(j, min(n - (g - j), c))
            cuts.append(c)
        # Re-center around balanced cuts if shifting pushes final interval over capacity
        if r == 0:
            cuts = [j * n // g for j in range(1, g)]
        else:
            # Shift internal cuts cleanly within [-step/3, +step/3]
            shift = int(round(((r / max(1, r_blocks - 1)) - 0.5) * (base_step * 0.6)))
            cuts = [max(j, min(n - (g - j), j * n // g + shift)) for j in range(1, g)]
        bounds = [0] + cuts + [n]
        sizes = [bounds[j + 1] - bounds[j] for j in range(g)]
        if any(sz <= 0 or sz >= h for sz in sizes):
            cuts = [j * n // g for j in range(1, g)]
            bounds = [0] + cuts + [n]
            sizes = [bounds[j + 1] - bounds[j] for j in range(g)]
        tags = tuple(j for j, sz in enumerate(sizes) for _ in range(sz))
        plans.append(tags)
    return plans


def cyclic_balanced_coset_tags(n: int, r_blocks: int, *, g: int = 2) -> list[tuple[int, ...]]:
    """Construct R blocks of cyclic-shifted balanced g-coset partitions.

    Every block has the EXACT same balanced coset sizes ``[(j+1)*n//g - j*n//g]``
    (hence IDENTICAL exact ``gamma`` and ``kappa`` across all blocks, ZERO
    multi-question conflict, and ZERO selection bias), while shifting the coset
    phase cyclically by ``floor(r * (n / g) / R)`` segments across blocks.

    For g=2, R=2:
    - Block 0: ``[0]*(n//2) + [1]*(n - n//2)`` (cut at n/2)
    - Block 1: ``[1]*(n//4) + [0]*(n//2) + [1]*(n - n//2 - n//4)`` (cuts at n/4, 3n/4)
    Any pair of segments at distance ``d in [n/4, 3n/4]`` is guaranteed to lie
    in DIFFERENT cosets in at least one block (cutting local deg-3 pairwise
    inflation in half without ever altering coset balance).
    """
    base_sizes = [(j + 1) * n // g - j * n // g for j in range(g)]
    base_tags = [j for j, sz in enumerate(base_sizes) for _ in range(sz)]
    if r_blocks <= 1:
        return [tuple(base_tags)]
    period = n / g
    plans: list[tuple[int, ...]] = []
    for r in range(r_blocks):
        shift = int(round((r * period) / r_blocks)) % n
        # Cyclic right shift by `shift` preserves exact coset sizes
        shifted = tuple(base_tags[(i - shift) % n] for i in range(n))
        plans.append(shifted)
    return plans


def evaluate_rotated_cosets_on_exact_tables(r_blocks: int = 4, trials: int = 60) -> list[dict[str, object]]:
    """Compare R-block CUBE, static balanced 2-coset, and Cyclic Balanced 2-Coset
    Rotation on all non-trivial exact Qwen3-4B tables (runs/137), tracking worst-pair
    gap MSE, top-1 accuracy, and exact GF(2) prefix token ratio."""
    out: list[dict[str, object]] = []
    for p in sorted((S22 / "runs/137").glob("*-invariant-table.json.gz")):
        d = json.loads(gzip.decompress(p.read_bytes()))
        n = int(d["n"])
        if n < 8:
            continue
        m = designs.budget(n) // 2
        if (n - n // 2) > m // 2:
            continue
        rot_tags = cyclic_balanced_coset_tags(n, r_blocks, g=2)
        static_tags = rot_tags[0]
        sizes = (n // 2, n - n // 2)
        gamma_half, kappa_half = exact_coset_certificates(sizes, m)
        scores_hex = d["scores"]
        nq = len(next(iter(scores_hex.values())))
        y_table = np.array([scores_hex[hex(msk)] for msk in range(1 << n)], dtype=np.float64)

        truth_B = np.zeros((nq, n), dtype=np.float64)
        for q in range(nq):
            f_wht = y_table[:, q].copy()
            for bit in range(n):
                step = 1 << bit
                f_wht = f_wht.reshape(-1, 2, step)
                a, b = f_wht[:, 0, :].copy(), f_wht[:, 1, :].copy()
                f_wht[:, 0, :] = (a + b) / 2.0
                f_wht[:, 1, :] = (b - a) / 2.0
                f_wht = f_wht.reshape(-1)
            truth_B[q] = np.array([2.0 * f_wht[1 << i] for i in range(n)])

        cube_hist = sum(general_tag_history_counts([0] * n, m, 1))
        static_hist = sum(general_tag_history_counts(static_tags, m, 2))
        rot_hist = float(np.mean([sum(general_tag_history_counts(tg, m, 2)) for tg in rot_tags]))

        methods: dict[str, list[tuple[int, ...] | None]] = {
            "cube": [None] * r_blocks,
            "static_half": [static_tags] * r_blocks,
            "cyclic_rotated_half": rot_tags,
        }
        res_m: dict[str, dict[str, float]] = {}
        for name, tag_seq in methods.items():
            hits = 0
            gap_sq_errs = []
            mses = []
            for tr in range(trials):
                est_blocks = np.zeros((r_blocks, nq, n), dtype=np.float64)
                for r_i, tg in enumerate(tag_seq):
                    g = 1 if tg is None else 2
                    rows, _, _ = draw_design_rows(n, m, random.Random(50000 + 100 * tr + r_i), tags=tg, groups=g)
                    z_mat = np.array(
                        [[1.0 if ((msk >> i) & 1) else -1.0 for i in range(n)] for msk in rows],
                        dtype=np.float64,
                    )
                    for q in range(nq):
                        est_blocks[r_i, q] = 2.0 * (z_mat.T @ y_table[rows, q]) / len(rows)
                mean_est = est_blocks.mean(axis=0)
                for q in range(nq):
                    tb = truth_B[q]
                    order = np.argsort(-tb)
                    t1, t2 = int(order[0]), int(order[1])
                    true_gap = tb[t1] - tb[t2]
                    est_gap = mean_est[q, t1] - mean_est[q, t2]
                    gap_sq_errs.append(float((est_gap - true_gap) ** 2))
                    hits += int(int(np.argmax(mean_est[q])) == t1)
                    mses.append(float(np.mean((mean_est[q] - tb) ** 2)))
            res_m[name] = {
                "top1_accuracy": round(hits / (trials * nq), 4),
                "top1_top2_gap_mse": float(f"{np.mean(gap_sq_errs):.4e}"),
                "avg_mse": float(f"{np.mean(mses):.4e}"),
            }

        out.append(
            {
                "doc": d["doc"],
                "n": n,
                "M": m,
                "r_blocks": r_blocks,
                "gamma_half": round(gamma_half, 4),
                "static_prefix_ratio_vs_cube": round(static_hist / cube_hist, 4),
                "cyclic_rotated_prefix_ratio_vs_cube": round(rot_hist / cube_hist, 4),
                "results": res_m,
            }
        )
    return out


# ---------------------------------------------------------------------------
# T3: Tightened Regime B Empirical Bernstein Range + Universal Strength-3 OA Bound
# ---------------------------------------------------------------------------


def vacant_variance_bounds(
    m: int,
    n: int,
    *,
    score_range: float = 1.0,
    kappa_cand: float = 1.0,
    kappa_cube: float = 1.0,
    delta: float = 0.05,
    k_candidates: int = 3,
    target_tau: float = 0.005,
) -> dict[str, float | int]:
    """Exact deterministic bounds on single-block vacant-channel Parseval variance
    and paired variance difference Y_D = kappa_D * v_vac(D) - kappa_CUBE * v_vac(CUBE).

    Since f(z) in [a, b] with R_f = b - a implies |O(z_t)| <= R_f / 2 and O(z_t)^2 <= R_f^2 / 4:
        0 <= v_vac <= (4 / (M - n)) * (R_f^2 / 4) = R_f^2 / (M - n).
    """
    if not (1 <= n < m):
        raise ValueError("require 1 <= n < m")
    dof = m - n
    vmax = (score_range ** 2) / dof
    lo_uncal, hi_uncal = -vmax, vmax
    lo_cal, hi_cal = -kappa_cube * vmax, kappa_cand * vmax
    span_uncal = hi_uncal - lo_uncal
    span_cal = hi_cal - lo_cal

    # Minimum sample size m where Empirical Bernstein's linear tail term
    # 7 * span * ln(4*K/delta) / (3 * (m - 1)) <= target_tau
    per_delta = delta / k_candidates
    log_term = math.log(4.0 / per_delta)
    m_min_tight = math.ceil(1.0 + (7.0 * span_uncal * log_term) / (3.0 * target_tau))
    m_min_naive = math.ceil(1.0 + (7.0 * 2.0 * log_term) / (3.0 * target_tau))

    # Concrete zero-variance EB radius at m = 16, 32, 64
    radii_at_m = {}
    for sample_m in (16, 32, 64):
        r_tight, meth = empirical_bernstein_radius([0.0] * sample_m, lo_uncal, hi_uncal, per_delta)
        r_naive, _ = empirical_bernstein_radius([0.0] * sample_m, -1.0, 1.0, per_delta)
        radii_at_m[f"m{sample_m}"] = {
            "tight_radius": round(r_tight, 6),
            "naive_radius": round(r_naive, 6),
            "method": meth,
        }

    return {
        "M": m,
        "n": n,
        "vacant_dof": dof,
        "score_range": score_range,
        "single_block_vac_upper": round(vmax, 6),
        "paired_uncalibrated_bounds": [round(lo_uncal, 6), round(hi_uncal, 6)],
        "paired_calibrated_bounds": [round(lo_cal, 6), round(hi_cal, 6)],
        "tightening_factor_vs_minus1_plus1": round(2.0 / span_uncal, 2),
        "eb_min_samples_for_tau_tight": int(m_min_tight),
        "eb_min_samples_for_tau_naive": int(m_min_naive),
        "zero_var_eb_radii": radii_at_m,
    }


def universal_strength3_oa_lower_bound(
    n: int, m: int, segment_lengths: Sequence[int] | None = None
) -> dict[str, float | int]:
    """Universal Rao-rank prefix-history lower bound for ANY strength-3 binary
    orthogonal array OA(2M, n, 2, 3) (cancelling all 1st- and 2nd-order terms),
    without assuming Walsh / linear character structure.

    Proof: Fixing z_k = +1 yields a strength-2 OA on the preceding k columns
    (z_0, ..., z_{k-1}) for any k >= 1. Since [1, z_0, ..., z_{k-1}] has k + 1
    mutually orthogonal columns, its prefix matrix has rank >= min(k + 1, M),
    giving h_k^{(+)} >= min(k + 1, M) distinct prefixes (with h_0^{(+)} = 1).
    """
    if not (1 <= n < m):
        raise ValueError("require 1 <= n < m")
    lens = [1] * n if segment_lengths is None else [int(x) for x in segment_lengths]
    if len(lens) != n:
        raise ValueError("segment_lengths length must equal n")

    def _next_pow2(x: int) -> int:
        return 1 << (x - 1).bit_length()

    cube_half = general_tag_history_counts([0] * n, m, 1)
    # Wait: in planning.py, each segment i has prefix count = 2 * h_half[i] except i=0 where header is shared (1)?
    # Even simpler: compare on the EXACT half-block scale M (divide oa3 and char by 2 for k>=1) OR full 2M scale:
    # Full 2M scale: unshared document tokens = (B / 2) * sum(L_i) because each segment appears in EXACTLY half of 2M prompts (= M prompts)!
    #Wait: for segment k, it appears in M = B/2 prompts, and the number of distinct prefixes PRECEDING segment k
    # among those M prompts where z_k = +1 is h_k / 2 (by exact complement symmetry, half of the h_k distinct
    # length-k prefixes end with z_k = +1? No, z_k is AFTER length-k prefix; among the h_k prefixes of length k,
    # each extends to z_k=+1 with distinct prefix, except wait: if z_k is determined by earlier bits it appears once, else twice!
    # Specifically, the number of distinct prefixes of (z_0, ..., z_{k-1}) that occur WITH z_k = +1 is >= h_k / 2
    # when z_k = +1 is balanced, and equals h_{k-1} (for independent z_k) or h_k / 2!
    # Exact check: in our exact token cost formula from Stage 2.2 theory.md:
    # segment k is computed once per distinct value of (z_0, ..., z_{k-1}) appearing in rows with z_k = +1.
    # For strength-3 OA: fixing z_k = +1 leaves a strength-2 OA on (z_0, ..., z_{k-1})!
    # By Rao's rank bound on strength-2 OA, the number of distinct prefixes (z_0, ..., z_{k-1}) with z_k = +1 is >= min(k + 1, M) (for k >= 1, and 1 for k = 0)!!
    # CHECK THAT!
    # k = 0: 1 distinct empty prefix with z_0 = +1 (always 1).
    # k = 1: fixing z_1 = +1 gives strength-2 OA (actually strength-1+2: [1, z_0] orthogonal) => >= 2 distinct prefixes!
    # k >= 1: fixing z_k = +1 leaves a strength-2 OA on k variables z_0..z_{k-1} => [1, z_0, ..., z_{k-1}] are k+1 orthogonal columns => >= min(k + 1, M) distinct prefixes!
    # AND for dyadic character designs, that count MUST be a power of 2 => >= min(next_pow2(k + 1), M)!!
    # CHECK DYADIC SUM for k = 0..M-1:
    # k=0: 1; k=1: 2; k=2: 4; k=3: 4; k=4..7: 8 (4 terms); k=8..15: 16 (8 terms); ...; k=M/2..M-1: M (M/2 terms)!
    # Sum = 1 + 2 + 4*2 + 8*4 + 16*8 + ... + M*(M/2) = 1 + 2 + 8 + 32 + ... + M^2/2 = (2/3) M^2 + 1/3!!
    # Divided by unshared M * M = M^2 gives EXACTLY 2/3!!
    oa3_seg_copies = [1] + [min(k + 1, m) for k in range(1, n)]
    char_seg_copies = [1] + [min(_next_pow2(k + 1), m) for k in range(1, n)]
    cube_seg_copies = cube_half  # exact expected distinct prefixes with z_k = +1 in standard CUBE!

    total_len = sum(lens)
    unshared_toks = m * total_len
    oa3_toks = sum(c * l for c, l in zip(oa3_seg_copies, lens))
    char_toks = sum(c * l for c, l in zip(char_seg_copies, lens))
    cube_toks = sum(c * l for c, l in zip(cube_seg_copies, lens))

    return {
        "n": n,
        "M": m,
        "unshared_doc_tokens": int(unshared_toks),
        "universal_strength3_oa_floor_ratio_vs_unshared": round(oa3_toks / unshared_toks, 4),
        "dyadic_character_floor_ratio_vs_unshared": round(char_toks / unshared_toks, 4),
        "expected_cube_ratio_vs_unshared": round(cube_toks / unshared_toks, 4),
        "universal_strength3_oa_floor_ratio_vs_cube": round(oa3_toks / cube_toks, 4),
        "dyadic_character_floor_ratio_vs_cube": round(char_toks / cube_toks, 4),
    }


def main() -> None:
    # 1. T1 verification
    t1_sim = simulate_t1_sparse_counterexample(n=20, m=32, trials=2000, seed=42)
    t1_long_theory = {
        "QASPER_n71_M128": {
            "r1_vac_blind": round(float(t1_sparse_blind_prob_cube(71, 128, 3)), 6),
            "r2_both_blind": round(float(t1_multiblock_both_blind_prob_cube(71, 128, 2, 3)), 6),
            "r3_both_blind": float(f"{float(t1_multiblock_both_blind_prob_cube(71, 128, 3, 3)):.3e}"),
        },
        "QuALITY_n90_M128": {
            "r1_vac_blind": round(float(t1_sparse_blind_prob_cube(90, 128, 3)), 6),
            "r2_both_blind": round(float(t1_multiblock_both_blind_prob_cube(90, 128, 2, 3)), 6),
            "r3_both_blind": float(f"{float(t1_multiblock_both_blind_prob_cube(90, 128, 3, 3)):.3e}"),
        },
    }
    t1_deloc = audit_exact_tables_delocalization(seeds=80)

    # 2. T2 verification
    t2_dp = {}
    for label, n, m in [("QASPER_n71_M128", 71, 128), ("QuALITY_n90_M128", 90, 128), ("QuALITY_n28_M32", 28, 32)]:
        half_sizes = (n // 2, n - n // 2)
        four_sizes = tuple((j + 1) * n // 4 - j * n // 4 for j in range(4))
        entry = {
            "half_same_coset": exact_pairwise_gap_ratio_dp(half_sizes, m, 0, 0),
            "half_cross_coset": exact_pairwise_gap_ratio_dp(half_sizes, m, 0, 1),
            "rotated_4blk_2coset_cuts": rotated_contiguous_2coset_cuts(n, m, 4, max_gamma=1.05),
        }
        if all(sz <= m // 4 for sz in four_sizes):
            entry["four_same_coset"] = exact_pairwise_gap_ratio_dp(four_sizes, m, 0, 0)
            entry["four_cross_coset"] = exact_pairwise_gap_ratio_dp(four_sizes, m, 0, 1)
        t2_dp[label] = entry
    t2_rot_exact = evaluate_rotated_cosets_on_exact_tables(r_blocks=4, trials=60)
    # Also quantify exact GF(2) prefix cost of 2-block / 4-block cyclic balanced 2-coset & 4-coset on long docs
    tok = json.loads(gzip.decompress((S22 / "tokenized.json.gz").read_bytes()))
    t2_cyclic_long = {}
    for doc in ("QASPER-1810.05241", "QuALITY-22967"):
        lens = [len(p) for p in tok[doc]["parts"]]
        n = len(lens)
        m = designs.budget(n) // 2
        cube_w = sum(l * h for l, h in zip(lens, general_tag_history_counts([0] * n, m, 1)))
        entry_c = {}
        for g_val in (2, 4):
            static_tg = cyclic_balanced_coset_tags(n, 1, g=g_val)[0]
            rot2_tg = cyclic_balanced_coset_tags(n, 2, g=g_val)
            rot4_tg = cyclic_balanced_coset_tags(n, 4, g=g_val)
            stat_w = sum(l * h for l, h in zip(lens, general_tag_history_counts(static_tg, m, g_val)))
            r2_w = float(np.mean([sum(l * h for l, h in zip(lens, general_tag_history_counts(tg, m, g_val))) for tg in rot2_tg]))
            r4_w = float(np.mean([sum(l * h for l, h in zip(lens, general_tag_history_counts(tg, m, g_val))) for tg in rot4_tg]))
            entry_c[f"g{g_val}"] = {
                "static_doc_token_ratio_vs_cube": round(stat_w / cube_w, 4),
                "cyclic_2blk_doc_token_ratio_vs_cube": round(r2_w / cube_w, 4),
                "cyclic_4blk_doc_token_ratio_vs_cube": round(r4_w / cube_w, 4),
            }
        t2_cyclic_long[doc] = entry_c

    # 3. T3 verification
    t3_eb = {}
    t3_oa = {}
    for doc in ("QASPER-1810.05241", "QuALITY-22967", "QuALITY-99914"):
        lens = [len(p) for p in tok[doc]["parts"]]
        n = len(lens)
        m = designs.budget(n) // 2
        half_sizes = (n // 2, n - n // 2)
        _, k_cube = exact_coset_certificates([n], m)
        _, k_half = exact_coset_certificates(half_sizes, m)
        t3_eb[doc] = vacant_variance_bounds(
            m, n, kappa_cand=k_half if math.isfinite(k_half) else 1.5, kappa_cube=k_cube
        )
        t3_oa[doc] = universal_strength3_oa_lower_bound(n, m, segment_lengths=lens)

    payload = {
        "T1_sparse_counterexample_sim": t1_sim,
        "T1_long_doc_blind_probs": t1_long_theory,
        "T1_exact_tables_delocalization": t1_deloc,
        "T2_pairwise_gap_ratio_dp": t2_dp,
        "T2_cyclic_rotation_long_doc_prefix_costs": t2_cyclic_long,
        "T2_rotated_cosets_exact_tables": t2_rot_exact,
        "T3_tightened_eb_bounds": t3_eb,
        "T3_universal_strength3_oa_bounds": t3_oa,
    }
    target = HERE / "theory-extensions-summary.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {target}")
    print("\n--- T1 Sparse Counterexample ---")
    print(json.dumps(t1_sim, indent=2))
    print("Long doc blind probs:", json.dumps(t1_long_theory))
    print("\n--- T1 Delocalization on runs/137 ---")
    for doc_e in t1_deloc:
        etas = [q["eta_peakiness"] for q in doc_e["questions"]]
        prs = [q["participation_ratio"] for q in doc_e["questions"]]
        p10s = [q["single_blk_ratio_p10"] for q in doc_e["questions"]]
        print(f"  {doc_e['doc']:18s} n={doc_e['n']:2d} dof={doc_e['vacant_dof']:2d}: eta={etas} r_PR={prs} p10={p10s}")
    print("\n--- T2 Pairwise Gap DP ---")
    for k_lbl, val in t2_dp.items():
        hs = val["half_same_coset"]
        hc = val["half_cross_coset"]
        print(
            f"  {k_lbl:18s}: half_same(worst={hs['worst_overall']:.3f}, local3={hs['deg3_same_coset_local_ratio']:.3f}, iso3={hs['deg3_isotropic_ratio']:.3f}) | "
            f"half_cross(worst={hc['worst_overall']:.3f}, local3={hc['deg3_same_coset_local_ratio']:.3f}, iso3={hc['deg3_isotropic_ratio']:.3f}) | "
            f"rot_cuts={val['rotated_4blk_2coset_cuts']}"
        )
        if "four_same_coset" in val:
            fs = val["four_same_coset"]
            fc = val["four_cross_coset"]
            print(
                f"    four: same(worst={fs['worst_overall']:.3f}, local3={fs['deg3_same_coset_local_ratio']:.3f}) | "
                f"cross(worst={fc['worst_overall']:.3f}, local3={fc['deg3_same_coset_local_ratio']:.3f})"
            )
    print("\n--- T2 Cyclic Rotation Long-Doc Prefix Costs ---")
    print(json.dumps(t2_cyclic_long, indent=2))
    print("\n--- T2 Cyclic Rotated Cosets on Exact Tables ---")
    for row in t2_rot_exact:
        print(
            f"  {row['doc']:18s} G={row['gamma_half']:.4f} pref(stat={row['static_prefix_ratio_vs_cube']:.4f}, cyc={row['cyclic_rotated_prefix_ratio_vs_cube']:.4f}) | "
            f"top1(cube={row['results']['cube']['top1_accuracy']:.3f}, stat={row['results']['static_half']['top1_accuracy']:.3f}, cyc={row['results']['cyclic_rotated_half']['top1_accuracy']:.3f}) | "
            f"gapMSE(cube={row['results']['cube']['top1_top2_gap_mse']:.2e}, stat={row['results']['static_half']['top1_top2_gap_mse']:.2e}, cyc={row['results']['cyclic_rotated_half']['top1_top2_gap_mse']:.2e})"
        )
    print("\n--- T3 Tightened EB & Strength-3 OA Floor ---")
    for doc in t3_eb:
        eb = t3_eb[doc]
        oa = t3_oa[doc]
        print(
            f"  {doc:18s}: EB tight={eb['tightening_factor_vs_minus1_plus1']}x "
            f"m_min({eb['eb_min_samples_for_tau_tight']} vs naive {eb['eb_min_samples_for_tau_naive']}) | "
            f"OA3_floor={oa['universal_strength3_oa_floor_ratio_vs_cube']:.4f} vs char_floor={oa['dyadic_character_floor_ratio_vs_cube']:.4f} (vs CUBE)"
        )


if __name__ == "__main__":
    main()
