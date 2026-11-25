# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy==2.2.6", "scikit-learn>=1.5,<1.7"]
# ///
"""Offline empirical benchmark of 2025-2026 surrogate-residual SOTA
(RegressionMSR, ProxySHAP-GBT, Antithetic-LASSO/Ridge, and Hybrid CrossFit-GBT
+ Balanced CUBE / Ordered Coset) against CUBE and ordered cosets on archived
Stage 2.2 real Qwen3-4B data (zero new GPU calls).

Evaluates:
1. All 3 archived long multi-question documents in runs/136:
   - QASPER-1810.05241 (n=71, M=128, 6 Qs)
   - QuALITY-99914 (n=28, M=32, 17 Qs)
   - QuALITY-22967 (n=90, M=128, 18 Qs)
   Ground truth = mean Banzhaf over held-out reference blocks 4..7 (independent
   from evaluation pool blocks 0..3).
2. All 4 non-trivial exact invariant Qwen3-4B tables in runs/137 (n=7..10),
   with exact FWHT ground-truth Banzhaf effects, including orthogonal
   composition of CrossFit-GBT with Ordered Half-Cosets.

Produces ``regression-msr-summary.json``.
"""

from __future__ import annotations

import gzip
import json
import random
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import designs  # noqa: E402
from integration import draw_design_rows  # noqa: E402


def exact_tree_banzhaf(tree, n_features: int) -> np.ndarray:
    """Exact uniform Banzhaf value of a single sklearn decision tree on {0, 1}^n.

    For binary inputs x_i in {0, 1}, every split ``x_f <= thresh`` with
    ``0 <= thresh < 1`` routes ``x_f = 0`` left and ``x_f = 1`` right.
    Along any root-to-leaf path restricting feature set ``S_leaf = L_leaf U R_leaf``
    (with ``L_leaf`` forced to 0 and ``R_leaf`` forced to 1), leaf value ``v``
    has uniform probability ``2^{-|S_leaf|}`` and contributes:
      - ``+v * 2^{-(|S_leaf| - 1)}`` to feature ``f in R_leaf``
      - ``-v * 2^{-(|S_leaf| - 1)}`` to feature ``f in L_leaf``
      - ``0`` to features not appearing on the path.
    Handles redundant splits along a path exactly via dict state ``path[f] in {0, 1}``.
    """
    phi = np.zeros(n_features, dtype=np.float64)
    left = tree.children_left
    right = tree.children_right
    feat = tree.feature
    thresh = tree.threshold
    value = tree.value[:, 0, 0]

    def dfs(node: int, fixed: dict[int, int]) -> None:
        if left[node] == -1:  # leaf
            v = float(value[node])
            if not fixed or v == 0.0:
                return
            scale = v * (0.5 ** (len(fixed) - 1))
            for f, bit in fixed.items():
                phi[f] += scale if bit == 1 else -scale
            return
        f = int(feat[node])
        th = float(thresh[node])
        # Binary feature x_f in {0, 1}: left iff x_f <= th
        go_left_0 = 0.0 <= th
        go_left_1 = 1.0 <= th
        if f in fixed:
            bit = fixed[f]
            if (bit == 0 and go_left_0) or (bit == 1 and go_left_1):
                dfs(int(left[node]), fixed)
            else:
                dfs(int(right[node]), fixed)
            return
        # Normal informative split (0 -> left, 1 -> right)
        if go_left_0 and not go_left_1:
            fixed[f] = 0
            dfs(int(left[node]), fixed)
            fixed[f] = 1
            dfs(int(right[node]), fixed)
            del fixed[f]
        elif go_left_0 and go_left_1:
            dfs(int(left[node]), fixed)
        elif not go_left_0 and not go_left_1:
            dfs(int(right[node]), fixed)
        else:
            fixed[f] = 1
            dfs(int(left[node]), fixed)
            fixed[f] = 0
            dfs(int(right[node]), fixed)
            del fixed[f]

    dfs(0, {})
    return phi


def exact_ensemble_banzhaf(model: GradientBoostingRegressor, n_features: int) -> np.ndarray:
    """Exact uniform Banzhaf value of a fitted sklearn GradientBoostingRegressor."""
    lr = float(model.learning_rate)
    total = np.zeros(n_features, dtype=np.float64)
    for est in model.estimators_.ravel():
        total += lr * exact_tree_banzhaf(est.tree_, n_features)
    return total


def _verify_exact_tree_banzhaf() -> float:
    """Brute-force 2^n verification on n=6 synthetic nonlinear function."""
    rng = np.random.default_rng(1234)
    n = 6
    all_x = np.array([[((m >> i) & 1) for i in range(n)] for m in range(1 << n)], dtype=np.float64)
    y = (
        0.3 * all_x[:, 0]
        - 0.2 * all_x[:, 1]
        + 0.4 * all_x[:, 0] * all_x[:, 2]
        + 0.25 * all_x[:, 2] * all_x[:, 3] * all_x[:, 4]
        + 0.1 * rng.standard_normal(1 << n)
    )
    model = GradientBoostingRegressor(
        n_estimators=25, max_depth=3, learning_rate=0.1, random_state=7
    )
    model.fit(all_x, y)
    exact_dp = exact_ensemble_banzhaf(model, n)
    preds = model.predict(all_x)
    # Exact Banzhaf via 2^n marginal averages
    brute = np.zeros(n, dtype=np.float64)
    for i in range(n):
        bit = 1 << i
        s = 0.0
        for m in range(1 << n):
            if (m & bit) == 0:
                s += preds[m | bit] - preds[m]
        brute[i] = s / (1 << (n - 1))
    max_diff = float(np.max(np.abs(exact_dp - brute)))
    if max_diff >= 1e-12:
        raise AssertionError(f"exact_ensemble_banzhaf mismatch: {max_diff}")
    return max_diff


def _fit_gbt(x: np.ndarray, y: np.ndarray, seed: int) -> GradientBoostingRegressor:
    model = GradientBoostingRegressor(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.08,
        min_samples_leaf=3,
        subsample=0.8,
        random_state=seed,
    )
    model.fit(x, y)
    return model


def msr_banzhaf(x01: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Exact conditional-mean uniform-Banzhaf Maximum Sample Reuse (MSR) estimator:
    mean(y[x_i == 1]) - mean(y[x_i == 0]) (0 if a column never takes both values).

    On perfectly balanced designs (CUBE / Ordered Cosets where sum(x_i) == m/2),
    this equals 2 * (z^T y) / m identically; on unbalanced i.i.d. uniform masks,
    conditional-mean normalization eliminates O(mu^2 / m) uncentered variance.
    """
    c1 = x01.sum(axis=0)
    c0 = len(y) - c1
    s1 = x01.T @ y
    s0 = (1.0 - x01).T @ y
    out = np.zeros(x01.shape[1], dtype=np.float64)
    mask = (c1 > 0) & (c0 > 0)
    out[mask] = s1[mask] / c1[mask] - s0[mask] / c0[mask]
    return out


def regression_msr_gbt_crossfit(x01: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """Unbiased 2-fold cross-fitted RegressionMSR with GBT surrogate (Witter et al., NeurIPS 2025)."""
    m, n = x01.shape
    half = m // 2
    idx = np.arange(m)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    a_idx, b_idx = idx[:half], idx[half:]

    est_parts = []
    for tr, ev, k in ((a_idx, b_idx, 0), (b_idx, a_idx, 1)):
        mdl = _fit_gbt(x01[tr], y[tr], seed * 17 + k)
        phi_hat = exact_ensemble_banzhaf(mdl, n)
        resid = y[ev] - mdl.predict(x01[ev])
        est_parts.append(phi_hat + msr_banzhaf(x01[ev], resid))
    return 0.5 * (est_parts[0] + est_parts[1])


def proxyshap_gbt_shared(x01: np.ndarray, y: np.ndarray, seed: int, *, adjust: bool) -> np.ndarray:
    """ProxySHAP-style shared-sample GBT estimator (Thies et al., NeurIPS 2026),
    with optional MSR residual adjustment on the same sample."""
    mdl = _fit_gbt(x01, y, seed * 19 + 3)
    phi_hat = exact_ensemble_banzhaf(mdl, x01.shape[1])
    if not adjust:
        return phi_hat
    resid = y - mdl.predict(x01)
    return phi_hat + msr_banzhaf(x01, resid)


def antithetic_linear_surrogate(
    x01: np.ndarray, y: np.ndarray, *, kind: str, alpha: float
) -> np.ndarray:
    """ContextCite-style LASSO or Ridge fit on {0, 1}^n (scaled to Banzhaf coefficients)."""
    if kind == "lasso":
        mdl = Lasso(alpha=alpha, max_iter=5000)
    elif kind == "ridge":
        mdl = Ridge(alpha=alpha)
    else:
        raise ValueError(kind)
    mdl.fit(x01, y)
    return mdl.coef_.astype(np.float64)


def hybrid_crossfit_gbt_balanced(
    x_blk0: np.ndarray,
    y_blk0: np.ndarray,
    x_blk1: np.ndarray,
    y_blk1: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Exact unbiased hybrid combining 2-fold CrossFit GBT with two independent
    balanced blocks (CUBE or ordered coset): preserves 1st + all even-order
    cancellation on residuals AND full prefix cache sharing."""
    n = x_blk0.shape[1]
    m0 = _fit_gbt(x_blk0, y_blk0, seed * 23 + 1)
    r1 = y_blk1 - m0.predict(x_blk1)
    est1 = exact_ensemble_banzhaf(m0, n) + msr_banzhaf(x_blk1, r1)

    m1 = _fit_gbt(x_blk1, y_blk1, seed * 23 + 2)
    r0 = y_blk0 - m1.predict(x_blk0)
    est0 = exact_ensemble_banzhaf(m1, n) + msr_banzhaf(x_blk0, r0)
    return 0.5 * (est0 + est1)


def _masks_to_x01(masks: Sequence[int], n: int) -> np.ndarray:
    arr = np.zeros((len(masks), n), dtype=np.float64)
    for r, msk in enumerate(masks):
        for i in range(n):
            if (msk >> i) & 1:
                arr[r, i] = 1.0
    return arr


def _metrics(est: np.ndarray, truth: np.ndarray) -> tuple[int, float, float]:
    true_top = int(np.argmax(truth))
    est_top = int(np.argmax(est))
    hit = int(est_top == true_top)
    regret = float(truth[true_top] - truth[est_top])
    mse = float(np.mean((est - truth) ** 2))
    return hit, regret, mse


def benchmark_long_docs() -> list[dict[str, object]]:
    """Run controlled held-out benchmark on all 3 archived Stage 2.2 runs/136 docs."""
    runs_dir = S22 / "runs/136"
    doc_names = ["QASPER-1810.05241", "QuALITY-99914", "QuALITY-22967"]
    results: list[dict[str, object]] = []

    for doc in doc_names:
        ref_blocks = []
        for idx in range(8):
            p = runs_dir / f"{doc}-reference-{idx}.json.gz"
            d = json.loads(gzip.decompress(p.read_bytes()))
            ref_blocks.append(d)
        n = int(ref_blocks[0]["n"])
        nq = len(next(iter(ref_blocks[0]["scores"].values())))
        block_budget = len(ref_blocks[0]["masks"])
        m_half = block_budget // 2

        def _parse_block(blk: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
            raw_masks = blk["masks"]
            int_masks = [int(x, 0) if isinstance(x, str) else int(x) for x in raw_masks]
            x01 = _masks_to_x01(int_masks, n)
            scores = np.array([blk["scores"][str(rm)] for rm in raw_masks], dtype=np.float64)  # (2M, nq)
            z = 2.0 * x01 - 1.0
            B_blk = (2.0 * (z.T @ scores) / len(int_masks)).T  # (nq, n)
            return x01, scores, B_blk, int_masks

        parsed_refs = [_parse_block(blk) for blk in ref_blocks]

        # Held-out ground truth from independent blocks 4..7
        truth_B = np.mean([parsed_refs[k][2] for k in range(4, 8)], axis=0)

        # Build evaluation pools from independent blocks 0..3
        pool_blocks = [(parsed_refs[k][0], parsed_refs[k][1], parsed_refs[k][2]) for k in range(4)]

        # Concatenate pool rows for i.i.d. uniform / antithetic sub-budget sampling
        all_x = np.vstack([pb[0] for pb in pool_blocks])
        all_y = np.vstack([pb[1] for pb in pool_blocks])
        n_total = len(all_x)

        # Build exact complementary pair index list inside blocks 0..3
        pair_indices = []
        full_mask = (1 << n) - 1
        offset = 0
        for k in range(4):
            masks = parsed_refs[k][3]
            by_mask_pos = {msk: offset + idx for idx, msk in enumerate(masks)}
            seen = set()
            for idx, msk in enumerate(masks):
                pos = offset + idx
                if pos in seen:
                    continue
                comp = full_mask ^ msk
                cpos = by_mask_pos.get(comp)
                if cpos is not None and cpos not in seen:
                    pair_indices.append((pos, cpos))
                    seen.add(pos)
                    seen.add(cpos)
            offset += len(masks)

        method_stats: dict[str, list[tuple[int, float, float]]] = {}

        def record(name: str, est: np.ndarray, truth: np.ndarray) -> None:
            method_stats.setdefault(name, []).append(_metrics(est, truth))

        # 1. Balanced CUBE (1-block = 2M) across blocks 0..3
        for k in range(4):
            for q in range(nq):
                record(f"CUBE_1blk_B{block_budget}", pool_blocks[k][2][q], truth_B[q])

        # 2. Balanced CUBE (2-block = 4M) across 6 pairs of blocks (0,1)...(2,3)
        blk_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        for b0, b1 in blk_pairs:
            for q in range(nq):
                est = 0.5 * (pool_blocks[b0][2][q] + pool_blocks[b1][2][q])
                record(f"CUBE_2blk_B{2 * block_budget}", est, truth_B[q])
                # Hybrid CrossFit-GBT + Balanced CUBE (2-block = 4M)
                hyb = hybrid_crossfit_gbt_balanced(
                    pool_blocks[b0][0],
                    pool_blocks[b0][1][:, q],
                    pool_blocks[b1][0],
                    pool_blocks[b1][1][:, q],
                    seed=100 * b0 + 10 * b1 + q,
                )
                record(f"Hybrid_CrossFitGBT_CUBE_B{2 * block_budget}", hyb, truth_B[q])

        # 3. Also include archived single-block Ordered Coset (half & four) from runs/136 multi-0..2!
        for rep in range(3):
            for tag_label, pat in [
                ("Ordered_Half_1blk", f"{doc}-multi-{rep}-cube-*-prefix.json.gz"),
                ("Ordered_Four_1blk", f"{doc}-multi-{rep}-four-None-prefix.json.gz"),
            ]:
                for fp in sorted(runs_dir.glob(pat)):
                    if "cube-None-" in fp.name:
                        continue
                    # Pick the balanced cut (closest to n//2) for Ordered_Half
                    if tag_label == "Ordered_Half_1blk":
                        cut_str = fp.name.split("-cube-")[1].split("-prefix")[0]
                        if cut_str == "None" or abs(int(cut_str) - n // 2) > 1:
                            continue
                    md = json.loads(gzip.decompress(fp.read_bytes()))
                    B_m = np.array(md["B"], dtype=np.float64)
                    for q in range(nq):
                        record(f"{tag_label}_B{block_budget}", B_m[q], truth_B[q])
                    break  # one balanced cut per rep

        # 4. Surrogate baselines across budgets (scaled appropriately for QuALITY-99914 vs 71/90 docs)
        if block_budget >= 256:
            surr_budgets = [64, 128, 256, 512]
            lin_budgets = [32, 64, 128, 256]
        else:
            surr_budgets = [32, 64, 128]
            lin_budgets = [32, 64, 128]

        n_trials = 6  # deterministic trials per question over independent pool draws
        for B in lin_budgets:
            if B > n_total or B // 2 > len(pair_indices):
                continue
            for tr in range(n_trials):
                rng = np.random.default_rng(7000 + 31 * B + tr)
                chosen_pairs = rng.choice(len(pair_indices), size=B // 2, replace=False)
                rows = [idx for p_i in chosen_pairs for idx in pair_indices[int(p_i)]]
                xb = all_x[rows]
                for q in range(nq):
                    yb = all_y[rows, q]
                    record(
                        f"Antithetic_LASSO_a1e-3_B{B}",
                        antithetic_linear_surrogate(xb, yb, kind="lasso", alpha=1e-3),
                        truth_B[q],
                    )
                    record(
                        f"Antithetic_Ridge_a1_B{B}",
                        antithetic_linear_surrogate(xb, yb, kind="ridge", alpha=1.0),
                        truth_B[q],
                    )

        for B in surr_budgets:
            if B > n_total:
                continue
            for tr in range(n_trials):
                rng = np.random.default_rng(9000 + 37 * B + tr)
                rows = rng.choice(n_total, size=B, replace=False)
                xb = all_x[rows]
                for q in range(nq):
                    yb = all_y[rows, q]
                    sd = 1000 * B + 50 * tr + q
                    record(f"RegressionMSR_GBT_B{B}", regression_msr_gbt_crossfit(xb, yb, sd), truth_B[q])
                    record(
                        f"ProxySHAP_GBT_SharedMSR_B{B}",
                        proxyshap_gbt_shared(xb, yb, sd, adjust=True),
                        truth_B[q],
                    )
                    record(
                        f"ProxySHAP_GBT_NoAdj_B{B}",
                        proxyshap_gbt_shared(xb, yb, sd, adjust=False),
                        truth_B[q],
                    )

        summary_rows = []
        for name, vals in sorted(method_stats.items()):
            hits = [v[0] for v in vals]
            regs = [v[1] for v in vals]
            mses = [v[2] for v in vals]
            summary_rows.append(
                {
                    "method": name,
                    "samples": len(vals),
                    "top1_accuracy": round(float(np.mean(hits)), 4),
                    "mean_regret": round(float(np.mean(regs)), 6),
                    "mean_mse": float(f"{np.mean(mses):.4e}"),
                }
            )
        results.append(
            {
                "doc": doc,
                "n": n,
                "M": m_half,
                "block_budget_2M": block_budget,
                "nq": nq,
                "heldout_reference_blocks": "4..7 (independent of eval pool 0..3)",
                "methods": summary_rows,
            }
        )
    return results


def benchmark_exact_tables() -> list[dict[str, object]]:
    """Evaluate RegressionMSR / ProxySHAP / Hybrid+OrderedHalf on exact Qwen3-4B tables (runs/137)."""
    out: list[dict[str, object]] = []
    for p in sorted((S22 / "runs/137").glob("*-invariant-table.json.gz")):
        d = json.loads(gzip.decompress(p.read_bytes()))
        n = int(d["n"])
        if n < 5:
            continue
        m = designs.budget(n) // 2
        B_blk = 2 * m
        scores_hex = d["scores"]
        nq = len(next(iter(scores_hex.values())))

        # Full exact table arrays + exact FWHT ground truth
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

        stats: dict[str, list[tuple[int, float, float]]] = {}

        def rec(name: str, est: np.ndarray, tr: np.ndarray) -> None:
            stats.setdefault(name, []).append(_metrics(est, tr))

        seeds = 20
        cut = n // 2
        half_tags = tuple(0 if i < cut else 1 for i in range(n))
        can_half = (cut < m // 2) and (n - cut < m // 2) and (m >= 4)

        for sd in range(seeds):
            # 1-block and 2-block CUBE
            r0, _, _ = draw_design_rows(n, m, random.Random(100 + sd), tags=None, groups=1)
            r1, _, _ = draw_design_rows(n, m, random.Random(500 + sd), tags=None, groups=1)
            x0, x1 = _masks_to_x01(r0, n), _masks_to_x01(r1, n)
            y0, y1 = y_table[r0], y_table[r1]

            # i.i.d. uniform draws of sizes B_blk and 2*B_blk
            rng = np.random.default_rng(2000 + sd)
            u1 = rng.integers(0, 1 << n, size=B_blk)
            u2 = rng.integers(0, 1 << n, size=2 * B_blk)
            xu1, yu1 = _masks_to_x01(u1.tolist(), n), y_table[u1]
            xu2, yu2 = _masks_to_x01(u2.tolist(), n), y_table[u2]

            if can_half:
                h0, _, _ = draw_design_rows(n, m, random.Random(300 + sd), tags=half_tags, groups=2)
                h1, _, _ = draw_design_rows(n, m, random.Random(700 + sd), tags=half_tags, groups=2)
                xh0, xh1 = _masks_to_x01(h0, n), _masks_to_x01(h1, n)
                yh0, yh1 = y_table[h0], y_table[h1]

            for q in range(nq):
                tr = truth_B[q]
                c0 = msr_banzhaf(x0, y0[:, q])
                c1 = msr_banzhaf(x1, y1[:, q])
                rec(f"CUBE_1blk_B{B_blk}", c0, tr)
                rec(f"CUBE_2blk_B{2 * B_blk}", 0.5 * (c0 + c1), tr)
                rec(
                    f"Hybrid_CrossFitGBT_CUBE_B{2 * B_blk}",
                    hybrid_crossfit_gbt_balanced(x0, y0[:, q], x1, y1[:, q], sd * 13 + q),
                    tr,
                )
                rec(f"RegressionMSR_GBT_B{B_blk}", regression_msr_gbt_crossfit(xu1, yu1[:, q], sd * 17 + q), tr)
                rec(f"RegressionMSR_GBT_B{2 * B_blk}", regression_msr_gbt_crossfit(xu2, yu2[:, q], sd * 19 + q), tr)
                rec(
                    f"ProxySHAP_GBT_NoAdj_B{B_blk}",
                    proxyshap_gbt_shared(xu1, yu1[:, q], sd * 23 + q, adjust=False),
                    tr,
                )
                if can_half:
                    bh0 = msr_banzhaf(xh0, yh0[:, q])
                    bh1 = msr_banzhaf(xh1, yh1[:, q])
                    rec(f"Ordered_Half_1blk_B{B_blk}", bh0, tr)
                    rec(f"Ordered_Half_2blk_B{2 * B_blk}", 0.5 * (bh0 + bh1), tr)
                    rec(
                        f"Hybrid_CrossFitGBT_OrderedHalf_B{2 * B_blk}",
                        hybrid_crossfit_gbt_balanced(xh0, yh0[:, q], xh1, yh1[:, q], sd * 29 + q),
                        tr,
                    )

        rows = []
        for name, vals in sorted(stats.items()):
            rows.append(
                {
                    "method": name,
                    "samples": len(vals),
                    "top1_accuracy": round(float(np.mean([v[0] for v in vals])), 4),
                    "mean_regret": round(float(np.mean([v[1] for v in vals])), 6),
                    "mean_mse": float(f"{np.mean([v[2] for v in vals]):.4e}"),
                }
            )
        out.append({"doc": d["doc"], "n": n, "M": m, "nq": nq, "methods": rows})
    return out


def main() -> None:
    max_diff = _verify_exact_tree_banzhaf()
    long_res = benchmark_long_docs()
    exact_res = benchmark_exact_tables()
    payload = {
        "source": "Archived Stage 2.2 real Qwen3-4B runs/136 and runs/137; zero new GPU calls",
        "exact_tree_banzhaf_bruteforce_max_diff": max_diff,
        "long_docs_runs136": long_res,
        "exact_tables_runs137": exact_res,
    }
    target = HERE / "regression-msr-summary.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Verified exact_ensemble_banzhaf max_diff={max_diff:.2e}; wrote {target}")
    for doc_entry in long_res:
        print(f"\n=== LONG DOC {doc_entry['doc']} (n={doc_entry['n']}, 2M={doc_entry['block_budget_2M']}, nq={doc_entry['nq']}) ===")
        for row in doc_entry["methods"]:
            print(
                f"  {row['method']:34s} | top1={row['top1_accuracy']:.4f} "
                f"| regret={row['mean_regret']:.6f} | mse={row['mean_mse']:.3e}"
            )
    for ex in exact_res:
        print(f"\n=== EXACT TABLE {ex['doc']} (n={ex['n']}, M={ex['M']}, nq={ex['nq']}) ===")
        for row in ex["methods"]:
            print(
                f"  {row['method']:38s} | top1={row['top1_accuracy']:.4f} "
                f"| regret={row['mean_regret']:.6f} | mse={row['mean_mse']:.3e}"
            )


if __name__ == "__main__":
    main()
