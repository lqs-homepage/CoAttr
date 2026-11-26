"""Upgraded Attribution Baselines & Statistical Uncertainty Toolkit (CoAttr v2).

Directly resolves Pillar II (statistical uncertainty, CUBE R=2) and Pillar V
(facing BLIP, Sketching-LOO, and single-pass AttnTrace head-on):
1. ``blip_two_stage_pruning``: Faithful two-stage backward minimal-sufficient
   provenance pruning (Lin et al., PVLDB 2026 BLIP) — distinct from scalar LOO.
2. ``sketching_loo_estimator``: Randomized sparse group-deletion sketching
   (Liaw et al. 2026 style) solving L1/L2 linear recovery under low budgets B.
3. ``attntrace_single_pass_scores``: Single-pass cross-layer attention aggregation
   over answer-to-context attention weights (AttnTrace / AT2 zero-ablation family).
4. ``cube_r2_average``: Clean multi-block CUBE R=2 reference baseline.
5. ``workload_cluster_bootstrap_ci`` & ``paired_sign_flip_pvalue``: Exact
   workload-cluster bootstrap 95% confidence intervals and paired significance tests.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
try:
    from sklearn.linear_model import Lasso, Ridge
except ImportError:  # pragma: no cover - fallback when sklearn is absent in serving venv
    Lasso = None
    Ridge = None


@dataclass(frozen=True)
class BlipPruningResult:
    """Output of two-stage backward minimal-sufficient provenance pruning (BLIP)."""

    sufficient_set: tuple[int, ...]
    scores: tuple[float, ...]  # binary indicator + fine-grained retention margin for ranking
    n_evals: int
    full_score: float
    pruned_score: float
    epsilon: float


def blip_two_stage_pruning(
    n: int,
    eval_mask: Callable[[int], float],
    *,
    epsilon: float = 0.05,
    block_size: int | None = None,
) -> BlipPruningResult:
    """Two-stage backward minimal-sufficient provenance pruning (BLIP [Lin et al. PVLDB'26]).

    Stage 1 (Coarse Block Pruning): partitions [0..n-1] into contiguous blocks of size
    b = block_size (default floor(sqrt(n))) and greedily drops any entire block whose
    removal keeps f(S) >= f(Omega) - epsilon.
    Stage 2 (Fine-Grained Chunk Refinement): iterates over surviving chunks and drops
    any single chunk whose removal keeps f(S) >= f(Omega) - epsilon, yielding a minimal
    sufficient provenance subset S*.

    To compare fairly on ranking metrics (Top-1 / Gold-Recall@k), surviving chunks are
    scored by their Stage-2 removal drop ``f(S*) - f(S* \\ {i})`` (strictly > 0 for
    critical chunks), while pruned chunks receive 0 (or coarse/fine slack <= 0).
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    full_mask = (1 << n) - 1
    cache: dict[int, float] = {}

    def query(msk: int) -> float:
        if msk not in cache:
            cache[msk] = float(eval_mask(msk))
        return cache[msk]

    full_val = query(full_mask)
    threshold = full_val - epsilon
    b = block_size if (block_size is not None and block_size >= 1) else max(2, math.isqrt(n))

    # Stage 1: Coarse block deletion
    cur_mask = full_mask
    for start in range(0, n, b):
        end = min(n, start + b)
        blk_bits = 0
        for i in range(start, end):
            blk_bits |= 1 << i
        cand = cur_mask & ~blk_bits
        if query(cand) >= threshold:
            cur_mask = cand

    # Stage 2: Fine-grained element-wise backward pruning inside surviving blocks
    survivors = [i for i in range(n) if (cur_mask >> i) & 1]
    drop_margins = [0.0] * n
    for i in survivors:
        cand = cur_mask & ~(1 << i)
        val_without = query(cand)
        if val_without >= threshold:
            cur_mask = cand
            drop_margins[i] = 0.0
        else:
            drop_margins[i] = full_val - val_without

    final_set = tuple(i for i in range(n) if (cur_mask >> i) & 1)
    pruned_val = query(cur_mask)
    # Score final minimal-sufficient chunks by their exact final necessity drop
    scores = [0.0] * n
    for i in final_set:
        val_without = query(cur_mask & ~(1 << i))
        scores[i] = max(1e-9, pruned_val - val_without)

    return BlipPruningResult(
        sufficient_set=final_set,
        scores=tuple(scores),
        n_evals=len(cache),
        full_score=full_val,
        pruned_score=pruned_val,
        epsilon=epsilon,
    )


def draw_sketching_loo_masks(
    n: int,
    budget: int,
    rng: random.Random,
    *,
    drop_prob: float = 0.25,
) -> list[int]:
    """Generate sparse group-deletion sketch masks (Sketching-LOO style):
    includes the full-context mask Omega plus (budget - 1) independent random
    sparse deletion masks where each chunk is dropped with probability ``drop_prob``."""
    if budget < 2:
        raise ValueError("budget must be >= 2")
    full = (1 << n) - 1
    masks = [full]
    for _ in range(budget - 1):
        msk = full
        dropped = 0
        for i in range(n):
            if rng.random() < drop_prob:
                msk &= ~(1 << i)
                dropped += 1
        if dropped == 0:
            msk &= ~(1 << rng.randrange(n))
        masks.append(msk)
    return masks


def fit_sketching_loo_scores(
    masks: Sequence[int],
    scores: Sequence[float],
    n: int,
    *,
    solver: str = "lasso",
    alpha: float = 1e-3,
) -> np.ndarray:
    """Recover per-chunk attribution weights from group-deletion sketch responses:
    models ``f(Omega) - f(S) ~= sum_{i not in S} w_i`` via L1 (LASSO) or L2 (Ridge)."""
    full = (1 << n) - 1
    # Identify full-context anchor score
    full_val = None
    for msk, val in zip(masks, scores, strict=True):
        if msk == full:
            full_val = float(val)
            break
    if full_val is None:
        full_val = float(max(scores))

    X_drop = np.zeros((len(masks), n), dtype=np.float64)
    y_drop = np.zeros(len(masks), dtype=np.float64)
    for r, (msk, val) in enumerate(zip(masks, scores, strict=True)):
        y_drop[r] = full_val - float(val)
        for i in range(n):
            if not ((msk >> i) & 1):
                X_drop[r, i] = 1.0

    if solver == "lasso":
        if Lasso is not None:
            mdl = Lasso(alpha=alpha, fit_intercept=False, max_iter=5000)
            mdl.fit(X_drop, y_drop)
            return mdl.coef_.astype(np.float64)
        # Pure-NumPy cyclic coordinate descent for (1/(2m))||Xw - y||^2 + alpha ||w||_1
        m_rows, n_feat = X_drop.shape
        w = np.zeros(n_feat, dtype=np.float64)
        col_norm2 = np.sum(X_drop * X_drop, axis=0) / m_rows
        r = y_drop.copy()
        for _ in range(1500):
            max_delta = 0.0
            for j in range(n_feat):
                cj = col_norm2[j]
                if cj <= 1e-15:
                    continue
                xj = X_drop[:, j]
                old = w[j]
                rho = float(np.dot(xj, r)) / m_rows + cj * old
                new = (rho - alpha) / cj if rho > alpha else ((rho + alpha) / cj if rho < -alpha else 0.0)
                if new != old:
                    r -= (new - old) * xj
                    w[j] = new
                    max_delta = max(max_delta, abs(new - old))
            if max_delta < 1e-8:
                break
        return w
    elif solver == "ridge":
        if Ridge is not None:
            mdl = Ridge(alpha=alpha, fit_intercept=False)
            mdl.fit(X_drop, y_drop)
            return mdl.coef_.astype(np.float64)
        gram = X_drop.T @ X_drop + alpha * np.eye(n, dtype=np.float64)
        return np.linalg.solve(gram, X_drop.T @ y_drop).astype(np.float64)
    else:
        raise ValueError(f"unknown solver {solver}")


def attntrace_single_pass_scores(
    attn_matrix: np.ndarray,
    chunk_token_spans: Sequence[tuple[int, int]],
    *,
    layer_slice: slice | None = None,
    head_top_k: int | None = None,
    length_normalize: bool = False,
) -> np.ndarray:
    """Single-pass cross-layer attention aggregation (AttnTrace-style white-box baseline).

    Parameters
    ----------
    attn_matrix : np.ndarray
        Attention weights from response tokens to context tokens, shape:
        - ``(n_layers, n_heads, n_resp_toks, seq_len)`` OR
        - ``(n_layers, n_resp_toks, seq_len)`` (already head-averaged).
    chunk_token_spans : Sequence[tuple[int, int]]
        Half-open token index spans ``(start, end)`` of each context chunk ``0..n-1``.
    layer_slice : slice | None
        Optional layer subset (default: upper half of layers ``L//2 :`` where
        retrieval/attribution heads concentrate).
    head_top_k : int | None
        If provided and 4D tensor is given, select top-k heads per layer by total
        attention mass directed onto context chunks (approximating AttnTrace head selection).
    length_normalize : bool
        If True, divide each chunk's summed attention mass by its token length.
    """
    arr = np.asarray(attn_matrix, dtype=np.float64)
    if arr.ndim == 3:
        # (L, T_resp, T_seq) -> add dummy head axis
        arr = arr[:, None, :, :]
    elif arr.ndim != 4:
        raise ValueError(f"expected 3D or 4D attention tensor, got shape {arr.shape}")

    n_layers, n_heads, _, seq_len = arr.shape
    sl = layer_slice if layer_slice is not None else slice(n_layers // 2, n_layers)
    sub = arr[sl]  # (L', H, T_resp, T_seq)
    # Average over response tokens -> (L', H, T_seq)
    resp_avg = sub.mean(axis=2)

    n_chunks = len(chunk_token_spans)
    chunk_mass = np.zeros((resp_avg.shape[0], n_heads, n_chunks), dtype=np.float64)
    for i, (s, e) in enumerate(chunk_token_spans):
        s_cl = max(0, min(seq_len, int(s)))
        e_cl = max(s_cl, min(seq_len, int(e)))
        if e_cl > s_cl:
            mass = resp_avg[:, :, s_cl:e_cl].sum(axis=2)
            if length_normalize:
                mass = mass / (e_cl - s_cl)
            chunk_mass[:, :, i] = mass

    if head_top_k is not None and 0 < head_top_k < n_heads:
        # Select heads with highest total attention onto the context chunks per layer
        head_energy = chunk_mass.sum(axis=2)  # (L', H)
        selected = np.zeros((chunk_mass.shape[0], n_chunks), dtype=np.float64)
        for l_idx in range(chunk_mass.shape[0]):
            top_h = np.argsort(-head_energy[l_idx])[:head_top_k]
            selected[l_idx] = chunk_mass[l_idx, top_h, :].mean(axis=0)
        return selected.mean(axis=0)

    return chunk_mass.mean(axis=(0, 1))


def cube_r2_average(blk0_effects: Sequence[Sequence[float]], blk1_effects: Sequence[Sequence[float]]) -> list[list[float]]:
    """Average two independent CUBE blocks (4M total budget) per question."""
    if len(blk0_effects) != len(blk1_effects):
        raise ValueError("block question counts must match")
    return [
        [0.5 * (float(a) + float(b)) for a, b in zip(r0, r1, strict=True)]
        for r0, r1 in zip(blk0_effects, blk1_effects, strict=True)
    ]


def workload_cluster_bootstrap_ci(
    per_workload_diffs: Sequence[float],
    *,
    weights: Sequence[float] | None = None,
    n_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 2027,
) -> dict[str, float]:
    """Non-parametric workload-cluster bootstrap CI (resampling workloads with replacement)."""
    vals = np.asarray(per_workload_diffs, dtype=np.float64)
    k = len(vals)
    if k < 2:
        raise ValueError("need at least 2 workload clusters for bootstrap")
    w = np.ones(k, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    if len(w) != k or np.any(w <= 0):
        raise ValueError("weights must be positive and match per_workload_diffs length")
    point = float(np.sum(w * vals) / np.sum(w))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(n_resamples, k))
    w_boot = w[idx]
    v_boot = vals[idx]
    boot_means = np.sum(w_boot * v_boot, axis=1) / np.sum(w_boot, axis=1)
    lo = float(np.quantile(boot_means, alpha / 2.0))
    hi = float(np.quantile(boot_means, 1.0 - alpha / 2.0))
    return {
        "point_estimate": point,
        "ci_low": lo,
        "ci_high": hi,
        "std_err": float(np.std(boot_means, ddof=1)),
        "clusters": float(k),
        "crosses_zero": float(lo <= 0.0 <= hi),
    }


def paired_sign_flip_pvalue(
    per_workload_diffs: Sequence[float],
    *,
    weights: Sequence[float] | None = None,
    n_resamples: int = 20000,
    seed: int = 2027,
) -> float:
    """Two-sided paired randomization (sign-flip) p-value across workload clusters."""
    vals = np.asarray(per_workload_diffs, dtype=np.float64)
    k = len(vals)
    if k < 1:
        raise ValueError("empty diffs")
    w = np.ones(k, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    obs = abs(float(np.sum(w * vals) / np.sum(w)))
    if k <= 16:
        # Exact enumeration over all 2^k sign assignments
        exceed = 0
        total = 1 << k
        for bits in range(total):
            sgn = np.array([1.0 if ((bits >> i) & 1) else -1.0 for i in range(k)], dtype=np.float64)
            stat = abs(float(np.sum(w * vals * sgn) / np.sum(w)))
            if stat >= obs - 1e-15:
                exceed += 1
        return exceed / total
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_resamples, k))
    stats = np.abs((signs * (w * vals)[None, :]).sum(axis=1) / np.sum(w))
    return float((np.sum(stats >= obs - 1e-15) + 1) / (n_resamples + 1))
