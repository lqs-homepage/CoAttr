"""Ordered Orthogonal-Array Designs for Context Attribution (CoAttr v2).

Supports two strength-3 fold-over Hadamard design families:
1. **Dyadic Ordered g-Cosets** (M = 2^p, g in {1, 2, 4, 8}, B = 2M):
   Standard CUBE (g = 1) and contiguous/rotated Walsh-Hadamard cosets.
2. **Kronecker Ordered Paley-Hadamard Cosets** (H_g kron H_L, M = g*L, B = 2*g*L):
   Combines Sylvester-Walsh H_g (g in {1, 2, 4, 8}) with Paley Type-I (L = p + 1,
   p == 3 mod 4) or Paley Type-II (L = 2*(q + 1), q == 1 mod 4 prime or p^2)
   Hadamard matrices.

Key theoretical guarantees verified for Kronecker Paley-Hadamard when L == 4 (mod 8):
- Exact strength-3 OA(2*g*L, n, 2, 3) after fold-over +/- Z: zero bias for all
  functions, exact recovery for all quadratic utilities, and exact Parseval
  energy conservation across all M = g*L orthonormal basis columns.
- Eliminates the 2x power-of-two budget cliff: unlocks dense non-power-of-two
  budgets B = 2*g*L in {48, 80, 96, 112, 144, 160, 176, 208, 224, 240, 288, ...}
  (e.g., B = 160 vs 256 at n = 65, 71: -37.5% rows; B = 224 vs 256 at n = 90).
- Preserves exact g-times early prefix compression: every segment in coset 0
  sees at most L = M / g distinct prefix histories (h_k^(+) <= L).
- Deterministic 0.00% single-block intra-coset 1-sparse blind spot (Theorem 2+):
  for ANY Hadamard matrix of order L == 4 (mod 8) and ANY four distinct columns
  a, b, c, d, <h_a * h_b * h_c, h_d> == 4 (mod 8), so |corr| >= 4/L > 0
  deterministically on every vacant column in the coset.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    d = 5
    while d * d <= n:
        if n % d == 0 or n % (d + 2) == 0:
            return False
        d += 6
    return True


def _prime_power_base(q: int) -> tuple[int, int] | None:
    """If q = p^k with prime p and k in {1, 2}, return (p, k); else None."""
    if q < 2:
        return None
    if _is_prime(q):
        return (q, 1)
    r = math.isqrt(q)
    if r * r == q and _is_prime(r):
        return (r, 2)
    return None


def _jacobsthal_prime(p: int) -> np.ndarray:
    """Exact p x p Jacobsthal matrix Q over GF(p) for odd prime p."""
    # Legendre symbol chi(x) = x^((p-1)//2) mod p mapped to {-1, 0, +1}
    chi = np.zeros(p, dtype=np.int8)
    exp = (p - 1) // 2
    for x in range(1, p):
        v = pow(x, exp, p)
        chi[x] = 1 if v == 1 else -1
    idx = np.arange(p, dtype=np.int32)
    diff = (idx[None, :] - idx[:, None]) % p
    return chi[diff]


def _jacobsthal_prime_square(p: int) -> np.ndarray:
    """Exact q x q Jacobsthal matrix Q over GF(p^2) for odd prime p (e.g., q=9, 25, 49)."""
    # Find a non-square d in GF(p) so GF(p^2) = GF(p)[t] / (t^2 - d)
    exp_p = (p - 1) // 2
    d_ns = next(x for x in range(2, p) if pow(x, exp_p, p) != 1)

    def mul(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
        return ((a[0] * b[0] + d_ns * a[1] * b[1]) % p, (a[0] * b[1] + a[1] * b[0]) % p)

    # Enumerate elements as (c0, c1) with index c0 * p + c1
    q = p * p
    squares: set[int] = set()
    for c0 in range(p):
        for c1 in range(p):
            if c0 == 0 and c1 == 0:
                continue
            s0, s1 = mul((c0, c1), (c0, c1))
            squares.add(s0 * p + s1)

    chi = np.full(q, -1, dtype=np.int8)
    chi[0] = 0
    for sq in squares:
        chi[sq] = 1

    Q = np.zeros((q, q), dtype=np.int8)
    for i in range(q):
        i0, i1 = divmod(i, p)
        for j in range(q):
            j0, j1 = divmod(j, p)
            d0 = (j0 - i0) % p
            d1 = (j1 - i1) % p
            Q[i, j] = chi[d0 * p + d1]
    return Q


def _jacobsthal_matrix(q: int) -> np.ndarray:
    bp = _prime_power_base(q)
    if bp is None or bp[0] == 2:
        raise ValueError(f"unsupported odd prime power q={q}")
    p, k = bp
    return _jacobsthal_prime(p) if k == 1 else _jacobsthal_prime_square(p)


def normalize_hadamard(H: np.ndarray) -> np.ndarray:
    """Normalize a Hadamard matrix so row 0 and column 0 are all +1."""
    H = np.asarray(H, dtype=np.int8).copy()
    # Flip rows where col 0 is -1
    H *= H[:, 0:1]
    # Flip cols where row 0 is -1
    H *= H[0:1, :]
    return H


@lru_cache(maxsize=64)
def sylvester_hadamard(g: int) -> np.ndarray:
    """Normalized Sylvester-Walsh-Hadamard matrix H_g for power-of-two g >= 1."""
    if g < 1 or (g & (g - 1)) != 0:
        raise ValueError(f"g must be a power of two, got {g}")
    H = np.array([[1]], dtype=np.int8)
    while H.shape[0] < g:
        H = np.block([[H, H], [H, -H]]).astype(np.int8)
    H.setflags(write=False)
    return H


@lru_cache(maxsize=64)
def paley_hadamard(L: int) -> np.ndarray:
    """Construct a normalized L x L Hadamard matrix via Paley Type-I or Type-II.

    Supports:
    - Power-of-two L via Sylvester (for completeness).
    - Paley Type-I: L = q + 1 where q == 3 (mod 4) is an odd prime (12, 20, 24, 44, 48, 60, 68, 84, ...).
    - Paley Type-II: L = 2*(q + 1) where q == 1 (mod 4) is p or p^2 (12, 20, 28, 36, 52, 60, 76, 84, 100, 108, ...).
    Covers EVERY L == 4 (mod 8) in {12, 20, 28, 36, 44, 52, 60, 68, 76, 84, 100, 108}.
    """
    if L < 1 or (L > 2 and L % 4 != 0):
        raise ValueError(f"Hadamard order L must be 1, 2, or a multiple of 4; got {L}")
    if (L & (L - 1)) == 0:
        return sylvester_hadamard(L)

    # Try Paley Type-I: q = L - 1 == 3 (mod 4)
    q1 = L - 1
    if q1 % 4 == 3 and _prime_power_base(q1) is not None:
        Q = _jacobsthal_matrix(q1)
        H = np.ones((L, L), dtype=np.int8)
        H[1:, 1:] = Q - np.eye(q1, dtype=np.int8)
        H = normalize_hadamard(H)
        H.setflags(write=False)
        return H

    # Try Paley Type-II: q = L // 2 - 1 == 1 (mod 4)
    if L % 2 == 0:
        q2 = L // 2 - 1
        if q2 % 4 == 1 and _prime_power_base(q2) is not None:
            Q = _jacobsthal_matrix(q2)
            C = np.ones((q2 + 1, q2 + 1), dtype=np.int8)
            C[0, 0] = 0
            C[1:, 1:] = Q
            A = np.array([[1, -1], [-1, -1]], dtype=np.int8)
            B = np.array([[1, 1], [1, -1]], dtype=np.int8)
            H = np.kron(np.eye(q2 + 1, dtype=np.int8), A) + np.kron(C, B)
            H = normalize_hadamard(H)
            H.setflags(write=False)
            return H

    raise ValueError(f"Paley Type-I/II construction not available for L={L}")


@lru_cache(maxsize=64)
def kronecker_paley_matrix(g: int, L: int) -> np.ndarray:
    """Normalized M x M Hadamard matrix K = H_g kron H_L (M = g * L).

    Coset r in {0, ..., g-1} occupies columns [r*L, (r+1)*L).
    Only column 0 = (0, 0) is the constant all-ones column; all other M - 1
    columns are balanced (+1/-1) and mutually orthogonal.
    """
    Hg = sylvester_hadamard(g)
    HL = paley_hadamard(L)
    K = np.kron(Hg, HL).astype(np.int8)
    K.setflags(write=False)
    return K


def supported_paley_orders(*, max_L: int = 84, mod8_eq4_only: bool = True) -> tuple[int, ...]:
    """Return sorted supported Paley Hadamard orders L."""
    orders = []
    for L in range(12, max_L + 1, 4):
        if mod8_eq4_only and (L % 8 != 4):
            continue
        try:
            paley_hadamard(L)
            orders.append(L)
        except ValueError:
            continue
    return tuple(orders)


def dyadic_half_budget(n: int) -> int:
    """Smallest power of two M strictly greater than n."""
    if n < 1:
        raise ValueError("n must be >= 1")
    return 1 << n.bit_length()


def cyclic_balanced_coset_tags(n: int, r_blocks: int, *, g: int = 2) -> list[tuple[int, ...]]:
    """Construct R cyclically rotated balanced g-coset tag vectors for n segments.

    Block b shifts the coset boundary by floor(b * n / (g * R)) modulo n,
    preserving exact coset sizes (floor/ceil(n/g)) and exact (gamma_g, kappa_g)
    in every block while separating mid-range contending pairs across blocks.
    """
    if g < 2 or (g & (g - 1)) != 0:
        raise ValueError("g must be a power of two >= 2")
    if r_blocks < 1:
        raise ValueError("r_blocks must be >= 1")
    base = [j for j in range(g) for _ in range((j + 1) * n // g - j * n // g)]
    out: list[tuple[int, ...]] = []
    for b in range(r_blocks):
        shift = (b * n) // (g * r_blocks)
        rotated = tuple(base[(i - shift) % n] for i in range(n))
        out.append(rotated)
    return out


@dataclass(frozen=True)
class DesignBlock:
    """One materialized strength-3 fold-over Hadamard design block (2*M rows)."""

    family: str  # "dyadic" or "kronecker_paley"
    n: int
    m: int
    g: int
    L: int
    tags: tuple[int, ...]
    column_indices: tuple[int, ...]
    shift: int
    rows: tuple[int, ...]  # 2*m integer bitmasks in complementary pairs (2t, 2t+1)


def draw_kronecker_paley_block(
    n: int,
    g: int,
    L: int,
    rng: random.Random,
    *,
    tags: Sequence[int] | None = None,
) -> DesignBlock:
    """Draw a strength-3 OA(2*g*L, n, 2, 3) from K = H_g kron H_L with random
    within-coset column assignment and independent uniform Rademacher sign shift.

    Capacity constraints:
    - Coset 0 excludes column 0 (constant all-ones), so n_0 <= L - 1.
    - Cosets r >= 1 use columns [r*L, (r+1)*L), so n_r <= L (prefer <= L - 1 to
      retain at least 1 vacant column inside every coset for 0.00% blind spot).
    - When ``tags is None``, uses balanced contiguous cosets (respecting n_0 <= L-1).
    """
    m = g * L
    if not (1 <= n < m):
        raise ValueError(f"require 1 <= n < g*L={m}, got n={n}")
    K = kronecker_paley_matrix(g, L)

    if tags is None:
        # Balanced contiguous partition; ensure coset 0 has <= L - 1 segments
        sizes = [(j + 1) * n // g - j * n // g for j in range(g)]
        if sizes[0] >= L:
            # Shift one segment from coset 0 to the first coset with room
            for j in range(1, g):
                if sizes[j] < L:
                    sizes[0] -= 1
                    sizes[j] += 1
                    break
        tag_tuple = tuple(j for j, sz in enumerate(sizes) for _ in range(sz))
    else:
        if len(tags) != n:
            raise ValueError("tags length must equal n")
        tag_tuple = tuple(int(t) for t in tags)

    by_group: dict[int, list[int]] = defaultdict(list)
    for idx, t in enumerate(tag_tuple):
        if not (0 <= t < g):
            raise ValueError(f"tag {t} out of range [0, {g})")
        by_group[t].append(idx)

    col_indices = [0] * n
    for r, indices in by_group.items():
        pool = list(range(1, L)) if r == 0 else list(range(r * L, (r + 1) * L))
        if len(indices) > len(pool):
            raise ValueError(f"coset {r} assigned {len(indices)} segments > capacity {len(pool)}")
        drawn = rng.sample(pool, len(indices))
        for idx, col in zip(indices, drawn, strict=True):
            col_indices[idx] = col

    # Independent uniform Rademacher sign shift s in {0, 1}^n (bit i == 1 means flip)
    shift = rng.getrandbits(n)
    all_bits = (1 << n) - 1
    sub = K[:, col_indices]  # shape (m, n), entries in {-1, +1}
    rows: list[int] = []
    for t in range(m):
        mask = 0
        row_t = sub[t]
        for i in range(n):
            # Map +1 -> bit 1, -1 -> bit 0, then XOR with random shift bit
            bit = 1 if row_t[i] == 1 else 0
            if (shift >> i) & 1:
                bit ^= 1
            if bit:
                mask |= 1 << i
        rows.append(mask)
        rows.append(all_bits ^ mask)

    return DesignBlock(
        family="kronecker_paley",
        n=n,
        m=m,
        g=g,
        L=L,
        tags=tag_tuple,
        column_indices=tuple(col_indices),
        shift=shift,
        rows=tuple(rows),
    )


def draw_dyadic_block(
    n: int,
    m: int,
    rng: random.Random,
    *,
    tags: Sequence[int] | None = None,
    g: int = 1,
) -> DesignBlock:
    """Draw a dyadic CUBE (g=1) or Ordered g-Coset block (M = 2^p)."""
    if not (1 <= n < m) or (m & (m - 1)) != 0:
        raise ValueError("m must be a power of two strictly greater than n")
    if tags is None:
        if g == 1:
            tag_tuple = tuple([0] * n)
            col_indices = rng.sample(range(1, m), n)
        else:
            sizes = [(j + 1) * n // g - j * n // g for j in range(g)]
            tag_tuple = tuple(j for j, sz in enumerate(sizes) for _ in range(sz))
            width = m // g
            col_indices = []
            for r, sz in enumerate(sizes):
                if sz > width:
                    raise ValueError(f"coset {r} size {sz} > {width}")
                pool = range(1, width) if (r == 0 and sz < width) else range(r * width, (r + 1) * width)
                # Note: standard dyadic Stage 2/3 allows label 0 in coset 0? Wait:
                # label 0 is constant all-+1, which must NEVER be assigned to an active feature!
                # Check: range(1, width) has width-1 non-constant labels!
                if r == 0 and sz > width - 1:
                    raise ValueError(f"coset 0 cannot hold {sz} non-constant columns (max {width - 1})")
                col_indices.extend(rng.sample(list(pool), sz))
    else:
        if len(tags) != n:
            raise ValueError("tags length must equal n")
        if g < 1 or (g & (g - 1)) != 0 or m % g != 0:
            raise ValueError("g must be a power of two dividing m")
        tag_tuple = tuple(int(t) for t in tags)
        width = m // g
        by_group: dict[int, list[int]] = defaultdict(list)
        for idx, t in enumerate(tag_tuple):
            by_group[t].append(idx)
        col_indices = [0] * n
        for r, indices in by_group.items():
            pool = list(range(1, width)) if r == 0 else list(range(r * width, (r + 1) * width))
            if len(indices) > len(pool):
                raise ValueError(f"coset {r} has {len(indices)} segments > non-constant capacity {len(pool)}")
            drawn = rng.sample(pool, len(indices))
            for idx, lbl in zip(indices, drawn, strict=True):
                col_indices[idx] = lbl

    shift = rng.getrandbits(n)
    all_bits = (1 << n) - 1
    rows: list[int] = []
    for t in range(m):
        mask = shift
        for i, lbl in enumerate(col_indices):
            if (lbl & t).bit_count() & 1:
                mask ^= 1 << i
        rows.append(mask)
        rows.append(all_bits ^ mask)

    return DesignBlock(
        family="dyadic",
        n=n,
        m=m,
        g=g,
        L=m // g,
        tags=tag_tuple,
        column_indices=tuple(col_indices),
        shift=shift,
        rows=tuple(rows),
    )


def exact_prefix_history_counts(rows: Sequence[int], n: int) -> list[int]:
    """Exact number of distinct prefixes seen before kept segment k (bit k == 1)
    across the materialized coalition masks ``rows``."""
    uniq = list(dict.fromkeys(rows))
    counts: list[int] = []
    for k in range(n):
        prefix_mask = (1 << k) - 1
        bit_k = 1 << k
        seen = {msk & prefix_mask for msk in uniq if (msk & bit_k)}
        counts.append(len(seen))
    return counts


def expected_paley_prefix_history_counts(
    n: int,
    g: int,
    L: int,
    *,
    tags: Sequence[int] | None = None,
    mc_draws: int = 16,
    seed: int = 2027,
) -> list[float]:
    """Deterministic expected prefix-history counts h_k^(+) for Kronecker Paley-Hadamard."""
    rng = random.Random(seed)
    acc = np.zeros(n, dtype=np.float64)
    for _ in range(mc_draws):
        blk = draw_kronecker_paley_block(n, g, L, rng, tags=tags)
        acc += np.array(exact_prefix_history_counts(blk.rows, n), dtype=np.float64)
    return (acc / mc_draws).tolist()


def select_best_paley_configs(
    n: int,
    *,
    groups: Sequence[int] = (2, 4),
    mod8_eq4_only: bool = True,
    min_vacant_per_coset: int = 1,
    strictly_smaller_than_dyadic: bool = True,
) -> list[dict[str, object]]:
    """Enumerate valid Kronecker Ordered Paley-Hadamard (g, L) configurations for n segments.

    Ensures:
    - coset 0 holds n_0 <= L - 1 - max(0, min_vacant_per_coset - 1) <= L - 1
      (note: column (0,0) is already a vacant non-active channel in Parseval energy,
       and requiring n_r <= L - min_vacant_per_coset guarantees >= min_vacant_per_coset
       vacant columns in EVERY coset, including >= 1 non-constant vacant column in coset 0
       when min_vacant_per_coset >= 2).
    - total half-budget M = g * L < dyadic_half_budget(n) if strictly_smaller_than_dyadic.
    """
    m_dyad = dyadic_half_budget(n)
    orders = supported_paley_orders(max_L=108, mod8_eq4_only=mod8_eq4_only)
    configs: list[dict[str, object]] = []
    for g in groups:
        max_coset_size = math.ceil(n / g)
        # Need L - 1 >= max_coset_size (since coset 0 has L - 1 non-constant columns)
        # and L - max_coset_size >= min_vacant_per_coset
        min_L = max(max_coset_size + 1, max_coset_size + min_vacant_per_coset)
        for L in orders:
            if L < min_L:
                continue
            m_pal = g * L
            if m_pal <= n:
                continue
            if strictly_smaller_than_dyadic and m_pal >= m_dyad:
                continue
            configs.append(
                {
                    "family": "kronecker_paley",
                    "name": f"paley_g{g}_L{L}",
                    "g": g,
                    "L": L,
                    "m": m_pal,
                    "budget_2m": 2 * m_pal,
                    "dyadic_m": m_dyad,
                    "row_saving_frac": round(1.0 - m_pal / m_dyad, 4),
                    "vacant_dof": m_pal - n,
                    "zero_intra_blind_spot": (L % 8 == 4),
                }
            )
            # Keep the smallest valid L (tightest budget) AND next L (more vacant DOF) per g
            if sum(1 for c in configs if c["g"] == g) >= 2:
                break
    configs.sort(key=lambda c: (c["m"], -int(c["g"])))
    return configs


def verify_oa_strength3(rows: Sequence[int], n: int, *, sample_triples: int = 200, seed: int = 0) -> dict[str, object]:
    """Verify exact strength-3 OA(B, n, 2, 3) moments on a materialized bitmask list."""
    B = len(rows)
    Z = np.empty((B, n), dtype=np.int8)
    for r, msk in enumerate(rows):
        for i in range(n):
            Z[r, i] = 1 if ((msk >> i) & 1) else -1
    max_s1 = int(np.max(np.abs(Z.sum(axis=0))))
    G = Z.astype(np.int32).T @ Z.astype(np.int32)
    np.fill_diagonal(G, 0)
    max_s2 = int(np.max(np.abs(G)))

    rng = random.Random(seed)
    max_s3 = 0
    if n >= 3:
        if math.comb(n, 3) <= sample_triples:
            triples = [(i, j, k) for i in range(n) for j in range(i + 1, n) for k in range(j + 1, n)]
        else:
            triples = [tuple(sorted(rng.sample(range(n), 3))) for _ in range(sample_triples)]
        for i, j, k in triples:
            val = int(np.sum(Z[:, i].astype(np.int32) * Z[:, j].astype(np.int32) * Z[:, k].astype(np.int32)))
            if abs(val) > max_s3:
                max_s3 = abs(val)
    return {
        "rows": B,
        "n": n,
        "max_abs_order1": max_s1,
        "max_abs_order2": max_s2,
        "max_abs_order3": max_s3,
        "is_strength3_oa": (max_s1 == 0 and max_s2 == 0 and max_s3 == 0),
    }


def verify_mod8_eq4_nonzero_quadruples(L: int, *, max_triples: int = 500, seed: int = 7) -> dict[str, object]:
    """Verify Theorem 2+ congruence: for L == 4 (mod 8), EVERY 4 distinct columns
    (a, b, c, d) of H_L satisfy <h_a * h_b * h_c, h_d> == 4 (mod 8) and |corr| >= 4/L."""
    H = paley_hadamard(L).astype(np.int32)
    rng = random.Random(seed)
    total_triples = math.comb(L, 3)
    if total_triples <= max_triples:
        triples = [(a, b, c) for a in range(L) for b in range(a + 1, L) for c in range(b + 1, L)]
    else:
        triples = [tuple(sorted(rng.sample(range(L), 3))) for _ in range(max_triples)]

    min_abs_inner = L
    max_abs_inner = 0
    congruence_violations = 0
    zero_count = 0
    checked_quads = 0

    for a, b, c in triples:
        prod = H[:, a] * H[:, b] * H[:, c]
        inners = H.T @ prod  # inner product with all L columns
        for d in range(L):
            if d in (a, b, c):
                assert int(inners[d]) == 0
                continue
            v = int(inners[d])
            checked_quads += 1
            if v == 0:
                zero_count += 1
            if abs(v) % 8 != 4:
                congruence_violations += 1
            if abs(v) < min_abs_inner:
                min_abs_inner = abs(v)
            if abs(v) > max_abs_inner:
                max_abs_inner = abs(v)

    return {
        "L": L,
        "L_mod_8": L % 8,
        "triples_checked": len(triples),
        "quads_checked": checked_quads,
        "zero_quads": zero_count,
        "mod8_eq4_violations": congruence_violations,
        "min_abs_corr": min_abs_inner / L,
        "max_abs_corr": max_abs_inner / L,
        "peak_sq_aliasing_vs_dyadic": (max_abs_inner / L) ** 2,
    }


def decode_hadamard_block(
    rows: Sequence[int],
    scores_by_mask: dict[int, Sequence[float]],
    n: int,
    nq: int,
) -> tuple[list[list[float]], list[list[float]]]:
    """Decode exact unbiased Banzhaf main effects B[q][i] and odd half-differences O[q][t]
    for ANY strength-3 fold-over Hadamard design (Dyadic or Kronecker Paley-Hadamard)."""
    m = len(rows) // 2
    effects = [[0.0] * n for _ in range(nq)]
    odd_by_q = [[0.0] * m for _ in range(nq)]
    for t in range(m):
        m_pos = rows[2 * t]
        m_neg = rows[2 * t + 1]
        s_pos = scores_by_mask[m_pos]
        s_neg = scores_by_mask[m_neg]
        for q in range(nq):
            o_val = 0.5 * (s_pos[q] - s_neg[q])
            odd_by_q[q][t] = o_val
            for i in range(n):
                sgn = 1.0 if ((m_pos >> i) & 1) else -1.0
                effects[q][i] += 2.0 * sgn * o_val / m
    return effects, odd_by_q
