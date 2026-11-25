"""Mathematically sound risk-safe selection, vacant-channel variance certification,
and contender-separating coset allocation for ordered context attribution (P-18).

Key guarantees enforced by this module:
1. Standalone-baseline net-saving contract: candidate end-to-end cost
   (``pilot_overhead + final_cost + correction_cost``) is compared against the
   standalone baseline cost (``baseline_cost``) without cancelling pilot overhead.
2. Dual-layer risk safety:
   - Zero-pilot analytical gate: exact worst-case total-MSE inflation ``gamma``
     and optional vacant-channel conservative calibration constant ``kappa``.
   - Empirical multi-sample gate: simultaneous Bonferroni + union-bound
     min(Empirical Bernstein, one-sided Hoeffding) upper bound on observable
     paired risk differences against CUBE, with explicit risk tolerance.
3. Zero-extra-query vacant Walsh-channel Parseval variance monitor with exact
   finite-sample conservative factor ``kappa_D`` (polynomial DP).
4. Contender-separating coset allocation with exact GF(2) projection-rank
   prefix-history recursion for arbitrary per-segment coset tag sequences.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from math import comb, isfinite, log, sqrt
from typing import Iterable, Sequence


def _finite_nonnegative(value: float, name: str) -> float:
    value = float(value)
    if not isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


@lru_cache(maxsize=None)
def signed_population(m: int, s: int) -> Fraction:
    """Finite-population signed hypergeometric character moment k_m(s)."""
    if s < 0 or s > m or s % 2 == 1:
        return Fraction(0)
    return Fraction((-1) ** (s // 2) * comb(m // 2, s // 2), comb(m, s))


@lru_cache(maxsize=None)
def native_alias(m: int, s: int) -> Fraction:
    """Exact aliasing probability of an even-order character in standard CUBE."""
    if s < 0 or s > m or s % 2 == 1:
        return Fraction(0)
    return (1 + (m - 1) * signed_population(m, s)) / m


def exact_kappa_cube(n: int, m: int) -> float:
    """Exact distribution-free conservative calibration constant kappa for CUBE.

    Ensures ``E[kappa * vacant_channel_var] >= (1/n) sum_i Var(B_hat_i)`` over
    every bounded response ``f`` when ``m > n``.  Returns ``inf`` if some odd
    degree attains full aliasing weight ``W_o(s) >= 1``.
    """
    if not (1 <= n < m) or (m & (m - 1)):
        raise ValueError("m must be a power of two strictly greater than n")
    worst = 0.0
    for s in range(3, n + 1, 2):
        wo = float(s * native_alias(m, s - 1) + (n - s) * native_alias(m, s + 1))
        if wo >= 1.0 - 1e-12:
            return float("inf")
        ratio = ((m - n) * wo) / (n * (1.0 - wo))
        if ratio > worst:
            worst = ratio
    return worst


def exact_coset_certificates(sizes: Sequence[int], m: int) -> tuple[float, float]:
    """Exact polynomial-DP worst-case total-MSE ratio ``gamma`` and vacant-channel
    conservative calibration factor ``kappa`` for arbitrary coset sizes.

    ``sizes`` gives the segment counts ``(n_0, ..., n_{g-1})`` across ``g = 2^d``
    cosets of width ``L = m // g``.  Because aliasing probabilities depend only
    on coset sizes (not within-coset segment positions), the exact same bounds
    apply to contiguous ordered cosets and to contender-separated coset partitions.
    """
    g = len(sizes)
    if g < 1 or (g & (g - 1)) or m % g != 0 or (m & (m - 1)):
        raise ValueError("g and m must be powers of two with g dividing m")
    n = sum(sizes)
    h = m // g
    if not (1 <= n < m) or any(sz < 0 or sz > h for sz in sizes):
        raise ValueError("coset sizes must be non-negative, <= m/g, and sum to n < m")
    if g == 1:
        return 1.0, exact_kappa_cube(n, m)

    k = [float(signed_population(h, v)) for v in range(h + 1)]
    po = [float(native_alias(m, v)) if v <= m else 0.0 for v in range(n + 2)]

    worst_gamma = 0.0
    worst_kappa = 0.0

    for eligible, nh in enumerate(sizes):
        if nh == 0:
            continue
        # DP tables indexed by (other_degree_sum t, xor_tag x) -> (min_prod, max_prod)
        lo = [[float("inf")] * g]
        hi = [[float("-inf")] * g]
        lo[0][0] = 1.0
        hi[0][0] = 1.0

        for tag, ng in enumerate(sizes):
            if tag == eligible:
                continue
            span = len(lo)
            next_lo = [[float("inf")] * g for _ in range(span + ng)]
            next_hi = [[float("-inf")] * g for _ in range(span + ng)]
            for t in range(span):
                row_lo = lo[t]
                row_hi = hi[t]
                for x in range(g):
                    lval = row_lo[x]
                    if not isfinite(lval):
                        continue
                    hval = row_hi[x]
                    for u in range(ng + 1):
                        ku = k[u]
                        nx = x ^ (tag if (u & 1) else 0)
                        a = lval * ku
                        b = hval * ku
                        if a > b:
                            a, b = b, a
                        nt = t + u
                        if a < next_lo[nt][nx]:
                            next_lo[nt][nx] = a
                        if b > next_hi[nt][nx]:
                            next_hi[nt][nx] = b
            lo, hi = next_lo, next_hi

        for u in range(nh + 1):
            wanted = 0 if (u & 1) else eligible
            coeff = (u * k[u - 1] if u > 0 else 0.0) + ((nh - u) * k[u + 1] if u < nh else 0.0)
            for t in range(len(lo)):
                s = t + u
                if s < 3 or (s & 1) == 0:
                    continue
                lval = lo[t][wanted]
                if not isfinite(lval):
                    continue
                prod = hi[t][wanted] if coeff >= 0.0 else lval
                wd = (nh + (h - 1) * coeff * prod) / h
                old = s * po[s - 1] + (n - s) * po[s + 1]
                if old > 1e-12:
                    ratio_g = wd / old
                    if ratio_g > worst_gamma:
                        worst_gamma = ratio_g
                elif wd > 1e-12:
                    worst_gamma = float("inf")

                if wd >= 1.0 - 1e-12:
                    worst_kappa = float("inf")
                elif isfinite(worst_kappa):
                    ratio_k = ((m - n) * wd) / (n * (1.0 - wd))
                    if ratio_k > worst_kappa:
                        worst_kappa = ratio_k

    return worst_gamma, worst_kappa


def general_tag_history_counts(tags: Sequence[int], m: int, g: int) -> list[float]:
    """Exact expected prefix-history counts ``h_0, ..., h_{n-1}`` for ANY sequence
    of coset tags ``tags[i] in {0, ..., g-1}`` using exact GF(2) subspace rank.

    When coset tags appear monotonically ``0, 1, ...``, this agrees identically
    with ``planning.group_history_counts``; it also supports contender-separated
    (non-contiguous) partitions where a late contender is placed in an earlier coset.
    """
    if g < 1 or (g & (g - 1)) or m % g != 0 or (m & (m - 1)):
        raise ValueError("g and m must be powers of two with g dividing m")
    if not tags:
        return []
    h = m // g
    # Maintain GF(2) row basis of seen coset-tag differences (tag ^ tags[0]).
    anchor = int(tags[0])
    basis: list[int] = []

    def insert_basis(vec: int) -> bool:
        v = vec
        for b in basis:
            v = min(v, v ^ b)
        if v:
            basis.append(v)
            basis.sort(reverse=True)
            return True
        return False

    profile: dict[int, float] = {0: 1.0}
    counts: list[float] = [1.0]
    projected = 0
    used = [0] * g
    used[anchor] = 1

    for raw_tag in tags[1:]:
        tag = int(raw_tag)
        if not (0 <= tag < g):
            raise ValueError(f"coset tag {tag} out of range [0, {g})")
        if used[tag] >= h:
            raise ValueError(f"coset {tag} exceeds capacity {h}")
        increase = insert_basis(tag ^ anchor)
        nxt: dict[int, float] = defaultdict(float)
        for dim, mass in profile.items():
            if increase:
                same = 0.0
            else:
                inside = (1 << (dim - projected)) - used[tag]
                same = inside / (h - used[tag])
            if not (-1e-12 <= same <= 1.0 + 1e-12):
                raise AssertionError(f"invalid conditional stay probability {same}")
            same = min(1.0, max(0.0, same))
            if same > 0.0:
                nxt[dim] += mass * same
            if same < 1.0:
                nxt[dim + 1] += mass * (1.0 - same)
        if increase:
            projected += 1
        used[tag] += 1
        profile = nxt
        counts.append(sum((1 << d) * p for d, p in profile.items()))
    return counts


def vacant_channel_variance(odd_values: Sequence[float], effects: Sequence[float], m: int) -> float:
    """Zero-extra-query Parseval vacant-channel variance estimate for one question.

    ``odd_values`` are the ``M`` complementary half-differences ``(f(z_t) - f(-z_t))/2``.
    ``effects`` are the ``n`` estimated Banzhaf main effects ``B_hat_i = 2 * theta_hat_i``.
    Returns ``(4 / (M - n)) * max(0, mean(O^2) - sum((B_hat_i / 2)^2))``.
    """
    n = len(effects)
    if len(odd_values) != m:
        raise ValueError(f"expected {m} odd complementary values, got {len(odd_values)}")
    if n >= m:
        raise ValueError("vacant-channel estimation requires strictly more rows M than sources n")
    e_total = sum(v * v for v in odd_values) / m
    e_occ = sum((0.5 * b) ** 2 for b in effects)
    return 4.0 * max(0.0, e_total - e_occ) / (m - n)


@dataclass(frozen=True)
class GapCertificate:
    """Zero-extra-query vacant-channel SNR certificate for one question."""

    top1: int
    top2: int
    estimated_gap: float
    vacant_variance: float
    vacant_dof: int
    dof_lower_tail_inflation: float
    calibrated_segment_variance: float
    estimated_gap_std: float
    snr: float
    same_coset: bool
    needs_confirmation: bool
    gate_reason: str


def _wilson_hilferty_variance_inflation(dof: int, *, z_lower: float = 1.6448536269514722) -> float:
    """Finite-DOF chi-square lower-tail reference inflation ``1 / q_{alpha}(chi2_d / d)``.

    Note: ``kappa_D`` guarantees an exact distribution-free *expectation* upper bound
    on per-segment MSE, NOT single-block high-probability coverage.  This reference
    inflation protects single-block SNR gating against left-tail underestimates of
    finite-DOF Parseval variance (and is paired with a hard ``min_vacant_dof`` guard).
    """
    if dof < 1:
        return float("inf")
    a = 2.0 / (9.0 * dof)
    cube_root = 1.0 - a - z_lower * sqrt(a)
    if cube_root <= 0.1:
        return float("inf")
    q_low = cube_root ** 3
    return 1.0 / q_low


def gap_snr_certificate(
    effects: Sequence[float],
    vacant_var: float,
    *,
    vacant_dof: int,
    kappa: float = 1.0,
    coset_of: Sequence[int] | None = None,
    same_coset_factor: float = 2.0,
    z_threshold: float = 2.0,
    min_vacant_dof: int = 16,
    min_gap_floor: float = 0.0,
) -> GapCertificate:
    """Certify top-1 vs top-2 separation from zero-extra-query vacant-channel variance.

    Conservative single-block safety guards:
    - If ``vacant_dof < min_vacant_dof`` (e.g., ``QuALITY-99914`` with ``M-n=4``)
      or ``kappa`` is non-finite, conservatively forces ``needs_confirmation = True``
      to prevent left-tail false-safety certification.
    - Otherwise inflates ``kappa * vacant_var`` by the Wilson–Hilferty finite-DOF
      lower-tail reference factor and triggers confirmation if
      ``gap <= z_threshold * gap_std`` or ``gap < min_gap_floor``.
    """
    if len(effects) < 2:
        raise ValueError("at least two segments are required to form a ranking gap")
    if vacant_dof < 0:
        raise ValueError("vacant_dof must be non-negative")
    order = sorted(range(len(effects)), key=lambda i: (-effects[i], i))
    top1, top2 = order[0], order[1]
    gap = float(effects[top1] - effects[top2])
    vacant_var = _finite_nonnegative(vacant_var, "vacant_var")
    same_coset = False if coset_of is None else (int(coset_of[top1]) == int(coset_of[top2]))

    if not isfinite(kappa):
        return GapCertificate(
            top1=top1,
            top2=top2,
            estimated_gap=gap,
            vacant_variance=vacant_var,
            vacant_dof=vacant_dof,
            dof_lower_tail_inflation=float("inf"),
            calibrated_segment_variance=float("inf"),
            estimated_gap_std=float("inf"),
            snr=0.0,
            same_coset=same_coset,
            needs_confirmation=True,
            gate_reason="non-finite kappa (saturated coset)",
        )
    if vacant_dof < min_vacant_dof:
        return GapCertificate(
            top1=top1,
            top2=top2,
            estimated_gap=gap,
            vacant_variance=vacant_var,
            vacant_dof=vacant_dof,
            dof_lower_tail_inflation=float("inf"),
            calibrated_segment_variance=float("inf"),
            estimated_gap_std=float("inf"),
            snr=0.0,
            same_coset=same_coset,
            needs_confirmation=True,
            gate_reason=f"vacant_dof {vacant_dof} below min_vacant_dof {min_vacant_dof}",
        )

    dof_infl = _wilson_hilferty_variance_inflation(vacant_dof)
    cal_var = kappa * dof_infl * vacant_var
    pair_mult = same_coset_factor if same_coset else 1.0
    gap_var = 2.0 * pair_mult * cal_var
    gap_std = sqrt(max(0.0, gap_var))
    if gap_std == 0.0:
        # Guard against rare zero single-block vacant variance realization on non-trivial gaps
        snr = float("inf") if gap > min_gap_floor else 0.0
        needs = gap <= max(min_gap_floor, 1e-4)
        reason = "zero vacant variance below safety gap floor" if needs else "certified (zero noise, large gap)"
    else:
        snr = gap / gap_std
        needs = (gap <= z_threshold * gap_std) or (gap < min_gap_floor)
        reason = "gap below finite-DOF SNR threshold" if needs else "certified above finite-DOF SNR threshold"
    return GapCertificate(
        top1=top1,
        top2=top2,
        estimated_gap=gap,
        vacant_variance=vacant_var,
        vacant_dof=vacant_dof,
        dof_lower_tail_inflation=dof_infl,
        calibrated_segment_variance=cal_var,
        estimated_gap_std=gap_std,
        snr=snr,
        same_coset=same_coset,
        needs_confirmation=needs,
        gate_reason=reason,
    )


@dataclass(frozen=True)
class CosetPartition:
    """A deterministic assignment of ``n`` segments into ``g`` cosets."""

    tags: tuple[int, ...]
    groups: int
    sizes: tuple[int, ...]
    gamma: float
    kappa: float
    separated_pair: tuple[int, int] | None
    is_contiguous: bool


def contender_separating_partition(
    n: int,
    m: int,
    contenders: tuple[int, int],
    *,
    g: int = 2,
    max_gamma: float = 1.05,
) -> CosetPartition | None:
    """Construct a coset partition that separates ``contenders = (i1, i2)`` across
    distinct cosets while preserving document order as much as possible.

    For ``g == 2``:
    1. First searches contiguous ordered cuts ``c`` strictly between ``min(i1, i2)``
       and ``max(i1, i2)`` with ``gamma <= max_gamma`` (zero out-of-order segments!).
    2. If no contiguous separating cut meets ``max_gamma``, takes the balanced
       contiguous cut ``c = n // 2`` and minimally swaps the later contender (if
       both are early) or earlier contender (if both are late) with the adjacent
       boundary segment from the opposite coset, keeping exact coset sizes and
       100% Banzhaf unbiasedness.
    """
    i1, i2 = int(contenders[0]), int(contenders[1])
    if i1 == i2 or not (0 <= i1 < n) or not (0 <= i2 < n):
        raise ValueError("contenders must be two distinct valid segment indices")
    if g < 2 or (g & (g - 1)) or m % g != 0 or n < g:
        return None
    h = m // g
    lo_idx, hi_idx = (i1, i2) if i1 < i2 else (i2, i1)

    if g == 2:
        low_cut = max(1, n - h + 1)
        high_cut = min(n - 1, h - 1)
        # Step 1: try contiguous separating cuts first (prefer cut closest to n // 2)
        candidates_cut = [c for c in range(max(low_cut, lo_idx + 1), min(high_cut, hi_idx) + 1)]
        candidates_cut.sort(key=lambda c: (abs(c - n / 2.0), c))
        for cut in candidates_cut:
            sizes = (cut, n - cut)
            gamma, kappa = exact_coset_certificates(sizes, m)
            if gamma <= max_gamma:
                tags = tuple(0 if i < cut else 1 for i in range(n))
                return CosetPartition(tags, 2, sizes, gamma, kappa, (i1, i2), True)

        # Step 2: balanced cut + minimal single swap to separate contenders
        cut = n // 2
        if not (low_cut <= cut <= high_cut):
            return None
        sizes = (cut, n - cut)
        gamma, kappa = exact_coset_certificates(sizes, m)
        if gamma > max_gamma:
            return None
        tags_list = [0 if i < cut else 1 for i in range(n)]
        if tags_list[i1] == tags_list[i2]:
            if tags_list[i1] == 0:
                # Both early: move the later contender (hi_idx) to coset 1; bring cut (first late) to coset 0
                partner = cut
                tags_list[hi_idx] = 1
                tags_list[partner] = 0
            else:
                # Both late: move the earlier contender (lo_idx) to coset 0; bring cut-1 (last early) to coset 1
                partner = cut - 1
                tags_list[lo_idx] = 0
                tags_list[partner] = 1
            return CosetPartition(tuple(tags_list), 2, sizes, gamma, kappa, (i1, i2), False)
        return CosetPartition(tuple(tags_list), 2, sizes, gamma, kappa, (i1, i2), True)

    # General g = 2^d >= 4: start from balanced contiguous ranges, swap second contender to adjacent coset if needed
    sizes_list = [(j + 1) * n // g - j * n // g for j in range(g)]
    if any(sz == 0 or sz >= h for sz in sizes_list):
        return None
    gamma, kappa = exact_coset_certificates(sizes_list, m)
    if gamma > max_gamma:
        return None
    tags_list = [j for j, sz in enumerate(sizes_list) for _ in range(sz)]
    is_contig = True
    if tags_list[i1] == tags_list[i2]:
        grp = tags_list[i1]
        other_grp = grp + 1 if grp + 1 < g else grp - 1
        # Pick boundary segment in other_grp nearest to hi_idx / lo_idx
        target_seg = hi_idx if other_grp > grp else lo_idx
        other_indices = [idx for idx, t in enumerate(tags_list) if t == other_grp]
        partner = other_indices[0] if other_grp > grp else other_indices[-1]
        tags_list[target_seg] = other_grp
        tags_list[partner] = grp
        is_contig = False
    return CosetPartition(tuple(tags_list), g, tuple(sizes_list), gamma, kappa, (i1, i2), is_contig)


@dataclass(frozen=True)
class CandidatePlan:
    """One non-baseline plan with analytical bounds, optional empirical pilot
    observations, and its independent delivery + correction price.
    """

    name: str
    final_cost: float
    observations: tuple[float, ...] | None = None
    correction_cost: float = 0.0
    gamma: float = 1.0
    kappa: float = 1.0
    lower: float = -1.0
    upper: float = 1.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("candidate name must be non-empty")
        final_cost = _finite_nonnegative(self.final_cost, "final_cost")
        correction_cost = _finite_nonnegative(self.correction_cost, "correction_cost")
        gamma = float(self.gamma)
        kappa = float(self.kappa)
        if gamma < 0.0 or gamma != gamma:
            raise ValueError("gamma must be non-negative and not NaN")
        if kappa < 0.0 or kappa != kappa:
            raise ValueError("kappa must be non-negative and not NaN")
        lower = float(self.lower)
        upper = float(self.upper)
        if not isfinite(lower) or not isfinite(upper) or lower >= upper:
            raise ValueError("observation bounds must be finite with lower < upper")
        obs: tuple[float, ...] | None
        if self.observations is None:
            obs = None
        else:
            obs = tuple(float(value) for value in self.observations)
            if not obs:
                raise ValueError("observations, when provided, must be non-empty")
            if any(not isfinite(value) or value < lower or value > upper for value in obs):
                raise ValueError("pilot observations must lie within the declared bounds")
        object.__setattr__(self, "final_cost", final_cost)
        object.__setattr__(self, "correction_cost", correction_cost)
        object.__setattr__(self, "gamma", gamma)
        object.__setattr__(self, "kappa", kappa)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "observations", obs)


@dataclass(frozen=True)
class CandidateBound:
    """Auditable risk and standalone-cost evaluation for one candidate."""

    name: str
    gamma: float
    kappa: float
    mean_difference: float | None
    radius: float | None
    upper_difference: float | None
    bound_method: str
    total_cost: float
    net_saving_vs_standalone: float
    accepted: bool
    reason: str


@dataclass(frozen=True)
class Selection:
    """Auditable output of :func:`select_safe_plan`."""

    selected: str
    used_fallback: bool
    standalone_baseline_cost: float
    selected_end_to_end_cost: float
    pilot_overhead: float
    confidence_delta: float
    risk_tolerance: float
    max_gamma: float
    minimum_saving: float
    bounds: tuple[CandidateBound, ...]


def hoeffding_radius(lower: float, upper: float, sample_count: int, delta: float, *, one_sided: bool = True) -> float:
    """Return Hoeffding radius for a bounded mean in ``[lower, upper]``."""
    lower = float(lower)
    upper = float(upper)
    if not isfinite(lower) or not isfinite(upper) or lower >= upper:
        raise ValueError("observation bounds must be finite with lower < upper")
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie strictly between zero and one")
    factor = 1.0 if one_sided else 2.0
    return (upper - lower) * sqrt(log(factor / delta) / (2.0 * sample_count))


def empirical_bernstein_radius(
    observations: Sequence[float],
    lower: float,
    upper: float,
    delta: float,
) -> tuple[float, str]:
    """Simultaneously valid one-sided upper radius via Maurer–Pontil (2009)
    Empirical Bernstein combined with one-sided Hoeffding via union bound.

    For ``m == 1``, falls back to exact one-sided Hoeffding at level ``delta``.
    For ``m >= 2``, allocates ``delta/2`` to Empirical Bernstein and ``delta/2``
    to one-sided Hoeffding and returns the tighter valid radius.
    """
    m = len(observations)
    if m < 1:
        raise ValueError("sample_count must be positive")
    lower = float(lower)
    upper = float(upper)
    if not isfinite(lower) or not isfinite(upper) or lower >= upper:
        raise ValueError("observation bounds must be finite with lower < upper")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie strictly between zero and one")
    if m == 1:
        return hoeffding_radius(lower, upper, 1, delta, one_sided=True), "hoeffding_one_sided"
    mean = sum(observations) / m
    sample_var = sum((x - mean) ** 2 for x in observations) / (m - 1)
    span = upper - lower
    # Union-bound allocation delta/2 each so min(r_eb, r_h) holds with probability >= 1 - delta.
    r_h = hoeffding_radius(lower, upper, m, delta / 2.0, one_sided=True)
    log_term = log(4.0 / delta)  # 2 / (delta / 2)
    r_eb = sqrt((2.0 * sample_var * log_term) / m) + (7.0 * span * log_term) / (3.0 * (m - 1))
    if r_eb <= r_h:
        return r_eb, "empirical_bernstein"
    return r_h, "hoeffding_union"


def select_safe_plan(
    candidates: Iterable[CandidatePlan],
    *,
    baseline_cost: float,
    pilot_overhead: float = 0.0,
    confidence_delta: float = 0.05,
    risk_tolerance: float = 0.0,
    max_gamma: float = 1.05,
    max_kappa: float | None = None,
    minimum_saving: float = 0.0,
    baseline_name: str = "cube",
) -> Selection:
    """Select the cheapest candidate passing BOTH risk and standalone-cost contracts.

    Key fix over the initial draft:
    - ``pilot_overhead`` is charged ONLY to adaptive candidates (it represents
      non-reusable pilot cost not paid by standalone CUBE).  Standalone CUBE
      pays ``baseline_cost`` (NOT ``pilot_overhead + baseline_cost``), so a
      candidate passes the price contract iff:
          ``pilot_overhead + final_cost + correction_cost <= baseline_cost - minimum_saving``.
    - If a candidate has ``observations is None``, safety is checked via the
      zero-pilot distribution-free analytical bounds (``gamma <= max_gamma`` and
      optional ``kappa <= max_kappa``).
    - If ``observations`` are provided, BOTH the analytical gate (if ``max_gamma``
      is finite) AND the simultaneous Bonferroni + Empirical-Bernstein/Hoeffding
      upper risk bound ``mean + radius <= risk_tolerance`` must pass.
    - If no candidate passes, returns ``baseline_name`` (CUBE fallback).
    """
    plans = tuple(candidates)
    if not plans:
        raise ValueError("at least one candidate is required")
    if len({plan.name for plan in plans}) != len(plans):
        raise ValueError("candidate names must be unique")
    if any(plan.name == baseline_name for plan in plans):
        raise ValueError("baseline must not be listed as a candidate")
    baseline_cost = _finite_nonnegative(baseline_cost, "baseline_cost")
    pilot_overhead = _finite_nonnegative(pilot_overhead, "pilot_overhead")
    minimum_saving = _finite_nonnegative(minimum_saving, "minimum_saving")
    risk_tolerance = float(risk_tolerance)
    if not isfinite(risk_tolerance):
        raise ValueError("risk_tolerance must be finite")
    if not 0.0 < confidence_delta < 1.0:
        raise ValueError("confidence_delta must lie strictly between zero and one")
    if max_gamma < 1.0 and isfinite(max_gamma):
        raise ValueError("max_gamma must be >= 1.0 or inf")
    if max_kappa is not None and max_kappa < 1.0 and isfinite(max_kappa):
        raise ValueError("max_kappa must be >= 1.0 or inf")

    empirical_count = sum(1 for p in plans if p.observations is not None)
    per_candidate_delta = confidence_delta / empirical_count if empirical_count > 0 else confidence_delta

    bounds: list[CandidateBound] = []
    accepted: list[CandidateBound] = []

    for plan in plans:
        total_cost = pilot_overhead + plan.final_cost + plan.correction_cost
        net_saving = baseline_cost - total_cost

        mean: float | None = None
        radius: float | None = None
        upper_diff: float | None = None
        method = "analytical_only"

        if plan.observations is not None:
            mean = sum(plan.observations) / len(plan.observations)
            radius, method = empirical_bernstein_radius(
                plan.observations, plan.lower, plan.upper, per_candidate_delta
            )
            upper_diff = mean + radius

        # Evaluate contracts in auditable priority order
        if plan.gamma > max_gamma:
            is_accepted = False
            reason = f"analytical gamma {plan.gamma:.4f} exceeds max_gamma {max_gamma:.4f}"
        elif max_kappa is not None and plan.kappa > max_kappa:
            is_accepted = False
            reason = f"vacant-channel kappa {plan.kappa:.4f} exceeds max_kappa {max_kappa:.4f}"
        elif upper_diff is not None and upper_diff > risk_tolerance:
            is_accepted = False
            reason = "empirical risk upper bound exceeds contract"
        elif net_saving < minimum_saving:
            is_accepted = False
            reason = "end-to-end cost (incl. pilot overhead) does not meet standalone saving contract"
        else:
            is_accepted = True
            reason = "risk and standalone price contracts pass"

        bound = CandidateBound(
            name=plan.name,
            gamma=plan.gamma,
            kappa=plan.kappa,
            mean_difference=mean,
            radius=radius,
            upper_difference=upper_diff,
            bound_method=method,
            total_cost=total_cost,
            net_saving_vs_standalone=net_saving,
            accepted=is_accepted,
            reason=reason,
        )
        bounds.append(bound)
        if is_accepted:
            accepted.append(bound)

    if accepted:
        chosen = min(accepted, key=lambda b: (b.total_cost, b.gamma, b.name))
        return Selection(
            selected=chosen.name,
            used_fallback=False,
            standalone_baseline_cost=baseline_cost,
            selected_end_to_end_cost=chosen.total_cost,
            pilot_overhead=pilot_overhead,
            confidence_delta=confidence_delta,
            risk_tolerance=risk_tolerance,
            max_gamma=max_gamma,
            minimum_saving=minimum_saving,
            bounds=tuple(bounds),
        )
    return Selection(
        selected=baseline_name,
        used_fallback=True,
        standalone_baseline_cost=baseline_cost,
        selected_end_to_end_cost=pilot_overhead + baseline_cost,
        pilot_overhead=pilot_overhead,
        confidence_delta=confidence_delta,
        risk_tolerance=risk_tolerance,
        max_gamma=max_gamma,
        minimum_saving=minimum_saving,
        bounds=tuple(bounds),
    )
