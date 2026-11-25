"""Unified Stage 3.1 execution adapter bridging planning, safe selection,
decoupled mask-trie scheduling, vacant-channel certification, and scoring.

Supports BOTH live SGLang execution (injecting ``serving.score_batch``) and
deterministic offline replay on archived Stage 2.2 real-model score tables so
the exact same selection, scheduling, and auditing code runs offline and on B200.
"""

from __future__ import annotations

import random
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from math import isfinite

from safe_selector import (
    CandidatePlan,
    CosetPartition,
    GapCertificate,
    Selection,
    contender_separating_partition,
    exact_coset_certificates,
    gap_snr_certificate,
    general_tag_history_counts,
    select_safe_plan,
    vacant_channel_variance,
)
from scheduler import RequestRow, ScheduleCostProfile, schedule_batches, simulate_schedule_cost


@dataclass(frozen=True)
class OptionSelection:
    """Selection plus the original option record chosen for delivery."""

    decision: Selection
    selected_option: Mapping[str, object]
    options_by_name: Mapping[str, Mapping[str, object]]


def option_name(option: Mapping[str, object], *, baseline_name: str = "cube") -> str:
    """Derive a stable ledger name from an existing planning option."""
    explicit = option.get("name")
    if explicit is not None:
        name = str(explicit)
        if not name:
            raise ValueError("option name must be non-empty")
        return name
    groups = option.get("groups")
    if groups is not None and int(groups) > 2:
        return f"ordered_cosets_{int(groups)}"
    cut = option.get("cut")
    if cut is None:
        return baseline_name
    return f"ordered_cut_{cut}"


def select_options(
    options: Iterable[Mapping[str, object]],
    pilot_differences: Mapping[str, Sequence[float]] | None = None,
    *,
    pilot_overhead: float = 0.0,
    baseline_name: str = "cube",
    cost_key: str = "work",
    correction_costs: Mapping[str, float] | None = None,
    confidence_delta: float = 0.05,
    risk_tolerance: float = 0.0,
    max_gamma: float = 1.05,
    max_kappa: float | None = None,
    minimum_saving: float = 0.0,
) -> OptionSelection:
    """Apply the fixed standalone-baseline safe contract to planning records.

    Works in BOTH modes:
    - ``pilot_differences is None``: zero-pilot analytical gate using ``gamma``
      (and optional ``kappa``) plus standalone cost savings.
    - ``pilot_differences`` provided: simultaneous Empirical-Bernstein / Hoeffding
      upper risk bound plus analytical gate and standalone net saving (charging
      ``pilot_overhead`` ONLY to non-baseline adaptive candidates).
    """
    rows = tuple(options)
    if not rows:
        raise ValueError("at least one planning option is required")
    named = tuple((option_name(row, baseline_name=baseline_name), row) for row in rows)
    names = [name for name, _ in named]
    if len(set(names)) != len(names):
        raise ValueError("planning option names must be unique")
    by_name = dict(named)
    if baseline_name not in by_name:
        raise ValueError("planning options must include the baseline")
    correction_costs = {} if correction_costs is None else dict(correction_costs)

    def cost(row: Mapping[str, object]) -> float:
        value = float(row[cost_key])
        if not isfinite(value) or value < 0.0:
            raise ValueError(f"{cost_key} must be finite and non-negative")
        return value

    plans: list[CandidatePlan] = []
    for name, row in named:
        if name == baseline_name:
            continue
        obs = None
        if pilot_differences is not None:
            if name not in pilot_differences:
                raise ValueError(f"missing pilot differences for {name}")
            obs = tuple(pilot_differences[name])
        plans.append(
            CandidatePlan(
                name=name,
                final_cost=cost(row),
                observations=obs,
                correction_cost=correction_costs.get(name, 0.0),
                gamma=float(row.get("gamma", 1.0)),
                kappa=float(row.get("kappa", 1.0)),
            )
        )
    if not plans:
        baseline_val = cost(by_name[baseline_name])
        decision = Selection(
            selected=baseline_name,
            used_fallback=True,
            standalone_baseline_cost=baseline_val,
            selected_end_to_end_cost=pilot_overhead + baseline_val,
            pilot_overhead=pilot_overhead,
            confidence_delta=confidence_delta,
            risk_tolerance=risk_tolerance,
            max_gamma=max_gamma,
            minimum_saving=minimum_saving,
            bounds=(),
        )
        return OptionSelection(decision, by_name[baseline_name], by_name)

    decision = select_safe_plan(
        plans,
        baseline_cost=cost(by_name[baseline_name]),
        pilot_overhead=pilot_overhead,
        confidence_delta=confidence_delta,
        risk_tolerance=risk_tolerance,
        max_gamma=max_gamma,
        max_kappa=max_kappa,
        minimum_saving=minimum_saving,
        baseline_name=baseline_name,
    )
    return OptionSelection(decision, by_name[decision.selected], by_name)


def draw_design_rows(
    n: int,
    m: int,
    rng: random.Random,
    *,
    tags: Sequence[int] | None = None,
    groups: int = 1,
) -> tuple[list[int], list[int], int]:
    """Generate ``2*m`` complementary coalition masks for CUBE or ANY coset tag assignment.

    Returns ``(rows, labels, shift)``.  When ``tags is None``, draws standard CUBE
    labels uniformly without replacement from ``1..m-1``.  When ``tags`` is given
    (contiguous OR contender-separated), draws labels uniformly without replacement
    inside each coset ``[j*L, (j+1)*L)`` of size ``L = m // groups``.
    """
    if not (1 <= n < m) or (m & (m - 1)):
        raise ValueError("m must be a power of two strictly greater than n")
    if tags is None:
        labels = rng.sample(range(1, m), n)
    else:
        if len(tags) != n:
            raise ValueError("tags length must equal n")
        if groups < 1 or (groups & (groups - 1)) or m % groups != 0:
            raise ValueError("groups must be a power of two dividing m")
        width = m // groups
        by_group: dict[int, list[int]] = defaultdict(list)
        for idx, tag in enumerate(tags):
            if not (0 <= int(tag) < groups):
                raise ValueError(f"tag {tag} out of range [0, {groups})")
            by_group[int(tag)].append(idx)
        labels = [0] * n
        for g_idx, indices in by_group.items():
            if len(indices) > width:
                raise ValueError(f"coset {g_idx} has {len(indices)} segments > capacity {width}")
            drawn = rng.sample(range(g_idx * width, (g_idx + 1) * width), len(indices))
            for idx, lbl in zip(indices, drawn, strict=True):
                labels[idx] = lbl

    shift = rng.getrandbits(n)
    all_bits = (1 << n) - 1
    rows: list[int] = []
    for t in range(m):
        mask = shift
        for i, lbl in enumerate(labels):
            if (lbl & t).bit_count() & 1:
                mask ^= 1 << i
        rows.append(mask)
        rows.append(all_bits ^ mask)
    return rows, labels, shift


def decode_balanced_block(
    rows: Sequence[int],
    scores_by_mask: Mapping[int, Sequence[float]],
    n: int,
    nq: int,
) -> tuple[list[list[float]], list[list[float]]]:
    """Decode Banzhaf main effects ``B[q][i]`` and odd complementary vectors ``O[q][t]``."""
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
                sgn = 1.0 if (m_pos >> i) & 1 else -1.0
                effects[q][i] += 2.0 * sgn * o_val / m
    return effects, odd_by_q


def estimate_design_work(
    segment_lengths: Sequence[int],
    header_length: int,
    tail_tokens_total: int,
    n_questions: int,
    m: int,
    *,
    tags: Sequence[int] | None = None,
    groups: int = 1,
    alpha_per_token: float = 1.0,
    beta_per_pair: float = 0.0001,
) -> dict[str, float]:
    """Input-only analytical physical work estimate supporting arbitrary coset tags."""
    n = len(segment_lengths)
    if tags is None:
        counts = general_tag_history_counts([0] * n, m, 1)
    else:
        counts = general_tag_history_counts(tags, m, groups)
    tokens = 0.0
    pairs = 0.0
    prefix = 0.0
    for rows_at_i, length in zip(counts, segment_lengths, strict=True):
        tokens += rows_at_i * length
        pairs += rows_at_i * (length * (header_length + prefix / 2.0) + length * (length + 1) / 2.0)
        prefix += length
    distinct_masks = 2.0 * counts[-1] if counts else 0.0
    tail_work_tokens = distinct_masks * tail_tokens_total
    tail_pairs = distinct_masks * tail_tokens_total * (header_length + prefix / 2.0)
    total_tokens = tokens + tail_work_tokens
    total_pairs = pairs + tail_pairs
    work = alpha_per_token * total_tokens + beta_per_pair * total_pairs
    return {
        "doc_tokens": tokens,
        "tail_tokens": tail_work_tokens,
        "tokens": total_tokens,
        "pairs": total_pairs,
        "distinct_masks": distinct_masks,
        "questions": float(n_questions),
        "work": work,
    }


def build_candidate_records(
    segment_lengths: Sequence[int],
    header_length: int,
    tail_tokens_total: int,
    n_questions: int,
    m: int,
    *,
    contenders: tuple[int, int] | None = None,
    max_gamma_for_sep: float = 1.05,
    alpha_per_token: float = 1.0,
    beta_per_pair: float = 0.0001,
) -> list[dict[str, object]]:
    """Build the unified candidate menu (CUBE, 2-coset cuts, 4-coset, optional contender-separated)."""
    n = len(segment_lengths)
    records: list[dict[str, object]] = []

    # 1. Standalone CUBE baseline
    g_cube, k_cube = exact_coset_certificates([n], m)
    c_cost = estimate_design_work(
        segment_lengths, header_length, tail_tokens_total, n_questions, m,
        tags=None, groups=1, alpha_per_token=alpha_per_token, beta_per_pair=beta_per_pair,
    )
    records.append({"name": "cube", "cut": None, "groups": 1, "tags": None, "gamma": g_cube, "kappa": k_cube, **c_cost})

    # 2. Contiguous 2-coset candidates (preserving >= 1 empty slot per coset)
    if m % 2 == 0 and n >= 2:
        half = m // 2
        low = max(1, n - half + 1)
        high = min(n - 1, half - 1)
        cuts: set[int] = set()
        if low <= high:
            cuts.update(max(low, min(high, round(p * n))) for p in (0.40, 0.45, 0.50, 0.55, 0.60))
            cuts.update((low, high))
        for cut in sorted(cuts):
            sizes = (cut, n - cut)
            gamma, kappa = exact_coset_certificates(sizes, m)
            tags = tuple(0 if i < cut else 1 for i in range(n))
            cost = estimate_design_work(
                segment_lengths, header_length, tail_tokens_total, n_questions, m,
                tags=tags, groups=2, alpha_per_token=alpha_per_token, beta_per_pair=beta_per_pair,
            )
            records.append(
                {
                    "name": f"ordered_cut_{cut}",
                    "cut": cut,
                    "groups": 2,
                    "tags": tags,
                    "gamma": gamma,
                    "kappa": kappa,
                    **cost,
                }
            )

    # 3. Balanced contiguous 4-coset candidate (preserving >= 1 empty slot per coset)
    if m % 4 == 0 and n >= 4:
        quarter = m // 4
        sizes4 = tuple((j + 1) * n // 4 - j * n // 4 for j in range(4))
        if all(1 <= sz < quarter for sz in sizes4):
            gamma4, kappa4 = exact_coset_certificates(sizes4, m)
            tags4 = tuple(j for j, sz in enumerate(sizes4) for _ in range(sz))
            cost4 = estimate_design_work(
                segment_lengths, header_length, tail_tokens_total, n_questions, m,
                tags=tags4, groups=4, alpha_per_token=alpha_per_token, beta_per_pair=beta_per_pair,
            )
            records.append(
                {
                    "name": "ordered_cosets_4",
                    "cut": None,
                    "groups": 4,
                    "tags": tags4,
                    "gamma": gamma4,
                    "kappa": kappa4,
                    **cost4,
                }
            )

    # 4. Contender-separating 2-coset candidate (for Block >= 2 using independent Block-1 contenders)
    if contenders is not None:
        sep: CosetPartition | None = contender_separating_partition(
            n, m, contenders, g=2, max_gamma=max_gamma_for_sep
        )
        if sep is not None and not sep.is_contiguous:
            cost_sep = estimate_design_work(
                segment_lengths, header_length, tail_tokens_total, n_questions, m,
                tags=sep.tags, groups=2, alpha_per_token=alpha_per_token, beta_per_pair=beta_per_pair,
            )
            records.append(
                {
                    "name": f"separated_2coset_{contenders[0]}_{contenders[1]}",
                    "cut": None,
                    "groups": 2,
                    "tags": sep.tags,
                    "gamma": sep.gamma,
                    "kappa": sep.kappa,
                    **cost_sep,
                }
            )

    return records


ScorerCallable = Callable[[Sequence[Sequence[int]], Sequence[Sequence[int]], Sequence[tuple[int, int]]], tuple[Sequence[float], int]]


@dataclass(frozen=True)
class AttributionRunLedger:
    """Complete auditable execution ledger for one document attribution call."""

    selected_plan: str
    used_fallback: bool
    layout: str
    n_sources: int
    n_questions: int
    m_pairs: int
    gamma: float
    kappa: float
    seed_namespace: str
    schedule_profile: ScheduleCostProfile
    effects: tuple[tuple[float, ...], ...]
    gap_certificates: tuple[GapCertificate, ...]
    triggered_questions: tuple[int, ...]
    confirmation_requests: int
    final_top1: tuple[int, ...]
    elapsed_seconds: float
    selection_summary: dict[str, object]


def execute_safe_attribution(
    *,
    prompt_builder: Callable[[int, Sequence[int]], Sequence[Sequence[int]]],
    answer_tokens: Sequence[Sequence[int]],
    scorer: ScorerCallable,
    segment_lengths: Sequence[int],
    header_length: int,
    m: int,
    seed: int,
    layout: str = "mask_trie_fanout",
    width: int = 32,
    max_live_masks: int | None = None,
    max_gamma: float = 1.05,
    max_kappa: float | None = None,
    minimum_saving_fraction: float = 0.02,
    z_threshold: float = 2.0,
    min_vacant_dof: int = 16,
    run_confirmation: bool = True,
    confirmation_pairs: int = 8,
    contenders_from_prior_block: tuple[int, int] | None = None,
    pilot_differences: Mapping[str, Sequence[float]] | None = None,
    pilot_overhead: float = 0.0,
    confidence_delta: float = 0.05,
    risk_tolerance: float = 0.0,
    seed_namespace: str = "stage3-delivery-v1",
) -> AttributionRunLedger:
    """Run the complete P-18 Stage 3.1 safe adaptive attribution pipeline.

    ``scorer(prompts, tails, mask_q_pairs)`` returns ``(scores, cached_tokens)``,
    allowing identical audited execution on live SGLang B200 or offline score tables.
    """
    t0 = time.perf_counter()
    n = len(segment_lengths)
    nq = len(answer_tokens)
    qs = list(range(nq))
    tail_total = sum(len(a) for a in answer_tokens)

    # 1. Build candidate menu and run safe contract selection
    records = build_candidate_records(
        segment_lengths,
        header_length,
        tail_total,
        nq,
        m,
        contenders=contenders_from_prior_block,
        max_gamma_for_sep=max_gamma,
    )
    baseline_work = float(next(r["work"] for r in records if r["name"] == "cube"))
    opt_sel = select_options(
        records,
        pilot_differences=pilot_differences,
        pilot_overhead=pilot_overhead,
        max_gamma=max_gamma,
        max_kappa=max_kappa,
        minimum_saving=baseline_work * minimum_saving_fraction,
        confidence_delta=confidence_delta,
        risk_tolerance=risk_tolerance,
    )
    chosen = opt_sel.selected_option
    chosen_tags = chosen["tags"]
    chosen_groups = int(chosen["groups"])
    chosen_gamma = float(chosen["gamma"])
    chosen_kappa = float(chosen["kappa"])

    # 2. Draw independent delivery masks from isolated seed namespace
    rng = random.Random(f"{seed_namespace}:{seed}:{opt_sel.decision.selected}")
    rows, _labels, _shift = draw_design_rows(n, m, rng, tags=chosen_tags, groups=chosen_groups)
    uniq_masks = list(dict.fromkeys(rows))

    # 3. Build request table and execute via unified fair scheduler
    by_mask: dict[int, list[RequestRow]] = defaultdict(list)
    for mask in uniq_masks:
        prompts = prompt_builder(mask, qs)
        for q, ptoks in zip(qs, prompts, strict=True):
            by_mask[mask].append((ptoks, mask, q))

    sched_profile = simulate_schedule_cost(by_mask, layout, n, width, max_live_masks=max_live_masks)
    score_cache: dict[tuple[int, int], float] = {}
    for batch in schedule_batches(by_mask, layout, n, width, max_live_masks=max_live_masks):
        prompts = [r[0] for r in batch]
        tails = [answer_tokens[r[2]] for r in batch]
        pairs = [(r[1], r[2]) for r in batch]
        vals, _ = scorer(prompts, tails, pairs)
        for (mask, q), val in zip(pairs, vals, strict=True):
            score_cache[(mask, q)] = float(val)

    scores_by_mask = {mask: [score_cache[(mask, q)] for q in qs] for mask in uniq_masks}
    effects, odd_by_q = decode_balanced_block(rows, scores_by_mask, n, nq)

    # 4. Zero-extra-query vacant-channel Parseval certification per question
    certs: list[GapCertificate] = []
    triggered: list[int] = []
    coset_of = None if chosen_tags is None else tuple(int(t) for t in chosen_tags)
    same_coset_factor = float(chosen_groups) if chosen_groups > 1 else 1.0
    for q in qs:
        v_vac = vacant_channel_variance(odd_by_q[q], effects[q], m)
        cert = gap_snr_certificate(
            effects[q],
            v_vac,
            vacant_dof=m - n,
            kappa=chosen_kappa,
            coset_of=coset_of,
            same_coset_factor=same_coset_factor,
            z_threshold=z_threshold,
            min_vacant_dof=min_vacant_dof,
        )
        certs.append(cert)
        if cert.needs_confirmation:
            triggered.append(q)

    # 5. Optional independent swap confirmation ONLY for triggered uncertain questions
    final_top1 = [c.top1 for c in certs]
    conf_reqs = 0
    if run_confirmation and triggered:
        conf_by_mask: dict[int, list[RequestRow]] = defaultdict(list)
        swap_pairs_by_q: dict[int, list[tuple[int, int]]] = {}
        for q in triggered:
            c = certs[q]
            i1, i2 = c.top1, c.top2
            free = ((1 << n) - 1) ^ ((1 << i1) | (1 << i2))
            swap_rng = random.Random(f"{seed_namespace}:confirm:{seed}:{q}:{min(i1, i2)}:{max(i1, i2)}")
            pairs_q: list[tuple[int, int]] = []
            for _ in range(confirmation_pairs):
                bg = swap_rng.getrandbits(n) & free
                for u in (bg, free ^ bg):
                    pairs_q.append((u | (1 << i1), u | (1 << i2)))
            swap_pairs_by_q[q] = pairs_q
            needed_masks = {msk for pair in pairs_q for msk in pair if (msk, q) not in score_cache}
            for msk in sorted(needed_masks):
                ptok = prompt_builder(msk, [q])[0]
                conf_by_mask[msk].append((ptok, msk, q))
        for batch in schedule_batches(conf_by_mask, layout, n, width, max_live_masks=max_live_masks):
            prompts = [r[0] for r in batch]
            tails = [answer_tokens[r[2]] for r in batch]
            pairs = [(r[1], r[2]) for r in batch]
            conf_reqs += len(batch)
            vals, _ = scorer(prompts, tails, pairs)
            for (mask, q), val in zip(pairs, vals, strict=True):
                score_cache[(mask, q)] = float(val)
        for q in triggered:
            pairs_q = swap_pairs_by_q[q]
            mean_diff = sum(score_cache[(a, q)] - score_cache[(b, q)] for a, b in pairs_q) / len(pairs_q)
            if mean_diff < 0.0:
                final_top1[q] = certs[q].top2

    elapsed = time.perf_counter() - t0
    return AttributionRunLedger(
        selected_plan=opt_sel.decision.selected,
        used_fallback=opt_sel.decision.used_fallback,
        layout=layout,
        n_sources=n,
        n_questions=nq,
        m_pairs=m,
        gamma=chosen_gamma,
        kappa=chosen_kappa,
        seed_namespace=seed_namespace,
        schedule_profile=sched_profile,
        effects=tuple(tuple(row) for row in effects),
        gap_certificates=tuple(certs),
        triggered_questions=tuple(triggered),
        confirmation_requests=conf_reqs,
        final_top1=tuple(final_top1),
        elapsed_seconds=elapsed,
        selection_summary=asdict(opt_sel.decision),
    )
