"""Comprehensive offline unit tests for P-18 Stage 3.1 core modules."""

import math
import random
import sys
import unittest
from pathlib import Path

COATTR_DIR = Path(__file__).resolve().parent.parent / 'coattr'
if str(COATTR_DIR) not in sys.path:
    sys.path.insert(0, str(COATTR_DIR))

import designs  # noqa: E402
import group_bound  # noqa: E402
import planning  # noqa: E402

from integration import (  # noqa: E402
    build_candidate_records,
    decode_balanced_block,
    draw_design_rows,
    execute_safe_attribution,
    option_name,
    select_options,
)
from safe_selector import (  # noqa: E402
    CandidatePlan,
    contender_separating_partition,
    empirical_bernstein_radius,
    exact_coset_certificates,
    exact_kappa_cube,
    gap_snr_certificate,
    general_tag_history_counts,
    hoeffding_radius,
    select_safe_plan,
    vacant_channel_variance,
)
from scheduler import mask_trie_fanout_batches, schedule_batches, simulate_schedule_cost  # noqa: E402


class SafeSelectorContractTests(unittest.TestCase):
    def test_radius_rejects_empty_sample(self):
        with self.assertRaises(ValueError):
            hoeffding_radius(-1.0, 1.0, 0, 0.05)

    def test_pilot_overhead_is_NOT_cancelled_from_standalone_baseline(self):
        """Regression test for the Stage 3.1 draft bug where pilot_cost was added
        to BOTH baseline and candidate, cancelling itself out of the price check."""
        # Candidate saves 2.0 in delivery (8.0 vs 10.0), but pilot_overhead is 3.0 ->
        # end-to-end is 11.0 > standalone CUBE 10.0, so MUST reject and fall back!
        decision = select_safe_plan(
            [CandidatePlan("two_cosets", final_cost=8.0, observations=tuple([-0.5] * 64))],
            baseline_cost=10.0,
            pilot_overhead=3.0,
            risk_tolerance=0.0,
            minimum_saving=0.0,
        )
        self.assertTrue(decision.used_fallback)
        self.assertEqual(decision.selected, "cube")
        self.assertEqual(decision.standalone_baseline_cost, 10.0)
        self.assertLess(decision.bounds[0].net_saving_vs_standalone, 0.0)
        self.assertIn("standalone saving contract", decision.bounds[0].reason)

    def test_empirical_bernstein_strictly_tighter_than_hoeffding_under_low_variance(self):
        """Real LLM attribution risk differences have tiny variance (~1e-5), not 1.0."""
        # 32 observations tightly clustered around 0.0005 in [-1, 1]
        obs = tuple(0.0005 + (0.0002 if i % 2 == 0 else -0.0002) for i in range(64))
        r_eb, method = empirical_bernstein_radius(obs, -1.0, 1.0, 0.05)
        r_h = hoeffding_radius(-1.0, 1.0, len(obs), 0.05, one_sided=True)
        # Note: for m=64, 7*R*log(80)/(3*63) ~ 0.325 vs Hoeffding ~ 0.306; for m=256 EB dominates strongly!
        obs256 = obs * 4
        r_eb256, method256 = empirical_bernstein_radius(obs256, -1.0, 1.0, 0.05)
        r_h256 = hoeffding_radius(-1.0, 1.0, len(obs256), 0.05, one_sided=True)
        self.assertEqual(method256, "empirical_bernstein")
        self.assertLess(r_eb256, 0.6 * r_h256)
        self.assertGreater(r_eb, 0.0)
        self.assertGreater(r_h, 0.0)
        self.assertIn(method, ("empirical_bernstein", "hoeffding_union"))

    def test_zero_pilot_analytical_gate_accepts_certified_plan_and_rejects_high_gamma(self):
        decision = select_safe_plan(
            [
                CandidatePlan("two_cosets_safe", final_cost=8.0, gamma=1.02, kappa=1.04),
                CandidatePlan("four_cosets_risky", final_cost=7.0, gamma=1.23, kappa=float("inf")),
            ],
            baseline_cost=10.0,
            pilot_overhead=0.0,
            max_gamma=1.05,
            max_kappa=1.50,
            minimum_saving=0.5,
        )
        self.assertFalse(decision.used_fallback)
        self.assertEqual(decision.selected, "two_cosets_safe")
        self.assertEqual(decision.selected_end_to_end_cost, 8.0)
        self.assertFalse(decision.bounds[1].accepted)
        self.assertIn("gamma", decision.bounds[1].reason)

    def test_uncertain_empirical_candidate_falls_back(self):
        decision = select_safe_plan(
            [CandidatePlan("two_cosets", final_cost=4.0, observations=(-0.2,))],
            baseline_cost=10.0,
            pilot_overhead=1.0,
        )
        self.assertTrue(decision.used_fallback)
        self.assertEqual(decision.selected, "cube")
        self.assertFalse(decision.bounds[0].accepted)

    def test_minimum_saving_enforced_with_correction_cost(self):
        decision = select_safe_plan(
            [CandidatePlan("two_cosets", final_cost=8.0, correction_cost=1.5, gamma=1.01)],
            baseline_cost=10.0,
            pilot_overhead=0.0,
            minimum_saving=1.0,
        )
        self.assertTrue(decision.used_fallback)
        self.assertEqual(decision.selected, "cube")

    def test_invalid_observation_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            CandidatePlan("bad", 1.0, (1.1,), lower=-1.0, upper=1.0)

    def test_select_options_preserves_record_and_requires_baseline(self):
        selected = select_options(
            [
                {"cut": None, "work": 10.0, "gamma": 1.0},
                {"cut": 2, "work": 7.5, "gamma": 1.02},
            ],
            pilot_differences=None,
            max_gamma=1.05,
            minimum_saving=1.0,
        )
        self.assertEqual(selected.decision.selected, "ordered_cut_2")
        self.assertIs(selected.selected_option, selected.options_by_name["ordered_cut_2"])
        self.assertEqual(option_name({"name": "four_cosets", "cut": 2}), "four_cosets")
        with self.assertRaises(ValueError):
            select_options([{"cut": 2, "work": 4.0}], pilot_differences=None)
        # Baseline-only menu (e.g., n = M - 1 saturated dimension) cleanly falls back to CUBE
        fb = select_options([{"cut": None, "work": 10.0, "gamma": 1.0}], pilot_differences=None)
        self.assertTrue(fb.decision.used_fallback)
        self.assertEqual(fb.decision.selected, "cube")


class MathematicalCertificateTests(unittest.TestCase):
    def test_exact_coset_certificates_match_stage2_dp(self):
        """Cross-check pure-Python DP against Stage 2.2 NumPy group_bound.factor_dp."""
        for n, m, g in [(37, 64, 2), (37, 64, 4), (52, 64, 2), (71, 128, 4), (90, 128, 4)]:
            sizes = [(j + 1) * n // g - j * n // g for j in range(g)]
            gamma, kappa = exact_coset_certificates(sizes, m)
            ref_gamma = group_bound.factor_dp(n, m, g)
            self.assertAlmostEqual(gamma, ref_gamma, places=10)
            self.assertGreaterEqual(kappa, 1.0)
        self.assertAlmostEqual(exact_kappa_cube(71, 128), 1.001761, places=5)
        self.assertAlmostEqual(exact_kappa_cube(90, 128), 1.001389, places=5)

    def test_general_tag_history_matches_planning_and_brute_force(self):
        """Verify GF(2) subspace-rank history recursion against planning.py and Monte Carlo."""
        for n, m, g in [(10, 16, 2), (12, 16, 4), (25, 32, 4), (37, 64, 2)]:
            tags = [j for j in range(g) for _ in range((j + 1) * n // g - j * n // g)]
            got = general_tag_history_counts(tags, m, g)
            ref = planning.group_history_counts(n, m, g)
            for a, b in zip(got, ref, strict=True):
                self.assertAlmostEqual(a, b, places=10)

        # Non-contiguous contender-separated tag sequence vs exact small brute-force enumeration
        # n=5, m=8, g=2 (L=4): tags = [0, 1, 0, 1, 0] -> coset 0 has 3 segs, coset 1 has 2 segs
        tags = (0, 1, 0, 1, 0)
        got = general_tag_history_counts(tags, 8, 2)
        # Exact enumeration over all 4*3*2 * 4*3 = 288 label assignments!
        import itertools

        totals = [0] * len(tags)
        count = 0
        for l0 in itertools.permutations(range(4), 3):
            for l1 in itertools.permutations(range(4, 8), 2):
                labels = [l0[0], l1[0], l0[1], l1[1], l0[2]]
                # Over the 2*M complementary rows (z and -z), count distinct prefixes before kept segment i (bit_i == 1)
                rows_2m = []
                for t in range(8):
                    bits = tuple((labels[k] & t).bit_count() & 1 for k in range(len(tags)))
                    rows_2m.append(bits)
                    rows_2m.append(tuple(1 - b for b in bits))
                for i in range(len(tags)):
                    seen = {row[:i] for row in rows_2m if row[i] == 1}
                    totals[i] += len(seen)
                count += 1
        exact = [tot / count for tot in totals]
        for a, b in zip(got, exact, strict=True):
            self.assertAlmostEqual(a, b, places=12)

    def test_vacant_channel_parseval_identity_and_unbiased_contender_separation(self):
        """Verify Parseval residual on a pure 3rd+5th-order synthetic function and
        unbiasedness of contender-separating coset designs."""
        n, m = 9, 16
        # Synthetic bounded response f(z) = 0.5 + 0.15*z0 + 0.12*z1 + 0.08*z0*z1*z2 + 0.05*z2*z3*z4*z5*z6
        true_B = [0.30, 0.24] + [0.0] * (n - 2)

        def eval_f(mask: int) -> float:
            z = [1.0 if (mask >> i) & 1 else -1.0 for i in range(n)]
            return 0.5 + 0.15 * z[0] + 0.12 * z[1] + 0.08 * z[0] * z[1] * z[2] + 0.05 * z[2] * z[3] * z[4] * z[5] * z[6]

        part = contender_separating_partition(n, m, (0, 1), g=2, max_gamma=2.0)
        self.assertIsNotNone(part)
        assert part is not None
        self.assertNotEqual(part.tags[0], part.tags[1])

        rng = random.Random(2027)
        acc = [0.0] * n
        mse_sum = 0.0
        vac_sum = 0.0
        trials = 400
        for _ in range(trials):
            rows, _, _ = draw_design_rows(n, m, rng, tags=part.tags, groups=part.groups)
            scores = {msk: [eval_f(msk)] for msk in set(rows)}
            effs, odd = decode_balanced_block(rows, scores, n, 1)
            for i in range(n):
                acc[i] += effs[0][i] / trials
            mse_sum += sum((effs[0][i] - true_B[i]) ** 2 for i in range(n)) / n
            vac_sum += vacant_channel_variance(odd[0], effs[0], m)

        # Unbiasedness check (within 4 Monte Carlo SEs)
        for i in range(n):
            self.assertAlmostEqual(acc[i], true_B[i], delta=0.015)
        # Vacant-channel variance is positive and tracks true MSE closely
        self.assertGreater(vac_sum / trials, 0.0)
        self.assertGreater(vac_sum / mse_sum, 0.5)
        self.assertLess(vac_sum / mse_sum, 2.5)

        cert = gap_snr_certificate(
            [0.30, 0.24, 0.0, 0.0],
            vacant_var=1e-5,
            vacant_dof=38,
            kappa=1.05,
            coset_of=(0, 1, 0, 1),
            z_threshold=2.0,
            min_vacant_dof=16,
        )
        self.assertEqual((cert.top1, cert.top2), (0, 1))
        self.assertFalse(cert.same_coset)
        self.assertFalse(cert.needs_confirmation)
        self.assertGreater(cert.dof_lower_tail_inflation, 1.4)

        # Low vacant DOF (e.g., QuALITY-99914 with M-n=4) MUST force conservative confirmation
        cert_low_dof = gap_snr_certificate(
            [0.30, 0.24, 0.0, 0.0],
            vacant_var=1e-8,
            vacant_dof=4,
            kappa=1.02,
            z_threshold=2.0,
            min_vacant_dof=16,
        )
        self.assertTrue(cert_low_dof.needs_confirmation)
        self.assertIn("min_vacant_dof", cert_low_dof.gate_reason)

    def test_early_contender_swap_prefix_wipeout_is_rejected_by_price_gate(self):
        """Verify that swapping an early contender pair like (2, 5) on n=71 inflates
        prefix work above CUBE and is automatically rejected by select_options."""
        recs = build_candidate_records([40] * 71, 20, 48, 1, 128, contenders=(2, 5))
        cube_work = float(next(r["work"] for r in recs if r["name"] == "cube"))
        sep_rec = next(r for r in recs if r["name"] == "separated_2coset_2_5")
        # Early swap wipes out prefix savings (work >= cube_work)
        self.assertGreaterEqual(float(sep_rec["work"]), cube_work * 0.99)
        # Passing only cube + early separated swap MUST fall back to cube!
        sel = select_options(
            [next(r for r in recs if r["name"] == "cube"), sep_rec],
            pilot_differences=None,
            max_gamma=1.05,
            minimum_saving=0.02 * cube_work,
        )
        self.assertTrue(sel.decision.used_fallback)
        self.assertEqual(sel.decision.selected, "cube")

        # Conversely, a late pair (60, 65) swap preserves substantial prefix savings!
        recs_late = build_candidate_records([40] * 71, 20, 48, 1, 128, contenders=(60, 65))
        sep_late = next(r for r in recs_late if r["name"] == "separated_2coset_60_65")
        self.assertLess(float(sep_late["work"]), 0.85 * cube_work)
        sel_late = select_options(
            [next(r for r in recs_late if r["name"] == "cube"), sep_late],
            pilot_differences=None,
            max_gamma=1.05,
            minimum_saving=0.02 * cube_work,
        )
        self.assertFalse(sel_late.decision.used_fallback)
        self.assertEqual(sel_late.decision.selected, "separated_2coset_60_65")


class SchedulerAndIntegrationTests(unittest.TestCase):
    def test_mask_trie_fanout_zero_mixed_batches_and_exact_ideal_uncached(self):
        """Verify mask_trie_fanout achieves zero mixed batches AND matches trie's exact
        Horton-Strahler zero-duplication uncached token count."""
        rng = random.Random(99)
        n = 12
        # Build 12 distinct segment token chunks of length 60 each + 8 distinct 12-token question tails
        parts = [[1000 * (i + 1) + j for j in range(60)] for i in range(n)]
        head = [1, 2, 3, 4]
        tails = [[90000 + q * 100 + j for j in range(12)] for q in range(8)]
        rows, _, _ = draw_design_rows(n, 16, rng, tags=[0] * 6 + [1] * 6, groups=2)
        by_mask = {}
        for msk in dict.fromkeys(rows):
            body = [tk for i in range(n) if (msk >> i) & 1 for tk in parts[i]]
            by_mask[msk] = [(head + body + tails[q], msk, q) for q in range(8)]

        p_trie = simulate_schedule_cost(by_mask, "trie", n, width=16)
        p_fan = simulate_schedule_cost(by_mask, "mask_trie_fanout", n, width=16)
        p_pref = simulate_schedule_cost(by_mask, "prefix", n, width=16)

        # Exact request preservation
        self.assertEqual(p_fan.requests, p_trie.requests)
        # Zero duplicate prefill: matches trie's ideal uncached tokens and <= prefix!
        self.assertEqual(p_fan.total_uncached_tokens, p_trie.total_uncached_tokens)
        self.assertLessEqual(p_fan.total_uncached_tokens, p_pref.total_uncached_tokens)
        # Zero mixed new-mask prefill + warmed-mask suffix batches!
        self.assertEqual(p_fan.mixed_prefill_suffix_batches, 0)
        self.assertGreater(p_trie.mixed_prefill_suffix_batches, 0)
        self.assertLess(p_fan.heavy_batches, p_trie.heavy_batches)
        # Every request appears exactly once
        emitted = [(r[1], r[2]) for b in mask_trie_fanout_batches(by_mask, 16) for r in b]
        self.assertEqual(len(emitted), len(set(emitted)))
        self.assertEqual(len(emitted), p_fan.requests)

    def test_end_to_end_execute_safe_attribution_auditable_ledger(self):
        """Exercise the full unified pipeline from candidate menu to confirmation ledger."""
        n, m = 9, 16
        parts = [[100 * (i + 1) + j for j in range(20)] for i in range(n)]
        head = [1, 2, 3]
        tails = [[5000 + q * 10 + j for j in range(6)] for q in range(2)]
        answers = [[7001, 7002], [7003, 7004]]

        def prompt_builder(mask, qs):
            body = [tk for i in range(n) if (mask >> i) & 1 for tk in parts[i]]
            return [head + body + tails[q] for q in qs]

        def scorer(_prompts, _tails, mask_q_pairs):
            vals = []
            for mask, q in mask_q_pairs:
                z0 = 1.0 if (mask >> 0) & 1 else -1.0
                z1 = 1.0 if (mask >> 1) & 1 else -1.0
                z2 = 1.0 if (mask >> 2) & 1 else -1.0
                if q == 0:
                    # Strong clear top-1 at segment 0
                    vals.append(0.5 + 0.25 * z0 + 0.02 * z1)
                else:
                    # Near-tie between segment 1 and 2 with 3rd-order noise
                    vals.append(0.5 + 0.08 * z1 + 0.075 * z2 + 0.04 * z0 * z1 * z2)
            return vals, 0

        ledger = execute_safe_attribution(
            prompt_builder=prompt_builder,
            answer_tokens=answers,
            scorer=scorer,
            segment_lengths=[20] * n,
            header_length=len(head),
            m=m,
            seed=7,
            layout="mask_trie_fanout",
            max_gamma=1.05,  # For n=9, M=16, 2-coset gamma=1.64 > 1.05 -> must fall back safely to CUBE!
            z_threshold=2.0,
            min_vacant_dof=6,  # Synthetic n=9, M=16 has 7 vacant DOF to exercise selective SNR gating
        )
        self.assertTrue(ledger.used_fallback)
        self.assertEqual(ledger.selected_plan, "cube")
        self.assertEqual(ledger.final_top1[0], 0)
        self.assertEqual(ledger.schedule_profile.mixed_batches, 0)
        self.assertIn(1, ledger.triggered_questions)
        self.assertGreater(ledger.confirmation_requests, 0)

        # Verify build_candidate_records creates contiguous + contender-separated options for larger n
        recs = build_candidate_records([40] * 71, 20, 48, 6, 128, contenders=(10, 12))
        names = [r["name"] for r in recs]
        self.assertIn("cube", names)
        self.assertIn("ordered_cosets_4", names)
        self.assertTrue(any(nm.startswith("separated_2coset_") or nm == "ordered_cut_11" for nm in names))
        # Verify schedule_batches rejects unknown layouts
        with self.assertRaises(ValueError):
            list(schedule_batches({}, "invalid_layout", 5))


if __name__ == "__main__":
    unittest.main()
