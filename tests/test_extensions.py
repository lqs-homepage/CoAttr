"""Comprehensive regression & brute-force verification suite for Stage 3.1
theoretical deepening modules:
- theory_extensions.py (T1, T2, T3)
- wave_merged_scheduler.py (T4)
- regression_msr_bench.py (exact tree Banzhaf & cross-fit hybrid)

All tests are additive and leave Stage-2-hashed test_safe_selector.py untouched.
"""

from __future__ import annotations

import itertools
import math
import random
import sys
import unittest
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np

COATTR_DIR = Path(__file__).resolve().parent.parent / 'coattr'
if str(COATTR_DIR) not in sys.path:
    sys.path.insert(0, str(COATTR_DIR))

import designs  # noqa: E402
from integration import draw_design_rows  # noqa: E402
from safe_selector import exact_coset_certificates, native_alias, signed_population, vacant_channel_variance  # noqa: E402
from theory_extensions import (  # noqa: E402
    combined_multiblock_variance,
    cyclic_balanced_coset_tags,
    exact_pairwise_gap_ratio_dp,
    t1_multiblock_both_blind_prob_cube,
    t1_sparse_blind_prob_cube,
    universal_strength3_oa_lower_bound,
    vacant_variance_bounds,
)
from wave_merged_scheduler import (  # noqa: E402
    GpuBatchCostParams,
    _extract_mask_trie_levels,
    _simulate_batches_profile,
    merged_mask_trie_batches,
    simulate_merged_mask_trie,
    width_aligned_mask_trie_batches,
)


class T1SparseAndMultiBlockTests(unittest.TestCase):
    def test_exact_single_and_multiblock_blind_formulas(self) -> None:
        # s=3: p_o(2)=0, p_o(4)=1/(M-3) => W_o(3) = (n-3)/(M-3)
        self.assertEqual(t1_sparse_blind_prob_cube(90, 128, 3), Fraction(87, 125))
        self.assertEqual(t1_sparse_blind_prob_cube(71, 128, 3), Fraction(68, 125))
        # R=1 equals single-block blind prob
        self.assertEqual(
            t1_multiblock_both_blind_prob_cube(90, 128, 1, 3),
            t1_sparse_blind_prob_cube(90, 128, 3),
        )
        # R=2 collapses by factor 1/(2*(M-3)) = 1/250
        self.assertEqual(
            t1_multiblock_both_blind_prob_cube(90, 128, 2, 3),
            Fraction(87, 125) * Fraction(1, 250),
        )
        # Strictly monotone decay with R
        p1 = t1_multiblock_both_blind_prob_cube(71, 128, 1, 5)
        p2 = t1_multiblock_both_blind_prob_cube(71, 128, 2, 5)
        p3 = t1_multiblock_both_blind_prob_cube(71, 128, 3, 5)
        self.assertGreater(p1, p2)
        self.assertGreater(p2, p3)

    def test_combined_multiblock_catches_zero_vacant_sparse_hit(self) -> None:
        # Block 0 hits occupied channel 0 (+0.5 error), block 1 hits occupied channel 1 (-0.5 error),
        # both blocks report 0.0 vacant variance — combined cross-block variance must catch it!
        b0 = [0.7, 0.2, 0.0, 0.0, 0.0]
        b1 = [0.2, -0.3, 0.0, 0.0, 0.0]
        comb = combined_multiblock_variance([b0, b1], [0.0, 0.0], kappa=1.2)
        self.assertEqual(comb["vacant_calibrated_avg_var"], 0.0)
        self.assertGreater(comb["cross_block_avg_var"], 0.01)
        self.assertEqual(comb["combined_avg_var"], comb["cross_block_avg_var"])
        self.assertGreater(comb["combined_max_seg_var"], 0.05)


class T2PairwiseDPAndCyclicRotationTests(unittest.TestCase):
    def _bruteforce_pairwise_ratio(self, sizes: tuple[int, ...], m: int, ci: int, cj: int) -> tuple[float, float]:
        """Exhaustively enumerate ALL coset-count tuples (u_0, ..., u_{g-1}) and verify DP."""
        g = len(sizes)
        h = m // g
        n = sum(sizes)
        rem = list(sizes)
        rem[ci] -= 1
        rem[cj] -= 1
        k = [float(signed_population(h, v)) for v in range(h + 1)]
        po = [float(native_alias(m, v)) if v <= m else 0.0 for v in range(n + 3)]

        best_even = 0.0
        best_odd = 0.0
        for counts in itertools.product(*(range(r + 1) for r in rem)):
            s = sum(counts)
            x = 0
            for c, u in enumerate(counts):
                if u & 1:
                    x ^= c
            if s >= 2 and s % 2 == 0:
                p0 = 0.0
                if x == 0:
                    prod = 1.0
                    for u in counts:
                        prod *= k[u]
                    p0 = (1.0 + (h - 1) * prod) / h
                pij = 0.0
                if x == (ci ^ cj):
                    prod = 1.0
                    for c, u in enumerate(counts):
                        sh = int(c == ci) + int(c == cj)
                        prod *= k[u + sh]
                    pij = (1.0 + (h - 1) * prod) / h
                pd = p0 + pij
                pc = po[s] + po[s + 2]
                best_even = max(best_even, pd / pc if pc > 1e-15 else (float("inf") if pd > 1e-15 else 0.0))
            elif s >= 3 and s % 2 == 1:
                pi = 0.0
                if x == ci:
                    prod = 1.0
                    for c, u in enumerate(counts):
                        prod *= k[u + int(c == ci)]
                    pi = (1.0 + (h - 1) * prod) / h
                pj = 0.0
                if x == cj:
                    prod = 1.0
                    for c, u in enumerate(counts):
                        prod *= k[u + int(c == cj)]
                    pj = (1.0 + (h - 1) * prod) / h
                pd = pi + pj
                pc = 2.0 * po[s + 1]
                best_odd = max(best_odd, pd / pc if pc > 1e-15 else (float("inf") if pd > 1e-15 else 0.0))
        return best_even, best_odd

    def test_pairwise_dp_matches_bruteforce_exactly(self) -> None:
        cases = [
            ((5, 5), 16, 0, 0),
            ((5, 5), 16, 0, 1),
            ((3, 3, 3, 3), 32, 0, 0),
            ((3, 3, 3, 3), 32, 0, 1),
            ((4, 3, 3, 2), 32, 1, 2),
            ((4, 3, 3, 2), 32, 2, 2),
        ]
        for sizes, m, ci, cj in cases:
            dp_res = exact_pairwise_gap_ratio_dp(sizes, m, ci, cj)
            bf_even, bf_odd = self._bruteforce_pairwise_ratio(sizes, m, ci, cj)
            self.assertAlmostEqual(dp_res["worst_even_branch"], bf_even, places=10)
            self.assertAlmostEqual(dp_res["worst_odd_branch"], bf_odd, places=10)
            self.assertAlmostEqual(dp_res["worst_overall"], max(bf_even, bf_odd), places=10)
            if ci != cj:
                self.assertEqual(dp_res["deg3_same_coset_local_ratio"], 0.0)
            elif sizes[ci] >= 4:
                self.assertGreater(dp_res["deg3_same_coset_local_ratio"], 1.5)
            else:
                self.assertEqual(dp_res["deg3_same_coset_local_ratio"], 0.0)

    def test_cyclic_balanced_rotation_preserves_sizes_and_separates_midrange_pairs(self) -> None:
        for n in (28, 71, 90):
            m = designs.budget(n) // 2
            rot2 = cyclic_balanced_coset_tags(n, 2, g=2)
            base_sizes = (n // 2, n - n // 2)
            base_gamma, base_kappa = exact_coset_certificates(base_sizes, m)
            for tg in rot2:
                self.assertEqual(len(tg), n)
                sizes = (tg.count(0), tg.count(1))
                self.assertEqual(sizes, base_sizes)
                g_val, k_val = exact_coset_certificates(sizes, m)
                self.assertEqual(g_val, base_gamma)
                self.assertEqual(k_val, base_kappa)
            # Every pair separated by d in [n//4 + 1, n//2] must be in DIFFERENT cosets in >=1 of the 2 blocks
            for i in range(n):
                for j in range(i + n // 4 + 1, min(n, i + n // 2)):
                    separated = any(tg[i] != tg[j] for tg in rot2)
                    self.assertTrue(separated, f"pair ({i}, {j}) for n={n} was never separated")


class T3BoundsAndOAFloorTests(unittest.TestCase):
    def test_deterministic_vacant_variance_upper_bound(self) -> None:
        # Adversarial bounded f in [0, 1] maximizing odd energy (pure higher-order +/-0.5)
        n, m = 10, 16
        rng = random.Random(77)
        bound = vacant_variance_bounds(m, n, score_range=1.0)
        vmax = float(bound["single_block_vac_upper"])
        self.assertAlmostEqual(vmax, 1.0 / (m - n), places=6)
        self.assertEqual(bound["tightening_factor_vs_minus1_plus1"], float(m - n))

        for _ in range(50):
            rows = designs.cube(n, 2 * m, rng, cut=None)
            # Random +/-0.5 odd values (corresponds to f(z) in {0, 1} with f(-z) = 1 - f(z))
            odd_vals = [0.5 if rng.random() < 0.5 else -0.5 for _ in range(m)]
            z_mat = np.array(
                [[1.0 if ((rows[2 * t] >> i) & 1) else -1.0 for i in range(n)] for t in range(m)],
                dtype=np.float64,
            )
            eff = (2.0 * (z_mat.T @ np.array(odd_vals)) / m).tolist()
            vac = vacant_channel_variance(odd_vals, eff, m)
            self.assertGreaterEqual(vac, 0.0)
            self.assertLessEqual(vac, vmax + 1e-12)

    def test_universal_strength3_oa_lower_bound_ordering(self) -> None:
        # Universal OA-3 floor <= dyadic character floor <= expected CUBE
        for n, m in [(28, 32), (71, 128), (90, 128)]:
            res = universal_strength3_oa_lower_bound(n, m)
            self.assertLessEqual(
                res["universal_strength3_oa_floor_ratio_vs_unshared"],
                res["dyadic_character_floor_ratio_vs_unshared"],
            )
            self.assertLessEqual(
                res["dyadic_character_floor_ratio_vs_unshared"],
                res["expected_cube_ratio_vs_unshared"],
            )
        # Asymptotically at n = M - 1, dyadic character floor approaches 2/3
        asym = universal_strength3_oa_lower_bound(1023, 1024)
        self.assertAlmostEqual(asym["dyadic_character_floor_ratio_vs_unshared"], 2.0 / 3.0, places=3)


class T4WaveMergedSchedulerTests(unittest.TestCase):
    def _synthetic_workload(self, n: int = 12, m: int = 16, nq: int = 6, seed: int = 11):
        rows, _, _ = draw_design_rows(n, m, random.Random(seed), tags=None, groups=1)
        parts = [[1000 * (i + 1) + j for j in range(40)] for i in range(n)]
        head = [1, 2, 3, 4]
        tails = [[90000 + q * 50 + j for j in range(12)] for q in range(nq)]
        bm = defaultdict(list)
        for msk in dict.fromkeys(rows):
            body = [tk for i in range(n) if (msk >> i) & 1 for tk in parts[i]]
            for q in range(nq):
                bm[msk].append((head + body + tails[q], msk, q))
        return bm

    def test_dp_matches_bruteforce_contiguous_level_merges_and_zero_mixed(self) -> None:
        bm = self._synthetic_workload(n=12, m=16, nq=5, seed=19)
        width = 8
        params = GpuBatchCostParams(c_batch=0.08, c_tok=1.5e-5, c_skew=5.0e-4, c_req=1e-4)

        batches, plan = merged_mask_trie_batches(bm, width, cost_params=params, return_merge_plan=True)
        prof = _simulate_batches_profile(batches, "merged_mask_trie")

        # Exact request conservation & zero mixed prefill+suffix batches
        expected_pairs = {(msk, q) for msk, reqs in bm.items() for _, _, q in reqs}
        seen_pairs = {(msk, q) for b in batches for _, msk, q in b}
        self.assertEqual(seen_pairs, expected_pairs)
        self.assertEqual(prof.mixed_prefill_suffix_batches, 0)
        self.assertTrue(all(len(b) <= width for b in batches))

        # Exhaustively evaluate ALL 2^(H-1) contiguous merges of mask-trie levels on prefill cost
        items, levels = _extract_mask_trie_levels(bm)
        h_levels = len(levels)
        self.assertGreaterEqual(h_levels, 2)

        def eval_partition_prefill_cost(cuts: list[int]) -> float:
            # cuts is sorted list of split indices strictly inside 1..h_levels-1
            bounds = [0] + cuts + [h_levels]
            children: list[dict[int, int]] = [{}]
            cost = 0.0
            for a, b in zip(bounds, bounds[1:]):
                idxs = [idx for ell in range(a, b) for idx in levels[ell]]
                uncs = []
                for idx in idxs:
                    toks = items[idx][2][0][0]
                    node, hit = 0, 0
                    for tk in toks:
                        nxt = children[node].get(tk)
                        if nxt is None:
                            break
                        node = nxt
                        hit += 1
                    uncs.append(len(toks) - hit)
                uncs.sort(reverse=True)
                for pos in range(0, len(uncs), width):
                    cost += params.predict_batch(uncs[pos : pos + width])
                for idx in idxs:
                    node = 0
                    for tk in items[idx][2][0][0]:
                        nxt = children[node].get(tk)
                        if nxt is None:
                            nxt = len(children)
                            children[node][tk] = nxt
                            children.append({})
                        node = nxt
            return cost

        dp_cuts = [b for _, b in plan[:-1]]
        dp_cost = eval_partition_prefill_cost(dp_cuts)

        best_bf = float("inf")
        for mask in range(1 << (h_levels - 1)):
            cuts = [i + 1 for i in range(h_levels - 1) if (mask >> i) & 1]
            c = eval_partition_prefill_cost(cuts)
            if c < best_bf:
                best_bf = c

        self.assertAlmostEqual(dp_cost, best_bf, places=10)

    def test_max_live_masks_cap_still_conserves_requests_and_zero_mixed(self) -> None:
        bm = self._synthetic_workload(n=10, m=16, nq=6, seed=23)
        prof, _ = simulate_merged_mask_trie(bm, width=8, max_live_masks=8)
        self.assertEqual(prof.requests, sum(len(v) for v in bm.values()))
        self.assertEqual(prof.mixed_prefill_suffix_batches, 0)

    def test_width_aligned_mask_trie_hits_minimum_heavy_batches_and_conserves_all(self) -> None:
        bm = self._synthetic_workload(n=12, m=16, nq=6, seed=29)
        width = 8
        batches = width_aligned_mask_trie_batches(bm, width=width)
        prof = _simulate_batches_profile(batches, "merged_mask_trie")
        expected_pairs = {(msk, q) for msk, reqs in bm.items() for _, _, q in reqs}
        seen_pairs = {(msk, q) for b in batches for _, msk, q in b}
        self.assertEqual(seen_pairs, expected_pairs)
        self.assertEqual(prof.mixed_prefill_suffix_batches, 0)
        self.assertEqual(prof.heavy_batches, math.ceil(len(bm) / width))


if __name__ == "__main__":
    unittest.main(verbosity=2)
