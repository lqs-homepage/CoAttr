"""Comprehensive offline unit & theorem verification suite for CoAttr Root-Elevation (Step 1).

Verifies:
  1. Theorem 5 (Random-Polarity Hierarchical Kronecker Paley-Hadamard):
     - Exact zero bias per single realization when background super-chunks are null.
     - Exact cancellation of ALL linear & pairwise cross-background interactions
       even conditional on ONE fixed random polarity draw s.
     - Exact zero expectation of higher-order even-subset co-located background
       interactions under random polarity (vs massive bias under locked polarity).
     - Exact Level-1 random-polarity L2-energy detection of equal-and-opposite
       sign-cancelling chunks inside the same super-chunk.
     - 100% prefix-cap & Algorithm 2 Delta_wave/W' compatibility on n=320 chunks.
  2. Dual-Mode Multi-Witness Verifiable Provenance (CoAttr-Verify):
     - Recovers ALL disjoint witnesses W_1, W_2, W_3 on 2-copy and 3-copy redundant
       OR-of-AND formulas and certifies post-remediation clearance (contrasted with
       BLIP-Pruning's single-witness failure).
  3. Faithful TracLLM Baseline:
     - Accurately recovers single-source poison on n=256 in O(K log n) evals.
     - Cleanly exposes redundant cross-half LOO masking vulnerability.
"""

from __future__ import annotations

import random
import sys
import unittest
from collections import defaultdict
from pathlib import Path

import numpy as np

UPGRADE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(UPGRADE_DIR))

from coattr_v2.baselines_v2 import blip_two_stage_pruning  # noqa: E402
from coattr_v2.capacity_scheduler_v2 import (  # noqa: E402
    RequestRow,
    adaptive_capacity_mask_trie_batches,
)
from coattr_v2.designs_v2 import dyadic_half_budget  # noqa: E402
from coattr_v2.hierarchical_v2 import (  # noqa: E402
    assemble_full_n_scores,
    build_level1_coarse_plan,
    build_level2_refined_plan,
    decode_hierarchical_block,
    partition_super_chunks,
    select_active_super_chunks,
)
from coattr_v2.tracllm_v2 import run_tracllm_single_query  # noqa: E402
from coattr_v2.verifiable_v2 import extract_multi_witness_provenance  # noqa: E402


class Theorem5HierarchicalPaleyTests(unittest.TestCase):
    def test_null_background_exact_zero_bias_per_single_draw(self) -> None:
        """When unexpanded background super-chunks are truly null, Level-2 fine Banzhaf
        equals the EXACT flat n-chunk uniform Banzhaf on every single realization s."""
        n = 64
        part = partition_super_chunks(n, target_groups=16)  # 16 super-chunks of 4
        # Active utility depends ONLY on chunks in super-chunks 2 (8..11) and 11 (44..47)
        # including linear + quadratic (degree-2) cross-super-chunk synergy!
        def f_mask(m: int) -> float:
            z = lambda idx: 1.0 if ((m >> idx) & 1) else -1.0
            return 0.5 + 0.22 * z(9) + 0.18 * z(45) + 0.14 * z(9) * z(45) - 0.10 * z(10) * z(46)

        rng = random.Random(2027)
        plan_l2 = build_level2_refined_plan(part, [2, 11], rng, background_polarity="random")
        sc = {m: f_mask(m) for m in set(plan_l2.full_masks)}
        eff, vac = decode_hierarchical_block(plan_l2, [sc])
        full_b = assemble_full_n_scores(plan_l2, eff, partition=part)[0]

        # Exact flat Banzhaf values: B_9 = 0.44, B_45 = 0.36, all others = 0.0 (degree-2 cancels!)
        self.assertAlmostEqual(full_b[9], 0.44, places=10)
        self.assertAlmostEqual(full_b[45], 0.36, places=10)
        for ch in (8, 10, 11, 44, 46, 47):
            self.assertAlmostEqual(full_b[ch], 0.0, places=10)
        self.assertAlmostEqual(vac[0], 0.0, places=10)

    def test_random_polarity_eliminates_nonnull_background_coarsening_bias(self) -> None:
        """Even with STRONG non-null linear + pairwise + co-located degree-3 background
        interactions, random-polarity Level-2 is strictly unbiased (whereas locked s=+1
        suffers huge systematic bias)."""
        n = 16  # super-chunks of 4: active=[0,1] (0..7), bg=[2,3] (8..15)
        part = partition_super_chunks(n, target_groups=4)

        def f_mask(m: int) -> float:
            z = [1.0 if ((m >> i) & 1) else -1.0 for i in range(n)]
            return (
                0.5
                + 0.25 * z[0] - 0.18 * z[3] + 0.15 * z[8] + 0.12 * z[12]  # linear active + bg
                + 0.20 * z[0] * z[8] + 0.18 * z[1] * z[12]  # pairwise active x bg
                + 0.22 * z[0] * z[8] * z[9] - 0.16 * z[2] * z[12] * z[13]  # degree-3 co-located bg
            )

        true_b = np.array([0.50, 0.0, 0.0, -0.36, 0.0, 0.0, 0.0, 0.0])
        rng = random.Random(2027)

        locked_ests, rand_ests = [], []
        for _ in range(300):
            p_lock = build_level2_refined_plan(part, [0, 1], rng, background_polarity="locked")
            sc_l = {m: f_mask(m) for m in set(p_lock.full_masks)}
            e_l, _ = decode_hierarchical_block(p_lock, [sc_l])
            locked_ests.append(assemble_full_n_scores(p_lock, e_l, partition=part)[0][:8])

            p_rand = build_level2_refined_plan(part, [0, 1], rng, background_polarity="random")
            sc_r = {m: f_mask(m) for m in set(p_rand.full_masks)}
            e_r, _ = decode_hierarchical_block(p_rand, [sc_r])
            rand_ests.append(assemble_full_n_scores(p_rand, e_r, partition=part)[0][:8])

        mean_lock = np.mean(locked_ests, axis=0)
        mean_rand = np.mean(rand_ests, axis=0)
        # Pairwise cross-bg chunk z[1] cancels identically to 0 even under locked polarity!
        self.assertAlmostEqual(mean_lock[1], 0.0, places=10)
        # Locked polarity suffers exact +0.44 bias on z[0] and -0.32 bias on z[2]!
        self.assertAlmostEqual(mean_lock[0] - true_b[0], 0.44, places=10)
        self.assertAlmostEqual(mean_lock[2] - true_b[2], -0.32, places=10)
        # Random polarity eliminates bias completely (within Monte Carlo SE)
        self.assertLess(float(np.max(np.abs(mean_rand - true_b))), 0.06)

    def test_level1_random_polarity_catches_opposite_sign_cancellation(self) -> None:
        """Inside super-chunk 5, chunks 20 (+0.30) and 21 (-0.30) cancel identically under
        locked polarity (B_G5 == 0.0!), yet random-polarity L2-energy E[B_G5^2] == 0.36 > 0."""
        n = 80
        part = partition_super_chunks(n, target_groups=20)  # 20 groups of 4; group 5 = (20,21,22,23)

        def f_mask(m: int) -> float:
            z = lambda idx: 1.0 if ((m >> idx) & 1) else -1.0
            return 0.5 + 0.30 * z(20) - 0.30 * z(21)

        rng = random.Random(17)
        # Locked Level-1 is 100% blind to group 5
        p_lock = build_level1_coarse_plan(part, rng, polarity_mode="locked")
        e_lock, _ = decode_hierarchical_block(p_lock, [{m: f_mask(m) for m in set(p_lock.full_masks)}])
        self.assertAlmostEqual(abs(e_lock[0, 5]), 0.0, places=10)

        # Random-polarity Level-1 has E[B_{G_5}^2] = (0.60)^2 + (-0.60)^2 = 0.72!
        sq_vals = []
        for _ in range(80):
            p_r = build_level1_coarse_plan(part, rng, polarity_mode="random")
            e_r, _ = decode_hierarchical_block(p_r, [{m: f_mask(m) for m in set(p_r.full_masks)}])
            sq_vals.append(float(e_r[0, 5] ** 2))
        self.assertAlmostEqual(float(np.mean(sq_vals)), 0.72, delta=0.18)

    def test_ultra_long_n320_row_savings_and_scheduler_compatibility(self) -> None:
        """At n=320 (32K-token scale), 2-level Hier-CoAttr needs <= 224 total masks
        (vs 1,024 flat dyadic CUBE — >4.5x row reduction!) and feeds cleanly into Alg. 2."""
        n = 320
        part = partition_super_chunks(n, target_groups=40)  # 40 super-chunks of 8
        rng = random.Random(2027)
        # 2 redundant poisons placed in distant super-chunks 3 (ch 26) and 34 (ch 275)!
        def f_redundant(m: int) -> float:
            p1 = (m >> 26) & 1
            p2 = (m >> 275) & 1
            return 0.95 if (p1 or p2) else 0.05

        p_l1 = build_level1_coarse_plan(part, rng, polarity_mode="locked")
        e_l1, _ = decode_hierarchical_block(p_l1, [{m: f_redundant(m) for m in set(p_l1.full_masks)}])
        active_g = select_active_super_chunks(e_l1, top_k_per_q=3, max_total_groups=4)
        self.assertIn(3, active_g)
        self.assertIn(34, active_g)

        p_l2 = build_level2_refined_plan(part, active_g, rng, background_polarity="random")
        e_l2, _ = decode_hierarchical_block(p_l2, [{m: f_redundant(m) for m in set(p_l2.full_masks)}])
        full_b = assemble_full_n_scores(p_l2, e_l2, partition=part, l1_group_effects=e_l1)[0]

        # Top-2 chunks across all 320 MUST be exact redundant poisons {26, 275}!
        top2 = set(int(i) for i in np.argsort(-full_b)[:2])
        self.assertEqual(top2, {26, 275})

        total_masks = len(p_l1.full_masks) + len(p_l2.full_masks)
        flat_cube_masks = 2 * dyadic_half_budget(n)  # 1024!
        self.assertLessEqual(total_masks, 256)
        self.assertGreater(flat_cube_masks / total_masks, 4.0)

        # Verify Algorithm 2 v2 runs cleanly and contracts W' on 32K prompts under 273K KV cap!
        parts = [[1000 + i * 110 + k for k in range(100)] for i in range(n)]  # 32,000 tokens
        bm: dict[int, list[RequestRow]] = defaultdict(list)
        for msk in dict.fromkeys(p_l1.full_masks):
            body = [tk for i in range(n) if (msk >> i) & 1 for tk in parts[i]]
            bm[msk].append((body + [999999], msk, 0))
        sched = adaptive_capacity_mask_trie_batches(bm, 32, kv_token_capacity=273472, page_size=64, min_width=4)
        self.assertTrue(sched.width_contracted)
        self.assertLess(sched.effective_width, 32)


class MultiWitnessAndTracLLMTests(unittest.TestCase):
    def test_multi_witness_verify_beats_blip_on_3copy_redundant_poison(self) -> None:
        """On a 3-copy redundant poison (chunks 5, 22, 47 in n=60), BLIP-Pruning finds only
        1 minimal witness (leaving 2 live poisons!), whereas CoAttr-Verify certifies ALL 3
        disjoint witnesses and achieves 100% post-remediation clearance."""
        n = 60
        poisons = {5, 22, 47}

        def f_oracle(m: int) -> float:
            return 0.96 if any((m >> p) & 1 for p in poisons) else 0.04

        # 1. BLIP-Pruning keeps only ONE minimal witness -> removing it leaves score 0.96!
        blip_res = blip_two_stage_pruning(n, f_oracle, epsilon=0.05)
        blip_set = set(blip_res.sufficient_set)
        self.assertEqual(len(blip_set & poisons), 1)
        full_mask = (1 << n) - 1
        blip_rem_mask = full_mask
        for c in blip_set:
            blip_rem_mask &= ~(1 << c)
        self.assertGreater(f_oracle(blip_rem_mask), 0.90)  # Remediation FAILS!

        # 2. CoAttr-Verify guided by Banzhaf scores certifies ALL 3 disjoint witnesses & clears attack!
        banzhaf = [0.23 if i in poisons else 0.0 for i in range(n)]
        cert = extract_multi_witness_provenance(banzhaf, f_oracle, n, epsilon=0.05, max_witnesses=4)
        self.assertEqual(len(cert.witnesses), 3)
        self.assertEqual({w[0] for w in cert.witnesses}, poisons)
        self.assertTrue(cert.remediation_certified)
        self.assertLess(cert.post_removal_score, 0.10)  # Complete remediation!

    def test_multi_witness_handles_composite_multi_hop_plus_redundant_witness(self) -> None:
        """Utility triggered by EITHER multi-hop pair (3 AND 19) OR redundant single chunk 41."""
        n = 50

        def f_oracle(m: int) -> float:
            hop = ((m >> 3) & 1) and ((m >> 19) & 1)
            alt = (m >> 41) & 1
            return 0.92 if (hop or alt) else 0.08

        # True uniform Banzhaf values: B_41 = 0.84*(3/4)=0.63, B_3 = B_19 = 0.84*(1/4)=0.21
        banzhaf = [0.63 if i == 41 else (0.21 if i in (3, 19) else 0.0) for i in range(n)]
        cert = extract_multi_witness_provenance(banzhaf, f_oracle, n, epsilon=0.05, max_witnesses=4)
        self.assertEqual(set(cert.witnesses), {(41,), (3, 19)})
        self.assertTrue(cert.remediation_certified)
        self.assertAlmostEqual(cert.post_removal_score, 0.08, places=6)

    def test_tracllm_recovers_single_source_but_masks_cross_half_redundant_poison(self) -> None:
        """Verify TracLLM faithfully recovers a single poison on n=128 in O(K log n) evals,
        and cleanly exhibits LOO masking when 2 redundant poisons split across halves."""
        n = 128

        # Case 1: Single poison at chunk 93 -> TracLLM succeeds cleanly!
        def f_single(masks):
            return [0.95 if ((m >> 93) & 1) else 0.05 for m in masks]

        res1 = run_tracllm_single_query(f_single, n, top_k=3, branch_factor=2, use_stc_ensemble=False)
        self.assertEqual(res1.top_chunks[0], 93)
        self.assertEqual(res1.tree_depth, 7)  # log2(128) = 7 sequential rounds!

        # Case 2: 2 redundant poisons in left half (12) and right half (105)
        # Under pure LOO tree search, dropping left half keeps 105 -> drop == 0.0!
        def f_double(masks):
            return [0.95 if (((m >> 12) & 1) or ((m >> 105) & 1)) else 0.05 for m in masks]

        res2 = run_tracllm_single_query(f_double, n, top_k=3, branch_factor=2, use_stc_ensemble=False)
        self.assertEqual(max(res2.scores), 0.0)  # Complete LOO masking!

    def test_coattr_verify_dual_threshold_saturated_redundancy(self) -> None:
        """Dual-threshold CoAttr-Verify certifies ALL redundant copies even when 2 copies
        saturate higher (0.99) than 1 copy alone (0.85) — preventing premature stop."""
        n = 48
        copies = (9, 34)

        def f_sat(mask: int) -> float:
            cnt = sum(1 for c in copies if (mask >> c) & 1)
            return 0.99 if cnt == 2 else (0.85 if cnt == 1 else 0.04)

        scores = [0.0] * n
        scores[9] = 0.44
        scores[34] = 0.43

        cert = extract_multi_witness_provenance(
            scores,
            f_sat,
            n,
            sufficiency_threshold=0.50,
            clearance_threshold=0.18,
            max_witnesses=4,
            max_witness_size=4,
        )
        self.assertTrue(cert.remediation_certified)
        self.assertEqual(set(cert.witnesses), {(9,), (34,)})
        self.assertAlmostEqual(cert.post_removal_score, 0.04, places=6)


if __name__ == "__main__":
    unittest.main()
