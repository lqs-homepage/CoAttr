"""Full-Scale Step-2 B200 Benchmark Runner for CoAttr Root-Elevation.

Supports three clean modes executed in parallel across 4 exclusive B200 GPUs:
  1. ``cuad``: Multi-Predicate Semantic SQL ``sem_filter`` compliance auditing on
     real lawyer-annotated CUAD contracts (standard n=78..91 + ultra-long n=213..295),
     comparing LOO, ContextCite/LASSO (matched budget B), TracLLM (USENIX'25),
     BLIP-Pruning (PVLDB'26), CoAttr Paley / Hier-CoAttr (random & hybrid L1 + L2
     locked-bg Theorem 5 ablation), CoAttr-Verify (dual-threshold), and optional AttnTrace.
  2. ``forensics``: Long-Context Adversarial Security Forensics (Std71 & Long240,
     Single / Redund2 / Redund3 / SplitInj, nq=5 incidents per workload = 40/model),
     evaluating Attack-Source Recall@k, Exact All-Sources, live Post-Sanitization
     Answer Recovery, and Certified Multi-Witness Sanitization Recovery.
  3. ``gemma4_extreme``: Extreme-Long Scaling (n=320..410, 32K..46K tokens) on
     gemma-4-31B-it (262K native context), including a 3-rep cold-cache physical
     scheduler microbenchmark comparing SGLang Native LPM, BatchLLM-DFS, Alg. 2 v1
     (fixed W=32), and Alg. 2 v2 (adaptive W' contraction) on 33K-token Hier-CoAttr!
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
UPGRADE_DIR = THIS_DIR.parent
for p in (str(THIS_DIR), str(UPGRADE_DIR), str(UPGRADE_DIR / "m2_bench")):
    if p not in sys.path:
        sys.path.insert(0, p)

from coattr_v2.baselines_v2 import blip_two_stage_pruning  # noqa: E402
from coattr_v2.designs_v2 import draw_kronecker_paley_block, select_best_paley_configs  # noqa: E402
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
import m2_common  # noqa: E402
import r2_common  # noqa: E402


def _save_atomic(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _strip_eos_ids(tokens: list[int], tokenizer) -> list[int]:
    ends = {tokenizer.eos_token_id, tokenizer.pad_token_id}
    for tok_str in ("<end_of_turn>", "<turn|>", "<|im_end|>"):
        tid = tokenizer.convert_tokens_to_ids(tok_str)
        if isinstance(tid, int) and tid >= 0:
            ends.add(tid)
    out = list(tokens)
    while out and out[-1] in ends:
        out.pop()
    return out


def _eval_cuad_ranking(
    engine,
    doc: r2_common.FastDocument,
    rec: dict,
    active_qs: list[int],
    scores_mat: np.ndarray,
    yes_ids: list[int],
    no_ids: list[int],
    kv_cap: int,
    temp: float,
    meta: dict,
    label: str,
) -> dict:
    full_mask = (1 << doc.n) - 1
    rec_ks, mrrs, flips_k3, flips_k5, mw_covs = [], [], [], [], []
    per_q_details = []
    rem_masks = []
    for qi, q in enumerate(active_qs):
        order = np.argsort(-scores_mat[qi])
        gold = set(rec["gold_supports"][q])
        k = max(1, len(gold))
        rec_ks.append(len(set(int(x) for x in order[: max(k, 3)]) & gold) / k)
        # MRR of first gold clause chunk
        rank1 = next((r + 1 for r, ch in enumerate(order) if int(ch) in gold), doc.n)
        mrrs.append(1.0 / rank1)
        m3 = full_mask
        for ch in order[:3]:
            m3 &= ~(1 << int(ch))
        m5 = full_mask
        for ch in order[:5]:
            m5 &= ~(1 << int(ch))
        rem_masks.extend([m3, m5])

    _, diff_rem, _ = r2_common.score_semfilter_logodds_batch(
        engine, doc, rem_masks, active_qs, yes_ids, no_ids, kv_cap, temp=temp
    )
    for qi, q in enumerate(active_qs):
        order = np.argsort(-scores_mat[qi])
        m3 = full_mask
        for ch in order[:3]:
            m3 &= ~(1 << int(ch))
        m5 = full_mask
        for ch in order[:5]:
            m5 &= ~(1 << int(ch))
        f3 = 1.0 if diff_rem[q][m3] <= 0.0 else 0.0
        f5 = 1.0 if diff_rem[q][m5] <= 0.0 else 0.0
        flips_k3.append(f3)
        flips_k5.append(f5)
        wgs = rec["gold_witness_groups"][q]
        mw_c = None
        if len(wgs) >= 2:
            t5 = set(int(x) for x in order[:5])
            mw_c = sum(1 for wg in wgs if set(wg) & t5) / len(wgs)
            mw_covs.append(mw_c)
        per_q_details.append(
            {
                "q": int(q),
                "category": rec["categories"][q],
                "clause_rec": round(float(rec_ks[qi]), 4),
                "mrr": round(float(mrrs[qi]), 4),
                "flip_k3": f3,
                "flip_k5": f5,
                "mw_cov_top5": round(float(mw_c), 4) if mw_c is not None else None,
            }
        )

    out = {
        "wall_s": round(float(meta["wall_s"]), 2),
        "clause_rec_at_maxk3": round(float(np.mean(rec_ks)), 4),
        "clause_mrr": round(float(np.mean(mrrs)), 4),
        "decision_flip_k3": round(float(np.mean(flips_k3)), 4),
        "decision_flip_k5": round(float(np.mean(flips_k5)), 4),
        "multi_witness_cov_top5": round(float(np.mean(mw_covs)), 4) if mw_covs else None,
        "per_q": per_q_details,
    }
    for k_ex in ("n_masks", "total_evals", "input_tokens", "cached_tokens", "uncached_tokens", "l2_groups", "l2_vars", "g", "L", "effective_width", "mean_witness_size", "witness_removal_flip_rate"):
        if k_ex in meta:
            out[k_ex] = meta[k_ex]
    print(
        f"    {label:28s} wall={out['wall_s']:6.2f}s rec={out['clause_rec_at_maxk3']:.4f} "
        f"mrr={out['clause_mrr']:.4f} mw5={out['multi_witness_cov_top5']} "
        f"flip3={out['decision_flip_k3']:.3f} flip5={out['decision_flip_k5']:.3f}",
        flush=True,
    )
    return out


def run_cuad_suite(
    engine,
    tokenizer,
    cuad_items: list[dict],
    kv_cap: int,
    ctx_limit: int,
    temp: float,
    out_dir: Path,
    *,
    max_q_per_doc: int = 10,
    run_l2_locked_ablation_count: int = 2,
) -> tuple[dict, dict[str, list[int]]]:
    yes_ids = tokenizer.encode("YES", add_special_tokens=False)
    no_ids = tokenizer.encode("NO", add_special_tokens=False)
    results: dict[str, dict] = {}
    active_qs_by_doc: dict[str, list[int]] = {}
    l2_ablation_done = 0

    for rec in cuad_items:
        doc = r2_common.FastDocument(tokenizer, rec)
        if doc.full_tokens > ctx_limit - 256:
            print(f"[CUAD SKIP] {rec['doc_id']} full_tokens={doc.full_tokens} > ctx_limit={ctx_limit}", flush=True)
            continue
        n = doc.n
        nq = len(rec["questions"])
        full_mask = (1 << n) - 1
        all_q = list(range(nq))

        m2_common.flush_cache(engine)
        gen_chk = engine.generate(
            input_ids=[doc.prompt(full_mask, q) for q in all_q],
            sampling_params={"max_new_tokens": 4, "temperature": 0.0},
        )
        gen_strs = [o["text"].strip() for o in gen_chk]

        m2_common.flush_cache(engine)
        _, diff_fe, _ = r2_common.score_semfilter_logodds_batch(
            engine, doc, [full_mask, 0], all_q, yes_ids, no_ids, kv_cap, temp=temp
        )
        yes_preds = [q for q in all_q if diff_fe[q][full_mask] > 0.0 and diff_fe[q][full_mask] > diff_fe[q][0] + 0.5]
        yes_preds_sorted = sorted(
            yes_preds,
            key=lambda q: (-len(rec["gold_witness_groups"][q]), -len(rec["gold_supports"][q]), q),
        )
        active_qs = yes_preds_sorted[:max_q_per_doc]
        if not active_qs:
            print(f"[CUAD SKIP] {rec['doc_id']} had 0 YES-triggered predicates", flush=True)
            continue
        active_qs_by_doc[rec["doc_id"]] = active_qs

        med_full = float(np.median([diff_fe[q][full_mask] for q in active_qs]))
        med_empty = float(np.median([diff_fe[q][0] for q in active_qs]))
        n_mw = sum(len(rec["gold_witness_groups"][q]) >= 2 for q in active_qs)
        print(
            f"\n[CUAD] {rec['doc_id']} (tier={rec['tier']}, n={n}, tok={doc.full_tokens}, "
            f"YES={len(yes_preds)}/{nq}, test_q={len(active_qs)}, mw={n_mw}, "
            f"med_full={med_full:.2f}, med_empty={med_empty:.2f})",
            flush=True,
        )

        doc_res = {
            "doc_id": rec["doc_id"],
            "tier": rec["tier"],
            "n": n,
            "full_tokens": doc.full_tokens,
            "yes_triggered": len(yes_preds),
            "greedy_yes_count": sum(g.upper().startswith("YES") for g in gen_strs),
            "tested_q": len(active_qs),
            "tested_multi_witness_q": n_mw,
            "median_diff_full": round(med_full, 3),
            "median_diff_empty": round(med_empty, 3),
            "temp": temp,
            "methods": {},
        }

        # 1. LOO
        m2_common.flush_cache(engine)
        loo_masks = [full_mask] + [full_mask ^ (1 << i) for i in range(n)]
        cal_loo, _, st_loo = r2_common.score_semfilter_logodds_batch(
            engine, doc, loo_masks, active_qs, yes_ids, no_ids, kv_cap, temp=temp
        )
        sc_loo = np.array([[cal_loo[q][full_mask] - cal_loo[q][full_mask ^ (1 << i)] for i in range(n)] for q in active_qs])
        doc_res["methods"]["loo"] = _eval_cuad_ranking(
            engine, doc, rec, active_qs, sc_loo, yes_ids, no_ids, kv_cap, temp, {**st_loo, "n_masks": len(loo_masks)}, "LOO"
        )

        # 2. CoAttr (Paley R=1 for standard; Hier-CoAttr random & hybrid for ultra/extreme_long)
        seed_base = m2_common.stable_seed("r2-cuad", rec["doc_id"])
        if rec["tier"] == "standard":
            m2_common.flush_cache(engine)
            cfg = select_best_paley_configs(n, groups=(2, 4), mod8_eq4_only=True, strictly_smaller_than_dyadic=True)[0]
            blk = draw_kronecker_paley_block(n, int(cfg["g"]), int(cfg["L"]), random.Random(seed_base))
            cal_p, _, st_p = r2_common.score_semfilter_logodds_batch(
                engine, doc, blk.rows, active_qs, yes_ids, no_ids, kv_cap, temp=temp
            )
            z_mat = np.array([[1.0 if ((blk.rows[2 * t] >> i) & 1) else -1.0 for i in range(n)] for t in range(blk.m)])
            sc_p = np.zeros((len(active_qs), n))
            for qi, q in enumerate(active_qs):
                odd = np.array([0.5 * (cal_p[q][blk.rows[2 * t]] - cal_p[q][blk.rows[2 * t + 1]]) for t in range(blk.m)])
                sc_p[qi] = 2.0 * (z_mat.T @ odd) / blk.m
            doc_res["methods"]["coattr_paley"] = _eval_cuad_ranking(
                engine, doc, rec, active_qs, sc_p, yes_ids, no_ids, kv_cap, temp,
                {**st_p, "n_masks": len(blk.rows), "g": blk.g, "L": blk.L}, "CoAttr Paley R=1"
            )
            best_scores = sc_p
            matched_budget = len(blk.rows)
        else:
            t_groups = 36 if n <= 300 else 44
            part = partition_super_chunks(n, target_groups=t_groups)
            max_g_l2 = min(part.num_groups, max(8, min(14, len(active_qs) * 2)))

            # 2a. Hier-CoAttr (random L1 + random L2, Theorem 5 flagship)
            m2_common.flush_cache(engine)
            r_rd = random.Random(seed_base + 1)
            p1_rd = build_level1_coarse_plan(part, r_rd, polarity_mode="random")
            c1_rd, _, st1_rd = r2_common.score_semfilter_logodds_batch(
                engine, doc, p1_rd.full_masks, active_qs, yes_ids, no_ids, kv_cap, temp=temp
            )
            e1_rd, _ = decode_hierarchical_block(p1_rd, [c1_rd[q] for q in active_qs])
            act_rd = select_active_super_chunks(e1_rd, top_k_per_q=2, max_total_groups=max_g_l2)
            p2_rd = build_level2_refined_plan(part, act_rd, r_rd, background_polarity="random")
            c2_rd, _, st2_rd = r2_common.score_semfilter_logodds_batch(
                engine, doc, p2_rd.full_masks, active_qs, yes_ids, no_ids, kv_cap, temp=temp
            )
            e2_rd, _ = decode_hierarchical_block(p2_rd, [c2_rd[q] for q in active_qs])
            full_rd = assemble_full_n_scores(
                p2_rd, e2_rd, partition=part, l1_group_effects=e1_rd, l1_chunk_polarities=p1_rd.chunk_polarities
            )
            meta_rd = {
                "wall_s": st1_rd["wall_s"] + st2_rd["wall_s"],
                "n_masks": len(p1_rd.full_masks) + len(p2_rd.full_masks),
                "l2_groups": len(act_rd),
                "l2_vars": p2_rd.n_vars,
                "input_tokens": st1_rd["input_tokens"] + st2_rd["input_tokens"],
                "cached_tokens": st1_rd["cached_tokens"] + st2_rd["cached_tokens"],
                "uncached_tokens": st1_rd["uncached_tokens"] + st2_rd["uncached_tokens"],
                "effective_width": min(st1_rd["effective_width"], st2_rd["effective_width"]),
            }
            doc_res["methods"]["hier_coattr_random"] = _eval_cuad_ranking(
                engine, doc, rec, active_qs, full_rd, yes_ids, no_ids, kv_cap, temp, meta_rd, "Hier-CoAttr (random, Thm.5)"
            )
            best_scores = full_rd
            matched_budget = meta_rd["n_masks"]

            # 2b. Theorem 5 live ablation: Level-2 LOCKED background polarity (on first 2 ultra-long docs)
            if l2_ablation_done < run_l2_locked_ablation_count:
                m2_common.flush_cache(engine)
                p2_lkbg = build_level2_refined_plan(part, act_rd, random.Random(seed_base + 9), background_polarity="locked")
                c2_lkbg, _, st2_lkbg = r2_common.score_semfilter_logodds_batch(
                    engine, doc, p2_lkbg.full_masks, active_qs, yes_ids, no_ids, kv_cap, temp=temp
                )
                e2_lkbg, _ = decode_hierarchical_block(p2_lkbg, [c2_lkbg[q] for q in active_qs])
                full_lkbg = assemble_full_n_scores(
                    p2_lkbg, e2_lkbg, partition=part, l1_group_effects=e1_rd, l1_chunk_polarities=p1_rd.chunk_polarities
                )
                doc_res["methods"]["hier_coattr_l2locked_bg_ablation"] = _eval_cuad_ranking(
                    engine, doc, rec, active_qs, full_lkbg, yes_ids, no_ids, kv_cap, temp,
                    {"wall_s": st1_rd["wall_s"] + st2_lkbg["wall_s"], "n_masks": len(p1_rd.full_masks) + len(p2_lkbg.full_masks)},
                    "Hier-CoAttr (L2-locked abl)",
                )
                l2_ablation_done += 1

            # 2c. Hier-CoAttr (hybrid L1: locked + random RMS -> random L2)
            m2_common.flush_cache(engine)
            r_lk = random.Random(seed_base + 2)
            p1_lk = build_level1_coarse_plan(part, r_lk, polarity_mode="locked")
            c1_lk, _, st1_lk = r2_common.score_semfilter_logodds_batch(
                engine, doc, p1_lk.full_masks, active_qs, yes_ids, no_ids, kv_cap, temp=temp
            )
            e1_lk, _ = decode_hierarchical_block(p1_lk, [c1_lk[q] for q in active_qs])
            act_hyb = select_active_super_chunks(
                e1_lk, top_k_per_q=2, max_total_groups=max_g_l2, extra_level1_scores_per_q=e1_rd
            )
            p2_hyb = build_level2_refined_plan(part, act_hyb, random.Random(seed_base + 3), background_polarity="random")
            c2_hyb, _, st2_hyb = r2_common.score_semfilter_logodds_batch(
                engine, doc, p2_hyb.full_masks, active_qs, yes_ids, no_ids, kv_cap, temp=temp
            )
            e2_hyb, _ = decode_hierarchical_block(p2_hyb, [c2_hyb[q] for q in active_qs])
            full_hyb = assemble_full_n_scores(
                p2_hyb, e2_hyb, partition=part, l1_group_effects=e1_lk, l1_chunk_polarities=p1_lk.chunk_polarities
            )
            meta_hyb = {
                "wall_s": st1_lk["wall_s"] + st1_rd["wall_s"] + st2_hyb["wall_s"],
                "n_masks": len(p1_lk.full_masks) + len(p1_rd.full_masks) + len(p2_hyb.full_masks),
                "l2_groups": len(act_hyb),
                "l2_vars": p2_hyb.n_vars,
                "input_tokens": st1_lk["input_tokens"] + st1_rd["input_tokens"] + st2_hyb["input_tokens"],
                "cached_tokens": st1_lk["cached_tokens"] + st1_rd["cached_tokens"] + st2_hyb["cached_tokens"],
                "uncached_tokens": st1_lk["uncached_tokens"] + st1_rd["uncached_tokens"] + st2_hyb["uncached_tokens"],
            }
            doc_res["methods"]["hier_coattr_hybrid"] = _eval_cuad_ranking(
                engine, doc, rec, active_qs, full_hyb, yes_ids, no_ids, kv_cap, temp, meta_hyb, "Hier-CoAttr (hybrid L1)"
            )

        # 3. ContextCite / LASSO (NeurIPS'24) with EXACT matched mask budget B
        m2_common.flush_cache(engine)
        cc_masks = r2_common.draw_contextcite_iid_masks(n, matched_budget, random.Random(seed_base + 77))
        cal_cc, _, st_cc = r2_common.score_semfilter_logodds_batch(
            engine, doc, cc_masks, active_qs, yes_ids, no_ids, kv_cap, temp=temp
        )
        t_fit0 = time.perf_counter()
        sc_cc = np.zeros((len(active_qs), n))
        for qi, q in enumerate(active_qs):
            y_q = [cal_cc[q][m] for m in cc_masks]
            sc_cc[qi] = r2_common.fit_contextcite_lasso(cc_masks, y_q, n, alpha=0.005)
        fit_s = time.perf_counter() - t_fit0
        doc_res["methods"]["contextcite_lasso"] = _eval_cuad_ranking(
            engine, doc, rec, active_qs, sc_cc, yes_ids, no_ids, kv_cap, temp,
            {**st_cc, "wall_s": st_cc["wall_s"] + fit_s, "n_masks": len(cc_masks)}, "ContextCite / LASSO"
        )

        # 4. TracLLM (USENIX Security'25)
        m2_common.flush_cache(engine)
        t_tr0 = time.perf_counter()
        sc_tr = np.zeros((len(active_qs), n))
        tr_evals = 0
        for qi, q in enumerate(active_qs):
            def eval_b_tr(ms):
                c, _, _ = r2_common.score_semfilter_logodds_batch(engine, doc, ms, [q], yes_ids, no_ids, kv_cap, temp=temp)
                return [c[q][m] for m in ms]
            res_tr = run_tracllm_single_query(
                eval_b_tr, n, top_k=5, branch_factor=2, use_stc_ensemble=True,
                full_score=r2_common.sigmoid(diff_fe[q][full_mask] / temp),
                empty_score=r2_common.sigmoid(diff_fe[q][0] / temp),
            )
            sc_tr[qi] = np.array(res_tr.scores)
            tr_evals += res_tr.total_evals
        doc_res["methods"]["tracllm"] = _eval_cuad_ranking(
            engine, doc, rec, active_qs, sc_tr, yes_ids, no_ids, kv_cap, temp,
            {"wall_s": time.perf_counter() - t_tr0, "total_evals": tr_evals}, "TracLLM (USENIX'25)"
        )

        # 5. BLIP-Pruning (PVLDB'26)
        m2_common.flush_cache(engine)
        t_bl0 = time.perf_counter()
        sc_bl = np.zeros((len(active_qs), n))
        blip_remed, blip_sizes, blip_evals = [], [], 0
        for qi, q in enumerate(active_qs):
            def eval_one_bl(m: int) -> float:
                c, _, _ = r2_common.score_semfilter_logodds_batch(engine, doc, [m], [q], yes_ids, no_ids, kv_cap, temp=temp)
                return c[q][m]
            b_res = blip_two_stage_pruning(n, eval_one_bl, epsilon=0.05)
            sc_bl[qi] = np.array(b_res.scores)
            blip_sizes.append(len(b_res.sufficient_set))
            blip_evals += b_res.n_evals
            m_rem = full_mask
            for ch in b_res.sufficient_set:
                m_rem &= ~(1 << int(ch))
            _, d_rem, _ = r2_common.score_semfilter_logodds_batch(engine, doc, [m_rem], [q], yes_ids, no_ids, kv_cap, temp=temp)
            blip_remed.append(1.0 if d_rem[q][m_rem] <= 0.0 else 0.0)
        doc_res["methods"]["blip_pruning"] = _eval_cuad_ranking(
            engine, doc, rec, active_qs, sc_bl, yes_ids, no_ids, kv_cap, temp,
            {
                "wall_s": time.perf_counter() - t_bl0,
                "total_evals": blip_evals,
                "mean_witness_size": round(float(np.mean(blip_sizes)), 2),
                "witness_removal_flip_rate": round(float(np.mean(blip_remed)), 4),
            },
            "BLIP-Pruning (VLDB'26)",
        )

        # 6. CoAttr-Verify (Dual-Threshold Multi-Witness Provenance)
        t_cv0 = time.perf_counter()
        cv_remed, cv_cert_ok, cv_n_wit, cv_evals, cv_union_sz = [], [], [], [], []
        for qi, q in enumerate(active_qs):
            def eval_one_cv(m: int) -> float:
                c, _, _ = r2_common.score_semfilter_logodds_batch(engine, doc, [m], [q], yes_ids, no_ids, kv_cap, temp=temp)
                return c[q][m]
            f_full_q = r2_common.sigmoid(diff_fe[q][full_mask] / temp)
            f_empty_q = r2_common.sigmoid(diff_fe[q][0] / temp)
            cert = extract_multi_witness_provenance(
                best_scores[qi],
                eval_one_cv,
                n,
                epsilon=0.08,
                sufficiency_threshold=min(f_full_q - 0.05, 0.55),
                clearance_threshold=0.50,
                max_witnesses=5,
                max_witness_size=6,
                full_score=f_full_q,
                empty_score=f_empty_q,
            )
            cv_n_wit.append(len(cert.witnesses))
            cv_evals.append(cert.verification_evals)
            cv_cert_ok.append(1.0 if cert.remediation_certified else 0.0)
            union_w = set(ch for w in cert.witnesses for ch in w)
            cv_union_sz.append(len(union_w))
            m_rem = full_mask
            for ch in union_w:
                m_rem &= ~(1 << int(ch))
            _, d_rem, _ = r2_common.score_semfilter_logodds_batch(engine, doc, [m_rem], [q], yes_ids, no_ids, kv_cap, temp=temp)
            cv_remed.append(1.0 if d_rem[q][m_rem] <= 0.0 else 0.0)
        doc_res["coattr_verify"] = {
            "extra_wall_s": round(time.perf_counter() - t_cv0, 2),
            "mean_witnesses_found": round(float(np.mean(cv_n_wit)), 2),
            "mean_union_chunks": round(float(np.mean(cv_union_sz)), 2),
            "mean_extra_evals_per_q": round(float(np.mean(cv_evals)), 1),
            "remediation_certified_rate": round(float(np.mean(cv_cert_ok)), 4),
            "witness_removal_flip_rate": round(float(np.mean(cv_remed)), 4),
        }
        print(
            f"    {'CoAttr-Verify (Multi-Wit)':28s} +wall={doc_res['coattr_verify']['extra_wall_s']:5.2f}s "
            f"wit={doc_res['coattr_verify']['mean_witnesses_found']:.2f} "
            f"union={doc_res['coattr_verify']['mean_union_chunks']:.2f} "
            f"cert_flip={doc_res['coattr_verify']['witness_removal_flip_rate']:.3f} "
            f"(vs BLIP {doc_res['methods']['blip_pruning']['witness_removal_flip_rate']:.3f})",
            flush=True,
        )

        results[rec["doc_id"]] = doc_res
        _save_atomic(out_dir / f"ckpt_cuad_{rec['doc_id']}.json", doc_res)
    return results, active_qs_by_doc


def run_forensics_suite(
    engine,
    tokenizer,
    forensics_items: list[dict],
    kv_cap: int,
    out_dir: Path,
) -> dict:
    results: dict[str, dict] = {}
    for rec in forensics_items:
        doc = r2_common.FastDocument(tokenizer, rec)
        n = doc.n
        nq = len(rec["questions"])
        all_q = list(range(nq))
        full_mask = (1 << n) - 1

        # 1. Verify LIVE HIJACKING (per-query clean mask removes ONLY that query's attack chunks!)
        m2_common.flush_cache(engine)
        clean_prompts, poison_prompts = [], []
        for q in all_q:
            cm = full_mask
            for ac in rec["gold_supports"][q]:
                cm &= ~(1 << ac)
            clean_prompts.append(doc.prompt(cm, q))
            poison_prompts.append(doc.prompt(full_mask, q))
        gen_outs = engine.generate(
            input_ids=clean_prompts + poison_prompts,
            sampling_params={"max_new_tokens": 32, "temperature": 0.0},
        )
        clean_texts = [o["text"].strip() for o in gen_outs[:nq]]
        poison_texts = [o["text"].strip() for o in gen_outs[nq:]]
        benign_ok = [rec["benign_answers"][q].lower() in clean_texts[q].lower() for q in all_q]
        hijacked = [rec["attack_answers"][q].lower() in poison_texts[q].lower() for q in all_q]
        print(
            f"\n[FORENSICS] {rec['doc_id']} (mode={rec['attack_mode']}, n={n}, tok={doc.full_tokens}, nq={nq}): "
            f"benign_ok={sum(benign_ok)}/{nq} hijacked={sum(hijacked)}/{nq}",
            flush=True,
        )
        for q in all_q:
            print(f"    q{q}: clean='{clean_texts[q][:48]}' | poison='{poison_texts[q][:48]}'", flush=True)

        # Zero-leakage observed response tokens
        ans_tok_map = {}
        for q in all_q:
            raw_ids = list(gen_outs[nq + q].get("output_ids") or tokenizer.encode(poison_texts[q], add_special_tokens=False))
            toks = _strip_eos_ids(raw_ids, tokenizer)
            ans_tok_map[q] = toks if toks else tokenizer.encode(rec["attack_answers"][q], add_special_tokens=False)

        # SplitInj live AND-synergy verification
        split_synergy = None
        if rec["attack_mode"] == "split_prompt_injection":
            syn_p = []
            for q in all_q:
                p_a, p_b = rec["gold_supports"][q]
                syn_p.append(doc.prompt(full_mask ^ (1 << p_a), q))
                syn_p.append(doc.prompt(full_mask ^ (1 << p_b), q))
            syn_outs = engine.generate(input_ids=syn_p, sampling_params={"max_new_tokens": 32, "temperature": 0.0})
            da = sum(
                rec["benign_answers"][q].lower() in syn_outs[2 * q]["text"].lower()
                and rec["attack_answers"][q].lower() not in syn_outs[2 * q]["text"].lower()
                for q in all_q
            )
            db = sum(
                rec["benign_answers"][q].lower() in syn_outs[2 * q + 1]["text"].lower()
                and rec["attack_answers"][q].lower() not in syn_outs[2 * q + 1]["text"].lower()
                for q in all_q
            )
            split_synergy = {"drop_A_recovers": da, "drop_B_recovers": db, "nq": nq}
            print(f"    [SPLIT SYNERGY] {split_synergy}", flush=True)

        def eval_forensic(scores_mat: np.ndarray, meta: dict, label: str) -> dict:
            rec_ks, rec_exact, sanitized_recover = [], [], []
            san_prompts = []
            for qi, q in enumerate(all_q):
                gold = set(rec["gold_supports"][q])
                k = len(gold)
                order = np.argsort(-scores_mat[qi])
                top_k_set = set(int(x) for x in order[:k])
                rec_ks.append(len(top_k_set & gold) / k)
                rec_exact.append(1.0 if top_k_set == gold else 0.0)
                san_mask = full_mask
                for ch in top_k_set:
                    san_mask &= ~(1 << ch)
                san_prompts.append(doc.prompt(san_mask, q))
            san_outs = engine.generate(input_ids=san_prompts, sampling_params={"max_new_tokens": 32, "temperature": 0.0})
            san_texts = []
            for q, o in enumerate(san_outs):
                txt = o["text"].strip()
                san_texts.append(txt[:60])
                recov = (rec["benign_answers"][q].lower() in txt.lower()) and (rec["attack_answers"][q].lower() not in txt.lower())
                sanitized_recover.append(1.0 if recov else 0.0)
            out = {
                "wall_s": round(float(meta["wall_s"]), 2),
                "attack_rec_at_k": round(float(np.mean(rec_ks)), 4),
                "exact_all_sources": round(float(np.mean(rec_exact)), 4),
                "post_sanitization_recovery": round(float(np.mean(sanitized_recover)), 4),
                "san_texts": san_texts,
            }
            for k_ex in ("n_masks", "total_evals", "input_tokens", "cached_tokens", "uncached_tokens", "l2_groups", "l2_vars", "witness_sanitization_recovery"):
                if k_ex in meta:
                    out[k_ex] = meta[k_ex]
            print(
                f"    {label:28s} wall={out['wall_s']:6.2f}s rec={out['attack_rec_at_k']:.4f} "
                f"exact={out['exact_all_sources']:.4f} recover={out['post_sanitization_recovery']:.4f}",
                flush=True,
            )
            return out

        w_res = {
            "doc_id": rec["doc_id"],
            "n": n,
            "full_tokens": doc.full_tokens,
            "mode": rec["attack_mode"],
            "nq": nq,
            "benign_ok": sum(benign_ok),
            "hijacked": sum(hijacked),
            "split_synergy": split_synergy,
            "methods": {},
        }

        seed_base = m2_common.stable_seed("r2-for", rec["doc_id"])

        # 1. LOO / CausalArmor
        m2_common.flush_cache(engine)
        loo_masks = [full_mask] + [full_mask ^ (1 << i) for i in range(n)]
        sc_m, st_loo = r2_common.score_answer_batch_sglang(engine, doc, loo_masks, all_q, ans_tok_map, kv_cap)
        sc_loo = np.array([[sc_m[q][full_mask] - sc_m[q][full_mask ^ (1 << i)] for i in range(n)] for q in all_q])
        w_res["methods"]["loo"] = eval_forensic(sc_loo, {**st_loo, "n_masks": len(loo_masks)}, "LOO / CausalArmor")

        # 2. CoAttr (Paley for Std71, Hier-CoAttr random & hybrid for Long240/Ext360)
        if n <= 90:
            m2_common.flush_cache(engine)
            cfg = select_best_paley_configs(n, groups=(2, 4), mod8_eq4_only=True, strictly_smaller_than_dyadic=True)[0]
            blk = draw_kronecker_paley_block(n, int(cfg["g"]), int(cfg["L"]), random.Random(seed_base))
            sc_p, st_p = r2_common.score_answer_batch_sglang(engine, doc, blk.rows, all_q, ans_tok_map, kv_cap)
            z_mat = np.array([[1.0 if ((blk.rows[2 * t] >> i) & 1) else -1.0 for i in range(n)] for t in range(blk.m)])
            b_p = np.zeros((nq, n))
            for q in all_q:
                odd = np.array([0.5 * (sc_p[q][blk.rows[2 * t]] - sc_p[q][blk.rows[2 * t + 1]]) for t in range(blk.m)])
                b_p[q] = 2.0 * (z_mat.T @ odd) / blk.m
            w_res["methods"]["coattr_paley"] = eval_forensic(b_p, {**st_p, "n_masks": len(blk.rows)}, "CoAttr Paley R=1")
            best_scores = b_p
            matched_budget = len(blk.rows)
        else:
            t_groups = 36 if n <= 300 else 44
            part = partition_super_chunks(n, target_groups=t_groups)
            max_g_l2 = min(part.num_groups, max(10, min(14, nq * 3)))

            # Random L1 + Random L2 (Theorem 5)
            m2_common.flush_cache(engine)
            r_rd = random.Random(seed_base + 1)
            p1_rd = build_level1_coarse_plan(part, r_rd, polarity_mode="random")
            s1_rd, st1_rd = r2_common.score_answer_batch_sglang(engine, doc, p1_rd.full_masks, all_q, ans_tok_map, kv_cap)
            e1_rd, _ = decode_hierarchical_block(p1_rd, [s1_rd[q] for q in all_q])
            act_rd = select_active_super_chunks(e1_rd, top_k_per_q=3, max_total_groups=max_g_l2)
            p2_rd = build_level2_refined_plan(part, act_rd, r_rd, background_polarity="random")
            s2_rd, st2_rd = r2_common.score_answer_batch_sglang(engine, doc, p2_rd.full_masks, all_q, ans_tok_map, kv_cap)
            e2_rd, _ = decode_hierarchical_block(p2_rd, [s2_rd[q] for q in all_q])
            full_rd = assemble_full_n_scores(
                p2_rd, e2_rd, partition=part, l1_group_effects=e1_rd, l1_chunk_polarities=p1_rd.chunk_polarities
            )
            meta_rd = {
                "wall_s": st1_rd["wall_s"] + st2_rd["wall_s"],
                "n_masks": len(p1_rd.full_masks) + len(p2_rd.full_masks),
                "l2_groups": len(act_rd),
                "l2_vars": p2_rd.n_vars,
                "input_tokens": st1_rd["input_tokens"] + st2_rd["input_tokens"],
                "cached_tokens": st1_rd["cached_tokens"] + st2_rd["cached_tokens"],
                "uncached_tokens": st1_rd["uncached_tokens"] + st2_rd["uncached_tokens"],
            }
            w_res["methods"]["hier_coattr_random"] = eval_forensic(full_rd, meta_rd, "Hier-CoAttr (random, Thm.5)")
            matched_budget = meta_rd["n_masks"]

            # Hybrid L1 (locked + random RMS -> random L2)
            m2_common.flush_cache(engine)
            r_lk = random.Random(seed_base + 2)
            p1_lk = build_level1_coarse_plan(part, r_lk, polarity_mode="locked")
            s1_lk, st1_lk = r2_common.score_answer_batch_sglang(engine, doc, p1_lk.full_masks, all_q, ans_tok_map, kv_cap)
            e1_lk, _ = decode_hierarchical_block(p1_lk, [s1_lk[q] for q in all_q])
            act_hyb = select_active_super_chunks(
                e1_lk, top_k_per_q=3, max_total_groups=max_g_l2, extra_level1_scores_per_q=e1_rd
            )
            p2_hyb = build_level2_refined_plan(part, act_hyb, random.Random(seed_base + 3), background_polarity="random")
            s2_hyb, st2_hyb = r2_common.score_answer_batch_sglang(engine, doc, p2_hyb.full_masks, all_q, ans_tok_map, kv_cap)
            e2_hyb, _ = decode_hierarchical_block(p2_hyb, [s2_hyb[q] for q in all_q])
            full_hyb = assemble_full_n_scores(
                p2_hyb, e2_hyb, partition=part, l1_group_effects=e1_lk, l1_chunk_polarities=p1_lk.chunk_polarities
            )
            meta_hyb = {
                "wall_s": st1_lk["wall_s"] + st1_rd["wall_s"] + st2_hyb["wall_s"],
                "n_masks": len(p1_lk.full_masks) + len(p1_rd.full_masks) + len(p2_hyb.full_masks),
                "l2_groups": len(act_hyb),
                "l2_vars": p2_hyb.n_vars,
                "input_tokens": st1_lk["input_tokens"] + st1_rd["input_tokens"] + st2_hyb["input_tokens"],
                "cached_tokens": st1_lk["cached_tokens"] + st1_rd["cached_tokens"] + st2_hyb["cached_tokens"],
                "uncached_tokens": st1_lk["uncached_tokens"] + st1_rd["uncached_tokens"] + st2_hyb["uncached_tokens"],
            }
            w_res["methods"]["hier_coattr_hybrid"] = eval_forensic(full_hyb, meta_hyb, "Hier-CoAttr (hybrid L1)")
            best_scores = full_hyb

        # 3. ContextCite / LASSO (matched budget B)
        m2_common.flush_cache(engine)
        cc_masks = r2_common.draw_contextcite_iid_masks(n, matched_budget, random.Random(seed_base + 77))
        sc_cc_raw, st_cc = r2_common.score_answer_batch_sglang(engine, doc, cc_masks, all_q, ans_tok_map, kv_cap)
        t_f0 = time.perf_counter()
        sc_cc = np.zeros((nq, n))
        for q in all_q:
            sc_cc[q] = r2_common.fit_contextcite_lasso(cc_masks, [sc_cc_raw[q][m] for m in cc_masks], n, alpha=0.005)
        w_res["methods"]["contextcite_lasso"] = eval_forensic(
            sc_cc, {**st_cc, "wall_s": st_cc["wall_s"] + (time.perf_counter() - t_f0), "n_masks": len(cc_masks)}, "ContextCite / LASSO"
        )

        # 4. TracLLM (USENIX Security'25)
        m2_common.flush_cache(engine)
        t0 = time.perf_counter()
        sc_tr = np.zeros((nq, n))
        tr_evals = 0
        for q in all_q:
            def eval_b_trf(ms):
                s, _ = r2_common.score_answer_batch_sglang(engine, doc, ms, [q], ans_tok_map, kv_cap)
                return [s[q][m] for m in ms]
            res_tr = run_tracllm_single_query(eval_b_trf, n, top_k=5, branch_factor=2, use_stc_ensemble=True)
            sc_tr[q] = np.array(res_tr.scores)
            tr_evals += res_tr.total_evals
        w_res["methods"]["tracllm"] = eval_forensic(
            sc_tr, {"wall_s": time.perf_counter() - t0, "total_evals": tr_evals}, "TracLLM (USENIX'25)"
        )

        # 5. BLIP-Pruning (PVLDB'26)
        m2_common.flush_cache(engine)
        t0 = time.perf_counter()
        sc_bl = np.zeros((nq, n))
        blip_san_prompts, blip_evals = [], 0
        for q in all_q:
            def eval_one_blf(m: int) -> float:
                s, _ = r2_common.score_answer_batch_sglang(engine, doc, [m], [q], ans_tok_map, kv_cap)
                return s[q][m]
            b_res = blip_two_stage_pruning(n, eval_one_blf, epsilon=0.05)
            sc_bl[q] = np.array(b_res.scores)
            blip_evals += b_res.n_evals
            sm = full_mask
            for ch in b_res.sufficient_set:
                sm &= ~(1 << int(ch))
            blip_san_prompts.append(doc.prompt(sm, q))
        b_outs = engine.generate(input_ids=blip_san_prompts, sampling_params={"max_new_tokens": 32, "temperature": 0.0})
        blip_wit_rec = [
            1.0 if (rec["benign_answers"][q].lower() in o["text"].lower() and rec["attack_answers"][q].lower() not in o["text"].lower()) else 0.0
            for q, o in enumerate(b_outs)
        ]
        w_res["methods"]["blip_pruning"] = eval_forensic(
            sc_bl,
            {"wall_s": time.perf_counter() - t0, "total_evals": blip_evals, "witness_sanitization_recovery": round(float(np.mean(blip_wit_rec)), 4)},
            "BLIP-Pruning (VLDB'26)",
        )

        # 6. CoAttr-Verify (Dual-Threshold Multi-Witness Sanitization)
        t_cv0 = time.perf_counter()
        sc_fe_for, _ = r2_common.score_answer_batch_sglang(engine, doc, [full_mask, 0], all_q, ans_tok_map, kv_cap)
        cv_san_prompts, cv_n_wit, cv_evals, cv_cert_ok = [], [], [], []
        for q in all_q:
            def eval_one_cvf(m: int) -> float:
                s, _ = r2_common.score_answer_batch_sglang(engine, doc, [m], [q], ans_tok_map, kv_cap)
                return s[q][m]
            f_full_q = sc_fe_for[q][full_mask]
            f_empty_q = sc_fe_for[q][0]
            lift = max(1e-6, f_full_q - f_empty_q)
            cert = extract_multi_witness_provenance(
                best_scores[q],
                eval_one_cvf,
                n,
                epsilon=0.08,
                sufficiency_threshold=f_empty_q + 0.45 * lift,
                clearance_threshold=f_empty_q + 0.15 * lift,
                max_witnesses=5,
                max_witness_size=4,
                full_score=f_full_q,
                empty_score=f_empty_q,
            )
            cv_n_wit.append(len(cert.witnesses))
            cv_evals.append(cert.verification_evals)
            cv_cert_ok.append(1.0 if cert.remediation_certified else 0.0)
            sm = full_mask
            for w in cert.witnesses:
                for ch in w:
                    sm &= ~(1 << int(ch))
            cv_san_prompts.append(doc.prompt(sm, q))
        cv_outs = engine.generate(input_ids=cv_san_prompts, sampling_params={"max_new_tokens": 32, "temperature": 0.0})
        cv_wit_rec = [
            1.0 if (rec["benign_answers"][q].lower() in o["text"].lower() and rec["attack_answers"][q].lower() not in o["text"].lower()) else 0.0
            for q, o in enumerate(cv_outs)
        ]
        w_res["coattr_verify"] = {
            "extra_wall_s": round(time.perf_counter() - t_cv0, 2),
            "mean_witnesses_found": round(float(np.mean(cv_n_wit)), 2),
            "mean_extra_evals_per_q": round(float(np.mean(cv_evals)), 1),
            "remediation_certified_rate": round(float(np.mean(cv_cert_ok)), 4),
            "witness_sanitization_recovery": round(float(np.mean(cv_wit_rec)), 4),
        }
        print(
            f"    {'CoAttr-Verify (Multi-Wit)':28s} +wall={w_res['coattr_verify']['extra_wall_s']:5.2f}s "
            f"wit={w_res['coattr_verify']['mean_witnesses_found']:.2f} "
            f"wit_rec={w_res['coattr_verify']['witness_sanitization_recovery']:.3f} "
            f"(vs BLIP {w_res['methods']['blip_pruning']['witness_sanitization_recovery']:.3f})",
            flush=True,
        )

        results[rec["doc_id"]] = w_res
        _save_atomic(out_dir / f"ckpt_for_{rec['doc_id']}.json", w_res)
    return results


def run_gemma4_extreme_scheduler_ablation(
    engine,
    tokenizer,
    rec: dict,
    kv_cap: int,
    reps: int = 3,
) -> dict:
    """Direct 3-rep cold-cache physical scheduler microbenchmark on a 33K-token Extreme-Long
    document (n=360) under tight Gemma-4 KV capacity comparing SGLang Native LPM,
    BatchLLM-DFS, Alg. 2 v1 (fixed W=32), and Alg. 2 v2 (adaptive W' contraction)."""
    doc = r2_common.FastDocument(tokenizer, rec)
    all_q = list(range(len(rec["questions"])))
    full_mask = (1 << doc.n) - 1
    gen_outs = engine.generate(
        input_ids=[doc.prompt(full_mask, q) for q in all_q],
        sampling_params={"max_new_tokens": 16, "temperature": 0.0},
    )
    ans_tok_map = {}
    for q in all_q:
        toks = _strip_eos_ids(list(gen_outs[q].get("output_ids") or tokenizer.encode(rec["attack_answers"][q], add_special_tokens=False)), tokenizer)
        ans_tok_map[q] = toks if toks else tokenizer.encode(rec["attack_answers"][q], add_special_tokens=False)

    part = partition_super_chunks(doc.n, target_groups=44)
    p1 = build_level1_coarse_plan(part, random.Random(202704), polarity_mode="random")
    masks = p1.full_masks  # 88..96 extreme-long ~16.5K-tok-avg masks!

    layouts = ["sglang_native_lpm", "batchllm_dfs", "capacity_aware_v1", "capacity_aware_v2"]
    summary: dict[str, list[dict]] = {lay: [] for lay in layouts}
    print(f"\n[GEMMA4 33K SCHEDULER ABLATION] {rec['doc_id']} n={doc.n} full_tok={doc.full_tokens} masks={len(masks)} kv_cap={kv_cap}", flush=True)
    for rep in range(reps):
        for lay in layouts:
            m2_common.flush_cache(engine)
            _, st = r2_common.score_answer_batch_sglang(
                engine, doc, masks, all_q, ans_tok_map, kv_cap, layout=lay, width=32, min_width=4
            )
            summary[lay].append(
                {
                    "rep": rep,
                    "wall_s": round(float(st["wall_s"]), 2),
                    "effective_width": int(st["effective_width"]),
                    "tile_masks": int(st["tile_masks"]),
                    "input_tokens": int(st["input_tokens"]),
                    "cached_tokens": int(st["cached_tokens"]),
                    "uncached_tokens": int(st["uncached_tokens"]),
                }
            )
            if rep == 0:
                print(
                    f"    {lay:20s} wall={st['wall_s']:6.2f}s W'={st['effective_width']:3d} "
                    f"tile={st['tile_masks']:3d} uncached={st['uncached_tokens']:8d}",
                    flush=True,
                )
    return {
        "doc_id": rec["doc_id"],
        "n": doc.n,
        "full_tokens": doc.full_tokens,
        "n_masks": len(masks),
        "nq": len(all_q),
        "kv_cap": kv_cap,
        "runs": summary,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--mode", choices=["cuad", "forensics", "both", "gemma4_extreme"], required=True)
    ap.add_argument("--cuad-json", type=Path)
    ap.add_argument("--forensics-json", type=Path)
    ap.add_argument("--extreme-json", type=Path)
    ap.add_argument("--cuad-slice", type=str, default=":")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--context-length", type=int, default=40960)
    ap.add_argument("--mem-fraction-static", type=float, default=0.78)
    ap.add_argument("--temp", type=float, default=4.0)
    ap.add_argument("--max-q-cuad", type=int, default=10)
    ap.add_argument("--run-attntrace", action="store_true")
    ap.add_argument("--port-offset", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    engine = m2_common.build_sglang_engine(
        args.model,
        context_length=args.context_length,
        mem_fraction_static=args.mem_fraction_static,
        port_offset=args.port_offset,
    )
    out_dir = args.out.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "model": args.model,
        "mode": args.mode,
        "context_length": args.context_length,
        "temp": args.temp,
    }
    cuad_items_used: list[dict] = []
    active_qs_by_doc: dict[str, list[int]] = {}
    yes_ids = tokenizer.encode("YES", add_special_tokens=False)
    no_ids = tokenizer.encode("NO", add_special_tokens=False)

    try:
        info = engine.get_server_info()
        kv_cap = int(info.get("max_total_num_tokens") or 0)
        payload["kv_cap"] = kv_cap
        payload["engine_info"] = {k: info.get(k) for k in ("version", "page_size", "max_total_num_tokens", "attention_backend", "schedule_policy", "dtype")}
        print(f"R2 ENGINE READY: model={Path(args.model).name} mode={args.mode} kv_cap={kv_cap} ctx={args.context_length}", flush=True)

        if args.mode in ("cuad", "both"):
            all_cuad = json.loads(args.cuad_json.read_text(encoding="utf-8"))
            s_parts = args.cuad_slice.split(":")
            s0 = int(s_parts[0]) if s_parts[0] else None
            s1 = int(s_parts[1]) if len(s_parts) > 1 and s_parts[1] else None
            cuad_items_used = all_cuad[s0:s1]
            cuad_res, active_qs_by_doc = run_cuad_suite(
                engine, tokenizer, cuad_items_used, kv_cap, args.context_length, args.temp, out_dir,
                max_q_per_doc=args.max_q_cuad,
            )
            payload["cuad"] = cuad_res

        if args.mode in ("forensics", "both"):
            for_items = json.loads(args.forensics_json.read_text(encoding="utf-8"))
            payload["forensics"] = run_forensics_suite(engine, tokenizer, for_items, kv_cap, out_dir)

        if args.mode == "gemma4_extreme":
            ext_pack = json.loads(args.extreme_json.read_text(encoding="utf-8"))
            cuad_ext_res, _ = run_cuad_suite(
                engine, tokenizer, ext_pack["cuad_extreme"], kv_cap, args.context_length, args.temp, out_dir,
                max_q_per_doc=8, run_l2_locked_ablation_count=1,
            )
            payload["cuad_extreme"] = cuad_ext_res
            payload["forensics_extreme"] = run_forensics_suite(engine, tokenizer, ext_pack["forensics_extreme"], kv_cap, out_dir)
            payload["scheduler_33k_ablation"] = run_gemma4_extreme_scheduler_ablation(
                engine, tokenizer, ext_pack["forensics_extreme"][0], kv_cap, reps=3
            )
    finally:
        engine.shutdown()

    # Optional HF AttnTrace on Standard CUAD contracts (runs AFTER SGLang shutdown to reuse HBM!)
    if args.run_attntrace and cuad_items_used:
        std_docs = [r for r in cuad_items_used if r["tier"] == "standard" and r["doc_id"] in active_qs_by_doc]
        if std_docs:
            print(f"\n=== RUNNING HF ATTNTRACE ON {len(std_docs)} STANDARD CUAD CONTRACTS ===", flush=True)
            at_raw = r2_common.run_attntrace_cuad_hf(args.model, std_docs, active_qs_by_doc)
            # Re-open lightweight SGLang engine ONLY to score top-3/top-5 decision flips?
            # Even cleaner: compute exact Clause Recall, MRR, and Multi-Witness Coverage offline!
            at_eval: dict[str, dict] = {}
            for rec in std_docs:
                did = rec["doc_id"]
                if did not in at_raw:
                    continue
                aqs = active_qs_by_doc[did]
                smat = np.asarray(at_raw[did]["scores"], dtype=np.float64)
                rec_ks, mrrs, mw_covs = [], [], []
                for qi, q in enumerate(aqs):
                    order = np.argsort(-smat[qi])
                    gold = set(rec["gold_supports"][q])
                    k = max(1, len(gold))
                    rec_ks.append(len(set(int(x) for x in order[: max(k, 3)]) & gold) / k)
                    rank1 = next((r + 1 for r, ch in enumerate(order) if int(ch) in gold), rec["n_chunks"])
                    mrrs.append(1.0 / rank1)
                    wgs = rec["gold_witness_groups"][q]
                    if len(wgs) >= 2:
                        t5 = set(int(x) for x in order[:5])
                        mw_covs.append(sum(1 for wg in wgs if set(wg) & t5) / len(wgs))
                at_eval[did] = {
                    "wall_s": round(float(at_raw[did]["wall_s"]), 2),
                    "clause_rec_at_maxk3": round(float(np.mean(rec_ks)), 4),
                    "clause_mrr": round(float(np.mean(mrrs)), 4),
                    "multi_witness_cov_top5": round(float(np.mean(mw_covs)), 4) if mw_covs else None,
                }
                print(f"    AttnTrace {did:26s} {at_eval[did]}", flush=True)
            payload["attntrace_cuad_std"] = at_eval

    _save_atomic(args.out, payload)
    print(f"\nSAVED FULL STEP-2 RESULT TO {args.out}", flush=True)


if __name__ == "__main__":
    main()
