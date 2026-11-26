"""M2 Job-B (GPU 1, Qwen3-32B SGLang):
Part B1: 5-Rep High-Precision Scheduler & Design Benchmark (direct Alg. 2 v2 on
         18Q, BatchLLM-DFS, Prefix Ablation-1, Global Trie, Alg. 2 v1, Paley-Hadamard,
         and single-question n_q=1 cliff).
Part B2: 32B Attribution Accuracy Validation (3 diverse workloads, 16 Qs: 8-block
         mixed reference truth [4 CUBE + 4 antithetic] + 3 seeds across CUBE,
         Half, Four, CUBE R=2, Cyclic Half R=2, Paley R=1, i.i.d. 2M/4M, LOO, BLIP).
Part B3: Continuous Multi-Tenant C_KV Quota Sweep (99K -> 160K -> 273K -> 512K)
         comparing Global Trie, Prefix 2-wave, BatchLLM-DFS, Alg. 2 v1, and Alg. 2 v2.
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from coattr_v2.baselines_v2 import blip_two_stage_pruning
from coattr_v2.designs_v2 import (
    cyclic_balanced_coset_tags,
    draw_dyadic_block,
    draw_kronecker_paley_block,
    dyadic_half_budget,
    select_best_paley_configs,
)
from m2_common import (
    Document,
    build_sglang_engine,
    decode_balanced_block_m2,
    execute_scheduled_sglang,
    flush_cache,
    load_all_m2_inputs,
    prepare_natural_answers_sglang,
    save_json,
    score_batch_sglang,
    stable_seed,
)

TIMING_WORKLOADS: list[tuple[str, int]] = [
    ("QuALITY-20027", 2),
    ("QASPER-1810.05241", 2),
    ("QASPER-1810.05241", 6),
    ("QuALITY-22967", 2),
    ("QuALITY-22967", 18),  # Direct Algorithm 2 v2 — eliminates Table 2 2-wv footnote!
    ("MuSiQue-SingleQ65", 1),  # Pure n_q=1 online RAG cliff on 32B!
]

ACCURACY_WORKLOADS: list[tuple[str, int]] = [
    ("QuALITY-20027", 2),
    ("QASPER-1810.05241", 6),
    ("MuSiQue-Pack88D", 8),
]

SWEEP_QUOTAS: list[tuple[int, float]] = [
    (99000, 0.72),
    (160000, 0.72),
    (273472, 0.72),
    (512000, 0.90),
]


def _draw_antithetic(n: int, size: int, rng: random.Random) -> list[int]:
    rows = []
    full = (1 << n) - 1
    for _ in range(size // 2):
        m = rng.getrandbits(n)
        rows.extend((m, full ^ m))
    return rows


def _msr_decode(masks: list[int], scores: dict[tuple[int, int], float], n: int, qs: list[int]) -> list[list[float]]:
    x01 = np.array([[1.0 if ((m >> i) & 1) else 0.0 for i in range(n)] for m in masks], dtype=np.float64)
    c1 = x01.sum(axis=0)
    c0 = len(masks) - c1
    valid = (c1 > 0) & (c0 > 0)
    effs = []
    for q in qs:
        y = np.array([scores[(m, q)] for m in masks], dtype=np.float64)
        out = np.zeros(n, dtype=np.float64)
        out[valid] = (x01.T @ y)[valid] / c1[valid] - ((1.0 - x01).T @ y)[valid] / c0[valid]
        effs.append(out.tolist())
    return effs


def main() -> None:
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-32B")
    ap.add_argument("--timing-reps", type=int, default=5)
    ap.add_argument("--acc-seeds", type=int, default=3)
    ap.add_argument("--sweep-reps", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    out = Path(os.environ["P18_OUTPUT"])
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    raw_by_ident = load_all_m2_inputs()

    # =========================================================================
    # PARTS B1 & B2: Default 32B Engine (mem_fraction_static=0.72, ~273K KV)
    # =========================================================================
    server = build_sglang_engine(args.model, max_running_requests=128, mem_fraction_static=0.72, port_offset=2)
    docs: dict[str, Document] = {}
    answers_map: dict[str, dict[int, dict]] = {}
    timing_summary: list[dict] = []
    acc_summary: list[dict] = []

    try:
        info = server.get_server_info()
        kv_cap = int(info.get("max_total_num_tokens") or 0)
        page_size = int(info.get("page_size") or 64)
        save_json(out / "engine-default.json", {k: info.get(k) for k in ("version", "page_size", "max_total_num_tokens", "attention_backend", "schedule_policy", "dtype")})
        print(f"JOB-B ENGINE {Path(args.model).name}: kv_cap={kv_cap} page={page_size} backend={info.get('attention_backend')}", flush=True)

        # Prepare natural answers for all needed docs
        max_nq: dict[str, int] = defaultdict(int)
        for ident, nq in TIMING_WORKLOADS + ACCURACY_WORKLOADS:
            max_nq[ident] = max(max_nq[ident], nq)
        for ident, nq_m in max_nq.items():
            doc = Document(tokenizer, raw_by_ident[ident])
            docs[ident] = doc
            qs_all = list(range(min(nq_m, len(doc.sides))))
            flush_cache(server)
            ans_list = prepare_natural_answers_sglang(server, doc, qs_all, max_new_tokens=args.max_new_tokens)
            answers_map[ident] = {a["q"]: a for a in ans_list}
            save_json(out / f"answers-32b-{ident}.json", ans_list)
            print(f"32B ANSWERS {ident} (n={doc.n}, nq={len(qs_all)}): lens={[a['token_len'] for a in ans_list]}", flush=True)

        # Warmup
        w_doc = docs["QuALITY-20027"]
        score_batch_sglang(server, [w_doc.prompts((1 << w_doc.n) - 1, [0])[0]], [answers_map["QuALITY-20027"][0]["tokens"]])
        flush_cache(server)

        # --- PART B1: 5-Rep High-Precision Scheduler + Design Benchmark ---
        print("\n=== PART B1: 32B 5-REP SCHEDULER & DESIGN BENCHMARK ===", flush=True)
        layouts = ["sglang_native_lpm", "batchllm_dfs", "prefix", "merged_mask_trie", "capacity_aware_v1", "capacity_aware_v2"]
        for ident, nq in TIMING_WORKLOADS:
            doc = docs[ident]
            qs = list(range(min(nq, len(doc.sides))))
            ans_by_q = answers_map[ident]
            m_half = dyadic_half_budget(doc.n)
            paley_cfgs = select_best_paley_configs(doc.n, groups=(2, 4), mod8_eq4_only=True, strictly_smaller_than_dyadic=True)
            best_paley = paley_cfgs[0] if paley_cfgs else None
            design_list = ["cube", "four"] + (["paley_r1"] if best_paley is not None else [])
            # Keep 18Q crisp: run 5 reps on cube/four/paley with all 6 layouts (for 18Q run 3 reps to stay fast)
            n_reps = 3 if nq >= 18 else args.timing_reps

            for rep in range(n_reps):
                masks_by_des: dict[str, tuple[list[int], dict]] = {}
                for des in design_list:
                    rng = random.Random(stable_seed("m2-32b-b1", ident, rep, des))
                    if des == "cube":
                        blk = draw_dyadic_block(doc.n, m_half, rng, g=1)
                        masks_by_des[des] = (list(blk.rows), {"budget": 2 * m_half})
                    elif des == "four":
                        blk = draw_dyadic_block(doc.n, m_half, rng, g=4)
                        masks_by_des[des] = (list(blk.rows), {"budget": 2 * m_half})
                    elif des == "paley_r1":
                        assert best_paley is not None
                        blk = draw_kronecker_paley_block(doc.n, int(best_paley["g"]), int(best_paley["L"]), rng)
                        masks_by_des[des] = (list(blk.rows), {"budget": int(best_paley["budget_2m"]), "paley_g": best_paley["g"], "paley_L": best_paley["L"]})

                # Run full layout comparison on `four` (and `cube` on `capacity_aware_v2`, `prefix`, `batchllm_dfs`, `sglang_native_lpm`; `paley_r1` on `capacity_aware_v2`)
                tasks = []
                for des in design_list:
                    if des == "four":
                        lay_sub = layouts
                    elif des == "cube":
                        lay_sub = ["sglang_native_lpm", "batchllm_dfs", "prefix", "capacity_aware_v2"]
                    else:
                        lay_sub = ["capacity_aware_v2", "batchllm_dfs"]
                    for lay in lay_sub:
                        tasks.append((des, lay))
                random.Random(stable_seed("m2-32b-b1-ord", ident, nq, rep)).shuffle(tasks)

                for des, lay in tasks:
                    rows, dmeta = masks_by_des[des]
                    flush_cache(server)
                    t0 = time.perf_counter()
                    scores, meas = execute_scheduled_sglang(
                        server, doc, rows, qs, ans_by_q, lay,
                        width=32, kv_token_capacity=kv_cap, page_size=page_size,
                    )
                    _, vacs, top1s, _ = decode_balanced_block_m2(rows, scores, doc.n, qs)
                    tot_s = time.perf_counter() - t0
                    rec = {
                        "part": "B1_timing",
                        "model": Path(args.model).name,
                        "kv_token_capacity": kv_cap,
                        "doc": ident,
                        "n": doc.n,
                        "nq": len(qs),
                        "rep": rep,
                        "design": des,
                        "layout": lay,
                        **dmeta,
                        "effective_width": meas["effective_width"],
                        "tile_masks": meas["tile_masks"],
                        "total_seconds": tot_s,
                        "service_seconds": meas["service_seconds"],
                        "requests": meas["requests"],
                        "batches": meas["batches"],
                        "input_tokens": meas["input_tokens"],
                        "cached_tokens": meas["cached_tokens"],
                        "uncached_tokens": meas["uncached_tokens"],
                        "top1": top1s,
                    }
                    timing_summary.append(rec)
                    save_json(out / f"b1-{ident}-nq{len(qs)}-r{rep}-{des}-{lay}.json.gz", rec)
                    if rep == 0:
                        print(
                            f"  B1 {ident:18s} nq={len(qs):2d} {des:8s} {lay:20s} "
                            f"(W'={meas['effective_width']:2d} tile={meas['tile_masks']:3d}) | tot={tot_s:6.2f}s unc={meas['uncached_tokens']:7d}",
                            flush=True,
                        )
            save_json(out / "b1-timing-summary.json", timing_summary)

        # --- PART B2: 32B Attribution Accuracy Validation (3 Workloads / 16 Qs) ---
        print("\n=== PART B2: 32B ATTRIBUTION ACCURACY VALIDATION ===", flush=True)
        for ident, nq in ACCURACY_WORKLOADS:
            doc = docs[ident]
            qs = list(range(min(nq, len(doc.sides))))
            ans_by_q = answers_map[ident]
            m_half = dyadic_half_budget(doc.n)
            budget_2m = 2 * m_half
            gold_support = raw_by_ident[ident].get("supporting_segments")
            paley_cfgs = select_best_paley_configs(doc.n, groups=(2, 4), mod8_eq4_only=True, strictly_smaller_than_dyadic=True)
            best_paley = paley_cfgs[0] if paley_cfgs else None

            # 8-block mixed reference truth (4 CUBE + 4 antithetic i.i.d.)
            ref_blocks = []
            for rb in range(4):
                flush_cache(server)
                blk = draw_dyadic_block(doc.n, m_half, random.Random(stable_seed("m2-32b-ref-c", ident, rb)), g=1)
                sc_r, _ = execute_scheduled_sglang(server, doc, blk.rows, qs, ans_by_q, "capacity_aware_v2", kv_token_capacity=kv_cap, page_size=page_size)
                er, _, _, _ = decode_balanced_block_m2(blk.rows, sc_r, doc.n, qs)
                ref_blocks.append(er)
            for rb in range(4):
                flush_cache(server)
                rows_a = _draw_antithetic(doc.n, budget_2m, random.Random(stable_seed("m2-32b-ref-a", ident, rb)))
                sc_r, _ = execute_scheduled_sglang(server, doc, rows_a, qs, ans_by_q, "capacity_aware_v2", kv_token_capacity=kv_cap, page_size=page_size)
                ref_blocks.append(_msr_decode(rows_a, sc_r, doc.n, qs))
            truth_B = np.mean(np.asarray(ref_blocks, dtype=np.float64), axis=0)
            truth_top1 = [int(np.argmax(truth_B[q])) for q in range(len(qs))]
            save_json(out / f"b2-ref-{ident}.json.gz", {"doc": ident, "truth_B": truth_B.tolist(), "truth_top1": truth_top1, "ref_blocks": ref_blocks})
            print(f"  32B REF {ident} (8-blk mixed): true_top1={truth_top1}", flush=True)

            for sd in range(args.acc_seeds):
                methods = ["cube", "half", "four", "cube_r2", "cyclic_half_r2", "iid_2M", "iid_4M", "loo"]
                if best_paley is not None:
                    methods.append("paley_r1")
                for mname in methods:
                    flush_cache(server)
                    t0 = time.perf_counter()
                    rng = random.Random(stable_seed("m2-32b-acc", ident, sd, mname))
                    if mname in ("cube", "half", "four"):
                        g_cnt = 1 if mname == "cube" else (2 if mname == "half" else 4)
                        blk = draw_dyadic_block(doc.n, m_half, rng, g=g_cnt)
                        sc, meas = execute_scheduled_sglang(server, doc, blk.rows, qs, ans_by_q, "capacity_aware_v2", kv_token_capacity=kv_cap, page_size=page_size)
                        effs, _, _, _ = decode_balanced_block_m2(blk.rows, sc, doc.n, qs)
                    elif mname in ("cube_r2", "cyclic_half_r2"):
                        g_cnt = 1 if mname == "cube_r2" else 2
                        tgs = [None, None] if g_cnt == 1 else cyclic_balanced_coset_tags(doc.n, 2, g=2)
                        be = []
                        tot_u = 0
                        for bi in range(2):
                            blk = draw_dyadic_block(doc.n, m_half, random.Random(stable_seed("m2-32b-r2", ident, sd, mname, bi)), tags=tgs[bi], g=g_cnt)
                            sc_b, meas_b = execute_scheduled_sglang(server, doc, blk.rows, qs, ans_by_q, "capacity_aware_v2", kv_token_capacity=kv_cap, page_size=page_size)
                            eb, _, _, _ = decode_balanced_block_m2(blk.rows, sc_b, doc.n, qs)
                            be.append(eb)
                            tot_u += meas_b["uncached_tokens"]
                        effs = np.mean(np.asarray(be, dtype=np.float64), axis=0).tolist()
                        meas = {"uncached_tokens": tot_u}
                    elif mname == "paley_r1":
                        assert best_paley is not None
                        blk = draw_kronecker_paley_block(doc.n, int(best_paley["g"]), int(best_paley["L"]), rng)
                        sc, meas = execute_scheduled_sglang(server, doc, blk.rows, qs, ans_by_q, "capacity_aware_v2", kv_token_capacity=kv_cap, page_size=page_size)
                        effs, _, _, _ = decode_balanced_block_m2(blk.rows, sc, doc.n, qs)
                    elif mname in ("iid_2M", "iid_4M"):
                        sz = budget_2m if mname == "iid_2M" else 2 * budget_2m
                        masks = [rng.getrandbits(doc.n) for _ in range(sz)]
                        sc, meas = execute_scheduled_sglang(server, doc, masks, qs, ans_by_q, "capacity_aware_v2", kv_token_capacity=kv_cap, page_size=page_size)
                        effs = _msr_decode(masks, sc, doc.n, qs)
                        raw_save = {"masks": [hex(m) for m in masks], "scores": {hex(m): [sc[(m, q)] for q in qs] for m in dict.fromkeys(masks)}}
                    elif mname == "loo":
                        fm = (1 << doc.n) - 1
                        masks = [fm, *[fm ^ (1 << i) for i in range(doc.n)]]
                        sc, meas = execute_scheduled_sglang(server, doc, masks, qs, ans_by_q, "capacity_aware_v2", kv_token_capacity=kv_cap, page_size=page_size)
                        effs = [[sc[(fm, q)] - sc[(fm ^ (1 << i), q)] for i in range(doc.n)] for q in qs]
                        raw_save = {}
                    else:
                        raise ValueError(mname)
                    if mname not in ("iid_2M", "iid_4M", "loo"):
                        raw_save = {}
                    tot_s = time.perf_counter() - t0
                    top1s = [int(np.argmax(effs[q])) for q in range(len(qs))]
                    hits = [int(top1s[q] == truth_top1[q]) for q in range(len(qs))]
                    regrets = [float(truth_B[q, truth_top1[q]] - truth_B[q, top1s[q]]) for q in range(len(qs))]
                    mses = [float(np.mean((np.asarray(effs[q]) - truth_B[q]) ** 2)) for q in range(len(qs))]
                    g1 = gr = None
                    if gold_support:
                        g1 = round(float(np.mean([int(top1s[q] in set(gold_support[q])) for q in range(len(qs))])), 4)
                        gr = round(float(np.mean([len(set(sorted(range(doc.n), key=lambda i: (-effs[q][i], i))[: len(gold_support[q])]) & set(gold_support[q])) / len(gold_support[q]) for q in range(len(qs))])), 4)
                    rec = {
                        "part": "B2_accuracy_32b",
                        "doc": ident,
                        "n": doc.n,
                        "nq": len(qs),
                        "seed": sd,
                        "method": mname,
                        "total_seconds": tot_s,
                        "uncached_tokens": meas["uncached_tokens"],
                        "top1_vs_ref": round(float(np.mean(hits)), 4),
                        "mean_regret_vs_ref": round(float(np.mean(regrets)), 6),
                        "mean_mse_vs_ref": float(f"{np.mean(mses):.4e}"),
                        "gold_top1_hit": g1,
                        "gold_recall_at_k": gr,
                    }
                    acc_summary.append(rec)
                    save_json(out / f"b2-{ident}-s{sd}-{mname}.json.gz", {**rec, **raw_save, "B": effs})
                    if sd == 0:
                        print(f"  B2 {ident:18s} s0 {mname:15s} | tot={tot_s:6.2f}s top1={rec['top1_vs_ref']:.3f} mse={rec['mean_mse_vs_ref']:.2e}", flush=True)
            save_json(out / "b2-accuracy-summary.json", acc_summary)
    finally:
        server.shutdown()
        gc.collect()
        time.sleep(3)

    # =========================================================================
    # PART B3: Continuous Multi-Tenant C_KV Quota Sweep (99K -> 160K -> 273K -> 512K)
    # =========================================================================
    print("\n=== PART B3: CONTINUOUS MULTI-TENANT C_KV QUOTA SWEEP (Qwen3-32B) ===", flush=True)
    sweep_summary: list[dict] = []
    sweep_docs = [("QASPER-1810.05241", 2), ("QuALITY-22967", 2)]
    sweep_layouts = ["merged_mask_trie", "prefix", "batchllm_dfs", "capacity_aware_v1", "capacity_aware_v2"]

    for q_idx, (target_kv, mem_frac) in enumerate(SWEEP_QUOTAS):
        srv = build_sglang_engine(
            args.model,
            max_running_requests=128,
            mem_fraction_static=mem_frac,
            max_total_tokens=target_kv,
            port_offset=20 + q_idx,
        )
        try:
            info = srv.get_server_info()
            actual_kv = int(info.get("max_total_num_tokens") or target_kv)
            page_size = int(info.get("page_size") or 64)
            print(f"\n--- SWEEP target_kv={target_kv} -> actual_kv={actual_kv} (page={page_size}) ---", flush=True)
            # Warmup
            w_doc = docs["QASPER-1810.05241"]
            score_batch_sglang(srv, [w_doc.prompts((1 << w_doc.n) - 1, [0])[0]], [answers_map["QASPER-1810.05241"][0]["tokens"]])
            flush_cache(srv)

            for ident, nq in sweep_docs:
                doc = docs[ident]
                qs = list(range(nq))
                ans_by_q = answers_map[ident]
                m_half = dyadic_half_budget(doc.n)
                for rep in range(args.sweep_reps):
                    blk = draw_dyadic_block(doc.n, m_half, random.Random(stable_seed("m2-sweep", ident, rep)), g=4)
                    lays = list(sweep_layouts)
                    random.Random(stable_seed("m2-sweep-ord", target_kv, ident, rep)).shuffle(lays)
                    for lay in lays:
                        flush_cache(srv)
                        t0 = time.perf_counter()
                        _, meas = execute_scheduled_sglang(
                            srv, doc, blk.rows, qs, ans_by_q, lay,
                            width=32, kv_token_capacity=actual_kv, page_size=page_size,
                        )
                        tot_s = time.perf_counter() - t0
                        rec = {
                            "part": "B3_ckv_sweep",
                            "target_kv": target_kv,
                            "actual_kv": actual_kv,
                            "doc": ident,
                            "n": doc.n,
                            "nq": nq,
                            "rep": rep,
                            "design": "four",
                            "layout": lay,
                            "effective_width": meas["effective_width"],
                            "tile_masks": meas["tile_masks"],
                            "total_seconds": tot_s,
                            "uncached_tokens": meas["uncached_tokens"],
                            "cached_tokens": meas["cached_tokens"],
                        }
                        sweep_summary.append(rec)
                        if rep == 0:
                            print(
                                f"  SWEEP kv={actual_kv:6d} {ident:18s} {lay:20s} "
                                f"(W'={meas['effective_width']:2d} tile={meas['tile_masks']:3d}) | tot={tot_s:6.2f}s unc={meas['uncached_tokens']:7d}",
                                flush=True,
                            )
            save_json(out / "b3-ckv-sweep-summary.json", sweep_summary)
        finally:
            srv.shutdown()
            gc.collect()
            time.sleep(3)

    print(f"JOB-B COMPLETE: B1={len(timing_summary)}, B2={len(acc_summary)}, B3={len(sweep_summary)}", flush=True)


if __name__ == "__main__":
    main()
