"""M2 Job-C (GPU 2, gemma-4-31B-it SGLang):
Conquering Extreme-Tight 99K HBM KV Capacity with Upgraded Algorithm 2
(Explicit Delta_wave + Adaptive Inner Wave-Width Contraction W') across ALL 6
workloads including 6Q and 18Q, 5 reps each, plus width ablation W' in {8, 16, 32}
and Kronecker Paley-Hadamard cliff comparison.
"""

from __future__ import annotations

import argparse
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from coattr_v2.designs_v2 import (
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

GEMMA_WORKLOADS: list[tuple[str, int]] = [
    ("QASPER-1912.06262", 2),
    ("QuALITY-20027", 2),
    ("QASPER-1810.05241", 2),
    ("QASPER-1810.05241", 6),
    ("QuALITY-22967", 2),
    ("QuALITY-22967", 18),
]


def main() -> None:
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="google/gemma-4-31B-it")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    out = Path(os.environ["P18_OUTPUT"])
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    raw_by_ident = load_all_m2_inputs()

    server = build_sglang_engine(args.model, max_running_requests=128, mem_fraction_static=0.72, port_offset=3)
    summary: list[dict] = []
    ablation_summary: list[dict] = []

    try:
        info = server.get_server_info()
        kv_cap = int(info.get("max_total_num_tokens") or 0)
        page_size = int(info.get("page_size") or 64)
        save_json(out / "engine.json", {k: info.get(k) for k in ("version", "page_size", "max_total_num_tokens", "attention_backend", "schedule_policy", "dtype")})
        print(f"JOB-C ENGINE {Path(args.model).name}: kv_cap={kv_cap} page={page_size} backend={info.get('attention_backend')}", flush=True)

        # Prepare natural answers
        max_nq: dict[str, int] = defaultdict(int)
        for ident, nq in GEMMA_WORKLOADS:
            max_nq[ident] = max(max_nq[ident], nq)
        docs: dict[str, Document] = {}
        answers_map: dict[str, dict[int, dict]] = {}
        for ident, nq_m in max_nq.items():
            doc = Document(tokenizer, raw_by_ident[ident])
            docs[ident] = doc
            qs_all = list(range(min(nq_m, len(doc.sides))))
            flush_cache(server)
            ans_list = prepare_natural_answers_sglang(server, doc, qs_all, max_new_tokens=args.max_new_tokens)
            answers_map[ident] = {a["q"]: a for a in ans_list}
            save_json(out / f"answers-gemma4-{ident}.json", ans_list)
            print(f"GEMMA4 ANSWERS {ident} (n={doc.n}, nq={len(qs_all)}): lens={[a['token_len'] for a in ans_list]}", flush=True)

        # Warmup
        w_doc = docs["QASPER-1912.06262"]
        score_batch_sglang(server, [w_doc.prompts((1 << w_doc.n) - 1, [0])[0]], [answers_map["QASPER-1912.06262"][0]["tokens"]])
        flush_cache(server)

        # 1. Explicit W' width ablation on QASPER-1810:2 & QuALITY-22967:2 (W' in {8, 16, 32})
        print("\n=== GEMMA-4 WIDTH ABLATION (W' in {8, 16, 32}) ===", flush=True)
        for ident in ("QASPER-1810.05241", "QuALITY-22967"):
            doc = docs[ident]
            qs = [0, 1]
            m_half = dyadic_half_budget(doc.n)
            for rep in range(3):
                blk = draw_dyadic_block(doc.n, m_half, random.Random(stable_seed("m2-gemma-w", ident, rep)), g=4)
                for w_test in (8, 16, 32):
                    flush_cache(server)
                    t0 = time.perf_counter()
                    _, meas = execute_scheduled_sglang(
                        server, doc, blk.rows, qs, answers_map[ident], "capacity_aware_v2",
                        width=w_test, kv_token_capacity=kv_cap, page_size=page_size, min_width=w_test,
                    )
                    tot_s = time.perf_counter() - t0
                    rec = {
                        "doc": ident, "rep": rep, "forced_width": w_test,
                        "tile_masks": meas["tile_masks"], "total_seconds": tot_s,
                        "uncached_tokens": meas["uncached_tokens"], "cached_tokens": meas["cached_tokens"],
                        "batches": meas["batches"],
                    }
                    ablation_summary.append(rec)
                    if rep == 0:
                        print(f"  W-ABL {ident:18s} W={w_test:2d} tile={meas['tile_masks']:3d} | tot={tot_s:6.2f}s unc={meas['uncached_tokens']:7d}", flush=True)
        save_json(out / "gemma4-width-ablation.json", ablation_summary)

        # 2. Full 6-Workload Matrix (including 6Q and 18Q, 5 reps on 2Q/6Q and 3 reps on 18Q)
        print("\n=== GEMMA-4 FULL 6-WORKLOAD MATRIX (INCL. 6Q & 18Q) ===", flush=True)
        layouts_four = ["sglang_native_lpm", "batchllm_dfs", "prefix", "capacity_aware_v1", "capacity_aware_v2"]
        layouts_cube = ["sglang_native_lpm", "batchllm_dfs", "prefix", "capacity_aware_v2"]
        for ident, nq in GEMMA_WORKLOADS:
            doc = docs[ident]
            qs = list(range(min(nq, len(doc.sides))))
            ans_by_q = answers_map[ident]
            m_half = dyadic_half_budget(doc.n)
            paley_cfgs = select_best_paley_configs(doc.n, groups=(2, 4), mod8_eq4_only=True, strictly_smaller_than_dyadic=True)
            best_paley = paley_cfgs[0] if paley_cfgs else None
            n_reps = 3 if nq >= 18 else args.reps

            for rep in range(n_reps):
                tasks = []
                masks_by_des = {}
                for des in (["cube", "four"] + (["paley_r1"] if best_paley is not None else [])):
                    rng = random.Random(stable_seed("m2-gemma-main", ident, rep, des))
                    if des == "cube":
                        blk = draw_dyadic_block(doc.n, m_half, rng, g=1)
                        masks_by_des[des] = (list(blk.rows), {"budget": 2 * m_half})
                        for lay in layouts_cube:
                            tasks.append((des, lay))
                    elif des == "four":
                        blk = draw_dyadic_block(doc.n, m_half, rng, g=4)
                        masks_by_des[des] = (list(blk.rows), {"budget": 2 * m_half})
                        for lay in layouts_four:
                            tasks.append((des, lay))
                    else:
                        assert best_paley is not None
                        blk = draw_kronecker_paley_block(doc.n, int(best_paley["g"]), int(best_paley["L"]), rng)
                        masks_by_des[des] = (list(blk.rows), {"budget": int(best_paley["budget_2m"]), "paley_g": best_paley["g"], "paley_L": best_paley["L"]})
                        for lay in ("capacity_aware_v2", "batchllm_dfs"):
                            tasks.append((des, lay))

                random.Random(stable_seed("m2-gemma-ord", ident, nq, rep)).shuffle(tasks)
                for des, lay in tasks:
                    rows, dmeta = masks_by_des[des]
                    flush_cache(server)
                    t0 = time.perf_counter()
                    scores, meas = execute_scheduled_sglang(
                        server, doc, rows, qs, ans_by_q, lay,
                        width=32, kv_token_capacity=kv_cap, page_size=page_size,
                    )
                    _, _, top1s, _ = decode_balanced_block_m2(rows, scores, doc.n, qs)
                    tot_s = time.perf_counter() - t0
                    rec = {
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
                        "width_contracted": meas["width_contracted"],
                        "total_seconds": tot_s,
                        "service_seconds": meas["service_seconds"],
                        "requests": meas["requests"],
                        "batches": meas["batches"],
                        "input_tokens": meas["input_tokens"],
                        "cached_tokens": meas["cached_tokens"],
                        "uncached_tokens": meas["uncached_tokens"],
                        "top1": top1s,
                    }
                    summary.append(rec)
                    save_json(out / f"gemma4-{ident}-nq{len(qs)}-r{rep}-{des}-{lay}.json.gz", rec)
                    if rep == 0:
                        print(
                            f"  GEMMA4 {ident:18s} nq={len(qs):2d} {des:8s} {lay:20s} "
                            f"(W'={meas['effective_width']:2d} tile={meas['tile_masks']:3d}) | tot={tot_s:6.2f}s unc={meas['uncached_tokens']:7d}",
                            flush=True,
                        )
            save_json(out / "gemma4-summary.json", summary)
    finally:
        server.shutdown()

    print(f"JOB-C COMPLETE: {len(summary)} main records, {len(ablation_summary)} width-ablation records.", flush=True)


if __name__ == "__main__":
    main()
