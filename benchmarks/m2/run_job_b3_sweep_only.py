"""Standalone 32B Continuous Multi-Tenant C_KV Quota Sweep (Backfill Runner).

Reuses the exact saved 32B natural answers from Job 209 (``answers-32b-*.json``)
and runs specified ``--quotas`` (e.g., ``273472:0.72,512000:0.90``) across
``QASPER-1810.05241:2`` and ``QuALITY-22967:2`` (3 reps, 5 layouts) so all 4
quotas (99K, 160K, 273K, 512K) complete in parallel with zero Slurm timeout risk.
"""

from __future__ import annotations

import argparse
import gc
import os
import random
import time
from pathlib import Path

from transformers import AutoTokenizer

from coattr_v2.designs_v2 import draw_dyadic_block, dyadic_half_budget
from m2_common import (
    Document,
    build_sglang_engine,
    execute_scheduled_sglang,
    flush_cache,
    load_all_m2_inputs,
    load_json,
    save_json,
    score_batch_sglang,
    stable_seed,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-32B")
    ap.add_argument("--answers-dir", type=Path, required=True)
    ap.add_argument("--quotas", type=str, default="273472:0.72,512000:0.90")
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    out = Path(os.environ["P18_OUTPUT"])
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    raw_by_ident = load_all_m2_inputs()

    sweep_docs = [("QASPER-1810.05241", 2), ("QuALITY-22967", 2)]
    sweep_layouts = ["merged_mask_trie", "prefix", "batchllm_dfs", "capacity_aware_v1", "capacity_aware_v2"]
    docs = {ident: Document(tokenizer, raw_by_ident[ident]) for ident, _ in sweep_docs}
    answers_map = {
        ident: {int(a["q"]): a for a in load_json(args.answers_dir / f"answers-32b-{ident}.json")}  # type: ignore[union-attr]
        for ident, _ in sweep_docs
    }

    quotas = [(int(p.split(":")[0]), float(p.split(":")[1])) for p in args.quotas.split(",") if p.strip()]
    sweep_summary: list[dict] = []

    for q_idx, (target_kv, mem_frac) in enumerate(quotas):
        srv = build_sglang_engine(
            args.model,
            max_running_requests=128,
            mem_fraction_static=mem_frac,
            max_total_tokens=target_kv,
            port_offset=50 + q_idx,
        )
        try:
            info = srv.get_server_info()
            actual_kv = int(info.get("max_total_num_tokens") or target_kv)
            page_size = int(info.get("page_size") or 64)
            print(f"\n--- SWEEP target_kv={target_kv} -> actual_kv={actual_kv} (page={page_size}) ---", flush=True)
            w_doc = docs["QASPER-1810.05241"]
            score_batch_sglang(srv, [w_doc.prompts((1 << w_doc.n) - 1, [0])[0]], [answers_map["QASPER-1810.05241"][0]["tokens"]])
            flush_cache(srv)

            for ident, nq in sweep_docs:
                doc = docs[ident]
                qs = list(range(nq))
                ans_by_q = answers_map[ident]
                m_half = dyadic_half_budget(doc.n)
                for rep in range(args.reps):
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

    print(f"B3-SWEEP COMPLETE: {len(sweep_summary)} records", flush=True)


if __name__ == "__main__":
    main()
