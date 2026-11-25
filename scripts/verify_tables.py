"""Verify and print CoAttr paper Tables 1-3 directly from archived data/."""
from __future__ import annotations
import csv, json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

def main() -> None:
    core = json.loads((DATA / "core-tables.json").read_text(encoding="utf-8"))
    with (DATA / "table-aggregate-57q.csv").open(encoding="utf-8") as f:
        agg = list(csv.DictReader(f))
    with (DATA / "table3a-ref-convergence.csv").open(encoding="utf-8") as f:
        ref = list(csv.DictReader(f))

    print("=== Table 1: 11 Workloads & 8-Block Split-Half Convergence ===")
    for r in ref:
        print(f"  {r['doc']:22s} n={int(r['n']):2d} nq={int(r['nq']):2d} split_half={100*float(r['split_half_top1_rate']):5.1f}% med_gap={float(r['median_top1_top2_gap']):.4f}")

    print("\n=== Table 2: Cross-Model Scaling (Same-Job Ratios & Joint Speedup) ===")
    for r in core["table2_cross_model_scheduler"]:
        if r["fair_four_over_cube"] is not None and r["joint_speedup_vs_cube_lpm"] is not None:
            print(f"  {r['track']:30s} {r['doc']:18s} {r['nq']:2d}Q | LPM={r['cube_lpm_s']:6.2f}s CUBE={r['cube_s']:5.2f}s CoAttr={r['four_s']:5.2f}s | ratio={r['fair_four_over_cube']:.3f}x joint={r['joint_speedup_vs_cube_lpm']:.2f}x")

    print("\n=== Table 3: 57-Question Multi-Dataset SOTA Aggregate ===")
    for r in agg:
        print(f"  {r['method']:32s} GPU={float(r['total_gpu_wall_s']):6.2f}s CPU={float(r['total_cpu_fit_s']):5.3f}s Tot={float(r['total_wall_s']):6.2f}s | W-Top1={float(r['weighted_top1_57q']):.4f} M-Top1={float(r['macro_top1_11wl']):.4f} MuSiQue25={float(r['musique_weighted_recall_25q']):.4f}")

if __name__ == "__main__":
    main()
