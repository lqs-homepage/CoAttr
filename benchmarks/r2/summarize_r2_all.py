"""Deterministic offline aggregator, workload-cluster bootstrap CI calculator,
and 1 Hz B200 hardware telemetry auditor for CoAttr Root-Elevation Step 2
(Slurm Jobs 247, 248, 249, 250)."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
UPGRADE_DIR = THIS_DIR.parent
sys.path.insert(0, str(UPGRADE_DIR))

from coattr_v2.baselines_v2 import paired_sign_flip_pvalue, workload_cluster_bootstrap_ci  # noqa: E402


def audit_telemetry(res_root: Path, job_ids: list[str]) -> dict:
    per_job = {}
    tot_rows = 0
    tot_therm = 0
    tot_pwr = 0
    for jid in job_ids:
        tpath = res_root / jid / "telemetry.csv"
        rows = 0
        therm = 0
        pwr = 0
        temps, utils, mems, watts = [], [], [], []
        with tpath.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows += 1
                # Strip spaces in keys
                rk = {k.strip(): (v.strip() if v else "") for k, v in r.items() if k}
                ts = rk.get("clocks_event_reasons.hw_thermal_slowdown", "")
                ps = rk.get("clocks_event_reasons.hw_power_brake_slowdown", "")
                if ts.lower() == "active":
                    therm += 1
                if ps.lower() == "active":
                    pwr += 1
                try:
                    temps.append(float(rk["temperature.gpu"]))
                    utils.append(float(rk["utilization.gpu [%]"].split()[0]))
                    mems.append(float(rk["memory.used [MiB]"].split()[0]))
                    watts.append(float(rk["power.draw [W]"].split()[0]))
                except Exception:
                    pass
        tot_rows += rows
        tot_therm += therm
        tot_pwr += pwr
        per_job[jid] = {
            "samples_1hz": rows,
            "hw_thermal_slowdown": therm,
            "hw_power_brake_slowdown": pwr,
            "mean_gpu_util_pct": round(float(np.mean(utils)), 1) if utils else None,
            "peak_mem_mib": round(float(np.max(mems)), 1) if mems else None,
            "max_temp_c": round(float(np.max(temps)), 1) if temps else None,
            "mean_power_w": round(float(np.mean(watts)), 1) if watts else None,
        }
    return {
        "total_1hz_samples": tot_rows,
        "total_thermal_slowdowns": tot_therm,
        "total_power_brake_slowdowns": tot_pwr,
        "per_job": per_job,
    }


def load_cuad_records(job_dirs: list[Path]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for jd in job_dirs:
        for p in sorted(jd.glob("ckpt_cuad_*.json")):
            rec = json.loads(p.read_text(encoding="utf-8"))
            out[rec["doc_id"]] = rec
    return out


def load_forensics_records(job_dirs: list[Path]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for jd in job_dirs:
        for p in sorted(jd.glob("ckpt_for_*.json")):
            rec = json.loads(p.read_text(encoding="utf-8"))
            out[rec["doc_id"]] = rec
    return out


def aggregate_cuad_tier(docs: dict[str, dict], tier: str) -> dict:
    sel = [d for d in docs.values() if d["tier"] == tier]
    sel.sort(key=lambda d: d["doc_id"])
    if not sel:
        return {}
    all_methods = sorted({m for d in sel for m in d["methods"].keys()})
    method_stats = {}
    for m in all_methods:
        sub = [d for d in sel if m in d["methods"]]
        walls = [d["methods"][m]["wall_s"] for d in sub]
        # Question-weighted averages AND document-macro averages
        q_recs, q_mrrs, q_f3, q_f5, q_mw = [], [], [], [], []
        doc_recs, doc_f5, doc_mw = [], [], []
        wit_flips = []
        tot_masks, tot_evals, tot_unc, tot_inp = [], [], [], []
        for d in sub:
            md = d["methods"][m]
            doc_recs.append(md["clause_rec_at_maxk3"])
            doc_f5.append(md["decision_flip_k5"])
            if md.get("multi_witness_cov_top5") is not None:
                doc_mw.append(md["multi_witness_cov_top5"])
            if md.get("witness_removal_flip_rate") is not None:
                wit_flips.append(md["witness_removal_flip_rate"])
            if "n_masks" in md:
                tot_masks.append(md["n_masks"])
            if "total_evals" in md:
                tot_evals.append(md["total_evals"])
            if "uncached_tokens" in md:
                tot_unc.append(md["uncached_tokens"])
            if "input_tokens" in md:
                tot_inp.append(md["input_tokens"])
            for pq in md.get("per_q", []):
                q_recs.append(pq["clause_rec"])
                q_mrrs.append(pq["mrr"])
                q_f3.append(pq["flip_k3"])
                q_f5.append(pq["flip_k5"])
                if pq.get("mw_cov_top5") is not None:
                    q_mw.append(pq["mw_cov_top5"])
        entry = {
            "n_docs": len(sub),
            "n_predicates": len(q_recs),
            "total_wall_s": round(float(np.sum(walls)), 2),
            "mean_wall_s": round(float(np.mean(walls)), 2),
            "w_clause_rec": round(float(np.mean(q_recs)), 4),
            "w_mrr": round(float(np.mean(q_mrrs)), 4),
            "w_flip_k3": round(float(np.mean(q_f3)), 4),
            "w_flip_k5": round(float(np.mean(q_f5)), 4),
            "w_mw_cov_top5": round(float(np.mean(q_mw)), 4) if q_mw else None,
            "macro_clause_rec": round(float(np.mean(doc_recs)), 4),
            "macro_flip_k5": round(float(np.mean(doc_f5)), 4),
            "macro_mw_cov_top5": round(float(np.mean(doc_mw)), 4) if doc_mw else None,
        }
        if wit_flips:
            entry["macro_witness_removal_flip"] = round(float(np.mean(wit_flips)), 4)
        if tot_masks:
            entry["mean_masks"] = round(float(np.mean(tot_masks)), 1)
        if tot_evals:
            entry["mean_evals"] = round(float(np.mean(tot_evals)), 1)
        if tot_unc and tot_inp:
            entry["kv_cache_hit_rate"] = round(1.0 - float(np.sum(tot_unc)) / max(1.0, float(np.sum(tot_inp))), 4)
        method_stats[m] = entry

    # CoAttr-Verify summary across tier
    cv_walls = [d["coattr_verify"]["extra_wall_s"] for d in sel]
    cv_wits = [d["coattr_verify"]["mean_witnesses_found"] for d in sel]
    cv_unions = [d["coattr_verify"]["mean_union_chunks"] for d in sel]
    cv_evals = [d["coattr_verify"]["mean_extra_evals_per_q"] for d in sel]
    cv_flips = [d["coattr_verify"]["witness_removal_flip_rate"] for d in sel]
    cv_certs = [d["coattr_verify"]["remediation_certified_rate"] for d in sel]
    # Weighted by tested_q
    q_weights = [d["tested_q"] for d in sel]
    w_cv_flip = float(np.average(cv_flips, weights=q_weights))
    w_blip_flip = float(np.average([d["methods"]["blip_pruning"]["witness_removal_flip_rate"] for d in sel], weights=q_weights))
    return {
        "tier": tier,
        "n_docs": len(sel),
        "doc_ids": [d["doc_id"] for d in sel],
        "n_chunks_range": [min(d["n"] for d in sel), max(d["n"] for d in sel)],
        "tokens_range": [min(d["full_tokens"] for d in sel), max(d["full_tokens"] for d in sel)],
        "total_predicates": int(sum(q_weights)),
        "total_multi_witness_predicates": int(sum(d["tested_multi_witness_q"] for d in sel)),
        "median_diff_full": round(float(np.median([d["median_diff_full"] for d in sel])), 2),
        "median_diff_empty": round(float(np.median([d["median_diff_empty"] for d in sel])), 2),
        "methods": method_stats,
        "coattr_verify": {
            "total_extra_wall_s": round(float(np.sum(cv_walls)), 2),
            "mean_extra_wall_s": round(float(np.mean(cv_walls)), 2),
            "mean_witnesses_found": round(float(np.mean(cv_wits)), 2),
            "mean_union_chunks": round(float(np.mean(cv_unions)), 2),
            "mean_extra_evals_per_q": round(float(np.mean(cv_evals)), 1),
            "w_remediation_certified_rate": round(float(np.average(cv_certs, weights=q_weights)), 4),
            "w_witness_removal_flip_rate": round(w_cv_flip, 4),
            "w_blip_witness_removal_flip_rate": round(w_blip_flip, 4),
        },
    }


def compute_cuad_12doc_bootstrap_cis(docs: dict[str, dict]) -> dict:
    """12-contract workload-cluster bootstrap 95% CIs & paired sign-flip p-values
    comparing flagship CoAttr (Paley on std / Hier-CoAttr random on ultra_long) vs baselines."""
    ordered = sorted(docs.values(), key=lambda d: d["doc_id"])
    flag_key = lambda d: "coattr_paley" if d["tier"] == "standard" else "hier_coattr_random"
    cis = {}
    for base in ("loo", "contextcite_lasso", "tracllm", "blip_pruning"):
        cis[base] = {}
        for metric_key, pq_field in (
            ("clause_rec", "clause_rec"),
            ("mrr", "mrr"),
            ("flip_k5", "flip_k5"),
            ("mw_cov_top5", "mw_cov_top5"),
        ):
            doc_diffs, doc_weights = [], []
            for d in ordered:
                fk = flag_key(d)
                pqa = d["methods"][fk]["per_q"]
                pqb = d["methods"][base]["per_q"]
                vals = []
                for xa, xb in zip(pqa, pqb, strict=True):
                    va, vb = xa.get(pq_field), xb.get(pq_field)
                    if va is not None and vb is not None:
                        vals.append(float(va) - float(vb))
                if vals:
                    doc_diffs.append(float(np.mean(vals)))
                    doc_weights.append(float(len(vals)))
            ci = workload_cluster_bootstrap_ci(doc_diffs, weights=doc_weights, n_resamples=10000, seed=2027)
            pval = paired_sign_flip_pvalue(doc_diffs, weights=doc_weights)
            cis[base][metric_key] = {
                "diff_mean": round(ci["point_estimate"], 4),
                "ci95": [round(ci["ci_low"], 4), round(ci["ci_high"], 4)],
                "p_value": round(pval, 4),
                "n_clusters": len(doc_diffs),
            }
    return cis


def aggregate_forensics(fdocs: dict[str, dict]) -> dict:
    ordered = sorted(fdocs.values(), key=lambda d: d["doc_id"])
    rows = {}
    for d in ordered:
        did = d["doc_id"]
        rows[did] = {
            "n": d["n"],
            "full_tokens": d["full_tokens"],
            "mode": d["mode"],
            "nq": d["nq"],
            "benign_ok": d["benign_ok"],
            "hijacked": d["hijacked"],
            "split_synergy": d.get("split_synergy"),
            "methods": {
                m: {k: v for k, v in mv.items() if k != "san_texts"}
                for m, mv in d["methods"].items()
            },
            "coattr_verify": d["coattr_verify"],
        }

    # Aggregate over Redundant (Redund2 + Redund3) across scales, and over ALL 8 workloads
    groups = {
        "all": ordered,
        "std71_all": [d for d in ordered if d["doc_id"].startswith("Std71")],
        "long240_all": [d for d in ordered if d["doc_id"].startswith("Long240")],
        "redundant_all": [d for d in ordered if "Redund" in d["doc_id"]],
        "single_all": [d for d in ordered if "Single" in d["doc_id"]],
        "split_all": [d for d in ordered if "SplitInj" in d["doc_id"]],
    }
    agg = {}
    for gname, gdocs in groups.items():
        if not gdocs:
            continue
        # Map flagship CoAttr: coattr_paley for Std71, hier_coattr_random & hier_coattr_hybrid for Long240/Ext360
        m_keys = ["loo", "contextcite_lasso", "tracllm", "blip_pruning", "coattr_flagship_random", "coattr_flagship_hybrid"]
        g_out = {"n_workloads": len(gdocs), "total_incidents": sum(d["nq"] for d in gdocs), "total_hijacked": sum(d["hijacked"] for d in gdocs), "methods": {}}
        for mk in m_keys:
            walls, recs, exacts, recovs, wit_recs = [], [], [], [], []
            for d in gdocs:
                if mk == "coattr_flagship_random":
                    src_k = "coattr_paley" if "coattr_paley" in d["methods"] else "hier_coattr_random"
                elif mk == "coattr_flagship_hybrid":
                    src_k = "coattr_paley" if "coattr_paley" in d["methods"] else "hier_coattr_hybrid"
                else:
                    src_k = mk
                if src_k not in d["methods"]:
                    continue
                md = d["methods"][src_k]
                walls.append(md["wall_s"])
                recs.append(md["attack_rec_at_k"])
                exacts.append(md["exact_all_sources"])
                recovs.append(md["post_sanitization_recovery"])
                if "witness_sanitization_recovery" in md:
                    wit_recs.append(md["witness_sanitization_recovery"])
            entry = {
                "total_wall_s": round(float(np.sum(walls)), 2),
                "mean_attack_rec": round(float(np.mean(recs)), 4),
                "mean_exact_all": round(float(np.mean(exacts)), 4),
                "mean_post_san_recovery": round(float(np.mean(recovs)), 4),
            }
            if wit_recs:
                entry["mean_witness_san_recovery"] = round(float(np.mean(wit_recs)), 4)
            g_out["methods"][mk] = entry
        g_out["coattr_verify"] = {
            "total_extra_wall_s": round(float(sum(d["coattr_verify"]["extra_wall_s"] for d in gdocs)), 2),
            "mean_witnesses_found": round(float(np.mean([d["coattr_verify"]["mean_witnesses_found"] for d in gdocs])), 2),
            "mean_witness_san_recovery": round(float(np.mean([d["coattr_verify"]["witness_sanitization_recovery"] for d in gdocs])), 4),
            "mean_remediation_certified": round(float(np.mean([d["coattr_verify"]["remediation_certified_rate"] for d in gdocs])), 4),
        }
        agg[gname] = g_out
    return {"per_workload": rows, "aggregates": agg}


def aggregate_scheduler_33k(raw_sched: dict) -> dict:
    runs = raw_sched["runs"]
    out = {k: raw_sched[k] for k in ("doc_id", "n", "full_tokens", "n_masks", "nq", "kv_cap")}
    stats = {}
    for lay, rlist in runs.items():
        ws = [r["wall_s"] for r in rlist]
        uncs = [r["uncached_tokens"] for r in rlist]
        inps = [r["input_tokens"] for r in rlist]
        stats[lay] = {
            "mean_wall_s": round(float(np.mean(ws)), 2),
            "std_wall_s": round(float(np.std(ws, ddof=1)) if len(ws) > 1 else 0.0, 2),
            "effective_width": rlist[0]["effective_width"],
            "tile_masks": rlist[0]["tile_masks"],
            "mean_uncached_tokens": int(round(float(np.mean(uncs)))),
            "kv_hit_rate": round(1.0 - float(np.mean(uncs)) / max(1.0, float(np.mean(inps))), 4),
            "reps": ws,
        }
    v2_w = stats["capacity_aware_v2"]["mean_wall_s"]
    v2_u = stats["capacity_aware_v2"]["mean_uncached_tokens"]
    for lay, st in stats.items():
        st["v2_speedup_over_this"] = round(st["mean_wall_s"] / v2_w, 2)
        st["v2_uncached_reduction_pct"] = round(100.0 * (1.0 - v2_u / max(1, st["mean_uncached_tokens"])), 1)
    out["layouts"] = stats
    return out


def main() -> None:
    res_root = UPGRADE_DIR / "r2_results"
    telemetry = audit_telemetry(res_root, ["247", "248", "249", "250"])

    # Load Qwen3-4B (247)
    cuad_4b = load_cuad_records([res_root / "247"])
    for_4b = load_forensics_records([res_root / "247"])

    # Load Qwen3-32B (248 + 249)
    cuad_32b = load_cuad_records([res_root / "248", res_root / "249"])
    for_32b = load_forensics_records([res_root / "249"])

    # Load Gemma-4-31B-it (250)
    gemma_sum = json.loads((res_root / "250" / "summary.json").read_text(encoding="utf-8"))
    cuad_gemma = load_cuad_records([res_root / "250"])
    for_gemma = load_forensics_records([res_root / "250"])
    sched_gemma = aggregate_scheduler_33k(gemma_sum["scheduler_33k_ablation"])

    master = {
        "telemetry_audit": telemetry,
        "cuad": {
            "qwen3_4b": {
                "standard": aggregate_cuad_tier(cuad_4b, "standard"),
                "ultra_long": aggregate_cuad_tier(cuad_4b, "ultra_long"),
                "bootstrap_12doc_ci": compute_cuad_12doc_bootstrap_cis(cuad_4b),
            },
            "qwen3_32b": {
                "standard": aggregate_cuad_tier(cuad_32b, "standard"),
                "ultra_long": aggregate_cuad_tier(cuad_32b, "ultra_long"),
                "bootstrap_12doc_ci": compute_cuad_12doc_bootstrap_cis(cuad_32b),
            },
            "gemma4_31b": {
                "extreme_long": aggregate_cuad_tier(cuad_gemma, "extreme_long"),
            },
        },
        "forensics": {
            "qwen3_4b": aggregate_forensics(for_4b),
            "qwen3_32b": aggregate_forensics(for_32b),
            "gemma4_31b": aggregate_forensics(for_gemma),
        },
        "gemma4_33k_scheduler_ablation": sched_gemma,
    }

    out_path = res_root / "r2-master-summary.json"
    out_path.write_text(json.dumps(master, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"WROTE {out_path}\n")

    # Pretty console printout for immediate audit
    print("=== 1 HZ HARDWARE TELEMETRY AUDIT (JOBS 247, 248, 249, 250) ===")
    print(f"  Total 1 Hz samples: {telemetry['total_1hz_samples']:,} | Thermal slowdowns: {telemetry['total_thermal_slowdowns']} | Power brake slowdowns: {telemetry['total_power_brake_slowdowns']}")
    for jid, st in telemetry["per_job"].items():
        print(f"    Job {jid}: {st['samples_1hz']:,} samples, util={st['mean_gpu_util_pct']}%, peak_mem={st['peak_mem_mib']} MiB, max_temp={st['max_temp_c']}C, therm={st['hw_thermal_slowdown']}, pwr={st['hw_power_brake_slowdown']}")

    print("\n=== BATTLEGROUND A: CUAD MULTI-PREDICATE SEMANTIC SQL ===")
    for mdl, tiers in (("qwen3_4b", ("standard", "ultra_long")), ("qwen3_32b", ("standard", "ultra_long")), ("gemma4_31b", ("extreme_long",))):
        for tr in tiers:
            t = master["cuad"][mdl][tr]
            print(f"\n  [{mdl} | {tr}] docs={t['n_docs']} preds={t['total_predicates']} (mw={t['total_multi_witness_predicates']}) n={t['n_chunks_range']} tok={t['tokens_range']} med_full={t['median_diff_full']} med_empty={t['median_diff_empty']}")
            for m, mv in t["methods"].items():
                print(f"    {m:30s} wall={mv['total_wall_s']:7.1f}s rec={mv['w_clause_rec']:.4f} mrr={mv['w_mrr']:.4f} mw5={mv['w_mw_cov_top5']} f3={mv['w_flip_k3']:.4f} f5={mv['w_flip_k5']:.4f}")
            cv = t["coattr_verify"]
            print(f"    {'CoAttr-Verify':30s} +wall={cv['total_extra_wall_s']:6.1f}s wit={cv['mean_witnesses_found']:.2f} union={cv['mean_union_chunks']:.2f} flip={cv['w_witness_removal_flip_rate']:.4f} (vs BLIP {cv['w_blip_witness_removal_flip_rate']:.4f})")

    print("\n=== CUAD 12-DOC CLUSTER BOOTSTRAP 95% CIs (CoAttr vs Baselines) ===")
    for mdl in ("qwen3_4b", "qwen3_32b"):
        print(f"  [{mdl}]")
        for base, metrics in master["cuad"][mdl]["bootstrap_12doc_ci"].items():
            s = " | ".join(f"{k}: {v['diff_mean']:+.4f} [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}] p={v['p_value']:.4f}" for k, v in metrics.items())
            print(f"    vs {base:18s}: {s}")

    print("\n=== BATTLEGROUND B: ADVERSARIAL SECURITY FORENSICS ===")
    for mdl in ("qwen3_4b", "qwen3_32b", "gemma4_31b"):
        print(f"\n  [{mdl}] per-workload:")
        for did, r in master["forensics"][mdl]["per_workload"].items():
            syn = f" syn={r['split_synergy']}" if r.get("split_synergy") else ""
            print(f"    {did:18s} n={r['n']:3d} tok={r['full_tokens']:5d} benign={r['benign_ok']}/{r['nq']} hijack={r['hijacked']}/{r['nq']}{syn}")
            for m, mv in r["methods"].items():
                wrec = f" wit_rec={mv['witness_sanitization_recovery']:.3f}" if "witness_sanitization_recovery" in mv else ""
                print(f"      {m:24s} wall={mv['wall_s']:6.1f}s rec={mv['attack_rec_at_k']:.3f} exact={mv['exact_all_sources']:.3f} recover={mv['post_sanitization_recovery']:.3f}{wrec}")
            cv = r["coattr_verify"]
            print(f"      {'CoAttr-Verify':24s} +wall={cv['extra_wall_s']:5.1f}s wit={cv['mean_witnesses_found']:.2f} cert={cv['remediation_certified_rate']:.3f} wit_rec={cv['witness_sanitization_recovery']:.3f}")

        print(f"  [{mdl}] aggregates (redundant_all / all):")
        for grp in ("redundant_all", "all"):
            if grp not in master["forensics"][mdl]["aggregates"]:
                continue
            ag = master["forensics"][mdl]["aggregates"][grp]
            print(f"    -- {grp} (workloads={ag['n_workloads']}, hijacked={ag['total_hijacked']}/{ag['total_incidents']}) --")
            for mk, mv in ag["methods"].items():
                print(f"      {mk:24s} wall={mv['total_wall_s']:7.1f}s rec={mv['mean_attack_rec']:.4f} exact={mv['mean_exact_all']:.4f} recover={mv['mean_post_san_recovery']:.4f}")
            cv = ag["coattr_verify"]
            print(f"      {'CoAttr-Verify':24s} +wall={cv['total_extra_wall_s']:6.1f}s wit={cv['mean_witnesses_found']:.2f} wit_rec={cv['mean_witness_san_recovery']:.4f}")

    print("\n=== BATTLEGROUND C: GEMMA-4 33K-TOKEN 3-REP PHYSICAL SCHEDULER ABLATION ===")
    for lay, st in sched_gemma["layouts"].items():
        print(f"  {lay:20s} wall={st['mean_wall_s']:6.2f}±{st['std_wall_s']:.2f}s W'={st['effective_width']:3d} tile={st['tile_masks']:2d} uncached={st['mean_uncached_tokens']:,} hit={st['kv_hit_rate']*100:.1f}% (v2 speedup={st['v2_speedup_over_this']:.2f}x, unc_red={st['v2_uncached_reduction_pct']:+.1f}%)")


if __name__ == "__main__":
    main()
