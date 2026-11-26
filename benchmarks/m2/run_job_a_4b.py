"""M2 Job-A (GPU 0, Qwen3-4B SGLang + Single-Pass AttnTrace):
Master Statistical Rigor, 27-Workload / 150-Question Scale-Up, 10 Seeds,
16-Block Multi-Family Reference Truth, Kronecker Paley-Hadamard Cliffs,
Sequential Closed-Loop Regime A, Live i.i.d. Regime B (m=120), BLIP-Pruning,
Sketching-LOO, Low-Budget B=64, Single-Question n_q=1, and Single-Pass AttnTrace.
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import random
import time
from pathlib import Path

import numpy as np

from coattr_v2.baselines_v2 import (
    attntrace_single_pass_scores,
    blip_two_stage_pruning,
    draw_sketching_loo_masks,
    fit_sketching_loo_scores,
)
from coattr_v2.designs_v2 import (
    cyclic_balanced_coset_tags,
    draw_dyadic_block,
    draw_kronecker_paley_block,
    dyadic_half_budget,
    select_best_paley_configs,
)
from coattr_v2.safe_selector_v2 import (
    exact_coset_certificates,
    gap_snr_certificate,
    parseval_active_mse_diff,
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

# Core 11 workloads (exact Stage 3.2 continuity, 57 Qs) -> 10 seeds + 8 antithetic ref blocks
CORE_11: list[tuple[str, int]] = [
    ("QASPER-1909.08970", 2),
    ("QuALITY-99914", 2),
    ("QASPER-1912.06262", 2),
    ("QuALITY-20027", 2),
    ("QASPER-1810.05241", 6),
    ("QuALITY-22967", 18),
    ("MuSiQue-Pack40A", 3),
    ("MuSiQue-Pack48B", 4),
    ("MuSiQue-Pack72C", 6),
    ("MuSiQue-Pack88D", 8),
    ("MuSiQue-Redundant24", 4),
]

# New 16 disjoint workloads (93 new Qs -> Total 27 workloads / 150 Qs!)
NEW_16: list[tuple[str, int]] = [
    ("QASPER-1902.09666", 6),
    ("QASPER-2001.01269", 6),
    ("QASPER-1811.08603", 4),
    ("QASPER-1908.10084", 6),
    ("QASPER-1809.06537", 5),
    ("QuALITY-20031", 8),
    ("QuALITY-23160", 7),
    ("QuALITY-60745", 8),
    ("QuALITY-32665", 7),
    ("QuALITY-63936", 6),
    ("MuSiQue-Pack36E", 4),
    ("MuSiQue-Pack52F", 5),
    ("MuSiQue-Pack65G", 6),
    ("MuSiQue-Pack70H", 6),
    ("MuSiQue-Pack84I", 8),
    ("MuSiQue-SingleQ65", 1),
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


def _run_blip_live(server, doc: Document, qs: list[int], ans_by_q: dict[int, dict], epsilon: float = 0.05) -> dict:
    """Run genuine two-stage BLIP minimal-sufficient backward pruning live on SGLang."""
    flush_cache(server)
    t0 = time.perf_counter()
    all_sets = []
    all_scores = []
    total_evals = 0
    total_in = 0
    total_cached = 0
    for q in qs:
        ans_toks = ans_by_q[q]["tokens"]

        def eval_one(msk: int) -> float:
            nonlocal total_in, total_cached
            prompt = doc.prompts(msk, [q])[0]
            vals, hit = score_batch_sglang(server, [prompt], [ans_toks])
            total_in += len(prompt) + len(ans_toks)
            total_cached += hit
            return float(vals[0])

        res = blip_two_stage_pruning(doc.n, eval_one, epsilon=epsilon)
        all_sets.append(list(res.sufficient_set))
        all_scores.append(list(res.scores))
        total_evals += res.n_evals
    wall_s = time.perf_counter() - t0
    return {
        "total_seconds": wall_s,
        "total_evals": total_evals,
        "input_tokens": total_in,
        "cached_tokens": total_cached,
        "uncached_tokens": max(0, total_in - total_cached),
        "sufficient_sets": all_sets,
        "B": all_scores,
        "top1": [int(np.argmax(s)) for s in all_scores],
    }


def _eval_vs_truth(effs: list[list[float]], truth_B: np.ndarray, gold_support: list[list[int]] | None) -> dict:
    nq, n = truth_B.shape
    truth_top1 = [int(np.argmax(truth_B[q])) for q in range(nq)]
    top1s = [int(np.argmax(effs[q])) for q in range(nq)]
    hits = [int(top1s[q] == truth_top1[q]) for q in range(nq)]
    regrets = [float(truth_B[q, truth_top1[q]] - truth_B[q, top1s[q]]) for q in range(nq)]
    mses = [float(np.mean((np.asarray(effs[q]) - truth_B[q]) ** 2)) for q in range(nq)]
    g1 = gr = None
    if gold_support:
        g1_l, gr_l = [], []
        for q in range(nq):
            gset = set(gold_support[q])
            if not gset:
                continue
            g1_l.append(int(top1s[q] in gset))
            ranked = sorted(range(n), key=lambda i: (-effs[q][i], i))
            gr_l.append(len(set(ranked[: len(gset)]) & gset) / len(gset))
        if g1_l:
            g1 = round(float(np.mean(g1_l)), 4)
            gr = round(float(np.mean(gr_l)), 4)
    return {
        "top1": top1s,
        "top1_vs_ref": round(float(np.mean(hits)), 4),
        "mean_regret_vs_ref": round(float(np.mean(regrets)), 6),
        "mean_mse_vs_ref": float(f"{np.mean(mses):.4e}"),
        "gold_top1_hit": g1,
        "gold_recall_at_k": gr,
    }


def run_single_pass_attntrace(
    model_path: str,
    docs_answers: list[tuple[Document, list[int], dict[int, dict]]],
    out: Path,
) -> list[dict]:
    """Run real single-pass cross-layer attention extraction on B200 (eager attention, bfloat16)."""
    import torch
    from transformers import AutoModelForCausalLM

    print("\n=== STARTING SINGLE-PASS ATTNTRACE EXTRACTION (HF PyTorch bfloat16 eager) ===", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="cuda",
        local_files_only=True,
    )
    model.eval()
    results = []
    # Warmup
    with torch.no_grad():
        inp_w = torch.tensor([[1, 2, 3, 4]], device="cuda")
        _ = model(input_ids=inp_w, output_attentions=True)
        torch.cuda.synchronize()

    for doc, qs, ans_by_q in docs_answers:
        t0 = time.perf_counter()
        scores_mean_all = []
        scores_topk_all = []
        for q in qs:
            prompt_ids, spans = doc.full_context_chunk_spans(q)
            ans_ids = ans_by_q[q]["tokens"]
            full_ids = prompt_ids + ans_ids
            # Cap extremely long sequences if > 14000 tokens to avoid OOM in eager quadratic attn
            if len(full_ids) > 14000:
                scores_mean_all.append([0.0] * doc.n)
                scores_topk_all.append([0.0] * doc.n)
                continue
            p_len = len(prompt_ids)
            a_len = len(ans_ids)
            inp = torch.tensor([full_ids], dtype=torch.long, device="cuda")
            with torch.no_grad():
                out_m = model(input_ids=inp, output_attentions=True, use_cache=False)
                # Extract ONLY answer-token query rows (positions p_len-1 .. p_len+a_len-2) across upper half layers
                n_layers = len(out_m.attentions)
                start_l = n_layers // 2
                # Shape per layer: (1, H, T_full, T_full) -> slice (H, a_len, p_len)
                slices = []
                for l_idx in range(start_l, n_layers):
                    att_l = out_m.attentions[l_idx][0, :, p_len - 1 : p_len - 1 + a_len, :p_len]
                    slices.append(att_l.float().cpu().numpy())
                attn_np = np.stack(slices, axis=0)  # (L/2, H, a_len, p_len)
                del out_m
            sc_m = attntrace_single_pass_scores(attn_np, spans, layer_slice=slice(None), head_top_k=None)
            sc_k = attntrace_single_pass_scores(attn_np, spans, layer_slice=slice(None), head_top_k=8)
            scores_mean_all.append(sc_m.tolist())
            scores_topk_all.append(sc_k.tolist())
        torch.cuda.synchronize()
        wall_s = time.perf_counter() - t0
        rec = {
            "doc": doc.ident,
            "n": doc.n,
            "nq": len(qs),
            "wall_seconds": wall_s,
            "attntrace_mean_B": scores_mean_all,
            "attntrace_top8_B": scores_topk_all,
            "top1_mean": [int(np.argmax(x)) for x in scores_mean_all],
            "top1_top8": [int(np.argmax(x)) for x in scores_topk_all],
        }
        results.append(rec)
        save_json(out / f"attntrace-{doc.ident}.json.gz", rec)
        print(f"ATTNTRACE {doc.ident:20s} nq={len(qs):2d} | wall={wall_s:.3f}s top1_top8={rec['top1_top8']}", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    save_json(out / "attntrace-summary.json", results)
    return results


def main() -> None:
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    ap.add_argument("--core-seeds", type=int, default=10)
    ap.add_argument("--new-seeds", type=int, default=3)
    ap.add_argument("--regime-b-queries", type=int, default=120)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    out = Path(os.environ["P18_OUTPUT"])
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    raw_by_ident = load_all_m2_inputs()

    server = build_sglang_engine(args.model, max_running_requests=128, mem_fraction_static=0.72, port_offset=1)
    docs_for_attntrace: list[tuple[Document, list[int], dict[int, dict]]] = []
    all_summary: list[dict] = []
    regime_b_records: list[dict] = []
    truth_by_doc: dict[str, np.ndarray] = {}

    try:
        info = server.get_server_info()
        kv_cap = int(info.get("max_total_num_tokens") or 0)
        page_size = int(info.get("page_size") or 64)
        save_json(out / "engine.json", {k: info.get(k) for k in ("version", "page_size", "max_total_num_tokens", "attention_backend", "schedule_policy", "dtype")})
        print(f"JOB-A ENGINE {Path(args.model).name}: kv_cap={kv_cap} page={page_size} backend={info.get('attention_backend')}", flush=True)

        # Warmup
        p_w = tokenizer.encode("What is 2+2? Answer:", add_special_tokens=False)
        a_w = tokenizer.encode(" 4", add_special_tokens=False)
        score_batch_sglang(server, [p_w, p_w], [a_w, a_w])
        flush_cache(server)

        workloads_plan = [(ident, nq, True) for ident, nq in CORE_11] + [(ident, nq, False) for ident, nq in NEW_16]

        for ident, nq, is_core in workloads_plan:
            raw = raw_by_ident[ident]
            doc = Document(tokenizer, raw)
            qs = list(range(min(nq, len(doc.sides))))
            flush_cache(server)
            ans_list = prepare_natural_answers_sglang(server, doc, qs, max_new_tokens=args.max_new_tokens)
            ans_by_q = {a["q"]: a for a in ans_list}
            save_json(out / f"answers-{ident}.json", ans_list)
            docs_for_attntrace.append((doc, qs, ans_by_q))

            m_half = dyadic_half_budget(doc.n)
            budget_2m = 2 * m_half
            gold_support = raw.get("supporting_segments")
            print(f"\n=== WORKLOAD {ident} ({'CORE' if is_core else 'NEW'}, n={doc.n}, nq={len(qs)}, B=2M={budget_2m}) ===", flush=True)

            # 1. Design-Agnostic Reference Truth:
            #    - Core 11: 8 fresh independent antithetic i.i.d. blocks (combines with Stage 3.2 8 CUBE blocks -> 16-block truth!)
            #    - New 16: 4 CUBE + 4 antithetic i.i.d. = 8-block mixed truth!
            ref_cube_blocks = []
            ref_anti_blocks = []
            n_cube_ref = 0 if is_core else 4
            n_anti_ref = 8 if is_core else 4
            for rb in range(n_cube_ref):
                flush_cache(server)
                blk = draw_dyadic_block(doc.n, m_half, random.Random(stable_seed("m2-ref-cube", ident, rb)), g=1)
                sc_r, meas_r = execute_scheduled_sglang(server, doc, blk.rows, qs, ans_by_q, "capacity_aware_v2", kv_token_capacity=kv_cap, page_size=page_size)
                eff_r, vac_r, _, _ = decode_balanced_block_m2(blk.rows, sc_r, doc.n, qs)
                ref_cube_blocks.append(eff_r)
                save_json(out / f"ref-cube-{ident}-{rb}.json.gz", {"doc": ident, "rb": rb, "B": eff_r, "vac": vac_r, "metrics": meas_r})

            for rb in range(n_anti_ref):
                flush_cache(server)
                rows_a = _draw_antithetic(doc.n, budget_2m, random.Random(stable_seed("m2-ref-anti", ident, rb)))
                sc_r, meas_r = execute_scheduled_sglang(server, doc, rows_a, qs, ans_by_q, "capacity_aware_v2", kv_token_capacity=kv_cap, page_size=page_size)
                eff_r = _msr_decode(rows_a, sc_r, doc.n, qs)
                ref_anti_blocks.append(eff_r)
                save_json(out / f"ref-anti-{ident}-{rb}.json.gz", {"doc": ident, "rb": rb, "B": eff_r, "metrics": meas_r})

            all_local_refs = ref_cube_blocks + ref_anti_blocks
            truth_B = np.mean(np.asarray(all_local_refs, dtype=np.float64), axis=0)
            truth_by_doc[ident] = truth_B

            # Check coset & Paley feasibility
            can_half = (doc.n >= 2) and (doc.n - doc.n // 2 <= m_half // 2 - 1)
            four_sizes = [(j + 1) * doc.n // 4 - j * doc.n // 4 for j in range(4)]
            can_four = (doc.n >= 4) and (m_half % 4 == 0) and all(1 <= sz <= m_half // 4 - 1 for sz in four_sizes)
            paley_cfgs = select_best_paley_configs(doc.n, groups=(2, 4), mod8_eq4_only=True, strictly_smaller_than_dyadic=True)
            best_paley = paley_cfgs[0] if paley_cfgs else None

            # 2. Run multi-seed evaluation (10 seeds on Core 11, 3 seeds on New 16)
            n_seeds = args.core_seeds if is_core else args.new_seeds
            for sd in range(n_seeds):
                # Core methods every seed
                methods = ["cube", "cube_r2", "iid_2M", "iid_4M", "antithetic_2M"]
                if can_half:
                    methods.extend(["half", "cyclic_half_r2"])
                if can_four:
                    methods.extend(["four", "cyclic_four_r2"])
                if best_paley is not None:
                    methods.extend(["paley_r1", "paley_cyclic_r2"])
                # Low-budget & LOO/Sketching on first 3 seeds (keeps runtime crisp!)
                if sd < 3:
                    methods.extend(["iid_64", "sketching_loo_64", "loo"])
                random.Random(stable_seed("m2-joba-order", ident, sd)).shuffle(methods)

                for mname in methods:
                    flush_cache(server)
                    t0 = time.perf_counter()
                    rng = random.Random(stable_seed("m2-joba-draw", ident, sd, mname))
                    extra: dict[str, object] = {}

                    if mname in ("cube", "half", "four"):
                        g_cnt = 1 if mname == "cube" else (2 if mname == "half" else 4)
                        blk = draw_dyadic_block(doc.n, m_half, rng, g=g_cnt)
                        sizes = [blk.tags.count(r) for r in range(g_cnt)]
                        g_v, k_v = exact_coset_certificates(sizes, m_half)
                        sc, meas = execute_scheduled_sglang(server, doc, blk.rows, qs, ans_by_q, "capacity_aware_v2", kv_token_capacity=kv_cap, page_size=page_size)
                        effs, vacs, _, _ = decode_balanced_block_m2(blk.rows, sc, doc.n, qs)
                        tot_s = time.perf_counter() - t0
                        certs = [
                            gap_snr_certificate(
                                effs[qi], vacs[qi], vacant_dof=m_half - doc.n, kappa=k_v,
                                coset_of=blk.tags if g_cnt > 1 else None,
                                same_coset_factor=float(g_cnt), z_threshold=2.0, min_vacant_dof=16,
                            )
                            for qi in range(len(qs))
                        ]
                        extra = {
                            "vacant_var": vacs,
                            "snr": [c.snr for c in certs],
                            "snr_triggered": [c.needs_confirmation for c in certs],
                            "gamma": g_v,
                            "kappa": k_v if math.isfinite(k_v) else "inf",
                            "coset_of": list(blk.tags),
                        }
                    elif mname in ("cube_r2", "cyclic_half_r2", "cyclic_four_r2"):
                        g_cnt = 1 if mname == "cube_r2" else (2 if mname == "cyclic_half_r2" else 4)
                        tag_plans = [None, None] if g_cnt == 1 else cyclic_balanced_coset_tags(doc.n, 2, g=g_cnt)
                        blk_effs, blk_vacs, blk_times = [], [], []
                        tot_req = tot_bat = tot_in = tot_c = tot_u = 0
                        for bi in range(2):
                            tb = time.perf_counter()
                            blk = draw_dyadic_block(doc.n, m_half, random.Random(stable_seed("m2-r2", ident, sd, mname, bi)), tags=tag_plans[bi], g=g_cnt)
                            sc_b, meas_b = execute_scheduled_sglang(server, doc, blk.rows, qs, ans_by_q, "capacity_aware_v2", kv_token_capacity=kv_cap, page_size=page_size)
                            eb, vb, _, _ = decode_balanced_block_m2(blk.rows, sc_b, doc.n, qs)
                            blk_times.append(time.perf_counter() - tb)
                            blk_effs.append(eb)
                            blk_vacs.append(vb)
                            tot_req += meas_b["requests"]
                            tot_bat += meas_b["batches"]
                            tot_in += meas_b["input_tokens"]
                            tot_c += meas_b["cached_tokens"]
                            tot_u += meas_b["uncached_tokens"]
                        effs = np.mean(np.asarray(blk_effs, dtype=np.float64), axis=0).tolist()
                        tot_s = time.perf_counter() - t0
                        meas = {"requests": tot_req, "batches": tot_bat, "input_tokens": tot_in, "cached_tokens": tot_c, "uncached_tokens": tot_u}
                        extra = {
                            "blk0_B": blk_effs[0],
                            "blk1_B": blk_effs[1],
                            "blk0_vac": blk_vacs[0],
                            "blk1_vac": blk_vacs[1],
                            "blk_seconds": blk_times,
                            "blk0_tags": list(tag_plans[0]) if tag_plans[0] is not None else None,
                            "g": g_cnt,
                        }
                    elif mname == "paley_r1":
                        assert best_paley is not None
                        blk = draw_kronecker_paley_block(doc.n, int(best_paley["g"]), int(best_paley["L"]), rng)
                        sc, meas = execute_scheduled_sglang(server, doc, blk.rows, qs, ans_by_q, "capacity_aware_v2", kv_token_capacity=kv_cap, page_size=page_size)
                        effs, vacs, _, _ = decode_balanced_block_m2(blk.rows, sc, doc.n, qs)
                        tot_s = time.perf_counter() - t0
                        extra = {"paley_g": best_paley["g"], "paley_L": best_paley["L"], "paley_budget": best_paley["budget_2m"], "vacant_var": vacs}
                    elif mname == "paley_cyclic_r2":
                        assert best_paley is not None
                        gp, Lp = int(best_paley["g"]), int(best_paley["L"])
                        # Ensure coset 0 never exceeds L-1 under rotation
                        rot_tags = cyclic_balanced_coset_tags(doc.n, 2, g=gp)
                        blk_effs, blk_vacs, blk_times = [], [], []
                        tot_req = tot_bat = tot_in = tot_c = tot_u = 0
                        for bi in range(2):
                            tb = time.perf_counter()
                            tg = rot_tags[bi] if rot_tags[bi].count(0) <= Lp - 1 else None
                            blk = draw_kronecker_paley_block(doc.n, gp, Lp, random.Random(stable_seed("m2-pr2", ident, sd, bi)), tags=tg)
                            sc_b, meas_b = execute_scheduled_sglang(server, doc, blk.rows, qs, ans_by_q, "capacity_aware_v2", kv_token_capacity=kv_cap, page_size=page_size)
                            eb, vb, _, _ = decode_balanced_block_m2(blk.rows, sc_b, doc.n, qs)
                            blk_times.append(time.perf_counter() - tb)
                            blk_effs.append(eb)
                            blk_vacs.append(vb)
                            tot_req += meas_b["requests"]
                            tot_bat += meas_b["batches"]
                            tot_in += meas_b["input_tokens"]
                            tot_c += meas_b["cached_tokens"]
                            tot_u += meas_b["uncached_tokens"]
                        effs = np.mean(np.asarray(blk_effs, dtype=np.float64), axis=0).tolist()
                        tot_s = time.perf_counter() - t0
                        meas = {"requests": tot_req, "batches": tot_bat, "input_tokens": tot_in, "cached_tokens": tot_c, "uncached_tokens": tot_u}
                        extra = {"paley_g": gp, "paley_L": Lp, "paley_budget": 4 * gp * Lp, "blk0_B": blk_effs[0], "blk1_B": blk_effs[1], "blk_seconds": blk_times}
                    elif mname in ("iid_2M", "iid_4M", "antithetic_2M", "iid_64"):
                        if mname == "iid_2M":
                            masks = [rng.getrandbits(doc.n) for _ in range(budget_2m)]
                        elif mname == "iid_4M":
                            masks = [rng.getrandbits(doc.n) for _ in range(2 * budget_2m)]
                        elif mname == "iid_64":
                            masks = [rng.getrandbits(doc.n) for _ in range(64)]
                        else:
                            masks = _draw_antithetic(doc.n, budget_2m, rng)
                        sc, meas = execute_scheduled_sglang(server, doc, masks, qs, ans_by_q, "capacity_aware_v2", kv_token_capacity=kv_cap, page_size=page_size)
                        effs = _msr_decode(masks, sc, doc.n, qs)
                        tot_s = time.perf_counter() - t0
                        # Save raw masks + scores on first 3 seeds for GBT/ProxySHAP/LASSO post-hoc decoding!
                        if sd < 3:
                            extra = {
                                "masks": [hex(m) for m in masks],
                                "scores": {hex(m): [sc[(m, q)] for q in qs] for m in dict.fromkeys(masks)},
                            }
                    elif mname == "sketching_loo_64":
                        masks = draw_sketching_loo_masks(doc.n, 64, rng, drop_prob=0.25)
                        sc, meas = execute_scheduled_sglang(server, doc, masks, qs, ans_by_q, "capacity_aware_v2", kv_token_capacity=kv_cap, page_size=page_size)
                        effs = [fit_sketching_loo_scores(masks, [sc[(m, q)] for m in masks], doc.n, solver="lasso", alpha=1e-4).tolist() for q in qs]
                        tot_s = time.perf_counter() - t0
                    elif mname == "loo":
                        full_m = (1 << doc.n) - 1
                        masks = [full_m, *[full_m ^ (1 << i) for i in range(doc.n)]]
                        sc, meas = execute_scheduled_sglang(server, doc, masks, qs, ans_by_q, "capacity_aware_v2", kv_token_capacity=kv_cap, page_size=page_size)
                        effs = [[sc[(full_m, q)] - sc[(full_m ^ (1 << i), q)] for i in range(doc.n)] for q in qs]
                        tot_s = time.perf_counter() - t0
                    else:
                        raise ValueError(mname)

                    metrics = _eval_vs_truth(effs, truth_B, gold_support)
                    rec = {
                        "doc": ident,
                        "is_core": is_core,
                        "n": doc.n,
                        "nq": len(qs),
                        "budget_2M": budget_2m,
                        "seed": sd,
                        "method": mname,
                        "total_seconds": tot_s,
                        "requests": meas["requests"],
                        "batches": meas["batches"],
                        "uncached_tokens": meas["uncached_tokens"],
                        "cached_tokens": meas["cached_tokens"],
                        **metrics,
                    }
                    all_summary.append(rec)
                    save_json(out / f"run-{ident}-s{sd}-{mname}.json.gz", {**rec, **extra, "B": effs})
                    if sd == 0:
                        print(
                            f"  s0 {mname:16s} | tot={tot_s:5.2f}s unc={meas['uncached_tokens']:7d} | "
                            f"top1={rec['top1_vs_ref']:.3f} mse={rec['mean_mse_vs_ref']:.2e}"
                            + (f" gold1={rec['gold_top1_hit']:.3f} rec@k={rec['gold_recall_at_k']:.3f}" if rec["gold_top1_hit"] is not None else ""),
                            flush=True,
                        )

            # 3. Genuine BLIP two-stage backward pruning (live on SGLang, once per workload)
            blip_res = _run_blip_live(server, doc, qs, ans_by_q, epsilon=0.05)
            blip_eval = _eval_vs_truth(blip_res["B"], truth_B, gold_support)
            blip_rec = {
                "doc": ident,
                "is_core": is_core,
                "n": doc.n,
                "nq": len(qs),
                "budget_2M": budget_2m,
                "seed": 0,
                "method": "blip_pruning",
                "total_seconds": blip_res["total_seconds"],
                "total_evals": blip_res["total_evals"],
                "uncached_tokens": blip_res["uncached_tokens"],
                "cached_tokens": blip_res["cached_tokens"],
                "mean_suff_size": round(float(np.mean([len(s) for s in blip_res["sufficient_sets"]])), 2),
                **blip_eval,
            }
            all_summary.append(blip_rec)
            save_json(out / f"run-{ident}-s0-blip_pruning.json.gz", {**blip_rec, **blip_res})
            print(
                f"  BLIP-Pruning     | tot={blip_rec['total_seconds']:5.2f}s evals={blip_rec['total_evals']:4d} "
                f"|S*|={blip_rec['mean_suff_size']:.1f} top1={blip_rec['top1_vs_ref']:.3f}"
                + (f" gold1={blip_rec['gold_top1_hit']:.3f} rec@k={blip_rec['gold_recall_at_k']:.3f}" if blip_rec["gold_top1_hit"] is not None else ""),
                flush=True,
            )

            # Checkpoint summary after every workload!
            save_json(out / "job-a-summary.json", all_summary)

        # 4. LIVE TRUE-INDEPENDENT REGIME B STREAM (m = 120 truly independent randomized-design queries!)
        #    Each query q gets its OWN independent randomized CUBE and Ordered Four draw (zero shared masks across queries!)
        print(f"\n=== LIVE TRUE-INDEPENDENT REGIME B STREAM (target m={args.regime_b_queries} queries) ===", flush=True)
        q_pool: list[tuple[Document, int, dict[int, dict]]] = []
        for doc, qs, ans_by_q in docs_for_attntrace:
            m_half = dyadic_half_budget(doc.n)
            four_sz = [(j + 1) * doc.n // 4 - j * doc.n // 4 for j in range(4)]
            if doc.n >= 24 and all(1 <= sz <= m_half // 4 - 1 for sz in four_sz):
                for q in qs:
                    q_pool.append((doc, q, ans_by_q))

        for q_idx, (doc, q, ans_by_q) in enumerate(q_pool[: args.regime_b_queries]):
            m_half = dyadic_half_budget(doc.n)
            truth_q = truth_by_doc[doc.ident][q]
            flush_cache(server)
            blk_c = draw_dyadic_block(doc.n, m_half, random.Random(stable_seed("m2-regb-c", q_idx, doc.ident, q)), g=1)
            sc_c, _ = execute_scheduled_sglang(server, doc, blk_c.rows, [q], ans_by_q, "capacity_aware_v2", kv_token_capacity=kv_cap, page_size=page_size)
            eff_c, vac_c, _, _ = decode_balanced_block_m2(blk_c.rows, sc_c, doc.n, [q])

            flush_cache(server)
            blk_4 = draw_dyadic_block(doc.n, m_half, random.Random(stable_seed("m2-regb-4", q_idx, doc.ident, q)), g=4)
            sc_4, _ = execute_scheduled_sglang(server, doc, blk_4.rows, [q], ans_by_q, "capacity_aware_v2", kv_token_capacity=kv_cap, page_size=page_size)
            eff_4, vac_4, _, _ = decode_balanced_block_m2(blk_4.rows, sc_4, doc.n, [q])

            zd = parseval_active_mse_diff(vac_4[0], vac_c[0], m_half, doc.n)
            mse_c = float(np.mean((np.asarray(eff_c[0]) - truth_q) ** 2))
            mse_4 = float(np.mean((np.asarray(eff_4[0]) - truth_q) ** 2))
            regime_b_records.append(
                {
                    "stream_idx": q_idx,
                    "doc": doc.ident,
                    "q": q,
                    "n": doc.n,
                    "m_half": m_half,
                    "vac_cube": vac_c[0],
                    "vac_four": vac_4[0],
                    "z_d": zd,
                    "true_mse_cube": mse_c,
                    "true_mse_four": mse_4,
                    "true_mse_diff": mse_4 - mse_c,
                }
            )
            if (q_idx + 1) % 20 == 0 or q_idx + 1 == min(len(q_pool), args.regime_b_queries):
                mean_z = float(np.mean([r["z_d"] for r in regime_b_records]))
                mean_d = float(np.mean([r["true_mse_diff"] for r in regime_b_records]))
                print(f"  Regime-B stream m={len(regime_b_records):3d}: mean(Z_D)={mean_z:+.3e} vs mean(true MSE diff)={mean_d:+.3e}", flush=True)
        save_json(out / "regime-b-stream.json", regime_b_records)

    finally:
        server.shutdown()
        gc.collect()

    # 5. Single-pass AttnTrace extraction (after SGLang frees GPU memory cleanly!)
    run_single_pass_attntrace(args.model, docs_for_attntrace, out)
    print(f"JOB-A COMPLETE: {len(all_summary)} eval records, {len(regime_b_records)} Regime-B independent queries.", flush=True)


if __name__ == "__main__":
    main()
