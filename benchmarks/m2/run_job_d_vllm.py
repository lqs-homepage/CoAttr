"""M2 Job-D (GPU 3, Isolated uv vLLM Automatic Prefix Caching, block_size=16):
Cross-Engine Real-Hardware Validation of CoAttr on vLLM APC (block_size=16)
across BOTH Qwen3-4B (4 workloads incl. 18Q & n_q=1 cliff) and Qwen3-32B
(3 workloads), 5 reps each, directly measuring vLLM full-block-only hashing
vs SGLang RadixAttention token-level caching.
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
from typing import Mapping, Sequence

import numpy as np

from coattr_v2.capacity_scheduler_v2 import RequestRow
from coattr_v2.designs_v2 import (
    draw_dyadic_block,
    draw_kronecker_paley_block,
    dyadic_half_budget,
    select_best_paley_configs,
)
from m2_common import (
    Document,
    build_m2_schedule,
    decode_balanced_block_m2,
    load_all_m2_inputs,
    save_json,
    stable_seed,
)

VLLM_4B_WORKLOADS: list[tuple[str, int]] = [
    ("QuALITY-20027", 2),
    ("QASPER-1810.05241", 6),
    ("QuALITY-22967", 18),
    ("MuSiQue-SingleQ65", 1),
]

VLLM_32B_WORKLOADS: list[tuple[str, int]] = [
    ("QuALITY-20027", 2),
    ("QASPER-1810.05241", 6),
    ("QuALITY-22967", 2),
]


def _reset_vllm_prefix_cache(llm) -> None:
    if hasattr(llm, "reset_prefix_cache"):
        llm.reset_prefix_cache()
    elif hasattr(llm, "llm_engine") and hasattr(llm.llm_engine, "reset_prefix_cache"):
        llm.llm_engine.reset_prefix_cache()
    else:
        raise RuntimeError("vLLM engine does not expose reset_prefix_cache()")


def _get_vllm_kv_capacity(llm, block_size: int) -> int:
    """Query exact HBM KV token capacity from vLLM v0 or v1 engine."""
    try:
        eng = llm.llm_engine
        # v0 engine
        if hasattr(eng, "cache_config") and getattr(eng.cache_config, "num_gpu_blocks", None):
            return int(eng.cache_config.num_gpu_blocks) * block_size
        # v1 engine
        if hasattr(eng, "engine_core"):
            cfg = getattr(eng, "vllm_config", None)
            if cfg and getattr(cfg.cache_config, "num_gpu_blocks", None):
                return int(cfg.cache_config.num_gpu_blocks) * block_size
    except Exception:
        pass
    return 0


def _prepare_answers_vllm(llm, doc: Document, qs: Sequence[int], max_new_tokens: int = 256) -> list[dict]:
    from vllm import SamplingParams

    full = (1 << doc.n) - 1
    prompts = [{"prompt_token_ids": p} for p in doc.prompts(full, qs)]
    sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    outputs = llm.generate(prompts, sampling_params=sp, use_tqdm=False)
    ends = {doc.tokenizer.eos_token_id, doc.tokenizer.pad_token_id}
    for tok_str in ("<end_of_turn>", "<turn|>", "<|im_end|>"):
        tid = doc.tokenizer.convert_tokens_to_ids(tok_str)
        if isinstance(tid, int) and tid >= 0:
            ends.add(tid)
    out = []
    for q_idx, req_out in zip(qs, outputs, strict=True):
        toks = list(req_out.outputs[0].token_ids)
        while toks and toks[-1] in ends:
            toks.pop()
        if not toks:
            toks = list(req_out.outputs[0].token_ids)[:1]
        out.append(
            {
                "q": int(q_idx),
                "tokens": [int(t) for t in toks],
                "token_len": len(toks),
                "text": doc.tokenizer.decode(toks),
            }
        )
    return out


def _score_batch_vllm(
    llm,
    prefixes: Sequence[list[int]],
    answers: Sequence[list[int]],
) -> tuple[list[float], int]:
    from vllm import SamplingParams

    prompts = [{"prompt_token_ids": p + a} for p, a in zip(prefixes, answers, strict=True)]
    # CRITICAL vLLM V1 setting: prompt_logprob_start = len(p) - 1 AND skip_reading_prefix_cache=False
    # so vLLM APC (block_size=16) reuses ALL cached prefix blocks before the answer tokens!
    sps = [
        SamplingParams(
            temperature=0.0,
            max_tokens=1,
            prompt_logprobs=1,
            prompt_logprob_start=max(0, len(p) - 1),
            skip_reading_prefix_cache=False,
        )
        for p in prefixes
    ]
    outputs = llm.generate(prompts, sampling_params=sps, use_tqdm=False)
    scores: list[float] = []
    cached = 0
    for req_out, ans in zip(outputs, answers, strict=True):
        plps = req_out.prompt_logprobs
        if plps is None or len(plps) < len(ans):
            raise RuntimeError("vLLM did not return prompt_logprobs")
        tail_lps = plps[-len(ans) :]
        probs = []
        for pos_dict, tid in zip(tail_lps, ans, strict=True):
            if pos_dict is None or tid not in pos_dict:
                raise RuntimeError(f"Missing answer token {tid} in vLLM prompt_logprobs")
            lp_obj = pos_dict[tid]
            lp_val = float(lp_obj.logprob if hasattr(lp_obj, "logprob") else lp_obj)
            probs.append(math.exp(lp_val))
        scores.append(sum(probs) / len(probs))
        c_toks = getattr(req_out, "num_cached_tokens", None)
        if c_toks is None and getattr(req_out, "metrics", None) is not None:
            c_toks = getattr(req_out.metrics, "num_cached_tokens", 0)
        cached += int(c_toks or 0)
    return scores, cached


def _execute_vllm_scheduled(
    llm,
    doc: Document,
    masks: Sequence[int],
    qs: Sequence[int],
    answers_by_q: Mapping[int, dict],
    layout: str,
    *,
    width: int = 32,
    kv_token_capacity: int | None = None,
    block_size: int = 16,
) -> tuple[dict[tuple[int, int], float], dict[str, object]]:
    t0 = time.perf_counter()
    uniq_masks = list(dict.fromkeys(masks))
    by_mask: dict[int, list[RequestRow]] = {}
    for m in uniq_masks:
        prompts = doc.prompts(m, qs)
        by_mask[m] = [(toks, m, q) for q, toks in zip(qs, prompts, strict=True)]
    batches, sched_meta = build_m2_schedule(
        by_mask, layout, width=width, kv_token_capacity=kv_token_capacity, page_size=block_size
    )
    t_prep = time.perf_counter()

    scores: dict[tuple[int, int], float] = {}
    total_in = 0
    total_cached = 0
    for batch in batches:
        prefixes = [r[0] for r in batch]
        tails = [answers_by_q[r[2]]["tokens"] for r in batch]
        vals, hit = _score_batch_vllm(llm, prefixes, tails)
        for (_, m, q), sc in zip(batch, vals, strict=True):
            scores[(m, q)] = sc
        total_in += sum(len(p) + len(t) for p, t in zip(prefixes, tails, strict=True))
        total_cached += hit
    t_end = time.perf_counter()

    return scores, {
        "requests": sum(len(b) for b in batches),
        "batches": len(batches),
        **sched_meta,
        "input_tokens": total_in,
        "cached_tokens": total_cached,
        "uncached_tokens": max(0, total_in - total_cached),
        "prepare_seconds": t_prep - t0,
        "service_seconds": t_end - t_prep,
    }


def run_vllm_model(
    model_path: str,
    workloads: list[tuple[str, int]],
    raw_by_ident: Mapping[str, dict],
    out: Path,
    *,
    reps: int = 5,
    block_size: int = 16,
    gpu_mem_util: float = 0.78,
) -> list[dict]:
    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM

    model_name = Path(model_path).name
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        enable_prefix_caching=True,
        block_size=block_size,
        max_model_len=16384,
        gpu_memory_utilization=gpu_mem_util,
        seed=1701,
        enforce_eager=False,
    )
    kv_cap = _get_vllm_kv_capacity(llm, block_size)
    # Fallback estimate if v1 engine hides num_gpu_blocks: use safe known B200 pool
    if kv_cap <= 0:
        kv_cap = 800000 if "4B" in model_name else 270000
    save_json(
        out / f"engine-vllm-{model_name}.json",
        {
            "engine": "vllm",
            "vllm_version": vllm.__version__,
            "torch_version": torch.__version__,
            "model": model_name,
            "block_size": block_size,
            "enable_prefix_caching": True,
            "kv_token_capacity": kv_cap,
            "gpu_memory_utilization": gpu_mem_util,
        },
    )
    print(f"\n=== VLLM {vllm.__version__} ENGINE {model_name}: block_size={block_size}, kv_cap={kv_cap} ===", flush=True)

    # Smoke check (2 sequential calls on a >64-token prefix to verify LIVE APC hits > 0!)
    p_s = tokenizer.encode("The quick brown fox jumps over the lazy dog. " * 12 + "What is the capital of France? Answer:", add_special_tokens=False)
    a_s = tokenizer.encode(" Paris.", add_special_tokens=False)
    sc_s1, hit_s1 = _score_batch_vllm(llm, [p_s], [a_s])
    sc_s2, hit_s2 = _score_batch_vllm(llm, [p_s], [a_s])
    _reset_vllm_prefix_cache(llm)
    print(f"VLLM SMOKE OK s1={sc_s1} hit1={hit_s1} | s2={sc_s2} hit2={hit_s2} (prompt_len={len(p_s)})", flush=True)
    if hit_s2 <= 0:
        raise RuntimeError(f"vLLM APC did not hit on identical repeated prefix! hit2={hit_s2}")

    # Prepare answers
    docs: dict[str, Document] = {}
    ans_map: dict[str, dict[int, dict]] = {}
    for ident, nq in workloads:
        if ident not in docs:
            doc = Document(tokenizer, raw_by_ident[ident])
            docs[ident] = doc
            qs_all = list(range(min(nq, len(doc.sides))))
            _reset_vllm_prefix_cache(llm)
            ans_list = _prepare_answers_vllm(llm, doc, qs_all, max_new_tokens=256)
            ans_map[ident] = {a["q"]: a for a in ans_list}
            save_json(out / f"answers-vllm-{model_name}-{ident}.json", ans_list)
            print(f"VLLM ANSWERS {model_name} {ident}: lens={[a['token_len'] for a in ans_list]}", flush=True)

    summary: list[dict] = []
    layouts_four = ["vllm_native_all", "batchllm_dfs", "prefix", "capacity_aware_v2"]
    layouts_cube = ["vllm_native_all", "batchllm_dfs", "capacity_aware_v2"]

    for ident, nq in workloads:
        doc = docs[ident]
        qs = list(range(min(nq, len(doc.sides))))
        ans_by_q = ans_map[ident]
        m_half = dyadic_half_budget(doc.n)
        paley_cfgs = select_best_paley_configs(doc.n, groups=(2, 4), mod8_eq4_only=True, strictly_smaller_than_dyadic=True)
        best_paley = paley_cfgs[0] if paley_cfgs else None
        n_reps = 3 if (nq >= 18 and "32B" in model_name) else reps

        for rep in range(n_reps):
            masks_by_des = {}
            tasks = []
            for des in (["cube", "four"] + (["paley_r1"] if best_paley is not None else [])):
                rng = random.Random(stable_seed("m2-vllm", model_name, ident, rep, des))
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
                    for lay in ("capacity_aware_v2", "vllm_native_all"):
                        tasks.append((des, lay))

            random.Random(stable_seed("m2-vllm-ord", model_name, ident, rep)).shuffle(tasks)
            for des, lay in tasks:
                rows, dmeta = masks_by_des[des]
                _reset_vllm_prefix_cache(llm)
                t0 = time.perf_counter()
                scores, meas = _execute_vllm_scheduled(
                    llm, doc, rows, qs, ans_by_q, lay,
                    width=32, kv_token_capacity=kv_cap, block_size=block_size,
                )
                _, _, top1s, _ = decode_balanced_block_m2(rows, scores, doc.n, qs)
                tot_s = time.perf_counter() - t0
                rec = {
                    "engine": "vllm",
                    "vllm_version": vllm.__version__,
                    "block_size": block_size,
                    "model": model_name,
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
                summary.append(rec)
                save_json(out / f"vllm-{model_name}-{ident}-nq{len(qs)}-r{rep}-{des}-{lay}.json.gz", rec)
                if rep == 0:
                    print(
                        f"  VLLM {model_name:9s} {ident:18s} nq={len(qs):2d} {des:8s} {lay:18s} | "
                        f"tot={tot_s:6.2f}s unc={meas['uncached_tokens']:7d} cached={meas['cached_tokens']:7d}",
                        flush=True,
                    )
        save_json(out / f"vllm-{model_name}-summary.json", summary)

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(3)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--block-size", type=int, default=16)
    args = ap.parse_args()

    out = Path(os.environ["P18_OUTPUT"])
    out.mkdir(parents=True, exist_ok=True)
    raw_by_ident = load_all_m2_inputs()

    s4 = run_vllm_model("Qwen/Qwen3-4B", VLLM_4B_WORKLOADS, raw_by_ident, out, reps=args.reps, block_size=args.block_size)
    s32 = run_vllm_model("Qwen/Qwen3-32B", VLLM_32B_WORKLOADS, raw_by_ident, out, reps=args.reps, block_size=args.block_size)
    save_json(out / "vllm-all-summary.json", s4 + s32)
    print(f"JOB-D COMPLETE: 4B={len(s4)} records, 32B={len(s32)} records", flush=True)


if __name__ == "__main__":
    main()
