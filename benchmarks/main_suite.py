"""Stage 3.2 Full Main Experiment Suite (M1-M4) on Exclusive B200.

Covers all four VLDB Strong Accept pillars beyond the E1+E2 outpost:
  - M1a (32B Capacity-Aware Scheduler Validation): live B200 head-to-head of
      prefix vs merged_mask_trie (global) vs capacity_aware_merged_mask_trie
      on Qwen3-32B (273K KV tokens) across B=128 (fits) and B=256 (exceeds KV).
  - M1b (Tuned-SGLang Concurrency Control): tests sglang_native_lpm with
      --max-running-requests 16 / 32 / 128 on Qwen3-4B to verify whether tuning
      SGLang concurrency closes the gap against wave-decoupled scheduling.
  - M2 (Second Model Family — Gemma-4-E4B-it / Gemma-4-31B-it): natural complete
      answers, capacity_aware_merged_mask_trie vs prefix vs batchllm_dfs vs
      sglang_native_lpm, fair same-scheduler half/four vs cube.
  - M3 (New Real Multi-Hop RAG Attribution — MuSiQue + 2025-2026 SOTA Pareto):
      runs on MuSiQue 20-paragraph native multi-hop instances, 48-paragraph
      multi-question RAG packs, and redundant-evidence audit cases, collecting
      exact B200 wall-clock + scores for:
        cube (1-blk & 8-blk ref), half, four, cyclic_half_r2, cyclic_four_r2,
        iid_uniform (B=M, 2M, 4M for RegressionMSR-GBT / ProxySHAP-GBT),
        antithetic (B=M, 2M for Kernel Banzhaf / LASSO / Ridge),
        loo (BLIP-style), and arc_jsd (ICLR 2026 top-k JSD over response tokens).
  - M4 (Full-Cost Regime A/B + Fallback + Natural-Answer 8-Block Truth on QASPER/QuALITY):
      runs all 6 confirmation docs (incl. no-gain fallback docs QASPER-1909.08970
      and QuALITY-99914) + multi-question docs with natural complete answers,
      measuring end-to-end safe selection, fallback sunk costs, cyclic rotation,
      and 8-block held-out natural-answer reference truth.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE.parent / 'coattr'))

import designs  # noqa: E402
from capacity_scheduler import capacity_aware_merged_mask_trie_batches  # noqa: E402
from safe_selector import (  # noqa: E402
    exact_coset_certificates,
    gap_snr_certificate,
    vacant_channel_variance,
)
from scheduler import RequestRow, schedule_batches  # noqa: E402
from theory_extensions import (  # noqa: E402
    combined_multiblock_variance,
    cyclic_balanced_coset_tags,
)
from integration import (  # noqa: E402
    build_candidate_records,
    draw_design_rows,
    select_options,
)

SYSTEM = (
    "Answer the question using the provided document. Be concise. "
    "For multiple choice return the option letter."
)
SLOT = "p18_document_slot_12c925"


def save_json(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".partial")
    data = (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    tmp.write_bytes(gzip.compress(data, mtime=0) if path.suffix == ".gz" else data)
    tmp.replace(path)


def load_json(path: Path) -> object:
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


class Document:
    def __init__(self, tokenizer, raw: dict, order: str = "document-first") -> None:
        self.raw = raw
        self.n = len(raw["segments"])
        self.ident = f"{raw['dataset']}-{raw['doc_id']}"
        self.tokenizer = tokenizer
        self.parts = [self.encode(p) for p in raw["segments"]]
        self.sides = []
        self.separable = True
        for q in raw["questions"]:
            question = "Question: " + q.strip()
            content = (
                "Context:\n\n" + SLOT + question
                if order == "document-first"
                else question + "\n\nContext:\n\n" + SLOT
            )
            chat = self._apply_template(tokenizer, content)
            left, right = chat.split(SLOT)
            h, t = self.encode(left), self.encode(right)
            self.sides.append((left, right, h, t))
        full = (1 << self.n) - 1
        samples = [0, full, full // 3, full ^ (full // 3), *(1 << i for i in range(self.n))]
        for q, (_, _, head, tail) in enumerate(self.sides):
            for mask in samples:
                if head + self.body(mask) + tail != self.text_ids(mask, q):
                    self.separable = False
                    break
            if not self.separable:
                break

    @staticmethod
    def _apply_template(tokenizer, content: str) -> str:
        # Try Qwen3 non-thinking template first, then standard system+user, then user-only
        for msgs in (
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}],
            [{"role": "user", "content": SYSTEM + "\n\n" + content}],
        ):
            for extra in ({"enable_thinking": False}, {}):
                try:
                    out = tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True, **extra
                    )
                    if SLOT in out:
                        return out
                except Exception:
                    continue
        raise RuntimeError("Failed to apply chat template preserving SLOT sentinel")

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def body(self, mask: int) -> list[int]:
        return list(chain.from_iterable(tokens for i, tokens in enumerate(self.parts) if (mask >> i) & 1))

    def text_ids(self, mask: int, q: int) -> list[int]:
        left, right, _, _ = self.sides[q]
        middle = "".join(s for i, s in enumerate(self.raw["segments"]) if (mask >> i) & 1)
        return self.encode(left + middle + right)

    def prompts(self, mask: int, qs: Sequence[int]) -> list[list[int]]:
        if not self.separable:
            return [self.text_ids(mask, q) for q in qs]
        body = self.body(mask)
        return [self.sides[q][2] + body + self.sides[q][3] for q in qs]


def build_engine(
    model_path: str,
    *,
    mode: str = "fast",
    context_length: int = 16384,
    max_running_requests: int = 128,
    mem_fraction_static: float = 0.72,
):
    import sglang as sgl

    port = 24000 + (int(os.environ.get("SLURM_JOB_ID", "0")) % 17000)
    if mode == "det":
        return sgl.Engine(
            model_path=model_path,
            dtype="bfloat16",
            tp_size=1,
            attention_backend="triton",
            sampling_backend="pytorch",
            disable_cuda_graph=True,
            schedule_policy="lpm",
            context_length=context_length,
            chunked_prefill_size=8192,
            mem_fraction_static=0.68,
            max_running_requests=max_running_requests,
            random_seed=1701,
            log_level="warning",
            enable_deterministic_inference=True,
            port=port,
        )
    kwargs = dict(
        model_path=model_path,
        dtype="bfloat16",
        tp_size=1,
        schedule_policy="lpm",
        context_length=context_length,
        chunked_prefill_size=8192,
        mem_fraction_static=mem_fraction_static,
        max_running_requests=max_running_requests,
        random_seed=1701,
        log_level="warning",
        enable_deterministic_inference=False,
        port=port,
    )
    attn = os.environ.get("P18_ATTN_BACKEND", "").strip()
    if attn:
        kwargs["attention_backend"] = attn
    return sgl.Engine(**kwargs)


def flush_cache(server) -> None:
    result = None
    for _ in range(50):
        result = server.flush_cache()
        if result is True or getattr(result, "success", False):
            return
        time.sleep(0.1)
    raise RuntimeError(f"Cache flush did not complete: {result!r}")


def score_batch(
    server,
    prefixes: Sequence[list[int]],
    answers: Sequence[list[int]],
    *,
    top_logprobs_num: int = 0,
) -> tuple[list[float], int, list[list[list[tuple[float, int]]]] | None]:
    tokens = [p + a for p, a in zip(prefixes, answers, strict=True)]
    gen_kwargs: dict[str, object] = dict(
        input_ids=tokens,
        sampling_params={"temperature": 0, "max_new_tokens": 1},
        return_logprob=True,
        logprob_start_len=[len(p) - 1 for p in prefixes],
    )
    if top_logprobs_num > 0:
        gen_kwargs["top_logprobs_num"] = top_logprobs_num
    response = server.generate(**gen_kwargs)
    scores: list[float] = []
    cached = 0
    top_lps_all: list[list[list[tuple[float, int]]]] | None = [] if top_logprobs_num > 0 else None
    for item, answer, requested in zip(response, answers, tokens, strict=True):
        meta = item["meta_info"]
        if int(meta["prompt_tokens"]) != len(requested):
            raise ValueError("Returned prompt length differs from requested length")
        rows = meta["input_token_logprobs"][-len(answer) :]
        if [int(row[1]) for row in rows] != list(answer) or any(r[0] is None for r in rows):
            raise ValueError(f"Answer logprob alignment failed: {rows!r} vs {answer!r}")
        probs = [math.exp(float(row[0])) for row in rows]
        if not all(math.isfinite(p) and 0.0 <= p <= 1.0001 for p in probs):
            raise ValueError(f"Invalid probabilities: {probs}")
        scores.append(sum(probs) / len(probs))
        cached += int(meta.get("cached_tokens", 0))
        if top_lps_all is not None:
            raw_top = meta.get("input_top_logprobs") or []
            tail_top = raw_top[-len(answer) :]
            parsed = []
            for pos_list in tail_top:
                if not pos_list:
                    parsed.append([])
                else:
                    parsed.append([(float(x[0]), int(x[1])) for x in pos_list if x[0] is not None])
            top_lps_all.append(parsed)
    return scores, cached, top_lps_all


def compute_arc_jsd_score(
    full_top_lps: Sequence[Sequence[tuple[float, int]]],
    ablate_top_lps: Sequence[Sequence[tuple[float, int]]],
) -> float:
    """ARC-JSD (Li et al., ICLR 2026) Eq. (1): mean token-level Jensen-Shannon
    Divergence between full-context and ablated-context next-token distributions
    (approximated faithfully from top-K logprobs + explicit tail probability mass)."""
    if not full_top_lps or len(full_top_lps) != len(ablate_top_lps):
        return 0.0
    jsds = []
    for p_list, q_list in zip(full_top_lps, ablate_top_lps):
        p_map = {tid: math.exp(lp) for lp, tid in p_list}
        q_map = {tid: math.exp(lp) for lp, tid in q_list}
        p_rem = max(0.0, 1.0 - sum(p_map.values()))
        q_rem = max(0.0, 1.0 - sum(q_map.values()))
        keys = set(p_map) | set(q_map)
        jsd = 0.0
        for k in keys:
            pk = p_map.get(k, 0.0)
            qk = q_map.get(k, 0.0)
            mk = 0.5 * (pk + qk)
            if pk > 1e-15 and mk > 1e-15:
                jsd += 0.5 * pk * math.log(pk / mk)
            if qk > 1e-15 and mk > 1e-15:
                jsd += 0.5 * qk * math.log(qk / mk)
        m_rem = 0.5 * (p_rem + q_rem)
        if p_rem > 1e-15 and m_rem > 1e-15:
            jsd += 0.5 * p_rem * math.log(p_rem / m_rem)
        if q_rem > 1e-15 and m_rem > 1e-15:
            jsd += 0.5 * q_rem * math.log(q_rem / m_rem)
        jsds.append(max(0.0, jsd))
    return float(sum(jsds) / len(jsds)) if jsds else 0.0


def prepare_natural_answers(server, doc: Document, qs: Sequence[int], max_new_tokens: int = 256) -> list[dict]:
    full = (1 << doc.n) - 1
    outputs = server.generate(
        input_ids=doc.prompts(full, qs),
        sampling_params={"temperature": 0, "max_new_tokens": max_new_tokens},
        return_logprob=True,
    )
    ends = {doc.tokenizer.eos_token_id, doc.tokenizer.pad_token_id}
    for tok_str in ("<end_of_turn>", "<turn|>", "<|im_end|>"):
        tid = doc.tokenizer.convert_tokens_to_ids(tok_str)
        if isinstance(tid, int) and tid >= 0:
            ends.add(tid)
    out = []
    for q_idx, row in zip(qs, outputs, strict=True):
        meta = row["meta_info"]
        tokens = list(row.get("output_ids") or [int(x[1]) for x in meta["output_token_logprobs"]])
        while tokens and tokens[-1] in ends:
            tokens.pop()
        if not tokens:
            # Fallback to single token if immediate EOS
            tokens = [int(meta["output_token_logprobs"][0][1])]
        out.append(
            {
                "q": int(q_idx),
                "tokens": [int(t) for t in tokens],
                "token_len": len(tokens),
                "text": doc.tokenizer.decode(tokens),
                "finish": meta.get("finish_reason"),
            }
        )
    return out


def batchllm_dfs_batches(by_mask: Mapping[int, Sequence[RequestRow]], width: int = 32) -> list[list[RequestRow]]:
    items = []
    for mask, reqs in by_mask.items():
        if not reqs:
            continue
        reqs_sorted = sorted(reqs, key=lambda r: r[2])
        items.append((reqs_sorted[0][0], mask, reqs_sorted))
    items.sort(key=lambda x: (x[0], x[1]))
    batches: list[list[RequestRow]] = []
    suffix_queue: list[RequestRow] = []

    def drain(*, flush_all: bool = False) -> None:
        while len(suffix_queue) >= width or (flush_all and suffix_queue):
            batches.append(suffix_queue[:width])
            del suffix_queue[:width]

    for pos in range(0, len(items), width):
        tile = items[pos : pos + width]
        batches.append([reqs[0] for _, _, reqs in tile])
        max_q = max(len(reqs) for _, _, reqs in tile)
        for q_idx in range(1, max_q):
            for _, _, reqs in tile:
                if q_idx < len(reqs):
                    suffix_queue.append(reqs[q_idx])
        drain(flush_all=False)
    drain(flush_all=True)
    return batches


def build_schedule(
    by_mask: Mapping[int, Sequence[RequestRow]],
    layout: str,
    n: int,
    *,
    width: int = 32,
    kv_token_capacity: int | None = None,
    page_size: int = 64,
) -> tuple[list[list[RequestRow]], int]:
    if layout == "capacity_aware_merged_mask_trie":
        return capacity_aware_merged_mask_trie_batches(
            by_mask, width=width, kv_token_capacity=kv_token_capacity, page_size=page_size
        )
    if layout == "merged_mask_trie":
        return capacity_aware_merged_mask_trie_batches(
            by_mask, width=width, kv_token_capacity=None, page_size=page_size
        )
    if layout == "batchllm_dfs":
        return batchllm_dfs_batches(by_mask, width=width), len(by_mask)
    if layout == "sglang_native_lpm":
        all_rows = [row for reqs in by_mask.values() for row in sorted(reqs, key=lambda r: r[2])]
        return ([all_rows] if all_rows else []), len(by_mask)
    return list(schedule_batches(by_mask, layout, n, width=width)), len(by_mask)


def execute_scheduled(
    server,
    doc: Document,
    masks: Sequence[int],
    qs: Sequence[int],
    answers_by_q: Mapping[int, dict],
    layout: str,
    *,
    width: int = 32,
    kv_token_capacity: int | None = None,
    page_size: int = 64,
    top_logprobs_num: int = 0,
) -> tuple[dict[tuple[int, int], float], dict[str, object], dict[tuple[int, int], list] | None]:
    t0 = time.perf_counter()
    uniq_masks = list(dict.fromkeys(masks))
    by_mask: dict[int, list[RequestRow]] = {}
    for m in uniq_masks:
        prompts = doc.prompts(m, qs)
        by_mask[m] = [(toks, m, q) for q, toks in zip(qs, prompts, strict=True)]
    batches, tile_used = build_schedule(
        by_mask, layout, doc.n, width=width, kv_token_capacity=kv_token_capacity, page_size=page_size
    )
    t_prep = time.perf_counter()

    scores: dict[tuple[int, int], float] = {}
    top_lps_map: dict[tuple[int, int], list] | None = {} if top_logprobs_num > 0 else None
    total_input = 0
    total_cached = 0
    for batch in batches:
        prompts = [r[0] for r in batch]
        tails = [answers_by_q[r[2]]["tokens"] for r in batch]
        vals, hit, top_lps = score_batch(server, prompts, tails, top_logprobs_num=top_logprobs_num)
        for idx, ((_, m, q), sc) in enumerate(zip(batch, vals, strict=True)):
            scores[(m, q)] = sc
            if top_lps_map is not None and top_lps is not None:
                top_lps_map[(m, q)] = top_lps[idx]
        total_input += sum(len(p) + len(t) for p, t in zip(prompts, tails, strict=True))
        total_cached += hit
    t_end = time.perf_counter()

    return scores, {
        "requests": sum(len(b) for b in batches),
        "batches": len(batches),
        "tile_masks": tile_used,
        "input_tokens": total_input,
        "cached_tokens": total_cached,
        "uncached_tokens": max(0, total_input - total_cached),
        "prepare_seconds": t_prep - t0,
        "service_seconds": t_end - t_prep,
    }, top_lps_map


def decode_balanced_block(
    masks: Sequence[int],
    scores: Mapping[tuple[int, int], float],
    n: int,
    qs: Sequence[int],
) -> tuple[list[list[float]], list[float], list[int], list[float]]:
    m_half = len(masks) // 2
    z_half = np.array(
        [[1.0 if ((masks[2 * t] >> i) & 1) else -1.0 for i in range(n)] for t in range(m_half)],
        dtype=np.float64,
    )
    effs, vacs, top1s, gaps = [], [], [], []
    for q in qs:
        odd = np.array(
            [0.5 * (scores[(masks[2 * t], q)] - scores[(masks[2 * t + 1], q)]) for t in range(m_half)],
            dtype=np.float64,
        )
        eff = (2.0 * (z_half.T @ odd) / m_half).tolist()
        vac = vacant_channel_variance(odd.tolist(), eff, m_half)
        order = sorted(range(n), key=lambda i: (-eff[i], i))
        effs.append(eff)
        vacs.append(vac)
        top1s.append(order[0])
        gaps.append(float(eff[order[0]] - eff[order[1]]))
    return effs, vacs, top1s, gaps


def load_all_inputs(extra_paths: Sequence[Path]) -> dict[str, dict]:
    raw_by_ident: dict[str, dict] = {}
    default_paths = [
        Path("data/confirm-inputs.json"),
        Path("data/multi-inputs.json"),
        Path("data/inputs-rag.json"),
    ]
    for dp in [*default_paths, *extra_paths]:
        if dp.is_file():
            for rec in load_json(dp):  # type: ignore[union-attr]
                ident = f"{rec['dataset']}-{rec['doc_id']}"
                if ident not in raw_by_ident or len(rec["questions"]) > len(raw_by_ident[ident]["questions"]):
                    raw_by_ident[ident] = rec
    return raw_by_ident


def run_suite_m1_cap_or_m2_family(
    server,
    tokenizer,
    raw_by_ident: Mapping[str, dict],
    out: Path,
    args,
    kv_cap: int,
    page_size: int,
) -> None:
    """Shared high-precision wall-clock runner for M1a (32B capacity-aware),
    M1b (tuned SGLang max_running_requests), and M2 (Gemma-4 second model family)."""
    workload_specs = [(s.split(":")[0].strip(), int(s.split(":")[1])) for s in args.workloads.split(",") if s.strip()]
    design_list = [x.strip() for x in args.designs.split(",") if x.strip()]
    layout_list = [x.strip() for x in args.layouts.split(",") if x.strip()]

    max_nq = defaultdict(int)
    for ident, nq in workload_specs:
        max_nq[ident] = max(max_nq[ident], nq)

    docs: dict[str, Document] = {}
    answers_map: dict[str, dict[int, dict]] = {}
    for ident, nq_m in max_nq.items():
        doc = Document(tokenizer, raw_by_ident[ident])
        docs[ident] = doc
        qs_all = list(range(min(nq_m, len(doc.sides))))
        flush_cache(server)
        ans_list = prepare_natural_answers(server, doc, qs_all, max_new_tokens=args.max_new_tokens)
        answers_map[ident] = {a["q"]: a for a in ans_list}
        print(f"ANSWERS {ident} (n={doc.n}, doc_toks={sum(len(p) for p in doc.parts)}, nq={len(qs_all)}): lens={[a['token_len'] for a in ans_list]}", flush=True)
        save_json(out / f"{ident}-natural-answers.json", ans_list)

    # Full-length warmup outside timing
    first_id = workload_specs[0][0]
    w_doc = docs[first_id]
    w_ans = answers_map[first_id][0]["tokens"]
    score_batch(server, [w_doc.prompts((1 << w_doc.n) - 1, [0])[0]], [w_ans])
    flush_cache(server)

    summary = []
    for ident, nq in workload_specs:
        doc = docs[ident]
        qs = list(range(min(nq, len(doc.sides))))
        ans_by_q = answers_map[ident]
        budget_size = designs.budget(doc.n)
        m_half = budget_size // 2

        for rep in range(args.reps):
            masks_by_design: dict[str, tuple[list[int], float, float]] = {}
            for des in design_list:
                rng = random.Random(designs.seed("stage32-main", ident, rep, des))
                if des == "cube":
                    rows = designs.cube(doc.n, budget_size, rng, cut=None)
                    g_v, k_v = exact_coset_certificates([doc.n], m_half)
                elif des == "half":
                    rows = designs.cube(doc.n, budget_size, rng, cut=doc.n // 2)
                    g_v, k_v = exact_coset_certificates([doc.n // 2, doc.n - doc.n // 2], m_half)
                elif des == "four":
                    rows = designs.partition_cube(doc.n, budget_size, rng, groups=4)
                    szs = [(j + 1) * doc.n // 4 - j * doc.n // 4 for j in range(4)]
                    g_v, k_v = exact_coset_certificates(szs, m_half)
                else:
                    raise ValueError(des)
                masks_by_design[des] = (rows, g_v, k_v)

            tasks = [(d, l) for d in design_list for l in layout_list]
            random.Random(designs.seed("stage32-order", args.suite, ident, len(qs), rep)).shuffle(tasks)

            for des, lay in tasks:
                masks, g_v, k_v = masks_by_design[des]
                flush_cache(server)
                t0 = time.perf_counter()
                scores, meas, _ = execute_scheduled(
                    server, doc, masks, qs, ans_by_q, lay,
                    width=args.width, kv_token_capacity=kv_cap, page_size=page_size,
                )
                effs, vacs, top1s, gaps = decode_balanced_block(masks, scores, doc.n, qs)
                tot_s = time.perf_counter() - t0
                rec = {
                    "suite": args.suite,
                    "model": Path(args.model).name,
                    "mode": args.mode,
                    "max_running_requests": args.max_running_requests,
                    "kv_token_capacity": kv_cap,
                    "page_size": page_size,
                    "doc": ident,
                    "n": doc.n,
                    "doc_tokens": sum(len(p) for p in doc.parts),
                    "nq": len(qs),
                    "mean_answer_tokens": round(float(np.mean([ans_by_q[q]["token_len"] for q in qs])), 2),
                    "budget": budget_size,
                    "rep": rep,
                    "design": des,
                    "layout": lay,
                    "tile_masks": meas["tile_masks"],
                    "gamma": round(g_v, 5) if math.isfinite(g_v) else "inf",
                    "kappa": round(k_v, 5) if math.isfinite(k_v) else "inf",
                    "total_seconds": tot_s,
                    "prepare_seconds": meas["prepare_seconds"],
                    "service_seconds": meas["service_seconds"],
                    "requests": meas["requests"],
                    "batches": meas["batches"],
                    "input_tokens": meas["input_tokens"],
                    "cached_tokens": meas["cached_tokens"],
                    "uncached_tokens": meas["uncached_tokens"],
                    "top1": top1s,
                    "gap": [round(x, 6) for x in gaps],
                    "vacant_var": [float(f"{v:.4e}") for v in vacs],
                }
                summary.append(rec)
                save_json(out / f"{ident}-nq{len(qs)}-r{rep}-{des}-{lay}.json.gz", {**rec, "B": effs})
                print(
                    f"RUN {args.suite} {ident:18s} nq={len(qs):2d} r={rep} {des:4s} {lay:32s} "
                    f"(tile={meas['tile_masks']:3d}) | tot={tot_s:6.2f}s unc={meas['uncached_tokens']:7d} cached={meas['cached_tokens']:7d}",
                    flush=True,
                )

    save_json(out / "suite-summary.json", summary)
    print(f"SUITE {args.suite} COMPLETE: {len(summary)} records", flush=True)


def run_suite_m3_rag_or_m4_regime(
    server,
    tokenizer,
    raw_by_ident: Mapping[str, dict],
    out: Path,
    args,
    kv_cap: int,
    page_size: int,
) -> None:
    """Full accuracy-vs-wall-clock benchmark (M3 RAG SOTA + ARC-JSD + M4 Regime A/B + Fallback).

    For each workload:
      1. Generates natural complete answers (max_new_tokens=256).
      2. Collects 8 independent CUBE blocks as held-out natural-answer Banzhaf reference truth
         (blocks 0..7; evaluated methods use completely disjoint random seed namespaces!).
      3. Times & logs EVERY statistical method on the SAME fastest fair scheduler
         (capacity_aware_merged_mask_trie):
           - cube (1-blk, B=2M)
           - half (1-blk, B=2M)
           - four (1-blk, B=2M, when valid)
           - cyclic_half_r2 (2-blk cyclic rotated 2-coset, B=4M)
           - cyclic_four_r2 (2-blk cyclic rotated 4-coset, B=4M, when valid)
           - regime_a_safe (zero-pilot analytical selector + SNR swap confirmation)
           - iid_uniform_2M (i.i.d. uniform B=2M for RegressionMSR-GBT & ProxySHAP-GBT)
           - iid_uniform_4M (i.i.d. uniform B=4M for RegressionMSR-GBT & ProxySHAP-GBT)
           - antithetic_2M (antithetic pairs B=2M for Kernel Banzhaf / LASSO / Ridge)
           - loo (BLIP-style n+1 deletions)
           - arc_jsd (ICLR 2026 n+1 deletions with top_logprobs_num=20 JSD divergence)
    """
    workload_specs = [(s.split(":")[0].strip(), int(s.split(":")[1])) for s in args.workloads.split(",") if s.strip()]
    fair_layout = "capacity_aware_merged_mask_trie"
    all_records = []

    for ident, nq in workload_specs:
        raw = raw_by_ident[ident]
        doc = Document(tokenizer, raw)
        qs = list(range(min(nq, len(doc.sides))))
        flush_cache(server)
        ans_list = prepare_natural_answers(server, doc, qs, max_new_tokens=args.max_new_tokens)
        ans_by_q = {a["q"]: a for a in ans_list}
        save_json(out / f"{ident}-natural-answers.json", ans_list)
        print(
            f"DOC {ident} (n={doc.n}, doc_toks={sum(len(p) for p in doc.parts)}, nq={len(qs)}): "
            f"ans_lens={[a['token_len'] for a in ans_list]}",
            flush=True,
        )

        budget_size = designs.budget(doc.n)
        m_half = budget_size // 2

        # 1. Held-out natural-answer Banzhaf reference truth (8 independent CUBE blocks)
        ref_blocks_B = []
        for rb in range(args.ref_blocks):
            flush_cache(server)
            rng_ref = random.Random(designs.seed("stage32-ref-truth", ident, rb))
            ref_masks = designs.cube(doc.n, budget_size, rng_ref, cut=None)
            sc_ref, meas_ref, _ = execute_scheduled(
                server, doc, ref_masks, qs, ans_by_q, fair_layout,
                width=args.width, kv_token_capacity=kv_cap, page_size=page_size,
            )
            eff_ref, vac_ref, _, _ = decode_balanced_block(ref_masks, sc_ref, doc.n, qs)
            ref_blocks_B.append(eff_ref)
            save_json(
                out / f"{ident}-ref-{rb}.json.gz",
                {
                    "doc": ident, "n": doc.n, "nq": len(qs), "ref_block": rb,
                    "masks": [hex(m) for m in ref_masks],
                    "scores": {hex(m): [sc_ref[(m, q)] for q in qs] for m in dict.fromkeys(ref_masks)},
                    "B": eff_ref, "vacant_var": vac_ref, "metrics": meas_ref,
                },
            )
        truth_B = np.mean(np.asarray(ref_blocks_B, dtype=np.float64), axis=0)  # (nq, n)
        truth_top1 = [int(np.argmax(truth_B[q])) for q in range(len(qs))]
        print(f"  REF TRUTH ({args.ref_blocks} blocks) DONE for {ident}; true_top1={truth_top1}", flush=True)

        # Check feasibility of half & four
        can_half = (doc.n >= 2) and (doc.n - doc.n // 2 <= m_half // 2)
        four_sizes = [(j + 1) * doc.n // 4 - j * doc.n // 4 for j in range(4)]
        can_four = (doc.n >= 4) and (m_half % 4 == 0) and all(0 < sz <= m_half // 4 for sz in four_sizes)

        # Regime A zero-pilot safe selector decision (input-only metadata)
        seg_lens = [len(p) for p in doc.parts]
        head_len = len(doc.sides[0][2])
        tail_total = sum(len(doc.sides[q][3]) + ans_by_q[q]["token_len"] for q in qs)
        t_plan0 = time.perf_counter()
        recs = build_candidate_records(seg_lens, head_len, tail_total, len(qs), m_half)
        base_work = float(next(r["work"] for r in recs if r["name"] == "cube"))
        sel = select_options(recs, pilot_differences=None, max_gamma=1.05, max_kappa=1.50, minimum_saving=0.02 * base_work)
        plan_sec = time.perf_counter() - t_plan0
        chosen_opt = sel.selected_option
        print(
            f"  REGIME-A SELECTOR {ident}: selected={sel.decision.selected} fallback={sel.decision.used_fallback} "
            f"gamma={chosen_opt['gamma']:.4f} kappa={chosen_opt['kappa']} plan={plan_sec*1000:.2f}ms",
            flush=True,
        )

        # 2. Evaluate all methods across `args.reps` independent reps
        full_mask = (1 << doc.n) - 1
        for rep in range(args.reps):
            methods_to_run = ["cube", "iid_2M", "iid_4M", "antithetic_2M", "loo", "arc_jsd"]
            if can_half:
                methods_to_run.extend(["half", "cyclic_half_r2"])
            if can_four:
                methods_to_run.extend(["four", "cyclic_four_r2"])
            random.Random(designs.seed("stage32-m34-order", ident, rep)).shuffle(methods_to_run)

            for mname in methods_to_run:
                flush_cache(server)
                t0 = time.perf_counter()
                rng = random.Random(designs.seed("stage32-m34-draw", ident, rep, mname))
                extra_payload: dict[str, object] = {}

                if mname in ("cube", "half", "four"):
                    if mname == "cube":
                        masks = designs.cube(doc.n, budget_size, rng, cut=None)
                        g_v, k_v = exact_coset_certificates([doc.n], m_half)
                        coset_tags = [0] * doc.n
                    elif mname == "half":
                        cut = doc.n // 2
                        masks = designs.cube(doc.n, budget_size, rng, cut=cut)
                        g_v, k_v = exact_coset_certificates([cut, doc.n - cut], m_half)
                        coset_tags = [0 if i < cut else 1 for i in range(doc.n)]
                    else:
                        masks = designs.partition_cube(doc.n, budget_size, rng, groups=4)
                        g_v, k_v = exact_coset_certificates(four_sizes, m_half)
                        coset_tags = [j for j, sz in enumerate(four_sizes) for _ in range(sz)]
                    sc, meas, _ = execute_scheduled(
                        server, doc, masks, qs, ans_by_q, fair_layout,
                        width=args.width, kv_token_capacity=kv_cap, page_size=page_size,
                    )
                    effs, vacs, top1s, gaps = decode_balanced_block(masks, sc, doc.n, qs)
                    tot_s = time.perf_counter() - t0
                    # Also run zero-extra-query Regime-A SNR certificate diagnostic
                    certs = [
                        gap_snr_certificate(
                            effs[qi], vacs[qi], vacant_dof=m_half - doc.n,
                            kappa=k_v, coset_of=coset_tags if mname != "cube" else None,
                            z_threshold=2.0, min_vacant_dof=16,
                        )
                        for qi in range(len(qs))
                    ]
                    extra_payload["snr_triggered"] = [c.needs_confirmation for c in certs]
                    extra_payload["vacant_var"] = vacs
                    extra_payload["gamma"] = round(g_v, 5) if math.isfinite(g_v) else "inf"
                    extra_payload["kappa"] = round(k_v, 5) if math.isfinite(k_v) else "inf"
                    raw_masks, raw_scores = masks, sc

                elif mname in ("cyclic_half_r2", "cyclic_four_r2"):
                    g_cnt = 2 if mname == "cyclic_half_r2" else 4
                    tag_plans = cyclic_balanced_coset_tags(doc.n, 2, g=g_cnt)
                    blk_effs = []
                    blk_vacs = []
                    blk_masks_all = []
                    blk_scores_all = []
                    tot_unc = tot_cached = tot_in = tot_req = tot_bat = 0
                    for r_i, tg in enumerate(tag_plans):
                        b_masks, _, _ = draw_design_rows(doc.n, m_half, random.Random(designs.seed("cyc", ident, rep, mname, r_i)), tags=tg, groups=g_cnt)
                        sc_r, meas_r, _ = execute_scheduled(
                            server, doc, b_masks, qs, ans_by_q, fair_layout,
                            width=args.width, kv_token_capacity=kv_cap, page_size=page_size,
                        )
                        e_r, v_r, _, _ = decode_balanced_block(b_masks, sc_r, doc.n, qs)
                        blk_effs.append(e_r)
                        blk_vacs.append(v_r)
                        blk_masks_all.append([hex(m) for m in b_masks])
                        blk_scores_all.append({hex(m): [sc_r[(m, q)] for q in qs] for m in dict.fromkeys(b_masks)})
                        tot_unc += meas_r["uncached_tokens"]
                        tot_cached += meas_r["cached_tokens"]
                        tot_in += meas_r["input_tokens"]
                        tot_req += meas_r["requests"]
                        tot_bat += meas_r["batches"]
                    effs = np.mean(np.asarray(blk_effs, dtype=np.float64), axis=0).tolist()
                    top1s = [int(np.argmax(e)) for e in effs]
                    # Combined T1-B variance across the 2 blocks
                    comb_vars = [
                        combined_multiblock_variance([blk_effs[0][qi], blk_effs[1][qi]], [blk_vacs[0][qi], blk_vacs[1][qi]])["combined_avg_var"]
                        for qi in range(len(qs))
                    ]
                    tot_s = time.perf_counter() - t0
                    meas = {"requests": tot_req, "batches": tot_bat, "input_tokens": tot_in, "cached_tokens": tot_cached, "uncached_tokens": tot_unc}
                    extra_payload["combined_multiblock_avg_var"] = comb_vars
                    extra_payload["block_masks"] = blk_masks_all
                    extra_payload["block_scores"] = blk_scores_all
                    raw_masks, raw_scores = None, None

                elif mname in ("iid_2M", "iid_4M", "antithetic_2M"):
                    if mname == "iid_2M":
                        masks = [rng.getrandbits(doc.n) for _ in range(budget_size)]
                    elif mname == "iid_4M":
                        masks = [rng.getrandbits(doc.n) for _ in range(2 * budget_size)]
                    else:
                        masks = designs.antithetic(doc.n, budget_size, rng)
                    sc, meas, _ = execute_scheduled(
                        server, doc, masks, qs, ans_by_q, fair_layout,
                        width=args.width, kv_token_capacity=kv_cap, page_size=page_size,
                    )
                    # Standard MSR estimate (RegressionMSR-GBT / ProxySHAP-GBT / Kernel Banzhaf decoded post-hoc from saved scores)
                    z_mat = np.array([[1.0 if ((m >> i) & 1) else -1.0 for i in range(doc.n)] for m in masks], dtype=np.float64)
                    effs = []
                    for q in qs:
                        yv = np.array([sc[(m, q)] for m in masks], dtype=np.float64)
                        effs.append((2.0 * (z_mat.T @ yv) / len(masks)).tolist())
                    top1s = [int(np.argmax(e)) for e in effs]
                    tot_s = time.perf_counter() - t0
                    raw_masks, raw_scores = masks, sc

                elif mname == "loo":
                    masks = [full_mask, *[full_mask ^ (1 << i) for i in range(doc.n)]]
                    sc, meas, _ = execute_scheduled(
                        server, doc, masks, qs, ans_by_q, fair_layout,
                        width=args.width, kv_token_capacity=kv_cap, page_size=page_size,
                    )
                    effs = [[sc[(full_mask, q)] - sc[(full_mask ^ (1 << i), q)] for i in range(doc.n)] for q in qs]
                    top1s = [int(np.argmax(e)) for e in effs]
                    tot_s = time.perf_counter() - t0
                    raw_masks, raw_scores = masks, sc

                elif mname == "arc_jsd":
                    masks = [full_mask, *[full_mask ^ (1 << i) for i in range(doc.n)]]
                    sc, meas, top_lps_map = execute_scheduled(
                        server, doc, masks, qs, ans_by_q, fair_layout,
                        width=args.width, kv_token_capacity=kv_cap, page_size=page_size,
                        top_logprobs_num=20,
                    )
                    assert top_lps_map is not None
                    effs = [
                        [compute_arc_jsd_score(top_lps_map[(full_mask, q)], top_lps_map[(full_mask ^ (1 << i), q)]) for i in range(doc.n)]
                        for q in qs
                    ]
                    top1s = [int(np.argmax(e)) for e in effs]
                    tot_s = time.perf_counter() - t0
                    raw_masks, raw_scores = masks, sc
                else:
                    raise ValueError(mname)

                # Compute metrics vs held-out natural-answer Banzhaf truth (+ gold supporting paragraphs if present!)
                hits = [int(top1s[qi] == truth_top1[qi]) for qi in range(len(qs))]
                regrets = [float(truth_B[qi, truth_top1[qi]] - truth_B[qi, top1s[qi]]) for qi in range(len(qs))]
                mses = [float(np.mean((np.asarray(effs[qi]) - truth_B[qi]) ** 2)) for qi in range(len(qs))]

                gold_support = raw.get("supporting_segments")  # optional list[list[int]] for MuSiQue RAG
                gold_top1 = gold_recall_k = None
                if gold_support:
                    g1 = []
                    gr = []
                    for qi in range(len(qs)):
                        gset = set(gold_support[qi])
                        if not gset:
                            continue
                        g1.append(int(top1s[qi] in gset))
                        ranked = sorted(range(doc.n), key=lambda i: (-effs[qi][i], i))
                        k = len(gset)
                        gr.append(len(set(ranked[:k]) & gset) / k)
                    if g1:
                        gold_top1 = round(float(np.mean(g1)), 4)
                        gold_recall_k = round(float(np.mean(gr)), 4)

                rec = {
                    "suite": args.suite,
                    "model": Path(args.model).name,
                    "doc": ident,
                    "n": doc.n,
                    "nq": len(qs),
                    "budget_2M": budget_size,
                    "rep": rep,
                    "method": mname,
                    "regime_a_selected": sel.decision.selected,
                    "regime_a_fallback": sel.decision.used_fallback,
                    "regime_a_plan_seconds": plan_sec,
                    "total_seconds": tot_s,
                    "requests": meas["requests"],
                    "batches": meas["batches"],
                    "uncached_tokens": meas["uncached_tokens"],
                    "cached_tokens": meas["cached_tokens"],
                    "top1_vs_ref": round(float(np.mean(hits)), 4),
                    "mean_regret_vs_ref": round(float(np.mean(regrets)), 6),
                    "mean_mse_vs_ref": float(f"{np.mean(mses):.4e}"),
                    "gold_top1_hit": gold_top1,
                    "gold_recall_at_k": gold_recall_k,
                    "top1": top1s,
                    "truth_top1": truth_top1,
                }
                all_records.append(rec)
                save_obj = {**rec, **extra_payload, "B": effs}
                if raw_masks is not None and raw_scores is not None:
                    save_obj["masks"] = [hex(m) for m in raw_masks]
                    save_obj["scores"] = {hex(m): [raw_scores[(m, q)] for q in qs] for m in dict.fromkeys(raw_masks)}
                save_json(out / f"{ident}-r{rep}-{mname}.json.gz", save_obj)
                print(
                    f"M34 {ident:20s} r={rep} {mname:15s} | tot={tot_s:6.2f}s unc={meas['uncached_tokens']:7d} | "
                    f"top1={rec['top1_vs_ref']:.3f} reg={rec['mean_regret_vs_ref']:.5f} mse={rec['mean_mse_vs_ref']:.2e}"
                    + (f" | gold1={gold_top1:.3f} rec@k={gold_recall_k:.3f}" if gold_top1 is not None else ""),
                    flush=True,
                )

    save_json(out / "suite-summary.json", all_records)
    print(f"SUITE {args.suite} COMPLETE: {len(all_records)} records", flush=True)


def main() -> None:
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["m1_cap32b", "m1_tuned_lpm", "m2_gemma4", "m3_rag_sota", "m4_regime_b"], required=True)
    ap.add_argument("--mode", choices=["fast", "det"], default="fast")
    ap.add_argument("--model", type=str, default=os.environ.get("P18_MODEL", "Qwen/Qwen3-4B"))
    ap.add_argument("--workloads", type=str, required=True)
    ap.add_argument("--designs", type=str, default="cube,half,four")
    ap.add_argument("--layouts", type=str, default="prefix,merged_mask_trie,capacity_aware_merged_mask_trie")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--ref-blocks", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-running-requests", type=int, default=128)
    ap.add_argument("--mem-fraction-static", type=float, default=0.72)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--extra-inputs", type=str, default="")
    ap.add_argument("--smoke-only", action="store_true")
    args = ap.parse_args()

    out = Path(os.environ["P18_OUTPUT"])
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    extra_paths = [Path(p) for p in args.extra_inputs.split(",") if p.strip()]
    raw_by_ident = load_all_inputs(extra_paths)

    server = build_engine(
        args.model,
        mode=args.mode,
        max_running_requests=args.max_running_requests,
        mem_fraction_static=args.mem_fraction_static,
    )
    try:
        info = server.get_server_info()
        kv_cap = int(info.get("max_total_num_tokens") or 0)
        page_size = int(info.get("page_size") or 64)
        save_json(out / "engine.json", {
            k: info.get(k) for k in (
                "version", "page_size", "mem_fraction_static", "max_total_num_tokens",
                "max_running_requests", "context_length", "attention_backend",
                "sampling_backend", "schedule_policy", "chunked_prefill_size",
                "disable_cuda_graph", "dtype", "tp_size", "enable_deterministic_inference",
            )
        })
        print(f"ENGINE {Path(args.model).name}: backend={info.get('attention_backend')} kv_cap={kv_cap} page={page_size} max_run={info.get('max_running_requests')}", flush=True)
        # Smoke check
        p = tokenizer.encode("What is the capital of France? Answer:", add_special_tokens=False)
        a = tokenizer.encode(" Paris.", add_special_tokens=False)
        sc, _, _ = score_batch(server, [p, p], [a, a])
        flush_cache(server)
        print(f"SMOKE OK scores={sc}", flush=True)
        if args.smoke_only:
            return

        if args.suite in ("m1_cap32b", "m1_tuned_lpm", "m2_gemma4"):
            run_suite_m1_cap_or_m2_family(server, tokenizer, raw_by_ident, out, args, kv_cap, page_size)
        else:
            run_suite_m3_rag_or_m4_regime(server, tokenizer, raw_by_ident, out, args, kv_cap, page_size)
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
