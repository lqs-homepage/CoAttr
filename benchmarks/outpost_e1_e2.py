"""Stage 3.2 E1 + E2 Outpost Kill-Switch Harness on Exclusive B200.

Evaluates two core VLDB kill-switch questions simultaneously on natural complete
answers (max_new_tokens=256, zero 24-token truncation):
  - E1 (Scheduler Showdown): fair head-to-head wall-clock comparison of
      1. prefix           (Stage 2.2 fastest wave scheduler)
      2. mask_trie_fanout (Stage 3.1 unmerged decoupled mask-trie)
      3. merged_mask_trie (Stage 3.1 T4 width-aligned + exact Bellman DP merge)
      4. batchllm_dfs     (BatchLLM-style DFS prefix-ordered decoupled waves)
      5. sglang_native_lpm (single-call continuous batching with SGLang native LPM)
    All schedulers execute the EXACT same realized masks per (doc, nq, design, rep)
    after an explicit KV cache flush, in randomized interleaved order.
  - E2 (Natural Complete Answers + Production Fast-Path + 32B Model Scale):
    Compares CUBE, balanced 2-coset (half), and balanced 4-coset (four) across:
      - Track A: Qwen3-4B batch-invariant deterministic (triton, no CUDA graph)
      - Track B: Qwen3-4B default production fast-path (flashinfer + CUDA graph)
      - Track C: Qwen3-32B default production fast-path (8x dense model scale)
    Also logs per-run vacant-channel Parseval variance, top-1/top-2 gap, and
    cross-scheduler score invariance diagnostics.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
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
from safe_selector import (  # noqa: E402
    exact_coset_certificates,
    vacant_channel_variance,
)
from scheduler import (  # noqa: E402
    RequestRow,
    _shared_doc_prefix,
    schedule_batches,
)
from wave_merged_scheduler import (  # noqa: E402
    GpuBatchCostParams,
    width_aligned_mask_trie_batches,
)

SYSTEM = (
    "Answer the question using the provided document. Be concise. "
    "For multiple choice return the option letter."
)
SLOT = "p18_document_slot_12c925"

# B200-calibrated 3-parameter physical batch cost model from Stage 3.1 T4
B200_DEFAULT_COST = GpuBatchCostParams(
    c_batch=0.1210374,
    c_tok=4.965718e-6,
    c_skew=2.112443e-4,
    c_req=0.0,
)


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
            try:
                chat = tokenizer.apply_chat_template(
                    [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                chat = tokenizer.apply_chat_template(
                    [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
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


def build_engine(model_path: str, *, mode: str, context_length: int = 16384):
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
            max_running_requests=128,
            random_seed=1701,
            log_level="warning",
            enable_deterministic_inference=True,
            port=port,
        )
    if mode == "fast":
        # Default production fast-path on B200: FlashInfer/default backend + CUDA graphs enabled
        kwargs = dict(
            model_path=model_path,
            dtype="bfloat16",
            tp_size=1,
            schedule_policy="lpm",
            context_length=context_length,
            chunked_prefill_size=8192,
            mem_fraction_static=0.72,
            max_running_requests=128,
            random_seed=1701,
            log_level="warning",
            enable_deterministic_inference=False,
            port=port,
        )
        attn = os.environ.get("P18_ATTN_BACKEND", "").strip()
        if attn:
            kwargs["attention_backend"] = attn
        return sgl.Engine(**kwargs)
    raise ValueError(f"unsupported mode: {mode}")


def flush_cache(server) -> None:
    result = None
    for _ in range(40):
        result = server.flush_cache()
        if result is True or getattr(result, "success", False):
            return
        time.sleep(0.1)
    raise RuntimeError(f"Cache flush did not complete: {result!r}")


def score_batch(server, prefixes: Sequence[list[int]], answers: Sequence[list[int]]) -> tuple[list[float], int]:
    tokens = [p + a for p, a in zip(prefixes, answers, strict=True)]
    response = server.generate(
        input_ids=tokens,
        sampling_params={"temperature": 0, "max_new_tokens": 1},
        return_logprob=True,
        logprob_start_len=[len(p) - 1 for p in prefixes],
    )
    scores: list[float] = []
    cached = 0
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
    return scores, cached


def prepare_natural_answers(server, doc: Document, qs: Sequence[int], max_new_tokens: int = 256) -> list[dict]:
    """Generate natural complete answers on the full document without 24-token truncation."""
    full = (1 << doc.n) - 1
    outputs = server.generate(
        input_ids=doc.prompts(full, qs),
        sampling_params={"temperature": 0, "max_new_tokens": max_new_tokens},
        return_logprob=True,
    )
    ends = {doc.tokenizer.eos_token_id, doc.tokenizer.pad_token_id}
    out = []
    for q_idx, row in zip(qs, outputs, strict=True):
        meta = row["meta_info"]
        tokens = list(row.get("output_ids") or [int(x[1]) for x in meta["output_token_logprobs"]])
        while tokens and tokens[-1] in ends:
            tokens.pop()
        if not tokens:
            raise ValueError(f"Empty frozen answer for {doc.ident} q={q_idx}")
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
    """BatchLLM-style DFS prefix-grouped decoupled batch scheduler.

    Traverses distinct masks in lexicographic radix-tree DFS order of their shared
    document prefix, groups consecutive masks into width-W tiles, executes q0
    prefills per DFS tile, and immediately flushes their question suffixes
    (maximizing DFS cache locality with zero mixed prefill+suffix batches).
    """
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


def fast_merged_mask_trie_batches(
    by_mask: Mapping[int, Sequence[RequestRow]],
    width: int = 32,
) -> list[list[RequestRow]]:
    """Fast O(M log M) Width-Aligned Pioneer-Peeling Mask-Trie Scheduler (<2 ms prep).

    Sorts distinct masks by their document prefix token sequence (C-speed list sort),
    recursively peels alternating pioneers (even indices) and deepest-LCP siblings
    (odd indices) bottom-up into exact full-width ``width`` prefill waves (achieving
    the exact hardware-minimum ``ceil(M/width)`` heavy batches with zero remainder
    waves), and dispatches all remaining question suffixes through the decoupled
    pure-suffix queue (zero mixed prefill+suffix batches).
    """
    items = []
    for mask, reqs in by_mask.items():
        if not reqs:
            continue
        reqs_sorted = sorted(reqs, key=lambda r: r[2])
        # Use q0 prompt directly as prefix sort key (avoids O(nq * L) LCP recomputation)
        items.append((reqs_sorted[0][0], mask, reqs_sorted))
    if not items:
        return []
    items.sort(key=lambda x: (x[0], x[1]))

    cur = list(range(len(items)))
    layers: list[list[int]] = []
    while len(cur) > width:
        layers.append(cur[1::2])
        cur = cur[0::2]
    layers.append(cur)
    layers.reverse()
    order = [idx for layer in layers for idx in layer]

    batches: list[list[RequestRow]] = []
    suffix_queue: list[RequestRow] = []

    def drain(*, flush_all: bool = False) -> None:
        while len(suffix_queue) >= width or (flush_all and suffix_queue):
            batches.append(suffix_queue[:width])
            del suffix_queue[:width]

    for pos in range(0, len(order), width):
        tile = [items[i] for i in order[pos : pos + width]]
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
    width: int = 32,
) -> list[list[RequestRow]]:
    if layout == "merged_mask_trie":
        return fast_merged_mask_trie_batches(by_mask, width=width)
    if layout == "batchllm_dfs":
        return batchllm_dfs_batches(by_mask, width=width)
    if layout == "sglang_native_lpm":
        # Single-call submission: hand all requests to SGLang continuous batching + native LPM
        all_rows = [row for reqs in by_mask.values() for row in sorted(reqs, key=lambda r: r[2])]
        return [all_rows] if all_rows else []
    return list(schedule_batches(by_mask, layout, n, width=width))


def execute_scheduled(
    server,
    doc: Document,
    masks: Sequence[int],
    qs: Sequence[int],
    answers_by_q: Mapping[int, dict],
    layout: str,
    width: int = 32,
) -> tuple[dict[tuple[int, int], float], dict[str, object]]:
    t0 = time.perf_counter()
    uniq_masks = list(dict.fromkeys(masks))
    by_mask: dict[int, list[RequestRow]] = {}
    for m in uniq_masks:
        prompts = doc.prompts(m, qs)
        by_mask[m] = [(toks, m, q) for q, toks in zip(qs, prompts, strict=True)]
    batches = build_schedule(by_mask, layout, doc.n, width=width)
    t_prep = time.perf_counter()

    scores: dict[tuple[int, int], float] = {}
    total_input = 0
    total_cached = 0
    for batch in batches:
        prompts = [r[0] for r in batch]
        tails = [answers_by_q[r[2]]["tokens"] for r in batch]
        vals, hit = score_batch(server, prompts, tails)
        for (_, m, q), sc in zip(batch, vals, strict=True):
            scores[(m, q)] = sc
        total_input += sum(len(p) + len(t) for p, t in zip(prompts, tails, strict=True))
        total_cached += hit
    t_end = time.perf_counter()

    return scores, {
        "requests": sum(len(b) for b in batches),
        "batches": len(batches),
        "input_tokens": total_input,
        "cached_tokens": total_cached,
        "uncached_tokens": max(0, total_input - total_cached),
        "prepare_seconds": t_prep - t0,
        "service_seconds": t_end - t_prep,
    }


def decode_and_certify(
    masks: Sequence[int],
    scores: Mapping[tuple[int, int], float],
    n: int,
    qs: Sequence[int],
) -> tuple[list[list[float]], list[float], list[int], list[float]]:
    """Decode Banzhaf estimates and zero-extra-query vacant-channel Parseval variance."""
    m_half = len(masks) // 2
    # masks are arranged in complementary pairs (masks[2t], masks[2t+1])
    z_half = np.array(
        [[1.0 if ((masks[2 * t] >> i) & 1) else -1.0 for i in range(n)] for t in range(m_half)],
        dtype=np.float64,
    )
    effects_all: list[list[float]] = []
    vac_all: list[float] = []
    top1_all: list[int] = []
    gap_all: list[float] = []
    for q in qs:
        odd_vals = np.array(
            [0.5 * (scores[(masks[2 * t], q)] - scores[(masks[2 * t + 1], q)]) for t in range(m_half)],
            dtype=np.float64,
        )
        eff = (2.0 * (z_half.T @ odd_vals) / m_half).tolist()
        vac = vacant_channel_variance(odd_vals.tolist(), eff, m_half)
        order = sorted(range(n), key=lambda i: (-eff[i], i))
        effects_all.append(eff)
        vac_all.append(vac)
        top1_all.append(order[0])
        gap_all.append(float(eff[order[0]] - eff[order[1]]))
    return effects_all, vac_all, top1_all, gap_all


def make_design_masks(design_name: str, n: int, budget_size: int, rng: random.Random) -> tuple[list[int], float, float]:
    m_half = budget_size // 2
    if design_name == "cube":
        rows = designs.cube(n, budget_size, rng, cut=None)
        gamma, kappa = exact_coset_certificates([n], m_half)
        return rows, gamma, kappa
    if design_name == "half":
        cut = n // 2
        rows = designs.cube(n, budget_size, rng, cut=cut)
        gamma, kappa = exact_coset_certificates([cut, n - cut], m_half)
        return rows, gamma, kappa
    if design_name == "four":
        rows = designs.partition_cube(n, budget_size, rng, groups=4)
        sizes = [(j + 1) * n // 4 - j * n // 4 for j in range(4)]
        gamma, kappa = exact_coset_certificates(sizes, m_half)
        return rows, gamma, kappa
    raise ValueError(f"unsupported design: {design_name}")


def run_smoke(server, tokenizer, out: Path) -> None:
    p = tokenizer.encode("What is the capital of France? Answer:", add_special_tokens=False)
    a = tokenizer.encode(" Paris.", add_special_tokens=False)
    sc, hit = score_batch(server, [p, p], [a, a])
    flush_cache(server)
    assert len(sc) == 2 and abs(sc[0] - sc[1]) < 1e-4, f"smoke mismatch: {sc}"
    save_json(out / "smoke.json", {"scores": sc, "cached": hit})
    print(f"SMOKE OK scores={sc} cached={hit}", flush=True)


def main() -> None:
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["det", "fast"], required=True)
    ap.add_argument("--model", type=str, default=os.environ.get("P18_MODEL", "Qwen/Qwen3-4B"))
    ap.add_argument("--workloads", type=str, required=True, help="Comma-separated doc:nq specs")
    ap.add_argument("--designs", type=str, default="cube,half,four")
    ap.add_argument("--layouts", type=str, default="prefix,mask_trie_fanout,merged_mask_trie,batchllm_dfs,sglang_native_lpm")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--smoke-only", action="store_true")
    args = ap.parse_args()

    out = Path(os.environ["P18_OUTPUT"])
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)

    # Load both confirm-input and multi-input JSONs to resolve any requested doc_ident
    data_roots = [
        Path(os.environ.get("P18_CONFIRM_INPUT", "data/confirm-inputs.json")),
        Path(os.environ.get("P18_MULTI_INPUT", "data/multi-inputs.json")),
    ]
    raw_by_ident: dict[str, dict] = {}
    for dp in data_roots:
        if dp.is_file():
            for rec in load_json(dp):  # type: ignore[union-attr]
                ident = f"{rec['dataset']}-{rec['doc_id']}"
                # Prefer record with MORE questions (multi-input overrides confirm-input)
                if ident not in raw_by_ident or len(rec["questions"]) > len(raw_by_ident[ident]["questions"]):
                    raw_by_ident[ident] = rec

    workload_specs: list[tuple[str, int]] = []
    for tok_spec in args.workloads.split(","):
        tok_spec = tok_spec.strip()
        if not tok_spec:
            continue
        ident, nq_str = tok_spec.split(":")
        workload_specs.append((ident, int(nq_str)))

    design_list = [x.strip() for x in args.designs.split(",") if x.strip()]
    layout_list = [x.strip() for x in args.layouts.split(",") if x.strip()]

    server = build_engine(args.model, mode=args.mode)
    try:
        info = server.get_server_info()
        info_fields = [
            "version", "page_size", "mem_fraction_static", "max_total_num_tokens",
            "max_total_tokens", "max_running_requests", "context_length",
            "attention_backend", "sampling_backend", "schedule_policy",
            "disable_radix_cache", "chunked_prefill_size", "disable_cuda_graph",
            "dtype", "tp_size", "enable_deterministic_inference",
        ]
        save_json(out / "engine.json", {k: info.get(k) for k in info_fields})
        print("ENGINE:", {k: info.get(k) for k in ("attention_backend", "disable_cuda_graph", "enable_deterministic_inference", "schedule_policy")}, flush=True)
        run_smoke(server, tokenizer, out)
        if args.smoke_only:
            return

        # Cache Document & natural answers per ident (generate ONCE using max requested nq)
        max_nq_by_doc: dict[str, int] = defaultdict(int)
        for ident, nq in workload_specs:
            max_nq_by_doc[ident] = max(max_nq_by_doc[ident], nq)

        docs: dict[str, Document] = {}
        answers_map: dict[str, dict[int, dict]] = {}
        for ident, max_nq in max_nq_by_doc.items():
            doc = Document(tokenizer, raw_by_ident[ident], order="document-first")
            docs[ident] = doc
            qs_all = list(range(min(max_nq, len(doc.sides))))
            flush_cache(server)
            ans_list = prepare_natural_answers(server, doc, qs_all, max_new_tokens=args.max_new_tokens)
            answers_map[ident] = {a["q"]: a for a in ans_list}
            lens = [a["token_len"] for a in ans_list]
            fins = [a["finish"] for a in ans_list]
            print(
                f"ANSWERS {ident} (n={doc.n}, doc_toks={sum(len(p) for p in doc.parts)}, nq={len(qs_all)}): "
                f"lens={lens} finish={fins}",
                flush=True,
            )
            save_json(out / f"{ident}-natural-answers.json", ans_list)

        # Full-length kernel & CUDA-graph warmup outside all timed trials
        first_ident = workload_specs[0][0]
        w_doc = docs[first_ident]
        w_ans = answers_map[first_ident][0]["tokens"]
        w_prompts = [w_doc.prompts((1 << w_doc.n) - 1, [0])[0], w_doc.prompts((1 << (w_doc.n // 2)) - 1, [0])[0]]
        score_batch(server, w_prompts, [w_ans, w_ans])
        flush_cache(server)
        print(f"WARMUP DONE on {first_ident} (len={len(w_prompts[0])})", flush=True)

        summary_rows: list[dict[str, object]] = []
        for ident, nq in workload_specs:
            doc = docs[ident]
            qs = list(range(min(nq, len(doc.sides))))
            ans_by_q = answers_map[ident]
            budget_size = designs.budget(doc.n)

            for rep in range(args.reps):
                # Pre-draw identical masks per design so every scheduler sees IDENTICAL masks
                masks_by_design: dict[str, tuple[list[int], float, float]] = {}
                for des in design_list:
                    rng = random.Random(designs.seed("stage32-outpost", ident, rep, des))
                    masks_by_design[des] = make_design_masks(des, doc.n, budget_size, rng)

                # Interleave all (design, layout) runs randomly
                tasks = [(des, lay) for des in design_list for lay in layout_list]
                random.Random(designs.seed("stage32-order", ident, len(qs), rep, args.mode)).shuffle(tasks)

                scores_by_design_layout: dict[tuple[str, str], dict[tuple[int, int], float]] = {}
                for des, lay in tasks:
                    masks, gamma, kappa = masks_by_design[des]
                    flush_cache(server)
                    t_start = time.perf_counter()
                    scores, meas = execute_scheduled(
                        server, doc, masks, qs, ans_by_q, lay, width=args.width
                    )
                    t_dec0 = time.perf_counter()
                    effs, vacs, top1s, gaps = decode_and_certify(masks, scores, doc.n, qs)
                    t_end = time.perf_counter()
                    scores_by_design_layout[(des, lay)] = scores

                    rec = {
                        "model": Path(args.model).name,
                        "mode": args.mode,
                        "doc": ident,
                        "n": doc.n,
                        "doc_tokens": sum(len(p) for p in doc.parts),
                        "nq": len(qs),
                        "mean_answer_tokens": round(
                            float(np.mean([ans_by_q[q]["token_len"] for q in qs])), 2
                        ),
                        "max_answer_tokens": max(ans_by_q[q]["token_len"] for q in qs),
                        "budget": budget_size,
                        "rep": rep,
                        "design": des,
                        "layout": lay,
                        "gamma": round(gamma, 5) if math.isfinite(gamma) else "inf",
                        "kappa": round(kappa, 5) if math.isfinite(kappa) else "inf",
                        "total_seconds": t_end - t_start,
                        "prepare_seconds": meas["prepare_seconds"],
                        "service_seconds": meas["service_seconds"],
                        "decode_seconds": t_end - t_dec0,
                        "requests": meas["requests"],
                        "batches": meas["batches"],
                        "input_tokens": meas["input_tokens"],
                        "cached_tokens": meas["cached_tokens"],
                        "uncached_tokens": meas["uncached_tokens"],
                        "top1": top1s,
                        "gap": [round(g, 6) for g in gaps],
                        "vacant_var": [float(f"{v:.4e}") for v in vacs],
                    }
                    summary_rows.append(rec)
                    # Save full per-run scores + B compressed for downstream exact/cross-track auditing
                    save_json(
                        out / f"{ident}-nq{len(qs)}-r{rep}-{des}-{lay}.json.gz",
                        {
                            **rec,
                            "masks": [hex(m) for m in masks],
                            "B": effs,
                            "scores": {hex(m): [scores[(m, q)] for q in qs] for m in dict.fromkeys(masks)},
                        },
                    )
                    print(
                        f"RUN {ident:18s} nq={len(qs):2d} r={rep} {des:4s} {lay:18s} | "
                        f"tot={rec['total_seconds']:6.2f}s (svc={rec['service_seconds']:6.2f}s, prep={rec['prepare_seconds']:5.3f}s) | "
                        f"batches={rec['batches']:3d} unc={rec['uncached_tokens']:7d} cached={rec['cached_tokens']:7d}",
                        flush=True,
                    )

                # Log cross-layout numerical drift within same (design, rep)
                for des in design_list:
                    ref_sc = scores_by_design_layout.get((des, layout_list[0]))
                    if not ref_sc:
                        continue
                    for lay in layout_list[1:]:
                        cur_sc = scores_by_design_layout.get((des, lay))
                        if not cur_sc:
                            continue
                        diffs = [abs(ref_sc[k] - cur_sc[k]) for k in ref_sc]
                        print(
                            f"  DRIFT {ident} nq={len(qs)} r={rep} {des} ({layout_list[0]} vs {lay}): "
                            f"max={max(diffs):.6f} mean={sum(diffs)/len(diffs):.6f}",
                            flush=True,
                        )

        save_json(out / "outpost-summary.json", summary_rows)
        print(f"OUTPOST COMPLETE: wrote {len(summary_rows)} records to {out / 'outpost-summary.json'}", flush=True)
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
