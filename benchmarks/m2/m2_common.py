"""Shared SGLang execution, tokenization, scheduling, and provenance helpers for M2."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import sys
import time
from itertools import chain
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

# Ensure coattr_v2 is importable regardless of working directory
THIS_DIR = Path(__file__).resolve().parent
for cand in (THIS_DIR, THIS_DIR.parent):
    if (cand / "coattr_v2").is_dir() and str(cand) not in sys.path:
        sys.path.insert(0, str(cand))

from coattr_v2.capacity_scheduler_v2 import (  # noqa: E402
    RequestRow,
    adaptive_capacity_mask_trie_batches,
)
from coattr_v2.safe_selector_v2 import vacant_channel_variance  # noqa: E402

SYSTEM = (
    "Answer the question using the provided document. Be concise. "
    "For multiple choice return the option letter."
)
SLOT = "p18_document_slot_12c925"


def stable_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("/".join(map(str, parts)).encode()).digest()[:8], "big")


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    data = (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    tmp.write_bytes(gzip.compress(data, mtime=0) if path.suffix == ".gz" else data)
    tmp.replace(path)


def load_json(path: Path) -> object:
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


class Document:
    """Identical chat-template and token-concatenation semantics as Stage 3.2."""

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

    def full_context_chunk_spans(self, q: int) -> tuple[list[int], list[tuple[int, int]]]:
        """Return full-context prompt token IDs and exact half-open (start, end) token span per chunk."""
        _, _, head, tail = self.sides[q]
        spans: list[tuple[int, int]] = []
        pos = len(head)
        body: list[int] = []
        for seg_toks in self.parts:
            spans.append((pos, pos + len(seg_toks)))
            body.extend(seg_toks)
            pos += len(seg_toks)
        return head + body + tail, spans


def load_all_m2_inputs(extra_paths: Sequence[Path] = ()) -> dict[str, dict]:
    raw_by_ident: dict[str, dict] = {}
    default_paths = [
        Path("./data"),
        Path("./data"),
        Path("./data"),
        Path("./data"),
        Path("./data"),
    ]
    for dp in [*default_paths, *extra_paths]:
        if dp.is_file():
            for rec in load_json(dp):  # type: ignore[union-attr]
                ident = f"{rec['dataset']}-{rec['doc_id']}"
                if ident not in raw_by_ident or len(rec["questions"]) > len(raw_by_ident[ident]["questions"]):
                    raw_by_ident[ident] = rec
    return raw_by_ident


def build_sglang_engine(
    model_path: str,
    *,
    context_length: int = 16384,
    max_running_requests: int = 128,
    mem_fraction_static: float = 0.72,
    max_total_tokens: int | None = None,
    port_offset: int = 0,
):
    import sglang as sgl

    job_id = int(os.environ.get("SLURM_JOB_ID", "100"))
    port = 24000 + ((job_id * 37 + port_offset * 13) % 16000)
    kwargs: dict[str, object] = dict(
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
    if max_total_tokens is not None and max_total_tokens > 0:
        kwargs["max_total_tokens"] = int(max_total_tokens)
    return sgl.Engine(**kwargs)


def flush_cache(server) -> None:
    result = None
    for _ in range(50):
        result = server.flush_cache()
        if result is True or getattr(result, "success", False):
            return
        time.sleep(0.1)
    raise RuntimeError(f"Cache flush did not complete: {result!r}")


def score_batch_sglang(
    server,
    prefixes: Sequence[list[int]],
    answers: Sequence[list[int]],
) -> tuple[list[float], int]:
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
            raise ValueError("Answer logprob alignment failed")
        probs = [math.exp(float(row[0])) for row in rows]
        scores.append(sum(probs) / len(probs))
        cached += int(meta.get("cached_tokens", 0))
    return scores, cached


def prepare_natural_answers_sglang(server, doc: Document, qs: Sequence[int], max_new_tokens: int = 256) -> list[dict]:
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


# ---------------------------------------------------------------------------
# Unified Scheduler Zoo (SGLang LPM, BatchLLM-DFS, Prefix Ablation-1,
# Global Trie, Capacity-Aware v1 [fixed W=32], Capacity-Aware v2 [adaptive W'])
# ---------------------------------------------------------------------------


def _prefix_2wave_batches(by_mask: Mapping[int, Sequence[RequestRow]], width: int = 32) -> list[list[RequestRow]]:
    """CoAttr Ablation-1: Paired-prefix decoupled 2-wave scheduler (Stage 2.2 baseline)."""
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

    for pos in range(0, len(items), 2 * width):
        chunk = items[pos : pos + 2 * width]
        pioneers = chunk[0::2]
        followers = chunk[1::2]
        for grp in (pioneers, followers):
            if not grp:
                continue
            batches.append([reqs[0] for _, _, reqs in grp])
            max_q = max(len(reqs) for _, _, reqs in grp)
            for q_idx in range(1, max_q):
                for _, _, reqs in grp:
                    if q_idx < len(reqs):
                        suffix_queue.append(reqs[q_idx])
            drain(flush_all=False)
    drain(flush_all=True)
    return batches


def _batchllm_dfs_batches(by_mask: Mapping[int, Sequence[RequestRow]], width: int = 32) -> list[list[RequestRow]]:
    """Client-side BatchLLM-DFS adaptation (lexicographic DFS tiles of width W + immediate suffix drain)."""
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


def build_m2_schedule(
    by_mask: Mapping[int, Sequence[RequestRow]],
    layout: str,
    *,
    width: int = 32,
    kv_token_capacity: int | None = None,
    page_size: int = 64,
    min_width: int = 4,
) -> tuple[list[list[RequestRow]], dict[str, object]]:
    if layout == "capacity_aware_v2":
        plan = adaptive_capacity_mask_trie_batches(
            by_mask,
            nominal_width=width,
            kv_token_capacity=kv_token_capacity,
            page_size=page_size,
            min_width=min_width,
        )
        return plan.batches, {
            "effective_width": plan.effective_width,
            "tile_masks": plan.max_tile_size,
            "tile_sizes": list(plan.tile_sizes),
            "predicted_peak_kv": plan.max_predicted_peak_kv,
            "predicted_delta_wave": plan.max_predicted_delta_wave,
            "width_contracted": plan.width_contracted,
        }
    if layout == "capacity_aware_v1":
        # Fixed nominal width W=32 without adaptive W' contraction (Stage 3.2 behavior)
        plan = adaptive_capacity_mask_trie_batches(
            by_mask,
            nominal_width=width,
            kv_token_capacity=kv_token_capacity,
            page_size=page_size,
            min_width=width,
        )
        return plan.batches, {
            "effective_width": plan.effective_width,
            "tile_masks": plan.max_tile_size,
            "tile_sizes": list(plan.tile_sizes),
            "predicted_peak_kv": plan.max_predicted_peak_kv,
            "predicted_delta_wave": plan.max_predicted_delta_wave,
            "width_contracted": False,
        }
    if layout == "merged_mask_trie":
        plan = adaptive_capacity_mask_trie_batches(
            by_mask, nominal_width=width, kv_token_capacity=None, page_size=page_size
        )
        return plan.batches, {"effective_width": width, "tile_masks": len(by_mask), "width_contracted": False}
    if layout == "prefix":
        return _prefix_2wave_batches(by_mask, width=width), {"effective_width": width, "tile_masks": min(2 * width, len(by_mask)), "width_contracted": False}
    if layout == "batchllm_dfs":
        return _batchllm_dfs_batches(by_mask, width=width), {"effective_width": width, "tile_masks": min(width, len(by_mask)), "width_contracted": False}
    if layout in ("sglang_native_lpm", "vllm_native_all"):
        all_rows = [row for reqs in by_mask.values() for row in sorted(reqs, key=lambda r: r[2])]
        return ([all_rows] if all_rows else []), {"effective_width": len(all_rows), "tile_masks": len(by_mask), "width_contracted": False}
    raise ValueError(f"unknown layout {layout}")


def execute_scheduled_sglang(
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
    min_width: int = 4,
) -> tuple[dict[tuple[int, int], float], dict[str, object]]:
    t0 = time.perf_counter()
    uniq_masks = list(dict.fromkeys(masks))
    by_mask: dict[int, list[RequestRow]] = {}
    for m in uniq_masks:
        prompts = doc.prompts(m, qs)
        by_mask[m] = [(toks, m, q) for q, toks in zip(qs, prompts, strict=True)]
    batches, sched_meta = build_m2_schedule(
        by_mask, layout, width=width, kv_token_capacity=kv_token_capacity, page_size=page_size, min_width=min_width
    )
    t_prep = time.perf_counter()

    scores: dict[tuple[int, int], float] = {}
    total_input = 0
    total_cached = 0
    for batch in batches:
        prompts = [r[0] for r in batch]
        tails = [answers_by_q[r[2]]["tokens"] for r in batch]
        vals, hit = score_batch_sglang(server, prompts, tails)
        for (_, m, q), sc in zip(batch, vals, strict=True):
            scores[(m, q)] = sc
        total_input += sum(len(p) + len(t) for p, t in zip(prompts, tails, strict=True))
        total_cached += hit
    t_end = time.perf_counter()

    return scores, {
        "requests": sum(len(b) for b in batches),
        "batches": len(batches),
        **sched_meta,
        "input_tokens": total_input,
        "cached_tokens": total_cached,
        "uncached_tokens": max(0, total_input - total_cached),
        "prepare_seconds": t_prep - t0,
        "service_seconds": t_end - t_prep,
    }


def decode_balanced_block_m2(
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
