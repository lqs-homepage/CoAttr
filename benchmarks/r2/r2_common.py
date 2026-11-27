"""Shared fast document handling, verified SGLang scoring with KV cache counters,
pure-NumPy ContextCite/LASSO surrogate solver, and HuggingFace AttnTrace helper
for Step-2 full-scale B200 Root-Elevation experiments."""

from __future__ import annotations

import math
import random
import sys
import time
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import Sequence

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
UPGRADE_DIR = THIS_DIR.parent
for p in (str(UPGRADE_DIR), str(UPGRADE_DIR / "m2_bench")):
    if p not in sys.path:
        sys.path.insert(0, p)

from coattr_v2.capacity_scheduler_v2 import RequestRow, adaptive_capacity_mask_trie_batches  # noqa: E402
import m2_common  # noqa: E402


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-min(x, 60.0)))
    z = math.exp(max(x, -60.0))
    return z / (1.0 + z)


class FastDocument:
    """Identical chat-template and token-concatenation semantics as m2_common.Document,
    optimized for ultra-long documents (n=200..420, 20K..48K tokens)."""

    def __init__(self, tokenizer, raw: dict, order: str = "document-first") -> None:
        self.raw = raw
        self.n = len(raw["segments"])
        self.ident = f"{raw['dataset']}-{raw['doc_id']}"
        self.tokenizer = tokenizer
        self.parts = [self.encode(p) for p in raw["segments"]]
        self.sides = []
        for q in raw["questions"]:
            question = "Question: " + q.strip()
            content = (
                "Context:\n\n" + m2_common.SLOT + question
                if order == "document-first"
                else question + "\n\nContext:\n\n" + m2_common.SLOT
            )
            chat = m2_common.Document._apply_template(tokenizer, content)
            left, right = chat.split(m2_common.SLOT)
            self.sides.append((left, right, self.encode(left), self.encode(right)))
        self.full_tokens = max(
            len(h) + sum(len(p) for p in self.parts) + len(t)
            for _, _, h, t in self.sides
        )

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def body(self, mask: int) -> list[int]:
        return list(chain.from_iterable(tokens for i, tokens in enumerate(self.parts) if (mask >> i) & 1))

    def prompt(self, mask: int, q: int) -> list[int]:
        _, _, head, tail = self.sides[q]
        return head + self.body(mask) + tail

    def full_context_chunk_spans(self, q: int) -> tuple[list[int], list[tuple[int, int]]]:
        _, _, head, tail = self.sides[q]
        spans: list[tuple[int, int]] = []
        pos = len(head)
        body: list[int] = []
        for seg in self.parts:
            spans.append((pos, pos + len(seg)))
            body.extend(seg)
            pos += len(seg)
        return head + body + tail, spans


def score_answer_batch_sglang(
    engine,
    doc: FastDocument,
    masks: Sequence[int],
    q_indices: Sequence[int],
    answer_token_ids_per_q: dict[int, list[int]],
    kv_cap: int,
    *,
    layout: str = "capacity_aware_v2",
    width: int = 32,
    page_size: int = 64,
    min_width: int = 4,
) -> tuple[dict[int, dict[int, float]], dict[str, object]]:
    """Teacher-forced geometric-mean probability scoring via unified scheduler zoo,
    returning scores and exact wall-clock + KV cache token accounting."""
    bm: dict[int, list[RequestRow]] = defaultdict(list)
    uniq_masks = list(dict.fromkeys(int(m) for m in masks))
    for m in uniq_masks:
        body = doc.body(m)
        for q in q_indices:
            _, _, head, tail = doc.sides[q]
            ans_ids = answer_token_ids_per_q[q]
            bm[m].append((head + body + tail + ans_ids, m, q))

    batches, sched_meta = m2_common.build_m2_schedule(
        bm,
        layout,
        width=width,
        kv_token_capacity=kv_cap,
        page_size=page_size,
        min_width=min_width,
    )
    t0 = time.perf_counter()
    scores: dict[int, dict[int, float]] = {q: {} for q in q_indices}
    total_input, total_cached = 0, 0
    for batch in batches:
        input_ids = [row[0] for row in batch]
        starts = [len(row[0]) - len(answer_token_ids_per_q[row[2]]) - 1 for row in batch]
        outs = engine.generate(
            input_ids=input_ids,
            sampling_params={"max_new_tokens": 1, "temperature": 0.0},
            return_logprob=True,
            logprob_start_len=starts,
        )
        for row, out in zip(batch, outs, strict=True):
            prompt, m, q = row
            ans_ids = answer_token_ids_per_q[q]
            a_len = len(ans_ids)
            meta = out["meta_info"]
            lp_list = meta["input_token_logprobs"][-a_len:]
            if [int(x[1]) for x in lp_list] != list(ans_ids) or any(x[0] is None for x in lp_list):
                raise ValueError(f"Answer logprob alignment failed for q={q}: {[x[1] for x in lp_list]} vs {ans_ids}")
            avg_lp = float(sum(float(x[0]) for x in lp_list) / a_len)
            scores[q][m] = math.exp(max(-30.0, avg_lp))
            total_input += len(prompt)
            total_cached += int(meta.get("cached_tokens", 0))
    wall = time.perf_counter() - t0
    stats = {
        "wall_s": wall,
        "requests": sum(len(b) for b in batches),
        "batches": len(batches),
        "input_tokens": total_input,
        "cached_tokens": total_cached,
        "uncached_tokens": max(0, total_input - total_cached),
        **sched_meta,
    }
    return scores, stats


def score_semfilter_logodds_batch(
    engine,
    doc: FastDocument,
    masks: Sequence[int],
    q_indices: Sequence[int],
    yes_ids: list[int],
    no_ids: list[int],
    kv_cap: int,
    *,
    temp: float = 4.0,
    layout: str = "capacity_aware_v2",
    width: int = 32,
    page_size: int = 64,
    min_width: int = 4,
) -> tuple[dict[int, dict[int, float]], dict[int, dict[int, float]], dict[str, object]]:
    """Score calibrated YES/NO log-odds utility sigma((lp_YES - lp_NO) / T) in [0, 1]
    supporting arbitrary tokenizer YES/NO lengths and exact KV token telemetry."""
    bm: dict[int, list[RequestRow]] = defaultdict(list)
    uniq_masks = list(dict.fromkeys(int(m) for m in masks))
    for m in uniq_masks:
        body = doc.body(m)
        for q in q_indices:
            _, _, head, tail = doc.sides[q]
            base = head + body + tail
            bm[m].append((base + yes_ids, m, 2 * q))
            bm[m].append((base + no_ids, m, 2 * q + 1))

    batches, sched_meta = m2_common.build_m2_schedule(
        bm,
        layout,
        width=width,
        kv_token_capacity=kv_cap,
        page_size=page_size,
        min_width=min_width,
    )
    t0 = time.perf_counter()
    raw_lp: dict[int, dict[int, float]] = defaultdict(dict)
    total_input, total_cached = 0, 0
    for batch in batches:
        input_ids = [row[0] for row in batch]
        starts = [len(row[0]) - (len(yes_ids) if (row[2] % 2 == 0) else len(no_ids)) - 1 for row in batch]
        outs = engine.generate(
            input_ids=input_ids,
            sampling_params={"max_new_tokens": 1, "temperature": 0.0},
            return_logprob=True,
            logprob_start_len=starts,
        )
        for row, out in zip(batch, outs, strict=True):
            prompt, m, qtag = row
            tgt = yes_ids if (qtag % 2 == 0) else no_ids
            meta = out["meta_info"]
            lp_list = meta["input_token_logprobs"][-len(tgt) :]
            if [int(x[1]) for x in lp_list] != list(tgt) or any(x[0] is None for x in lp_list):
                raise ValueError(f"Semfilter alignment failed: {[x[1] for x in lp_list]} vs {tgt}")
            raw_lp[qtag][m] = float(sum(float(x[0]) for x in lp_list))
            total_input += len(prompt)
            total_cached += int(meta.get("cached_tokens", 0))
    wall = time.perf_counter() - t0

    cal_scores: dict[int, dict[int, float]] = {q: {} for q in q_indices}
    raw_diffs: dict[int, dict[int, float]] = {q: {} for q in q_indices}
    for q in q_indices:
        for m in uniq_masks:
            diff = raw_lp[2 * q][m] - raw_lp[2 * q + 1][m]
            raw_diffs[q][m] = diff
            cal_scores[q][m] = sigmoid(diff / temp)

    stats = {
        "wall_s": wall,
        "requests": sum(len(b) for b in batches),
        "batches": len(batches),
        "input_tokens": total_input,
        "cached_tokens": total_cached,
        "uncached_tokens": max(0, total_input - total_cached),
        **sched_meta,
    }
    return cal_scores, raw_diffs, stats


def draw_contextcite_iid_masks(n: int, budget_b: int, rng: random.Random) -> list[int]:
    """Draw ``budget_b`` i.i.d. Bernoulli(1/2) ablation masks (ContextCite, NeurIPS'24)."""
    masks: list[int] = []
    for _ in range(budget_b):
        m = 0
        for i in range(n):
            if rng.random() < 0.5:
                m |= 1 << i
        masks.append(m)
    return masks


def fit_contextcite_lasso(
    masks: Sequence[int],
    y_vals: Sequence[float],
    n: int,
    *,
    alpha: float = 0.005,
    max_iter: int = 600,
    tol: float = 1e-6,
) -> np.ndarray:
    """Pure-NumPy coordinate-descent LASSO matching ContextCite's linear surrogate
    min_{w, b} (1/(2B)) ||X w + b - y||_2^2 + alpha ||w||_1 on binary {0,1} mask features."""
    B = len(masks)
    X = np.array([[(m >> i) & 1 for i in range(n)] for m in masks], dtype=np.float64)
    y = np.asarray(y_vals, dtype=np.float64)

    x_mean = X.mean(axis=0)
    y_mean = float(y.mean())
    Xc = X - x_mean
    yc = y - y_mean
    col_norm2 = np.sum(Xc * Xc, axis=0)

    w = np.zeros(n, dtype=np.float64)
    r = yc.copy()  # residual = yc - Xc @ w
    lam = float(alpha) * B

    for _ in range(max_iter):
        max_delta = 0.0
        for j in range(n):
            denom = col_norm2[j]
            if denom <= 1e-12:
                continue
            old = w[j]
            xj = Xc[:, j]
            rho = float(np.dot(xj, r)) + denom * old
            if rho > lam:
                new = (rho - lam) / denom
            elif rho < -lam:
                new = (rho + lam) / denom
            else:
                new = 0.0
            diff = new - old
            if diff != 0.0:
                w[j] = new
                r -= diff * xj
                if abs(diff) > max_delta:
                    max_delta = abs(diff)
        if max_delta < tol:
            break
    return w


def run_attntrace_cuad_hf(
    model_path: str,
    cuad_docs: Sequence[dict],
    active_qs_by_doc: dict[str, list[int]],
) -> dict[str, dict]:
    """Single-pass cross-layer AttnTrace extraction over standard CUAD contracts on HuggingFace
    (called AFTER SGLang engine.shutdown() so full GPU HBM is available)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from coattr_v2.baselines_v2 import attntrace_single_pass_scores

    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=True,
        trust_remote_code=True,
    ).to("cuda:0")
    model.eval()
    yes_id = tok.encode("YES", add_special_tokens=False)[0]
    out: dict[str, dict] = {}
    try:
        for rec in cuad_docs:
            did = rec["doc_id"]
            aqs = active_qs_by_doc.get(did)
            if not aqs:
                continue
            doc = FastDocument(tok, rec)
            t0 = time.perf_counter()
            scores_mat = np.zeros((len(aqs), doc.n), dtype=np.float64)
            for qi, q in enumerate(aqs):
                prompt_ids, spans = doc.full_context_chunk_spans(q)
                full_ids = prompt_ids + [yes_id]
                inp = torch.tensor([full_ids], dtype=torch.long, device="cuda:0")
                with torch.no_grad():
                    res = model(input_ids=inp, output_attentions=True, use_cache=False)
                # Extract attention from last prompt position predicting YES across all layers & heads
                ans_pos = len(prompt_ids) - 1
                layer_rows = []
                for attn_l in res.attentions:
                    # shape (1, H, S, S) -> take [0, :, ans_pos, :len(prompt_ids)]
                    layer_rows.append(attn_l[0, :, ans_pos, : len(prompt_ids)].float().cpu().numpy())
                attn_stack = np.stack(layer_rows, axis=0)  # (L, H, P)
                # Add singleton answer dimension (L, H, 1, P)
                sc = attntrace_single_pass_scores(attn_stack[:, :, None, :], spans)
                scores_mat[qi] = np.asarray(sc, dtype=np.float64)
                del res, inp
                torch.cuda.empty_cache()
            wall = time.perf_counter() - t0
            out[did] = {"wall_s": wall, "scores": scores_mat.tolist()}
            print(f"  [AttnTrace HF] {did} (n={doc.n}, nq={len(aqs)}): wall={wall:.2f}s", flush=True)
    finally:
        del model
        torch.cuda.empty_cache()
    return out
