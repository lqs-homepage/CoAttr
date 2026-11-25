"""Build deterministic real multi-hop RAG attribution workloads from MuSiQue validation split.

Creates ``inputs-rag.json`` in the exact schema consumed by ``main_suite.py``:
- 4 unmodified real multi-hop MuSiQue retrieved-corpus packs:
    * MuSiQue-Pack40A (n=40, 3 multi-hop Qs, 2M=128)
    * MuSiQue-Pack48B (n=48, 4 multi-hop Qs, 2M=128)
    * MuSiQue-Pack72C (n=72, 6 multi-hop Qs, 2M=256)
    * MuSiQue-Pack88D (n=88, 8 multi-hop Qs, 2M=256)
- 1 controlled redundancy-stress pack (explicitly flagged ``stress_test=True``)
  where each question's first gold supporting passage has a near-duplicate
  corroborating passage from a second source in the retrieved context, directly
  testing LOO / ARC-JSD blindness vs Uniform Banzhaf under redundant evidence:
    * MuSiQue-Redundant24 (n=24, 4 multi-hop Qs, 2M=64)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _fmt_para(p: dict) -> str:
    title = (p.get("title") or "").strip()
    text = (p.get("paragraph_text") or "").strip()
    return f"[{title}] {text}\n\n" if title else f"{text}\n\n"


def build_pack(
    doc_id: str,
    rows: list[dict],
    target_n: int,
    *,
    duplicate_first_support: bool = False,
) -> dict:
    """Assemble an ordered retrieved-paragraph RAG pack of exact size ``target_n``
    preserving exact gold ``supporting_segments`` indices for every question."""
    segments: list[str] = []
    seen_key: dict[tuple[str, str], int] = {}
    questions: list[str] = []
    gold_answers: list[str] = []
    supporting_segments: list[list[int]] = []

    # First insert all supporting paragraphs of the selected questions, interleaved with
    # their native distractor paragraphs in original MuSiQue order.
    # Step 1: collect all supporting paragraphs first so every question's evidence is guaranteed present
    supp_keys_per_q: list[list[tuple[str, str]]] = []
    distractor_pool: list[tuple[str, str, str]] = []  # (title, text, formatted)

    for row in rows:
        questions.append(row["question"])
        gold_answers.append(row["answer"])
        q_supps = []
        for p in row["paragraphs"]:
            key = (p.get("title", ""), p.get("paragraph_text", ""))
            if p.get("is_supporting"):
                q_supps.append(key)
                if key not in seen_key:
                    seen_key[key] = -1  # mark as required supporting
            else:
                distractor_pool.append((key[0], key[1], _fmt_para(p)))
        supp_keys_per_q.append(q_supps)

    # Interleave supporting & distractor paragraphs deterministically across document positions
    # (so gold supporting evidence is distributed naturally across early/middle/late positions!)
    ordered_entries: list[tuple[tuple[str, str], str, bool]] = []
    added: set[tuple[str, str]] = set()
    # Collect unique supporting entries in question order
    supp_entries: list[tuple[tuple[str, str], str]] = []
    for row in rows:
        for p in row["paragraphs"]:
            if p.get("is_supporting"):
                key = (p.get("title", ""), p.get("paragraph_text", ""))
                if key not in added:
                    added.add(key)
                    supp_entries.append((key, _fmt_para(p)))

    if duplicate_first_support:
        # Controlled redundancy stress test: add a second corroborating source echoing the
        # first supporting paragraph of each question (explicitly tagged as corroborating duplicate)
        dup_entries: list[tuple[tuple[str, str], str, tuple[str, str]]] = []
        for q_idx, q_supps in enumerate(supp_keys_per_q):
            if q_supps:
                orig_key = q_supps[0]
                dup_key = (orig_key[0] + f" (Mirror Source {q_idx + 1})", orig_key[1])
                dup_text = f"[{dup_key[0]}] {orig_key[1]}\n\n"
                dup_entries.append((dup_key, dup_text, orig_key))
        # Append duplicates as separate segments
        extra_supp_map: dict[tuple[str, str], tuple[str, str]] = {}
        for dk, dt, ok in dup_entries:
            supp_entries.append((dk, dt))
            extra_supp_map[ok] = dk
    else:
        extra_supp_map = {}

    dist_entries: list[tuple[tuple[str, str], str]] = []
    for t, tx, fm in distractor_pool:
        key = (t, tx)
        if key not in added:
            added.add(key)
            dist_entries.append((key, fm))

    n_supp = len(supp_entries)
    n_dist_needed = max(0, target_n - n_supp)
    dist_selected = dist_entries[:n_dist_needed]

    # Spread supporting paragraphs evenly among distractors so evidence spans full document
    final_items: list[tuple[tuple[str, str], str]] = []
    total = n_supp + len(dist_selected)
    supp_positions = {round((i + 0.5) * total / n_supp) for i in range(n_supp)} if n_supp > 0 else set()
    # Ensure exact count of supp_positions == n_supp
    pos_list = sorted(supp_positions)
    while len(pos_list) < n_supp:
        for cand in range(total):
            if cand not in pos_list:
                pos_list.append(cand)
                break
        pos_list.sort()
    supp_pos_set = set(pos_list[:n_supp])

    si = di = 0
    for idx in range(total):
        if idx in supp_pos_set and si < n_supp:
            final_items.append(supp_entries[si])
            si += 1
        elif di < len(dist_selected):
            final_items.append(dist_selected[di])
            di += 1
        elif si < n_supp:
            final_items.append(supp_entries[si])
            si += 1

    key_to_idx = {k: i for i, (k, _) in enumerate(final_items)}
    segments = [txt for _, txt in final_items]

    for q_supps in supp_keys_per_q:
        idxs = [key_to_idx[k] for k in q_supps if k in key_to_idx]
        if duplicate_first_support and q_supps and q_supps[0] in extra_supp_map:
            dk = extra_supp_map[q_supps[0]]
            if dk in key_to_idx:
                idxs.append(key_to_idx[dk])
        supporting_segments.append(sorted(set(idxs)))

    return {
        "dataset": "MuSiQue",
        "doc_id": doc_id,
        "split": "validation",
        "stress_test_redundancy": duplicate_first_support,
        "source_ids": [r["id"] for r in rows],
        "segments": segments,
        "questions": questions,
        "gold_answers": gold_answers,
        "supporting_segments": supporting_segments,
    }


def main() -> None:
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    rows = json.loads(src.read_text(encoding="utf-8"))
    # Select clean 2-hop / 3-hop / 4-hop questions with >=2 supporting paragraphs and non-empty text
    valid = [
        r for r in rows
        if r.get("answerable", True)
        and sum(1 for p in r["paragraphs"] if p.get("is_supporting")) >= 2
        and len(r["paragraphs"]) >= 15
    ]
    # Disjoint question slices across all 5 packs (zero question overlap!)
    packs = [
        build_pack("Pack40A", valid[0:3], target_n=40),
        build_pack("Pack48B", valid[3:7], target_n=48),
        build_pack("Pack72C", valid[7:13], target_n=72),
        build_pack("Pack88D", valid[13:21], target_n=88),
        build_pack("Redundant24", valid[21:25], target_n=24, duplicate_first_support=True),
    ]
    dst.write_text(json.dumps(packs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for p in packs:
        print(
            f"  {p['dataset']}-{p['doc_id']:12s}: n={len(p['segments']):2d} nq={len(p['questions'])} "
            f"chars={sum(len(s) for s in p['segments']):5d} supp={p['supporting_segments']} "
            f"redundancy={p['stress_test_redundancy']}"
        )


if __name__ == "__main__":
    main()
