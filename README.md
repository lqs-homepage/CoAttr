# CoAttr: Statistical–Physical Co-Execution for Ordered Context Attribution Queries

Official code, unit tests, benchmark runners, and archived evaluation data for **CoAttr** (*CoAttr: Statistical–Physical Co-Execution for Ordered Context Attribution Queries*).

**Authors:** Qiushi Li, Anbang Wu, Haolin Ma, Heng Lin, Ju Ren, Tianyu Wo, Chunming Hu  
*(Beihang University, Hohai University, Tsinghua University)*

---

## Overview

**CoAttr** compiles declarative ordered context attribution queries `ATTRIBUTION(C, Q; B, delta)` onto unmodified prefix-caching LLM serving engines (e.g., SGLang RadixAttention, vLLM APC) without altering frozen model weights or attention semantics:

1. **Logical Design Rewriter (`coattr/designs.py`, `coattr/theory_extensions.py`)**: Strength-3 ordered $g$-coset orthogonal arrays approaching the universal $(2/3, 6/7)$ prefix-rank frontier (Theorem 1), paired with $R$-block cyclic balanced rotation (Proposition 1).
2. **Cost-Based Safe Planner (`coattr/safe_selector.py`, `coattr/integration.py`)**: Millisecond zero-probe selector (`Regime A`, Algorithm 1) guarded by exact polynomial-DP total-MSE inflation ($\gamma_g \le 1.05$) and vacant-channel calibration ($\kappa_g \le 1.50$) with automatic CUBE fallback, zero-extra-query vacant-channel SNR certificates, and Parseval-tightened empirical Bernstein bounds (`Regime B`, Theorem 2).
3. **Capacity-Aware Physical Executor (`coattr/capacity_scheduler.py`, `coattr/scheduler.py`, `coattr/wave_merged_scheduler.py`)**: Decoupled width-aligned pioneer peeling and paged-LCP capacity-aware subtree tiling (Algorithm 2, Theorem 3) eliminating mixed prefill–suffix stalls and 32B-scale HBM LRU thrashing.

---

## Repository Structure

```text
├── coattr/            # Core optimizer, orthogonal coset designs, safe planner, capacity-aware scheduler, MSR baselines
├── benchmarks/        # Live SGLang GPU benchmark runners (Qwen3-4B/32B, Gemma-4-31B) & MuSiQue pack builder
├── tests/             # 23 deterministic offline unit tests covering Theorems 1-3, Proposition 1, and all modules
├── data/              # Archived MuSiQue input packs (25Q) and aggregated CSV/JSON tables across all 11 workloads (57Q)
└── scripts/           # Offline table verification and publication-figure reproduction scripts
```

---

## Quickstart (Offline Verification — No GPU Required)

All theoretical verifications, 23 unit tests, table summaries (Tables 1–3), and experimental figures (Figures 5, 7, 8, 9) run locally in seconds using [`uv`](https://docs.astral.sh/uv/) (Python 3.11+):

```bash
# 1. Run all 23 unit tests (designs, DP certificates, Parseval bounds, cyclic rotation, capacity tiling, MSR parity)
uv run --with pytest --with numpy --with scikit-learn pytest -v

# 2. Verify and print Tables 1-3 directly from archived benchmark data
uv run scripts/verify_tables.py

# 3. Reproduce Figures 5, 7, 8, and 9 (PDF + 300-DPI PNG into figures/)
uv run --with numpy --with matplotlib scripts/reproduce_figures.py
```

---

## Running Live GPU Benchmarks (SGLang)

The live benchmarks in `benchmarks/` run against an unmodified [SGLang](https://github.com/sgl-project/sglang) server with RadixAttention enabled:

```bash
# Build MuSiQue organic + controlled redundancy retrieval packs (already archived in data/inputs-rag.json)
uv run benchmarks/build_rag_inputs.py --out data/inputs-rag.json

# Cross-model scheduler & HBM KV capacity ablation (Qwen3-4B / Qwen3-32B / gemma-4-31B-it)
uv run benchmarks/outpost_e1_e2.py --model Qwen/Qwen3-32B --help

# Full 11-workload, 57-query, 15-estimator evaluation suite
uv run benchmarks/main_suite.py --model Qwen/Qwen3-4B --help
```

---

## License

Released under the [MIT License](LICENSE).
