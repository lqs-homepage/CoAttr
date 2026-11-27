# CoAttr: Statistical–Physical Co-Execution for Ordered Context Attribution Queries

Official code, unit tests, dual-engine benchmark runners (`SGLang` + `vLLM 0.34.0` APC), and archived evaluation data for **CoAttr** (*CoAttr: Statistical–Physical Co-Execution for Ordered Context Attribution Queries*).

**Authors:** Qiushi Li, Anbang Wu, Haolin Ma, Heng Lin, Ju Ren, Tianyu Wo, Chunming Hu  
*(Beihang University, Hohai University, Tsinghua University)*

---

## Overview

**CoAttr** compiles declarative ordered context attribution queries `ATTRIBUTION(C, Q; B, delta)` onto unmodified prefix-caching LLM serving engines (**SGLang RadixAttention** and **vLLM Automatic Prefix Caching**) without altering frozen model weights or attention semantics:

1. **Logical Design Rewriter (`coattr_v2/designs_v2.py`, `coattr/designs.py`)**:
   - Universal strength-3 prefix-rank lower bound and ordered dyadic $g$-cosets approaching the $(2/3, 6/7)$ prefix frontier (Theorem 1).
   - **Kronecker Ordered Paley–Hadamard Cosets** ($H_g \otimes H_L$, $L \equiv 4 \pmod 8$, Paley Type-I/II) that eliminate $2\times$ power-of-two budget cliffs (cutting prefix tokens by $36\%$--$59\%$) while deterministically driving single-block intra-coset 1-sparse blind spots to **0.00%** (Theorem 2).
   - Bidirectional degree-3 pairwise coupling analysis and all-$g$ multi-block cyclic balanced rotation ($g \in \{2, 4\}$, Proposition 1).
2. **Cost-Based Safe Planner (`coattr_v2/safe_selector_v2.py`, `coattr/safe_selector.py`)**:
   - Exact polynomial-DP total-MSE guard ($\gamma_g \le 1.05$) and vacant-channel calibration ($\kappa_g \le 1.50$) with automatic CUBE fallback ($<10$\,ms CPU).
   - **Sign-corrected unbiased active Mean-MSE Parseval estimator** $Z_D = \frac{M-n}{n}(V_r(\mathrm{CUBE}) - V_r(D))$ providing exact $n\times$ range tightening ($71\times$--$90\times$ on long documents) for streaming Empirical Bernstein certification (`Regime B`, Theorem 3).
   - **Closed-loop `Regime A`** (Algorithm 1): Wilson--Hilferty finite-DoF SNR alerting with lazy selective $R=2$ cyclic rotation reusing 100% of Block-1 evaluations.
3. **Capacity-Aware Physical Executor (`coattr_v2/capacity_scheduler_v2.py`, `coattr/capacity_scheduler.py`)**:
   - Decoupled width-aligned alternating pioneer peeling with explicit transient in-wave duplicate-prefix accounting ($\Delta_{\mathrm{wave}}$), closed-loop **adaptive inner wave-width contraction** ($W' \in \{32 \to 16 \to 8 \to 4\}$), and general $kW'$ paged-LCP subtree tiling (Algorithm 2, Theorem 4) supporting both SGLang (`page_size=64`) and vLLM APC (`block_size=16`).
4. **Baselines & Statistical Audit (`coattr_v2/baselines_v2.py`)**:
   - Faithful implementations of `BLIP-Pruning`, `Sketching-LOO`, `AttnTrace` single-pass attention extraction, `CUBE R=2`, question-weighted workload-cluster bootstrap 95% CIs, and paired sign-flip permutation tests.

---

## Repository Structure

```text
├── coattr_v2/         # Upgraded CoAttr v2 modules (Kronecker Paley-Hadamard, Z_D Parseval, Adaptive-W' Alg. 2, Baselines)
├── coattr/            # Original CoAttr v1 core library & Stage 3.1/3.2 modules
├── benchmarks/        # Live GPU benchmark runners for SGLang & vLLM 0.34.0 APC (Qwen3-4B/32B, gemma-4-31B-it, 27 workloads / 150Q)
├── tests/             # 48 deterministic offline unit tests (23 v1 + 25 v2) verifying Theorems 1-4, Prop. 1, and all modules
├── data/              # Archived authoritative evaluation tables (data/m2-master-summary.json covering 27 workloads / 150Q, 32B acc, 4-quota C_KV, Gemma-4 99K, vLLM 0.34.0)
└── scripts/           # Offline table verification and publication-figure reproduction scripts
```

---

## Quickstart (Offline Verification — No GPU Required)

All theoretical verifications and **48 deterministic unit tests** run locally in seconds using [`uv`](https://docs.astral.sh/uv/) (Python 3.11+):

```bash
# 1. Run all 48 unit tests (Paley-Hadamard mod-8 congruence, Z_D Parseval tightening,
#    all-g cyclic separation, adaptive-W' delta_wave simulator, BLIP/Sketching/AttnTrace, bootstrap CIs)
uv run --with pytest --with numpy --with scikit-learn pytest -v

# 2. Verify archived 11-workload Stage 3 tables & inspect 27-workload / 150Q M2 master summary
uv run scripts/verify_tables.py
uv run python -c "import json; d = json.load(open('data/m2-master-summary.json')); print('Sections:', list(d.keys()))"
```

---

## Running Live GPU Benchmarks (SGLang & vLLM)

The live benchmark suites in `benchmarks/m2/` run against unmodified **SGLang** (`page_size=64`) and **vLLM 0.34.0** (`enable_prefix_caching=True`, `block_size=16`, per-request `prompt_logprob_start` + `skip_reading_prefix_cache=False`):

```bash
# 27-workload / 150-query SOTA comparison + live m=120 Regime B stream (Qwen3-4B)
uv run benchmarks/m2/run_job_a_4b.py --model Qwen/Qwen3-4B --help

# Qwen3-32B 5-rep timings (incl. 18Q & SingleQ65 1Q), 16Q attribution accuracy, & 4-quota C_KV sweep
uv run benchmarks/m2/run_job_b_32b.py --model Qwen/Qwen3-32B --help

# gemma-4-31B-it (99K tight KV pool) adaptive-W' width ablation & 6-workload 5-rep sweep
uv run benchmarks/m2/run_job_c_gemma4.py --model google/gemma-4-31B-it --help

# Cross-engine vLLM 0.34.0 APC (block_size=16) 5-rep evaluation on Qwen3-4B & Qwen3-32B
uv run benchmarks/m2/run_job_d_vllm.py --help
```

---

## License

Released under the [MIT License](LICENSE).

## Ultra-Long Scaling & Multi-Witness Provenance (`Hier-CoAttr` & `CoAttr-Verify`)

- **`coattr_v2/hierarchical_v2.py`**: Two-level random-polarity hierarchical Kronecker Paley--Hadamard attribution (`Hier-CoAttr`, Theorem 5) scaling cleanly to $n = 200$--$400+$ chunks ($20\text{K}$--$48\text{K}$ tokens) while guaranteeing exact flat-Banzhaf unbiasedness for arbitrary Boolean utilities $f : \{\pm 1\}^n \to \mathbb{R}$.
- **`coattr_v2/verifiable_v2.py`**: Dual-threshold $(\tau_{\mathrm{suff}}, \tau_{\mathrm{clear}})$ multi-witness verifiable provenance (`CoAttr-Verify`, Algorithm 3) certifying all disjoint minimal sufficient witnesses $W_1, \dots, W_m$ and complete post-remediation clearance.
- **`coattr_v2/tracllm_v2.py`**: Faithful `TracLLM` (USENIX Security'25) hierarchical perturbation tree baseline.
- **`benchmarks/r2/` & `data/r2-master-summary.json`**: End-to-end CUAD multi-predicate `sem_filter` compliance auditing (14 commercial contracts, 204 predicates), long-context adversarial security forensics (18 workloads, 90 live incidents), and 33K-token tight-KV physical scheduling benchmarks (`gemma-4-31B-it`).
- Run all 56 offline unit tests:
  ```bash
  uv run --with pytest --with numpy --with scikit-learn pytest -q tests/
  ```
