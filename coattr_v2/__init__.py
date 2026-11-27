"""CoAttr v2 — Upgraded core library for Strong Accept milestone (M1–M4).

Modules:
- ``designs_v2``: Dyadic Ordered g-Cosets + Kronecker Ordered Paley-Hadamard Cosets
  (H_g x H_L, L == 4 mod 8 via Paley Type-I and Type-II over GF(p) / GF(p^2)).
- ``safe_selector_v2``: Sign-corrected unbiased active Mean-MSE Parseval estimator Z_D,
  exact DP certificates (gamma_g, kappa_g), bidirectional Prop. 1(1) coupling,
  g=4 cyclic arc separation, and true sequential closed-loop Regime A + SNR R=2.
- ``capacity_scheduler_v2``: Explicit transient pioneer duplicate-prefix accounting
  (Delta_wave), closed-loop adaptive inner wave-width contraction W' (conquering
  tight KV regimes like gemma-4 99K), and dual SGLang/vLLM paged-KV simulators.
- ``baselines_v2``: Faithful BLIP two-stage minimal-sufficient pruning, Sketching-LOO,
  single-pass cross-layer attention aggregation (AttnTrace-style), CUBE R=2, and
  workload-cluster bootstrap 95% CI / paired significance testing.
- ``hierarchical_v2``: 2-level Random-Polarity Hierarchical Kronecker Paley-Hadamard
  attribution (Hier-CoAttr, Theorem 5) scaling cleanly to n = 200..512+ ultra-long
  contexts with exact flat-Banzhaf unbiasedness and zero sign-cancellation blindness.
- ``verifiable_v2``: Multi-Witness Verifiable Provenance (CoAttr-Verify) certifying
  all disjoint minimal sufficient witnesses W_1..W_m and post-remediation clearance.
- ``tracllm_v2``: Faithful TracLLM (Wang et al., USENIX Security'25) top-down
  hierarchical perturbation search tree with LOO + STC ensembling.
"""
