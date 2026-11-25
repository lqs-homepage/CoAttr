# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy==2.2.6", "matplotlib>=3.9"]
# ///
"""Reproduce paper Figures 5, 7, 8, and 9 (PDF + 300-DPI PNG) directly from data/."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
S31 = ROOT / "data"
S32 = ROOT / "data"
S33 = ROOT / "data"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Linux Libertine O", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.0,
    "axes.titlesize": 8.6,
    "axes.titleweight": "bold",
    "axes.labelsize": 8.0,
    "xtick.labelsize": 7.1,
    "ytick.labelsize": 7.4,
    "legend.fontsize": 7.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "grid.linewidth": 0.55,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
})

COL = {
    "cap": "#0B5394",
    "half": "#117A65",
    "four": "#0B5394",
    "prefix": "#D97706",
    "glob_mrg": "#5DADE2",
    "bllm": "#8E44AD",
    "lpm128": "#95A5A6",
    "lpm32": "#5D6D7E",
    "cube": "#4A5568",
    "regmsr": "#C0392B",
    "proxy": "#E67E22",
    "kbanz": "#7F8C8D",
    "arcjsd": "#8E44AD",
    "loo": "#B03A2E",
}


def save_fig(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG_DIR / f"{stem}.pdf")
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300)
    plt.close(fig)
    print(f"Saved {stem}.pdf and {stem}.png")


def make_fig7_scheduler_kv(s32_summary: dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.20, 3.05), gridspec_kw={"width_ratios": [1.08, 1.0]})

    j151 = {(r["doc"], r["nq"], r["design"]): r for r in s32_summary["jobs"]["151"]["scheduler_table"]}
    j169 = {(r["doc"], r["nq"], r["design"]): r for r in s32_summary["jobs"]["169"]["scheduler_table"]}
    wl_a = [
        ("QASPER-1810.05241", 2, "QASP-1810\n(2Q)"),
        ("QASPER-1810.05241", 6, "QASP-1810\n(6Q)"),
        ("QuALITY-22967", 2, "QuAL-22967\n(2Q)"),
        ("QuALITY-22967", 18, "QuAL-22967\n(18Q)"),
    ]
    x = np.arange(len(wl_a))
    w = 0.145
    series_a = [
        ("Native LPM (128)", [j151[(d, q, "four")]["sglang_native_lpm_s"] for d, q, _ in wl_a], COL["lpm128"], "///"),
        ("Native LPM (32)", [j169[(d, q, "four")]["sglang_native_lpm_s"] for d, q, _ in wl_a], COL["lpm32"], "\\\\"),
        ("BatchLLM-DFS", [j151[(d, q, "four")]["batchllm_dfs_s"] for d, q, _ in wl_a], COL["bllm"], ""),
        ("Prefix (2-wave)", [j169[(d, q, "four")]["prefix_s"] for d, q, _ in wl_a], COL["prefix"], ""),
        ("CoAttr Cap-Trie", [j169[(d, q, "four")]["capacity_aware_merged_mask_trie_s"] for d, q, _ in wl_a], COL["cap"], ""),
    ]
    offsets = (np.arange(len(series_a)) - (len(series_a) - 1) / 2.0) * w
    for idx, (lbl, vals, color, hatch) in enumerate(series_a):
        bars = ax1.bar(
            x + offsets[idx], vals, width=w * 0.92, label=lbl,
            color=color, edgecolor="#222222", linewidth=0.45, hatch=hatch, zorder=3
        )
        if idx == len(series_a) - 1:
            lpm32_vals = series_a[1][1]
            bllm_vals = series_a[2][1]
            for b, v_co, v_lpm, v_bl in zip(bars, vals, lpm32_vals, bllm_vals):
                sp = v_lpm / v_co
                y_pos = max(v_co + 0.85, v_bl + 0.45) if v_co < 10 else v_co + 0.9
                ax1.text(
                    b.get_x() + b.get_width() * 0.55, y_pos, f"{sp:.2f}×",
                    ha="center", va="bottom", fontsize=6.6, fontweight="bold", color=COL["cap"],
                )
    ax1.set_xticks(x)
    ax1.set_xticklabels([lbl for _, _, lbl in wl_a])
    ax1.set_ylabel("End-to-End Wall-Clock (s)")
    ax1.set_ylim(0, 34.5)
    ax1.legend(
        loc="lower center", bbox_to_anchor=(0.48, 1.01), ncol=3,
        frameon=False, handlelength=1.1, columnspacing=0.65, handletextpad=0.35, fontsize=6.6,
    )
    ax1.set_title("(a) Qwen3-4B Scheduler Showdown (4-Coset)", pad=28)

    j165 = {(r["doc"], r["nq"], r["design"]): r for r in s32_summary["jobs"]["165"]["scheduler_table"]}
    wl_b = [
        ("QuALITY-20027", 2, "QuAL-20027\n(fits KV)"),
        ("QASPER-1810.05241", 2, "QASP-1810\n(2Q, >273K)"),
        ("QASPER-1810.05241", 6, "QASP-1810\n(6Q, >273K)"),
        ("QuALITY-22967", 2, "QuAL-22967\n(2Q, >273K)"),
    ]
    xb = np.arange(len(wl_b))
    wb = 0.24
    v_glob = [j165[(d, q, "half")]["merged_mask_trie_s"] for d, q, _ in wl_b]
    v_pre = [j165[(d, q, "half")]["prefix_s"] for d, q, _ in wl_b]
    v_cap = [j165[(d, q, "half")]["capacity_aware_merged_mask_trie_s"] for d, q, _ in wl_b]
    tiles = [j165[(d, q, "half")]["capacity_aware_merged_mask_trie_tile"] for d, q, _ in wl_b]

    ax2.bar(xb - wb, v_glob, width=wb * 0.90, label="Global Trie (unbounded)", color=COL["glob_mrg"], edgecolor="#222222", linewidth=0.45, hatch="///", zorder=3)
    ax2.bar(xb, v_pre, width=wb * 0.90, label="Prefix (2-wave)", color=COL["prefix"], edgecolor="#222222", linewidth=0.45, zorder=3)
    b3 = ax2.bar(xb + wb, v_cap, width=wb * 0.90, label="CoAttr Cap-Aware Trie", color=COL["cap"], edgecolor="#222222", linewidth=0.45, zorder=3)

    ax2.axvline(0.5, color="#555555", linestyle=":", linewidth=0.95, zorder=2)
    ax2.text(0.0, 48.2, "Fits KV", ha="center", va="top", fontsize=6.6, style="italic", color="#444444")
    ax2.text(2.0, 48.2, "Exceeds 273K KV (LRU Thrashing Regime)", ha="center", va="top", fontsize=6.6, style="italic", color="#444444")

    for idx, (bar_c, vg, vp, vc, tl) in enumerate(zip(b3, v_glob, v_pre, v_cap, tiles)):
        if idx == 0:
            vs_pre = (1.0 - vc / vp) * 100.0
            note = f"t={tl}\n(-{vs_pre:.1f}%\nvs pre)"
            y_pos = max(vp, vc) + 1.0
        else:
            rec = (1.0 - vc / vg) * 100.0
            note = f"t={tl}\n(-{rec:.1f}%)"
            y_pos = vc + 1.0
        ax2.text(
            bar_c.get_x() + bar_c.get_width() * 0.52, y_pos, note,
            ha="center", va="bottom", fontsize=6.4, fontweight="bold", color=COL["cap"],
        )

    ax2.set_xticks(xb)
    ax2.set_xticklabels([lbl for _, _, lbl in wl_b], fontsize=6.8)
    ax2.set_ylabel("End-to-End Wall-Clock (s)")
    ax2.set_ylim(0, 50.5)
    ax2.legend(
        loc="lower center", bbox_to_anchor=(0.50, 1.01), ncol=2,
        frameon=False, handlelength=1.1, columnspacing=0.8, handletextpad=0.35, fontsize=6.6,
    )
    ax2.set_title("(b) Qwen3-32B HBM KV Phase Transition (2-Coset)", pad=28)

    fig.tight_layout(w_pad=2.0)
    save_fig(fig, "fig7_scheduler_kv")


def make_fig8_cross_model(core_tables: dict) -> None:
    t2 = core_tables["table2_cross_model_scheduler"]
    by_tk = {(r["track"], r["doc"], r["nq"]): r for r in t2}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.20, 2.95), gridspec_kw={"width_ratios": [1.14, 1.0]})

    groups = [
        ("QASP\n1912·2Q", "QASPER-1912.06262", 2),
        ("QuAL\n20027·2Q", "QuALITY-20027", 2),
        ("QASP\n1810·2Q", "QASPER-1810.05241", 2),
        ("QASP\n1810·6Q", "QASPER-1810.05241", 6),
        ("QuAL\n22967·2Q", "QuALITY-22967", 2),
        ("QuAL\n22967·18Q", "QuALITY-22967", 18),
    ]
    x = np.arange(len(groups))
    w = 0.25

    def get_four(tr: str, doc: str, nq: int) -> float | None:
        r = by_tk.get((tr, doc, nq))
        return float(r["fair_four_over_cube"]) if (r and r["fair_four_over_cube"] is not None) else None

    m4_vals = [
        get_four("Qwen3-4B (fast, max_run=32)", d, q) or get_four("Qwen3-4B (fast, trtllm_mha)", d, q)
        for _, d, q in groups
    ]
    m32_vals = [get_four("Qwen3-32B (fast, outpost)", d, q) for _, d, q in groups]
    mg_vals = [get_four("Gemma-4-31B-it (fast, 99K KV)", d, q) for _, d, q in groups]

    h1 = ax1.bar(x - w, [v or 0 for v in m4_vals], width=w * 0.90, label="Qwen3-4B (fast)", color=COL["half"], edgecolor="#222", linewidth=0.45, zorder=3)
    h2 = ax1.bar(x, [v or 0 for v in m32_vals], width=w * 0.90, label="Qwen3-32B (8× scale)", color=COL["cap"], edgecolor="#222", linewidth=0.45, zorder=3)
    g_idx = [i for i, v in enumerate(mg_vals) if v is not None]
    h3 = ax1.bar(np.array(g_idx) + w, [mg_vals[i] for i in g_idx], width=w * 0.90, label="Gemma-4-31B-it (2nd family)", color=COL["prefix"], edgecolor="#222", linewidth=0.45, zorder=3)

    ax1.axhline(1.0, color="#333333", linestyle="-", linewidth=0.85, zorder=2)
    ax1.axhline(6.0 / 7.0, color="#C0392B", linestyle="--", linewidth=0.9, zorder=4)
    ax1.axhline(2.0 / 3.0, color="#555555", linestyle=":", linewidth=0.9, zorder=4)
    ax1.text(5.42, 1.008, "CUBE = 1.0", ha="right", va="bottom", fontsize=6.3, color="#333333")
    ax1.text(2.45, 6.0 / 7.0 + 0.095, "6/7 attn-pair limit", ha="center", va="bottom", fontsize=6.3, color="#C0392B", fontweight="bold")
    ax1.annotate("", xy=(2.45, 6.0 / 7.0), xytext=(2.45, 6.0 / 7.0 + 0.092), arrowprops=dict(arrowstyle="->", color="#C0392B", lw=0.65))
    ax1.text(
        5.42, 2.0 / 3.0 - 0.015, "2/3 token limit",
        ha="right", va="top", fontsize=6.2, color="#444444", style="italic",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=1.0), zorder=6,
    )

    for i, (v4, v32, vg) in enumerate(zip(m4_vals, m32_vals, mg_vals)):
        if v32 is not None:
            top_bar = max(v for v in (v4, v32, vg) if v is not None)
            y_lbl = top_bar + 0.014
            if abs(y_lbl - 6.0 / 7.0) < 0.026:
                y_lbl = 6.0 / 7.0 + 0.016
            ax1.text(i, y_lbl, f"-{(1.0 - v32) * 100:.0f}%", ha="center", va="bottom", fontsize=6.4, color=COL["cap"], fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels([g[0] for g in groups], fontsize=6.6)
    ax1.set_ylabel("Wall-Clock Ratio (4-Coset / CUBE)")
    ax1.set_ylim(0.60, 1.07)
    ax1.set_title("(a) Fair Same-Scheduler Ratio (Natural Answers)", pad=6)

    j_groups = [
        ("QASP\n1912·2Q", "QASPER-1912.06262", 2),
        ("QuAL\n20027·2Q", "QuALITY-20027", 2),
        ("QASP\n1810·2Q", "QASPER-1810.05241", 2),
        ("QASP\n1810·6Q", "QASPER-1810.05241", 6),
        ("QuAL\n22967·18Q", "QuALITY-22967", 18),
    ]
    xj = np.arange(len(j_groups))
    wj = 0.25

    def get_sp(tr: str, doc: str, nq: int) -> float | None:
        r = by_tk.get((tr, doc, nq))
        return float(r["joint_speedup_vs_cube_lpm"]) if (r and r["joint_speedup_vs_cube_lpm"] is not None) else None

    sp4 = [
        get_sp("Qwen3-4B (fast, max_run=32)", d, q) or get_sp("Qwen3-4B (fast, trtllm_mha)", d, q)
        for _, d, q in j_groups
    ]
    sp32 = [get_sp("Qwen3-32B (fast, outpost)", d, q) for _, d, q in j_groups]
    spg = [get_sp("Gemma-4-31B-it (fast, 99K KV)", d, q) for _, d, q in j_groups]

    ax2.bar(xj - wj, [v or 0 for v in sp4], width=wj * 0.90, color=COL["half"], edgecolor="#222", linewidth=0.45, zorder=3)
    b32 = ax2.bar(xj, [v or 0 for v in sp32], width=wj * 0.90, color=COL["cap"], edgecolor="#222", linewidth=0.45, zorder=3)
    gi = [i for i, v in enumerate(spg) if v is not None]
    ax2.bar(np.array(gi) + wj, [spg[i] for i in gi], width=wj * 0.90, color=COL["prefix"], edgecolor="#222", linewidth=0.45, zorder=3)

    for b, v in zip(b32, sp32):
        if v is not None:
            ax2.text(b.get_x() + b.get_width() / 2.0, v + 0.09, f"{v:.2f}×", ha="center", va="bottom", fontsize=6.5, fontweight="bold", color=COL["cap"])

    ax2.axhline(1.0, color="#333333", linestyle=":", linewidth=0.8)
    ax2.set_xticks(xj)
    ax2.set_xticklabels([g[0] for g in j_groups], fontsize=6.6)
    ax2.set_ylabel("Joint Speedup vs CUBE@LPM (×)")
    ax2.set_ylim(0, 4.45)
    ax2.set_title("(b) Joint End-to-End Speedup (Fast-Path)", pad=6)

    fig.legend(
        handles=[h1, h2, h3],
        labels=["Qwen3-4B (fast)", "Qwen3-32B (8× scale)", "Gemma-4-31B-it (2nd family)"],
        loc="upper center", bbox_to_anchor=(0.52, 0.995), ncol=3,
        frameon=False, fontsize=7.6, handlelength=1.3, columnspacing=1.5,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91), w_pad=2.2)
    save_fig(fig, "fig8_cross_model")


def make_fig9_sota_pareto(core_tables: dict) -> None:
    t1 = {r["doc"]: r for r in core_tables["table1_grand_pareto"]}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.20, 2.90))
    leader = dict(arrowstyle="-", color="#777777", lw=0.55, shrinkA=1, shrinkB=3)

    m88 = t1["MuSiQue-Pack88D"]["methods"]
    pts_a = [
        ("LOO (BLIP)", m88["loo"]["wall_s"], m88["loo"]["gold_recall_k"], COL["loo"], "v", (9.2, 0.415), "left", False),
        ("ARC-JSD (ICLR'26)", m88["arc_jsd"]["wall_s"], m88["arc_jsd"]["gold_recall_k"], COL["arcjsd"], "D", (10.5, 0.488), "left", False),
        ("Kernel Banzhaf", m88["KernelBanzhaf_AntitheticRidge_2M"]["wall_s"], m88["KernelBanzhaf_AntitheticRidge_2M"]["gold_recall_k"], COL["kbanz"], "s", (16.5, 0.560), "left", True),
        ("CUBE (1-blk)", m88["cube"]["wall_s"], m88["cube"]["gold_recall_k"], COL["cube"], "o", (17.2, 0.625), "left", True),
        ("ProxySHAP 2M\n(NeurIPS'26)", m88["ProxySHAP_SharedMSR_2M"]["wall_s"], m88["ProxySHAP_SharedMSR_2M"]["gold_recall_k"], COL["proxy"], "P", (17.8, 0.688), "left", True),
        ("RegMSR-GBT 2M\n(NeurIPS'25)", m88["RegressionMSR_GBT_2M"]["wall_s"], m88["RegressionMSR_GBT_2M"]["gold_recall_k"], COL["regmsr"], "^", (16.8, 0.805), "center", True),
        (f"CoAttr Ordered\nHalf (1-blk, {m88['half']['wall_s']:.1f}s)", m88["half"]["wall_s"], m88["half"]["gold_recall_k"], COL["half"], "*", (4.65, 0.808), "left", True),
        (f"CoAttr Ordered\nFour (1-blk, {m88['four']['wall_s']:.1f}s)", m88["four"]["wall_s"], m88["four"]["gold_recall_k"], COL["four"], "*", (4.65, 0.695), "left", True),
        ("CoAttr Cyclic Four\n(R=2)", m88["cyclic_four_r2"]["wall_s"], m88["cyclic_four_r2"]["gold_recall_k"], COL["cap"], "h", (25.6, 0.805), "center", True),
        ("RegMSR-GBT 4M", m88["RegressionMSR_GBT_4M"]["wall_s"], m88["RegressionMSR_GBT_4M"]["gold_recall_k"], COL["regmsr"], "^", (31.5, 0.670), "right", True),
    ]
    for lbl, xv, yv, col, mk, xy_txt, ha, use_arrow in pts_a:
        sz = 115 if mk == "*" else (65 if mk == "h" else 46)
        ax1.scatter(xv, yv, color=col, marker=mk, s=sz, edgecolors="#111111", linewidths=0.55, zorder=4)
        ax1.annotate(
            lbl, (xv, yv), xytext=xy_txt, textcoords="data", ha=ha, va="center",
            fontsize=6.3, fontweight=("bold" if "CoAttr" in lbl else "normal"),
            color=(col if "CoAttr" in lbl else "#222222"),
            arrowprops=(leader if use_arrow else None), zorder=5,
        )
    ax1.plot(
        [m88["loo"]["wall_s"], m88["arc_jsd"]["wall_s"], m88["half"]["wall_s"]],
        [m88["loo"]["gold_recall_k"], m88["arc_jsd"]["gold_recall_k"], m88["half"]["gold_recall_k"]],
        linestyle="--", color=COL["half"], linewidth=1.0, alpha=0.75, zorder=2,
    )
    ax1.set_xlabel("B200 End-to-End Wall-Clock (s)")
    ax1.set_ylabel("Gold Multi-Hop Recall@k")
    ax1.set_xlim(4.4, 32.0)
    ax1.set_ylim(0.38, 0.855)
    ax1.set_title("(a) Multi-Hop RAG (MuSiQue-Pack88D, 11K tok)", pad=6)

    mq = t1["QuALITY-22967"]["methods"]
    four_save = (1.0 - mq["four"]["wall_s"] / mq["cube"]["wall_s"]) * 100.0
    pts_b = [
        ("LOO (BLIP)", mq["loo"]["wall_s"], mq["loo"]["top1_vs_ref"], COL["loo"], "v", (5.8, 0.265), "left", True),
        ("ARC-JSD (ICLR'26)", mq["arc_jsd"]["wall_s"], mq["arc_jsd"]["top1_vs_ref"], COL["arcjsd"], "D", (10.2, 0.165), "left", False),
        ("Kernel Banzhaf", mq["KernelBanzhaf_AntitheticRidge_2M"]["wall_s"], mq["KernelBanzhaf_AntitheticRidge_2M"]["top1_vs_ref"], COL["kbanz"], "s", (5.9, 0.605), "left", True),
        ("ProxySHAP 2M", mq["ProxySHAP_SharedMSR_2M"]["wall_s"], mq["ProxySHAP_SharedMSR_2M"]["top1_vs_ref"], COL["proxy"], "P", (6.2, 0.455), "left", True),
        ("RegMSR-GBT 2M", mq["RegressionMSR_GBT_2M"]["wall_s"], mq["RegressionMSR_GBT_2M"]["top1_vs_ref"], COL["regmsr"], "^", (15.6, 0.455), "left", True),
        ("CUBE (1-blk)", mq["cube"]["wall_s"], mq["cube"]["top1_vs_ref"], COL["cube"], "o", (15.5, 0.785), "left", False),
        (f"CoAttr Ordered Four\n(1-blk, -{four_save:.1f}%)", mq["four"]["wall_s"], mq["four"]["top1_vs_ref"], COL["four"], "*", (5.9, 0.835), "left", True),
        (f"CoAttr Cyclic Half\n(R=2, Best {mq['cyclic_half_r2']['top1_vs_ref']:.3f})", mq["cyclic_half_r2"]["wall_s"], mq["cyclic_half_r2"]["top1_vs_ref"], COL["half"], "*", (16.2, 0.895), "left", True),
        ("CoAttr Cyclic Four\n(R=2)", mq["cyclic_four_r2"]["wall_s"], mq["cyclic_four_r2"]["top1_vs_ref"], COL["cap"], "h", (17.2, 0.715), "left", True),
        ("RegMSR-GBT 4M", mq["RegressionMSR_GBT_4M"]["wall_s"], mq["RegressionMSR_GBT_4M"]["top1_vs_ref"], COL["regmsr"], "^", (20.2, 0.630), "left", True),
        ("ProxySHAP 4M", mq["ProxySHAP_SharedMSR_4M"]["wall_s"], mq["ProxySHAP_SharedMSR_4M"]["top1_vs_ref"], COL["proxy"], "P", (23.0, 0.525), "left", True),
    ]
    for lbl, xv, yv, col, mk, xy_txt, ha, use_arrow in pts_b:
        sz = 115 if mk == "*" else (65 if mk == "h" else 46)
        ax2.scatter(xv, yv, color=col, marker=mk, s=sz, edgecolors="#111111", linewidths=0.55, zorder=4)
        ax2.annotate(
            lbl, (xv, yv), xytext=xy_txt, textcoords="data", ha=ha, va="center",
            fontsize=6.2, fontweight=("bold" if "CoAttr" in lbl else "normal"),
            color=(col if "CoAttr" in lbl else "#222222"),
            arrowprops=(leader if use_arrow else None), zorder=5,
        )
    ax2.plot(
        [mq["loo"]["wall_s"], mq["four"]["wall_s"], mq["cube"]["wall_s"], mq["cyclic_half_r2"]["wall_s"]],
        [mq["loo"]["top1_vs_ref"], mq["four"]["top1_vs_ref"], mq["cube"]["top1_vs_ref"], mq["cyclic_half_r2"]["top1_vs_ref"]],
        linestyle="--", color=COL["half"], linewidth=1.0, alpha=0.75, zorder=2,
    )
    ax2.set_xlabel("B200 End-to-End Wall-Clock (s)")
    ax2.set_ylabel("Top-1 Accuracy vs 8-Block Truth")
    ax2.set_xlim(5.5, 32.0)
    ax2.set_ylim(0.08, 0.965)
    ax2.set_title("(b) Hard 18-Q Long-Doc (QuALITY-22967)", pad=6)

    fig.tight_layout(w_pad=2.1)
    save_fig(fig, "fig9_sota_pareto")


def make_fig5_theory_guarantees(theory_json: dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.20, 2.70))

    rs = np.array([1, 2, 3])
    qasper = theory_json["T1_long_doc_blind_probs"]["QASPER_n71_M128"]
    quality = theory_json["T1_long_doc_blind_probs"]["QuALITY_n90_M128"]
    sim = theory_json["T1_sparse_counterexample_sim"]

    y_q_vac = [quality["r1_vac_blind"] ** r for r in rs]
    y_q_comb = [quality["r1_vac_blind"], quality["r2_both_blind"], quality["r3_both_blind"]]
    y_a_comb = [qasper["r1_vac_blind"], qasper["r2_both_blind"], qasper["r3_both_blind"]]

    ax1.semilogy(rs, y_q_vac, "s--", color=COL["regmsr"], linewidth=1.5, markersize=5.2, label="Vacant-only (QuALITY n=90)")
    ax1.semilogy(rs, y_q_comb, "o-", color=COL["cap"], linewidth=1.8, markersize=5.6, label="CoAttr Joint (QuALITY n=90)")
    ax1.semilogy(rs, y_a_comb, "^-", color=COL["half"], linewidth=1.5, markersize=5.2, label="CoAttr Joint (QASPER n=71)")
    ax1.semilogy([1, 2], [sim["r1_vac_blind_empirical"], sim["r2_combined_blind_empirical"]], "D", color=COL["prefix"], markersize=5.2, label="MC Verified (n=20, M=32)")

    drop_x = y_q_comb[0] / y_q_comb[1]
    ax1.annotate(
        f"{y_q_comb[0]*100:.1f}% blind\n(single block)", (1, y_q_comb[0]), xytext=(1.35, 0.16),
        arrowprops=dict(arrowstyle="->", color="#444444", lw=0.7), fontsize=6.7, color="#333333",
    )
    ax1.annotate(
        f"{y_q_comb[1]*100:.2f}% ({drop_x:.0f}× drop!)", (2, y_q_comb[1]), xytext=(2.08, 0.032),
        arrowprops=dict(arrowstyle="->", color=COL["cap"], lw=0.8), fontsize=6.7, fontweight="bold", color=COL["cap"],
    )
    ax1.annotate(
        f"{y_q_comb[2]:.1e}", (3, y_q_comb[2]), xytext=(2.68, 1.3e-4),
        arrowprops=dict(arrowstyle="->", color=COL["cap"], lw=0.75), fontsize=6.7, fontweight="bold", color=COL["cap"],
    )

    ax1.set_xticks([1, 2, 3])
    ax1.set_xticklabels(["R = 1", "R = 2", "R = 3"])
    ax1.set_xlim(0.85, 3.22)
    ax1.set_ylim(3e-6, 1.7)
    ax1.set_xlabel("Independent Design Blocks (R)")
    ax1.set_ylabel("1-Sparse Blind-Spot Prob. (log)")
    ax1.set_title("(a) Multi-Block Blind-Spot Collapse (Thm. 2)", pad=6)
    ax1.legend(loc="lower left", frameon=True, framealpha=0.92, fontsize=6.4, handlelength=1.4)

    ms = np.arange(8, 261, 2)
    delta = 0.05 / 3.0
    log_eb = math.log(4.0 / delta)
    log_hoeff = math.log(2.0 / delta)

    def eb_zero_var(span: float, m_arr: np.ndarray) -> np.ndarray:
        return (7.0 * span * log_eb) / (3.0 * (m_arr - 1.0))

    def hoeff_union(span: float, m_arr: np.ndarray) -> np.ndarray:
        return span * np.sqrt(log_hoeff / (2.0 * m_arr))

    def union_zero_var(span: float, m_arr: np.ndarray) -> np.ndarray:
        return np.minimum(eb_zero_var(span, m_arr), hoeff_union(span, m_arr))

    r_naive = union_zero_var(2.0, ms)
    r_qual = union_zero_var(2.0 / 38.0, ms)
    r_qasp = union_zero_var(2.0 / 57.0, ms)

    ms_small = ms[ms <= 70]
    ax2.semilogy(ms_small, eb_zero_var(2.0, ms_small), ":", color=COL["regmsr"], linewidth=0.9, alpha=0.55)
    ax2.semilogy(ms_small, eb_zero_var(2.0 / 38.0, ms_small), ":", color=COL["prefix"], linewidth=0.9, alpha=0.55)
    ax2.semilogy(ms_small, eb_zero_var(2.0 / 57.0, ms_small), ":", color=COL["cap"], linewidth=0.9, alpha=0.55)

    ax2.semilogy(ms, r_naive, "--", color=COL["regmsr"], linewidth=1.5, label="Naive [-1, 1] min(EB, Hoeff.)")
    ax2.semilogy(ms, r_qual, "-", color=COL["prefix"], linewidth=1.8, label="Parseval Tight (QuALITY, 38×)")
    ax2.semilogy(ms, r_qasp, "-", color=COL["cap"], linewidth=1.8, label="Parseval Tight (QASPER, 57×)")
    ax2.axvline(70, color="#6B7280", linestyle="-.", linewidth=0.75, alpha=0.75)
    ax2.text(68, 1.55, r"Hoeff. $\leftarrow m{=}70 \rightarrow$ EB", ha="center", va="top", fontsize=5.8, color="#4B5563",
             bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.8))
    ax2.axhline(0.005, color="#333333", linestyle=":", linewidth=0.9)
    ax2.text(252, 0.0058, r"Target $\tau = 0.005$", ha="right", va="bottom", fontsize=6.7, color="#333333")

    ax2.scatter([91, 136], [0.005, 0.005], color=[COL["cap"], COL["prefix"]], s=38, zorder=5, edgecolors="#111")
    ax2.annotate(
        "m=91\n(vs 5,117)", (91, 0.005), xytext=(28, 0.00105),
        arrowprops=dict(arrowstyle="->", color=COL["cap"], lw=0.8), fontsize=6.7, fontweight="bold", color=COL["cap"],
    )
    ax2.annotate(
        "m=136", (136, 0.005), xytext=(158, 0.00135),
        arrowprops=dict(arrowstyle="->", color=COL["prefix"], lw=0.8), fontsize=6.7, fontweight="bold", color=COL["prefix"],
    )

    ax2.set_xlabel("Independent Samples / Questions (m)")
    ax2.set_ylabel("Certified Union Radius (log)")
    ax2.set_xlim(8, 260)
    ax2.set_ylim(5e-4, 2.5)
    ax2.set_title("(b) Parseval Tight Certificate (Thm. 2)", pad=6)
    ax2.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=6.1)

    fig.tight_layout(w_pad=2.2)
    save_fig(fig, "fig5_theory_guarantees")


def main() -> None:
    core_tables = json.loads((S33 / "core-tables.json").read_text(encoding="utf-8"))
    s32_summary = json.loads((S32 / "stage-3.2-summary.json").read_text(encoding="utf-8"))
    theory_json = json.loads((S31 / "theory-extensions-summary.json").read_text(encoding="utf-8"))

    make_fig5_theory_guarantees(theory_json)
    make_fig7_scheduler_kv(s32_summary)
    make_fig8_cross_model(core_tables)
    make_fig9_sota_pareto(core_tables)
    print("Regenerated all 4 Stage 3.3 figures with consistent CoAttr / Theorem 2 naming!")


if __name__ == "__main__":
    main()
