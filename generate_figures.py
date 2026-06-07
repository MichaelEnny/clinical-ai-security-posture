"""
Figure Generator for Paper 3 Security Evaluation
Reads tables/ CSVs produced by run_all_tests.py and renders publication figures.

Usage:
    cd third_paper/code
    python generate_figures.py
"""

import os
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

CODE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(CODE_DIR)
TABLES    = os.path.join(ROOT_DIR, "tables")
FIGURES   = os.path.join(ROOT_DIR, "figures")
os.makedirs(FIGURES, exist_ok=True)

PALETTE = {
    "icu":        "#2563EB",
    "ed":         "#059669",
    "attack":     "#DC2626",
    "defense":    "#7C3AED",
    "sps_bar":    "#1D4ED8",
    "PASS":       "#16A34A",
    "PARTIAL":    "#D97706",
    "FAIL":       "#DC2626",
    "UNTESTED":   "#6B7280",
}


# ── Fig 1: Adversarial Attack Results ─────────────────────────────────────────

def figure_adversarial():
    path = os.path.join(TABLES, "adversarial_results.csv")
    mi_path = os.path.join(TABLES, "membership_inference_results.csv")
    if not os.path.exists(path):
        print(f"  Missing: {path}")
        return

    raw = pd.read_csv(path)
    # Pivot long-format to wide per dataset
    fgsm = raw[raw["attack"] == "FGSM"].set_index("dataset")
    ba   = raw[raw["attack"] == "Boundary"].set_index("dataset")
    datasets = ["icu_mortality", "ed_admission"]
    labels = ["ICU Mortality", "ED Admission"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("Adversarial Attack Results — ICU Mortality and ED Admission Models",
                 fontsize=12, fontweight="bold", y=1.01)

    x = np.arange(len(datasets))
    width = 0.35

    # Panel A: Flip Rate
    ax = axes[0]
    fgsm_flip = [fgsm.loc[d, "flip_rate"] if d in fgsm.index else 0 for d in datasets]
    ba_flip   = [ba.loc[d, "flip_rate"]   if d in ba.index   else 0 for d in datasets]
    bars1 = ax.bar(x - width/2, fgsm_flip, width, label="FGSM (ε=0.1)", color=PALETTE["attack"], alpha=0.85)
    bars2 = ax.bar(x + width/2, ba_flip,   width, label="Boundary", color=PALETTE["defense"], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Prediction Flip Rate")
    ax.set_title("Flip Rate by Attack Type", fontsize=10)
    ax.legend(fontsize=8); ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    for b in list(bars1) + list(bars2):
        if b.get_height() > 0:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                    f"{b.get_height():.0%}", ha="center", va="bottom", fontsize=8)

    # Panel B: AUROC drops
    ax = axes[1]
    fgsm_drop = [fgsm.loc[d, "auroc_drop"] if d in fgsm.index else 0 for d in datasets]
    ba_drop   = [ba.loc[d, "auroc_drop"]   if d in ba.index   else 0 for d in datasets]
    clean_auc = [fgsm.loc[d, "auroc_clean"] if d in fgsm.index else 0 for d in datasets]
    bars3 = ax.bar(x - width/2, fgsm_drop, width, label="FGSM AUROC drop", color=PALETTE["attack"], alpha=0.85)
    bars4 = ax.bar(x + width/2, ba_drop,   width, label="Boundary AUROC drop", color=PALETTE["defense"], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("AUROC Drop"); ax.set_title("AUROC Degradation by Attack Type", fontsize=10)
    ax.legend(fontsize=8); ax.set_ylim(0, 0.5)
    for b in list(bars3) + list(bars4):
        if b.get_height() > 0:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                    f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    # Add clean baseline line
    for i, cl in enumerate(clean_auc):
        ax.annotate(f"Clean: {cl:.3f}", xy=(i, 0.02), ha="center", fontsize=7.5,
                    color=PALETTE["icu"], style="italic")

    # Panel C: Membership Inference
    ax = axes[2]
    if os.path.exists(mi_path):
        mi_df = pd.read_csv(mi_path)
        mi_numeric = mi_df[pd.to_numeric(mi_df["mi_accuracy"], errors="coerce").notna()].copy()
        mi_numeric["mi_accuracy"] = pd.to_numeric(mi_numeric["mi_accuracy"])
        mi_numeric["mi_auroc"]    = pd.to_numeric(mi_numeric["mi_auroc"])
        risks  = mi_numeric["privacy_risk"].tolist()
        colors = [{"LOW": PALETTE["PASS"], "MODERATE": PALETTE["PARTIAL"], "HIGH": PALETTE["FAIL"]}.get(r, PALETTE["UNTESTED"]) for r in risks]
        ax.scatter(mi_numeric["mi_auroc"], mi_numeric["mi_accuracy"], c=colors, s=180, zorder=5, edgecolors="white", linewidths=0.8)
        for _, row in mi_numeric.iterrows():
            ds_label = row["dataset"].replace("_", " ").title()
            ax.annotate(f"{ds_label}\n({row['privacy_risk']})", (row["mi_auroc"], row["mi_accuracy"]),
                        textcoords="offset points", xytext=(8, -4), fontsize=8)
    ax.axhline(0.50, ls="--", color="gray", lw=1.2, label="Baseline (0.50)")
    ax.axhline(0.55, ls=":", color=PALETTE["PARTIAL"], lw=1, label="MODERATE threshold")
    ax.axhline(0.60, ls=":", color=PALETTE["FAIL"],    lw=1, label="HIGH threshold")
    ax.set_xlabel("Attack AUROC"); ax.set_ylabel("Inference Accuracy")
    ax.set_title("Membership Inference (Shadow Model)", fontsize=10)
    ax.set_xlim(0.40, 0.65); ax.set_ylim(0.40, 0.70)
    patches = [mpatches.Patch(color=PALETTE["PASS"], label="LOW risk"),
               mpatches.Patch(color=PALETTE["PARTIAL"], label="MODERATE risk"),
               mpatches.Patch(color=PALETTE["FAIL"], label="HIGH risk")]
    ax.legend(handles=patches, fontsize=8, loc="upper left")

    plt.tight_layout()
    out = os.path.join(FIGURES, "adversarial_results.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── Fig 2: MITRE ATT&CK Coverage Map ─────────────────────────────────────────

def figure_mitre():
    path = os.path.join(TABLES, "mitre_mapping.csv")
    if not os.path.exists(path):
        print(f"  Missing: {path}")
        return

    df = pd.read_csv(path)

    fig, ax = plt.subplots(figsize=(13, 6))

    tactics_order = [
        "Initial Access", "Credential Access", "Collection",
        "Command and Control", "Impact", "Exfiltration"
    ]
    tactic_map = {t: i for i, t in enumerate(tactics_order)}
    df["tactic_idx"] = df["tactic"].map(tactic_map).fillna(len(tactics_order))

    result_colors = {"PASS": PALETTE["PASS"], "PARTIAL": PALETTE["PARTIAL"],
                     "FAIL": PALETTE["FAIL"], None: PALETTE["UNTESTED"]}

    y_positions = list(range(len(df)))
    for i, row in df.iterrows():
        color = result_colors.get(row["result"], PALETTE["UNTESTED"])
        bar = ax.barh(i, 1, left=row["tactic_idx"], height=0.6,
                      color=color, alpha=0.85, edgecolor="white")
        label = f"{row['id']}: {row['technique'][:40]}..."  \
                if len(str(row['technique'])) > 40 else f"{row['id']}: {row['technique']}"
        ax.text(row["tactic_idx"] + 0.05, i, label,
                va="center", ha="left", fontsize=7.5)

    ax.set_xticks(range(len(tactics_order)))
    ax.set_xticklabels([t.replace(" ", "\n") for t in tactics_order], fontsize=9)
    ax.set_yticks([])
    ax.set_title("MITRE ATT&CK for Healthcare — Clinical AI Threat Coverage Map",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(-0.1, len(tactics_order))
    ax.set_ylim(-0.5, len(df) - 0.5)

    legend_patches = [
        mpatches.Patch(color=PALETTE["PASS"], label="PASS — Mitigated"),
        mpatches.Patch(color=PALETTE["PARTIAL"], label="PARTIAL — Mitigated (incomplete)"),
        mpatches.Patch(color=PALETTE["FAIL"], label="FAIL — Unmitigated"),
        mpatches.Patch(color=PALETTE["UNTESTED"], label="UNTESTED"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.tight_layout()
    out = os.path.join(FIGURES, "mitre_coverage_map.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── Fig 3: SPS Sensitivity Analysis ──────────────────────────────────────────

def figure_sps_sensitivity():
    sps_path = os.path.join(TABLES, "sps_full.json")
    abl_path = os.path.join(TABLES, "sps_sensitivity.csv")
    if not os.path.exists(sps_path):
        print(f"  Missing: {sps_path}")
        return

    with open(sps_path) as f:
        full = json.load(f)

    sps_result = full["sps_result"]
    base_sps = sps_result["sps"]
    dims = sps_result["dimensions"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Security Posture Score (SPS) — Summary (Score: {base_sps}/100)",
                 fontsize=13, fontweight="bold")

    # Panel A: SPS breakdown by dimension
    ax = axes[0]
    dim_names = [
        "Adversarial\nRobustness", "Threat\nCoverage",
        "Access\nControl", "Compliance\nPosture"
    ]
    dim_keys = [
        "adversarial_input_robustness", "threat_coverage",
        "access_control_integrity", "compliance_posture"
    ]
    subscores    = [dims[k]["subscore"] for k in dim_keys]
    contributions = [dims[k]["weighted_contribution"] for k in dim_keys]
    weights      = [dims[k]["weight"] for k in dim_keys]

    bar_colors = ["#3B82F6", "#10B981", "#8B5CF6", "#F59E0B"]
    bars = ax.bar(dim_names, contributions, color=bar_colors, alpha=0.85, edgecolor="white")
    ax.axhline(base_sps * max(weights), ls="--", color="gray", lw=1)
    ax.set_ylabel("Weighted SPS Contribution")
    ax.set_title(f"SPS Dimension Contributions\n(Total SPS = {base_sps})", fontsize=10)
    for i, (bar, ss, w) in enumerate(zip(bars, subscores, weights)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"Sub: {ss}\n(×{w})", ha="center", va="bottom", fontsize=8)

    # SPS threshold lines
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks([])
    ax.axhline(65 * max(weights) / 100, ls=":", color="#F59E0B",
               lw=1.5, label="Conditional deployment (65)")
    ax.axhline(80 * max(weights) / 100, ls=":", color="#10B981",
               lw=1.5, label="Production deployment (80)")
    ax.legend(fontsize=8)

    # Panel B: Sensitivity ablations
    ax = axes[1]
    if os.path.exists(abl_path):
        abl_df = pd.read_csv(abl_path)
        scenarios = abl_df["scenario"].tolist()
        sps_vals  = abl_df["sps"].tolist()
        drops     = abl_df["sps_drop"].tolist()
        y_pos = list(range(len(scenarios)))
        ax.barh(y_pos, sps_vals, color=PALETTE["sps_bar"], alpha=0.75, label="Scenario SPS")
        ax.barh(y_pos, [base_sps]*len(scenarios), height=0.15,
                color="black", alpha=0.5, label=f"Base SPS ({base_sps})")
        ax.set_yticks(y_pos)
        ax.set_yticklabels([s[:45] for s in scenarios], fontsize=8)
        ax.set_xlabel("SPS Score")
        ax.set_title("Sensitivity Analysis — SPS Under Ablation", fontsize=10)
        ax.axvline(65, ls="--", color="#F59E0B", lw=1.2, label="Conditional threshold (65)")
        ax.axvline(80, ls="--", color="#10B981", lw=1.2, label="Production threshold (80)")
        ax.axvline(base_sps, ls="-", color="black", lw=1.5, alpha=0.5)
        ax.set_xlim(0, 100)
        ax.legend(fontsize=8, loc="lower right")
        for i, (v, d) in enumerate(zip(sps_vals, drops)):
            ax.text(v + 0.5, i, f"−{d}", va="center", fontsize=8, color=PALETTE["attack"])

    plt.tight_layout()
    out = os.path.join(FIGURES, "sps_sensitivity.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── Fig 4: Compliance Heatmap ─────────────────────────────────────────────────

def figure_compliance():
    path = os.path.join(TABLES, "compliance_checklist.csv")
    if not os.path.exists(path):
        print(f"  Missing: {path}")
        return

    df = pd.read_csv(path)
    status_val = {"PASS": 1.0, "PARTIAL": 0.5, "FAIL": 0.0}

    fig, ax = plt.subplots(figsize=(10, 5))
    items = df["id"].tolist()
    statuses = df["status"].fillna("UNTESTED").tolist()
    safeguards = df["safeguard"].tolist()
    colors = [{"PASS": PALETTE["PASS"], "PARTIAL": PALETTE["PARTIAL"],
               "FAIL": PALETTE["FAIL"]}.get(s, PALETTE["UNTESTED"]) for s in statuses]

    bars = ax.barh(items, [1]*len(items), color=colors, alpha=0.85,
                   edgecolor="white", height=0.65)
    for i, (item, status, safeguard) in enumerate(zip(items, statuses, safeguards)):
        ax.text(0.02, i, f"{item}: {safeguard[:55]}", va="center", fontsize=8.5, color="white",
                fontweight="bold" if status == "PASS" else "normal")
        ax.text(0.98, i, status, va="center", ha="right", fontsize=8.5,
                color="white", fontweight="bold")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("HIPAA / HL7 FHIR Compliance Posture", fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    legend_patches = [
        mpatches.Patch(color=PALETTE["PASS"], label="PASS"),
        mpatches.Patch(color=PALETTE["PARTIAL"], label="PARTIAL"),
        mpatches.Patch(color=PALETTE["FAIL"], label="FAIL"),
        mpatches.Patch(color=PALETTE["UNTESTED"], label="UNTESTED"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)

    plt.tight_layout()
    out = os.path.join(FIGURES, "compliance_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Generating paper figures...")
    figure_adversarial()
    figure_mitre()
    figure_sps_sensitivity()
    figure_compliance()
    print(f"\nAll figures saved to: {FIGURES}")


if __name__ == "__main__":
    main()
