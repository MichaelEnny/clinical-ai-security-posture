"""Figures for the R2 revision. Reads results_r2/r2_results.json."""
import os, json, textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "results_r2", "r2_results.json")))
ROOT = os.path.dirname(os.path.dirname(HERE))  # third_paper/
FIG = os.path.join(ROOT, "figures"); os.makedirs(FIG, exist_ok=True)
COL = {"icu_mortality": "#2563EB", "ed_admission": "#D97706"}
LAB = {"icu_mortality": "ICU mortality", "ed_admission": "ED admission"}
INK, MUTED, GRID = "#1f2328", "#57606a", "#e5e7eb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300, "font.family": "DejaVu Sans"})
MODELS = ["icu_mortality", "ed_admission"]


def grid(ax):
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)


def fig_adversarial():
    adv = pd.DataFrame(R["adversarial"])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    attacks = ["FGSM", "Boundary"]
    x = np.arange(2); w = 0.36
    for k, (metric, ci, ttl, yl) in enumerate([("flip_rate", "flip_ci", "Prediction flip rate", "Flip rate"),
                                               ("auroc_drop", "drop_ci", "AUROC drop", "Clean minus adversarial AUROC")]):
        ax = axes[k]
        for j, m in enumerate(MODELS):
            sub = adv[adv.model == m].set_index("attack").loc[attacks]
            v = sub[metric].values; lo = [c[0] for c in sub[ci]]; hi = [c[1] for c in sub[ci]]
            err = np.array([v - lo, np.array(hi) - v])
            ax.bar(x + (j - 0.5) * w, v, w * 0.94, color=COL[m], label=LAB[m])
            ax.errorbar(x + (j - 0.5) * w, v, yerr=err, fmt="none", ecolor=INK, capsize=3, lw=1)
            for xi, vi, hii in zip(x + (j - 0.5) * w, v, hi):
                ax.text(xi, hii + (0.008 if metric == "flip_rate" else 0.003), f"{vi:.0%}" if metric == "flip_rate" else f"{vi:.3f}",
                        ha="center", va="bottom", fontsize=8.5, color=INK)
        ax.set_xticks(x); ax.set_xticklabels(["FGSM (ε = 0.1)", "Boundary (nearest target)"])
        ax.set_title(ttl, color=INK); ax.set_ylabel(yl); grid(ax); ax.axhline(0, color=MUTED, lw=0.8)
        if metric == "flip_rate":
            ax.set_ylim(0, 0.40); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        else:
            ax.set_ylim(-0.01, 0.15)
        ax.legend(frameon=False, loc="upper left")
    ax = axes[2]
    mi = pd.DataFrame(R["membership_inference"])
    ax.axhspan(0.469, 0.531, color=GRID, label="95% band under chance (n = 1,000)")
    ax.axhline(0.5, color=MUTED, lw=1, ls="--")
    ax.axhline(0.55, color=MUTED, lw=0.8, ls=":"); ax.text(1.45, 0.553, "MODERATE threshold", ha="right", fontsize=8, color=MUTED)
    for i, m in enumerate(MODELS):
        r = mi[mi.model == m].iloc[0]
        ax.plot(i, r.accuracy, "o", ms=10, color=COL[m], mec="white", mew=2)
        ax.text(i + 0.1, r.accuracy, f"{r.accuracy:.3f}\n(p = {r.p_binom:.2f})", va="center", fontsize=8.5, color=INK)
    ax.set_xticks([0, 1]); ax.set_xticklabels([LAB[m] for m in MODELS]); ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0.40, 0.62); ax.set_ylabel("Attack accuracy"); ax.set_title("Membership inference", color=INK)
    ax.legend(frameon=False, loc="upper left", fontsize=8); grid(ax)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "adversarial_results.png")); plt.close(fig)


def fig_attack_sensitivity():
    sw = pd.DataFrame(R["epsilon_sweep"]); ab = pd.DataFrame(R["lime_ablation"])
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    for m in MODELS:
        s = sw[sw.model == m]
        axes[0].plot(s.eps, s.flip_rate, "-o", color=COL[m], lw=2, ms=6, mec="white", mew=1.5, label=LAB[m])
        axes[1].plot(s.eps, s.auroc_drop, "-o", color=COL[m], lw=2, ms=6, mec="white", mew=1.5, label=LAB[m])
        a = ab[(ab.model == m) & (ab.get("sampling", "training_distribution") != "around_instance")].groupby("lime_samples").flip_rate.agg(["mean", "min", "max"])
        axes[2].errorbar(a.index, a["mean"], yerr=[a["mean"] - a["min"], a["max"] - a["mean"]], fmt="-o",
                         color=COL[m], lw=2, ms=6, capsize=3, label=LAB[m])
    axes[0].set_title("FGSM flip rate vs. ε", color=INK); axes[0].set_xlabel("ε (training-set SD units)")
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0)); axes[0].set_ylabel("Flip rate")
    axes[1].set_title("FGSM AUROC drop vs. ε", color=INK); axes[1].set_xlabel("ε (training-set SD units)")
    axes[1].set_ylabel("AUROC drop")
    axes[2].set_xscale("log"); axes[2].set_xticks([50, 500, 2000, 5000]); axes[2].set_xticklabels(["50", "500", "2,000", "5,000"])
    axes[2].set_title("FGSM (ε = 0.1) vs. LIME neighborhood size", color=INK); axes[2].set_xlabel("LIME perturbation samples")
    axes[2].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0)); axes[2].set_ylabel("Flip rate"); axes[2].set_ylim(0, 0.2)
    bu = pd.DataFrame(R["budget"]); ax = axes[3]
    ax.axhspan(80, 92, color="#e8f0fe"); ax.axhspan(65, 80, color="#fdf3e1")
    ax.axhline(80, color=MUTED, lw=0.8, ls=":")
    f = bu[bu.config.str.startswith("FGSM")]; bd = bu[bu.config.str.startswith("Boundary")]
    ax.plot(f.l2_sd, f.combined_sps, "-s", color=INK, ms=6, lw=1.5, label="FGSM, ε sweep")
    ax.plot(bd.l2_sd, bd.combined_sps, "D", color="#7C3AED", ms=7, mec="white", mew=1, label="Boundary, alternative targets")
    nn = bd[bd.config.str.contains("nearest")]
    ax.plot(nn.l2_sd, nn.combined_sps, "D", color="#7C3AED", ms=11, mec=INK, mew=1.5, label="Boundary, nearest target (primary)")
    ax.text(0.05, 80.4, "production tier (≥ 80)", fontsize=8, color=MUTED); ax.text(0.05, 76.0, "conditional tier", fontsize=8, color=MUTED)
    ax.set_xlim(0, 2.6); ax.set_ylim(72, 92)
    ax.set_xlabel("ICU perturbation size, L2 (SD units)"); ax.set_ylabel("Combined SPS")
    ax.set_title("Combined SPS vs. perturbation size", color=INK)
    for a in axes:
        grid(a); a.legend(frameon=False, loc="upper left" if a is not ax else "lower left", fontsize=8 if a is ax else 10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "attack_sensitivity.png")); plt.close(fig)


def fig_sps_sensitivity():
    se = pd.DataFrame(R["sensitivity"]); base = R["sps"]["combined"]["sps"]
    labels = ["Adversarial AUROC drop\nfixed at 0.25", "No threat mitigations\ntested", "Degraded audit\nand RBAC",
              "Minimal compliance\nposture", "No-defense baseline\n(all dimensions)"]
    d = (se.sps - base).values
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    y = np.arange(len(d))[::-1]
    ax.barh(y, d, color=["#2563EB" if v > 0 else "#57606a" for v in d], height=0.6)
    for yi, v, s in zip(y, d, se.sps):
        ax.text(v + (0.6 if v > 0 else -0.6), yi, f"{v:+.2f}  (SPS {s:.2f})", va="center",
                ha="left" if v > 0 else "right", fontsize=9, color=INK)
    ax.axvline(0, color=INK, lw=1); ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel(f"ΔSPS = scenario SPS − baseline SPS ({base:.2f})"); ax.set_xlim(-62, 6)
    ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.set_title("SPS sensitivity to single-dimension and joint ablations", color=INK)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "sps_sensitivity.png")); plt.close(fig)


def fig_mitre():
    mt = pd.read_csv(os.path.join(HERE, "mitre_mapping_r2.csv"))
    tactics = ["Initial Access", "Credential Access", "Collection", "Command and Control", "Impact", "Exfiltration"]
    names = {"CUSTOM-MI": "Membership Inference", "CUSTOM-ADV": "Adversarial Input",
             "T1071": "Application Layer Protocol"}
    stat = {"PASS": "#16A34A", "PARTIAL": "#D97706", "FAIL": "#DC2626"}
    fig, ax = plt.subplots(figsize=(11, 5.6))
    counts = {t: 0 for t in tactics}
    for _, r in mt.iterrows():
        c = tactics.index(r.tactic); k = counts[r.tactic]; counts[r.tactic] += 1
        ax.add_patch(plt.Rectangle((c + 0.04, k + 0.08), 0.92, 0.84, color=stat[r.result]))
        nm = names.get(r.id, r.technique.split(":")[0])
        ax.text(c + 0.5, k + 0.5, f"{r.id}\n{textwrap.fill(nm, 17)}\n[{r.result}]", ha="center", va="center", fontsize=7.6,
                color="white", fontweight="bold")
    ax.set_xlim(0, len(tactics)); ax.set_ylim(0, max(counts.values()))
    ax.set_xticks(np.arange(len(tactics)) + 0.5); ax.set_xticklabels([t.replace(" and ", "\nand ") for t in tactics])
    ax.set_yticks([]); [ax.spines[s].set_visible(False) for s in ["left", "top", "right"]]
    ax.set_title("MITRE ATT&CK Enterprise techniques mapped to the clinical AI inference system\n"
                 f"({(mt.result == 'PASS').sum()} PASS, {(mt.result == 'PARTIAL').sum()} PARTIAL, "
                 f"{(mt.result == 'FAIL').sum()} FAIL; status also printed in each cell)", color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=v) for v in stat.values()]
    ax.legend(handles, ["PASS: mitigated and tested", "PARTIAL: mitigation incomplete or not fully tested",
                        "FAIL: unmitigated (none observed)"], frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8.5)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "mitre_coverage_map.png")); plt.close(fig)


if __name__ == "__main__":
    fig_adversarial(); fig_attack_sensitivity(); fig_sps_sensitivity(); fig_mitre(); print("ok")
