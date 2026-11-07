"""l2_sd: ICU perturbation size (exact for FGSM; median among flipped ICU patients for the line search).
Combined SPS as a function of attack configuration / perturbation budget.
Reads results_r2/r2_results.json, adds a 'budget' table. Run after r2_pipeline.py
from the third_paper/ directory (needs code/sps on the path)."""
import os, sys, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.getcwd(), "code", "sps"))
import sps as S

P = os.path.join(HERE, "results_r2", "r2_results.json")
R = json.load(open(P))
base = dict(threat_mitigated=11, threat_tested=11, threat_total=12, threat_passed=6, threat_partial=6,
            audit_completeness_rate=0.9737, rbac_enforcement_rate=1.0, compliance_score=0.8182)
rows = []
sw = pd.DataFrame(R["epsilon_sweep"])
for eps in sorted(sw.eps.unique()):
    s = sw[sw.eps == eps].set_index("model")
    d = float(s.auroc_drop.max())
    rows.append(dict(config=f"FGSM eps={eps}", l2_sd=round(eps * np.sqrt(6), 3), l2_sd_ed=round(eps * np.sqrt(6), 3),
                     icu_flip=float(s.loc["icu_mortality", "flip_rate"]), icu_drop=float(s.loc["icu_mortality", "auroc_drop"]),
                     ed_flip=float(s.loc["ed_admission", "flip_rate"]), ed_drop=float(s.loc["ed_admission", "auroc_drop"]),
                     combined_sps=S.compute_sps(auroc_drop=round(d, 4), **base)["sps"]))
adv = pd.DataFrame(R["adversarial"]); b = adv[adv.attack == "Boundary"].set_index("model")
bv = pd.DataFrame(R["boundary_variants"])
variants = [("Boundary, nearest target", b)] + [
    (f"Boundary, {t} target" + (f" (seed {sd})" if t == "random" else ""), bv[(bv.target == t) & (bv.seed == sd)].set_index("model"))
    for t, sd in [("first", 0)] + [("random", s) for s in range(5)]]
for lab, s in variants:
    d = float(s.auroc_drop.max())
    rows.append(dict(config=lab, l2_sd=float(s.loc["icu_mortality", "median_l2_std"]), l2_sd_ed=float(s.loc["ed_admission", "median_l2_std"]),
                     icu_flip=float(s.loc["icu_mortality", "flip_rate"]), icu_drop=float(s.loc["icu_mortality", "auroc_drop"]),
                     ed_flip=float(s.loc["ed_admission", "flip_rate"]), ed_drop=float(s.loc["ed_admission", "auroc_drop"]),
                     combined_sps=S.compute_sps(auroc_drop=round(d, 4), **base)["sps"]))
R["budget"] = rows
json.dump(R, open(P, "w"), indent=2)
pd.DataFrame(rows).to_csv(os.path.join(HERE, "results_r2", "budget.csv"), index=False)
print(pd.DataFrame(rows).round(3).to_string())
