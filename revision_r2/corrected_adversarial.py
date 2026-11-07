"""
DIAGNOSTIC ONLY (not used for the paper). Re-runs the R1 models with the
clipping fixed but without data cleaning, to show the size of the R1 error.

Corrected adversarial battery for the R2 revision.

Fix: the models consume standardized (z-score) features, but the original
code clipped those standardized values to raw clinical bounds (e.g. heart
rate 30-200). Clipping alone moved every feature to an extreme value and
produced the original Table 2 results. Here perturbations are applied in
standardized space, mapped back to raw units with the cohort scaler,
clipped to the clinical bounds, and mapped forward again. Plausibility is
checked in raw units.

Adds: bootstrap 95% CIs (reviewer C2), LIME neighborhood ablation (C3),
and an epsilon sweep. Run from the third_paper/ directory.
"""
import os, sys, json, pickle, warnings, time
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from lime.lime_tabular import LimeTabularExplainer

ROOT = os.getcwd()
sys.path.insert(0, os.path.join(ROOT, "code", "adversarial"))
from tabular_fgsm import CLINICAL_BOUNDS  # raw-unit bounds

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
N_BOOT = 2000
COHORTS = [("icu_mortality", "icu_cohort"), ("ed_admission", "ed_cohort")]


class RawSpace:
    def __init__(self, scaler, feats):
        self.mu, self.sd = scaler.mean_, scaler.scale_
        self.lo = np.array([CLINICAL_BOUNDS.get(f, (-np.inf, np.inf))[0] for f in feats])
        self.hi = np.array([CLINICAL_BOUNDS.get(f, (-np.inf, np.inf))[1] for f in feats])

    def to_raw(self, Z):
        return Z * self.sd + self.mu

    def to_std(self, R):
        return (R - self.mu) / self.sd

    def clip_std(self, Z):
        return self.to_std(np.clip(self.to_raw(Z), self.lo, self.hi))

    def plausible(self, Z):
        R = self.to_raw(Z)
        return ((R >= self.lo - 1e-9) & (R <= self.hi + 1e-9)).all(axis=1)


def lime_grad(explainer, model, x, n_feat, n_samples, seed):
    explainer.random_state = np.random.RandomState(seed)
    exp = explainer.explain_instance(x, model.predict_proba, num_features=n_feat,
                                     num_samples=n_samples)
    g = np.zeros(n_feat)
    for idx, w in exp.as_map()[1]:
        g[idx] = w
    return g


def fgsm(model, explainer, rs, X, eps, n_samples, seed=42, grads=None):
    preds = model.predict(X)
    if grads is None:
        grads = np.array([lime_grad(explainer, model, x, X.shape[1], n_samples, seed + i)
                          for i, x in enumerate(X)])
    s = np.sign(grads)
    s[preds == 1] *= -1
    X_adv_unclipped = X + eps * s
    X_adv = rs.clip_std(X_adv_unclipped)
    return X_adv, X_adv_unclipped, grads


def boundary(model, rs, X, X_pool, step=0.05, max_steps=100):
    pool_pred = model.predict(X_pool)
    X_adv, steps, flipped = X.copy(), [], []
    for i, x in enumerate(X):
        p0 = model.predict(x[None])[0]
        cand = np.where(pool_pred != p0)[0]
        if len(cand) == 0:
            steps.append(max_steps); flipped.append(False); continue
        seed = X_pool[cand[0]]  # first opposite-class sample, as in the original code
        cur, done = x.copy(), False
        for k in range(1, max_steps + 1):
            d = seed - cur
            n = np.linalg.norm(d)
            if n < 1e-10:
                break
            nxt = rs.clip_std(cur + step * d / n)
            if model.predict(nxt[None])[0] != p0:
                cur, done = nxt, True
                steps.append(k); break
            cur = nxt
        if not done:
            steps.append(max_steps)
        flipped.append(done)
        X_adv[i] = cur
    return X_adv, np.array(steps), np.array(flipped)


def metrics(model, X, Xa, y):
    p0, pa = model.predict_proba(X)[:, 1], model.predict_proba(Xa)[:, 1]
    flips = model.predict(X) != model.predict(Xa)
    return p0, pa, flips


def boot_ci(y, p0, pa, flips, rng):
    fr, dr = [], []
    n = len(y)
    while len(fr) < N_BOOT:
        b = rng.randint(0, n, n)
        if len(np.unique(y[b])) < 2:
            continue
        fr.append(flips[b].mean())
        dr.append(roc_auc_score(y[b], p0[b]) - roc_auc_score(y[b], pa[b]))
    q = lambda a: [round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3)]
    return q(fr), q(dr), np.array(fr), np.array(dr)


def main():
    t0 = time.time()
    summary, ablation, sweep, boots, perturb = [], [], [], {}, []
    for name, d in COHORTS:
        model = pickle.load(open(f"code/system/models/{name}_rf.pkl", "rb"))
        feats = json.load(open(f"code/system/models/{name}_meta.json"))["feature_names"]
        scaler = pickle.load(open(f"data/mimic/{d}/scaler.pkl", "rb"))
        rs = RawSpace(scaler, feats)
        Xtr = pd.read_csv(f"data/mimic/{d}/train_X.csv").values
        X = pd.read_csv(f"data/mimic/{d}/test_X.csv").values
        y = pd.read_csv(f"data/mimic/{d}/test_y.csv").values.ravel()
        rng = np.random.RandomState(42)  # same draws as run_all_tests.py
        i_f = rng.choice(len(X), 100, replace=False)
        i_b = rng.choice(len(X), 100, replace=False)
        expl = LimeTabularExplainer(Xtr, feature_names=feats, mode="classification",
                                    discretize_continuous=False, random_state=42)

        # Original (buggy) clip, no perturbation: shows the artifact
        Zc = X[i_f].copy()
        for j, f in enumerate(feats):
            if f in CLINICAL_BOUNDS:
                Zc[:, j] = np.clip(Zc[:, j], *CLINICAL_BOUNDS[f])
        p0, pa, fl = metrics(model, X[i_f], Zc, y[i_f])
        summary.append(dict(model=name, attack="Clip-only artifact (original code, eps=0)",
                            n=100, auroc_clean=roc_auc_score(y[i_f], p0),
                            auroc_adv=roc_auc_score(y[i_f], pa), flip_rate=fl.mean()))

        # FGSM, eps=0.1, LIME n=50 (corrected)
        Xa, Xu, G50 = fgsm(model, expl, rs, X[i_f], 0.1, 50)
        p0, pa, fl = metrics(model, X[i_f], Xa, y[i_f])
        bci = boot_ci(y[i_f], p0, pa, fl, np.random.RandomState(7))
        dR = np.abs(rs.to_raw(Xa) - rs.to_raw(X[i_f]))
        for j, f in enumerate(feats):
            perturb.append(dict(model=name, feature=f, sd_raw=float(rs.sd[j]),
                                median_abs_change_raw=float(np.median(dR[:, j])),
                                max_abs_change_raw=float(dR[:, j].max())))
        summary.append(dict(model=name, attack="FGSM eps=0.1 (LIME n=50)", n=100,
                            auroc_clean=roc_auc_score(y[i_f], p0), auroc_adv=roc_auc_score(y[i_f], pa),
                            flip_rate=fl.mean(), flip_ci=bci[0], drop_ci=bci[1],
                            plausibility_unclipped=rs.plausible(Xu).mean(),
                            plausibility_after_clip=rs.plausible(Xa).mean(),
                            flip_rate_no_clip=(model.predict(Xu) != model.predict(X[i_f])).mean()))
        boots[f"{name}_fgsm"] = bci[2]

        # Boundary attack (corrected)
        Xb, st, fb = boundary(model, rs, X[i_b], X)
        p0b, pab, flb = metrics(model, X[i_b], Xb, y[i_b])
        bcb = boot_ci(y[i_b], p0b, pab, flb, np.random.RandomState(8))
        dRb = np.abs(rs.to_raw(Xb) - rs.to_raw(X[i_b]))
        summary.append(dict(model=name, attack="Boundary step=0.05 (<=100 steps)", n=100,
                            auroc_clean=roc_auc_score(y[i_b], p0b), auroc_adv=roc_auc_score(y[i_b], pab),
                            flip_rate=flb.mean(), flip_ci=bcb[0], drop_ci=bcb[1],
                            mean_steps=float(st[fb].mean()) if fb.any() else None,
                            median_steps=float(np.median(st[fb])) if fb.any() else None,
                            median_linf_std=float(np.median(np.abs(Xb - X[i_b]).max(1)[fb])) if fb.any() else None,
                            median_hr_change_bpm=float(np.median(dRb[fb, 0])) if fb.any() else None,
                            plausibility=rs.plausible(Xb).mean()))
        boots[f"{name}_boundary"] = bcb[2]
        print(name, "core done", round(time.time() - t0), "s", flush=True)

        # C3: LIME neighborhood ablation, eps=0.1
        for ns in [50, 500, 2000, 5000]:
            for seed in ([42, 1042, 2042] if ns <= 500 else [42]):
                Xa2, _, G = fgsm(model, expl, rs, X[i_f], 0.1, ns, seed=seed)
                p0, pa, fl = metrics(model, X[i_f], Xa2, y[i_f])
                agree = (np.sign(G) == np.sign(G50)).mean()
                ablation.append(dict(model=name, lime_samples=ns, seed=seed, flip_rate=fl.mean(),
                                     auroc_drop=roc_auc_score(y[i_f], p0) - roc_auc_score(y[i_f], pa),
                                     sign_agreement_with_n50=agree))
                print(" ablation", name, ns, seed, round(fl.mean(), 3), round(time.time() - t0), "s", flush=True)

        # Epsilon sweep, LIME n=50 gradients reused
        for eps in [0.05, 0.1, 0.25, 0.5, 1.0]:
            Xa3, _, _ = fgsm(model, expl, rs, X[i_f], eps, 50, grads=G50)
            p0, pa, fl = metrics(model, X[i_f], Xa3, y[i_f])
            sweep.append(dict(model=name, eps=eps, flip_rate=fl.mean(),
                              auroc_drop=roc_auc_score(y[i_f], p0) - roc_auc_score(y[i_f], pa)))

    # ICU vs ED flip-rate difference CI (C2's specific question)
    for atk in ["fgsm", "boundary"]:
        diff = boots[f"ed_admission_{atk}"] - boots[f"icu_mortality_{atk}"]
        summary.append(dict(model="ED minus ICU", attack=f"{atk} flip-rate difference",
                            diff_ci=[round(float(np.percentile(diff, 2.5)), 3),
                                     round(float(np.percentile(diff, 97.5)), 3)]))

    pd.DataFrame(summary).to_csv(f"{OUT}/corrected_table2.csv", index=False)
    pd.DataFrame(ablation).to_csv(f"{OUT}/lime_ablation.csv", index=False)
    pd.DataFrame(sweep).to_csv(f"{OUT}/epsilon_sweep.csv", index=False)
    pd.DataFrame(perturb).to_csv(f"{OUT}/raw_perturbation_sizes.csv", index=False)
    json.dump(summary, open(f"{OUT}/corrected_table2.json", "w"), indent=2, default=float)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(pd.DataFrame(summary)); print(pd.DataFrame(ablation)); print(pd.DataFrame(sweep))
    print(pd.DataFrame(perturb))


if __name__ == "__main__":
    main()
