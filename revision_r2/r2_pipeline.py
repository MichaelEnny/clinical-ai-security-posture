"""
R2 corrected evaluation pipeline (run from third_paper/).

Corrections relative to the R1 code:
  1. Raw vital signs are winsorized to the physiological bounds of Table 1
     before standardization (the R1 scaler was fit on uncleaned charting
     errors, e.g. SpO2 = 390,709).
  2. Adversarial samples are clipped to clinical bounds in RAW units, then
     re-standardized. R1 clipped standardized values to raw bounds, which
     moved every feature to an extreme value and produced the R1 Table 2.
  3. Only the six vital signs are perturbed; demographics, acuity and pain
     are held fixed. FGSM and boundary attacks use the same patients.
  4. n = 500 per model, bootstrap 95% CIs, seeded membership inference with
     the full training set as the shadow pool for both models, calibration
     metrics (Brier, ECE), LIME neighborhood ablation and epsilon sweep.
"""
import os, sys, json, time, pickle, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss
from lime.lime_tabular import LimeTabularExplainer

ROOT = os.getcwd()
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_r2"); os.makedirs(OUT, exist_ok=True)
MOD = os.path.join(HERE, "models_r2"); os.makedirs(MOD, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, "code", "sps"))
import sps as sps_module

N_ATTACK, N_BOOT, N_MI = 500, 2000, 500
VITAL_BOUNDS = {  # Table 1, raw units
    "heart_rate": (30, 200), "heartrate": (30, 200),
    "resp_rate": (5, 60), "resprate": (5, 60),
    "spo2": (70, 100), "o2sat": (70, 100),
    "sbp": (60, 220), "dbp": (30, 140),
    "temperature_f": (94, 107), "temperature": (94, 107),
}
OTHER_BOUNDS = {"pain": (0, 10)}
COHORTS = [("icu_mortality", "icu_cohort"), ("ed_admission", "ed_cohort")]
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)


# ---------------------------------------------------------------- data
def load_clean(d):
    base = f"data/mimic/{d}"
    old = pickle.load(open(f"{base}/scaler.pkl", "rb"))
    tr = pd.read_csv(f"{base}/train_X_raw.csv")
    te = pd.read_csv(f"{base}/test_X_raw.csv")
    va = pd.DataFrame(old.inverse_transform(pd.read_csv(f"{base}/val_X.csv").values), columns=tr.columns)
    ys = {k: pd.read_csv(f"{base}/{k}_y.csv").values.ravel() for k in ["train", "val", "test"]}
    feats = list(tr.columns)
    clip_report = []
    for f in feats:
        b = VITAL_BOUNDS.get(f) or OTHER_BOUNDS.get(f)
        if b is None:
            continue
        n_out = int(((tr[f] < b[0]) | (tr[f] > b[1])).sum())
        clip_report.append(dict(cohort=d, feature=f, lower=b[0], upper=b[1],
                                train_values_clipped=n_out, pct=round(100 * n_out / len(tr), 3)))
        for df in (tr, va, te):
            df[f] = df[f].clip(*b)
    sc = StandardScaler().fit(tr.values)
    Z = {k: sc.transform(v.values) for k, v in [("train", tr), ("val", va), ("test", te)]}
    return feats, sc, Z, ys, clip_report


class Space:
    def __init__(self, sc, feats):
        self.mu, self.sd = sc.mean_, sc.scale_
        self.vital = np.array([f in VITAL_BOUNDS for f in feats])
        self.lo = np.array([VITAL_BOUNDS.get(f, (-np.inf, np.inf))[0] for f in feats], float)
        self.hi = np.array([VITAL_BOUNDS.get(f, (-np.inf, np.inf))[1] for f in feats], float)

    def raw(self, Z): return Z * self.sd + self.mu
    def std(self, R): return (R - self.mu) / self.sd
    def clip(self, Z): return self.std(np.clip(self.raw(Z), self.lo, self.hi))
    def plausible(self, Z):
        R = self.raw(Z); return ((R >= self.lo - 1e-6) & (R <= self.hi + 1e-6)).all(1)


def ece(y, p, bins=10):
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & (p < b) if b < 1 else (p >= a) & (p <= b)
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e


# ---------------------------------------------------------------- attacks
def lime_grads(model, expl, X, n_samples, seed):
    G = np.zeros_like(X)
    for i, x in enumerate(X):
        expl.random_state = np.random.RandomState(seed + i)
        ex = expl.explain_instance(x, model.predict_proba, num_features=X.shape[1], num_samples=n_samples)
        for j, w in ex.as_map()[1]:
            G[i, j] = w
    return G


def fgsm(model, sp, X, G, eps):
    s = np.sign(G) * sp.vital  # vitals only
    s[model.predict(X) == 1] *= -1
    Xu = X + eps * s
    return sp.clip(Xu), Xu


def boundary(model, sp, X, X_pool, step=0.05, max_steps=100, target="nearest", seed=0):
    """Target-directed line search (batched). Each patient's vital signs move in
    0.05-SD steps toward a target record with the opposite predicted class.
    target: 'nearest' = per-patient nearest opposite-class pool record in
    vital-sign space (primary); 'first' = first opposite-class record in pool
    order (R1 anchor); 'random' = one random opposite-class record per class."""
    pool_pred = model.predict(X_pool)
    p0s = model.predict(X)
    if target == "nearest":
        from sklearn.neighbors import NearestNeighbors
        seeds = np.zeros_like(X)
        for c in np.unique(p0s):
            opp = X_pool[pool_pred != c]
            nn = NearestNeighbors(n_neighbors=1).fit(opp[:, sp.vital])
            rows = np.where(p0s == c)[0]
            seeds[rows] = opp[nn.kneighbors(X[rows][:, sp.vital], return_distance=False)[:, 0]]
    else:
        rng = np.random.RandomState(seed)
        pick = {}
        for c in np.unique(p0s):
            cand = np.where(pool_pred != c)[0]
            pick[c] = X_pool[cand[0] if target == "first" else rng.choice(cand)]
        seeds = np.array([pick[c] for c in p0s])
    cur = X.copy()
    steps = np.full(len(X), max_steps)
    flipped = np.zeros(len(X), bool)
    active = np.ones(len(X), bool)
    for k in range(1, max_steps + 1):
        if not active.any():
            break
        d = (seeds[active] - cur[active]) * sp.vital
        n = np.linalg.norm(d, axis=1, keepdims=True)
        moving = n[:, 0] > 1e-10
        upd = cur[active].copy()
        upd[moving] = sp.clip(upd[moving] + step * d[moving] / n[moving])
        cur[active] = upd
        ids = np.where(active)[0]
        now = model.predict(cur[ids]) != p0s[ids]
        flipped[ids[now]] = True
        steps[ids[now]] = k
        active[ids[now]] = False
        active[ids[~moving]] = False
    return cur, steps, flipped


def boot(y, p0, pa, fl, seed):
    rng = np.random.RandomState(seed); F, D = [], []
    while len(F) < N_BOOT:
        b = rng.randint(0, len(y), len(y))
        if y[b].min() == y[b].max():
            continue
        F.append(fl[b].mean()); D.append(roc_auc_score(y[b], p0[b]) - roc_auc_score(y[b], pa[b]))
    return np.array(F), np.array(D)


ci = lambda a: [round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3)]


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    perf, clips, adv, abl, sweep, raws, mi_rows, bvar = [], [], [], [], [], [], [], []
    boots = {}
    for name, d in COHORTS:
        feats, sc, Z, ys, cr = load_clean(d); clips += cr
        sp = Space(sc, feats)
        rf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5,
                                    class_weight="balanced", random_state=42, n_jobs=-1)
        rf.fit(Z["train"], ys["train"])
        rf.set_params(n_jobs=1)
        pickle.dump(rf, open(f"{MOD}/{name}_rf.pkl", "wb")); pickle.dump(sc, open(f"{MOD}/{name}_scaler.pkl", "wb"))
        Xte, yte = Z["test"], ys["test"]
        pt = rf.predict_proba(Xte)[:, 1]
        rng = np.random.RandomState(11); aucs = []
        for _ in range(N_BOOT):
            b = rng.randint(0, len(yte), len(yte)); aucs.append(roc_auc_score(yte[b], pt[b]))
        perf.append(dict(model=name, n_train=len(ys["train"]), n_val=len(ys["val"]), n_test=len(yte),
                         prevalence=float(yte.mean()), auroc=roc_auc_score(yte, pt), auroc_ci=ci(aucs),
                         brier=brier_score_loss(yte, pt), ece=ece(yte, pt),
                         brier_train=brier_score_loss(ys["train"], rf.predict_proba(Z["train"])[:, 1]),
                         auroc_train=roc_auc_score(ys["train"], rf.predict_proba(Z["train"])[:, 1]),
                         brier_prevalence_only=float(yte.mean() * (1 - yte.mean())),
                         precision_pos=float(((rf.predict(Xte) == 1) & (yte == 1)).sum() / max(1, (rf.predict(Xte) == 1).sum()))))
        log(name, "trained", perf[-1], round(time.time() - t0), "s")

        idx = np.random.RandomState(42).choice(len(Xte), N_ATTACK, replace=False)
        X, y = Xte[idx], yte[idx]
        expl = LimeTabularExplainer(Z["train"], feature_names=feats, mode="classification",
                                    discretize_continuous=False, random_state=42)
        p0 = rf.predict_proba(X)[:, 1]
        G50 = lime_grads(rf, expl, X, 50, 42)

        # FGSM eps=0.1
        Xa, Xu = fgsm(rf, sp, X, G50, 0.1)
        pa = rf.predict_proba(Xa)[:, 1]; fl = rf.predict(Xa) != rf.predict(X)
        F, D = boot(y, p0, pa, fl, 7); boots[(name, "FGSM")] = (F, D)
        adv.append(dict(model=name, attack="FGSM", n=N_ATTACK, auroc_clean=roc_auc_score(y, p0),
                        auroc_adv=roc_auc_score(y, pa), auroc_drop=roc_auc_score(y, p0) - roc_auc_score(y, pa),
                        drop_ci=ci(D), flip_rate=fl.mean(), flip_ci=ci(F),
                        in_bounds_without_clipping=sp.plausible(Xu).mean(),
                        flip_rate_without_clipping=(rf.predict(Xu) != rf.predict(X)).mean(),
                        out_of_bounds_inputs_flipped=int(((rf.predict(Xu) != rf.predict(X)) & ~sp.plausible(Xu)).sum()),
                        median_l2_std=float(np.median(np.linalg.norm(Xa - X, axis=1)))))
        dR = np.abs(sp.raw(Xa) - sp.raw(X))
        for j, f in enumerate(feats):
            if sp.vital[j]:
                raws.append(dict(model=name, feature=f, sd_raw=float(sp.sd[j]), eps_raw=0.1 * float(sp.sd[j]),
                                 max_abs_change=float(dR[:, j].max())))

        # Boundary (primary: nearest opposite-class target)
        Xb, st, fb = boundary(rf, sp, X, Xte, target="nearest")
        pb = rf.predict_proba(Xb)[:, 1]
        F, D = boot(y, p0, pb, fb, 8); boots[(name, "Boundary")] = (F, D)
        dRb = sp.raw(Xb) - sp.raw(X)
        adv.append(dict(model=name, attack="Boundary", n=N_ATTACK, auroc_clean=roc_auc_score(y, p0),
                        auroc_adv=roc_auc_score(y, pb), auroc_drop=roc_auc_score(y, p0) - roc_auc_score(y, pb),
                        drop_ci=ci(D), flip_rate=fb.mean(), flip_ci=ci(F),
                        median_steps=float(np.median(st[fb])) if fb.any() else None,
                        iqr_steps=[float(np.percentile(st[fb], 25)), float(np.percentile(st[fb], 75))] if fb.any() else None,
                        single_step_flips=int((st[fb] == 1).sum()),
                        median_l2_std=float(np.median(np.linalg.norm((Xb - X)[fb], axis=1))) if fb.any() else None,
                        median_abs_change_raw={f: float(np.median(np.abs(dRb[fb, j]))) for j, f in enumerate(feats) if sp.vital[j]}))
        log(name, "attacks done", adv[-2:], round(time.time() - t0), "s")

        # Boundary target sensitivity (review: anchor choice)
        for tgt, sd in [("first", 0)] + [("random", s) for s in range(5)]:
            Xv, stv, fv = boundary(rf, sp, X, Xte, target=tgt, seed=sd)
            bvar.append(dict(model=name, target=tgt, seed=sd, flip_rate=fv.mean(),
                             auroc_drop=roc_auc_score(y, p0) - roc_auc_score(y, rf.predict_proba(Xv)[:, 1]),
                             median_steps=float(np.median(stv[fv])) if fv.any() else None,
                             median_l2_std=float(np.median(np.linalg.norm((Xv - X)[fv], axis=1))) if fv.any() else None))
            log(" boundary variant", bvar[-1], round(time.time() - t0), "s")

        # LIME sampled around the instance (review: default samples around training mean)
        expl_inst = LimeTabularExplainer(Z["train"], feature_names=feats, mode="classification",
                                         discretize_continuous=False, sample_around_instance=True, random_state=42)
        for ns in [50, 500]:
            G = lime_grads(rf, expl_inst, X, ns, 42)
            Xa2, _ = fgsm(rf, sp, X, G, 0.1)
            fl2 = rf.predict(Xa2) != rf.predict(X)
            abl.append(dict(model=name, lime_samples=ns, seed=42, sampling="around_instance", flip_rate=fl2.mean(),
                            auroc_drop=roc_auc_score(y, p0) - roc_auc_score(y, rf.predict_proba(Xa2)[:, 1]),
                            sign_agreement_with_n50_seed42=float((np.sign(G[:, sp.vital]) == np.sign(G50[:, sp.vital])).mean())))
            log(" ablation", abl[-1], round(time.time() - t0), "s")

        # C3 LIME ablation
        for ns, seeds in [(50, [42, 1042, 2042]), (500, [42, 1042, 2042]), (2000, [42]), (5000, [42])]:
            for s in seeds:
                G = G50 if (ns == 50 and s == 42) else lime_grads(rf, expl, X, ns, s)
                Xa2, _ = fgsm(rf, sp, X, G, 0.1)
                fl2 = rf.predict(Xa2) != rf.predict(X)
                vm = sp.vital
                abl.append(dict(model=name, lime_samples=ns, seed=s, sampling="training_distribution", flip_rate=fl2.mean(),
                                auroc_drop=roc_auc_score(y, p0) - roc_auc_score(y, rf.predict_proba(Xa2)[:, 1]),
                                sign_agreement_with_n50_seed42=float((np.sign(G[:, vm]) == np.sign(G50[:, vm])).mean())))
                log(" ablation", abl[-1], round(time.time() - t0), "s")

        # epsilon sweep
        for eps in [0.05, 0.1, 0.25, 0.5, 1.0]:
            Xa3, Xu3 = fgsm(rf, sp, X, G50, eps)
            fl3 = rf.predict(Xa3) != rf.predict(X)
            sweep.append(dict(model=name, eps=eps, flip_rate=fl3.mean(),
                              auroc_drop=roc_auc_score(y, p0) - roc_auc_score(y, rf.predict_proba(Xa3)[:, 1]),
                              in_bounds_without_clipping=sp.plausible(Xu3).mean(),
                              hr_change_bpm=eps * float(sp.sd[0])))

        # Membership inference (Shokri et al.), seeded, full training set as shadow pool
        rng = np.random.RandomState(42)
        Xtr, ytr = Z["train"], ys["train"]
        AX, AY = [], []
        for k in range(4):
            m = np.zeros(len(Xtr), bool); m[np.random.RandomState(42 + k).choice(len(Xtr), len(Xtr) // 2, replace=False)] = True
            sh = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight="balanced",
                                        random_state=42 + k, n_jobs=-1).fit(Xtr[m], ytr[m])
            AX += [sh.predict_proba(Xtr[m]), sh.predict_proba(Xtr[~m])]
            AY += [np.ones(m.sum()), np.zeros((~m).sum())]
        atk = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1).fit(np.vstack(AX), np.concatenate(AY))
        im, inm = rng.choice(len(Xtr), N_MI, replace=False), rng.choice(len(Xte), N_MI, replace=False)
        q = np.vstack([rf.predict_proba(Xtr[im]), rf.predict_proba(Xte[inm])])
        t = np.concatenate([np.ones(N_MI), np.zeros(N_MI)])
        pr = atk.predict(q); acc = (pr == t).mean()
        mi_rows.append(dict(model=name, shadow_pool=len(Xtr), shadow_train_each=len(Xtr) // 2, n_eval=2 * N_MI,
                            accuracy=acc, auroc=roc_auc_score(t, atk.predict_proba(q)[:, 1]),
                            correct=int((pr == t).sum()), p_binom=binomtest(int((pr == t).sum()), 2 * N_MI, 0.5).pvalue,
                            risk="HIGH" if acc > 0.60 else ("MODERATE" if acc > 0.55 else "LOW")))
        # Loss-threshold baseline (Yeom et al. 2018): member if loss < mean training loss
        eps_ = 1e-12
        def loss(Xs, ys_):
            p = np.clip(rf.predict_proba(Xs)[:, 1], eps_, 1 - eps_)
            return -(ys_ * np.log(p) + (1 - ys_) * np.log(1 - p))
        thr = loss(Xtr, ytr).mean()
        lq = np.concatenate([loss(Xtr[im], ytr[im]), loss(Xte[inm], yte[inm])])
        prl = (lq < thr).astype(float); cl = int((prl == t).sum())
        mi_rows[-1].update(dict(yeom_threshold=float(thr), yeom_accuracy=cl / (2 * N_MI), yeom_correct=cl,
                                yeom_p=binomtest(cl, 2 * N_MI, 0.5).pvalue, yeom_auroc=roc_auc_score(t, -lq)))
        log(name, "MI", mi_rows[-1], round(time.time() - t0), "s")

    # ICU vs ED flip-rate difference (reviewer C2)
    diffs = []
    for a in ["FGSM", "Boundary"]:
        dd = boots[("ed_admission", a)][0] - boots[("icu_mortality", a)][0]
        obs = [r["flip_rate"] for r in adv if r["attack"] == a]
        diffs.append(dict(attack=a, observed_ed_minus_icu=float(obs[1] - obs[0]),
                          bootstrap_mean=float(np.mean(dd)), ci=ci(dd)))

    # SPS (combined / ICU / ED) with bootstrap CI propagated from D1
    # T1078 rated PARTIAL in R2: demo account passwords are hard-coded in source.
    base = dict(threat_mitigated=11, threat_tested=11, threat_total=12, threat_passed=6, threat_partial=6,
                audit_completeness_rate=0.9737, rbac_enforcement_rate=1.0, compliance_score=0.8182)
    advdf = pd.DataFrame(adv)
    sps_out = {}
    for label, models in [("combined", ["icu_mortality", "ed_admission"]), ("icu", ["icu_mortality"]), ("ed", ["ed_admission"])]:
        sub = advdf[advdf.model.isin(models)]
        worst = sub.loc[sub.auroc_drop.idxmax()]
        r = sps_module.compute_sps(auroc_drop=round(float(worst.auroc_drop), 4), **base)
        Dw = boots[(worst.model, worst.attack)][1]
        sps_b = [sps_module.compute_sps(auroc_drop=float(x), **base)["sps"] for x in Dw]
        sps_out[label] = dict(sps=r["sps"], sps_ci=ci(sps_b), worst=f"{worst.model} {worst.attack}",
                              worst_drop=float(worst.auroc_drop), dims={k: v["subscore"] for k, v in r["dimensions"].items()},
                              contrib={k: v["weighted_contribution"] for k, v in r["dimensions"].items()},
                              tier=r["deployment_recommendation"])
        sps_out[label]["p_ge_80"] = float(np.mean(np.array(sps_b) >= 80))
    sens = sps_module.sensitivity_analysis(dict(auroc_drop=round(sps_out["combined"]["worst_drop"], 4), **base))
    # combined SPS if the boundary attack used each alternative target
    fgsm_max = advdf[advdf.attack == "FGSM"].auroc_drop.max()
    for r in bvar:
        same = [v["auroc_drop"] for v in bvar if v["target"] == r["target"] and v["seed"] == r["seed"]]
        r["combined_sps_if_used"] = sps_module.compute_sps(auroc_drop=round(float(max(max(same), fgsm_max)), 4), **base)["sps"]

    res = dict(performance=perf, clipping=clips, adversarial=adv, flip_difference=diffs, lime_ablation=abl,
               epsilon_sweep=sweep, raw_eps=raws, membership_inference=mi_rows, sps=sps_out, sensitivity=sens,
               boundary_variants=bvar)
    json.dump(res, open(f"{OUT}/r2_results.json", "w"), indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    for k in ["performance", "clipping", "adversarial", "lime_ablation", "epsilon_sweep", "raw_eps", "membership_inference", "sensitivity", "boundary_variants"]:
        pd.DataFrame(res[k]).to_csv(f"{OUT}/{k}.csv", index=False)
    log("SPS", json.dumps(sps_out, indent=1, default=str)); log("diffs", diffs); log("DONE", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
