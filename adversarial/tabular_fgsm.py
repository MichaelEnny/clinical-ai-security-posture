"""
Tabular FGSM Adversarial Attack
LIME-guided gradient perturbation for scikit-learn tree models.
Adapted from image FGSM: uses surrogate linear model (LIME) to estimate
perturbation direction, then clips to clinically plausible bounds.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from lime.lime_tabular import LimeTabularExplainer


CLINICAL_BOUNDS = {
    "heart_rate":      (30.0,  200.0),
    "resp_rate":       (5.0,   60.0),
    "spo2":            (70.0,  100.0),
    "sbp":             (60.0,  220.0),
    "dbp":             (30.0,  140.0),
    "temperature_f":   (94.0,  107.0),
    "anchor_age":      (18.0,  100.0),
    "gender_enc":      (0.0,   1.0),
    "heartrate":       (30.0,  200.0),
    "temperature":     (94.0,  107.0),
    "o2sat":           (70.0,  100.0),
    "resprate":        (5.0,   60.0),
    "pain":            (0.0,   10.0),
    "acuity":          (1.0,   5.0),
}


class TabularFGSM:
    def __init__(self, model, feature_names, X_train,
                 epsilon=0.1, n_lime_samples=500):
        self.model = model
        self.feature_names = feature_names
        self.X_train = np.array(X_train)
        self.epsilon = epsilon
        self.n_lime_samples = n_lime_samples

        self.explainer = LimeTabularExplainer(
            training_data=self.X_train,
            feature_names=feature_names,
            mode="classification",
            discretize_continuous=False,
            random_state=42,
        )

        self.feature_bounds = []
        for feat in feature_names:
            if feat in CLINICAL_BOUNDS:
                self.feature_bounds.append(CLINICAL_BOUNDS[feat])
            else:
                col_idx = feature_names.index(feat)
                lo = float(self.X_train[:, col_idx].min())
                hi = float(self.X_train[:, col_idx].max())
                self.feature_bounds.append((lo, hi))

    def _get_lime_gradient(self, x):
        exp = self.explainer.explain_instance(
            x,
            self.model.predict_proba,
            num_features=len(self.feature_names),
            num_samples=self.n_lime_samples,
        )
        coef_map = dict(exp.as_list())
        grad = np.zeros(len(self.feature_names))
        for i, feat in enumerate(self.feature_names):
            for key, val in coef_map.items():
                if feat in key:
                    grad[i] = val
                    break
        return grad

    def perturb(self, x):
        x = np.array(x, dtype=float)
        original_pred = self.model.predict([x])[0]

        grad = self._get_lime_gradient(x)

        sign_grad = np.sign(grad)
        if original_pred == 1:
            sign_grad = -sign_grad

        x_adv = x + self.epsilon * sign_grad

        for i, (lo, hi) in enumerate(self.feature_bounds):
            x_adv[i] = np.clip(x_adv[i], lo, hi)

        return x_adv

    def attack(self, X_test, y_test=None):
        X_test = np.array(X_test)
        X_adv = np.zeros_like(X_test)

        for i, x in enumerate(X_test):
            X_adv[i] = self.perturb(x)
            if (i + 1) % 100 == 0:
                print(f"  Perturbed {i+1}/{len(X_test)} samples")

        orig_preds = self.model.predict(X_test)
        adv_preds = self.model.predict(X_adv)
        flip_mask = orig_preds != adv_preds
        flip_rate = flip_mask.mean()

        perturbation_mag = np.abs(X_adv - X_test)
        linf_norms = perturbation_mag.max(axis=1)
        mean_linf = linf_norms[flip_mask].mean() if flip_mask.any() else 0.0

        results = {
            "n_samples": len(X_test),
            "flip_rate": round(float(flip_rate), 4),
            "mean_linf_norm_flipped": round(float(mean_linf), 4),
            "epsilon": self.epsilon,
        }

        if y_test is not None:
            y_test = np.array(y_test)
            orig_prob = self.model.predict_proba(X_test)[:, 1]
            adv_prob = self.model.predict_proba(X_adv)[:, 1]
            auroc_clean = roc_auc_score(y_test, orig_prob)
            auroc_adv = roc_auc_score(y_test, adv_prob)
            results["auroc_clean"] = round(auroc_clean, 4)
            results["auroc_adversarial"] = round(auroc_adv, 4)
            results["auroc_drop"] = round(auroc_clean - auroc_adv, 4)

        plausible = self._check_clinical_plausibility(X_adv)
        results["clinical_plausibility_rate"] = round(float(plausible), 4)

        return X_adv, results

    def _check_clinical_plausibility(self, X_adv):
        in_bounds = np.ones(len(X_adv), dtype=bool)
        for i, (lo, hi) in enumerate(self.feature_bounds):
            in_bounds &= (X_adv[:, i] >= lo) & (X_adv[:, i] <= hi)
        return in_bounds.mean()

    def attack_with_defense(self, X_test, y_test=None):
        X_test = np.array(X_test)
        X_adv, results_no_defense = self.attack(X_test, y_test)

        X_defended = X_adv.copy()
        for i, (lo, hi) in enumerate(self.feature_bounds):
            X_defended[:, i] = np.clip(X_defended[:, i], lo, hi)

        defended_preds = self.model.predict(X_defended)
        orig_preds = self.model.predict(X_test)
        flip_rate_defended = (orig_preds != defended_preds).mean()
        results_no_defense["flip_rate_with_input_validation"] = round(
            float(flip_rate_defended), 4
        )

        if y_test is not None:
            y_test = np.array(y_test)
            defended_prob = self.model.predict_proba(X_defended)[:, 1]
            auroc_defended = roc_auc_score(y_test, defended_prob)
            results_no_defense["auroc_with_input_validation"] = round(
                auroc_defended, 4
            )

        return X_adv, X_defended, results_no_defense
