"""
Iterative Boundary Attack for Tabular Data
Moves test samples toward the decision boundary and measures
how many steps are required to flip the model's prediction.
"""

import numpy as np
from sklearn.metrics import roc_auc_score


class BoundaryAttack:
    def __init__(self, model, feature_names, feature_bounds=None,
                 step_size=0.05, max_steps=100):
        self.model = model
        self.feature_names = feature_names
        self.step_size = step_size
        self.max_steps = max_steps

        if feature_bounds is None:
            self.feature_bounds = [(None, None)] * len(feature_names)
        else:
            self.feature_bounds = feature_bounds

    def _clip(self, x):
        x = x.copy()
        for i, (lo, hi) in enumerate(self.feature_bounds):
            if lo is not None:
                x[i] = max(x[i], lo)
            if hi is not None:
                x[i] = min(x[i], hi)
        return x

    def _find_opposite_class_sample(self, x, X_pool):
        orig_pred = self.model.predict([x])[0]
        for candidate in X_pool:
            if self.model.predict([candidate])[0] != orig_pred:
                return candidate
        return None

    def attack_single(self, x, X_pool):
        x = np.array(x, dtype=float)
        orig_pred = self.model.predict([x])[0]

        opposite = self._find_opposite_class_sample(x, X_pool)
        if opposite is None:
            return x, 0, False

        current = x.copy()
        for step in range(1, self.max_steps + 1):
            direction = opposite - current
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                break
            direction = direction / norm
            candidate = self._clip(current + self.step_size * direction)
            new_pred = self.model.predict([candidate])[0]
            if new_pred != orig_pred:
                perturbation = np.max(np.abs(candidate - x))
                return candidate, step, True
            current = candidate

        perturbation = np.max(np.abs(current - x))
        return current, self.max_steps, False

    def attack(self, X_test, y_test=None, X_pool=None):
        X_test = np.array(X_test)
        if X_pool is None:
            X_pool = X_test

        results_per_sample = []
        X_adv = X_test.copy()

        for i, x in enumerate(X_test):
            x_adv, steps, flipped = self.attack_single(x, X_pool)
            X_adv[i] = x_adv
            results_per_sample.append({
                "steps_to_flip": steps,
                "flipped": flipped,
                "perturbation_linf": float(np.max(np.abs(x_adv - x))),
            })
            if (i + 1) % 100 == 0:
                print(f"  Boundary attack: {i+1}/{len(X_test)} samples")

        flip_rate = np.mean([r["flipped"] for r in results_per_sample])
        steps_to_flip = [
            r["steps_to_flip"] for r in results_per_sample if r["flipped"]
        ]
        mean_steps = np.mean(steps_to_flip) if steps_to_flip else self.max_steps
        mean_linf = np.mean([r["perturbation_linf"] for r in results_per_sample])

        summary = {
            "n_samples": len(X_test),
            "flip_rate": round(float(flip_rate), 4),
            "mean_steps_to_flip": round(float(mean_steps), 2),
            "mean_perturbation_linf": round(float(mean_linf), 4),
            "step_size": self.step_size,
            "max_steps": self.max_steps,
        }

        if y_test is not None:
            y_test = np.array(y_test)
            orig_prob = self.model.predict_proba(X_test)[:, 1]
            adv_prob = self.model.predict_proba(X_adv)[:, 1]
            summary["auroc_clean"] = round(roc_auc_score(y_test, orig_prob), 4)
            summary["auroc_adversarial"] = round(roc_auc_score(y_test, adv_prob), 4)
            summary["auroc_drop"] = round(
                summary["auroc_clean"] - summary["auroc_adversarial"], 4
            )

        return X_adv, summary
