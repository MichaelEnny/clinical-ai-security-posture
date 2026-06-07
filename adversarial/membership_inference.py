"""
Shadow Model Membership Inference Attack
Implements Shokri et al. (2017) adapted for tabular clinical data.
Determines whether an attacker can identify if a patient's record
was in the training set by querying the prediction API.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report
)
from sklearn.model_selection import train_test_split


class MembershipInferenceAttack:
    def __init__(self, target_model, n_shadow_models=4,
                 shadow_model_size=0.5, random_state=42):
        self.target_model = target_model
        self.n_shadow_models = n_shadow_models
        self.shadow_model_size = shadow_model_size
        self.random_state = random_state
        self.attack_model = None

    def _train_shadow_model(self, X, y, seed):
        X = np.array(X)
        y = np.array(y)
        n = int(len(X) * self.shadow_model_size)
        idx = np.random.RandomState(seed).choice(len(X), size=n, replace=False)
        mask = np.zeros(len(X), dtype=bool)
        mask[idx] = True

        X_shadow_train = X[mask]
        y_shadow_train = y[mask]
        X_shadow_out = X[~mask]
        y_shadow_out = y[~mask]

        shadow = RandomForestClassifier(
            n_estimators=100, max_depth=8,
            class_weight="balanced", random_state=seed
        )
        shadow.fit(X_shadow_train, y_shadow_train)

        in_probs = shadow.predict_proba(X_shadow_train)
        out_probs = shadow.predict_proba(X_shadow_out)

        in_labels = np.ones(len(in_probs))
        out_labels = np.zeros(len(out_probs))

        attack_X = np.vstack([in_probs, out_probs])
        attack_y = np.concatenate([in_labels, out_labels])
        return attack_X, attack_y

    def train_attack_model(self, X_pool, y_pool):
        X_pool = np.array(X_pool)
        y_pool = np.array(y_pool)

        all_attack_X = []
        all_attack_y = []

        for i in range(self.n_shadow_models):
            ax, ay = self._train_shadow_model(
                X_pool, y_pool, seed=self.random_state + i
            )
            all_attack_X.append(ax)
            all_attack_y.append(ay)

        attack_X = np.vstack(all_attack_X)
        attack_y = np.concatenate(all_attack_y)

        self.attack_model = RandomForestClassifier(
            n_estimators=100, random_state=self.random_state
        )
        self.attack_model.fit(attack_X, attack_y)
        print(f"  Attack model trained on {len(attack_X):,} shadow model outputs")

    def infer(self, X_members, X_nonmembers):
        X_members = np.array(X_members)
        X_nonmembers = np.array(X_nonmembers)

        member_probs = self.target_model.predict_proba(X_members)
        nonmember_probs = self.target_model.predict_proba(X_nonmembers)

        attack_X = np.vstack([member_probs, nonmember_probs])
        true_labels = np.concatenate([
            np.ones(len(X_members)),
            np.zeros(len(X_nonmembers))
        ])

        predicted = self.attack_model.predict(attack_X)
        predicted_proba = self.attack_model.predict_proba(attack_X)[:, 1]

        accuracy = accuracy_score(true_labels, predicted)
        auroc = roc_auc_score(true_labels, predicted_proba)

        n_members_at_risk = int(accuracy * len(X_members))

        results = {
            "n_members_tested": len(X_members),
            "n_nonmembers_tested": len(X_nonmembers),
            "membership_inference_accuracy": round(float(accuracy), 4),
            "membership_inference_auroc": round(float(auroc), 4),
            "baseline_accuracy": 0.50,
            "above_baseline": accuracy > 0.55,
            "privacy_risk": "HIGH" if accuracy > 0.60 else (
                "MODERATE" if accuracy > 0.55 else "LOW"
            ),
            "estimated_patients_at_risk": n_members_at_risk,
        }

        print(f"\nMembership Inference Results:")
        print(f"  Accuracy: {accuracy:.4f} (baseline: 0.50)")
        print(f"  AUROC:    {auroc:.4f}")
        print(f"  Privacy risk: {results['privacy_risk']}")
        print(f"  Estimated patients at re-identification risk: {n_members_at_risk:,}")

        return results

    def run_full_attack(self, X_train, y_train, X_test, y_test):
        print("Training shadow models for membership inference attack...")
        self.train_attack_model(X_train, y_train)

        print("Running membership inference on target model outputs...")
        n_eval = min(500, len(X_train), len(X_test))
        idx_member = np.random.choice(len(X_train), n_eval, replace=False)
        idx_nonmember = np.random.choice(len(X_test), n_eval, replace=False)

        return self.infer(
            np.array(X_train)[idx_member],
            np.array(X_test)[idx_nonmember]
        )
