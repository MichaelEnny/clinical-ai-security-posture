"""
Runs only the ED admission membership inference attack (shadow model approach).
Uses a capped pool of 5,000 training samples for tractable shadow model training.
"""

import os, sys, json, pickle, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CODE_DIR)
sys.path.insert(0, os.path.join(CODE_DIR, "adversarial"))

MODEL_DIR = os.path.join(CODE_DIR, "system", "models")
DATA_DIR  = os.path.join(ROOT_DIR, "data", "mimic", "ed_cohort")

from membership_inference import MembershipInferenceAttack

def main():
    print("Loading ED admission model and data...", flush=True)
    with open(os.path.join(MODEL_DIR, "ed_admission_rf.pkl"), "rb") as f:
        model = pickle.load(f)

    X_train = pd.read_csv(os.path.join(DATA_DIR, "train_X.csv")).values
    y_train = pd.read_csv(os.path.join(DATA_DIR, "train_y.csv")).values.ravel()
    X_test  = pd.read_csv(os.path.join(DATA_DIR, "test_X.csv")).values
    y_test  = pd.read_csv(os.path.join(DATA_DIR, "test_y.csv")).values.ravel()

    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}", flush=True)

    rng = np.random.RandomState(42)
    pool_size = min(5000, len(X_train))
    idx = rng.choice(len(X_train), pool_size, replace=False)
    X_pool = X_train[idx]
    y_pool = y_train[idx]

    print(f"\n[MI] shadow_models=4, pool={pool_size}", flush=True)

    mia = MembershipInferenceAttack(
        target_model=model,
        n_shadow_models=4,
        random_state=42,
    )
    mi_res = mia.run_full_attack(X_pool, y_pool, X_test, y_test)

    print(f"\nResult JSON: {json.dumps(mi_res, indent=2)}", flush=True)

    # Save result
    out = os.path.join(ROOT_DIR, "tables", "ed_mi_result.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(mi_res, f, indent=2)
    print(f"\nSaved: {out}", flush=True)

if __name__ == "__main__":
    main()
