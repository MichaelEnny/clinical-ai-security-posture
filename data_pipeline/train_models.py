"""
Model Training Script
Trains Random Forest classifiers on all datasets:
  - MIMIC-IV ICU (primary: mortality prediction)
  - MIMIC-IV-ED (primary: admission prediction)
  - UCI CKD (pilot)
  - Wisconsin WDBC (pilot)
Saves models as .pkl files in code/system/models/
"""

import os
import pickle
import json

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, classification_report
)

BASE = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
)))
DATA_DIR = os.path.join(BASE, "data")
MODEL_DIR = os.path.join(BASE, "code", "system", "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def train_and_save(name, X_train, y_train, X_test, y_test,
                   feature_names, label_name="label"):
    print(f"\nTraining {name}...")
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"  Class distribution (train): {dict(pd.Series(y_train).value_counts())}")

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    y_prob = rf.predict_proba(X_test)[:, 1]
    y_pred = rf.predict(X_test)
    auroc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)

    print(f"  AUROC: {auroc:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, digits=3))

    model_path = os.path.join(MODEL_DIR, f"{name}_rf.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(rf, f)

    meta = {
        "model_name": name,
        "model_type": "RandomForestClassifier",
        "n_estimators": 200,
        "feature_names": feature_names,
        "label_name": label_name,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "auroc": round(auroc, 4),
        "accuracy": round(acc, 4),
        "model_version": "1.0",
    }
    meta_path = os.path.join(MODEL_DIR, f"{name}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved: {model_path}")
    return rf, auroc


def load_cohort(cohort_dir):
    X_train = pd.read_csv(os.path.join(cohort_dir, "train_X.csv"))
    X_test = pd.read_csv(os.path.join(cohort_dir, "test_X.csv"))
    y_train = pd.read_csv(os.path.join(cohort_dir, "train_y.csv")).iloc[:, 0]
    y_test = pd.read_csv(os.path.join(cohort_dir, "test_y.csv")).iloc[:, 0]
    with open(os.path.join(cohort_dir, "feature_names.txt")) as f:
        features = [line.strip() for line in f.readlines()]
    return X_train, X_test, y_train, y_test, features


def prepare_ckd():
    full = pd.read_csv(os.path.join(DATA_DIR, "ckd", "ckd_full.csv"))
    full["label"] = full["class"].str.strip().apply(
        lambda x: 1 if x == "ckd" else 0
    )
    full = full.drop(columns=["class"])

    cat_cols = full.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        full[col] = full[col].fillna("missing")
        full[col] = le.fit_transform(full[col].astype(str))

    num_cols = full.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != "label"]
    for col in num_cols:
        full[col] = full[col].fillna(full[col].median())

    feature_cols = [c for c in full.columns if c != "label"]
    X = full[feature_cols]
    y = full["label"]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_cols
    )
    X_test_s = pd.DataFrame(
        scaler.transform(X_test), columns=feature_cols
    )

    ckd_out = os.path.join(DATA_DIR, "ckd", "processed")
    os.makedirs(ckd_out, exist_ok=True)
    X_train_s.to_csv(os.path.join(ckd_out, "train_X.csv"), index=False)
    X_test_s.to_csv(os.path.join(ckd_out, "test_X.csv"), index=False)
    y_train.to_csv(os.path.join(ckd_out, "train_y.csv"), index=False)
    y_test.to_csv(os.path.join(ckd_out, "test_y.csv"), index=False)
    with open(os.path.join(ckd_out, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(ckd_out, "feature_names.txt"), "w") as f:
        f.write("\n".join(feature_cols))

    return X_train_s, X_test_s, y_train.reset_index(drop=True), \
           y_test.reset_index(drop=True), feature_cols


def prepare_wdbc():
    full = pd.read_csv(os.path.join(DATA_DIR, "breast_cancer", "wdbc_full.csv"))
    feature_cols = [c for c in full.columns if c != "label"]
    X = full[feature_cols]
    y = full["label"]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_cols
    )
    X_test_s = pd.DataFrame(
        scaler.transform(X_test), columns=feature_cols
    )

    wdbc_out = os.path.join(DATA_DIR, "breast_cancer", "processed")
    os.makedirs(wdbc_out, exist_ok=True)
    X_train_s.to_csv(os.path.join(wdbc_out, "train_X.csv"), index=False)
    X_test_s.to_csv(os.path.join(wdbc_out, "test_X.csv"), index=False)
    y_train.to_csv(os.path.join(wdbc_out, "train_y.csv"), index=False)
    y_test.to_csv(os.path.join(wdbc_out, "test_y.csv"), index=False)
    with open(os.path.join(wdbc_out, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(wdbc_out, "feature_names.txt"), "w") as f:
        f.write("\n".join(feature_cols))

    return X_train_s, X_test_s, y_train.reset_index(drop=True), \
           y_test.reset_index(drop=True), feature_cols


def main():
    results = {}

    icu_dir = os.path.join(DATA_DIR, "mimic", "icu_cohort")
    if os.path.exists(os.path.join(icu_dir, "train_X.csv")):
        X_tr, X_te, y_tr, y_te, feats = load_cohort(icu_dir)
        _, auroc = train_and_save("icu_mortality", X_tr, y_tr, X_te, y_te, feats)
        results["icu_mortality"] = auroc
    else:
        print("ICU cohort not found. Run build_icu_cohort.py first.")

    ed_dir = os.path.join(DATA_DIR, "mimic", "ed_cohort")
    if os.path.exists(os.path.join(ed_dir, "train_X.csv")):
        X_tr, X_te, y_tr, y_te, feats = load_cohort(ed_dir)
        _, auroc = train_and_save("ed_admission", X_tr, y_tr, X_te, y_te, feats)
        results["ed_admission"] = auroc
    else:
        print("ED cohort not found. Run build_ed_cohort.py first.")

    print("\nPreparing CKD pilot dataset...")
    X_tr, X_te, y_tr, y_te, feats = prepare_ckd()
    _, auroc = train_and_save("ckd", X_tr, y_tr, X_te, y_te, feats)
    results["ckd"] = auroc

    print("\nPreparing WDBC pilot dataset...")
    X_tr, X_te, y_tr, y_te, feats = prepare_wdbc()
    _, auroc = train_and_save("wdbc", X_tr, y_tr, X_te, y_te, feats)
    results["wdbc"] = auroc

    print("\n=== Training Summary ===")
    for name, auroc in results.items():
        print(f"  {name}: AUROC = {auroc:.4f}")

    summary_path = os.path.join(MODEL_DIR, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
