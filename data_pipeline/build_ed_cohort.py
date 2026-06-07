"""
ED Admission Cohort Builder
Joins MIMIC-IV-ED edstays and triage tables.
Label: disposition == ADMITTED -> 1, else 0
Features: triage vitals + demographics
"""

import gzip
import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

BASE_DATA = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "mimic"
)
ED_DIR = os.path.join(BASE_DATA, "ed_v2_2", "ed")
OUT_DIR = os.path.join(BASE_DATA, "ed_cohort")
os.makedirs(OUT_DIR, exist_ok=True)

FEATURE_COLS = [
    "heartrate", "temperature", "sbp", "dbp",
    "o2sat", "resprate", "pain", "acuity", "gender_enc"
]

ADMITTED_DISPOSITIONS = {"ADMITTED"}


def load_edstays():
    path = os.path.join(ED_DIR, "edstays.csv.gz")
    df = pd.read_csv(path, compression="gzip")
    df = df[["subject_id", "stay_id", "hadm_id",
             "gender", "race", "intime", "outtime", "disposition"]]
    df["label"] = df["disposition"].apply(
        lambda x: 1 if str(x).upper().strip() in ADMITTED_DISPOSITIONS else 0
    )
    print(f"  ED stays loaded: {len(df):,}")
    print(f"  Admission rate: {df['label'].mean():.2%}")
    return df


def load_triage():
    path = os.path.join(ED_DIR, "triage.csv.gz")
    df = pd.read_csv(path, compression="gzip")
    df = df[["subject_id", "stay_id", "heartrate", "temperature",
             "sbp", "dbp", "o2sat", "resprate", "pain", "acuity"]]
    print(f"  Triage rows loaded: {len(df):,}")
    return df


def build_cohort():
    print("Building ED admission cohort...")

    stays = load_edstays()
    triage = load_triage()

    cohort = stays.merge(triage, on=["subject_id", "stay_id"], how="inner")
    print(f"  After triage join: {len(cohort):,} stays")

    cohort["gender_enc"] = (cohort["gender"].str.upper() == "M").astype(int)

    vital_cols = ["heartrate", "temperature", "sbp", "dbp",
                  "o2sat", "resprate", "pain", "acuity"]
    cohort = cohort.dropna(subset=vital_cols)
    print(f"  After dropping missing vitals: {len(cohort):,} stays")

    cohort["pain"] = pd.to_numeric(cohort["pain"], errors="coerce")
    cohort = cohort.dropna(subset=["pain"])
    print(f"  After cleaning pain field: {len(cohort):,} stays")

    X = cohort[FEATURE_COLS].copy()
    meta = cohort[["stay_id", "subject_id", "race", "label"]].copy()

    X_train, X_temp, m_train, m_temp = train_test_split(
        X, meta, test_size=0.30, random_state=42, stratify=meta["label"]
    )
    X_val, X_test, m_val, m_test = train_test_split(
        X_temp, m_temp, test_size=0.50, random_state=42,
        stratify=m_temp["label"]
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=FEATURE_COLS, index=X_train.index
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val), columns=FEATURE_COLS, index=X_val.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=FEATURE_COLS, index=X_test.index
    )

    X_train_scaled.to_csv(os.path.join(OUT_DIR, "train_X.csv"), index=False)
    X_val_scaled.to_csv(os.path.join(OUT_DIR, "val_X.csv"), index=False)
    X_test_scaled.to_csv(os.path.join(OUT_DIR, "test_X.csv"), index=False)

    X_train.to_csv(os.path.join(OUT_DIR, "train_X_raw.csv"), index=False)
    X_test.to_csv(os.path.join(OUT_DIR, "test_X_raw.csv"), index=False)

    m_train[["label"]].to_csv(os.path.join(OUT_DIR, "train_y.csv"), index=False)
    m_val[["label"]].to_csv(os.path.join(OUT_DIR, "val_y.csv"), index=False)
    m_test[["label"]].to_csv(os.path.join(OUT_DIR, "test_y.csv"), index=False)

    m_test.to_csv(os.path.join(OUT_DIR, "test_demographics.csv"), index=False)

    with open(os.path.join(OUT_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    with open(os.path.join(OUT_DIR, "feature_names.txt"), "w") as f:
        f.write("\n".join(FEATURE_COLS))

    print(f"\nED cohort saved to: {OUT_DIR}")
    print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    print(f"  Features: {FEATURE_COLS}")
    return X_train, X_val, X_test, m_train, m_val, m_test


if __name__ == "__main__":
    build_cohort()
