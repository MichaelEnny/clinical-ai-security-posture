"""
ICU Mortality Cohort Builder
Joins MIMIC-IV icustays, admissions, patients, and chartevents
to produce a clean train/val/test split for ICU mortality prediction.
Label: hospital_expire_flag (died in hospital = 1, survived = 0)
Features: first-24-hour mean vitals from chartevents
"""

import os
import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DATA = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "mimic"
)
OUT_DIR = os.path.join(BASE_DATA, "icu_cohort")
os.makedirs(OUT_DIR, exist_ok=True)

VITAL_ITEMS = {
    220045: "heart_rate",
    220210: "resp_rate",
    220277: "spo2",
    220179: "sbp",
    220180: "dbp",
    223761: "temperature_f",
}

FIRST_24H_MINUTES = 1440
CHUNK_SIZE = 2_000_000


def load_icustays():
    path = os.path.join(BASE_DATA, "icu", "icustays.csv.gz")
    df = pd.read_csv(path, compression="gzip")
    df = df[["subject_id", "hadm_id", "stay_id", "intime", "outtime", "los"]]
    df["intime"] = pd.to_datetime(df["intime"])
    print(f"  ICU stays loaded: {len(df):,}")
    return df


def load_admissions():
    path = os.path.join(BASE_DATA, "hosp", "admissions.csv.gz")
    df = pd.read_csv(path, compression="gzip")
    df = df[["subject_id", "hadm_id", "admission_type", "race",
             "insurance", "hospital_expire_flag"]]
    print(f"  Admissions loaded: {len(df):,}")
    return df


def load_patients():
    path = os.path.join(BASE_DATA, "hosp", "patients.csv.gz")
    df = pd.read_csv(path, compression="gzip")
    df = df[["subject_id", "gender", "anchor_age"]]
    print(f"  Patients loaded: {len(df):,}")
    return df


def extract_first24h_vitals(stay_intime: dict):
    """
    Read chartevents in chunks using pandas.
    stay_intime: dict mapping stay_id (int) -> intime (datetime)
    Returns DataFrame with mean first-24h vitals per stay.
    """
    path = os.path.join(BASE_DATA, "icu", "chartevents.csv.gz")
    vital_item_ids = set(VITAL_ITEMS.keys())
    stay_ids_set = set(stay_intime.keys())

    all_chunks = []
    chunk_num = 0

    print("  Processing chartevents in chunks (this takes several minutes)...")

    for chunk in pd.read_csv(
        path,
        compression="gzip",
        chunksize=CHUNK_SIZE,
        usecols=["stay_id", "charttime", "itemid", "valuenum"],
        dtype={"stay_id": "Int64", "itemid": "Int64", "valuenum": "float32"},
    ):
        chunk_num += 1
        if chunk_num % 50 == 0:
            print(f"    Processed {chunk_num * CHUNK_SIZE:,} rows...")

        chunk = chunk[chunk["stay_id"].isin(stay_ids_set)]
        if chunk.empty:
            continue

        chunk = chunk[chunk["itemid"].isin(vital_item_ids)]
        if chunk.empty:
            continue

        chunk = chunk.dropna(subset=["valuenum"])
        if chunk.empty:
            continue

        chunk["intime"] = chunk["stay_id"].map(stay_intime)
        chunk["charttime"] = pd.to_datetime(chunk["charttime"])
        chunk["offset_min"] = (
            chunk["charttime"] - chunk["intime"]
        ).dt.total_seconds() / 60.0

        chunk = chunk[(chunk["offset_min"] >= 0) & (chunk["offset_min"] <= FIRST_24H_MINUTES)]
        if chunk.empty:
            continue

        all_chunks.append(chunk[["stay_id", "itemid", "valuenum"]])

    print(f"  Finished reading chartevents ({chunk_num * CHUNK_SIZE:,}+ rows total)")

    if not all_chunks:
        print("  WARNING: No vitals found. Check item IDs and stay_id types.")
        return pd.DataFrame(columns=["stay_id"] + list(VITAL_ITEMS.values()))

    vitals_raw = pd.concat(all_chunks, ignore_index=True)
    print(f"  Vital rows collected: {len(vitals_raw):,}")
    print(f"  Stays with at least one vital: {vitals_raw['stay_id'].nunique():,}")

    vitals_raw["vital_name"] = vitals_raw["itemid"].map(VITAL_ITEMS)
    vitals_pivot = (
        vitals_raw.groupby(["stay_id", "vital_name"])["valuenum"]
        .mean()
        .unstack("vital_name")
        .reset_index()
    )

    for col in VITAL_ITEMS.values():
        if col not in vitals_pivot.columns:
            vitals_pivot[col] = np.nan

    return vitals_pivot


def build_cohort():
    print("Building ICU mortality cohort...")

    stays = load_icustays()
    admissions = load_admissions()
    patients = load_patients()

    cohort = stays.merge(admissions, on=["subject_id", "hadm_id"], how="inner")
    cohort = cohort.merge(patients, on="subject_id", how="inner")
    cohort = cohort[cohort["anchor_age"] >= 18].copy()
    print(f"  After adult filter: {len(cohort):,} stays")

    stay_intime = dict(zip(
        cohort["stay_id"].astype(int),
        cohort["intime"]
    ))

    vitals_df = extract_first24h_vitals(stay_intime)
    vitals_df["stay_id"] = vitals_df["stay_id"].astype(int)
    cohort["stay_id"] = cohort["stay_id"].astype(int)

    cohort = cohort.merge(vitals_df, on="stay_id", how="inner")
    print(f"  After vitals join: {len(cohort):,} stays")

    vital_cols = list(VITAL_ITEMS.values())
    cohort = cohort.dropna(subset=vital_cols)
    print(f"  After dropping missing vitals: {len(cohort):,} stays")

    cohort["gender_enc"] = (cohort["gender"] == "M").astype(int)

    feature_cols = vital_cols + ["anchor_age", "gender_enc"]
    label_col = "hospital_expire_flag"

    print(f"  Mortality rate: {cohort[label_col].mean():.2%}")

    X = cohort[feature_cols].copy()
    y = cohort[[label_col, "stay_id", "subject_id", "race"]].copy()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y[label_col]
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42,
        stratify=y_temp[label_col]
    )

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index
    )
    X_val_s = pd.DataFrame(
        scaler.transform(X_val), columns=feature_cols, index=X_val.index
    )
    X_test_s = pd.DataFrame(
        scaler.transform(X_test), columns=feature_cols, index=X_test.index
    )

    X_train_s.to_csv(os.path.join(OUT_DIR, "train_X.csv"), index=False)
    X_val_s.to_csv(os.path.join(OUT_DIR, "val_X.csv"), index=False)
    X_test_s.to_csv(os.path.join(OUT_DIR, "test_X.csv"), index=False)
    X_train.to_csv(os.path.join(OUT_DIR, "train_X_raw.csv"), index=False)
    X_test.to_csv(os.path.join(OUT_DIR, "test_X_raw.csv"), index=False)

    y_train[[label_col]].to_csv(os.path.join(OUT_DIR, "train_y.csv"), index=False)
    y_val[[label_col]].to_csv(os.path.join(OUT_DIR, "val_y.csv"), index=False)
    y_test[[label_col]].to_csv(os.path.join(OUT_DIR, "test_y.csv"), index=False)
    y_test[["stay_id", "subject_id", "race", label_col]].to_csv(
        os.path.join(OUT_DIR, "test_demographics.csv"), index=False
    )

    with open(os.path.join(OUT_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    with open(os.path.join(OUT_DIR, "feature_names.txt"), "w") as f:
        f.write("\n".join(feature_cols))

    print(f"\nCohort saved to: {OUT_DIR}")
    print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    print(f"  Features: {feature_cols}")


if __name__ == "__main__":
    build_cohort()
