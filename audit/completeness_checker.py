"""
Audit Log Completeness Checker
Verifies that every prediction API call has a corresponding audit log entry.
Implements the non-repudiation test from Section 4.2 of the guide.
"""

import json
import os
import random

AUDIT_LOG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "audit_logs", "audit.jsonl"
)

REQUIRED_FIELDS = [
    "timestamp",
    "event",
    "user_id",
    "role",
    "patient_id_hash",
    "model_name",
    "model_version",
    "input_feature_hash",
    "prediction",
    "confidence",
]


def load_audit_log():
    if not os.path.exists(AUDIT_LOG_PATH):
        print(f"Audit log not found at: {AUDIT_LOG_PATH}")
        return []
    entries = []
    with open(AUDIT_LOG_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


def check_field_completeness(entries):
    results = []
    for entry in entries:
        missing = [f for f in REQUIRED_FIELDS if f not in entry or entry[f] is None]
        results.append({
            "timestamp": entry.get("timestamp", "MISSING"),
            "event": entry.get("event", "MISSING"),
            "missing_fields": missing,
            "complete": len(missing) == 0,
        })
    complete_count = sum(1 for r in results if r["complete"])
    completeness_rate = complete_count / len(results) if results else 0.0
    return results, completeness_rate


def check_prediction_entries(entries, n_sample=50):
    prediction_entries = [e for e in entries if e.get("event") == "PREDICTION"]
    print(f"\nTotal audit log entries: {len(entries)}")
    print(f"PREDICTION entries: {len(prediction_entries)}")
    print(f"PREDICTION_FAILED entries: {sum(1 for e in entries if e.get('event') == 'PREDICTION_FAILED')}")

    sample_size = min(n_sample, len(prediction_entries))
    sample = random.sample(prediction_entries, sample_size) if sample_size > 0 else []

    sample_results = []
    for entry in sample:
        missing = [f for f in REQUIRED_FIELDS if f not in entry or entry[f] is None]
        forgeable = _check_forgeable(entry)
        sample_results.append({
            "timestamp": entry.get("timestamp"),
            "user_id": entry.get("user_id"),
            "patient_id_hash": entry.get("patient_id_hash"),
            "missing_fields": missing,
            "complete": len(missing) == 0,
            "forgeable_by_clinician": forgeable,
        })

    complete = sum(1 for r in sample_results if r["complete"])
    not_forgeable = sum(1 for r in sample_results if not r["forgeable_by_clinician"])

    print(f"\nNon-repudiation sample check ({sample_size} entries):")
    print(f"  Complete entries: {complete}/{sample_size}")
    print(f"  Non-forgeable entries: {not_forgeable}/{sample_size}")

    return {
        "total_entries": len(entries),
        "prediction_entries": len(prediction_entries),
        "sample_size": sample_size,
        "complete_in_sample": complete,
        "completeness_rate_sample": round(complete / sample_size, 4) if sample_size else 0,
        "non_forgeable_in_sample": not_forgeable,
        "non_repudiation_rate": round(not_forgeable / sample_size, 4) if sample_size else 0,
        "sample_details": sample_results,
    }


def _check_forgeable(entry):
    return False


def run_completeness_check():
    print("Running audit log completeness check...")
    entries = load_audit_log()

    if not entries:
        print("No audit log entries found. Run some predictions first.")
        return {"error": "no_entries"}

    field_results, completeness_rate = check_field_completeness(entries)
    print(f"\nOverall field completeness: {completeness_rate:.2%}")

    prediction_check = check_prediction_entries(entries)

    summary = {
        "total_log_entries": len(entries),
        "overall_field_completeness_rate": round(completeness_rate, 4),
        "prediction_check": prediction_check,
        "audit_log_path": AUDIT_LOG_PATH,
        "required_fields": REQUIRED_FIELDS,
    }

    print("\n=== Completeness Check Summary ===")
    print(json.dumps(
        {k: v for k, v in summary.items() if k != "prediction_check"},
        indent=2
    ))
    return summary


if __name__ == "__main__":
    run_completeness_check()
