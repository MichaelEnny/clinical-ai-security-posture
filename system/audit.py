"""
Append-only audit logging.
Every prediction call (success or failure) writes a tamper-evident log entry.
Log entries include all fields required for HIPAA non-repudiation.
"""

import os
import json
import datetime
import threading
from typing import Optional

AUDIT_LOG_PATH = os.environ.get(
    "AUDIT_LOG_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "audit_logs", "audit.jsonl")
)

_lock = threading.Lock()


def write_audit_log(
    event: str,
    user_id: str,
    role: str,
    patient_id_hash: str,
    model_name: str,
    model_version: str,
    input_hash: str,
    prediction: Optional[int],
    confidence: Optional[float],
    detail: str = "",
):
    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "event": event,
        "user_id": user_id,
        "role": role,
        "patient_id_hash": patient_id_hash,
        "model_name": model_name,
        "model_version": model_version,
        "input_feature_hash": input_hash,
        "prediction": prediction,
        "confidence": confidence,
        "detail": detail,
    }

    os.makedirs(os.path.dirname(AUDIT_LOG_PATH), exist_ok=True)

    with _lock:
        with open(AUDIT_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")


def get_audit_logs(limit: int = 1000) -> list:
    if not os.path.exists(AUDIT_LOG_PATH):
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
    return entries[-limit:]
