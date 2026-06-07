"""
Threat Coverage Scorer
Computes the Threat Coverage dimension of SPS from MITRE ATT&CK results.
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from mitre_mapper import get_vectors, get_coverage_stats, record_result


def run_coverage_scoring():
    vectors = get_vectors()
    stats = get_coverage_stats()

    print("Threat Coverage Scoring")
    print(f"  Total techniques mapped:        {stats['total_techniques']}")
    print(f"  Mitigations implemented:        {stats['mitigations_implemented']}")
    print(f"  Techniques tested:              {stats['techniques_tested']}")
    print(f"  Results: {stats['passed']} PASS | {stats['partial']} PARTIAL")
    print(f"  Coverage fraction: {stats['coverage_fraction']:.4f} ({stats['coverage_pct']}%)")

    threat_subscore = stats["coverage_fraction"] * 100

    result = {
        "threat_coverage_stats": stats,
        "threat_subscore_raw": round(threat_subscore, 2),
        "vectors": vectors,
    }
    return result


def export_mitre_table(output_path: str):
    vectors = get_vectors()
    rows = []
    for v in vectors:
        rows.append({
            "ID": v["id"],
            "Technique": v["technique"],
            "Tactic": v["tactic"],
            "System Layer": v["system_layer"],
            "Mitigation": v["mitigation"],
            "Implemented": "Yes" if v["mitigation_implemented"] else "No",
            "Tested": "Yes" if v["tested"] else "No",
            "Result": v["result"] or "PENDING",
            "Notes": v["notes"],
        })

    import csv
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"MITRE ATT&CK table exported to: {output_path}")


if __name__ == "__main__":
    result = run_coverage_scoring()
    print(json.dumps(result["threat_coverage_stats"], indent=2))

    table_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "tables", "mitre_mapping.csv"
    )
    os.makedirs(os.path.dirname(table_path), exist_ok=True)
    export_mitre_table(table_path)
