"""
Security Posture Score (SPS) Computation
Composite score (0-100) across four dimensions:
  1. Adversarial Input Robustness  (35%)
  2. Threat Coverage               (25%)
  3. Access Control Integrity      (20%)
  4. Compliance Posture            (20%)

SPS >= 65: conditional deployment
SPS >= 80: production deployment
"""

import json
import datetime


WEIGHTS = {
    "adversarial": 0.35,
    "threat":      0.25,
    "access":      0.20,
    "compliance":  0.20,
}

THRESHOLDS = {
    "conditional_deployment": 65,
    "production_deployment":  80,
}


def compute_adversarial_subscore(auroc_drop: float) -> float:
    robustness = max(0.0, 1.0 - auroc_drop)
    return round(robustness * 100, 2)


def compute_threat_subscore(mitigated: int, tested: int, total: int,
                            passed: int, partial: int) -> float:
    effective = passed + 0.5 * partial
    coverage = effective / total if total > 0 else 0.0
    return round(coverage * 100, 2)


def compute_access_subscore(audit_completeness_rate: float,
                            rbac_enforcement_rate: float) -> float:
    combined = (audit_completeness_rate + rbac_enforcement_rate) / 2.0
    return round(combined * 100, 2)


def compute_compliance_subscore(compliance_score: float) -> float:
    return round(compliance_score * 100, 2)


def compute_sps(
    auroc_drop: float,
    threat_mitigated: int,
    threat_tested: int,
    threat_total: int,
    threat_passed: int,
    threat_partial: int,
    audit_completeness_rate: float,
    rbac_enforcement_rate: float,
    compliance_score: float,
) -> dict:

    d1 = compute_adversarial_subscore(auroc_drop)
    d2 = compute_threat_subscore(
        threat_mitigated, threat_tested, threat_total,
        threat_passed, threat_partial
    )
    d3 = compute_access_subscore(audit_completeness_rate, rbac_enforcement_rate)
    d4 = compute_compliance_subscore(compliance_score)

    sps = (
        WEIGHTS["adversarial"] * d1
        + WEIGHTS["threat"]    * d2
        + WEIGHTS["access"]    * d3
        + WEIGHTS["compliance"] * d4
    )
    sps = round(sps, 2)

    if sps >= THRESHOLDS["production_deployment"]:
        deployment_recommendation = "PRODUCTION DEPLOYMENT RECOMMENDED"
    elif sps >= THRESHOLDS["conditional_deployment"]:
        deployment_recommendation = "CONDITIONAL DEPLOYMENT (address gaps before full rollout)"
    else:
        deployment_recommendation = "NOT RECOMMENDED FOR DEPLOYMENT"

    return {
        "sps": sps,
        "deployment_recommendation": deployment_recommendation,
        "dimensions": {
            "adversarial_input_robustness": {
                "subscore": d1,
                "weight": WEIGHTS["adversarial"],
                "weighted_contribution": round(WEIGHTS["adversarial"] * d1, 2),
                "inputs": {"auroc_drop": auroc_drop},
            },
            "threat_coverage": {
                "subscore": d2,
                "weight": WEIGHTS["threat"],
                "weighted_contribution": round(WEIGHTS["threat"] * d2, 2),
                "inputs": {
                    "mitigated": threat_mitigated,
                    "tested": threat_tested,
                    "total": threat_total,
                    "passed": threat_passed,
                    "partial": threat_partial,
                },
            },
            "access_control_integrity": {
                "subscore": d3,
                "weight": WEIGHTS["access"],
                "weighted_contribution": round(WEIGHTS["access"] * d3, 2),
                "inputs": {
                    "audit_completeness_rate": audit_completeness_rate,
                    "rbac_enforcement_rate": rbac_enforcement_rate,
                },
            },
            "compliance_posture": {
                "subscore": d4,
                "weight": WEIGHTS["compliance"],
                "weighted_contribution": round(WEIGHTS["compliance"] * d4, 2),
                "inputs": {"compliance_score": compliance_score},
            },
        },
        "thresholds": THRESHOLDS,
        "computed_at": datetime.datetime.utcnow().isoformat(),
    }


def sensitivity_analysis(base_inputs: dict) -> list:
    base_result = compute_sps(**base_inputs)
    base_sps = base_result["sps"]

    ablations = []

    no_adversarial_defense = base_inputs.copy()
    no_adversarial_defense["auroc_drop"] = 0.25
    r = compute_sps(**no_adversarial_defense)
    ablations.append({
        "scenario": "Adversarial AUROC drop fixed at 0.25",
        "sps": r["sps"],
        "sps_drop": round(base_sps - r["sps"], 2),
        "dimension_affected": "adversarial_input_robustness",
    })

    no_threat_coverage = base_inputs.copy()
    no_threat_coverage["threat_passed"] = 0
    no_threat_coverage["threat_partial"] = 0
    r = compute_sps(**no_threat_coverage)
    ablations.append({
        "scenario": "No threat mitigations tested",
        "sps": r["sps"],
        "sps_drop": round(base_sps - r["sps"], 2),
        "dimension_affected": "threat_coverage",
    })

    no_audit = base_inputs.copy()
    no_audit["audit_completeness_rate"] = 0.60
    no_audit["rbac_enforcement_rate"] = 0.80
    r = compute_sps(**no_audit)
    ablations.append({
        "scenario": "Degraded audit and RBAC",
        "sps": r["sps"],
        "sps_drop": round(base_sps - r["sps"], 2),
        "dimension_affected": "access_control_integrity",
    })

    no_compliance = base_inputs.copy()
    no_compliance["compliance_score"] = 0.50
    r = compute_sps(**no_compliance)
    ablations.append({
        "scenario": "Minimal compliance posture",
        "sps": r["sps"],
        "sps_drop": round(base_sps - r["sps"], 2),
        "dimension_affected": "compliance_posture",
    })

    no_defense_baseline = {
        "auroc_drop": 0.30,
        "threat_mitigated": 0,
        "threat_tested": 0,
        "threat_total": base_inputs["threat_total"],
        "threat_passed": 0,
        "threat_partial": 0,
        "audit_completeness_rate": 0.50,
        "rbac_enforcement_rate": 0.50,
        "compliance_score": 0.40,
    }
    r = compute_sps(**no_defense_baseline)
    ablations.append({
        "scenario": "No-defense baseline (lower bound)",
        "sps": r["sps"],
        "sps_drop": round(base_sps - r["sps"], 2),
        "dimension_affected": "all",
    })

    return ablations


def print_report(result: dict):
    print("\n" + "=" * 50)
    print("  SECURITY POSTURE SCORE (SPS) REPORT")
    print("=" * 50)
    print(f"  SPS: {result['sps']} / 100")
    print(f"  Recommendation: {result['deployment_recommendation']}")
    print()
    for dim_name, dim in result["dimensions"].items():
        print(f"  {dim_name}")
        print(f"    Subscore: {dim['subscore']} | Weight: {dim['weight']} | "
              f"Contribution: {dim['weighted_contribution']}")
    print("=" * 50)


if __name__ == "__main__":
    example_inputs = {
        "auroc_drop": 0.04,
        "threat_mitigated": 10,
        "threat_tested": 9,
        "threat_total": 12,
        "threat_passed": 8,
        "threat_partial": 1,
        "audit_completeness_rate": 1.0,
        "rbac_enforcement_rate": 1.0,
        "compliance_score": 0.87,
    }

    result = compute_sps(**example_inputs)
    print_report(result)

    print("\nSensitivity Analysis:")
    ablations = sensitivity_analysis(example_inputs)
    for a in ablations:
        print(f"  {a['scenario']}: SPS = {a['sps']} (drop = {a['sps_drop']})")

    print(json.dumps(result, indent=2))
