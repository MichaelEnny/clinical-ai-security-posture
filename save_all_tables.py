"""Save all measured results to tables/ directory."""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
import pandas as pd

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CODE_DIR)
TABLES = os.path.join(ROOT_DIR, "tables")
os.makedirs(TABLES, exist_ok=True)

sys.path.insert(0, os.path.join(CODE_DIR, "threat_model"))
sys.path.insert(0, os.path.join(CODE_DIR, "sps"))
sys.path.insert(0, os.path.join(CODE_DIR, "compliance"))

import mitre_mapper
import sps as sps_module
import hipaa_fhir_checklist as hf

# ── MITRE ATT&CK results ──────────────────────────────────────────────────────
mitre_mapper.record_result("T1566", "PASS",
    "JWT authentication required; unauthenticated access denied on all endpoints")
mitre_mapper.record_result("T1078", "PASS",
    "No default credentials; 3 accounts explicitly provisioned with unique passwords")
mitre_mapper.record_result("T1110", "PASS",
    "Rate limiting active on /auth/token; repeated failures return 401")
mitre_mapper.record_result("T1005", "PARTIAL",
    "DB not deployed in test env (Docker not installed); RBAC prevents unauthorized API access")
mitre_mapper.record_result("T1071", "PARTIAL",
    "TLS 1.3 in production config; dev env runs HTTP on localhost — upgrade required before production")
mitre_mapper.record_result("T1565", "PARTIAL",
    "Input bounds defense implemented but insufficient: FGSM flip_rate 28% (ICU), 48% (ED) "
    "even after bounds clipping because all adversarial samples remain within bounds by design")
mitre_mapper.record_result("T1530", "PASS",
    "Auditor role confirmed read-only (HTTP 403 on POST /predict); clinician cannot access other patients")
mitre_mapper.record_result("T1041", "PARTIAL",
    "Rate limiting configured (100 preds/hr/user); automated exfiltration stress test not conducted")
mitre_mapper.record_result("T1499", "PASS",
    "FastAPI + uvicorn stable under 35 concurrent predictions; no downtime or degradation observed")
mitre_mapper.record_result("T1190", "PASS",
    "Pydantic schema validation rejects malformed JSON, oversized payloads, and injection patterns (422 response)")
mitre_mapper.record_result("CUSTOM-MI", "PASS",
    "Shadow model MI: ICU accuracy=0.491 (< 0.50 baseline), AUROC=0.493; "
    "ED accuracy=0.487, AUROC=0.496; both LOW privacy risk — RF with balanced weights resists MI")
mitre_mapper.record_result("CUSTOM-ADV", "PARTIAL",
    "LIME-guided FGSM: flip_rate 28-51% at epsilon=0.1; boundary attack achieves flip in 1 step; "
    "input validation defense ineffective since all perturbations remain within clinical bounds")

stats = mitre_mapper.get_coverage_stats()
vectors = mitre_mapper.get_vectors()
mitre_df = pd.DataFrame([{
    "id": v["id"], "technique": v["technique"], "tactic": v["tactic"],
    "system_layer": v["system_layer"], "mitigation_implemented": v["mitigation_implemented"],
    "tested": v["tested"], "result": v["result"], "notes": v["notes"],
} for v in vectors])
mitre_df.to_csv(os.path.join(TABLES, "mitre_mapping.csv"), index=False)
print(f"MITRE: {stats['passed']} PASS, {stats['partial']} PARTIAL | Coverage: {stats['coverage_pct']}%")

# ── Adversarial results ───────────────────────────────────────────────────────
adv_data = [
    {"dataset":"icu_mortality","attack":"FGSM","epsilon":0.1,"n_samples":100,
     "auroc_clean":0.792,"auroc_adversarial":0.500,"auroc_drop":0.292,
     "flip_rate":0.28,"flip_rate_with_defense":0.28,"clinical_plausibility_rate":1.0},
    {"dataset":"ed_admission","attack":"FGSM","epsilon":0.1,"n_samples":100,
     "auroc_clean":0.789,"auroc_adversarial":0.560,"auroc_drop":0.228,
     "flip_rate":0.48,"flip_rate_with_defense":0.48,"clinical_plausibility_rate":1.0},
    {"dataset":"icu_mortality","attack":"Boundary","step_size":0.05,"n_samples":100,
     "auroc_clean":0.792,"auroc_adversarial":0.449,"auroc_drop":0.343,
     "flip_rate":0.28,"mean_steps_to_flip":1.0},
    {"dataset":"ed_admission","attack":"Boundary","step_size":0.05,"n_samples":100,
     "auroc_clean":0.789,"auroc_adversarial":0.522,"auroc_drop":0.266,
     "flip_rate":0.51,"mean_steps_to_flip":1.0},
]
pd.DataFrame(adv_data).to_csv(os.path.join(TABLES, "adversarial_results.csv"), index=False)
print("Adversarial results table saved.")

# ── Membership inference results ──────────────────────────────────────────────
mi_data = [
    {"dataset":"icu_mortality","mi_accuracy":0.491,"mi_auroc":0.493,
     "privacy_risk":"LOW","patients_at_risk":245,"baseline":0.50,"n_shadow_models":4,"pool_size":28920},
    {"dataset":"ed_admission","mi_accuracy":0.487,"mi_auroc":0.496,
     "privacy_risk":"LOW","patients_at_risk":243,"baseline":0.50,"n_shadow_models":4,"pool_size":5000},
]
pd.DataFrame(mi_data).to_csv(os.path.join(TABLES, "membership_inference_results.csv"), index=False)

# ── RBAC results ──────────────────────────────────────────────────────────────
rbac_data = [
    {"endpoint":"POST /predict","role":"clinician","expected":"ALLOWED","status_code":200,"result":"PASS"},
    {"endpoint":"POST /predict","role":"admin","expected":"ALLOWED","status_code":200,"result":"PASS"},
    {"endpoint":"POST /predict","role":"auditor","expected":"DENIED (403)","status_code":403,"result":"PASS"},
    {"endpoint":"GET /predictions/history","role":"clinician","expected":"ALLOWED","status_code":200,"result":"PASS"},
    {"endpoint":"GET /predictions/history","role":"admin","expected":"ALLOWED","status_code":200,"result":"PASS"},
    {"endpoint":"GET /predictions/history","role":"auditor","expected":"DENIED (403)","status_code":403,"result":"PASS"},
    {"endpoint":"GET /admin/users","role":"admin","expected":"ALLOWED","status_code":200,"result":"PASS"},
    {"endpoint":"GET /admin/users","role":"clinician","expected":"DENIED (403)","status_code":403,"result":"PASS"},
    {"endpoint":"GET /admin/users","role":"auditor","expected":"DENIED (403)","status_code":403,"result":"PASS"},
    {"endpoint":"GET /audit/logs","role":"auditor","expected":"ALLOWED","status_code":200,"result":"PASS"},
    {"endpoint":"GET /audit/logs","role":"admin","expected":"ALLOWED","status_code":200,"result":"PASS"},
    {"endpoint":"GET /audit/logs","role":"clinician","expected":"DENIED (403)","status_code":403,"result":"PASS"},
    {"endpoint":"GET /health","role":"clinician","expected":"ALLOWED","status_code":200,"result":"PASS"},
    {"endpoint":"GET /health","role":"admin","expected":"ALLOWED","status_code":200,"result":"PASS"},
    {"endpoint":"GET /health","role":"auditor","expected":"ALLOWED","status_code":200,"result":"PASS"},
]
pd.DataFrame(rbac_data).to_csv(os.path.join(TABLES, "rbac_results.csv"), index=False)
print("RBAC results saved: 15/15 PASS (100%)")

# ── Compliance checklist ──────────────────────────────────────────────────────
hf.run_automated_checks()
hf.record_result("HIPAA-01", "PARTIAL",
    "PostgreSQL pgcrypto in production config; not active in dev env (Docker not installed)")
hf.record_result("HIPAA-02", "PARTIAL",
    "TLS 1.3 in production config; dev env runs HTTP on localhost only")
hf.record_result("FHIR-01", "PARTIAL",
    "Features correspond to LOINC-coded clinical observations; formal LOINC mapping not documented")
hf.record_result("FHIR-02", "PARTIAL",
    "JSON output with prediction and confidence; FHIR ClinicalImpression format not implemented in v1")
hf.record_result("FHIR-04", "PASS",
    "OpenAPI schema auto-generated by FastAPI; model_version field in all responses")

scores = hf.compute_compliance_score()
comp_df = pd.DataFrame([{
    "id": i["id"], "category": i["category"], "safeguard": i["safeguard"],
    "status": i["status"], "weight": i["weight"], "evidence": i["evidence"],
} for i in scores["checklist"]])
comp_df.to_csv(os.path.join(TABLES, "compliance_checklist.csv"), index=False)
print(f"Compliance: HIPAA={scores['hipaa_pct']}%, FHIR={scores['fhir_pct']}%, Overall={scores['overall_compliance_pct']}%")

# ── SPS computation ───────────────────────────────────────────────────────────
sps_inputs = {
    "auroc_drop": 0.3426,
    "threat_mitigated": stats["mitigations_implemented"],
    "threat_tested": stats["techniques_tested"],
    "threat_total": stats["total_techniques"],
    "threat_passed": stats["passed"],
    "threat_partial": stats["partial"],
    "audit_completeness_rate": 0.9737,
    "rbac_enforcement_rate": 1.00,
    "compliance_score": scores["overall_compliance_score"],
}

result = sps_module.compute_sps(**sps_inputs)
sps_module.print_report(result)

ablations = sps_module.sensitivity_analysis(sps_inputs)
print("\nSensitivity Analysis:")
for a in ablations:
    print(f"  {a['scenario']}: SPS={a['sps']} (drop={a['sps_drop']})")

sps_summary = {
    "sps": result["sps"],
    "recommendation": result["deployment_recommendation"],
    "computed_at": result["computed_at"],
}
for k, v in result["dimensions"].items():
    sps_summary[f"d_{k}_subscore"] = v["subscore"]
    sps_summary[f"d_{k}_contribution"] = v["weighted_contribution"]
sps_summary.update(sps_inputs)

pd.DataFrame([sps_summary]).to_csv(os.path.join(TABLES, "sps_results.csv"), index=False)
pd.DataFrame(ablations).to_csv(os.path.join(TABLES, "sps_sensitivity.csv"), index=False)

with open(os.path.join(TABLES, "sps_full.json"), "w") as f:
    json.dump({"sps_result": result, "sensitivity": ablations, "inputs": sps_inputs}, f, indent=2, default=str)

print(f"\nAll tables saved to: {TABLES}")
print(f"Files: {sorted(os.listdir(TABLES))}")
