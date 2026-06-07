"""
MITRE ATT&CK for Healthcare — Attack Vector Mapper
Maps clinical AI system components to relevant MITRE techniques
and records test results for Threat Coverage SPS dimension.
"""

ATTACK_VECTORS = [
    {
        "id": "T1566",
        "technique": "Phishing: Spearphishing Link",
        "tactic": "Initial Access",
        "system_layer": "Authentication",
        "mitigation": "MFA enforcement on all user accounts",
        "mitigation_implemented": True,
        "tested": True,
        "result": None,
        "notes": "Simulated phishing attempt against clinician credentials",
    },
    {
        "id": "T1078",
        "technique": "Valid Accounts: Default Accounts",
        "tactic": "Credential Access",
        "system_layer": "API",
        "mitigation": "No default credentials; all accounts require explicit provisioning",
        "mitigation_implemented": True,
        "tested": True,
        "result": None,
        "notes": "Verified no default admin/root accounts exist in deployment",
    },
    {
        "id": "T1110",
        "technique": "Brute Force: Password Spraying",
        "tactic": "Credential Access",
        "system_layer": "API",
        "mitigation": "Rate limiting: 10 requests per minute per IP; account lockout after 5 failures",
        "mitigation_implemented": True,
        "tested": True,
        "result": None,
        "notes": "Tested with automated spray tool against /auth/token endpoint",
    },
    {
        "id": "T1005",
        "technique": "Data from Local System",
        "tactic": "Collection",
        "system_layer": "Database",
        "mitigation": "AES-256 encryption at rest; filesystem access requires OS-level credentials",
        "mitigation_implemented": True,
        "tested": True,
        "result": None,
        "notes": "Verified PostgreSQL data directory encrypted; access denied without DB credentials",
    },
    {
        "id": "T1071",
        "technique": "Application Layer Protocol: Web Protocols",
        "tactic": "Command and Control",
        "system_layer": "API",
        "mitigation": "TLS 1.3 enforced on all API endpoints; plaintext connections rejected",
        "mitigation_implemented": True,
        "tested": True,
        "result": None,
        "notes": "Verified TLS 1.3; HTTP connections return 301 redirect",
    },
    {
        "id": "T1565",
        "technique": "Data Manipulation",
        "tactic": "Impact",
        "system_layer": "Model Input",
        "mitigation": "Input validation with clinical bounds checking; values outside plausible ranges rejected",
        "mitigation_implemented": True,
        "tested": True,
        "result": None,
        "notes": "Partial: bounds checking implemented for numeric features; categorical validation pending",
    },
    {
        "id": "T1530",
        "technique": "Data from Cloud Storage",
        "tactic": "Collection",
        "system_layer": "Data Layer",
        "mitigation": "RBAC on all storage access; audit log for every read operation",
        "mitigation_implemented": True,
        "tested": True,
        "result": None,
        "notes": "Verified auditor role cannot write; clinician cannot read other patients",
    },
    {
        "id": "T1041",
        "technique": "Exfiltration Over C2 Channel",
        "tactic": "Exfiltration",
        "system_layer": "API Output",
        "mitigation": "Output rate limiting: max 100 predictions per hour per user",
        "mitigation_implemented": True,
        "tested": False,
        "result": None,
        "notes": "Rate limiting implemented; automated exfiltration test not yet conducted",
    },
    {
        "id": "T1499",
        "technique": "Endpoint Denial of Service",
        "tactic": "Impact",
        "system_layer": "API",
        "mitigation": "Request throttling; Docker resource limits on API container",
        "mitigation_implemented": True,
        "tested": True,
        "result": None,
        "notes": "Load tested at 500 concurrent requests; API remained responsive",
    },
    {
        "id": "T1190",
        "technique": "Exploit Public-Facing Application",
        "tactic": "Initial Access",
        "system_layer": "API",
        "mitigation": "Input sanitization; Pydantic schema validation on all endpoints",
        "mitigation_implemented": True,
        "tested": True,
        "result": None,
        "notes": "Malformed JSON, oversized payloads, and SQL injection patterns tested",
    },
    {
        "id": "CUSTOM-MI",
        "technique": "Model Inversion / Membership Inference",
        "tactic": "Collection",
        "system_layer": "Model",
        "mitigation": "Differential privacy (optional); confidence score rounding to 2 decimal places",
        "mitigation_implemented": False,
        "tested": True,
        "result": None,
        "notes": "Membership inference attack conducted via shadow model approach (Shokri et al. 2017)",
    },
    {
        "id": "CUSTOM-ADV",
        "technique": "Adversarial Input: Gradient-Guided Attack",
        "tactic": "Impact",
        "system_layer": "Model Input",
        "mitigation": "Adversarial training with augmented examples; input bounds enforcement",
        "mitigation_implemented": True,
        "tested": True,
        "result": None,
        "notes": "LIME-guided FGSM attack and boundary attack both conducted",
    },
]


def record_result(technique_id: str, result: str, notes: str = ""):
    for v in ATTACK_VECTORS:
        if v["id"] == technique_id:
            v["result"] = result
            if notes:
                v["notes"] = notes
            return True
    return False


def get_coverage_stats():
    total = len(ATTACK_VECTORS)
    implemented = sum(1 for v in ATTACK_VECTORS if v["mitigation_implemented"])
    tested = sum(1 for v in ATTACK_VECTORS if v["tested"])
    passed = sum(1 for v in ATTACK_VECTORS if v["result"] == "PASS")
    partial = sum(1 for v in ATTACK_VECTORS if v["result"] == "PARTIAL")

    coverage_fraction = (passed + 0.5 * partial) / total if total > 0 else 0.0

    return {
        "total_techniques": total,
        "mitigations_implemented": implemented,
        "techniques_tested": tested,
        "passed": passed,
        "partial": partial,
        "coverage_fraction": round(coverage_fraction, 4),
        "coverage_pct": round(coverage_fraction * 100, 1),
    }


def get_vectors():
    return ATTACK_VECTORS
