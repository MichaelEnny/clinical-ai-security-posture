# Evaluating Security-by-Design in Clinical AI
## Adversarial Testing, Threat Assessment, and Security Posture Scoring

This is the code for the thesis's security posture. It runs adversarial attacks against two clinical AI models trained on MIMIC-IV data, tests access controls and audit logging, scores HIPAA and FHIR compliance, and computes a Security Posture Score (SPS) for each system.

The two models under evaluation are an ICU mortality predictor and an emergency department admission classifier. Both are Random Forest classifiers served through a FastAPI inference API with JWT-based role access control.


> **Revision R2 correction.** The adversarial results in earlier versions of the paper came from a clipping bug in `adversarial/`. The corrected pipeline that produces every number in the current paper is in [`revision_r2/`](revision_r2/README.md). Run that instead of `run_all_tests.py` for the adversarial, membership inference and SPS results.

---

## What You Need

Python 3.11 or 3.12 is recommended. The code runs on Python 3.13 but requires one adjustment: passlib does not work with bcrypt 4.x on Python 3.13. The auth module in this repo already uses bcrypt directly to avoid that issue.

Install dependencies:

```
pip install -r system/requirements.txt
pip install bcrypt lime
```

You also need the MIMIC-IV datasets. Access requires credentialed PhysioNet approval. Once you have access, place the raw files at:

```
third_paper/data/mimic/icu/          (icustays, chartevents, admissions, patients)
third_paper/data/mimic/ed_v2_2/ed/   (edstays, triage)
```

---

## Folder Structure

```
code/
  data_pipeline/       build cohorts and train models
  system/              FastAPI inference server, auth, audit logging, Docker config
  adversarial/         FGSM, boundary attack, membership inference
  threat_model/        MITRE ATT&CK mapping and coverage scoring
  audit/               RBAC enforcement tests and audit log completeness checks
  compliance/          HIPAA and FHIR checklist runner
  sps/                 Security Posture Score computation and sensitivity analysis
  run_all_tests.py     runs the full evaluation in sequence
  save_all_tables.py   saves all results to third_paper/tables/
  generate_figures.py  produces the four paper figures
  run_ed_mi_only.py    standalone ED membership inference (faster, capped at 5k samples)
```

---

## Step 1: Build the Data and Train Models

Run these three scripts in order from the `code/` directory:

```
python data_pipeline/build_icu_cohort.py
python data_pipeline/build_ed_cohort.py
python data_pipeline/train_models.py
```

`build_icu_cohort.py` reads chartevents (which is large) and takes 20 to 40 minutes. The other two are fast.

After this step, trained model files appear in `system/models/` and cohort CSVs appear in `data/mimic/icu_cohort/` and `data/mimic/ed_cohort/`.

Model performance after training:

| Model | Test AUROC |
|---|---|
| ICU mortality | 0.772 |
| ED admission | 0.750 |
| CKD (pilot) | 1.000 |
| WDBC (pilot) | 0.993 |

---

## Step 2: Start the Inference Server

From the `system/` directory:

```
uvicorn main:app --host 0.0.0.0 --port 8000
```

Or with Docker Compose if you have Docker installed:

```
docker-compose up -d
```

Confirm the server is running:

```
curl http://localhost:8000/health
```

Expected response: `{"status":"ok","version":"1.0"}`

Three test accounts are provisioned at startup: `clinician_alice`, `admin_bob`, and `auditor_carol`. Passwords are in `system/auth.py`. Do not use these accounts outside of a local test environment.

---

## Step 3: Run the Full Evaluation

With the server running, execute from the `code/` directory:

```
python run_all_tests.py
```

This script runs everything in sequence: FGSM and boundary attacks on ICU and ED models, membership inference on ICU, MITRE ATT&CK coverage scoring, RBAC tests against the live API, audit log completeness checks, HIPAA and FHIR compliance scoring, and final SPS computation. Results are written to `third_paper/tables/`.

Expected runtime: 15 to 25 minutes. FGSM runs on 100 test samples per model using 50 LIME neighborhood samples each. If you reduce these numbers the runtime drops but results will vary more.

Membership inference for the ED model takes longer due to dataset size. Run it separately if needed:

```
python run_ed_mi_only.py
```

This caps the training pool at 5,000 samples to keep it tractable.

---

## Step 4: Save Tables and Generate Figures

```
python save_all_tables.py
python generate_figures.py
```

`save_all_tables.py` writes eight CSV files and a JSON summary to `third_paper/tables/`. `generate_figures.py` writes four PNG files to `third_paper/figures/`.

---

## Key Results

| Metric | Value |
|---|---|
| ICU FGSM flip rate | 28% at epsilon = 0.1 |
| ED FGSM flip rate | 48% at epsilon = 0.1 |
| Clinical plausibility rate | 100% (all adversarial samples stayed within physiological bounds) |
| ICU membership inference accuracy | 0.491 (below 0.500 baseline, LOW risk) |
| ED membership inference accuracy | 0.487 (below 0.500 baseline, LOW risk) |
| RBAC enforcement | 15 / 15 tests passed |
| Audit completeness | 97.37% |
| HIPAA score | 86.7% |
| FHIR score | 71.4% |
| Security Posture Score | 78.9 / 100 (conditional deployment tier) |

The 100% clinical plausibility rate is the most operationally significant finding. Every adversarial sample produced by FGSM stayed within normal physiological ranges, so input validation based on value bounds gives no protection. The models sit close enough to their decision boundaries that a perturbation of 0.05 standardized units flips predictions in a single step.

---

## API Endpoints

| Endpoint | Allowed Roles |
|---|---|
| POST /auth/token | all (login) |
| POST /predict | clinician, admin |
| GET /predictions/history | clinician, admin |
| GET /admin/users | admin |
| GET /audit/logs | admin, auditor |
| GET /health | all |

Tokens expire after 8 hours. Expired tokens return HTTP 401.

---

## Notes on Python Version Compatibility

If you see a bcrypt error on startup, your environment has passlib and bcrypt 4.x together. The `system/auth.py` file in this repo bypasses passlib entirely by calling bcrypt directly. If you pulled from an older branch and still see the error, check that auth.py does not import from `passlib.context`.

---

## License

MIT license. See the [LICENSE](LICENSE) file for the full license text.
