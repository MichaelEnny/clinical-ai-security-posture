# Revision R2: corrected evaluation pipeline

This folder reproduces every adversarial, membership-inference, calibration and SPS result in the R2 version of the paper.

## Correction

In earlier versions, `adversarial/tabular_fgsm.py` and `adversarial/boundary_attack.py` clipped **standardized** (z-score) features to clinical bounds given in **raw** units (for example, heart rate 30-200 bpm). That moved every attacked feature to an extreme value, and the earlier adversarial results came from that clipping step, not from the perturbation. The scaler was also fit on uncleaned vitals that contain charting errors.

`r2_pipeline.py` fixes this:

1. Raw vital signs are winsorized to the physiological bounds (Table 1 of the paper) before standardization, and pain to 0-10. The models are retrained with the original hyperparameters.
2. Adversarial inputs are mapped back to raw units, clipped, and re-standardized. Plausibility is checked in raw units.
3. Only the six vital signs are perturbed. FGSM and the line-search attack use the same 500 test patients per model. The line search walks toward each patient's nearest opposite-class record; the fixed target used in earlier versions and five random targets are run as sensitivity analyses.
4. Bootstrap 95% CIs (2,000 paired resamples), LIME ablation (50 to 5,000 samples; training-distribution and patient-centred sampling), epsilon sweep, seeded shadow-model and loss-threshold membership inference, Brier score and ECE, and SPS with a bootstrap interval and by attack configuration.
5. T1078 (Valid Accounts) is rated PARTIAL because the demonstration accounts in `system/auth.py` have passwords in source code. Replace them before any real deployment.

The attack modules in `adversarial/` now raise an error if they are called without the fitted scaler, so the original bug cannot recur silently. `run_all_tests.py` predates this fix and will stop at that check; use the pipeline below for the paper's results.

## Run

From the `third_paper/` directory, with the cohort CSVs from `data_pipeline/` in place:

```
python code/revision_r2/r2_pipeline.py      # about 10 minutes; writes results_r2/ and models_r2/
python code/revision_r2/r2_budget.py        # adds SPS by attack configuration
python code/revision_r2/r2_figures.py       # writes the paper's figures to third_paper/figures/
```

`corrected_adversarial.py` is a diagnostic only. It reruns the R1 models with the clipping fixed but without data cleaning, to show the size of the R1 error. It is not used for any number in the paper.

Requirements: the packages in `system/requirements.txt`, plus `lime` and `scipy`.
