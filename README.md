# Digital proxies of reward, motivation and impulsivity

A reproducible Snakemake pipeline that asks whether passive smartphone and
wearable signals behave like proxies for reward, motivation and impulsivity
(RM&I), and where that claim breaks down.

**Data (both free, no registration):**
- [StudentLife](https://studentlife.cs.dartmouth.edu/): 48 Dartmouth students, 10 weeks of passive phone sensing, daily EMA, pre/post PHQ-9
- [Depresjon](https://zenodo.org/record/1219550): wrist actigraphy from 23 depressed patients and 32 controls, clinician-rated MADRS

## What it does

```
StudentLife raw ─► sl_features (per participant, parallel) ─► sl_panel ─┬─► mixed_models
                                                                        ├─► phq_anchor
                                                                        └─► ml_loso
Depresjon raw ─────────────────────────────────────────────► depresjon ─┴─► report.md
```

1. **Features** per person-day: time at home, location entropy and variance, places visited,
   distance, physical activity, ambient conversation, night-time phone use, dark exposure.
   Person-level circadian movement (Saeb et al., 2015).
2. **Within/between mixed models.** Each feature is split into a person mean and a daily
   deviation. The two coefficients answer different questions ("are people who stay home
   more lower in mood?" vs "is a person lower on days they stay home more?"). Random slopes
   estimate how much that daily link differs between people. Same-day and next-day models,
   with BH-FDR correction across all tests.
3. **PHQ-9 anchor.** PHQ-9 exists only pre/post, so it is used as a person-level anchor
   with bootstrap CIs, not as a repeated outcome.
4. **Cross-validation leakage demo.** Random K-fold vs leave-one-subject-out.
5. **Clinical replication.** Nonparametric rest-activity rhythms (IS, IV, RA, M10, L5) in
   Depresjon: patients vs controls, and severity within patients.
6. **Construct validity table** (`config/construct_validity.tsv`): for each feature, the
   RM&I process it is claimed to proxy, competing explanations, and what would validate it.

## Run

```bash
conda env create -f envs/environment.yml && conda activate rmi-proxies

# Smoke test on synthetic data with a planted effect
python tests/make_synthetic.py
snakemake -c4 --configfile config/test.yaml

# Real data: download and unpack to the paths in config/config.yaml, then
snakemake -c4
```

Unit tests: `pytest -q tests`

## Assumptions to check on real data

- **PAM grid orientation** (`pam_layout`) and **stress recode** (`stress_recode`) are
  configurable. Verify both against Pollak et al. (2011) and `EMA/EMA_definition.json`.
- **Coverage gate**: person-days with < 18 h of GPS samples are dropped. Missingness is
  unlikely to be random, so report how many days this removes.
- **Night-time phone use** treats any unlocked time as use, so it is only meaningful on
  high-coverage days.
