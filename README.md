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

## Checks before interpreting (v2)

- `results/checks/verification.md` prints the Stress and PAM definitions from `EMA_definition.json`,
  raw answer counts, and **empirical sign checks** on the coding: valence and stress should move in
  opposite directions, valence should be lower and stress higher in people with higher PHQ-9. The PAM
  grid is scored both ways so a swapped mapping shows up. It also lists model rows that are identical
  at full precision, near-duplicate features (|r| > 0.95), and participants with the least data.
- `results/tables/missingness_tests.csv`: does data completeness correlate with PHQ-9?
- `results/tables/slope_heterogeneity.csv` and `figures/person_slopes.png`: for models where people
  genuinely differ in their within-person slope, each person's slope, robustness to the two most
  extreme people, and `slope_moderation_phq.csv`: whether PHQ-9 predicts a person's slope
  (cross-level interaction).
- Depresjon work-status and recording-length checks: rank partial correlations adjusting for work
  (1 = working, 2 = unemployed/sick leave/pension) and recording days, and every metric recomputed
  on each person's first 7 complete days.
- Depresjon inpatient sensitivity: MADRS correlations adjusted for inpatient status (partial Spearman),
  inpatients vs outpatients, and outpatients vs controls, with FDR applied within that family.
- `divergent` now fires only when within and between estimates have opposite signs **and** both CIs
  exclude zero.
- Random-slope likelihood-ratio tests (`slope_lrt_p`, `slope_lrt_p_fdr`) test whether person-to-person
  variation in within-person slopes is real.

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

## Coding checks and remaining assumptions

| Item | Status | Evidence |
| --- | --- | --- |
| Stress recode (`stress_recode`) | Confirmed | Matches the codes in `EMA/EMA_definition.json` |
| PAM valence direction (`pam_layout`) | Checked empirically | Within-person r(valence, stress) = -0.39; Mood-question cross-check in `results/checks/verification.md` |
| PAM arousal direction | **Not verified** | No published mapping found; arousal is not reported |
| Depresjon inpatient code | Confirmed | 1 = inpatient, 2 = outpatient ([dataset page](https://datasets.simula.no/depresjon/)) |
| Participants | Confirmed | 49 IDs in the public archive (u00-u59 with gaps); the paper reports 48 |

Still assumptions:

- **Coverage gate**: person-days with < 18 h of GPS samples are dropped. `missingness_tests.csv`
  tests whether data loss relates to PHQ-9.
- **Night-time phone use** treats any unlocked time as use, so it is only meaningful on
  high-coverage days.
