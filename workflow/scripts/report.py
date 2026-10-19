"""Assemble results into a single markdown report."""
import json
import pandas as pd


def md_table(df, floatfmt="{:.3f}"):
    if df is None or df.empty:
        return "_No results._\n"
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in r]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def ci(df, est):
    df = df.copy()
    df["95% CI"] = df.apply(lambda r: f"[{r.ci_low:.2f}, {r.ci_high:.2f}]", axis=1)
    return df.drop(columns=["ci_low", "ci_high"])


summary = json.load(open(snakemake.input.summary))
mm = pd.read_csv(snakemake.input.mixed)
phq = pd.read_csv(snakemake.input.phq)
ml = pd.read_csv(snakemake.input.ml)
dep = pd.read_csv(snakemake.input.dep_tests)
cv = pd.read_csv(snakemake.input.construct, sep="\t")
miss = pd.read_csv(snakemake.input.miss)
dep_in = pd.read_csv(snakemake.input.dep_inpatient)


def read_or_empty(path):
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


slope_sum = read_or_empty(snakemake.input.slope_summary)
slope_mod = read_or_empty(snakemake.input.slope_mod)
dep_conf = pd.read_csv(snakemake.input.dep_confound)

L = ["# Digital proxies of reward, motivation and impulsivity: StudentLife + Depresjon\n"]
L.append("## 1. Data\n")
L.append(md_table(pd.DataFrame([summary]).T.reset_index().rename(columns={"index": "quantity", 0: "value"})))

L.append("\n## 2. Within- vs between-person associations (mixed models)\n")
L.append("Standardised betas. *Between*: are people who do more of X on average higher/lower in the outcome? "
         "*Within*: is a given person higher/lower on days they do more of X than usual? "
         "`within_slope_sd` is the SD of person-specific within-slopes: the heterogeneity a precision-medicine "
         "framework would need to explain. `divergent` is True only when the within and between estimates have "
         "opposite signs AND both 95% CIs exclude zero. p-values are BH-FDR corrected across all tests.\n")
bad = mm[mm.level == "not_estimable"]
mm = mm[mm.level != "not_estimable"]
if len(bad):
    L.append("Not estimable (near-constant feature or singular design): "
             + ", ".join(f"{r.feature}->{r.outcome}{' (next-day)' if r.lagged else ''}" for r in bad.itertuples()) + "\n")
for lagged, title in ((False, "Same-day"), (True, "Next-day (controlling for today's outcome)")):
    t = mm[mm.lagged == lagged][["outcome", "feature", "level", "beta_std", "ci_low", "ci_high",
                                 "p_fdr", "within_slope_sd", "divergent", "n_persons", "n_obs"]]
    L.append(f"\n### {title}\n")
    L.append(md_table(ci(t, "beta_std")))

L.append("\n### Is the between-person variation in slopes real?\n")
L.append("Likelihood-ratio test of the random-slope model against a random-intercept-only model (REML, same fixed effects; "
         "boundary-corrected p from a 50:50 mixture of chi-square(1) and chi-square(2)). BH-FDR across these tests only. "
         "Only rows with FDR < 0.10 are listed; an empty table means no feature shows clear person-to-person differences in its slope.\n\n")
lrt = mm[(mm.level == "within") & mm.slope_lrt_p.notna()][["outcome", "feature", "lagged", "beta_std", "within_slope_sd",
                                                          "slope_lrt_stat", "slope_lrt_p", "slope_lrt_p_fdr"]]
L.append(md_table(lrt[lrt.slope_lrt_p_fdr < 0.10].sort_values("slope_lrt_p")))
L.append(f"\n{int((lrt.slope_lrt_p_fdr < 0.10).sum())} of {len(lrt)} random-slope tests pass FDR < 0.10.\n")

L.append("\n### Person-specific slopes\n")
L.append("For each model whose slope variation passed FDR < 0.05: the spread of person-specific slopes (shrunken "
         "estimates), how many people go against the average, and whether the variation survives dropping the two "
         "most extreme people. Plot: `figures/person_slopes.png`.\n\n")
if len(slope_sum):
    L.append(md_table(slope_sum))
else:
    L.append("_No model passed the threshold._\n")
L.append("\n### Does depression predict a person's slope?\n")
L.append("`interaction` is the cross-level interaction (within-feature x PHQ-9, standardised) in the mixed model: how "
         "much a person's slope changes per 1 SD higher PHQ-9. PHQ-9 at the start of term precedes the daily data; the "
         "end-of-term score is secondary. `rho_slope_vs_phq` is a descriptive check on the shrunken slopes. "
         "BH-FDR across these tests.\n\n")
if len(slope_mod):
    t = slope_mod.copy()
    t["95% CI"] = t.apply(lambda r: f"[{r.ci_low:.2f}, {r.ci_high:.2f}]", axis=1)
    t["rho 95% CI"] = t.apply(lambda r: f"[{r.rho_ci_low:.2f}, {r.rho_ci_high:.2f}]", axis=1)
    L.append(md_table(t.drop(columns=["ci_low", "ci_high", "rho_ci_low", "rho_ci_high"])))

L.append("\n## 3. Person-level anchor: feature means vs post-term PHQ-9\n")
L.append("Exploratory. n is small; read the CI widths, not the p-values.\n\n")
if len(phq):
    L.append(md_table(ci(phq, "rho")))

L.append("\n### Is missing data related to depression?\n")
L.append("Spearman correlations between each person's data completeness and PHQ-9. If completeness correlates with "
         "PHQ-9, missing days are not random and the estimates above may be biased.\n\n")
if len(miss):
    L.append(md_table(ci(miss, "rho")))

L.append("\n## 4. Why cross-validation design matters\n")
L.append("Random K-fold lets the model see other days from the same person, so it can learn who someone is "
         "rather than how behaviour relates to mood. LOSO is the honest estimate for a new person.\n\n")
L.append(md_table(ml))

L.append("\n## 5. Clinical replication: Depresjon rest-activity rhythms\n")
L.append("Between-person only (one recording block per participant). MADRS correlations use patients only.\n\n")
L.append("### Original analyses\n")
L.append(md_table(ci(dep[dep.family == "original"].drop(columns="family"), "estimate")))
L.append("\n### Inpatient sensitivity analyses\n")
L.append("Hypothesis tested: ward routines make inpatients look more regular. Inpatient code confirmed from the dataset "
         "page (1 = inpatient). FDR within this family.\n\n")
L.append(md_table(dep_in))
L.append("\n")
L.append(md_table(ci(dep[dep.family == "inpatient_sensitivity"].drop(columns="family"), "estimate")))
L.append("\n### Work status and recording length\n")
L.append("Two further explanations. (i) People in work have weekday/weekend differences, which lowers interdaily "
         "stability; patients on sick leave may live more uniform days. (ii) Recording length varies, and rhythm "
         "metrics from fewer days are less reliable. Tests adjust for these (rank partial correlation) or compare "
         "everyone on the same number of days. FDR within this family.\n\n")
L.append(md_table(dep_conf))
L.append("\n")
L.append(md_table(ci(dep[dep.family == "confound_sensitivity"].drop(columns="family"), "estimate")))

L.append("\n## 6. Construct validity of each proxy\n")
L.append(md_table(cv))
L.append(
    "\nPassive behaviour is many-to-one with its causes: staying home can reflect anhedonia, an exam, rain or "
    "introversion. None of these features measures reward, motivation or impulsivity directly. To become valid "
    "RM&I proxies they need anchoring against (i) task-based measures of the construct, such as effort-based choice "
    "(EEfRT) and delay discounting, collected in the same people, and (ii) neurobiological markers of the circuits "
    "those tasks engage. That anchoring is the work this analysis cannot do with public data.\n")

L.append("\n## 7. Limitations\n")
L.append("- StudentLife: 49 participant IDs in the public archive (the paper reports 48), one term, one campus; not a clinical sample.\n"
         "- Data completeness showed no detectable relationship with PHQ-9 (see missingness table), but with n = 38-46 a moderate relationship cannot be ruled out.\n"
         "- Next-day models are weak temporal evidence, not causal estimates.\n"
         "- Stress coding and the Depresjon inpatient code are confirmed from the dataset documentation. PAM valence "
         "direction is checked empirically (against stress and the separate Mood question); PAM arousal is not "
         "verified and is not reported. See results/checks/verification.md.\n")
open(snakemake.output[0], "w").write("\n".join(L))
