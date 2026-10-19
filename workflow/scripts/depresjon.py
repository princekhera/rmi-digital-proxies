import sys; sys.path.insert(0, snakemake.params.src)  # noqa: E702
import glob, os
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from rmi.rhythm import rhythm_summary
from rmi.models import bootstrap_spearman, hedges_g_boot, partial_spearman_boot

root, cfg = snakemake.params.dep_dir, snakemake.params.cfg
rows = []
for group in ("condition", "control"):
    for f in sorted(glob.glob(os.path.join(root, group, f"{group}_*.csv"))):
        pid = os.path.basename(f).replace(".csv", "")
        minute = pd.read_csv(f)
        eq = rhythm_summary(minute, max_days=cfg.get("depresjon_equal_days", 7))
        rows.append({"number": pid, "group": group, **rhythm_summary(minute),
                     **{f"{k}_eq": v for k, v in eq.items() if k != "n_days"}})
rhythm = pd.DataFrame(rows)
scores = pd.read_csv(os.path.join(root, "scores.csv"))
rhythm = rhythm.merge(scores, on="number", how="left")
# Depresjon codes inpatient status numerically. ASSUMPTION (config):
# depresjon_inpatient_value is the code meaning 'inpatient'.
rhythm["is_inpatient"] = np.where(rhythm["inpatient"].isna(), np.nan,
                                  (rhythm["inpatient"] == cfg["depresjon_inpatient_value"]).astype(float))
# work: 1 = working or studying, 2 = unemployed / sick leave / pension
# (datasets.simula.no/depresjon). People in work have weekday/weekend
# differences, which lowers interdaily stability.
rhythm["not_working"] = np.where(rhythm["work"].isna(), np.nan, (rhythm["work"] == 2).astype(float))
rhythm.to_csv(snakemake.output.metrics, index=False)

metrics = ["mean_activity", "IS", "IV", "RA", "M10", "L5"]
cond, ctrl = rhythm[rhythm.group == "condition"], rhythm[rhythm.group == "control"]
inp, outp = cond[cond.is_inpatient == 1], cond[cond.is_inpatient == 0]
res = []


def add(metric, family, test, est, lo, hi, p, n):
    res.append({"metric": metric, "family": family, "test": test, "estimate": est,
                "ci_low": lo, "ci_high": hi, "p": p, "n": n})


for m in metrics:
    # Original analyses
    add(m, "original", "condition_vs_control_hedges_g",
        *hedges_g_boot(cond[m].to_numpy(float), ctrl[m].to_numpy(float)), len(cond) + len(ctrl))
    rho, lo, hi, p, n = bootstrap_spearman(cond[m].to_numpy(float), cond["madrs1"].to_numpy(float))
    add(m, "original", "spearman_vs_MADRS_within_patients", rho, lo, hi, p, n)

    # Inpatient sensitivity analyses
    r, lo, hi, p, n = partial_spearman_boot(cond[m], cond["madrs1"], cond["is_inpatient"])
    add(m, "inpatient_sensitivity", "spearman_vs_MADRS_adjusted_for_inpatient", r, lo, hi, p, n)
    if len(inp) >= 3 and len(outp) >= 3:
        add(m, "inpatient_sensitivity", "inpatient_vs_outpatient_hedges_g",
            *hedges_g_boot(inp[m].to_numpy(float), outp[m].to_numpy(float)), len(inp) + len(outp))
        add(m, "inpatient_sensitivity", "outpatients_vs_controls_hedges_g",
            *hedges_g_boot(outp[m].to_numpy(float), ctrl[m].to_numpy(float)), len(outp) + len(ctrl))

# Confound sensitivity: work status and recording length -----------------
eq_days = cfg.get("depresjon_equal_days", 7)
for m in ["IS", "IV", "RA"]:
    z_work, z_days = cond["not_working"], cond["n_days"]
    add(m, "confound_sensitivity", "spearman_vs_MADRS_adjusted_for_work",
        *partial_spearman_boot(cond[m], cond["madrs1"], z_work))
    add(m, "confound_sensitivity", "spearman_vs_MADRS_adjusted_for_recording_days",
        *partial_spearman_boot(cond[m], cond["madrs1"], z_days))
    add(m, "confound_sensitivity", "spearman_vs_MADRS_adjusted_for_work_and_days",
        *partial_spearman_boot(cond[m], cond["madrs1"], np.column_stack([z_work, z_days])))
    rho, lo, hi, p, n = bootstrap_spearman(cond[f"{m}_eq"].to_numpy(float), cond["madrs1"].to_numpy(float))
    add(m, "confound_sensitivity", f"spearman_vs_MADRS_first_{eq_days}_days_only", rho, lo, hi, p, n)
    a, b = cond[f"{m}_eq"].to_numpy(float), ctrl[f"{m}_eq"].to_numpy(float)
    if np.isfinite(a).sum() >= 3 and np.isfinite(b).sum() >= 3:
        add(m, "confound_sensitivity", f"condition_vs_control_hedges_g_first_{eq_days}_days_only",
            *hedges_g_boot(a, b), int(np.isfinite(a).sum() + np.isfinite(b).sum()))
    rho, lo, hi, p, n = bootstrap_spearman(rhythm[m].to_numpy(float), rhythm["n_days"].to_numpy(float))
    add(m, "confound_sensitivity", "spearman_vs_recording_days_all_participants", rho, lo, hi, p, n)
    wk = rhythm[rhythm["not_working"] == 0][m].to_numpy(float)
    nw = rhythm[rhythm["not_working"] == 1][m].to_numpy(float)
    if len(wk) >= 3 and len(nw) >= 3:
        add(m, "confound_sensitivity", "not_working_vs_working_hedges_g_all_participants",
            *hedges_g_boot(nw, wk), len(wk) + len(nw))

res = pd.DataFrame(res).dropna(subset=["p"])
# FDR within each family, so the sensitivity tests do not dilute the originals.
res["p_fdr"] = np.nan
for fam, g in res.groupby("family"):
    res.loc[g.index, "p_fdr"] = multipletests(g["p"], method="fdr_bh")[1]
res.to_csv(snakemake.output.tests, index=False)

# How inpatient status relates to severity: the core of the confound story.
madrs_in = inp["madrs1"].dropna()
madrs_out = outp["madrs1"].dropna()
pd.DataFrame([{
    "n_patients": len(cond), "n_inpatients": len(inp), "n_outpatients": len(outp),
    "madrs_mean_inpatients": madrs_in.mean() if len(madrs_in) else np.nan,
    "madrs_mean_outpatients": madrs_out.mean() if len(madrs_out) else np.nan,
    "inpatient_code_assumed": cfg["depresjon_inpatient_value"],
}]).to_csv(snakemake.output.inpatient_summary, index=False)

# Who works, by group, and how recording length differs: the confound picture.
conf = rhythm.groupby("group").agg(
    n=("number", "size"),
    n_work_recorded=("not_working", lambda v: int(v.notna().sum())),
    share_not_working=("not_working", "mean"),
    median_recording_days=("n_days", "median"),
    min_recording_days=("n_days", "min"),
    n_with_equal_days=("IS_eq", lambda v: int(v.notna().sum())),
).reset_index()
conf.to_csv(snakemake.output.confound_summary, index=False)
