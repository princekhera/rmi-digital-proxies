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
        rows.append({"number": pid, "group": group, **rhythm_summary(pd.read_csv(f))})
rhythm = pd.DataFrame(rows)
scores = pd.read_csv(os.path.join(root, "scores.csv"))
rhythm = rhythm.merge(scores, on="number", how="left")
# Depresjon codes inpatient status numerically. ASSUMPTION (config):
# depresjon_inpatient_value is the code meaning 'inpatient'.
rhythm["is_inpatient"] = np.where(rhythm["inpatient"].isna(), np.nan,
                                  (rhythm["inpatient"] == cfg["depresjon_inpatient_value"]).astype(float))
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
