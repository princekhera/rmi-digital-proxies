"""Is missing data related to depression? If students with higher PHQ-9 lose
more days (phone off, not carried, EMA ignored), every estimate is biased."""
import sys; sys.path.insert(0, snakemake.params.src)  # noqa: E702
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from rmi.models import bootstrap_spearman

cfg = snakemake.params.cfg
raw = pd.concat([pd.read_csv(f) for f in snakemake.input.daily if pd.read_csv(f).shape[0]], ignore_index=True)
panel = pd.read_csv(snakemake.input.panel)
person = pd.read_csv(snakemake.input.person)

per = raw.groupby("uid").agg(
    days_with_gps=("day", "nunique"),
    mean_coverage_h=("coverage_h", "mean"),
    days_pass_coverage=("coverage_h", lambda s: int((s >= cfg["min_coverage_hours"]).sum())),
).reset_index()
per = per.merge(panel.groupby("uid")["day"].nunique().rename("days_in_panel").reset_index(), on="uid", how="left")
per["days_in_panel"] = per["days_in_panel"].fillna(0)
per["frac_pass_coverage"] = per["days_pass_coverage"] / per["days_with_gps"]
per["frac_in_panel"] = per["days_in_panel"] / per["days_with_gps"]
per = per.merge(person[["uid", "phq9_pre", "phq9_post"]], on="uid", how="left")
per.to_csv(snakemake.output.per_person, index=False)

rows = []
for m in ["days_with_gps", "mean_coverage_h", "frac_pass_coverage", "frac_in_panel"]:
    for phq in ["phq9_pre", "phq9_post"]:
        rho, lo, hi, p, n = bootstrap_spearman(per[m].to_numpy(float), per[phq].to_numpy(float))
        rows.append({"missingness_measure": m, "phq": phq, "rho": rho, "ci_low": lo, "ci_high": hi, "p": p, "n": n})
res = pd.DataFrame(rows).dropna(subset=["p"])
if len(res):
    res["p_fdr"] = multipletests(res["p"], method="fdr_bh")[1]
res.to_csv(snakemake.output.tests, index=False)
