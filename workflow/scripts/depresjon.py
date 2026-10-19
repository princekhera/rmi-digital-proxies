import sys; sys.path.insert(0, snakemake.params.src)  # noqa: E702
import glob, os
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from rmi.rhythm import rhythm_summary
from rmi.models import bootstrap_spearman, hedges_g_boot

root = snakemake.params.dep_dir
rows = []
for group in ("condition", "control"):
    for f in sorted(glob.glob(os.path.join(root, group, f"{group}_*.csv"))):
        pid = os.path.basename(f).replace(".csv", "")
        rows.append({"number": pid, "group": group, **rhythm_summary(pd.read_csv(f))})
rhythm = pd.DataFrame(rows)
scores = pd.read_csv(os.path.join(root, "scores.csv"))
rhythm = rhythm.merge(scores, on="number", how="left")
rhythm.to_csv(snakemake.output.metrics, index=False)

metrics = ["mean_activity", "IS", "IV", "RA", "M10", "L5"]
cond, ctrl = rhythm[rhythm.group == "condition"], rhythm[rhythm.group == "control"]
res = []
for m in metrics:
    g, lo, hi, p = hedges_g_boot(cond[m].to_numpy(float), ctrl[m].to_numpy(float))
    res.append({"metric": m, "test": "condition_vs_control_hedges_g", "estimate": g, "ci_low": lo, "ci_high": hi, "p": p, "n": len(rhythm)})
    rho, lo, hi, p, n = bootstrap_spearman(cond[m].to_numpy(float), cond["madrs1"].to_numpy(float))
    res.append({"metric": m, "test": "spearman_vs_MADRS_within_patients", "estimate": rho, "ci_low": lo, "ci_high": hi, "p": p, "n": n})
res = pd.DataFrame(res).dropna(subset=["p"])
res["p_fdr"] = multipletests(res["p"], method="fdr_bh")[1]
res.to_csv(snakemake.output.tests, index=False)
