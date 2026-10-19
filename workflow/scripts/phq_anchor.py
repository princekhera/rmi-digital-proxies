import sys; sys.path.insert(0, snakemake.params.src)  # noqa: E702
import pandas as pd
from statsmodels.stats.multitest import multipletests
from rmi.models import bootstrap_spearman

cfg = snakemake.params.cfg
person = pd.read_csv(snakemake.input.person)
rows = []
for f in cfg["features"] + ["circadian_movement"]:
    if f not in person:
        continue
    rho, lo, hi, p, n = bootstrap_spearman(person[f].to_numpy(float), person["phq9_post"].to_numpy(float))
    rows.append({"feature": f, "rho": rho, "ci_low": lo, "ci_high": hi, "p": p, "n": n})
res = pd.DataFrame(rows).dropna(subset=["p"])
if len(res):
    res["p_fdr"] = multipletests(res["p"], method="fdr_bh")[1]
res.to_csv(snakemake.output[0], index=False)
