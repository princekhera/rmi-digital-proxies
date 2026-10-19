"""Do people really differ in how behaviour tracks their mood, and do those
differences relate to depression?

For every within-person model whose random slope passed the FDR threshold:

1. Person-specific slopes (fixed effect + random deviation).
2. Robustness: refit without the 2 most extreme people. If the slope
   variation disappears, it was driven by a few individuals.
3. Moderation: does PHQ-9 predict a person's slope? Tested properly as a
   cross-level interaction (fw x PHQ-9) in the mixed model, using PHQ-9 at
   the START of term (measured before the daily data, so the order is
   right) and, secondarily, at the end.
4. A caterpillar plot of the person-specific slopes.
"""
import sys; sys.path.insert(0, snakemake.params.src)  # noqa: E702
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from rmi.models import (OPTIMISERS, bootstrap_spearman, fit_slope_model, person_slopes,
                        prepare_design, slope_lrt_p)

cfg = snakemake.params.cfg
thr = cfg.get("slope_fdr_threshold", 0.05)
panel = pd.read_csv(snakemake.input.panel)
person = pd.read_csv(snakemake.input.person)
mm = pd.read_csv(snakemake.input.mixed)
sel = mm[(mm.level == "within") & (mm.slope_lrt_p_fdr < thr)].sort_values("slope_lrt_p")

slope_rows, summ, mods = [], [], []
for r in sel.itertuples():
    label = f"{r.feature} -> {'next-day ' if r.lagged else ''}{r.outcome}"
    d, formula = prepare_design(panel, r.outcome, r.feature, bool(r.lagged))
    try:
        res = fit_slope_model(d, formula)
    except Exception as e:
        summ.append({"model": label, "note": f"fit failed: {type(e).__name__}"})
        continue
    ps = person_slopes(res)
    fe = float(res.fe_params["fw"])
    for uid, v in ps.items():
        slope_rows.append({"model": label, "uid": uid, "person_slope": v, "fixed_effect": fe})

    # Robustness: drop the 2 people furthest from the average slope.
    extreme = (ps - fe).abs().sort_values(ascending=False).index[:2]
    d2 = d[~d["uid"].isin(extreme)]
    rob_sd, rob_p = np.nan, np.nan
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r1 = smf.mixedlm(formula, d2, groups=d2["uid"], re_formula="~fw").fit(reml=True, method=OPTIMISERS)
            r0 = smf.mixedlm(formula, d2, groups=d2["uid"]).fit(reml=True, method=OPTIMISERS)
        rob_sd = float(np.sqrt(max(r1.cov_re.iloc[1, 1], 0)))
        rob_p = slope_lrt_p(r1.llf, r0.llf)[1]
    except Exception:
        pass

    sign_fe = np.sign(fe) if fe != 0 else 1
    summ.append({
        "model": label, "n_persons": len(ps), "fixed_effect": fe,
        "slope_sd": float(np.sqrt(max(res.cov_re.iloc[1, 1], 0))),
        "share_opposite_sign": float((np.sign(ps) == -sign_fe).mean()),
        "share_below_-0.1": float((ps < -0.1).mean()), "share_above_+0.1": float((ps > 0.1).mean()),
        "slope_sd_without_2_extreme": rob_sd, "slope_lrt_p_without_2_extreme": rob_p,
        "dropped": ", ".join(extreme), "note": "",
    })

    # Moderation by PHQ-9 (cross-level interaction).
    for phq in ("phq9_pre", "phq9_post"):
        dm = d.merge(person[["uid", phq]], on="uid", how="left").dropna(subset=[phq])
        pz = person.set_index("uid")[phq].reindex(dm["uid"].unique())
        dm["phq_z"] = (dm[phq] - pz.mean()) / pz.std(ddof=1)
        f2 = formula + " + phq_z + fw:phq_z"
        est = lo = hi = p = np.nan
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rm = smf.mixedlm(f2, dm, groups=dm["uid"], re_formula="~fw").fit(reml=True, method=OPTIMISERS)
            ci = rm.conf_int()
            est, lo, hi, p = (float(rm.params["fw:phq_z"]), float(ci.loc["fw:phq_z", 0]),
                              float(ci.loc["fw:phq_z", 1]), float(rm.pvalues["fw:phq_z"]))
        except Exception:
            pass
        # Descriptive two-stage check: shrunken slopes vs PHQ-9.
        both = ps.rename("s").to_frame().join(person.set_index("uid")[phq])
        rho, rlo, rhi, rp, rn = bootstrap_spearman(both["s"].to_numpy(float), both[phq].to_numpy(float))
        mods.append({"model": label, "phq": phq, "interaction": est, "ci_low": lo, "ci_high": hi, "p": p,
                     "n_persons": dm["uid"].nunique(), "rho_slope_vs_phq": rho,
                     "rho_ci_low": rlo, "rho_ci_high": rhi})

slopes = pd.DataFrame(slope_rows)
summary = pd.DataFrame(summ)
mod = pd.DataFrame(mods)
if len(mod) and mod["p"].notna().any():
    m = mod["p"].notna()
    mod.loc[m, "p_fdr"] = multipletests(mod.loc[m, "p"], method="fdr_bh")[1]
slopes.to_csv(snakemake.output.slopes, index=False)
summary.to_csv(snakemake.output.summary, index=False)
mod.to_csv(snakemake.output.moderation, index=False)

# Caterpillar plot: one panel per model, people sorted by their slope.
models = list(summary["model"]) if len(summary) else []
fig, axes = plt.subplots(max(len(models), 1), 1, figsize=(8, 2.2 * max(len(models), 1)), squeeze=False)
for ax, name in zip(axes[:, 0], models or ["(no model passed the threshold)"]):
    s = slopes[slopes.model == name].sort_values("person_slope") if len(slopes) else pd.DataFrame()
    if len(s):
        x = np.arange(len(s))
        ax.axhline(0, color="0.6", lw=1)
        ax.axhline(s["fixed_effect"].iloc[0], color="tab:blue", lw=1, ls="--", label="average slope")
        ax.scatter(x, s["person_slope"], s=14, color=np.where(s["person_slope"] < 0, "0.35", "tab:blue"))
        ax.set_xticks([])
        ax.set_ylabel("slope (SD)")
        ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.set_title(name, fontsize=10, loc="left")
axes[-1, 0].set_xlabel("participants, sorted by their own slope")
fig.tight_layout()
fig.savefig(snakemake.output.plot, dpi=150)
