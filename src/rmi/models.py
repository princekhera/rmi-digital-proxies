"""Within/between mixed models, person-level anchors, and leakage-aware CV."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests


# lbfgs (the statsmodels default) can stall at the zero-variance boundary and
# return NaN SEs; bfgs then powell are more robust for these small designs.
OPTIMISERS = ["bfgs", "powell"]


def _finite(res) -> bool:
    return bool(np.isfinite(res.bse[["fb", "fw"]]).all())


def slope_lrt_p(llf_slope: float, llf_intercept: float) -> tuple[float, float]:
    """Likelihood-ratio test for adding a random slope (variance + covariance).

    The slope variance sits on the boundary of its parameter space (it cannot
    be negative) under the null, so the usual chi-square(2) reference is too
    conservative. Self & Liang (1987): use a 50:50 mixture of chi-square(1)
    and chi-square(2)."""
    lr = max(2.0 * (llf_slope - llf_intercept), 0.0)
    p = 0.5 * stats.chi2.sf(lr, 1) + 0.5 * stats.chi2.sf(lr, 2)
    return float(lr), float(p)


def flag_divergent(between: pd.Series, within: pd.Series) -> bool:
    """True only when between and within estimates have opposite signs AND
    both 95% CIs exclude zero. Opposite signs on two near-zero estimates are
    noise, not divergence."""
    def clear(r):
        return (r["ci_low"] > 0) or (r["ci_high"] < 0)
    return bool(clear(between) and clear(within)
                and np.sign(between["beta_std"]) != np.sign(within["beta_std"]))


def within_between(d: pd.DataFrame, col: str, uid: str = "uid") -> tuple[pd.Series, pd.Series]:
    """Split a time-varying predictor into a person mean (between) and a
    deviation from that mean (within). Both are standardised so coefficients
    are comparable: between by the SD of person means, within by the pooled
    within-person SD."""
    pm = d.groupby(uid)[col].transform("mean")
    dev = d[col] - pm
    person_means = d.groupby(uid)[col].mean()
    b = (pm - person_means.mean()) / person_means.std(ddof=1)
    w = dev / dev.std(ddof=1)
    return b, w


def fit_within_between(df, outcome, feature, lagged=False, random_slope=True, uid="uid", min_persons=10):
    y = f"{outcome}_next" if lagged else outcome
    cols = [uid, y, feature] + ([outcome] if lagged else [])
    d = df[cols].dropna().copy()
    if d[uid].nunique() < min_persons or d[feature].std() == 0:
        return None
    d["fb"], d["fw"] = within_between(d, feature, uid)
    d["y"] = (d[y] - d[y].mean()) / d[y].std(ddof=1)
    formula = "y ~ fb + fw"
    if lagged:
        # Control for today's outcome (within-person) so 'fw' asks whether
        # today's behaviour predicts tomorrow's mood beyond today's mood.
        _, d["yw_today"] = within_between(d, outcome, uid)
        formula += " + yw_today"

    res, slope_sd, used_slope = None, np.nan, False
    lrt_stat, lrt_p = np.nan, np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Intercept-only model: needed both as a fallback and as the null
        # model for the random-slope likelihood-ratio test. Both use REML,
        # which is valid because the fixed effects are identical.
        res_int = None
        try:
            res_int = smf.mixedlm(formula, d, groups=d[uid]).fit(reml=True, method=OPTIMISERS)
            if not _finite(res_int):
                res_int = None
        except Exception:
            res_int = None
        if random_slope:
            try:
                res = smf.mixedlm(formula, d, groups=d[uid], re_formula="~fw").fit(reml=True, method=OPTIMISERS)
                if res.converged and _finite(res):
                    slope_sd, used_slope = float(np.sqrt(max(res.cov_re.iloc[1, 1], 0))), True
                    if res_int is not None:
                        lrt_stat, lrt_p = slope_lrt_p(res.llf, res_int.llf)
                else:
                    res = None
            except Exception:
                res = None
        if res is None:
            if res_int is None:  # e.g. near-constant feature -> singular design
                return [{"outcome": outcome, "feature": feature, "lagged": lagged,
                         "level": "not_estimable", "note": "intercept model failed"}]
            res = res_int

    ci = res.conf_int()
    rows = []
    for term, level in (("fb", "between"), ("fw", "within")):
        rows.append({
            "outcome": outcome, "feature": feature, "lagged": lagged, "level": level,
            "beta_std": res.params[term], "ci_low": ci.loc[term, 0], "ci_high": ci.loc[term, 1],
            "p": res.pvalues[term], "n_persons": d[uid].nunique(), "n_obs": len(d),
            # SD of person-specific within slopes: how much the
            # behaviour-mood link differs between people.
            "within_slope_sd": slope_sd if level == "within" else np.nan,
            # Is that between-person variation in slopes real? LRT vs a
            # random-intercept-only model (boundary-corrected p-value).
            "slope_lrt_stat": lrt_stat if level == "within" else np.nan,
            "slope_lrt_p": lrt_p if level == "within" else np.nan,
            "random_slope": used_slope, "note": "",
        })
    return rows


def run_mixed_models(panel, features, outcomes, random_slope=True, alpha=0.05):
    rows = []
    for o in outcomes:
        for f in features:
            for lagged in (False, True):
                r = fit_within_between(panel, o, f, lagged=lagged, random_slope=random_slope)
                if r:
                    rows.extend(r)
    res = pd.DataFrame(rows)
    skipped = res[res["level"] == "not_estimable"] if not res.empty else res
    res = res[res["level"] != "not_estimable"].copy() if not res.empty else res
    if not res.empty:
        # Benjamini-Hochberg across every test in the family, not per table.
        res["p_fdr"] = multipletests(res["p"], method="fdr_bh", alpha=alpha)[1]
        res["divergent"] = False
        key = ["outcome", "feature", "lagged"]
        for _, g in res.groupby(key):
            if len(g) == 2:
                b = g[g["level"] == "between"].iloc[0]
                w = g[g["level"] == "within"].iloc[0]
                if flag_divergent(b, w):
                    res.loc[g.index, "divergent"] = True
        # Separate FDR family for the random-slope tests.
        res["slope_lrt_p_fdr"] = np.nan
        m = res["slope_lrt_p"].notna()
        if m.any():
            res.loc[m, "slope_lrt_p_fdr"] = multipletests(res.loc[m, "slope_lrt_p"], method="fdr_bh")[1]
    return pd.concat([res, skipped], ignore_index=True)


def bootstrap_spearman(x, y, n_boot=2000, seed=0):
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = np.asarray(x)[m], np.asarray(y)[m]
    if len(x) < 8 or np.unique(x).size < 2 or np.unique(y).size < 2:
        # Too few people, or a constant variable: correlation undefined.
        return np.nan, np.nan, np.nan, np.nan, len(x)
    rho, p = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        if np.unique(x[i]).size > 1 and np.unique(y[i]).size > 1:
            boots.append(stats.spearmanr(x[i], y[i])[0])
    if not boots:
        return float(rho), np.nan, np.nan, float(p), len(x)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return float(rho), float(lo), float(hi), float(p), len(x)


def hedges_g_boot(a, b, n_boot=2000, seed=0):
    a, b = np.asarray(a)[~np.isnan(a)], np.asarray(b)[~np.isnan(b)]

    def g(a, b):
        na, nb = len(a), len(b)
        sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
        j = 1 - 3 / (4 * (na + nb) - 9)
        return j * (a.mean() - b.mean()) / sp

    rng = np.random.default_rng(seed)
    est = g(a, b)
    boots = [g(rng.choice(a, len(a)), rng.choice(b, len(b))) for _ in range(n_boot)]
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    p = stats.mannwhitneyu(a, b).pvalue
    return float(est), float(lo), float(hi), float(p)


def partial_spearman_boot(x, y, z, n_boot=2000, seed=0):
    """Spearman correlation between x and y after removing a binary or
    numeric covariate z: rank all three, regress the ranks of x and y on the
    rank of z, correlate the residuals. Bootstrap CI over participants."""
    x, y, z = (np.asarray(v, dtype=float) for v in (x, y, z))
    m = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    x, y, z = x[m], y[m], z[m]
    if len(x) < 8 or np.unique(z).size < 2:
        return np.nan, np.nan, np.nan, np.nan, len(x)

    def pr(x, y, z):
        rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
        Z = np.column_stack([np.ones_like(rz), rz])
        ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
        ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
        return stats.pearsonr(ex, ey)

    r, p = pr(x, y, z)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        if np.unique(z[i]).size > 1 and np.unique(x[i]).size > 2 and np.unique(y[i]).size > 2:
            boots.append(pr(x[i], y[i], z[i])[0])
    if not boots:
        return float(r), np.nan, np.nan, float(p), len(x)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return float(r), float(lo), float(hi), float(p), len(x)
