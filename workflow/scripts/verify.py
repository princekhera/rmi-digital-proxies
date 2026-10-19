"""Checks to run before trusting any result. Writes a markdown report.

1. EMA definitions for Stress and PAM, straight from EMA_definition.json.
2. Raw answer distributions.
3. Empirical sanity checks on the outcome coding (expected signs), including
   the PAM grid read both ways.
4. Rows in the model table with identical estimates.
5. Participant IDs and their data.
"""
import sys; sys.path.insert(0, snakemake.params.src)  # noqa: E702
import json, os
import numpy as np
import pandas as pd
from rmi.io_studentlife import load_ema, study_day, to_local
from rmi.outcomes import pam_to_valence_arousal

sl, cfg = snakemake.params.sl_dir, snakemake.params.cfg
L = ["# Verification report\n", "Run before interpreting any daily-mood or stress result.\n"]

# 1. EMA definitions -------------------------------------------------------
L.append("## 1. EMA definitions (from the dataset itself)\n")
defn_path = os.path.join(sl, "EMA", "EMA_definition.json")
if os.path.exists(defn_path):
    try:
        defs = json.load(open(defn_path))
        items = defs if isinstance(defs, list) else list(defs.values()) if isinstance(defs, dict) else []
        hits = [it for it in items if any(k in json.dumps(it) for k in ("Stress", "PAM"))]
        if not hits:
            hits = [defs]
        for it in hits:
            L.append("```json\n" + json.dumps(it, indent=2)[:4000] + "\n```\n")
    except Exception as e:
        L.append(f"Could not parse {defn_path}: {e}\n")
else:
    L.append(f"**Not found:** `{defn_path}`. Find the definition file and check the stress codes by hand.\n")
L.append(f"\nCurrent stress recode in config (raw code: recoded value, higher = more stressed): `{cfg['stress_recode']}`\n")
L.append(f"\nCurrent PAM layout in config: `{cfg['pam_layout']}`\n")

# 2. Raw distributions -----------------------------------------------------
L.append("\n## 2. Raw answer distributions\n")
stress = load_ema(sl, "Stress")
pam = load_ema(sl, "PAM")
if not stress.empty and "level" in stress:
    L.append("Stress `level` counts:\n\n```\n" + stress["level"].astype(str).value_counts().sort_index().to_string() + "\n```\n")
if not pam.empty and "picture_idx" in pam:
    L.append("PAM `picture_idx` counts:\n\n```\n" + pam["picture_idx"].astype(str).value_counts().sort_index(key=lambda s: pd.to_numeric(s, errors='coerce')).to_string() + "\n```\n")

# 3. Coding sanity checks ----------------------------------------------------
L.append("\n## 3. Do the outcomes behave as their coding claims?\n")
L.append("If valence and stress are both coded correctly, they should move in opposite directions "
         "(within a person, and between people), and valence should be lower in people with higher PHQ-9.\n\n")


def daily(df, valcol):
    d = df.copy()
    d["day"] = study_day(to_local(d["resp_time"], cfg["tz"]), cfg["day_start_hour"])
    return d.groupby(["uid", "day"])[valcol].mean().reset_index()


def within_r(df, a, b):
    d = df[["uid", a, b]].dropna()
    if len(d) < 30:
        return np.nan, len(d)
    da = d[a] - d.groupby("uid")[a].transform("mean")
    db = d[b] - d.groupby("uid")[b].transform("mean")
    return float(np.corrcoef(da, db)[0, 1]), len(d)


checks = []
if not stress.empty and not pam.empty:
    recode = {int(k): v for k, v in cfg["stress_recode"].items()}
    stress["stress"] = pd.to_numeric(stress["level"], errors="coerce").map(recode)
    sd = daily(stress, "stress")
    layouts = {
        "configured": cfg["pam_layout"],
        "alternative (axes swapped)": {"valence_axis": "row" if cfg["pam_layout"]["valence_axis"] == "col" else "col",
                                        "arousal_top_high": cfg["pam_layout"].get("arousal_top_high", True)},
    }
    person = pd.read_csv(snakemake.input.person)
    for name, lay in layouts.items():
        va = pam_to_valence_arousal(pam["picture_idx"], lay)
        p = pd.concat([pam[["uid", "resp_time"]].reset_index(drop=True), va], axis=1)
        pdly = daily(p, "valence").merge(daily(p, "arousal"), on=["uid", "day"])
        m = pdly.merge(sd, on=["uid", "day"])
        r_w, n_w = within_r(m, "valence", "stress")
        pm = m.groupby("uid")[["valence", "stress"]].mean()
        r_b = float(pm.corr(method="spearman").iloc[0, 1]) if len(pm) > 5 else np.nan
        pv = pdly.groupby("uid")["valence"].mean().rename("v").reset_index().merge(person[["uid", "phq9_post"]], on="uid")
        r_phq = float(pv[["v", "phq9_post"]].corr(method="spearman").iloc[0, 1]) if pv["phq9_post"].notna().sum() > 5 else np.nan
        checks.append({"PAM layout": name, "valence vs stress, within-person r (expect < 0)": r_w,
                       "valence vs stress, between-person rho (expect < 0)": r_b,
                       "valence vs PHQ-9 post, rho (expect < 0)": r_phq, "person-days": n_w})
    sp = sd.groupby("uid")["stress"].mean().rename("s").reset_index().merge(person[["uid", "phq9_post"]], on="uid")
    r_sphq = float(sp[["s", "phq9_post"]].corr(method="spearman").iloc[0, 1]) if sp["phq9_post"].notna().sum() > 5 else np.nan
    L.append("```\n" + pd.DataFrame(checks).round(3).T.to_string(header=False) + "\n```")
    L.append(f"\n\nStress (recoded) vs PHQ-9 post, between-person rho (expect > 0): **{r_sphq:.3f}**\n")
    L.append("\n**How to read this:** the configured layout should show the expected signs, and more clearly than the "
             "alternative. If the alternative layout matches expectations better, the PAM mapping is probably wrong. "
             "If stress vs PHQ-9 is negative, the stress recode is probably reversed.\n")

# 3b. Independent check: the separate Mood EMA -----------------------------------
# Mood asks 'Do you feel AT ALL happy?' (1 yes, 2 no) then 'how happy' (1-4),
# and the same for sad. Score each 0-4 (0 = 'no'), balance = happy - sad.
L.append("\n### 3b. Independent check against the Mood question\n")
mood = load_ema(sl, "Mood")
if (not mood.empty and {"happyornot", "sadornot"} <= set(mood.columns)
        and not pam.empty and not stress.empty):
    def score(flag, level):
        f = pd.to_numeric(mood[flag], errors="coerce")
        lv = pd.to_numeric(mood.get(level), errors="coerce")
        return np.where(f == 2, 0.0, np.where(f == 1, lv, np.nan))
    mood["mood_balance"] = score("happyornot", "happy") - score("sadornot", "sad")
    md = daily(mood.dropna(subset=["mood_balance"]), "mood_balance")
    va = pam_to_valence_arousal(pam["picture_idx"], cfg["pam_layout"])
    pv = daily(pd.concat([pam[["uid", "resp_time"]].reset_index(drop=True), va], axis=1), "valence")
    m = md.merge(pv, on=["uid", "day"])
    r_mv, n_mv = within_r(m, "mood_balance", "valence")
    r_ms, n_ms = within_r(md.merge(sd, on=["uid", "day"]), "mood_balance", "stress")
    L.append(f"Mood balance (happy minus sad) vs PAM valence, within-person r (expect > 0): "
             f"**{r_mv:.3f}** on {n_mv} person-days\n\n")
    L.append(f"Mood balance vs stress, within-person r (expect < 0): **{r_ms:.3f}** on {n_ms} person-days\n")
    L.append("\nThe Mood question was asked on fewer days than PAM, so expect fewer person-days here.\n")
else:
    L.append("Mood EMA not found or missing the happy/sad fields; check skipped.\n")

# 4. Identical rows ------------------------------------------------------------
L.append("\n## 4. Rows in the model table with identical estimates\n")
mm = pd.read_csv(snakemake.input.mixed)
mm = mm[mm["level"].isin(["between", "within"])].copy()
key = ["beta_std", "ci_low", "ci_high", "p"]
mm["_k"] = mm[key].round(12).astype(str).agg("|".join, axis=1)
dups = mm[mm.duplicated("_k", keep=False)].sort_values("_k")
if len(dups):
    L.append("These rows match to 12 decimal places, which is very unlikely by chance. Check the feature columns in `panel.csv` "
             "for two features that are identical or perfectly correlated.\n\n```\n"
             + dups[["outcome", "feature", "lagged", "level"] + key].to_string(index=False, float_format=lambda v: f"{v:.10f}") + "\n```\n")
else:
    L.append("No rows match at full precision. Rows that look identical in the report differ beyond the 3rd decimal: coincidence of rounding.\n")
near = mm.copy(); near["_k3"] = near[key].round(3).astype(str).agg("|".join, axis=1)
nd = near[near.duplicated("_k3", keep=False)].sort_values("_k3")
if len(nd):
    L.append("\nRows identical at 3 decimals (as printed in the report), shown at full precision:\n\n```\n"
             + nd[["outcome", "feature", "lagged", "level"] + key].to_string(index=False, float_format=lambda v: f"{v:.10f}") + "\n```\n")
panel = pd.read_csv(snakemake.input.panel)
feats = [f for f in cfg["features"] if f in panel]
corr = panel[feats].corr().abs()
pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(feats) for b in feats[i + 1:] if corr.loc[a, b] > 0.95]
if pairs:
    L.append("\nFeature pairs with |r| > 0.95 in the panel (near-duplicates):\n\n" + "\n".join(f"- {a} vs {b}: r = {r:.4f}" for a, b, r in pairs) + "\n")

# 5. Participants ---------------------------------------------------------------
L.append("\n## 5. Participants\n")
raw = pd.concat([pd.read_csv(f) for f in snakemake.input.daily if pd.read_csv(f).shape[0]], ignore_index=True)
per = raw.groupby("uid")["day"].nunique().rename("days_with_gps").to_frame()
per = per.join(panel.groupby("uid")["day"].nunique().rename("days_in_panel"), how="left").fillna(0)
person = pd.read_csv(snakemake.input.person).set_index("uid")
per = per.join(person[["phq9_pre", "phq9_post"]], how="left")
L.append(f"GPS files found: **{len(snakemake.input.daily)}** (the StudentLife paper reports 48 participants).\n\n")
L.append("Fewest days in the panel (check these for a stray or test ID):\n\n```\n" + per.sort_values("days_in_panel").head(5).to_string() + "\n```\n")

open(snakemake.output[0], "w").write("\n".join(L))
