import sys; sys.path.insert(0, snakemake.params.src)  # noqa: E702
import json
import pandas as pd
from rmi.outcomes import daily_outcomes, add_lead_lag
from rmi.io_studentlife import load_phq9

cfg = snakemake.params.cfg
feats = pd.concat([pd.read_csv(f) for f in snakemake.input.daily if pd.read_csv(f).shape[0]], ignore_index=True)
feats["day"] = pd.to_datetime(feats["day"])
n_days_raw = len(feats)
feats = feats[feats["coverage_h"] >= cfg["min_coverage_hours"]]

out = daily_outcomes(snakemake.params.sl_dir, cfg)
out["day"] = pd.to_datetime(out["day"])
panel = feats.merge(out, on=["uid", "day"], how="inner")
panel = add_lead_lag(panel, cfg["outcomes"])
panel.to_csv(snakemake.output.panel, index=False)

person = pd.DataFrame([json.load(open(f)) for f in snakemake.input.person])
means = panel.groupby("uid")[cfg["features"]].mean().reset_index()
person = person.merge(means, on="uid", how="left")
phq = load_phq9(snakemake.params.sl_dir)
for t in ("pre", "post"):
    person = person.merge(phq[phq["type"] == t][["uid", "phq9_total"]]
                          .rename(columns={"phq9_total": f"phq9_{t}"}), on="uid", how="left")
person.to_csv(snakemake.output.person, index=False)

json.dump({
    "person_days_raw": int(n_days_raw),
    "person_days_after_coverage_gate": int(len(feats)),
    "person_days_in_panel": int(len(panel)),
    "participants_in_panel": int(panel["uid"].nunique()),
    "participants_with_phq9_post": int(person["phq9_post"].notna().sum()),
}, open(snakemake.output.summary, "w"), indent=2)
