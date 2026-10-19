"""Generate synthetic data with the StudentLife and Depresjon file layouts.

Used only to smoke-test the pipeline. A known within-person effect is
planted (more time at home than usual -> lower valence) alongside an
opposite-signed between-person effect, so the within/between decomposition
has something to recover.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
TZ = "America/New_York"


def studentlife(root, n_users=14, n_days=25):
    for sub in ("gps", "activity", "conversation", "phonelock", "dark"):
        os.makedirs(f"{root}/sensing/{sub}", exist_ok=True)
    os.makedirs(f"{root}/survey", exist_ok=True)
    start = pd.Timestamp("2013-03-27", tz=TZ)
    phq_rows = []
    for u in range(n_users):
        uid = f"u{u:02d}"
        home = np.array([43.70, -72.29]) + rng.normal(0, 0.01, 2)
        places = home + rng.normal(0, 0.02, (3, 2))
        person_home = rng.uniform(0.45, 0.85)
        gps, act, conv, lock, dark, pam, stress = [], [], [], [], [], [], []
        for d in range(n_days):
            day0 = start + pd.Timedelta(days=d)
            p_day = np.clip(person_home + rng.normal(0, 0.12), 0.05, 0.98)
            for slot in range(144):
                t = day0 + pd.Timedelta(minutes=10 * slot)
                ts = int(t.timestamp())
                at_home = t.hour < 7 or rng.random() < p_day
                loc = home if at_home else places[rng.integers(3)]
                lat, lon = loc + rng.normal(0, 0.0002, 2)
                gps.append([ts, "network", "wifi", 30, lat, lon, 0, 0, 0, "stationary"])
                act.append([ts, int(rng.random() < (0.05 if at_home else 0.3))])
            for _ in range(rng.integers(2, 8)):
                s = int((day0 + pd.Timedelta(hours=rng.uniform(9, 22))).timestamp())
                conv.append([s, s + int(rng.uniform(60, 1800))])
            for h in np.sort(rng.uniform(0, 24, 20)):
                s = int((day0 + pd.Timedelta(hours=h)).timestamp())
                lock.append([s, s + int(rng.uniform(600, 5400))])
            s = int((day0 + pd.Timedelta(hours=rng.uniform(23, 25))).timestamp())
            dark.append([s, s + int(rng.uniform(5, 9) * 3600)])
            # Planted effects: within-person negative, between-person positive.
            val = 2.5 + 2.0 * (person_home - 0.65) - 4.0 * (p_day - person_home) + rng.normal(0, 0.4)
            vcol = int(np.clip(np.round(val), 1, 4)) - 1
            for _ in range(rng.integers(1, 4)):
                r = int((day0 + pd.Timedelta(hours=rng.uniform(10, 22))).timestamp())
                pam.append({"picture_idx": str(rng.integers(0, 4) * 4 + vcol + 1), "resp_time": r})
            stress.append({"level": str(rng.integers(1, 6)), "resp_time": int((day0 + pd.Timedelta(hours=20)).timestamp())})
        cols = "time,provider,network_type,accuracy,latitude,longitude,altitude,bearing,speed,travelstate".split(",")
        pd.DataFrame(gps, columns=cols).to_csv(f"{root}/sensing/gps/gps_{uid}.csv", index=False)
        pd.DataFrame(act, columns=["timestamp", " activity inference"]).to_csv(f"{root}/sensing/activity/activity_{uid}.csv", index=False)
        pd.DataFrame(conv, columns=["start_timestamp", " end_timestamp"]).to_csv(f"{root}/sensing/conversation/conversation_{uid}.csv", index=False)
        pd.DataFrame(lock, columns=["start", "end"]).to_csv(f"{root}/sensing/phonelock/phonelock_{uid}.csv", index=False)
        pd.DataFrame(dark, columns=["start", "end"]).to_csv(f"{root}/sensing/dark/dark_{uid}.csv", index=False)
        for q, data in (("PAM", pam), ("Stress", stress)):
            os.makedirs(f"{root}/EMA/response/{q}", exist_ok=True)
            json.dump(data, open(f"{root}/EMA/response/{q}/{q}_{uid}.json", "w"))
        opts = ["Not at all", "Several days", "More than half the days", "Nearly every day"]
        for typ in ("pre", "post"):
            phq_rows.append([uid, typ] + [opts[rng.integers(0, 4)] for _ in range(9)] + ["Somewhat difficult"])
    items = [f"item{i}" for i in range(1, 10)]
    pd.DataFrame(phq_rows, columns=["uid", "type"] + items + ["Response"]).to_csv(f"{root}/survey/PHQ-9.csv", index=False)


def depresjon(root, n_each=10, n_days=10):
    rows = []
    for group in ("condition", "control"):
        os.makedirs(f"{root}/{group}", exist_ok=True)
        for i in range(1, n_each + 1):
            madrs = rng.integers(12, 35) if group == "condition" else np.nan
            amp = 300 - (5 * madrs if group == "condition" else 0)
            t = pd.date_range("2003-05-07 12:00", periods=n_days * 1440, freq="min")
            h = t.hour + t.minute / 60
            act = np.clip(amp * (np.sin((h - 9) / 24 * 2 * np.pi) + 0.6) + rng.normal(0, 80, len(t)), 0, None)
            pd.DataFrame({"timestamp": t.strftime("%Y-%m-%d %H:%M:%S"), "date": t.strftime("%Y-%m-%d"),
                          "activity": act.round()}).to_csv(f"{root}/{group}/{group}_{i}.csv", index=False)
            rows.append({"number": f"{group}_{i}", "days": n_days, "gender": rng.integers(1, 3),
                         "age": "35-39", "madrs1": madrs, "madrs2": madrs})
    pd.DataFrame(rows).to_csv(f"{root}/scores.csv", index=False)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "data/synthetic"
    studentlife(f"{out}/studentlife/dataset")
    depresjon(f"{out}/depresjon")
    print(f"Synthetic data written to {out}")
