"""Loaders for the StudentLife dataset (Wang et al., UbiComp 2014).

The raw CSVs have known quirks: leading spaces in column names and trailing
commas. Every loader strips column names and tolerates missing files, because
not every participant has every sensor stream.
"""
from __future__ import annotations

import glob
import json
import os

import pandas as pd


def read_csv_clean(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, index_col=False)
    df.columns = [c.strip() for c in df.columns]
    return df


def sensor_path(sl_dir: str, stream: str, uid: str) -> str:
    return os.path.join(sl_dir, "sensing", stream, f"{stream}_{uid}.csv")


def to_local(ts: pd.Series, tz: str) -> pd.Series:
    return pd.to_datetime(ts.astype("int64"), unit="s", utc=True).dt.tz_convert(tz)


def study_day(local: pd.Series, day_start_hour: int) -> pd.Series:
    """Assign each timestamp to a 'study day' that starts at day_start_hour,
    so 01:00 activity belongs to the previous evening, not the next day."""
    return (local - pd.Timedelta(hours=day_start_hour)).dt.date


def ema_dir(sl_dir: str) -> str:
    # The docs say EMA/responses; some copies of the archive use EMA/response.
    for name in ("response", "responses"):
        p = os.path.join(sl_dir, "EMA", name)
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(f"No EMA response folder under {sl_dir}/EMA")


def load_ema(sl_dir: str, question: str) -> pd.DataFrame:
    rows = []
    pattern = os.path.join(ema_dir(sl_dir), question, f"{question}_u*.json")
    for f in sorted(glob.glob(pattern)):
        uid = os.path.basename(f).split("_")[-1].replace(".json", "")
        with open(f) as fh:
            try:
                entries = json.load(fh)
            except json.JSONDecodeError:
                continue
        for e in entries or []:
            if isinstance(e, dict) and "resp_time" in e:
                rows.append({"uid": uid, **e})
    return pd.DataFrame(rows)


PHQ_MAP = {
    "not at all": 0,
    "several days": 1,
    "more than half the days": 2,
    "nearly every day": 3,
}


def load_phq9(sl_dir: str) -> pd.DataFrame:
    """Return uid, type (pre/post), phq9_total. Items are the 9 columns after
    uid and type; a trailing 'difficulty' column, if present, is excluded."""
    df = read_csv_clean(os.path.join(sl_dir, "survey", "PHQ-9.csv"))
    if df.empty:
        return pd.DataFrame(columns=["uid", "type", "phq9_total"])
    items = df.columns[2:11]
    scored = df[items].apply(lambda c: c.astype(str).str.strip().str.lower().map(PHQ_MAP))
    out = df[["uid", "type"]].copy()
    out["type"] = out["type"].astype(str).str.strip().str.lower()
    # Require all 9 items; partial sums would bias totals downward.
    out["phq9_total"] = scored.sum(axis=1, min_count=9)
    return out
