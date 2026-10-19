"""Daily EMA outcomes from StudentLife.

PHQ-9 exists only pre and post term, so it cannot be a repeated outcome.
Daily EMA (PAM affect, stress) is the within-person outcome; PHQ-9 is a
between-person anchor only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .io_studentlife import load_ema, study_day, to_local


def pam_to_valence_arousal(idx: pd.Series, layout: dict) -> pd.DataFrame:
    """Photographic Affect Meter (Pollak et al., 2011): 16 images on a 4x4
    grid. VERIFY the grid orientation against the PAM paper before trusting
    results; it is configurable in config.yaml (pam_layout)."""
    i = pd.to_numeric(idx, errors="coerce") - 1
    row, col = i // 4, i % 4
    valence = col + 1 if layout["valence_axis"] == "col" else row + 1
    arousal = row + 1 if layout["valence_axis"] == "col" else col + 1
    if layout.get("arousal_top_high", True):
        arousal = 5 - arousal
    return pd.DataFrame({"valence": valence, "arousal": arousal})


def daily_outcomes(sl_dir: str, cfg: dict) -> pd.DataFrame:
    parts = []

    pam = load_ema(sl_dir, "PAM")
    if not pam.empty and "picture_idx" in pam:
        va = pam_to_valence_arousal(pam["picture_idx"], cfg["pam_layout"])
        pam = pd.concat([pam[["uid", "resp_time"]].reset_index(drop=True), va], axis=1)
        pam["day"] = study_day(to_local(pam["resp_time"], cfg["tz"]), cfg["day_start_hour"])
        parts.append(pam.groupby(["uid", "day"]).agg(
            valence=("valence", "mean"), arousal=("arousal", "mean"),
            n_pam=("valence", "count")).reset_index())

    stress = load_ema(sl_dir, "Stress")
    if not stress.empty and "level" in stress:
        recode = {int(k): v for k, v in cfg["stress_recode"].items()}
        stress["stress"] = pd.to_numeric(stress["level"], errors="coerce").map(recode)
        stress["day"] = study_day(to_local(stress["resp_time"], cfg["tz"]), cfg["day_start_hour"])
        parts.append(stress.groupby(["uid", "day"]).agg(stress=("stress", "mean")).reset_index())

    if not parts:
        raise RuntimeError("No usable EMA outcomes found")
    out = parts[0]
    for p in parts[1:]:
        out = out.merge(p, on=["uid", "day"], how="outer")
    return out


def add_lead_lag(panel: pd.DataFrame, outcomes: list[str]) -> pd.DataFrame:
    """Next-day outcome, matched on exact calendar day so gaps are not
    silently bridged."""
    p = panel.sort_values(["uid", "day"]).copy()
    p["day"] = pd.to_datetime(p["day"])
    nxt = p[["uid", "day"] + outcomes].copy()
    nxt["day"] = nxt["day"] - pd.Timedelta(days=1)
    nxt = nxt.rename(columns={o: f"{o}_next" for o in outcomes})
    return p.merge(nxt, on=["uid", "day"], how="left")
