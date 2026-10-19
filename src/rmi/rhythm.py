"""Nonparametric rest-activity rhythm metrics (Van Someren et al., 1999).

IS  interdaily stability: 0 (no day-to-day consistency) to 1 (identical days)
IV  intradaily variability: ~0 (smooth) to ~2 (white noise); higher = fragmented
RA  relative amplitude: (M10 - L5) / (M10 + L5)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def to_hourly(minute: pd.DataFrame, ts_col="timestamp", val_col="activity") -> pd.Series:
    s = minute.set_index(pd.to_datetime(minute[ts_col]))[val_col].astype(float)
    return s.resample("h").mean()


def complete_days(hourly: pd.Series) -> pd.Series:
    """Keep only calendar days with 24 non-missing hours."""
    ok = hourly.groupby(hourly.index.date).transform(lambda x: x.notna().sum() == 24)
    return hourly[ok.astype(bool)]


def interdaily_stability(x: pd.Series) -> float:
    x = x.dropna()
    n = len(x)
    if n < 48:
        return np.nan
    xbar = x.mean()
    hour_means = x.groupby(x.index.hour).mean()
    num = n * ((hour_means - xbar) ** 2).sum()
    den = 24 * ((x - xbar) ** 2).sum()
    return float(num / den) if den > 0 else np.nan


def intradaily_variability(x: pd.Series) -> float:
    v = x.dropna().to_numpy()
    n = len(v)
    if n < 48:
        return np.nan
    den = (n - 1) * ((v - v.mean()) ** 2).sum()
    return float(n * (np.diff(v) ** 2).sum() / den) if den > 0 else np.nan


def m10_l5(x: pd.Series) -> tuple[float, float]:
    """Most active 10 h and least active 5 h of the average 24 h profile,
    with windows allowed to wrap past midnight."""
    prof = x.groupby(x.index.hour).mean().reindex(range(24)).to_numpy()
    wrap = np.concatenate([prof, prof])
    m10 = max(wrap[i:i + 10].mean() for i in range(24))
    l5 = min(wrap[i:i + 5].mean() for i in range(24))
    return float(m10), float(l5)


def rhythm_summary(minute: pd.DataFrame) -> dict:
    h = complete_days(to_hourly(minute))
    m10, l5 = m10_l5(h) if len(h) else (np.nan, np.nan)
    return {
        "n_days": len(h) // 24,
        "mean_activity": float(h.mean()) if len(h) else np.nan,
        "IS": interdaily_stability(h),
        "IV": intradaily_variability(h),
        "M10": m10,
        "L5": l5,
        "RA": (m10 - l5) / (m10 + l5) if (m10 + l5) > 0 else np.nan,
    }
