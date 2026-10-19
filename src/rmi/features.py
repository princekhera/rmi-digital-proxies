"""Person-day behavioural features from passive smartphone sensing.

Definitions follow Saeb et al. (2015, JMIR) where possible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import lombscargle
from sklearn.cluster import DBSCAN

from .io_studentlife import study_day, to_local

EARTH_R_M = 6_371_000.0


def haversine_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R_M * np.arcsin(np.sqrt(a))


def shannon_entropy(weights) -> float:
    w = np.asarray(weights, dtype=float)
    w = w[w > 0]
    if w.size == 0:
        return np.nan
    p = w / w.sum()
    return float(-(p * np.log(p)).sum())


def circadian_movement(g: pd.DataFrame, band_h=(23.5, 24.5)) -> float:
    """Saeb et al. 2015: log of Lomb-Scargle power in the ~24 h band for
    latitude + longitude. Person-level only: one day cannot estimate a rhythm."""
    if len(g) < 50:
        return np.nan
    t = (g["time"].to_numpy(dtype=float) - g["time"].min()) / 3600.0
    if t.max() < 72:
        return np.nan
    freqs = 2 * np.pi / np.linspace(band_h[0], band_h[1], 25)
    e = 0.0
    for col in ("latitude", "longitude"):
        y = g[col].to_numpy(dtype=float)
        y = y - y.mean()
        if np.allclose(y, 0):
            continue
        e += lombscargle(t, y, freqs, normalize=True).mean()
    return float(np.log(e + 1e-10))


def gps_daily(gps: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Daily mobility features, plus person-level circadian movement."""
    if gps.empty:
        return pd.DataFrame(), {}
    g = gps.dropna(subset=["latitude", "longitude"]).copy()
    if "accuracy" in g:
        g = g[g["accuracy"] <= cfg["gps_max_accuracy_m"]]
    g = g.sort_values("time").reset_index(drop=True)
    if g.empty:
        return pd.DataFrame(), {}
    g["local"] = to_local(g["time"], cfg["tz"])
    g["day"] = study_day(g["local"], cfg["day_start_hour"])

    # Time each sample represents = gap to the next sample, capped so long
    # recording gaps are not credited to one location.
    gap = g["time"].shift(-1) - g["time"]
    g["dt_min"] = (gap / 60).clip(upper=cfg["gps_max_gap_min"]).fillna(0)

    if "travelstate" in g:
        stationary = g["travelstate"].astype(str).str.strip().eq("stationary")
    else:
        stationary = g.get("speed", pd.Series(0, index=g.index)).fillna(0) < cfg["stationary_speed_ms"]
    st = g[stationary]
    g["cluster"] = -1
    if len(st) >= cfg["dbscan_min_samples"]:
        db = DBSCAN(
            eps=cfg["dbscan_eps_m"] / EARTH_R_M,
            min_samples=cfg["dbscan_min_samples"],
            metric="haversine",
            algorithm="ball_tree",
        ).fit(np.radians(st[["latitude", "longitude"]].to_numpy()))
        g.loc[st.index, "cluster"] = db.labels_

    # Home = the cluster where the most night-time (00-06 local) is spent.
    night = g[(g["local"].dt.hour < 6) & (g["cluster"] >= 0)]
    home = night.groupby("cluster")["dt_min"].sum().idxmax() if len(night) else None

    rows = []
    for day, d in g.groupby("day"):
        w = d[d["cluster"] >= 0].groupby("cluster")["dt_min"].sum()
        n = int((w > 0).sum())
        ent = shannon_entropy(w.values)
        if len(d) > 1:
            dist = haversine_m(d["latitude"].values[:-1], d["longitude"].values[:-1],
                               d["latitude"].values[1:], d["longitude"].values[1:]).sum()
            loc_var = float(np.log(d["latitude"].var() + d["longitude"].var() + 1e-10))
        else:
            dist, loc_var = 0.0, np.nan
        known = d["dt_min"].sum()
        rows.append({
            "day": day,
            "coverage_h": d["local"].dt.floor("h").nunique(),
            "loc_var": loc_var,
            "loc_entropy": ent,
            "loc_entropy_norm": ent / np.log(n) if n > 1 else (0.0 if n == 1 else np.nan),
            "n_places": n,
            "home_frac": (d.loc[d["cluster"] == home, "dt_min"].sum() / known)
            if (home is not None and known > 0) else np.nan,
            "distance_km": dist / 1000,
        })
    return pd.DataFrame(rows), {"circadian_movement": circadian_movement(g)}


def activity_daily(act: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if act.empty:
        return pd.DataFrame()
    col = [c for c in act.columns if "activity" in c.lower()][0]
    a = act[act[col].isin([0, 1, 2])].copy()  # 3 = unknown, dropped
    a["day"] = study_day(to_local(a["timestamp"], cfg["tz"]), cfg["day_start_hour"])
    return (a.groupby("day")[col].apply(lambda s: s.isin([1, 2]).mean())
            .rename("active_frac").reset_index())


def conversation_daily(conv: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if conv.empty:
        return pd.DataFrame()
    c = conv.rename(columns={"start_timestamp": "start", "end_timestamp": "end"}).copy()
    c["day"] = study_day(to_local(c["start"], cfg["tz"]), cfg["day_start_hour"])
    c["min"] = (c["end"] - c["start"]) / 60
    return c.groupby("day").agg(conv_n=("min", "size"), conv_min=("min", "sum")).reset_index()


def _overlap_min(starts, ends, w0, w1) -> float:
    lo = np.maximum(starts, w0)
    hi = np.minimum(ends, w1)
    return float(np.clip(hi - lo, 0, None).sum() / 60)


def interval_daily(iv: pd.DataFrame, cfg: dict, prefix: str) -> pd.DataFrame:
    """phonelock / dark streams (start, end in epoch s): episodes and minutes
    per day, plus minutes overlapping the night window."""
    if iv.empty:
        return pd.DataFrame()
    iv = iv.copy()
    iv["day"] = study_day(to_local(iv["start"], cfg["tz"]), cfg["day_start_hour"])
    iv["min"] = (iv["end"] - iv["start"]) / 60
    out = iv.groupby("day").agg(**{f"{prefix}_n": ("min", "size"),
                                   f"{prefix}_min": ("min", "sum")}).reset_index()

    off_h = (cfg["night_start_hour"] - cfg["day_start_hour"]) % 24
    len_h = (cfg["night_end_hour"] - cfg["night_start_hour"]) % 24
    s, e = iv["start"].to_numpy(dtype=float), iv["end"].to_numpy(dtype=float)
    night = []
    for day in out["day"]:
        w0 = (pd.Timestamp(day).tz_localize(cfg["tz"])
              + pd.Timedelta(hours=cfg["day_start_hour"] + off_h)).timestamp()
        night.append(_overlap_min(s, e, w0, w0 + len_h * 3600))
    out[f"{prefix}_night_min"] = night
    if prefix == "lock":
        # Unlocked night minutes. Only valid on high-coverage days: a dead
        # phone also looks 'unlocked'. The panel step applies that gate.
        out["night_phone_use_min"] = len_h * 60 - out["lock_night_min"]
    return out


def participant_features(streams: dict, cfg: dict) -> tuple[pd.DataFrame, dict]:
    daily, person = gps_daily(streams.get("gps", pd.DataFrame()), cfg)
    if daily.empty:
        return daily, person
    for part in (
        activity_daily(streams.get("activity", pd.DataFrame()), cfg),
        conversation_daily(streams.get("conversation", pd.DataFrame()), cfg),
        interval_daily(streams.get("phonelock", pd.DataFrame()), cfg, "lock"),
        interval_daily(streams.get("dark", pd.DataFrame()), cfg, "dark"),
    ):
        if not part.empty:
            daily = daily.merge(part, on="day", how="left")
    return daily, person
