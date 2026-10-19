"""Unit tests for the metric definitions. These pin down behaviour on
signals whose answer is known analytically."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rmi.features import haversine_m, shannon_entropy  # noqa: E402
from rmi.models import within_between  # noqa: E402
from rmi.rhythm import interdaily_stability, intradaily_variability, m10_l5  # noqa: E402

IDX = pd.date_range("2020-01-01", periods=24 * 14, freq="h")


def test_entropy_uniform_equals_log_k():
    assert shannon_entropy([5, 5, 5, 5]) == pytest.approx(np.log(4))


def test_entropy_single_place_is_zero():
    assert shannon_entropy([10]) == pytest.approx(0.0)


def test_haversine_one_degree_latitude():
    assert haversine_m(0, 0, 1, 0) == pytest.approx(111_195, rel=1e-3)


def test_is_identical_days_is_one():
    x = pd.Series(np.tile(np.sin(np.arange(24) / 24 * 2 * np.pi) + 2, 14), index=IDX)
    assert interdaily_stability(x) == pytest.approx(1.0)


def test_iv_white_noise_near_two():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(size=len(IDX)), index=IDX)
    assert intradaily_variability(x) == pytest.approx(2.0, abs=0.25)


def test_iv_smooth_signal_low():
    x = pd.Series(np.tile(np.sin(np.arange(24) / 24 * 2 * np.pi), 14), index=IDX)
    assert intradaily_variability(x) < 0.1


def test_m10_l5_wraps_midnight():
    prof = np.zeros(24)
    prof[[22, 23, 0, 1, 2]] = 10  # active block spans midnight
    x = pd.Series(np.tile(prof, 14), index=IDX)
    m10, l5 = m10_l5(x)
    assert m10 == pytest.approx(5.0) and l5 == pytest.approx(0.0)


def test_within_component_has_zero_person_means():
    df = pd.DataFrame({"uid": list("aaabbb"), "x": [1, 2, 3, 10, 20, 30.0]})
    b, w = within_between(df, "x")
    assert w.groupby(df["uid"]).mean().abs().max() < 1e-12
    assert b.groupby(df["uid"]).nunique().max() == 1


# --- added: divergent flag, random-slope LRT, partial Spearman -------------
from rmi.models import flag_divergent, partial_spearman_boot, slope_lrt_p  # noqa: E402


def _row(beta, lo, hi):
    return pd.Series({"beta_std": beta, "ci_low": lo, "ci_high": hi})


def test_divergent_requires_both_cis_to_exclude_zero():
    assert flag_divergent(_row(0.3, 0.1, 0.5), _row(-0.4, -0.6, -0.2))
    assert not flag_divergent(_row(0.03, -0.1, 0.16), _row(-0.02, -0.07, 0.03))  # noise
    assert not flag_divergent(_row(0.3, 0.1, 0.5), _row(-0.05, -0.12, 0.02))     # one clear, one not
    assert not flag_divergent(_row(0.3, 0.1, 0.5), _row(0.2, 0.1, 0.3))          # same sign


def test_slope_lrt_mixture_pvalue():
    lr, p = slope_lrt_p(-100.0, -100.0)
    assert lr == 0 and p == pytest.approx(1.0)
    _, p_small = slope_lrt_p(-90.0, -100.0)          # LR = 20
    assert p_small < 0.001
    _, p_neg = slope_lrt_p(-100.5, -100.0)            # optimiser noise -> LR clipped at 0
    assert p_neg == pytest.approx(1.0)


def test_partial_spearman_removes_confounder():
    rng = np.random.default_rng(1)
    z = rng.integers(0, 2, 200).astype(float)          # e.g. inpatient
    x = z * 2 + rng.normal(0, 0.3, 200)                # both driven by z only
    y = z * 2 + rng.normal(0, 0.3, 200)
    raw = np.corrcoef(x, y)[0, 1]
    r, lo, hi, p, n = partial_spearman_boot(x, y, z, n_boot=200)
    assert raw > 0.8 and abs(r) < 0.2
