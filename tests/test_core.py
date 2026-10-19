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
