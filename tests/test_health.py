"""Tests des analyses charge x sante sur des series synthetiques a effet connu."""
import math
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from trailcoach import health


def _days(n):
    start = date(2025, 1, 1)
    return [start + timedelta(days=i) for i in range(n)]


@pytest.fixture
def daily():
    """300 jours : la FC repos monte de 0,1 bpm par km couru LA VEILLE, sur une
    tendance de fond qui baisse de 5 bpm (gain de forme)."""
    rng = np.random.default_rng(0)
    n = 300
    km = np.where(rng.random(n) < 0.5, rng.uniform(5, 25, n), 0.0)
    trend = np.linspace(60, 55, n)
    rhr = trend + 0.1 * np.roll(km, 1) + rng.normal(0, 0.5, n)
    rhr[0] = trend[0]
    return pd.DataFrame({
        "run_km": km,
        "run_elev_m": km * 10,
        "resting_hr": rhr,
        "stress_avg": rng.normal(25, 5, n),
        "acwr": rng.uniform(0.5, 1.8, n),
    }, index=_days(n))


def test_baseline_deviation_excludes_current_day():
    s = pd.Series([10.0] * 12 + [20.0])
    dev = health.baseline_deviation(s, window=28, min_periods=10)
    assert dev.iloc[-1] == 10.0
    assert math.isnan(dev.iloc[5])


def test_spearman_autocorrelation_reduces_n():
    t = pd.Series(np.arange(200, dtype=float))
    noise = pd.Series(np.random.default_rng(1).normal(0, 1, 200))
    c = health.spearman(t, t + noise * 5)
    assert c.rho > 0.9
    assert c.n == 200
    assert c.n_eff < 20
    independent = health.spearman(t, t + noise * 5, autocorrelated=False)
    assert independent.n_eff == 200


def test_spearman_too_few_points():
    c = health.spearman(pd.Series([1.0, 2.0]), pd.Series([2.0, 1.0]))
    assert math.isnan(c.rho)


def test_bh_adjust():
    q = health.bh_adjust(pd.Series([0.01, 0.04, 0.03, math.nan]))
    assert q.iloc[0] == pytest.approx(0.03)
    assert q.iloc[1] == pytest.approx(0.04)
    assert q.iloc[2] == pytest.approx(0.04)
    assert math.isnan(q.iloc[3])


def test_lagged_correlations_find_planted_lag(daily):
    corr = health.lagged_correlations(daily)
    rhr = corr[(corr["load"] == "run_km") & (corr["marker"] == "resting_hr_dev")].set_index("lag")
    assert rhr.loc[1, "rho"] > 0.5
    assert rhr.loc[1, "q"] < 0.01
    assert abs(rhr.loc[0, "rho"]) < 0.2
    stress = corr[corr["marker"] == "stress_avg_dev"]
    assert (stress["q"] > 0.05).all()


def test_lag_controls_alternating_days():
    """Sortie un jour sur deux, stress lie a la seule charge du jour meme :
    sans controle, J+1 sortirait fortement negatif."""
    rng = np.random.default_rng(2)
    n = 300
    km = np.where(np.arange(n) % 2 == 0, rng.uniform(5, 25, n), 0.0)
    d = pd.DataFrame({"run_km": km, "run_elev_m": km * 10,
                      "stress_avg": 25 + 0.5 * km + rng.normal(0, 1, n)}, index=_days(n))
    raw = health.spearman(d["run_km"], health.baseline_deviation(d["stress_avg"]).shift(-1))
    assert raw.rho < -0.5
    corr = health.lagged_correlations(d)
    corr = corr[corr["load"] == "run_km"].set_index(["marker", "lag"])
    assert corr.loc[("stress_avg_dev", 0), "rho"] > 0.5
    assert abs(corr.loc[("stress_avg_dev", 1), "rho"]) < 0.2


def test_garmin_load_empty_before_first_measure():
    d = pd.DataFrame({"run_km": [0.0, 5, 0, 8], "activity_load": [math.nan, math.nan, math.nan, 80]},
                     index=_days(4))
    d.loc[d.index[1], "activity_load"] = 50
    load = health.load_features(d)["garmin_load"]
    assert math.isnan(load.iloc[0])
    assert list(load.iloc[1:]) == [50, 0, 80]


def _acts(rows):
    base = {"is_run": True, "sport": "running", "moving_min": 45.0, "gap_pace_s_per_km": math.nan}
    df = pd.DataFrame([{**base, **r} for r in rows])
    df["date"] = df["start_local"].dt.date
    return df


def test_run_efficiency_filters_and_reference():
    start = datetime(2025, 3, 1, 8)
    rows = [{"start_local": start + timedelta(days=i), "avg_hr": 140.0, "pace_s_per_km": 360.0}
            for i in range(6)]
    rows[-1]["pace_s_per_km"] = 327.27               # 10 % plus rapide a meme FC
    rows.append({"start_local": start + timedelta(days=7), "avg_hr": 180.0,
                 "pace_s_per_km": 280.0})             # fractionne : hors endurance
    rows.append({"start_local": start + timedelta(days=8), "avg_hr": 140.0,
                 "pace_s_per_km": 360.0, "sport": "treadmill_running"})
    runs = health.run_efficiency(_acts(rows), hr_max=190)
    assert len(runs) == 6
    assert runs["efficiency"].iloc[0] == pytest.approx(60000 / 360 / 140)
    assert math.isnan(runs["eff_dev"].iloc[4])       # moins de 5 sorties avant
    assert runs["eff_dev"].iloc[5] == pytest.approx(0.10, abs=0.001)


def test_acwr_bands(daily):
    bands = health.acwr_bands(daily)
    assert list(bands.index) == ["< 0,8", "0,8 - 1,3", "1,3 - 1,5", ">= 1,5"]
    assert bands["days"].sum() == len(daily)


def test_monthly_trend(daily):
    trend = health.monthly_trend(daily)
    jan = trend.iloc[0]
    assert jan["km_per_week"] == pytest.approx(daily["run_km"].iloc[:31].sum() / 31 * 7)
