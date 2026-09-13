"""Tests du loader d'export Garmin sur un export synthetique.

Chaque fixture reproduit un piege observe sur un export reel : unites en cm et
en ms, sentinelle 511, nuits non mesurees, champs absents ou nuls.
"""
import json
import math
from datetime import date, datetime, timezone

import pytest

from trailcoach.garmin_export import GarminExport

MS_2026_05_03 = int(datetime(2026, 5, 3, tzinfo=timezone.utc).timestamp() * 1000)
H = 3_600_000


def _write(root, rel, data):
    path = root / "DI_CONNECT" / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _activity(**over):
    a = {
        "activityId": 1,
        "activityType": "running",
        "startTimeGmt": MS_2026_05_03 + 6 * H,
        "startTimeLocal": MS_2026_05_03 + 8 * H,
        "distance": 1_000_000.0,          # cm -> 10 km
        "duration": 3_600_000.0,          # ms -> 60 min
        "movingDuration": 3_000_000.0,
        "elevationGain": 25_000.0,        # cm -> 250 m
        "avgSpeed": 0.25,                 # x10 -> 2.5 m/s -> 400 s/km
        "avgStrideLength": 90.0,          # cm
        "avgHr": 140.0,
        "maxHr": 170.0,
        "activityTrainingLoad": 100.0,
        "hrTimeInZone_2": 1_800_000.0,    # ms -> 30 min
    }
    a.update(over)
    return a


@pytest.fixture
def export(tmp_path):
    _write(tmp_path, "DI-Connect-Fitness/x_summarizedActivities.json", [{
        "summarizedActivitiesExport": [
            _activity(),
            _activity(activityId=2, activityType="walking", startTimeLocal=None,
                      startTimeGmt=MS_2026_05_03 + 30 * H, maxHr=120.0),
        ]}])
    _write(tmp_path, "DI-Connect-Aggregator/UDSFile_a.json", [
        {"calendarDate": "2026-05-03", "restingHeartRate": 55, "currentDayRestingHeartRate": 99,
         "allDayStress": {"aggregatorList": [{"type": "TOTAL", "averageStressLevel": 20}]},
         "bodyBattery": {"bodyBatteryStatList": [
             {"bodyBatteryStatType": "HIGHEST", "statsValue": 90}]}},
        {"calendarDate": "2026-05-05", "restingHeartRate": 57,
         "allDayStress": {"aggregatorList": [{"type": "TOTAL", "averageStressLevel": -1}]}},
    ])
    # chevauchement entre deux fichiers : meme jour lu deux fois
    _write(tmp_path, "DI-Connect-Aggregator/UDSFile_b.json", [
        {"calendarDate": "2026-05-05", "restingHeartRate": 58},
    ])
    _write(tmp_path, "DI-Connect-Wellness/x_sleepData.json", [
        {"retro": False},
        {"calendarDate": "2026-05-03", "sleepWindowConfirmationType": "ENHANCED_CONFIRMED_FINAL",
         "deepSleepSeconds": 3600, "lightSleepSeconds": 18000, "remSleepSeconds": 5400,
         "sleepScores": {"overallScore": 80}},
        {"calendarDate": "2026-05-04", "sleepWindowConfirmationType": "UNCONFIRMED",
         "sleepStartTimestampGMT": "2026-05-03T20:00:00.0",
         "sleepEndTimestampGMT": "2026-05-04T06:00:00.0"},
        {"calendarDate": "2026-05-05", "sleepWindowConfirmationType": "OFF_WRIST",
         "deepSleepSeconds": 0, "lightSleepSeconds": 0},
    ])
    _write(tmp_path, "DI-Connect-Metrics/TrainingReadinessDTO_a.json", [
        {"calendarDate": "2026-05-03", "timestamp": "2026-05-03T18:00:00.0",
         "inputContext": "AFTER_POST_EXERCISE_RESET", "score": 40, "hrvWeeklyAverage": 48},
        {"calendarDate": "2026-05-03", "timestamp": "2026-05-03T07:00:00.0",
         "inputContext": "AFTER_WAKEUP_RESET", "score": 80, "hrvWeeklyAverage": 511},
    ])
    _write(tmp_path, "DI-Connect-Metrics/MetricsAcuteTrainingLoad_a.json", [
        {"calendarDate": MS_2026_05_03, "timestamp": 1, "dailyTrainingLoadAcute": 300},
        {"calendarDate": MS_2026_05_03, "timestamp": 2, "dailyTrainingLoadAcute": 350},
    ])
    return GarminExport(tmp_path)


def test_activities_units(export):
    run = export.activities().iloc[0]
    assert run["distance_km"] == pytest.approx(10.0)
    assert run["duration_min"] == pytest.approx(60.0)
    assert run["elev_gain_m"] == pytest.approx(250.0)
    assert run["pace_s_per_km"] == pytest.approx(400.0)
    assert run["stride_m"] == pytest.approx(0.9)
    assert run["hr_zone2_min"] == pytest.approx(30.0)
    assert run["start_local"] == datetime(2026, 5, 3, 8, 0)
    assert run["utc_offset_h"] == 2.0
    assert run["is_run"]


def test_activity_with_null_local_time(export):
    walk = export.activities().iloc[1]
    assert walk["date"] == date(2026, 5, 4)
    assert walk["utc_offset_h"] is None or math.isnan(walk["utc_offset_h"])


def test_daily_one_row_per_calendar_day(export):
    daily = export.daily()
    assert daily.index.is_unique
    assert list(daily.index) == [date(2026, 5, 3), date(2026, 5, 4), date(2026, 5, 5)]
    assert daily.loc[date(2026, 5, 4), "run_km"] == 0
    assert daily.loc[date(2026, 5, 3), "run_km"] == pytest.approx(10.0)


def test_daily_health_traps(export):
    daily = export.daily()
    d3, d5 = daily.loc[date(2026, 5, 3)], daily.loc[date(2026, 5, 5)]
    assert d3["resting_hr"] == 55                 # pas l'artefact 99
    assert d5["resting_hr"] == 58                 # doublon : la derniere lue
    assert math.isnan(d5["stress_avg"])           # -1 = pas assez mesure
    assert d3["body_battery_high"] == 90
    assert d3["sleep_h"] == pytest.approx(7.5)
    assert d3["sleep_score"] == 80
    assert math.isnan(daily.loc[date(2026, 5, 4), "sleep_h"])   # fenetre par defaut
    assert math.isnan(d5["sleep_h"])                            # montre non portee
    assert d3["readiness"] == 80                  # valeur du reveil
    assert math.isnan(d3["hrv_weekly"])           # 511 = sentinelle
    assert d3["acute_load"] == 350                # derniere mise a jour du jour


def test_metrics_with_missing_keys_do_not_crash(tmp_path):
    _write(tmp_path, "DI-Connect-Metrics/TrainingReadinessDTO_a.json",
           [{"calendarDate": "2026-05-03", "score": 70}])
    _write(tmp_path, "DI-Connect-Metrics/MetricsMaxMetData_a.json",
           [{"calendarDate": "2026-05-03", "sport": "CYCLING", "vo2MaxValue": 50}])
    daily = GarminExport(tmp_path).daily()
    assert daily.loc[date(2026, 5, 3), "readiness"] == 70
    assert "vo2max" not in daily


def test_profile(export):
    prof = export.profile()
    assert prof["observed_hr_max"] == 170         # la marche ne compte pas
    assert prof["resting_hr_median_90d"] == pytest.approx(56.5)


def test_missing_export_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        GarminExport(tmp_path / "nope")
