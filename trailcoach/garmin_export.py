"""Lecture de l'export RGPD Garmin Connect ("Exporter vos donnees").

L'export est un dossier (une fois dezippe) qui contient `DI_CONNECT/`. On en
tire deux tables normalisees, identiques d'un athlete a l'autre :

- `activities()` : une ligne par activite, unites converties ;
- `daily()` : une ligne par jour calendaire, sans trou, qui aligne sante
  (FC repos, stress, Body Battery, sommeil, readiness, VO2max) et charge.

et un `profile()` qui sert a initialiser les reperes de l'athlete.

Pieges des JSON de l'export, verifies sur un export reel :

| champ | unite reelle |
|---|---|
| `distance`, `elevationGain/Loss`, `avgStrideLength` | centimetres |
| `duration`, `movingDuration`, `hrTimeInZone_*` | millisecondes |
| `avgSpeed`, `avgGradeAdjustedSpeed`, `lactateThresholdSpeed` | m/s divises par 10 |
| `startTimeGmt` / `startTimeLocal` | ms epoch ; leur ecart donne le fuseau du jour |
| `MetricsAcuteTrainingLoad.calendarDate` | ms epoch a minuit UTC, pas une date ISO |
| `allDayStress.averageStressLevel` | -1 ou -2 quand la montre n'a pas assez mesure |
| `sleepData.calendarDate` | date du REVEIL : la nuit precedant ce jour |
| `sleepData` sans phases ou `OFF_WRIST` | pas une mesure : fenetre par defaut (10 h) ou montre non portee |
| `hrvWeeklyAverage` = 511 | valeur sentinelle "pas de donnee", pas une HRV |
| `currentDayRestingHeartRate` | artefacts (105 quand `restingHeartRate` donne 71) : preferer ce dernier |

Plusieurs fichiers de metriques ont plusieurs lignes par jour (readiness
recalculee apres chaque seance, charge mise a jour en continu). On garde la
readiness du reveil et la derniere valeur de charge du jour.

Aucune coordonnee GPS ni nom de lieu n'est repris dans les tables.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

RUN_TYPES = ("running", "trail_running", "treadmill_running", "track_running",
             "ultra_run", "virtual_run", "obstacle_run", "indoor_running")


def _ms_to_local(gmt_ms: float, local_ms: float | None) -> datetime:
    # startTimeLocal est un faux epoch : l'heure locale ecrite comme si c'etait UTC
    ms = local_ms if local_ms is not None else gmt_ms
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).replace(tzinfo=None)


def _div(v, k):
    return v / k if v is not None else None


def _pace(speed_ms: float | None) -> float | None:
    return 1000 / speed_ms if speed_ms else None


class GarminExport:
    def __init__(self, root: str | Path):
        root = Path(root).expanduser()
        self.root = root if root.name == "DI_CONNECT" else root / "DI_CONNECT"
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"{self.root} introuvable : passer le dossier de l'export dezippe"
            )

    # -- Acces fichiers ---------------------------------------------------

    def _files(self, pattern: str) -> list[Path]:
        return sorted(self.root.glob(pattern))

    def _records(self, pattern: str) -> list[dict]:
        """Concatene les listes JSON de tous les fichiers qui matchent."""
        out: list[dict] = []
        for f in self._files(pattern):
            data = json.loads(f.read_text(encoding="utf-8"))
            out.extend(data if isinstance(data, list) else [data])
        return out

    # -- Activites --------------------------------------------------------

    def raw_activities(self) -> list[dict]:
        out = []
        for block in self._records("DI-Connect-Fitness/*_summarizedActivities.json"):
            out.extend(block.get("summarizedActivitiesExport", []))
        return out

    def activities(self) -> pd.DataFrame:
        rows = []
        for a in self.raw_activities():
            gmt, local = a.get("startTimeGmt"), a.get("startTimeLocal")
            if gmt is None:
                continue
            start = _ms_to_local(gmt, local)
            speed = _div(a.get("avgSpeed"), 0.1)
            gap = _div(a.get("avgGradeAdjustedSpeed"), 0.1)
            row = {
                "activity_id": a.get("activityId"),
                "start_local": start,
                "date": start.date(),
                "utc_offset_h": round((local - gmt) / 3.6e6, 2) if local is not None else None,
                "sport": a.get("activityType"),
                "is_run": a.get("activityType") in RUN_TYPES,
                "distance_km": _div(a.get("distance"), 1e5),
                "duration_min": _div(a.get("duration"), 6e4),
                "moving_min": _div(a.get("movingDuration"), 6e4),
                "elev_gain_m": _div(a.get("elevationGain"), 100),
                "elev_loss_m": _div(a.get("elevationLoss"), 100),
                "pace_s_per_km": _pace(speed),
                "gap_pace_s_per_km": _pace(gap),
                "avg_hr": a.get("avgHr"),
                "max_hr": a.get("maxHr"),
                "cadence_spm": a.get("avgDoubleCadence"),
                "stride_m": _div(a.get("avgStrideLength"), 100),
                "ground_contact_ms": a.get("avgGroundContactTime"),
                "vert_osc_cm": a.get("avgVerticalOscillation"),
                "vert_ratio_pct": a.get("avgVerticalRatio"),
                "avg_power": a.get("avgPower"),
                "norm_power": a.get("normPower"),
                "training_load": a.get("activityTrainingLoad"),
                "aerobic_te": a.get("aerobicTrainingEffect"),
                "anaerobic_te": a.get("anaerobicTrainingEffect"),
                "te_label": a.get("trainingEffectLabel"),
                "vo2max": a.get("vO2MaxValue"),
                "rpe": a.get("workoutRpe"),
                "feel": a.get("workoutFeel"),
                "body_battery_delta": a.get("differenceBodyBattery"),
                "workout_id": a.get("workoutId"),
            }
            # index Garmin : 0 = sous la zone 1, puis zones 1 a 5 (voire 6)
            for i in range(7):
                row[f"hr_zone{i}_min"] = _div(a.get(f"hrTimeInZone_{i}"), 6e4)
            rows.append(row)
        df = pd.DataFrame(rows)
        return df.sort_values("start_local").reset_index(drop=True) if not df.empty else df

    # -- Sante quotidienne ------------------------------------------------

    def _uds(self) -> pd.DataFrame:
        rows = []
        for u in self._records("DI-Connect-Aggregator/UDSFile_*.json"):
            if not u.get("calendarDate"):
                continue
            stress = next((s for s in (u.get("allDayStress") or {}).get("aggregatorList", [])
                           if s.get("type") == "TOTAL"), {})
            bb = u.get("bodyBattery") or {}
            bb_stats = {s.get("bodyBatteryStatType"): s.get("statsValue")
                        for s in bb.get("bodyBatteryStatList", [])}
            avg_stress = stress.get("averageStressLevel")
            rows.append({
                "date": date.fromisoformat(u["calendarDate"]),
                "resting_hr": u.get("restingHeartRate") or u.get("currentDayRestingHeartRate"),
                "min_hr": u.get("minHeartRate"),
                "steps": u.get("totalSteps"),
                "moderate_min": u.get("moderateIntensityMinutes"),
                "vigorous_min": u.get("vigorousIntensityMinutes"),
                "stress_avg": avg_stress if avg_stress is not None and avg_stress >= 0 else None,
                "stress_max": stress.get("maxStressLevel"),
                "stress_high_min": _div(stress.get("highDuration"), 60),
                "body_battery_high": bb_stats.get("HIGHEST"),
                "body_battery_low": bb_stats.get("LOWEST"),
                "body_battery_charged": bb.get("chargedValue"),
                "body_battery_drained": bb.get("drainedValue"),
                "respiration_awake": (u.get("respiration") or {}).get("avgWakingRespirationValue"),
            })
        return pd.DataFrame(rows)

    def _sleep(self) -> pd.DataFrame:
        rows = []
        for s in self._records("DI-Connect-Wellness/*_sleepData.json"):
            # sans phases mesurees, la nuit n'est qu'une fenetre par defaut ou
            # une montre non portee : on l'ignore plutot que d'inventer 10 h
            asleep = sum(s.get(k) or 0 for k in
                         ("deepSleepSeconds", "lightSleepSeconds", "remSleepSeconds"))
            if "calendarDate" not in s or not asleep:
                continue
            scores = s.get("sleepScores") or {}
            rows.append({
                "date": date.fromisoformat(s["calendarDate"]),
                "sleep_h": _div(asleep, 3600),
                "deep_h": _div(s.get("deepSleepSeconds"), 3600),
                "rem_h": _div(s.get("remSleepSeconds"), 3600),
                "awake_min": _div(s.get("awakeSleepSeconds"), 60),
                "sleep_score": scores.get("overallScore"),
                "sleep_stress": s.get("avgSleepStress"),
                "respiration_sleep": s.get("averageRespiration"),
            })
        return pd.DataFrame(rows)

    def _metrics(self, pattern: str, columns: list[str]) -> pd.DataFrame:
        """Records d'un fichier de metriques, avec toutes les `columns` garanties.

        Une cle absente de tous les records (export ancien ou partiel) donne
        une colonne vide au lieu de faire planter l'import.
        """
        recs = [r for r in self._records(pattern) if r.get("calendarDate") is not None]
        return pd.DataFrame(recs).reindex(columns=["calendarDate", *columns])

    def _readiness(self) -> pd.DataFrame:
        df = self._metrics("DI-Connect-Metrics/TrainingReadinessDTO_*.json",
                           ["timestamp", "inputContext", "score", "hrvWeeklyAverage", "recoveryTime"])
        if df.empty:
            return pd.DataFrame()
        # la valeur du reveil d'abord, sinon la premiere du jour
        df["_rank"] = (df["inputContext"] != "AFTER_WAKEUP_RESET").astype(int)
        df = df.sort_values(["calendarDate", "_rank", "timestamp"]).drop_duplicates("calendarDate")
        return pd.DataFrame({
            "date": df["calendarDate"].map(date.fromisoformat),
            "readiness": df["score"],
            "hrv_weekly": df["hrvWeeklyAverage"].where(df["hrvWeeklyAverage"] != 511),
            "recovery_time_min": df["recoveryTime"],
        })

    def _load(self) -> pd.DataFrame:
        df = self._metrics("DI-Connect-Metrics/MetricsAcuteTrainingLoad_*.json",
                           ["timestamp", "dailyTrainingLoadAcute", "dailyTrainingLoadChronic",
                            "dailyAcuteChronicWorkloadRatio"])
        if df.empty:
            return pd.DataFrame()
        df = df.sort_values("timestamp").drop_duplicates("calendarDate", keep="last")
        return pd.DataFrame({
            "date": df["calendarDate"].map(
                lambda ms: datetime.fromtimestamp(ms / 1000, tz=timezone.utc).date()),
            "acute_load": df["dailyTrainingLoadAcute"],
            "chronic_load": df["dailyTrainingLoadChronic"],
            "acwr": df["dailyAcuteChronicWorkloadRatio"],
        })

    def _status(self) -> pd.DataFrame:
        df = self._metrics("DI-Connect-Metrics/TrainingHistory_*.json",
                           ["timestamp", "trainingStatus"])
        if df.empty:
            return pd.DataFrame()
        df = df.sort_values("timestamp").drop_duplicates("calendarDate", keep="last")
        return pd.DataFrame({
            "date": df["calendarDate"].map(date.fromisoformat),
            "training_status": df["trainingStatus"],
        })

    def _vo2max(self) -> pd.DataFrame:
        df = self._metrics("DI-Connect-Metrics/MetricsMaxMetData_*.json",
                           ["updateTimestamp", "sport", "vo2MaxValue"])
        df = df[(df["sport"] == "RUNNING") & df["vo2MaxValue"].notna()]
        if df.empty:
            return pd.DataFrame()
        df = df.sort_values("updateTimestamp").drop_duplicates("calendarDate", keep="last")
        return pd.DataFrame({
            "date": df["calendarDate"].map(date.fromisoformat),
            "vo2max": df["vo2MaxValue"],
        })

    def _hrv_night(self) -> pd.DataFrame:
        rows = []
        for h in self._records("DI-Connect-Wellness/*_healthStatusData.json"):
            hrv = next((m.get("value") for m in h.get("metrics", []) if m.get("type") == "HRV"), None)
            if hrv:
                rows.append({"date": date.fromisoformat(h["calendarDate"]), "hrv_night": hrv})
        return pd.DataFrame(rows)

    def daily(self, acts: pd.DataFrame | None = None) -> pd.DataFrame:
        """Une ligne par jour, du premier au dernier jour couvert, sans trou.

        Les colonnes d'activite portent sur le jour local de la seance ; le
        sommeil porte sur la nuit qui PRECEDE la date. Un jour sans activite
        vaut 0 km, un jour sans mesure de sante reste vide (NaN). Si une source
        a plusieurs lignes pour un meme jour (sieste, fichiers qui se
        chevauchent), la derniere lue l'emporte.
        """
        acts = self.activities() if acts is None else acts
        parts = [p for p in (self._uds(), self._sleep(), self._readiness(), self._load(),
                             self._status(), self._vo2max(), self._hrv_night()) if not p.empty]
        if not acts.empty:
            runs = acts[acts["is_run"]]
            parts.append(pd.DataFrame({
                "run_km": runs.groupby("date")["distance_km"].sum(),
                "run_elev_m": runs.groupby("date")["elev_gain_m"].sum(),
                "run_min": runs.groupby("date")["moving_min"].sum(),
                "activity_load": acts.groupby("date")["training_load"].sum(min_count=1),
                "n_activities": acts.groupby("date").size(),
            }).reset_index())
        if not parts:
            return pd.DataFrame()

        dates = [d for p in parts for d in p["date"]]
        df = pd.DataFrame(index=pd.date_range(min(dates), max(dates), freq="D").date)
        df.index.name = "date"
        for p in parts:
            df = df.join(p.drop_duplicates("date", keep="last").set_index("date"), how="left")
        for col in ("run_km", "run_elev_m", "run_min", "n_activities"):
            if col in df:
                df[col] = df[col].fillna(0)
        return df

    # -- Profil -----------------------------------------------------------

    def profile(self, acts: pd.DataFrame | None = None, daily: pd.DataFrame | None = None) -> dict:
        """Reperes pour initialiser l'athlete (voir config.Athlete)."""
        acts = self.activities() if acts is None else acts
        prof: dict = {}

        zones = self._records("DI-Connect-Wellness/*_heartRateZones.json")
        z = next((z for z in zones if z.get("sport") == "RUNNING"), zones[0] if zones else {})
        prof["garmin_hr_max"] = z.get("maxHeartRateUsed")
        prof["garmin_hr_rest"] = z.get("restingHeartRateUsed")
        prof["garmin_zone_method"] = z.get("trainingMethod")

        bio = next(iter(self._records("DI-Connect-Wellness/*_bioMetrics_latest.json")), {})
        prof["garmin_lthr"] = bio.get("lactateThresholdHeartRate") or z.get("lactateThresholdHeartRateUsed")
        lt_speed = _div(bio.get("lactateThresholdSpeed"), 0.1)
        prof["garmin_lt_pace_s_per_km"] = round(_pace(lt_speed)) if lt_speed else None

        runs = acts[acts["is_run"]] if not acts.empty else acts
        if not runs.empty and runs["max_hr"].notna().any():
            # la valeur max brute peut etre un artefact capteur : on donne aussi le 2e plus haut
            top = runs["max_hr"].dropna().sort_values(ascending=False)
            prof["observed_hr_max"] = int(top.iloc[0])
            prof["observed_hr_max_2nd"] = int(top.iloc[1]) if len(top) > 1 else None
            last = runs["start_local"].max()
            recent = runs[runs["start_local"] >= last - timedelta(days=90)]
            prof["runs_per_week_90d"] = round(len(recent) / (90 / 7), 1)
            prof["km_per_week_90d"] = round(recent["distance_km"].sum() / (90 / 7), 1)
            prof["longest_run_km_90d"] = round(recent["distance_km"].max(), 1)

        daily = self.daily(acts) if daily is None else daily
        if "resting_hr" in daily and daily["resting_hr"].notna().any():
            prof["resting_hr_median_90d"] = float(daily["resting_hr"].dropna().tail(90).median())
        if "vo2max" in daily and daily["vo2max"].notna().any():
            prof["vo2max_latest"] = float(daily["vo2max"].dropna().iloc[-1])
        return prof
