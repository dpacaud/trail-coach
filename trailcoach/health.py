"""Charge d'entrainement et marqueurs de sante, sur les tables de garmin_export.

Ce sont des associations sur UN athlete, pas des causes : la chaleur, une
maladie ou une semaine chargee au travail bougent les memes marqueurs. Quatre
precautions sont appliquees partout :

1. ECART A LA REFERENCE. La FC repos, le stress ou le Body Battery derivent sur
   des mois (forme, saison). On correle l'ecart a la mediane des 28 jours
   PRECEDENTS, sinon la tendance de fond cree des correlations fantomes.
2. AUTOCORRELATION. Deux jours consecutifs se ressemblent : 300 jours ne font
   pas 300 observations independantes. Le n effectif suit Bretherton et al.
   (1999), n * (1 - r1x.r1y) / (1 + r1x.r1y), et sert au calcul de p.
3. TESTS MULTIPLES. Sur des dizaines de couples charge x marqueur x decalage,
   quelques-uns sortent au hasard. Les p sont corriges par Benjamini-Hochberg (q).
4. CIRCULARITE. Readiness, Body Battery et HRV hebdo sont calcules par Garmin
   en integrant deja la charge : leur lien avec elle est en partie mecanique.

Et un piege propre aux decalages : si l'athlete court un jour sur deux, le
lendemain d'une grosse sortie est souvent un jour de repos. Une correlation
"charge J -> stress J+1" negative peut alors seulement dire qu'un jour de repos
est moins stressant. Pour J+1 a J+3, on controle donc la charge du jour J+lag
(correlation partielle sur les rangs).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

GARMIN_DERIVED = {"readiness", "body_battery_high_dev", "hrv_weekly"}
DEVIATION_COLS = ("resting_hr", "stress_avg", "body_battery_high", "respiration_awake")
RAW_COLS = ("readiness", "hrv_weekly", "sleep_h", "sleep_score")
STEADY_HR = (0.65, 0.85)


@dataclass(frozen=True)
class Corr:
    rho: float
    n: int
    n_eff: float
    p: float


def baseline_deviation(s: pd.Series, window: int = 28, min_periods: int = 10) -> pd.Series:
    """Ecart a la mediane des `window` jours precedents, le jour meme exclu."""
    return s - s.shift(1).rolling(window, min_periods=min_periods).median()


def spearman(x: pd.Series, y: pd.Series, autocorrelated: bool = True,
             control: pd.Series | None = None) -> Corr:
    """Rho de Spearman, avec p approche (Fisher z, erreur type 1.06 / sqrt(n - 3)).

    `autocorrelated` : series temporelles regulieres, dont on reduit le n.
    `control` : si fourni, correlation partielle de x et y a `control` egal.
    """
    frame = pd.concat([x, y] + ([control] if control is not None else []), axis=1).dropna()
    n = len(frame)
    a, b = frame.iloc[:, 0], frame.iloc[:, 1]
    if n < 5 or a.nunique() < 3 or b.nunique() < 3:
        return Corr(math.nan, n, math.nan, math.nan)
    ra, rb = a.rank(), b.rank()
    rho = ra.corr(rb)
    k = 0
    if control is not None:
        rc = frame.iloc[:, 2].rank()
        rac, rbc = ra.corr(rc), rb.corr(rc)
        denom = math.sqrt(max((1 - rac ** 2) * (1 - rbc ** 2), 0.0))
        if not denom or math.isnan(denom):
            return Corr(math.nan, n, math.nan, math.nan)
        rho, k = (rho - rac * rbc) / denom, 1

    n_eff = float(n)
    if autocorrelated:
        r = (x.autocorr(1) or 0.0) * (y.autocorr(1) or 0.0)
        if r > 0:
            n_eff = n * (1 - r) / (1 + r)
    if n_eff - k <= 3:
        return Corr(rho, n, n_eff, math.nan)
    z = math.atanh(max(min(rho, 0.999999), -0.999999)) * math.sqrt(n_eff - k - 3) / 1.06
    return Corr(rho, n, n_eff, math.erfc(abs(z) / math.sqrt(2)))


def bh_adjust(p: pd.Series) -> pd.Series:
    """q-values de Benjamini-Hochberg (les NaN restent NaN)."""
    valid = p.dropna().sort_values()
    m = len(valid)
    if not m:
        return p.copy()
    q = valid * m / pd.Series(range(1, m + 1), index=valid.index)
    q = q[::-1].cummin()[::-1].clip(upper=1.0)
    return q.reindex(p.index)


# -- Variables ------------------------------------------------------------

def load_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Charge du jour. `garmin_load` vaut 0 les jours sans activite, mais reste
    vide avant la premiere mesure (montre sans cette metrique)."""
    out = pd.DataFrame(index=daily.index)
    for col in ("run_km", "run_elev_m"):
        if col in daily:
            out[col] = daily[col]
    if "activity_load" in daily and daily["activity_load"].notna().any():
        first = daily["activity_load"].first_valid_index()
        load = daily["activity_load"].fillna(0.0)
        load[daily.index < first] = math.nan
        out["garmin_load"] = load
        out["garmin_load_3d"] = load.rolling(3).sum()
    return out


def health_features(daily: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=daily.index)
    for col in DEVIATION_COLS:
        if col in daily:
            out[f"{col}_dev"] = baseline_deviation(daily[col])
    for col in RAW_COLS:
        if col in daily:
            out[col] = daily[col]
    return out


# -- Analyses -------------------------------------------------------------

def lagged_correlations(daily: pd.DataFrame, lags=(0, 1, 2, 3), min_n: int = 30) -> pd.DataFrame:
    """Charge du jour J contre marqueur a J + lag, pour chaque couple.

    La table `daily` doit etre continue (une ligne par jour) : le decalage se
    fait en lignes. A partir de J+1, la charge du jour J+lag est controlee.
    """
    loads, health = load_features(daily), health_features(daily)
    rows = []
    for load in loads:
        # controler la charge du SEUL jour J+lag : une somme glissante decalee
        # recouvrirait le jour J et effacerait l'effet qu'on cherche
        day_load = loads[load.removesuffix("_3d")]
        for marker in health:
            for lag in lags:
                control = day_load.shift(-lag) if lag else None
                c = spearman(loads[load], health[marker].shift(-lag), control=control)
                if c.n >= min_n and not math.isnan(c.rho):
                    rows.append({"load": load, "marker": marker, "lag": lag,
                                 "rho": c.rho, "n": c.n, "n_eff": c.n_eff, "p": c.p,
                                 "garmin_derived": marker in GARMIN_DERIVED})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["q"] = bh_adjust(df["p"])
    return df


def run_efficiency(acts: pd.DataFrame, hr_max: float, min_minutes: float = 20) -> pd.DataFrame:
    """Efficacite aerobie des sorties en endurance : metres par battement.

    Vitesse ajustee a la pente (GAP) si disponible, divisee par la FC moyenne.
    Seules les sorties dont la FC moyenne tombe entre 65 et 85 % de la FC max
    comptent : une seance de fractionne n'est pas comparable a un footing.
    Le tapis est exclu (vitesse estimee par l'accelerometre).
    `eff_dev` est l'ecart relatif a la mediane des 42 jours precedents.
    """
    r = acts[acts["is_run"] & (acts["sport"] != "treadmill_running")
             & (acts["moving_min"] >= min_minutes) & acts["avg_hr"].notna()].copy()
    frac = r["avg_hr"] / hr_max
    r = r[(frac >= STEADY_HR[0]) & (frac <= STEADY_HR[1])]
    pace = r["gap_pace_s_per_km"].fillna(r["pace_s_per_km"])
    r["efficiency"] = 60000 / pace / r["avg_hr"]
    r = r.dropna(subset=["efficiency"]).sort_values("start_local")
    ref = (r.set_index("start_local")["efficiency"]
           .rolling("42D", closed="left", min_periods=5).median())
    r["eff_dev"] = r["efficiency"].to_numpy() / ref.to_numpy() - 1
    return r.reset_index(drop=True)


def session_context(runs: pd.DataFrame, daily: pd.DataFrame, min_n: int = 20) -> pd.DataFrame:
    """Etat du jour (ou des jours precedents) contre efficacite de la sortie."""
    health, loads = health_features(daily), load_features(daily)
    ctx = pd.DataFrame(index=daily.index)
    for col in ("resting_hr_dev", "readiness", "sleep_h", "sleep_score", "body_battery_high_dev"):
        if col in health:
            ctx[col] = health[col]
    if "stress_avg_dev" in health:
        ctx["stress_prev_day_dev"] = health["stress_avg_dev"].shift(1)
    ctx["run_km_prev7d"] = loads["run_km"].shift(1).rolling(7).sum()
    if "garmin_load" in loads:
        ctx["garmin_load_prev2d"] = loads["garmin_load"].shift(1).rolling(2).sum()

    joined = runs[["date", "eff_dev"]].join(ctx, on="date")
    rows = []
    for col in ctx:
        c = spearman(joined[col], joined["eff_dev"], autocorrelated=False)
        if c.n >= min_n and not math.isnan(c.rho):
            rows.append({"context": col, "rho": c.rho, "n": c.n, "p": c.p,
                         "garmin_derived": col in GARMIN_DERIVED})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["q"] = bh_adjust(df["p"])
    return df


def acwr_bands(daily: pd.DataFrame) -> pd.DataFrame:
    """Marqueurs du lendemain selon le ratio charge aigue / chronique du jour."""
    if "acwr" not in daily or daily["acwr"].notna().sum() == 0:
        return pd.DataFrame()
    health = health_features(daily)
    bands = pd.cut(daily["acwr"], [0, 0.8, 1.3, 1.5, math.inf], right=False,
                   labels=["< 0,8", "0,8 - 1,3", "1,3 - 1,5", ">= 1,5"])
    cols = [c for c in ("resting_hr_dev", "stress_avg_dev", "readiness") if c in health]
    nxt = health[cols].shift(-1)
    out = nxt.groupby(bands, observed=False).agg(["mean", "count"])
    out.columns = [f"{c}_{s}" for c, s in out.columns]
    out.insert(0, "days", bands.value_counts().reindex(out.index))
    return out


def monthly_trend(daily: pd.DataFrame, runs: pd.DataFrame | None = None) -> pd.DataFrame:
    idx = pd.to_datetime(daily.index)
    month = idx.to_period("M")
    days = pd.Series(1, index=daily.index).groupby(month).sum()
    agg = {"km_per_week": daily["run_km"].groupby(month).sum() / days * 7,
           "elev_per_week": daily["run_elev_m"].groupby(month).sum() / days * 7}
    for col, fn in (("resting_hr", "median"), ("readiness", "median"), ("vo2max", "max")):
        if col in daily:
            agg[col] = daily[col].groupby(month).agg(fn)
    out = pd.DataFrame(agg)
    if runs is not None and not runs.empty:
        out["efficiency"] = runs.groupby(pd.to_datetime(runs["start_local"]).dt.to_period("M"))[
            "efficiency"].median()
    return out
