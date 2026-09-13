#!/usr/bin/env python3
"""Rapport charge d'entrainement x sante, a partir des tables de garmin_import.

    python scripts/health_report.py --athlete prenom
    python scripts/health_report.py --dir data/prenom --hr-max 185

Ecrit correlations.csv et session_context.csv a cote des tables.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trailcoach import data_dir, load_dotenv                       # noqa: E402
from trailcoach import health                                      # noqa: E402

LABELS = {
    "run_km": "km courus",
    "run_elev_m": "D+ couru",
    "garmin_load": "charge Garmin",
    "garmin_load_3d": "charge Garmin 3 j",
    "resting_hr_dev": "FC repos (ecart)",
    "stress_avg_dev": "stress moyen (ecart)",
    "body_battery_high_dev": "Body Battery max (ecart)",
    "respiration_awake_dev": "respiration eveil (ecart)",
    "readiness": "Training Readiness",
    "hrv_weekly": "HRV hebdo",
    "sleep_h": "sommeil (h)",
    "sleep_score": "score sommeil",
    "stress_prev_day_dev": "stress de la veille (ecart)",
    "run_km_prev7d": "km des 7 jours avant",
    "garmin_load_prev2d": "charge des 2 jours avant",
}
STRONG = 0.2
Q_MAX = 0.05


def label(col):
    return LABELS.get(col, col)


def mark(row):
    return "*" if row["q"] < Q_MAX and abs(row["rho"]) >= STRONG else " "


def section(title):
    print(f"\n{title}\n{'-' * len(title)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--athlete", default="garmin", help="sous-dossier de data_dir")
    ap.add_argument("--dir", type=Path, help="dossier contenant daily.csv et activities.csv")
    ap.add_argument("--hr-max", type=float, help="FC max (defaut : profile.json)")
    ap.add_argument("--months", type=int, default=18, help="mois affiches dans la tendance")
    args = ap.parse_args()

    load_dotenv()
    src = args.dir or data_dir() / args.athlete
    daily = pd.read_csv(src / "daily.csv", index_col="date", parse_dates=["date"])
    daily.index = daily.index.date
    acts = pd.read_csv(src / "activities.csv", parse_dates=["start_local"])
    acts["date"] = acts["start_local"].dt.date
    prof_path = src / "profile.json"
    prof = json.loads(prof_path.read_text()) if prof_path.exists() else {}
    hr_max = args.hr_max or prof.get("garmin_hr_max") or prof.get("observed_hr_max")

    print(f"Donnees : {src}   {daily.index.min()} -> {daily.index.max()} ({len(daily)} jours)")

    # -- 1. Charge -> marqueurs des jours suivants
    corr = health.lagged_correlations(daily)
    section("1. Charge du jour J -> marqueurs a J+0 .. J+3  (rho de Spearman)")
    print("   * = |rho| >= 0,2 et q < 0,05 apres correction des tests multiples")
    print("   (G) = indicateur Garmin qui integre deja la charge")
    print("   J+1 a J+3 : a charge egale le jour J+lag (sinon on mesure l'alternance sortie / repos)\n")
    if corr.empty:
        print("   pas assez de jours communs")
    else:
        for marker, grp in corr.groupby("marker", sort=False):
            g = " (G)" if grp["garmin_derived"].iloc[0] else ""
            n_eff = grp["n_eff"].median()
            print(f"   {label(marker) + g:32} n~{grp['n'].median():.0f} jours, n eff~{n_eff:.0f}")
            for load, lg in grp.groupby("load", sort=False):
                cells = {r.lag: f"{r.rho:+.2f}{mark(r._asdict())}" for r in lg.itertuples()}
                line = "  ".join(f"J+{lag} {cells.get(lag, '   -  ')}" for lag in (0, 1, 2, 3))
                print(f"      {label(load):20} {line}")
        corr.to_csv(src / "correlations.csv", index=False)

    # -- 2. ACWR
    bands = health.acwr_bands(daily)
    section("2. Ratio charge aigue / chronique du jour -> lendemain (moyennes)")
    if bands.empty:
        print("   pas d'ACWR dans l'export")
    else:
        cols = [c for c in bands.columns if c.endswith("_mean")]
        print("   " + f"{'ACWR':10} {'jours':>6} " + " ".join(f"{label(c[:-5]):>22}" for c in cols))
        for band, r in bands.iterrows():
            vals = " ".join(f"{'-':>22}" if math.isnan(r[c]) else
                            f"{r[c]:>+22.1f}" if "_dev" in c else f"{r[c]:>22.1f}" for c in cols)
            print(f"   {band:10} {r['days']:>6.0f} {vals}")

    # -- 3. Etat avant la seance -> efficacite
    section("3. Etat avant la sortie -> efficacite aerobie (m/battement, ecart a 6 semaines)")
    if not hr_max:
        print("   FC max inconnue : passer --hr-max")
        runs = pd.DataFrame()
    else:
        runs = health.run_efficiency(acts, hr_max)
        ctx = health.session_context(runs, daily) if not runs.empty else pd.DataFrame()
        print(f"   {len(runs)} sorties en endurance (FC moy 65-85 % de {hr_max:.0f}), hors tapis\n")
        if ctx.empty:
            print("   pas assez de sorties avec contexte")
        else:
            for r in ctx.sort_values("rho", key=abs, ascending=False).itertuples():
                g = " (G)" if r.garmin_derived else ""
                print(f"   {label(r.context) + g:34} rho {r.rho:+.2f}{mark(r._asdict())}  n={r.n}")
            ctx.to_csv(src / "session_context.csv", index=False)

    # -- 4. Tendance
    section(f"4. Tendance mensuelle ({args.months} derniers mois)")
    trend = health.monthly_trend(daily, runs if not runs.empty else None).tail(args.months)
    print(trend.round(2).to_string(na_rep="-"))

    # -- 5. Lecture
    section("5. Ce qui ressort")
    found = False
    if not corr.empty:
        best = (corr[(corr["q"] < Q_MAX) & (corr["rho"].abs() >= STRONG)]
                .sort_values("rho", key=abs, ascending=False)
                .drop_duplicates(["load", "marker"]))
        for r in best.head(8).itertuples():
            found = True
            sens = "hausse" if r.rho > 0 else "baisse"
            g = "  [en partie mecanique, calcule par Garmin]" if r.garmin_derived else ""
            print(f"   - {label(r.load)} a J -> {sens} de {label(r.marker)} a J+{r.lag} "
                  f"(rho {r.rho:+.2f}, n eff {r.n_eff:.0f}){g}")
    if not found:
        print("   aucune association nette avec la charge")
    print("\n   Associations sur un seul athlete, pas des causes : chaleur, maladie et vie")
    print("   perso bougent les memes marqueurs. A lire comme des pistes a surveiller.")


if __name__ == "__main__":
    main()
