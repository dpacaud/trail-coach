#!/usr/bin/env python3
"""Initialise les donnees d'un athlete depuis son export RGPD Garmin Connect.

    python scripts/garmin_import.py ~/Downloads/<export_dezippe> --athlete prenom

Ecrit dans <data_dir>/<athlete>/ : activities.csv, daily.csv, profile.json.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trailcoach import data_dir, load_dotenv                       # noqa: E402
from trailcoach.garmin_export import GarminExport                  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export", help="dossier de l'export Garmin dezippe (contient DI_CONNECT)")
    ap.add_argument("--athlete", default="garmin", help="sous-dossier de sortie")
    ap.add_argument("--out", type=Path, help="dossier de sortie (defaut : data_dir/athlete)")
    args = ap.parse_args()

    load_dotenv()
    exp = GarminExport(args.export)
    out = args.out or data_dir() / args.athlete
    out.mkdir(parents=True, exist_ok=True)

    acts = exp.activities()
    daily = exp.daily(acts)
    prof = exp.profile(acts, daily)

    acts.to_csv(out / "activities.csv", index=False)
    daily.to_csv(out / "daily.csv")
    (out / "profile.json").write_text(json.dumps(prof, indent=2, default=str))

    print(f"Ecrit dans {out}/\n")
    if not acts.empty:
        print(f"{len(acts)} activites du {acts['date'].min()} au {acts['date'].max()}")
        for sport, n in acts["sport"].value_counts().head(8).items():
            print(f"  {sport:22} {n}")
        runs = acts[acts["is_run"]]
        print("\nCourse par annee :")
        by_year = runs.groupby(runs["start_local"].dt.year).agg(
            sorties=("activity_id", "size"), km=("distance_km", "sum"),
            dplus=("elev_gain_m", "sum"), heures=("moving_min", lambda m: m.sum() / 60))
        for year, r in by_year.iterrows():
            print(f"  {year}  {r.sorties:4.0f} sorties  {r.km:7.0f} km  "
                  f"{r.dplus:7.0f} m D+  {r.heures:5.0f} h")

    if not daily.empty:
        print(f"\nTable quotidienne : {len(daily)} jours. Couverture par colonne :")
        for col in daily.columns:
            s = daily[col].dropna()
            if col in ("run_km", "run_elev_m", "run_min", "n_activities"):
                continue
            first = s.index.min() if len(s) else "-"
            print(f"  {col:22} {len(s):5} jours   depuis {first}")

    print("\nProfil :")
    for k, v in prof.items():
        print(f"  {k:26} {v}")
    if prof.get("observed_hr_max"):
        print("\nPour .env (a relire : la FC max brute peut etre un artefact capteur) :")
        print(f"  ATHLETE_HR_MAX={prof['observed_hr_max']}")
        if prof.get("garmin_lthr"):
            print(f"  ATHLETE_LTHR={prof['garmin_lthr']}")
        if prof.get("resting_hr_median_90d"):
            print(f"  ATHLETE_HR_REST={prof['resting_hr_median_90d']:.0f}")


if __name__ == "__main__":
    main()
