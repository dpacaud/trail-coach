#!/usr/bin/env python3
"""Analyse une seance a partir de son fichier FIT.

    python scripts/analyze_session.py seance.fit
    python scripts/analyze_session.py export_garmin.zip --efforts
"""
import argparse
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trailcoach import Athlete, load_dotenv                        # noqa: E402
from trailcoach import fit as fitmod                               # noqa: E402
from trailcoach.metrics import (                                   # noqa: E402
    detect_efforts, effort_cadence_threshold, fmt_pace, hr_drift, recoveries, splits,
    stride_response, time_below, zone_distribution, zone_edges,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="fichier .fit ou .zip Garmin")
    ap.add_argument("--efforts", action="store_true", help="detecter les repetitions")
    ap.add_argument("--cadence-threshold", type=float,
                    help="seuil absolu de cadence (pas/min) pour --efforts, ex. en cote raide")
    ap.add_argument("--margin", type=float, default=8.0,
                    help="ecart au-dessus de la mediane de cadence en course (defaut 8)")
    ap.add_argument("--splits", action="store_true", help="detail par kilometre")
    ap.add_argument("--target-hr", type=int, help="consigne de FC moyenne a verifier")
    args = ap.parse_args()

    load_dotenv()
    ath = Athlete.from_env()
    sess = fitmod.load(args.path, tz=ath.tz)
    S, m = sess.samples, sess.meta

    print(f"Depart {sess.start_local:%Y-%m-%d %H:%M} (local)   "
          f"temp moy {m.get('avg_temperature')}C max {m.get('max_temperature')}C")
    asc = m.get("total_ascent") or 0
    print(f"{sess.distance_km:.2f} km / {asc} m D+ ({asc / max(sess.distance_km, .01):.1f} m/km)   "
          f"{int(sess.duration_s)//60}min{int(sess.duration_s)%60:02d}   "
          f"{fmt_pace(sess.pace_s_per_km)}")
    print(f"FC moy {m.get('avg_heart_rate')} / max {m.get('max_heart_rate')}")

    edges = zone_edges(ath.hr_max)
    print(f"\nZones (FC max {ath.hr_max}) : "
          f"<{edges[0]} | {edges[0]}-{edges[1]-1} | {edges[1]}-{edges[2]-1} | "
          f"{edges[2]}-{edges[3]-1} | >{edges[3]-1}")
    for name, pct in zone_distribution(S, ath.hr_max).items():
        print(f"  {name:14} {pct:5.1f} %")

    if args.target_hr:
        avg = m.get("avg_heart_rate")
        verdict = "TENUE" if avg and avg <= args.target_hr else "depassee"
        print(f"\nConsigne FC moyenne <= {args.target_hr} : {avg} -> {verdict}")
        print(f"  temps sous {args.target_hr} : {time_below(S, args.target_hr):.0f} %")
    if ath.lthr:
        print(f"  temps au-dessus du seuil ({ath.lthr}) : "
              f"{100 - time_below(S, ath.lthr):.0f} %")

    d = hr_drift(S)
    if d:
        print(f"\nDerive FC : {d.first_half_hr:.0f} -> {d.second_half_hr:.0f} "
              f"= {d.pct:+.1f} %   (allure {fmt_pace(d.first_half_pace)} -> "
              f"{fmt_pace(d.second_half_pace)})")

    run, walk = sess.running(), sess.walking()
    if run:
        print(f"\nCadence en courant : {st.mean([s.cadence for s in run]):.1f} pas/min")
    if S:
        print(f"Marche : {len(walk)}s ({100*len(walk)/len(S):.0f} %)")

    sr = stride_response(S)
    if sr:
        print(f"\nReponse de la foulee (debut -> fin) :")
        print(f"  allure   {fmt_pace(sr['pace_start'])} -> {fmt_pace(sr['pace_end'])}")
        print(f"  cadence  {sr['cadence_start']:.1f} -> {sr['cadence_end']:.1f} pas/min")
        print(f"  foulee   {sr['step_len_start']:.0f} -> {sr['step_len_end']:.0f} mm")
        dc = sr['cadence_end'] - sr['cadence_start']
        print("  -> " + ("la foulee absorbe le ralentissement, cadence tenue : bon signe"
                        if abs(dc) < 4 else
                        "la cadence decroche avec l'allure : a corriger"))

    if args.efforts:
        if args.cadence_threshold:
            threshold, origin = args.cadence_threshold, "absolu"
        else:
            threshold = effort_cadence_threshold(S, args.margin)
            origin = f"mediane course + {args.margin:g}"
        eff = detect_efforts(S, cadence_threshold=threshold) if threshold else []
        borne = f"cadence >= {threshold:.1f}, {origin}" if threshold else "pas de cadence en course"
        print(f"\n{len(eff)} repetitions detectees (borne = {borne}) :")
        for i, e in enumerate(eff, 1):
            print(f"  {i:2}  a {e.start_s//60:>2}:{e.start_s%60:02d}  {e.duration_s:>3}s  "
                  f"D+ {e.ascent_m:5.1f}m  FCmax {e.hr_max:3d}  "
                  f"cad {e.cadence:5.1f}  {fmt_pace(e.pace_s_per_km)}")
        rec = recoveries(S, eff)
        if rec:
            print("\n  recuperations :")
            for i, r in enumerate(rec, 1):
                print(f"    apres {i}: {r['duration_s']:>3}s  marche {r['walk_pct']:3.0f} %  "
                      f"FC {r['hr_in']} -> {r['hr_min']} (baisse {r['hr_drop']})")

    if args.splits:
        print("\nPar kilometre :")
        for s in splits(S):
            print(f"  km {s['index']:2}  {s['ascent_m']:+6.1f}m  {fmt_pace(s['pace_s_per_km']):>9}  "
                  f"FC {s['hr_avg']:5.1f} (max {s['hr_max']})  cad {s['cadence'] or 0:5.1f}")


if __name__ == "__main__":
    main()
