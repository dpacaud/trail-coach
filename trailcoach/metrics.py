"""Metriques d'analyse d'une seance.

Toutes les fonctions prennent une liste de Sample (voir trailcoach.fit) et
restent independantes de la source : un flux Strava converti en Sample
fonctionne aussi, avec la reserve sur le lissage rappelee dans fit.py.
"""
from __future__ import annotations

import statistics as st
from dataclasses import dataclass

from .fit import Sample

# Bornes en % de la FC max. Le decoupage 60/70/80/90 est celui de Garmin en
# mode "percent_max_hr". Strava applique ses propres seuils : a FC identique on
# peut etre en Z3 sur un ecran et en Z4 sur l'autre sans que personne ait tort.
# C'est pourquoi toute consigne donnee a un athlete doit etre en BATTEMENTS
# ABSOLUS, jamais en numero de zone.
ZONE_BOUNDS = (0.60, 0.70, 0.80, 0.90)
ZONE_NAMES = ("Z1 recup", "Z2 endurance", "Z3 tempo", "Z4 seuil", "Z5 max")


def zone_edges(hr_max: int) -> list[int]:
    return [round(hr_max * b) for b in ZONE_BOUNDS]


def zone_distribution(samples: list[Sample], hr_max: int) -> dict[str, float]:
    """Part du temps passee dans chaque zone, ponderee par la duree reelle."""
    pts = [s for s in samples if s.hr]
    if len(pts) < 2:
        return {}
    edges = zone_edges(hr_max)
    secs = [0.0] * 5
    for prev, cur in zip(pts, pts[1:]):
        dt = cur.t - prev.t
        hr = cur.hr
        idx = sum(1 for e in edges if hr >= e)
        secs[idx] += dt
    total = sum(secs) or 1
    return {name: 100 * s / total for name, s in zip(ZONE_NAMES, secs)}


def time_below(samples: list[Sample], bpm: int) -> float:
    """Part du temps sous un seuil de FC donne, en pourcent."""
    pts = [s for s in samples if s.hr]
    if not pts:
        return 0.0
    return 100 * sum(1 for s in pts if s.hr < bpm) / len(pts)


@dataclass
class Drift:
    first_half_hr: float
    second_half_hr: float
    first_half_pace: float | None
    second_half_pace: float | None

    @property
    def pct(self) -> float:
        return 100 * (self.second_half_hr - self.first_half_hr) / self.first_half_hr


def hr_drift(samples: list[Sample]) -> Drift | None:
    """Derive cardiaque entre les deux moities de la seance.

    A interpreter avec prudence sur un aller-retour : un parcours qui descend
    a l'aller et remonte au retour gonfle mecaniquement la derive. Comparer le
    denivele net de chaque moitie avant de conclure quoi que ce soit.
    Au-dela de deux heures, 8 a 10 % sont normaux meme au frais.
    """
    pts = [s for s in samples if s.hr and s.speed and s.speed > 1.2]
    if len(pts) < 20:
        return None
    mid = pts[len(pts) // 2].t
    h1 = [s for s in pts if s.t < mid]
    h2 = [s for s in pts if s.t >= mid]
    if not h1 or not h2:
        return None
    pace = lambda g: 1000 / st.mean([s.speed for s in g])
    return Drift(
        first_half_hr=st.mean([s.hr for s in h1]),
        second_half_hr=st.mean([s.hr for s in h2]),
        first_half_pace=pace(h1),
        second_half_pace=pace(h2),
    )


def _keep(span, lo: int, hi: int | None) -> bool:
    d = span[1] - span[0]
    return d >= lo and (hi is None or d <= hi)


@dataclass
class Effort:
    start_s: int
    end_s: int
    ascent_m: float
    hr_max: int
    cadence: float
    pace_s_per_km: float | None

    @property
    def duration_s(self) -> int:
        return self.end_s - self.start_s + 1


def effort_cadence_threshold(samples: list[Sample], margin: float = 8.0) -> float | None:
    """Seuil relatif de detection des efforts : mediane de la cadence en course + `margin`."""
    run = [s.cadence for s in samples
           if s.cadence and s.cadence > 140 and s.speed and s.speed > 1.5]
    return st.median(run) + margin if run else None


def detect_efforts(
    samples: list[Sample],
    cadence_threshold: float | None = None,
    min_duration_s: int = 12,
    max_duration_s: int | None = 180,
    margin: float = 8.0,
) -> list[Effort]:
    """Detecte les repetitions d'une seance d'intervalles.

    LA borne fiable est la CADENCE, qui saute instantanement d'environ 125 en
    marche a 165 en effort, et retombe aussi net. Ne pas utiliser :

    - la VITESSE, car en cote une repetition intense peut etre plus lente qu'un
      footing a plat, et parce que les flux lisses noient les transitions ;
    - un SEUIL DE PUISSANCE, car il coupe la rampe de montee en puissance des
      8 a 10 premieres secondes et raccourcit artificiellement chaque effort.

    Sur une seance reelle de 6 x 45 s, la detection par puissance donnait
    30 a 39 s, la detection par cadence 46 a 51 s. Seule la seconde etait juste.

    Le seuil est RELATIF par defaut : mediane de la cadence en course + `margin`.
    Un seuil absolu se trompe des que la cadence de footing de l'athlete depasse
    la valeur choisie : a 165 pas/min en footing, un seuil a 150 classe
    l'echauffement entier comme un effort.

    `max_duration_s` ecarte les blocs continus (echauffement, retour au calme)
    qui passeraient le seuil. Le mettre a None pour detecter des efforts longs.

    Limite connue : sur des cotes raides ou la cadence reste basse pendant
    l'effort, l'ecart avec le footing se resserre. Passer alors un
    `cadence_threshold` absolu, ou borner les efforts autrement.
    """
    if cadence_threshold is None:
        cadence_threshold = effort_cadence_threshold(samples, margin)
        if cadence_threshold is None:
            return []
    efforts: list[Effort] = []
    cur: list[int] | None = None
    for s in samples:
        if s.cadence and s.cadence >= cadence_threshold:
            if cur is None:
                cur = [s.t, s.t]
            else:
                cur[1] = s.t
        else:
            if cur and _keep(cur, min_duration_s, max_duration_s):
                efforts.append(cur)
            cur = None
    if cur and _keep(cur, min_duration_s, max_duration_s):
        efforts.append(cur)

    out = []
    for a, b in efforts:
        seg = [s for s in samples if a <= s.t <= b]
        alts = [s.alt for s in seg if s.alt is not None]
        speeds = [s.speed for s in seg if s.speed and s.speed > 0.5]
        hrs = [s.hr for s in seg if s.hr]
        cads = [s.cadence for s in seg if s.cadence]
        out.append(Effort(
            start_s=a, end_s=b,
            ascent_m=(max(alts) - alts[0]) if alts else 0.0,
            hr_max=max(hrs) if hrs else 0,
            cadence=st.mean(cads) if cads else 0.0,
            pace_s_per_km=(1000 / st.mean(speeds)) if speeds else None,
        ))
    return out


def recoveries(samples: list[Sample], efforts: list[Effort]) -> list[dict]:
    """Caracterise les recuperations entre repetitions.

    Renvoie notamment `hr_drop`, la baisse de FC obtenue. Deux constats de
    terrain : une recuperation de moins de 30 s ne fait rien baisser, car le
    coeur met 20 a 30 s a seulement reagir ; et par forte chaleur la baisse
    reste d'environ 1 bpm meme sur 60 s, la charge thermique plafonnant le
    systeme. Marcher pour faire baisser la FC ne fonctionne qu'au frais.
    """
    out = []
    for e1, e2 in zip(efforts, efforts[1:]):
        seg = [s for s in samples if e1.end_s < s.t < e2.start_s]
        if not seg:
            continue
        hrs = [s.hr for s in seg if s.hr]
        cads = [s.cadence for s in seg if s.cadence]
        walked = [c for c in cads if c < 140]
        out.append({
            "duration_s": e2.start_s - e1.end_s,
            "hr_in": hrs[0] if hrs else None,
            "hr_min": min(hrs) if hrs else None,
            "hr_drop": (hrs[0] - min(hrs)) if hrs else None,
            "walk_pct": 100 * len(walked) / len(cads) if cads else 0,
        })
    return out


def splits(samples: list[Sample], every_m: int = 1000) -> list[dict]:
    """Decoupe par distance : allure, FC, denivele, cadence."""
    pts = [s for s in samples if s.dist is not None]
    if not pts:
        return []
    out = []
    k = 1
    while True:
        seg = [s for s in pts if (k - 1) * every_m <= s.dist < k * every_m]
        if len(seg) < 10:
            break
        speeds = [s.speed for s in seg if s.speed and s.speed > 1.2]
        hrs = [s.hr for s in seg if s.hr]
        cads = [s.cadence for s in seg if s.cadence and s.cadence > 140]
        alts = [s.alt for s in seg if s.alt is not None]
        out.append({
            "index": k,
            "pace_s_per_km": 1000 / st.mean(speeds) if speeds else None,
            "hr_avg": st.mean(hrs) if hrs else None,
            "hr_max": max(hrs) if hrs else None,
            "cadence": st.mean(cads) if cads else None,
            "ascent_m": (alts[-1] - alts[0]) if alts else None,
        })
        k += 1
    return out


def stride_response(samples: list[Sample]) -> dict | None:
    """Comment la foulee absorbe un ralentissement.

    Le geste protecteur pour une articulation fragile consiste a RACCOURCIR la
    foulee en gardant la cadence, plutot qu'a ralentir la cadence. Cette
    fonction compare le premier et le dernier tiers de la seance pour voir
    lequel des deux a bouge.
    """
    run = [s for s in samples if s.cadence and s.cadence > 140 and s.step_len]
    if len(run) < 60:
        return None
    n = len(run) // 3
    a, b = run[:n], run[-n:]
    f = lambda g, attr: st.mean([getattr(s, attr) for s in g if getattr(s, attr)])
    return {
        "cadence_start": f(a, "cadence"),
        "cadence_end": f(b, "cadence"),
        "step_len_start": f(a, "step_len"),
        "step_len_end": f(b, "step_len"),
        "pace_start": 1000 / f(a, "speed"),
        "pace_end": 1000 / f(b, "speed"),
    }


def fmt_pace(sec_per_km: float | None) -> str:
    if not sec_per_km:
        return "-"
    return f"{int(sec_per_km) // 60}:{int(sec_per_km) % 60:02d}/km"
