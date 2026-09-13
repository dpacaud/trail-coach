"""Lecture des fichiers FIT Garmin.

Pourquoi ce module existe alors que Strava expose deja des flux : Strava lisse
`velocity_smooth` au point de rendre inexploitable toute seance a intervalles
courts. Sur une seance de 6 x 45 s analysee via Strava, on peut conclure a des
repetitions de 30-35 s non marchees en descente, la ou le FIT brut montre
46 a 51 s et 100 % des recuperations marchees. Le FIT porte en plus la
puissance, la longueur de foulee, le temps de contact et l'oscillation
verticale, absents de l'API Strava.

Conventions Garmin a connaitre, chacune est un piege :

- `timestamp` est en UTC. Une seance lancee a 19h28 locale apparait a 17h28.
- `cadence` est en cycles par minute (une jambe). Les pas par minute valent
  ``(cadence + fractional_cadence) * 2``.
- `enhanced_speed` et `enhanced_altitude` sont plus precis que `speed`/`altitude`.
- L'altitude est barometrique, donc bien meilleure que celle du GPS.
"""
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from fitparse import FitFile


@dataclass
class Sample:
    """Un enregistrement a 1 Hz."""

    t: int                      # secondes depuis le debut
    dist: float | None = None   # metres cumules
    speed: float | None = None  # m/s
    alt: float | None = None    # metres
    hr: int | None = None       # bpm
    cadence: float | None = None  # PAS par minute (deja x2)
    step_len: float | None = None  # mm
    stance: float | None = None    # ms
    vert_osc: float | None = None  # mm
    vert_ratio: float | None = None  # %
    power: int | None = None    # watts
    temp: int | None = None     # degres C


@dataclass
class Session:
    """Une activite FIT normalisee."""

    start_utc: datetime
    tz: str = "Europe/Paris"
    samples: list[Sample] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    laps: list[dict] = field(default_factory=list)

    @property
    def start_local(self) -> datetime:
        return self.start_utc.replace(tzinfo=timezone.utc).astimezone(ZoneInfo(self.tz))

    @property
    def distance_km(self) -> float:
        return (self.meta.get("total_distance") or 0) / 1000

    @property
    def duration_s(self) -> float:
        return self.meta.get("total_timer_time") or 0

    @property
    def pace_s_per_km(self) -> float | None:
        return self.duration_s / self.distance_km if self.distance_km else None

    def running(self, min_cadence: float = 140, min_speed: float = 1.5) -> list[Sample]:
        """Echantillons ou l'athlete court reellement.

        Filtrer sur la cadence ET la vitesse evite de compter les arrets, la
        marche et les transitions dans une moyenne de foulee ou d'allure.
        """
        return [
            s for s in self.samples
            if s.cadence and s.cadence >= min_cadence
            and s.speed and s.speed >= min_speed
        ]

    def walking(self, max_cadence: float = 140) -> list[Sample]:
        """Echantillons en marche.

        La cadence discrimine bien mieux que la vitesse : une marche rapide
        atteint 1,7 m/s, au-dessus des seuils de vitesse naifs, mais sa cadence
        reste sous 135 alors que la course est au-dessus de 150.
        """
        return [s for s in self.samples if s.cadence and 0 < s.cadence < max_cadence]


def _first(d: dict, *names):
    for n in names:
        if d.get(n) is not None:
            return d[n]
    return None


def load(path: str | Path, tz: str = "Europe/Paris") -> Session:
    """Charge un .fit (ou un .zip qui en contient un) en Session normalisee."""
    path = Path(path)
    if path.suffix.lower() == ".zip":
        # FitFile ne lit que l'en-tete a l'ouverture, les messages a la demande :
        # on charge le FIT en memoire, un handle de zip serait deja ferme
        with zipfile.ZipFile(path) as z:
            name = next((n for n in z.namelist() if n.lower().endswith(".fit")), None)
            if name is None:
                raise FileNotFoundError(f"aucun fichier .fit dans {path}")
            fit = FitFile(io.BytesIO(z.read(name)))
    else:
        fit = FitFile(str(path))

    records, t0 = [], None
    for msg in fit.get_messages("record"):
        d = {f.name: f.value for f in msg}
        ts = d.get("timestamp")
        if ts is None:
            continue
        if t0 is None:
            t0 = ts
        cad = d.get("cadence")
        records.append(Sample(
            t=int((ts - t0).total_seconds()),
            dist=d.get("distance"),
            speed=_first(d, "enhanced_speed", "speed"),
            alt=_first(d, "enhanced_altitude", "altitude"),
            hr=d.get("heart_rate"),
            cadence=((cad + (d.get("fractional_cadence") or 0)) * 2) if cad is not None else None,
            step_len=d.get("step_length"),
            stance=d.get("stance_time"),
            vert_osc=d.get("vertical_oscillation"),
            vert_ratio=d.get("vertical_ratio"),
            power=d.get("power"),
            temp=d.get("temperature"),
        ))

    meta: dict = {}
    sessions = list(fit.get_messages("session"))
    if sessions:
        meta = {f.name: f.value for f in sessions[0]}

    laps = []
    for lap in fit.get_messages("lap"):
        laps.append({f.name: f.value for f in lap})

    return Session(
        start_utc=meta.get("start_time") or t0 or datetime.now(),
        tz=tz,
        samples=records,
        meta=meta,
        laps=laps,
    )


def temperature_curve(sess: Session, window_s: int = 1200) -> list[tuple[int, float]]:
    """Temperature moyenne par fenetre. Utile pour separer derive thermique et fatigue.

    Une sortie longue lancee tot le matin en ete se termine souvent 4 a 5 degres
    plus haut qu'elle n'a commence, ce qui suffit a expliquer une derive
    cardiaque que l'on attribuerait sinon a la fatigue.
    """
    out = []
    if not sess.samples:
        return out
    end = sess.samples[-1].t
    for a in range(0, end + 1, window_s):
        vals = [s.temp for s in sess.samples if a <= s.t < a + window_s and s.temp is not None]
        if vals:
            out.append((a, sum(vals) / len(vals)))
    return out
