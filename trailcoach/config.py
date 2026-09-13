"""Configuration lue depuis l'environnement, jamais depuis le code.

Charge un fichier .env s'il existe, sans dependance externe.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> None:
    """Injecte les variables d'un .env dans os.environ (sans ecraser l'existant)."""
    p = Path(path)
    if not p.exists():
        return
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _req(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Variable d'environnement {name} manquante. "
            f"Copie .env.example en .env et remplis-la."
        )
    return value


@dataclass(frozen=True)
class Athlete:
    """Reperes physiologiques.

    hr_max doit etre la valeur OBSERVEE sur l'historique, pas une estimation
    sur l'age ni l'auto-detection de la montre : celles-ci se trompent souvent
    de plusieurs battements, et une FC max trop haute fait afficher des zones
    trop basses, ce qui flatte l'athlete au mauvais moment.

    lthr est la FC moyenne tenue sur un effort maximal d'environ une heure.
    Une course de 10 km bouclee entre 55 et 65 minutes en donne une bonne mesure.
    """

    hr_max: int = 196
    lthr: int | None = 178
    hr_rest: int | None = 61
    weight_kg: float | None = None
    tz: str = "Europe/Paris"

    @classmethod
    def from_env(cls) -> "Athlete":
        def _opt(name, cast, default=None):
            v = os.environ.get(name, "").strip()
            return cast(v) if v else default

        return cls(
            hr_max=_opt("ATHLETE_HR_MAX", int, 196),
            lthr=_opt("ATHLETE_LTHR", int),
            hr_rest=_opt("ATHLETE_HR_REST", int),
            weight_kg=_opt("ATHLETE_WEIGHT_KG", float),
            tz=os.environ.get("ATHLETE_TZ", "Europe/Paris"),
        )


@dataclass(frozen=True)
class StravaConfig:
    client_id: str
    client_secret: str
    callback_port: int = 8723

    @classmethod
    def from_env(cls) -> "StravaConfig":
        return cls(
            client_id=_req("STRAVA_CLIENT_ID"),
            client_secret=_req("STRAVA_CLIENT_SECRET"),
            callback_port=int(os.environ.get("STRAVA_CALLBACK_PORT", "8723")),
        )


def data_dir() -> Path:
    d = Path(os.environ.get("TRAILCOACH_DATA_DIR", "./data")).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    return d
