# trailcoach

Lire ses données Strava et Garmin, et construire un plan de préparation trail
sur une méthode explicite.

Deux choses dans ce dépôt :

- **Du code** pour récupérer ses activités Strava et décoder ses fichiers FIT
  Garmin, avec les conventions et les pièges documentés.
- **[`docs/METHODE.md`](docs/METHODE.md)**, la méthode de construction d'un plan,
  chaque règle étiquetée selon qu'elle est mesurée, d'usage, ou observée sur un
  cycle réel.

Aucune donnée personnelle n'est versionnée. Le `.gitignore` exclut les jetons,
les exports et les fichiers FIT.

## Installation

```bash
git clone <url> && cd trail-coach
uv venv && uv pip install -e .      # ou : python -m venv .venv && pip install -e .
cp .env.example .env                 # puis remplir
```

## Configuration

Les identifiants ne sont **jamais** dans le code. Créer une application sur
<https://www.strava.com/settings/api> (Authorization Callback Domain :
`localhost`), puis renseigner `.env` :

```
STRAVA_CLIENT_ID=...
STRAVA_CLIENT_SECRET=...
ATHLETE_HR_MAX=196
ATHLETE_LTHR=178
```

`ATHLETE_HR_MAX` doit être la valeur **observée** sur l'historique, pas une
estimation sur l'âge ni l'auto-détection de la montre. Voir la section
correspondante de la méthode : une FC max trop haute fait afficher des zones trop
basses, ce qui rassure l'athlète au mauvais moment.

## Usage

Récupérer Strava (OAuth au premier lancement, jetons mis en cache) :

```bash
python scripts/fetch_strava.py --after 2026-08-16
```

Analyser une séance depuis son FIT, y compris le `.zip` téléchargé depuis
Garmin Connect :

```bash
python scripts/analyze_session.py seance.zip --target-hr 150
python scripts/analyze_session.py seance_de_cotes.fit --efforts
python scripts/analyze_session.py sortie_longue.fit --splits
```

En bibliothèque :

```python
from trailcoach import Athlete, load_dotenv
from trailcoach import fit
from trailcoach.metrics import detect_efforts, hr_drift, zone_distribution

load_dotenv()
ath = Athlete.from_env()
sess = fit.load("seance.fit", tz=ath.tz)

print(sess.start_local)                        # heure LOCALE, le FIT stocke en UTC
print(zone_distribution(sess.samples, ath.hr_max))
print(hr_drift(sess.samples).pct)
for e in detect_efforts(sess.samples):         # borne sur la cadence, pas la vitesse
    print(e.duration_s, e.cadence)
```

## Pourquoi lire le FIT et pas seulement Strava

Strava lisse `velocity_smooth` au point de rendre inexploitable toute séance à
intervalles courts. Sur une séance réelle de 6 × 45 s, l'analyse via Strava
concluait à des répétitions de 30 à 35 s sans marche en descente ; le FIT brut
montrait 46 à 51 s et 100 % des récupérations marchées.

Le FIT porte en plus la puissance, la longueur de foulée, le temps de contact au
sol et l'oscillation verticale, absents de l'API Strava. C'est la longueur de
foulée qui permet de vérifier le geste protecteur pour une articulation fragile :
raccourcir la foulée sans baisser la cadence quand on ralentit.

## Conventions Garmin

| Piège | Réalité |
|---|---|
| `timestamp` | UTC, pas local |
| `cadence` | cycles/min, une jambe. Pas/min = `(cadence + fractional_cadence) * 2` |
| `speed`, `altitude` | préférer `enhanced_speed` et `enhanced_altitude` |
| bornes d'un effort | détecter sur la **cadence**, pas la vitesse ni la puissance |
| marche vs course | cadence < 140 vs > 150. Une marche rapide atteint 1,7 m/s |

## Licence

MIT.
