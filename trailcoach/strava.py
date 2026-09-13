"""Client Strava minimal : OAuth, activites, flux.

Les identifiants viennent de l'environnement (voir .env.example), jamais du code.
Les jetons sont ecrits dans le repertoire de donnees, hors depot.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from .config import StravaConfig, data_dir

API_BASE = "https://www.strava.com/api/v3"
SCOPE = "activity:read_all"
STREAM_KEYS = "time,distance,altitude,heartrate,cadence,velocity_smooth"


def _http_json(url: str, data: dict | None = None, token: str | None = None):
    body = urllib.parse.urlencode(data).encode() if data else None
    req = urllib.request.Request(url, data=body)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


class StravaClient:
    def __init__(self, cfg: StravaConfig | None = None, tokens_path: Path | None = None):
        self.cfg = cfg or StravaConfig.from_env()
        self.tokens_path = tokens_path or (data_dir() / "strava_tokens.json")

    # -- OAuth ------------------------------------------------------------

    def _wait_for_code(self) -> str:
        holder: dict[str, str] = {}

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                if "code" in qs:
                    holder["code"] = qs["code"][0]
                    body = b"<h2>Autorisation Strava OK, tu peux fermer cet onglet.</h2>"
                else:
                    body = b"<h2>Pas de code dans la requete.</h2>"
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = HTTPServer(("localhost", self.cfg.callback_port), Handler)
        while "code" not in holder:
            server.handle_request()
        server.server_close()
        return holder["code"]

    def authorize(self) -> dict:
        redirect_uri = f"http://localhost:{self.cfg.callback_port}/callback"
        url = "https://www.strava.com/oauth/authorize?" + urllib.parse.urlencode({
            "client_id": self.cfg.client_id,
            "response_type": "code",
            "redirect_uri": redirect_uri,
            "approval_prompt": "force",
            "scope": SCOPE,
        })
        print("Ouvre cette URL et clique Autoriser :\n")
        print(f"  {url}\n")
        print(f"(en attente du retour sur localhost:{self.cfg.callback_port}...)", flush=True)
        code = self._wait_for_code()
        tokens = _http_json("https://www.strava.com/oauth/token", {
            "client_id": self.cfg.client_id,
            "client_secret": self.cfg.client_secret,
            "code": code,
            "grant_type": "authorization_code",
        })
        self._save_tokens(tokens)
        return tokens

    def _save_tokens(self, tokens: dict) -> None:
        self.tokens_path.parent.mkdir(parents=True, exist_ok=True)
        self.tokens_path.write_text(json.dumps({
            "access_token": tokens["access_token"],
            "refresh_token": tokens["refresh_token"],
            "expires_at": tokens["expires_at"],
        }))
        self.tokens_path.chmod(0o600)

    def access_token(self) -> str:
        if not self.tokens_path.exists():
            return self.authorize()["access_token"]
        tokens = json.loads(self.tokens_path.read_text())
        if tokens["expires_at"] > time.time() + 60:
            return tokens["access_token"]
        refreshed = _http_json("https://www.strava.com/oauth/token", {
            "client_id": self.cfg.client_id,
            "client_secret": self.cfg.client_secret,
            "refresh_token": tokens["refresh_token"],
            "grant_type": "refresh_token",
        })
        self._save_tokens(refreshed)
        return refreshed["access_token"]

    # -- Lecture ----------------------------------------------------------

    def activities(self, after: str | datetime) -> list[dict]:
        """Toutes les activites posterieures a `after` (date ISO ou datetime)."""
        if isinstance(after, str):
            after = datetime.fromisoformat(after)
        ts = int(after.replace(tzinfo=timezone.utc).timestamp())
        token = self.access_token()
        out, page = [], 1
        while True:
            batch = _http_json(
                f"{API_BASE}/athlete/activities?after={ts}&per_page=100&page={page}",
                token=token,
            )
            if not batch:
                break
            out.extend(batch)
            page += 1
        return out

    def streams(self, activity_id: int) -> dict:
        return _http_json(
            f"{API_BASE}/activities/{activity_id}/streams"
            f"?keys={STREAM_KEYS}&key_by_type=true",
            token=self.access_token(),
        )

    def sync(self, after: str, out_dir: Path | None = None, sleep: float = 0.5) -> dict:
        """Recupere activites + flux, en incremental.

        ATTENTION : ecrit activities_<after>.json, pas un fichier unique. Ecraser
        un activities.json commun fait perdre l'historique des fetchs precedents,
        erreur classique quand on enchaine plusieurs fenetres.
        """
        out_dir = out_dir or data_dir()
        streams_dir = out_dir / "streams"
        streams_dir.mkdir(parents=True, exist_ok=True)

        acts = self.activities(after)
        (out_dir / f"activities_{after}.json").write_text(json.dumps(acts, indent=1))

        runs = [a for a in acts if a.get("type") == "Run"]
        fetched = 0
        for a in runs:
            target = streams_dir / f"{a['id']}.json"
            if target.exists():
                continue
            try:
                target.write_text(json.dumps(self.streams(a["id"])))
                fetched += 1
                time.sleep(sleep)  # rester sous 100 req / 15 min
            except Exception as exc:  # noqa: BLE001
                print(f"  {a['id']} erreur flux: {exc}")
        return {"activities": len(acts), "runs": len(runs), "streams_fetched": fetched}
