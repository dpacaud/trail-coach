#!/usr/bin/env python3
"""Recupere activites et flux Strava depuis une date.

    python scripts/fetch_strava.py --after 2026-08-16
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trailcoach import load_dotenv, data_dir          # noqa: E402
from trailcoach.strava import StravaClient            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--after", required=True, help="date ISO, ex 2026-08-16")
    args = ap.parse_args()

    load_dotenv()
    client = StravaClient()
    print(f"Recuperation depuis {args.after}...")
    res = client.sync(args.after)
    print(f"  {res['activities']} activites ({res['runs']} Run), "
          f"{res['streams_fetched']} nouveaux flux")
    print(f"  -> {data_dir()}")


if __name__ == "__main__":
    main()
