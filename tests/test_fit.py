"""Tests de trailcoach.fit sur un fichier FIT synthetique, construit octet par octet."""
import struct
import zipfile
from datetime import datetime, timezone

import pytest
from fitparse.records import Crc

from trailcoach import fit

FIT_EPOCH = datetime(1989, 12, 31, tzinfo=timezone.utc)
START = datetime(2026, 5, 3, 6, 0, tzinfo=timezone.utc)

# message `record` (global 20) : (numero de champ, taille, type de base)
RECORD_FIELDS = [(253, 4, 0x86),   # timestamp, s depuis l'epoque FIT
                 (3, 1, 0x02),     # heart_rate
                 (4, 1, 0x02),     # cadence, cycles/min
                 (5, 4, 0x86),     # distance, cm
                 (6, 2, 0x84)]     # speed, mm/s


def _fit_bytes(samples):
    """samples : liste de (t_s, hr, cadence_cycles, distance_m, speed_ms)."""
    data = struct.pack("<BBBHB", 0x40, 0, 0, 20, len(RECORD_FIELDS))
    data += b"".join(struct.pack("<BBB", *f) for f in RECORD_FIELDS)
    t0 = int((START - FIT_EPOCH).total_seconds())
    for t, hr, cad, dist, speed in samples:
        data += struct.pack("<BIBBIH", 0x00, t0 + t, hr, cad, round(dist * 100), round(speed * 1000))
    header = struct.pack("<BBHI4sH", 14, 0x10, 2093, len(data), b".FIT", 0)
    return header + data + struct.pack("<H", Crc.calculate(header + data))


SAMPLES = [(0, 120, 80, 0.0, 2.5), (1, 125, 84, 2.5, 2.6), (2, 130, 85, 5.1, 2.7)]


def _check(sess):
    assert [s.t for s in sess.samples] == [0, 1, 2]
    assert [s.hr for s in sess.samples] == [120, 125, 130]
    assert [s.cadence for s in sess.samples] == [160, 168, 170]   # pas/min = cycles x 2
    assert sess.samples[2].dist == pytest.approx(5.1)
    assert sess.samples[1].speed == pytest.approx(2.6)
    assert sess.start_utc == START.replace(tzinfo=None)


def test_load_fit(tmp_path):
    path = tmp_path / "seance.fit"
    path.write_bytes(_fit_bytes(SAMPLES))
    _check(fit.load(path))


def test_load_zip(tmp_path):
    """fitparse lit les messages a la demande : le handle du zip doit rester
    lisible apres l'ouverture."""
    path = tmp_path / "seance.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("readme.txt", "pas un fit")
        z.writestr("123456_ACTIVITY.fit", _fit_bytes(SAMPLES))
    _check(fit.load(path))


def test_zip_without_fit(tmp_path):
    path = tmp_path / "vide.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("readme.txt", "pas un fit")
    with pytest.raises(FileNotFoundError):
        fit.load(path)
