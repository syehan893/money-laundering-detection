"""
scripts/baseline/train_ensemble_beragam.py — Anggota ensemble dengan konfigurasi beragam
==========================================================================================
Melatih tiga anggota ensemble tambahan yang konfigurasinya sengaja dibedakan,
untuk digabung dengan tiga seed v5 yang sudah ada.

Mengapa dibedakan, bukan sekadar seed baru
-------------------------------------------
Ensemble bekerja karena anggotanya membuat kesalahan yang berbeda. Beberapa seed
dengan konfigurasi identik menghasilkan model yang sangat mirip, sehingga
rata-ratanya hanya meredam sedikit varians. Anggota dengan bias induktif berbeda
menghasilkan kesalahan yang lebih tidak berkorelasi, dan rata-ratanya lebih kuat.

Diagnosis yang melandasi pilihan konfigurasinya
-----------------------------------------------
Ketiga seed v5 dilatih dengan `fn_cost = 12.0` berbanding `fp_cost = 1.0`. Model
karenanya takut melewatkan pencucian uang dua belas kali lebih besar daripada
takut salah tuduh, dan melempar kasus ambigu ke sisi positif dengan penuh
keyakinan. Akibatnya terukur: 62 false positive yang tersisa punya skor median
0,9998 -- model yakin salah, sehingga menggeser ambang tidak menyentuhnya sama
sekali. Sementara 80 false negative justru berkumpul tepat di bawah ambang
(median 0,5144).

Anggota baru di sini memakai `fn_cost` yang jauh lebih rendah untuk menekan
keyakinan berlebih itu. GAT punya keunggulan 15 false negative atas XGBoost
(80 berbanding 95), yang memberi ruang untuk ditukar dengan pengurangan false
positive.

Usage:
    python scripts/baseline/train_ensemble_beragam.py
    python scripts/baseline/train_ensemble_beragam.py --only C
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")


def _find_root(start: str) -> str:
    """Akar proyek = direktori terdekat di atas berkas ini yang memuat backend/config.py."""
    d = os.path.dirname(os.path.abspath(start))
    for _ in range(6):
        if os.path.exists(os.path.join(d, "backend", "config.py")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError(f"Akar proyek tidak ditemukan dari {start}")


PROJECT_ROOT = _find_root(__file__)
sys.path.insert(0, PROJECT_ROOT)

# Tiga arah bias yang berbeda. `tag` menentukan nama berkas keluaran.
ANGGOTA = {
    "C": {
        "tag": "costlow", "seed": 7,
        "fn_cost": 6.0, "dropout": 0.25, "label_smoothing": 0.03,
        "catatan": "biaya FN separuh -- menyasar langsung 62 FP berkeyakinan tinggi",
    },
    "D": {
        "tag": "costlow_reg", "seed": 99,
        "fn_cost": 5.0, "dropout": 0.35, "label_smoothing": 0.05,
        "catatan": "biaya FN lebih rendah lagi + regularisasi kuat, menekan keyakinan berlebih",
    },
    "E": {
        "tag": "wide", "seed": 512,
        "fn_cost": 8.0, "dropout": 0.30, "hidden_dim": 96, "label_smoothing": 0.02,
        "catatan": "kapasitas lebih besar, biaya FN menengah -- bias induktif berbeda",
    },
}


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    ap = argparse.ArgumentParser(description="Latih anggota ensemble beragam.")
    ap.add_argument("--only", choices=sorted(ANGGOTA), help="Latih satu anggota saja.")
    args = ap.parse_args()

    from backend.ml.train import main as train_main

    dipilih = {args.only: ANGGOTA[args.only]} if args.only else ANGGOTA
    for nama, cfg in dipilih.items():
        cfg = dict(cfg)
        catatan = cfg.pop("catatan")
        seed = cfg.pop("seed")
        print("\n" + "=" * 74)
        print(f"  Anggota {nama} — {catatan}")
        print(f"  seed={seed}  " + "  ".join(f"{k}={v}" for k, v in cfg.items() if k != "tag"))
        print("=" * 74)
        train_main("v5", seed=seed, override=cfg)


if __name__ == "__main__":
    main()
