"""
per_pola_v5.py — Jumlah transaksi terdeteksi dan tidak terdeteksi per pola
===========================================================================
Menghitung, untuk tiap pola money laundering pada data uji, berapa transaksi
yang terdeteksi dan yang lolos oleh ensemble v5 pada threshold yang dipilih di
data validasi. Hasilnya ditampilkan di halaman Dashboard dan menjadi sumber
Tabel 13 (Recall per Pola) pada naskah.

Masukan : data/raw/test_v4.csv, data/results/test_probs_v5_ensemble.npy,
          data/results/evaluation_v5.json (threshold)
Keluaran: data/results/per_pola_v5.json

Pemakaian:
    python scripts/eval/per_pola_v5.py
"""

from __future__ import annotations

import json
import os
import sys


def _find_root(start: str) -> str:
    path = os.path.abspath(start)
    while True:
        if os.path.exists(os.path.join(path, "backend", "config.py")):
            return path
        parent = os.path.dirname(path)
        if parent == path:
            raise RuntimeError("Akar proyek tidak ditemukan")
        path = parent


sys.path.insert(0, _find_root(__file__))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import backend.config as C  # noqa: E402

LABELS = {
    "Fan_Out": "Fan-Out",
    "Fan_In": "Fan-In",
    "Scatter_Gather": "Scatter-Gather (Smurfing)",
    "Gather_Scatter": "Gather-Scatter (Mixer)",
    "Simple_Cycle": "Simple Cycle (Round-Tripping)",
}


def main() -> None:
    with open(C.resolve("evaluation_v5.json"), encoding="utf-8") as f:
        threshold = float(json.load(f)["threshold"])

    probs = np.load(C.resolve("test_probs_v5_ensemble.npy"))
    test = pd.read_csv(C.resolve("test_v4.csv"), usecols=["Is_laundering", "Laundering_type"])
    if len(test) != len(probs):
        raise SystemExit(f"Jumlah baris berbeda: test {len(test):,} vs probabilitas {len(probs):,}")

    flagged = probs >= threshold
    positive = test["Is_laundering"].values == 1

    rows = []
    for pola in C.THESIS_PATTERNS:
        mask = positive & (test["Laundering_type"].values == pola)
        total = int(mask.sum())
        caught = int((flagged & mask).sum())
        rows.append({
            "pola": pola,
            "label": LABELS.get(pola, pola),
            "jumlah": total,
            "terdeteksi": caught,
            "tidak_terdeteksi": total - caught,
            "recall": round(caught / total, 6) if total else 0.0,
        })

    total = int(positive.sum())
    caught = int((flagged & positive).sum())
    out = {
        "versi": "v5 (ensemble 3 seed)",
        "threshold": threshold,
        "per_pola": rows,
        "total": {"jumlah": total, "terdeteksi": caught, "tidak_terdeteksi": total - caught,
                  "recall": round(caught / total, 6) if total else 0.0},
    }

    path = C.resolve("per_pola_v5.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"{'Pola':<32}{'Jumlah':>9}{'Terdeteksi':>12}{'Lolos':>8}{'Recall':>10}")
    for r in rows + [{"label": "Total", **out["total"]}]:
        print(f"{r['label']:<32}{r['jumlah']:>9,}{r['terdeteksi']:>12,}"
              f"{r['tidak_terdeteksi']:>8,}{r['recall'] * 100:>9.2f}%")
    print(f"\nDisimpan: {path}")


if __name__ == "__main__":
    main()
