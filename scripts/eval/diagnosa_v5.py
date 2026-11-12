"""
scripts/diagnosa_v5.py — Verifikasi apakah dua akar kegagalan v4 tertutup di v5
===============================================================================
Menjalankan kembali diagnosis yang dipakai untuk merancang v5, pada model v5,
sehingga perbaikannya dapat dinilai pada ukuran yang sama.

Dua akar kegagalan v4:
  1. Prior kategorikal mendominasi saat graf tidak informatif -- Payment_type
     "Cash Deposit" menyumbang 323 dari 480 FP (FPR 3,24%, 28x rata-rata).
  2. Model nyaris buta pada akun berderajat rendah -- FNR 96,3% pada derajat < 10
     berbanding 0,030% pada derajat >= 80.

Usage:
    python scripts/diagnosa_v5.py
"""

from __future__ import annotations

import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

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

import backend.config as C


def muat(versi: str):
    """Kembalikan (dataframe test dengan kolom p, ambang) untuk sebuah versi."""
    if versi == "v4":
        probs = np.load(C.resolve("test_probs_v4.npy"))
        thr = json.load(open(C.resolve("evaluation_v4.json"), encoding="utf-8"))["threshold"]
    else:
        probs = np.load(C.resolve("test_probs_v5_ensemble.npy"))
        thr = json.load(open(C.resolve("evaluation_v5.json"), encoding="utf-8"))["threshold"]
    t = pd.read_csv(C.TEST_CSV_V4_PATH)
    t["p"] = probs
    return t, thr


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    hasil = {}
    for versi in ("v4", "v5"):
        try:
            t, thr = muat(versi)
        except FileNotFoundError:
            print(f"  [lewati] {versi}: berkas hasil belum ada")
            continue
        deg = pd.concat([t.Sender_account, t.Receiver_account]).value_counts()
        t["deg"] = t.Sender_account.map(deg).fillna(0) + t.Receiver_account.map(deg).fillna(0)
        t["salah_fp"] = (t.Is_laundering == 0) & (t.p >= thr)
        t["salah_fn"] = (t.Is_laundering == 1) & (t.p < thr)
        hasil[versi] = (t, thr)

    if "v5" not in hasil:
        print("Hasil v5 belum tersedia.")
        return

    print("=" * 78)
    print("  Diagnosis ulang: apakah akar kegagalan v4 tertutup di v5?")
    print("=" * 78)

    print("\n[1] FPR per Payment_type  (akar masalah: prior kategorikal)")
    print(f"  {'tipe':<18}{'FP v4':>8}{'FPR v4':>10}{'FP v5':>8}{'FPR v5':>10}{'perubahan':>12}")
    t4 = hasil.get("v4", (None,))[0]
    t5 = hasil["v5"][0]
    for pt in t5.Payment_type.value_counts().index:
        row = []
        for t in (t4, t5):
            if t is None:
                row += [None, None]; continue
            neg = t[(t.Is_laundering == 0) & (t.Payment_type == pt)]
            row += [int(neg.salah_fp.sum()), neg.salah_fp.mean()]
        if row[0] is None:
            continue
        delta = f"{(row[2]-row[0]):+d}"
        print(f"  {pt:<18}{row[0]:>8}{row[1]*100:>9.4f}%{row[2]:>8}{row[3]*100:>9.4f}%{delta:>12}")

    print("\n[2] FPR & FNR per derajat graf  (akar masalah: konteks kosong)")
    print(f"  {'derajat':<10}{'FPR v4':>10}{'FPR v5':>10}   {'FNR v4':>10}{'FNR v5':>10}")
    for lo, hi, lab in [(0, 10, "0-9"), (10, 20, "10-19"), (20, 40, "20-39"),
                        (40, 80, "40-79"), (80, 1e9, "80+")]:
        vals = []
        for t in (t4, t5):
            s = t[(t.deg >= lo) & (t.deg < hi)]
            neg, pos = s[s.Is_laundering == 0], s[s.Is_laundering == 1]
            vals += [neg.salah_fp.mean() if len(neg) else 0.0,
                     pos.salah_fn.mean() if len(pos) else 0.0]
        print(f"  {lab:<10}{vals[0]*100:>9.4f}%{vals[2]*100:>9.4f}%   "
              f"{vals[1]*100:>9.4f}%{vals[3]*100:>9.4f}%")

    print("\n[3] Ringkasan")
    for versi in ("v4", "v5"):
        if versi not in hasil:
            continue
        t, thr = hasil[versi]
        fp, fn = int(t.salah_fp.sum()), int(t.salah_fn.sum())
        print(f"  {versi}: FP {fp:>4}  FN {fn:>4}  total {fp+fn:>4}  (ambang {thr:.4f})")
    fp4, fn4 = int(t4.salah_fp.sum()), int(t4.salah_fn.sum())
    fp5, fn5 = int(t5.salah_fp.sum()), int(t5.salah_fn.sum())
    print(f"  Perubahan total kesalahan: {fp4+fn4} -> {fp5+fn5} "
          f"({(fp5+fn5-fp4-fn4)/(fp4+fn4)*100:+.1f}%)")


if __name__ == "__main__":
    main()
