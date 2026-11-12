"""
scripts/eval/ringkasan_perbandingan.py — Perbandingan pada titik operasi yang disamakan
========================================================================================
Membandingkan GAT, XGBoost, dan Random Forest secara adil.

Masalah dengan membandingkan titik operasi masing-masing
---------------------------------------------------------
GAT pada titik seimbangnya mencatat FP 62 / FN 80; XGBoost mencatat FP 53 / FN 95.
Menyimpulkan "XGBoost lebih baik pada false positive" dari dua angka itu keliru,
karena keduanya berdiri di tempat yang berbeda pada kurva trade-off masing-masing.
Setiap model dapat menukar FP dengan FN hanya dengan menggeser ambangnya.

Cara yang benar
---------------
1. **Kurva trade-off penuh.** Untuk setiap jumlah FP yang mungkin, berapa FN yang
   dihasilkan tiap model. Ini tidak bergantung pada pilihan ambang sama sekali.
2. **Titik yang disamakan di validation.** Ambang GAT digeser sampai FN-nya di
   validation menyamai FN XGBoost di validation, lalu keduanya dilaporkan di test.
   Penyamaan dilakukan di validation supaya test tetap tidak tersentuh.

Keluaran:
  - data/results/ringkasan_perbandingan.json
  - docs/images/kurva_tradeoff_metode.png

Usage:
    python scripts/eval/ringkasan_perbandingan.py
"""

from __future__ import annotations

import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch


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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import backend.config as C

INK = "#0b0b0b"
INK_SOFT = "#52514e"
SURFACE = "#fcfcfb"
AKSEN = {"GAT v5": "#256abf", "XGBoost": "#7d6ba8", "Random Forest": "#b8733a"}
IMG = os.path.join(PROJECT_ROOT, "docs", "images", "kurva_tradeoff_metode.png")


def kurva(p: np.ndarray, y: np.ndarray):
    """Pasangan (FP, FN) menyusuri seluruh ambang, dari ambang tertinggi ke terendah."""
    order = np.argsort(-p)
    ys = y[order]
    fp = np.cumsum(~ys)
    tp = np.cumsum(ys)
    return fp, int(y.sum()) - tp


def fn_pada_fp(p: np.ndarray, y: np.ndarray, target_fp: int) -> int:
    fp, fn = kurva(p, y)
    i = np.searchsorted(fp, target_fp, side="right") - 1
    return int(fn[i]) if i >= 0 else int(y.sum())


def ambang_untuk_fn(p: np.ndarray, y: np.ndarray, target_fn: int) -> float:
    """Ambang tertinggi yang masih menghasilkan FN <= target."""
    order = np.argsort(-p)
    ys, ps = y[order], p[order]
    fn = int(y.sum()) - np.cumsum(ys)
    ok = np.where(fn <= target_fn)[0]
    return float(ps[ok[0]]) if len(ok) else 0.0


def confusion(y: np.ndarray, pred: np.ndarray) -> dict:
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    tn = int((~pred & ~y).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp, "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / max(prec + rec, 1e-12),
            "fpr": fp / max(fp + tn, 1), "fnr": fn / max(fn + tp, 1)}


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    data = torch.load(C.DATA_V5_PATH, weights_only=False)
    y = data.y.numpy().astype(bool)
    val_m = data.val_mask.numpy()
    test_m = data.test_mask.numpy()

    P = {
        "GAT v5": np.load(C.resolve("probs_v5_ensemble_full.npy")),
        "XGBoost": np.load(C.resolve("probs_xgboost_full.npy")),
        "Random Forest": np.load(C.resolve("probs_random_forest_full.npy")),
    }

    print("=" * 84)
    print("  1. Kurva trade-off penuh pada TEST — tidak bergantung pilihan ambang")
    print("=" * 84)
    print(f"  {'FP':>6}" + "".join(f"{n:>16}" for n in P))
    baris_kurva = []
    for target in (40, 46, 50, 53, 60, 62, 70, 80, 100):
        fns = {n: fn_pada_fp(P[n][test_m], y[test_m], target) for n in P}
        baris_kurva.append({"fp": target, **fns})
        best = min(fns, key=fns.get)
        print(f"  {target:>6}" + "".join(
            f"{('* ' if n == best else '  ') + str(fns[n]):>16}" for n in P))
    print("  (* = FN terkecil pada tingkat FP tersebut)")

    print("\n" + "=" * 84)
    print("  2. Titik disamakan DI VALIDATION, dilaporkan DI TEST")
    print("=" * 84)
    # Acuan: titik seimbang XGBoost di validation
    with open(C.resolve("baseline_xgboost.json"), encoding="utf-8") as f:
        thr_x = json.load(f)["threshold"]
    cm_x_val = confusion(y[val_m], P["XGBoost"][val_m] >= thr_x)
    print(f"  Acuan XGBoost di validation: FP {cm_x_val['fp']}  FN {cm_x_val['fn']}")

    hasil = {}
    for nama in P:
        if nama == "XGBoost":
            thr = thr_x
        else:
            # Geser ambang sampai FN di validation menyamai FN XGBoost di validation
            thr = ambang_untuk_fn(P[nama][val_m], y[val_m], cm_x_val["fn"])
        cmv = confusion(y[val_m], P[nama][val_m] >= thr)
        cmt = confusion(y[test_m], P[nama][test_m] >= thr)
        hasil[nama] = {"threshold": thr,
                       "validation": {k: cmv[k] for k in ("fp", "fn")},
                       "test": {k: cmt[k] for k in ("tn", "fp", "fn", "tp")},
                       "metrik_test": {k: cmt[k] for k in
                                       ("precision", "recall", "f1", "fpr", "fnr")}}
        print(f"  {nama:<16} val FP {cmv['fp']:>4} FN {cmv['fn']:>4}  |  "
              f"test FP {cmt['fp']:>4} FN {cmt['fn']:>4} total {cmt['fp'] + cmt['fn']:>4}")

    g, x = hasil["GAT v5"]["test"], hasil["XGBoost"]["test"]
    print(f"\n  Pada FN validation yang sama, di test GAT mencatat FP {g['fp']} "
          f"berbanding {x['fp']} milik XGBoost "
          f"({'unggul' if g['fp'] < x['fp'] else 'tidak unggul'}), "
          f"dan FN {g['fn']} berbanding {x['fn']} "
          f"({'unggul' if g['fn'] < x['fn'] else 'tidak unggul'}).")

    # ── Gambar kurva ──
    # Dibatasi pada wilayah operasi yang relevan. Pada rentang penuh seluruh
    # kurva menumpuk di dasar sumbu dan selisih antar-metode tidak terbaca:
    # FN melonjak ke ribuan begitu FP ditekan di bawah 30.
    fig, ax = plt.subplots(figsize=(9.5, 6.0), facecolor=SURFACE)
    for nama, p in P.items():
        fp, fn = kurva(p[test_m], y[test_m])
        sel = (fp >= 35) & (fp <= 140)
        ax.plot(fp[sel], fn[sel], color=AKSEN[nama], linewidth=2.4, label=nama)
    for nama, h in hasil.items():
        t = h["test"]
        if 35 <= t["fp"] <= 140:
            ax.scatter([t["fp"]], [t["fn"]], color=AKSEN[nama],
                       s=110, zorder=5, edgecolor="white", linewidth=1.8)
            ax.annotate(f"{t['fp']} / {t['fn']}", (t["fp"], t["fn"]),
                        textcoords="offset points", xytext=(0, 13), ha="center",
                        fontsize=9.5, fontweight="bold", color=AKSEN[nama])
    ax.set_xlim(35, 140)
    ax.set_ylim(65, 145)

    ax.set_xlabel("False Positive", fontsize=11.5, color=INK_SOFT)
    ax.set_ylabel("False Negative", fontsize=11.5, color=INK_SOFT)
    ax.set_title("Kurva Trade-off FP–FN pada Test Set",
                 fontsize=14.5, fontweight="bold", color=INK, pad=14)
    ax.text(0.5, 1.015, "Semakin dekat ke kiri-bawah semakin baik; titik bulat = titik operasi terpilih (FP / FN)",
            transform=ax.transAxes, ha="center", fontsize=9.5, color=INK_SOFT)
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d8d6d1")
    ax.tick_params(colors=INK_SOFT, length=0)
    ax.grid(color="#e8e6e1", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=10.5, loc="upper right")
    ax.set_axisbelow(True)
    fig.tight_layout()
    os.makedirs(os.path.dirname(IMG), exist_ok=True)
    fig.savefig(IMG, dpi=160, facecolor=SURFACE)
    plt.close(fig)

    with open(C.resolve("ringkasan_perbandingan.json"), "w", encoding="utf-8") as f:
        json.dump({"kurva_tradeoff_test": baris_kurva,
                   "titik_disamakan_di_validation": hasil}, f, indent=2, ensure_ascii=False)
    print(f"\n  Gambar : {os.path.relpath(IMG, PROJECT_ROOT)}")
    print("  Angka  : data/results/ringkasan_perbandingan.json")


if __name__ == "__main__":
    main()
