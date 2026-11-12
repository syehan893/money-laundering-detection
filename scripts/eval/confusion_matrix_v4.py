"""
scripts/confusion_matrix_v4.py — Gambar hasil evaluasi model v4
==================================================================
Menghasilkan gambar siap pakai untuk naskah skripsi berdasarkan model dan data
terbaru (dataset v4: lima pola sesuai definisi BAB II, atribut realistis).

Keluaran:
  - docs/images/confusion_matrix_v4.png       confusion matrix + metrik
  - docs/images/recall_per_pola_v4.png        recall tiap pola
  - docs/images/tradeoff_fpr_fnr_v4.png       kurva trade-off FPR vs FNR
  - data/evaluation_v4.json                   seluruh angka dalam bentuk JSON

Usage:
    python scripts/confusion_matrix_v4.py
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import backend.config as C

# Palet mengikuti skill dataviz: ramp biru sekuensial + slot kategorikal
SEQ_BLUE = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"]
SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#e5e4e0"

IMG_DIR = os.path.join(PROJECT_ROOT, "docs", "images")
OUT_JSON = C.resolve("evaluation_v4.json")

POLA_LABEL = {
    "Fan_Out": "Fan-Out",
    "Fan_In": "Fan-In",
    "Scatter_Gather": "Scatter-Gather\n(Smurfing)",
    "Gather_Scatter": "Gather-Scatter\n(Mixer)",
    "Simple_Cycle": "Simple Cycle\n(Round-Tripping)",
}


def _id(n) -> str:
    """Pemisah ribuan gaya Indonesia."""
    return f"{int(n):,}".replace(",", ".")


def _pct(x: float, digits: int = 2) -> str:
    return f"{x*100:.{digits}f}".replace(".", ",") + "%"


def load() -> tuple:
    """Muat probabilitas uji, label, dan metrik model v4."""
    probs_path = C.resolve("test_probs_v4.npy")
    for path in (probs_path, C.METRICS_V4_PATH, C.TEST_CSV_V4_PATH):
        if not os.path.exists(path):
            raise SystemExit(f"Berkas belum ada: {path}\n"
                             f"Jalankan `python -m backend.ml.train --version v4` lebih dulu.")
    probs = np.load(probs_path)
    with open(C.METRICS_V4_PATH, encoding="utf-8") as f:
        metrics = json.load(f)
    test = pd.read_csv(C.TEST_CSV_V4_PATH, usecols=["Is_laundering", "Laundering_type"])
    return probs, test["Is_laundering"].values, test["Laundering_type"].values, metrics


def draw_confusion(cm: dict, rates: dict, threshold: float, n_total: int) -> None:
    """Confusion matrix 2x2 dengan metrik turunannya."""
    counts = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]], dtype=float)
    # Warna dinormalisasi per baris; tanpa itu sel FP/FN yang bernilai ratusan
    # tidak akan terlihat di samping TN yang bernilai ratusan ribu.
    row_rate = counts / counts.sum(axis=1, keepdims=True)

    fig = plt.figure(figsize=(11.5, 5.4), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1], wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    ax_txt = fig.add_subplot(gs[0, 1])
    for a in (ax, ax_txt):
        a.set_facecolor(SURFACE)

    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    ax.imshow(row_rate, cmap=cmap, vmin=0.0, vmax=1.0, aspect="equal")

    for i in range(2):
        for j in range(2):
            dark = row_rate[i, j] > 0.55
            ax.text(j, i - 0.11, _id(counts[i, j]), ha="center", va="center",
                    fontsize=17, fontweight="bold", color="#ffffff" if dark else INK)
            ax.text(j, i + 0.17, _pct(row_rate[i, j]), ha="center", va="center",
                    fontsize=10, color="#e8eef8" if dark else INK_SOFT)

    ax.set_xticks([0, 1], ["Prediksi\nNormal", "Prediksi\nLaundering"], fontsize=10.5)
    ax.set_yticks([0, 1], ["Aktual\nNormal", "Aktual\nLaundering"], fontsize=10.5)
    ax.set_title("Confusion Matrix", fontsize=13.5, fontweight="bold", color=INK, pad=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, colors=INK_SOFT)

    # Panel metrik
    ax_txt.axis("off")
    rows = [
        ("Precision", f"{rates['precision']:.4f}".replace(".", ",")),
        ("Recall", f"{rates['recall']:.4f}".replace(".", ",")),
        ("F1-Score", f"{rates['f1']:.4f}".replace(".", ",")),
        ("Accuracy", _pct(rates["accuracy"], 2)),
        ("", ""),
        ("False Positive Rate", _pct(rates["fpr"], 4)),
        ("False Negative Rate", _pct(rates["fnr"], 4)),
        ("", ""),
        ("Threshold", f"{threshold:.4f}".replace(".", ",")),
        ("Transaksi diuji", _id(n_total)),
    ]
    y = 0.95
    for label, value in rows:
        if label:
            ax_txt.text(0.02, y, label, fontsize=11.5, color=INK_SOFT, va="top")
            ax_txt.text(0.98, y, value, fontsize=11.5, color=INK, va="top",
                        ha="right", fontweight="bold")
        y -= 0.095
    ax_txt.set_title("Metrik Evaluasi", fontsize=13.5, fontweight="bold", color=INK, pad=12)

    fig.suptitle("Hasil Evaluasi Model GAT pada Data Uji",
                 fontsize=15, fontweight="bold", color=INK, y=0.99)
    fig.text(0.5, 0.915,
             "Dataset v4 — lima pola pencucian uang dengan atribut mengikuti distribusi transaksi normal",
             ha="center", fontsize=10.5, color=INK_SOFT)
    fig.subplots_adjust(top=0.80, bottom=0.10, left=0.07, right=0.96)
    fig.savefig(os.path.join(IMG_DIR, "confusion_matrix_v4.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)


def draw_recall_per_pattern(per_pattern: list) -> None:
    """Recall tiap pola, diurutkan dari tertinggi."""
    rows = sorted(per_pattern, key=lambda r: -r["recall"])
    labels = [POLA_LABEL.get(r["pola"], r["pola"]) for r in rows]
    values = [r["recall"] * 100 for r in rows]

    fig, ax = plt.subplots(figsize=(10.5, 5.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    y = np.arange(len(rows))
    ax.barh(y, values, height=0.58, color=SERIES_1)

    for i, r in enumerate(rows):
        text = f"{r['recall']*100:.1f}%".replace(".", ",")
        detail = f"({_id(r['terdeteksi'])} dari {_id(r['jumlah'])})"
        # Label ditaruh di dalam bar bila muat, agar tidak terpotong tepi kanan
        if values[i] > 55:
            ax.text(values[i] - 1.5, i, f"{text}  {detail}", va="center", ha="right",
                    fontsize=10.5, color="#ffffff", fontweight="bold")
        else:
            ax.text(values[i] + 1.5, i, f"{text}  {detail}", va="center", ha="left",
                    fontsize=10.5, color=INK, fontweight="bold")

    ax.set_yticks(y, labels, fontsize=10.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 104)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"], fontsize=10)
    ax.set_xlabel("Recall — proporsi transaksi laundering yang terdeteksi",
                  fontsize=11, color=INK_SOFT)
    ax.set_title("Tingkat Deteksi per Pola Pencucian Uang",
                 fontsize=14, fontweight="bold", color=INK, pad=28)
    ax.text(0.5, 1.03, "Kelima pola yang menjadi lingkup penelitian, diuji pada data yang "
                       "atributnya tidak membedakan kelas",
            transform=ax.transAxes, ha="center", fontsize=10.5, color=INK_SOFT)

    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.yaxis.grid(False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(length=0, colors=INK_SOFT)

    fig.subplots_adjust(top=0.80, bottom=0.13, left=0.19, right=0.97)
    fig.savefig(os.path.join(IMG_DIR, "recall_per_pola_v4.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)


def draw_tradeoff(curve: list) -> None:
    """Kurva trade-off: FNR sebagai fungsi anggaran FPR."""
    fpr = [c["fpr"] * 100 for c in curve]
    fnr = [c["fnr"] * 100 for c in curve]

    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    ax.plot(fpr, fnr, color=SERIES_1, linewidth=2, marker="o", markersize=7,
            markerfacecolor=SERIES_1, markeredgecolor=SURFACE, markeredgewidth=2)

    for c in curve:
        if abs(c["fpr"] - 0.001) < 2e-4 or abs(c["fpr"] - 0.0002) < 1e-4:
            ax.annotate(
                f"FPR {_pct(c['fpr'], 3)} → FNR {_pct(c['fnr'], 2)}",
                xy=(c["fpr"] * 100, c["fnr"] * 100), xytext=(12, 10),
                textcoords="offset points", fontsize=10.5, color=INK, fontweight="bold")

    ax.axvline(0.1, color=SERIES_2, linewidth=1.5)
    ax.annotate("target FPR 0,1%", xy=(0.1, max(fnr) * 0.9), xytext=(7, 0),
                textcoords="offset points", fontsize=10.5, color=SERIES_2, fontweight="bold")

    ax.set_xlabel("False Positive Rate (%)", fontsize=11, color=INK_SOFT)
    ax.set_ylabel("False Negative Rate (%)", fontsize=11, color=INK_SOFT)
    ax.set_title("Trade-off antara False Positive dan False Negative",
                 fontsize=14, fontweight="bold", color=INK, pad=26)
    ax.text(0.5, 1.03, "Menurunkan satu jenis kesalahan selalu menaikkan yang lain; "
                       "titik operasi dipilih sesuai kapasitas investigasi",
            transform=ax.transAxes, ha="center", fontsize=10.5, color=INK_SOFT)

    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_SOFT, length=0)

    fig.subplots_adjust(top=0.79, bottom=0.13, left=0.09, right=0.97)
    fig.savefig(os.path.join(IMG_DIR, "tradeoff_fpr_fnr_v4.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    os.makedirs(IMG_DIR, exist_ok=True)
    probs, labels, types, metrics = load()
    threshold = float(metrics["optimal_threshold"])
    pred = probs >= threshold

    tp = int((pred & (labels == 1)).sum())
    tn = int((~pred & (labels == 0)).sum())
    fp = int((pred & (labels == 0)).sum())
    fn = int((~pred & (labels == 1)).sum())
    cm = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    rates = {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "accuracy": (tp + tn) / len(labels),
        "fpr": fp / max(fp + tn, 1),
        "fnr": fn / max(fn + tp, 1),
    }

    per_pattern = []
    for name in sorted(set(types[labels == 1])):
        sel = (labels == 1) & (types == name)
        total = int(sel.sum())
        hit = int((pred & sel).sum())
        per_pattern.append({"pola": name, "jumlah": total, "terdeteksi": hit,
                            "recall": hit / max(total, 1)})

    neg_sorted = np.sort(probs[labels == 0])[::-1]
    n_neg, n_pos = len(neg_sorted), int((labels == 1).sum())
    curve = []
    for budget in (0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01):
        k = min(int(n_neg * budget), n_neg - 1)
        thr_b = float(neg_sorted[k])
        pr = probs >= thr_b
        b_fp = int((pr & (labels == 0)).sum())
        b_fn = int((~pr & (labels == 1)).sum())
        curve.append({"anggaran": budget, "threshold": thr_b, "fp": b_fp, "fn": b_fn,
                      "fpr": b_fp / n_neg, "fnr": b_fn / n_pos,
                      "recall": (n_pos - b_fn) / n_pos})

    draw_confusion(cm, rates, threshold, len(labels))
    draw_recall_per_pattern(per_pattern)
    draw_tradeoff(curve)

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"versi": "v4", "threshold": threshold, "confusion_matrix": cm,
                   "metrik": rates, "per_pola": per_pattern, "trade_off": curve,
                   "epoch_terbaik": metrics.get("best_epoch"),
                   "hyperparameter": metrics.get("hyperparameters")},
                  f, indent=2, ensure_ascii=False)

    # ── Ringkasan terminal ──
    print("=" * 72)
    print("  Evaluasi Model v4 — Data Uji")
    print("=" * 72)
    print(f"\n  Threshold {threshold:.4f} | {_id(len(labels))} transaksi diuji\n")
    print(f"    TN {_id(tn):>9}   FP {_id(fp):>6}")
    print(f"    FN {_id(fn):>9}   TP {_id(tp):>6}\n")
    print(f"    Precision {rates['precision']:.4f}   Recall {rates['recall']:.4f}   "
          f"F1 {rates['f1']:.4f}")
    print(f"    FPR {_pct(rates['fpr'], 4)}   FNR {_pct(rates['fnr'], 4)}   "
          f"Akurasi {_pct(rates['accuracy'], 2)}")

    print(f"\n  {'pola':<18}{'jumlah':>9}{'terdeteksi':>12}{'recall':>9}")
    for r in sorted(per_pattern, key=lambda x: -x["recall"]):
        print(f"  {r['pola']:<18}{_id(r['jumlah']):>9}{_id(r['terdeteksi']):>12}"
              f"{r['recall']*100:>8.1f}%")

    print(f"\n  {'anggaran FPR':>13}{'FP':>8}{'FN':>8}{'FNR':>10}{'recall':>9}")
    for c in curve:
        print(f"  {c['anggaran']*100:>12.2f}%{_id(c['fp']):>8}{_id(c['fn']):>8}"
              f"{c['fnr']*100:>9.3f}%{c['recall']*100:>8.1f}%")

    print(f"\n  Gambar : {IMG_DIR}")
    print(f"  Angka  : {OUT_JSON}")


if __name__ == "__main__":
    main()
