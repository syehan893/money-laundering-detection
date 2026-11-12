"""
scripts/baseline/confusion_matrix_perbandingan.py — Confusion matrix tiga metode
=================================================================================
Menyajikan confusion matrix Random Forest, XGBoost, dan GAT v5 berdampingan pada
sumbu yang sama, sehingga selisihnya terbaca langsung tanpa perlu membandingkan
tiga gambar terpisah.

Ketiganya dievaluasi pada test set yang sama dengan ambang yang sama-sama dipilih
di validation memakai kriteria minimum FP+FN.

Keluaran:
  - docs/images/confusion_matrix_perbandingan.png
  - docs/images/perbandingan_metode_bar.png

Usage:
    python scripts/baseline/confusion_matrix_perbandingan.py
"""

from __future__ import annotations

import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np


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

SEQ_BLUE = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"]
INK = "#0b0b0b"
INK_SOFT = "#52514e"
SURFACE = "#fcfcfb"
# Satu warna beraksen per metode, dipakai konsisten di kedua gambar
AKSEN = {"Random Forest": "#b8733a", "XGBoost": "#7d6ba8",
         "GAT v5 (ensemble)": "#256abf", "GAT v6 (ensemble beragam)": "#104281"}

IMG_CM = os.path.join(PROJECT_ROOT, "docs", "images", "confusion_matrix_perbandingan.png")
IMG_BAR = os.path.join(PROJECT_ROOT, "docs", "images", "perbandingan_metode_bar.png")


def _id(n) -> str:
    """Pemisah ribuan gaya Indonesia."""
    return f"{int(n):,}".replace(",", ".")


def _pct(x: float, digits: int = 2) -> str:
    return f"{x * 100:.{digits}f}".replace(".", ",") + "%"


def _dec(x: float, digits: int = 4) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def _nama_pendek(nama: str) -> str:
    """Nama metode tanpa keterangan cara penggabungan, untuk label gambar."""
    return (nama.replace(" (ensemble beragam)", "")
                .replace(" (ensemble 3 seed)", "")
                .replace(" (ensemble)", ""))


def panel(ax_cm, ax_txt, cm: dict, mt: dict, nama: str, warna: str) -> None:
    """Satu kolom: confusion matrix di atas, metrik ringkas di bawah."""
    counts = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]], dtype=float)
    # Dinormalisasi per baris; tanpa itu sel FP/FN tak terlihat di samping TN
    # yang bernilai ratusan ribu.
    rate = counts / counts.sum(axis=1, keepdims=True)

    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    ax_cm.imshow(rate, cmap=cmap, vmin=0.0, vmax=1.0, aspect="equal")
    for i in range(2):
        for j in range(2):
            gelap = rate[i, j] > 0.55
            ax_cm.text(j, i - 0.12, _id(counts[i, j]), ha="center", va="center",
                       fontsize=13, fontweight="bold",
                       color="#ffffff" if gelap else INK)
            ax_cm.text(j, i + 0.18, _pct(rate[i, j]), ha="center", va="center",
                       fontsize=8.5, color="#e8eef8" if gelap else INK_SOFT)

    ax_cm.set_xticks([0, 1], ["Prediksi\nNormal", "Prediksi\nLaundering"], fontsize=9)
    ax_cm.set_yticks([0, 1], ["Aktual\nNormal", "Aktual\nLaundering"], fontsize=9)
    # Keterangan cara penggabungan dibuang dari judul panel; hal itu dijelaskan
    # di dokumen, dan di gambar hanya membuat judul panjang tanpa menambah makna.
    judul = _nama_pendek(nama)
    ax_cm.set_title(judul, fontsize=12.5, fontweight="bold", color=warna, pad=10)
    for s in ax_cm.spines.values():
        s.set_visible(False)
    ax_cm.tick_params(length=0, colors=INK_SOFT)

    ax_txt.axis("off")
    baris = [("FP + FN", _id(cm["fp"] + cm["fn"])), ("Precision", _dec(mt["precision"])),
             ("Recall", _dec(mt["recall"])), ("F1-Score", _dec(mt["f1"])),
             ("FPR", _pct(mt["fpr"], 3)), ("FNR", _pct(mt["fnr"], 3))]
    for idx, (label, nilai) in enumerate(baris):
        y = 0.92 - idx * 0.165
        tebal = "bold" if idx == 0 else "normal"
        ax_txt.text(0.04, y, label, fontsize=10, color=INK_SOFT, va="center")
        ax_txt.text(0.96, y, nilai, fontsize=10.5, color=warna if idx == 0 else INK,
                    va="center", ha="right", fontweight="bold" if idx == 0 else tebal)


def gambar_confusion(hasil: dict) -> None:
    n = len(hasil)
    fig = plt.figure(figsize=(min(4.6, 18.0 / n) * n, 6.8), facecolor=SURFACE)
    gs = fig.add_gridspec(2, n, height_ratios=[2.4, 1], hspace=0.28, wspace=0.24)
    for k, (nama, h) in enumerate(hasil.items()):
        ax_cm = fig.add_subplot(gs[0, k])
        ax_txt = fig.add_subplot(gs[1, k])
        ax_cm.set_facecolor(SURFACE)
        ax_txt.set_facecolor(SURFACE)
        panel(ax_cm, ax_txt, h["confusion_matrix"], h["metrik"], nama,
              AKSEN.get(nama, INK))

    fig.suptitle(f"Perbandingan {n} Metode pada Test Set yang Sama",
                 fontsize=15.5, fontweight="bold", color=INK, y=0.985)
    fig.text(0.5, 0.925,
             "489.137 transaksi, fitur dan split identik, ambang sama-sama dipilih "
             "di validation dengan kriteria minimum FP+FN",
             ha="center", fontsize=10, color=INK_SOFT)
    fig.subplots_adjust(top=0.845, bottom=0.05, left=0.06, right=0.97)
    os.makedirs(os.path.dirname(IMG_CM), exist_ok=True)
    fig.savefig(IMG_CM, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def gambar_batang(hasil: dict) -> None:
    """
    Batang FP dan FN berdampingan.

    Skala linear dari nol, bukan logaritmik: seluruh nilai berkumpul di rentang
    sempit (puluhan), sehingga skala log hanya memotong batang di bawah dan
    memaksa sumbu menampilkan notasi ilmiah. Memulai sumbu dari nol juga menjaga
    perbandingan tetap jujur -- memotong sumbu akan melebih-lebihkan selisih
    yang sebenarnya hanya beberapa satuan.
    """
    nama = list(hasil.keys())
    fp = [hasil[n]["confusion_matrix"]["fp"] for n in nama]
    fn = [hasil[n]["confusion_matrix"]["fn"] for n in nama]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), facecolor=SURFACE)
    for ax, nilai, judul in zip(axes, (fp, fn), ("False Positive", "False Negative")):
        warna = [AKSEN.get(n, INK) for n in nama]
        bar = ax.bar(range(len(nama)), nilai, color=warna, width=0.58)
        ax.set_ylim(0, max(nilai) * 1.22)
        ax.set_title(judul, fontsize=13, fontweight="bold", color=INK, pad=10)
        ax.set_xticks(range(len(nama)), [_nama_pendek(n) for n in nama], fontsize=10.5)
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_color("#d8d6d1")
        ax.spines["bottom"].set_color("#d8d6d1")
        ax.tick_params(colors=INK_SOFT, length=0)
        ax.grid(axis="y", color="#e8e6e1", linewidth=0.8)
        ax.set_axisbelow(True)
        for b, v in zip(bar, nilai):
            ax.text(b.get_x() + b.get_width() / 2, v + max(nilai) * 0.03, _id(v),
                    ha="center", fontsize=11, fontweight="bold", color=INK)

    fig.suptitle("Jumlah Kesalahan per Metode",
                 fontsize=14.5, fontweight="bold", color=INK, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(IMG_BAR, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    with open(C.resolve("baseline_comparison.json"), encoding="utf-8") as f:
        hasil = json.load(f)["perbandingan"]

    gambar_confusion(hasil)
    gambar_batang(hasil)

    print("=" * 74)
    print("  Perbandingan tiga metode (test set 489.137 transaksi)")
    print("=" * 74)
    print(f"  {'metode':<26}{'FP':>7}{'FN':>7}{'total':>8}{'FPR':>10}{'FNR':>10}{'F1':>9}")
    for nama, h in hasil.items():
        cm, mt = h["confusion_matrix"], h["metrik"]
        print(f"  {nama:<26}{cm['fp']:>7}{cm['fn']:>7}{cm['fp'] + cm['fn']:>8}"
              f"{_pct(mt['fpr'], 3):>10}{_pct(mt['fnr'], 3):>10}{_dec(mt['f1']):>9}")
    print(f"\n  Gambar : {os.path.relpath(IMG_CM, PROJECT_ROOT)}")
    print(f"           {os.path.relpath(IMG_BAR, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
