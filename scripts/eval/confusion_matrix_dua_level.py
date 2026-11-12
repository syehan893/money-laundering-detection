"""
scripts/confusion_matrix_dua_level.py — Confusion matrix tingkat transaksi & akun
===================================================================================
Model mengklasifikasikan **transaksi** (edge pada graf), sehingga confusion matrix
utamanya bersatuan transaksi. Namun analis AML bekerja pada tingkat **akun**:
yang diinvestigasi adalah rekeningnya, bukan satu transaksi terpisah.

Skrip ini menyajikan keduanya berdampingan agar perbedaannya jelas.

Aturan agregasi ke tingkat akun
--------------------------------
  Label sebenarnya : akun dianggap terlibat pencucian uang bila ikut serta dalam
                     minimal satu transaksi berlabel laundering.
  Prediksi model   : akun ditandai bila minimal satu transaksinya ditandai model.

Aturan "minimal satu" dipilih karena sesuai cara kerja sistem peringatan AML —
satu transaksi mencurigakan sudah cukup memunculkan alert atas rekening tersebut.
Sensitivitas terhadap aturan lain (minimal dua, minimal tiga) ikut dilaporkan.

Keluaran:
  - docs/images/confusion_matrix_dua_level.png
  - data/evaluation_dua_level.json

Usage:
    python scripts/confusion_matrix_dua_level.py
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

SEQ_BLUE = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"]
INK = "#0b0b0b"
INK_SOFT = "#52514e"
SURFACE = "#fcfcfb"

# Versi dipilih lewat --version; v5 memakai probabilitas ensemble.
SUMBER = {
    "v4": {"probs": "test_probs_v4.npy", "eval": "evaluation_v4.json",
           "img": "confusion_matrix_dua_level.png", "json": "evaluation_dua_level.json"},
    "v5": {"probs": "test_probs_v5_ensemble.npy", "eval": "evaluation_v5.json",
           "img": "confusion_matrix_dua_level_v5.png", "json": "evaluation_dua_level_v5.json"},
}


def _id(n) -> str:
    """Pemisah ribuan gaya Indonesia."""
    return f"{int(n):,}".replace(",", ".")


def _pct(x: float, digits: int = 2) -> str:
    return f"{x*100:.{digits}f}".replace(".", ",") + "%"


def _dec(x: float, digits: int = 4) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Confusion matrix beserta metrik turunannya."""
    tp = int((y_pred & y_true).sum())
    fp = int((y_pred & ~y_true).sum())
    fn = int((~y_pred & y_true).sum())
    tn = int((~y_pred & ~y_true).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
        "fpr": fp / max(fp + tn, 1),
        "fnr": fn / max(fn + tp, 1),
        "total": tp + tn + fp + fn,
    }


def draw_panel(ax_cm, ax_txt, cm: dict, title: str, subtitle: str, unit: str) -> None:
    """Satu panel: confusion matrix + daftar metrik di bawahnya."""
    counts = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]], dtype=float)
    # Warna dinormalisasi per baris; tanpa itu sel FP/FN tidak terlihat
    # di samping TN yang bernilai ratusan ribu.
    rate = counts / counts.sum(axis=1, keepdims=True)

    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    ax_cm.imshow(rate, cmap=cmap, vmin=0.0, vmax=1.0, aspect="equal")

    for i in range(2):
        for j in range(2):
            dark = rate[i, j] > 0.55
            ax_cm.text(j, i - 0.12, _id(counts[i, j]), ha="center", va="center",
                       fontsize=15, fontweight="bold", color="#ffffff" if dark else INK)
            ax_cm.text(j, i + 0.18, _pct(rate[i, j]), ha="center", va="center",
                       fontsize=9.5, color="#e8eef8" if dark else INK_SOFT)

    ax_cm.set_xticks([0, 1], [f"Prediksi\nNormal", f"Prediksi\nLaundering"], fontsize=10)
    ax_cm.set_yticks([0, 1], [f"Aktual\nNormal", f"Aktual\nLaundering"], fontsize=10)
    # Judul panel di atas, penjelasannya tepat di bawahnya agar tidak
    # berdesakan dengan subjudul gambar
    ax_cm.set_title(title, fontsize=13.5, fontweight="bold", color=INK, pad=30)
    ax_cm.text(0.5, 1.055, subtitle, transform=ax_cm.transAxes, ha="center",
               fontsize=10.5, color=INK_SOFT)
    for spine in ax_cm.spines.values():
        spine.set_visible(False)
    ax_cm.tick_params(length=0, colors=INK_SOFT)

    ax_txt.axis("off")
    rows = [
        ("Precision", _dec(cm["precision"])),
        ("Recall", _dec(cm["recall"])),
        ("F1-Score", _dec(cm["f1"])),
        ("Accuracy", _pct(cm["accuracy"], 2)),
        ("FPR", _pct(cm["fpr"], 3)),
        ("FNR", _pct(cm["fnr"], 3)),
        (f"Total {unit}", _id(cm["total"])),
    ]
    x_pairs = [(0.04, 0.46), (0.54, 0.96)]
    for idx, (label, value) in enumerate(rows):
        col, row = idx % 2, idx // 2
        xl, xr = x_pairs[col]
        y = 0.88 - row * 0.235
        ax_txt.text(xl, y, label, fontsize=11, color=INK_SOFT, va="center")
        ax_txt.text(xr, y, value, fontsize=11, color=INK, va="center",
                    ha="right", fontweight="bold")


def main(versi: str = "v4") -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    src = SUMBER[versi]
    img_path = os.path.join(PROJECT_ROOT, "docs", "images", src["img"])
    json_path = C.resolve(src["json"])
    probs = np.load(C.resolve(src["probs"]))
    with open(C.resolve(src["eval"]), encoding="utf-8") as f:
        threshold = json.load(f)["threshold"]

    test = pd.read_csv(C.TEST_CSV_V4_PATH,
                       usecols=["Sender_account", "Receiver_account", "Is_laundering"])
    test["flag"] = (probs >= threshold).astype(int)

    # ── Tingkat transaksi ──
    cm_tx = confusion(test["Is_laundering"].values.astype(bool),
                      test["flag"].values.astype(bool))

    # ── Tingkat akun ──
    # Tiap transaksi menyumbang ke dua akun: pengirim dan penerima.
    long = pd.concat([
        test[["Sender_account", "Is_laundering", "flag"]].rename(
            columns={"Sender_account": "akun"}),
        test[["Receiver_account", "Is_laundering", "flag"]].rename(
            columns={"Receiver_account": "akun"}),
    ])
    per_akun = long.groupby("akun").agg(
        n_tx=("flag", "size"), n_laundering=("Is_laundering", "sum"), n_flag=("flag", "sum"))

    y_akun = (per_akun["n_laundering"] > 0).values
    cm_akun = confusion(y_akun, (per_akun["n_flag"] >= 1).values)

    # Sensitivitas aturan agregasi
    sensitivitas = []
    for k in (1, 2, 3):
        c = confusion(y_akun, (per_akun["n_flag"] >= k).values)
        c["aturan"] = f"minimal {k} transaksi ditandai"
        sensitivitas.append(c)

    # ── Gambar ──
    fig = plt.figure(figsize=(13.5, 6.6), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, height_ratios=[2.5, 1], hspace=0.30, wspace=0.22)
    panels = [
        (gs[0, 0], gs[1, 0], cm_tx, "Tingkat Transaksi",
         "Satuan analisis: setiap transaksi (edge pada graf)", "transaksi"),
        (gs[0, 1], gs[1, 1], cm_akun, "Tingkat Akun",
         "Akun ditandai bila minimal satu transaksinya ditandai", "akun"),
    ]
    for cell_cm, cell_txt, cm, title, subtitle, unit in panels:
        ax_cm = fig.add_subplot(cell_cm)
        ax_txt = fig.add_subplot(cell_txt)
        ax_cm.set_facecolor(SURFACE)
        ax_txt.set_facecolor(SURFACE)
        draw_panel(ax_cm, ax_txt, cm, title, subtitle, unit)

    fig.suptitle(f"Confusion Matrix pada Dua Satuan Analisis ({versi})",
                 fontsize=15.5, fontweight="bold", color=INK, y=0.985)
    fig.text(0.5, 0.925,
             "Model memberi skor per transaksi; hasilnya diagregasi ke tingkat akun "
             "untuk kebutuhan investigasi",
             ha="center", fontsize=10.5, color=INK_SOFT)
    fig.subplots_adjust(top=0.825, bottom=0.05, left=0.07, right=0.96)
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    fig.savefig(img_path, dpi=160, facecolor=SURFACE)
    plt.close(fig)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"threshold": threshold, "tingkat_transaksi": cm_tx,
                   "tingkat_akun": cm_akun, "sensitivitas_aturan": sensitivitas},
                  f, indent=2, ensure_ascii=False)

    # ── Ringkasan terminal ──
    print("=" * 74)
    print("  Confusion Matrix — Dua Satuan Analisis")
    print("=" * 74)
    for name, cm, unit in [("TINGKAT TRANSAKSI", cm_tx, "transaksi"),
                           ("TINGKAT AKUN", cm_akun, "akun")]:
        print(f"\n  {name}  ({_id(cm['total'])} {unit})")
        print(f"    TN {_id(cm['tn']):>9}   FP {_id(cm['fp']):>6}")
        print(f"    FN {_id(cm['fn']):>9}   TP {_id(cm['tp']):>6}")
        print(f"    Precision {_dec(cm['precision'])}  Recall {_dec(cm['recall'])}  "
              f"F1 {_dec(cm['f1'])}")
        print(f"    FPR {_pct(cm['fpr'], 3)}  FNR {_pct(cm['fnr'], 3)}  "
              f"Akurasi {_pct(cm['accuracy'], 2)}")

    print(f"\n  Sensitivitas aturan agregasi ke tingkat akun:")
    print(f"    {'aturan':<28}{'FP':>8}{'FN':>7}{'Precision':>11}{'Recall':>9}")
    for c in sensitivitas:
        print(f"    {c['aturan']:<28}{_id(c['fp']):>8}{_id(c['fn']):>7}"
              f"{_dec(c['precision']):>11}{_dec(c['recall']):>9}")

    print(f"\n  Gambar : {img_path}")
    print(f"  Angka  : {json_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Confusion matrix tingkat transaksi & akun.")
    parser.add_argument("--version", choices=["v4", "v5"], default="v4")
    main(parser.parse_args().version)
