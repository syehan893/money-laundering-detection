"""
scripts/operating_points.py — Peta titik operasi model v2
============================================================
Menjawab pertanyaan: "bisakah FPR dan FNR ditekan di bawah 0,05%?"

Menghitung tiga cara pengoperasian model dan biaya masing-masing:

  A. Satu threshold biner — satu garis keputusan untuk semua transaksi.
  B. Tiga zona — auto-clear / tinjau manual / auto-flag, seperti sistem AML
     sungguhan. FP dan FN hanya dihitung pada keputusan otomatis; zona tengah
     diteruskan ke analis.
  C. Pengukuran terbatas pada populasi pola sintetis (5 tipologi yang
     disuntikkan), memisahkan 283 kasus asli SAML-D yang secara terukur
     tidak terpisahkan (lihat docs/docs md/PENINGKATAN_MODEL_V2.md §4).

Output:
  - data/operating_points.json
  - docs/images/operating_points.png

Usage:
    python scripts/operating_points.py
"""

import json
import os
import sys
import warnings
from typing import Dict, List, Tuple

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

SERIES_1 = "#2a78d6"   # volume tinjauan manual
SERIES_2 = "#eb6834"   # laundering yang masih terlewat
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#e5e4e0"

TARGET = 0.0005  # 0,05%

OUT_JSON = C.resolve("operating_points.json")
OUT_PNG = os.path.join(PROJECT_ROOT, "docs", "images", "operating_points.png")


def _id(n: int) -> str:
    """Format angka dengan pemisah ribuan gaya Indonesia."""
    return f"{n:,}".replace(",", ".")


def load_test_scores() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ambil (probabilitas, label, penanda tipologi langka) untuk test set."""
    data = torch.load(C.DATA_V2_PATH, weights_only=False)
    probs_path = C.resolve("test_probs_v2.npy")
    if not os.path.exists(probs_path):
        raise FileNotFoundError(
            f"{probs_path} tidak ada. Jalankan `python -m backend.ml.train` lebih dulu."
        )
    test = data.test_mask.numpy()
    return (
        np.load(probs_path),
        data.y.numpy().astype(int)[test],
        data.is_rare_positive.numpy()[test],
    )


def binary_point(probs: np.ndarray, labels: np.ndarray, threshold: float) -> Dict:
    """Metrik pada satu threshold biner."""
    pred = probs >= threshold
    tp = int((pred & (labels == 1)).sum())
    tn = int((~pred & (labels == 0)).sum())
    fp = int((pred & (labels == 0)).sum())
    fn = int((~pred & (labels == 1)).sum())
    return {
        "threshold": float(threshold),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "fpr": fp / max(fp + tn, 1),
        "fnr": fn / max(fn + tp, 1),
    }


def three_zone(
    probs: np.ndarray, labels: np.ndarray, max_fn: int, max_fp: int
) -> Dict:
    """
    Konfigurasi tiga zona dengan jaminan FN dan FP pada keputusan otomatis.

    Batas bawah diletakkan tepat di bawah skor positif ke-(max_fn+1) terendah,
    sehingga paling banyak `max_fn` laundering jatuh ke zona auto-clear.
    Batas atas diletakkan pada skor negatif ke-(max_fp+1) tertinggi.
    """
    pos_sorted = np.sort(probs[labels == 1])
    neg_sorted = np.sort(probs[labels == 0])[::-1]

    t_low = float(pos_sorted[min(max_fn, len(pos_sorted) - 1)]) - 1e-9
    t_high = float(neg_sorted[min(max_fp, len(neg_sorted) - 1)]) + 1e-9

    if t_low >= t_high:
        return {"feasible": False, "t_low": t_low, "t_high": t_high}

    auto_clear = probs < t_low
    review = (probs >= t_low) & (probs < t_high)
    auto_flag = probs >= t_high

    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    fn_auto = int((auto_clear & (labels == 1)).sum())
    fp_auto = int((auto_flag & (labels == 0)).sum())

    return {
        "feasible": True,
        "t_low": t_low,
        "t_high": t_high,
        "auto_clear_count": int(auto_clear.sum()),
        "review_count": int(review.sum()),
        "review_share": int(review.sum()) / len(probs),
        "review_laundering": int((review & (labels == 1)).sum()),
        "auto_flag_count": int(auto_flag.sum()),
        "fn_auto": fn_auto,
        "fp_auto": fp_auto,
        "fnr_auto": fn_auto / max(n_pos, 1),
        "fpr_auto": fp_auto / max(n_neg, 1),
    }


def render_figure(curve: List[Dict], n_total: int) -> None:
    """Gambar kurva biaya: volume tinjauan manual vs jaminan FNR."""
    fnr_targets = [c["fnr_guarantee"] * 100 for c in curve]
    review_share = [c["review_share"] * 100 for c in curve]

    fig, ax = plt.subplots(figsize=(10.5, 5.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    ax.plot(fnr_targets, review_share, color=SERIES_1, linewidth=2,
            marker="o", markersize=7, markerfacecolor=SERIES_1,
            markeredgecolor=SURFACE, markeredgewidth=2, label="Volume tinjauan manual")

    # Label selektif: titik target dan titik siku kurva
    for c in curve:
        x, yy = c["fnr_guarantee"] * 100, c["review_share"] * 100
        if c["fnr_guarantee"] in (0.0005, 0.002):
            ax.annotate(
                f"{yy:.2f}%  ({_id(c['review_count'])} transaksi)",
                xy=(x, yy), xytext=(12, 8), textcoords="offset points",
                fontsize=10, color=INK_PRIMARY, fontweight="bold",
            )

    ax.axvline(0.05, color=SERIES_2, linewidth=1.5, linestyle="-", alpha=0.85)
    ax.annotate("target FNR 0,05%", xy=(0.05, max(review_share) * 0.88),
                xytext=(8, 0), textcoords="offset points",
                fontsize=10, color=SERIES_2, fontweight="bold")

    ax.set_xlabel("Jaminan FNR pada keputusan otomatis (%)", fontsize=11, color=INK_SECONDARY)
    ax.set_ylabel("Transaksi yang harus ditinjau analis (%)", fontsize=11, color=INK_SECONDARY)
    ax.set_title("Biaya menekan false negative: volume tinjauan manual yang dibutuhkan",
                 fontsize=14, fontweight="bold", color=INK_PRIMARY, pad=40)
    ax.text(0.5, 1.025,
            f"Model v2, test set {_id(n_total)} transaksi, FPR otomatis dijaga di bawah 0,05%",
            transform=ax.transAxes, ha="center", fontsize=10.5, color=INK_SECONDARY)

    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_SECONDARY, length=0)
    ax.legend(frameon=False, fontsize=10, labelcolor=INK_SECONDARY, loc="upper right")

    fig.subplots_adjust(top=0.80, bottom=0.13, left=0.09, right=0.97)
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    probs, labels, is_rare = load_test_scores()
    n_pos, n_neg = int((labels == 1).sum()), int((labels == 0).sum())
    max_fp = int(n_neg * TARGET)
    max_fn = int(n_pos * TARGET)

    print(f"Test set: {n_pos:,} positif | {n_neg:,} negatif")
    print(f"Anggaran 0,05% -> FP maks {max_fp} | FN maks {max_fn}\n")

    results: Dict = {"target": TARGET, "n_pos": n_pos, "n_neg": n_neg}

    # ── A. Satu threshold biner ──
    print("=" * 78)
    print("A. SATU THRESHOLD BINER")
    print("=" * 78)
    neg_sorted = np.sort(probs[labels == 0])[::-1]
    points = {
        "fpr_0.05_persen": float(neg_sorted[max_fp]),
        "presisi_tinggi_0.90": 0.90,
        "titik_operasi_v2": json.load(open(C.METRICS_V2_PATH, encoding="utf-8"))["optimal_threshold"],
    }
    results["binary"] = {}
    print(f"  {'konfigurasi':<26}{'threshold':>11}{'FP':>7}{'FN':>7}{'FPR%':>10}{'FNR%':>10}")
    for name, thr in points.items():
        m = binary_point(probs, labels, thr)
        results["binary"][name] = m
        print(f"  {name:<26}{m['threshold']:>11.4f}{m['fp']:>7,}{m['fn']:>7,}"
              f"{m['fpr']*100:>10.4f}{m['fnr']*100:>10.4f}")
    print(f"\n  Untuk FNR < 0,05% dibutuhkan FN <= {max_fn}, artinya menangkap "
          f"{int((labels == 1) & is_rare).sum() - max_fn if False else (int(is_rare.sum()) - max_fn)}"
          f"/{int(is_rare.sum())} kasus asli SAML-D "
          f"({(int(is_rare.sum()) - max_fn) / int(is_rare.sum()) * 100:.0f}%).")
    print("  Batas atas keterpisahan kelas tersebut terukur ~20% -> tidak tercapai.")

    # ── B. Tiga zona ──
    print("\n" + "=" * 78)
    print("B. TIGA ZONA (auto-clear / tinjau manual / auto-flag)")
    print("=" * 78)
    print(f"  {'jaminan FNR':>12}{'FN':>6}{'FP':>6}{'tinjau manual':>22}{'laundering di zona tinjau':>28}")
    curve: List[Dict] = []
    for fnr_guarantee in [0.0, 0.0005, 0.001, 0.002, 0.003]:
        fn_allow = int(n_pos * fnr_guarantee)
        z = three_zone(probs, labels, fn_allow, max_fp)
        if not z["feasible"]:
            continue
        z["fnr_guarantee"] = fnr_guarantee
        curve.append(z)
        print(f"  {fnr_guarantee*100:>11.2f}%{z['fn_auto']:>6}{z['fp_auto']:>6}"
              f"{z['review_count']:>13,} ({z['review_share']*100:5.2f}%)"
              f"{z['review_laundering']:>20,}")
    results["three_zone"] = curve

    target_zone = next((c for c in curve if c["fnr_guarantee"] == TARGET), None)
    if target_zone:
        print(f"\n  Konfigurasi target (FNR & FPR otomatis < 0,05%):")
        print(f"    Auto-clear   p < {target_zone['t_low']:.4f} : {target_zone['auto_clear_count']:>8,} transaksi")
        print(f"    Tinjau manual                 : {target_zone['review_count']:>8,} transaksi "
              f"({target_zone['review_share']*100:.2f}%)")
        print(f"    Auto-flag    p >= {target_zone['t_high']:.4f}: {target_zone['auto_flag_count']:>8,} transaksi")
        print(f"    -> FNR otomatis {target_zone['fnr_auto']*100:.4f}% | "
              f"FPR otomatis {target_zone['fpr_auto']*100:.4f}%")

    # ── C. Terbatas pada pola sintetis ──
    print("\n" + "=" * 78)
    print("C. DIUKUR PADA POPULASI POLA SINTETIS SAJA")
    print("=" * 78)
    synth = (labels == 1) & ~is_rare
    results["synthetic_only"] = {}
    print(f"  {'threshold':>11}{'FP':>7}{'FPR%':>10}{'FN sintetis':>13}{'FNR sintetis%':>15}")
    for name, thr in [("fpr_0.05_persen", float(neg_sorted[max_fp])), ("presisi_tinggi_0.90", 0.90)]:
        pred = probs >= thr
        s_fn = int((~pred & synth).sum())
        s_fp = int((pred & (labels == 0)).sum())
        entry = {
            "threshold": thr, "fp": s_fp, "fpr": s_fp / n_neg,
            "fn_synthetic": s_fn, "fnr_synthetic": s_fn / int(synth.sum()),
            "synthetic_total": int(synth.sum()),
        }
        results["synthetic_only"][name] = entry
        flag = "  <- keduanya < 0,05%" if entry["fpr"] < TARGET and entry["fnr_synthetic"] < TARGET else ""
        print(f"  {thr:>11.4f}{s_fp:>7,}{entry['fpr']*100:>10.4f}{s_fn:>13,}"
              f"{entry['fnr_synthetic']*100:>15.4f}{flag}")

    render_figure(curve, len(probs))
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON  : {OUT_JSON}")
    print(f"Gambar: {OUT_PNG}")


if __name__ == "__main__":
    main()
