"""
scripts/compare_v1_v2.py — Bandingkan model GAT v1 dan v2
============================================================
Menghasilkan perbandingan apple-to-apple antara model lama (`best_model.pt`)
dan model hasil perbaikan (`best_model_v2.pt`).

Kedua model dibandingkan pada **jumlah false positive yang identik**, bukan
pada threshold masing-masing. Tanpa penyetaraan ini, perbandingan tidak adil:
model dengan threshold lebih longgar otomatis terlihat punya recall lebih baik.

Output:
  - data/model_comparison_v1_v2.json
  - docs/images/confusion_matrix_v1_vs_v2.png

Usage:
    python scripts/compare_v1_v2.py
"""

import json
import os
import sys
import warnings
from typing import Dict, Tuple

warnings.filterwarnings("ignore")

import joblib
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
from matplotlib.colors import LinearSegmentedColormap

import backend.config as C
from backend.ml.legacy.model import EdgeGATModel
from backend.ml.model import EdgeGATModelV2

# ─── Palet (mengikuti skill dataviz: ramp biru sekuensial + slot kategorikal 1/2) ──
SEQ_BLUE = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"]
SERIES_1 = "#2a78d6"   # v1
SERIES_2 = "#eb6834"   # v2
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#e5e4e0"

OUT_JSON = C.resolve("model_comparison_v1_v2.json")
OUT_PNG = os.path.join(PROJECT_ROOT, "docs", "images", "confusion_matrix_v1_vs_v2.png")


def load_v1_scores(device: torch.device) -> Tuple[np.ndarray, object]:
    """Skor seluruh edge dari model v1 (fitur lama)."""
    data = torch.load(C.DATA_PATH, weights_only=False)
    model = EdgeGATModel(
        node_feat_dim=C.NODE_FEAT_DIM, edge_feat_dim=C.EDGE_FEAT_DIM,
        hidden_dim=C.HIDDEN_DIM, num_heads=C.NUM_HEADS, num_layers=C.NUM_LAYERS,
        dropout=0.0, use_layer_norm=C.USE_LAYER_NORM, use_residual=C.USE_RESIDUAL,
    ).to(device)
    model.load_state_dict(torch.load(C.MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        probs = model.predict(
            data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device)
        ).cpu().numpy()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return probs, data


def load_v2_scores(device: torch.device) -> Tuple[np.ndarray, object]:
    """Skor seluruh edge dari model v2 (fitur stasioner + embedding)."""
    data = torch.load(C.DATA_V2_PATH, weights_only=False)
    artifacts = joblib.load(C.ENCODERS_V2_PATH)
    model = EdgeGATModelV2(
        node_feat_dim=C.NODE_FEAT_DIM_V2, edge_numeric_dim=C.EDGE_NUMERIC_DIM_V2,
        cat_cardinalities=artifacts["cat_cardinalities"],
        cat_emb_dims=(C.CURRENCY_EMB_DIM, C.CURRENCY_EMB_DIM,
                      C.LOCATION_EMB_DIM, C.LOCATION_EMB_DIM, C.PAYMENT_TYPE_EMB_DIM),
        hidden_dim=C.HIDDEN_DIM_V2, num_heads=C.NUM_HEADS_V2, num_layers=C.NUM_LAYERS_V2,
        dropout=0.0, use_layer_norm=C.USE_LAYER_NORM_V2, use_residual=C.USE_RESIDUAL_V2,
    ).to(device)
    model.load_state_dict(torch.load(C.MODEL_V2_PATH, map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        probs = model.predict(
            data.x.to(device), data.edge_index.to(device),
            data.edge_attr.to(device), data.edge_cat.to(device),
        ).cpu().numpy()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return probs, data


def metrics_at_fp_budget(
    probs: np.ndarray, labels: np.ndarray, is_rare: np.ndarray, n_fp: int
) -> Dict:
    """
    Hitung metrik pada jumlah false positive yang tepat sama.

    Threshold diturunkan dari distribusi skor negatif sehingga tiap model
    dievaluasi dengan beban investigasi yang identik.
    """
    neg_sorted = np.sort(probs[labels == 0])[::-1]
    thr = float(neg_sorted[min(n_fp, len(neg_sorted) - 1)])
    pred = probs >= thr

    tp = int((pred & (labels == 1)).sum())
    tn = int((~pred & (labels == 0)).sum())
    fp = int((pred & (labels == 0)).sum())
    fn = int((~pred & (labels == 1)).sum())

    rare_mask = (labels == 1) & is_rare
    synth_mask = (labels == 1) & ~is_rare

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "threshold": thr,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "fpr": fp / max(fp + tn, 1),
        "fnr": fn / max(fn + tp, 1),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
        "rare_caught": int((pred & rare_mask).sum()),
        "rare_total": int(rare_mask.sum()),
        "synthetic_caught": int((pred & synth_mask).sum()),
        "synthetic_total": int(synth_mask.sum()),
    }


def _draw_confusion(ax, cm: Dict[str, int], title: str, subtitle: str) -> None:
    """Gambar satu confusion matrix; warna mengikuti proporsi per baris."""
    counts = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]], dtype=float)
    # Warna dinormalisasi per baris: tanpa ini sel FP/FN (ratusan) tidak akan
    # terlihat di samping TN (ratusan ribu).
    rates = counts / counts.sum(axis=1, keepdims=True)

    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    ax.imshow(rates, cmap=cmap, vmin=0.0, vmax=1.0, aspect="equal")

    for i in range(2):
        for j in range(2):
            value = int(counts[i, j])
            rate = rates[i, j]
            ax.text(j, i - 0.10, f"{value:,}", ha="center", va="center",
                    fontsize=15, fontweight="bold",
                    color="#ffffff" if rate > 0.55 else INK_PRIMARY)
            ax.text(j, i + 0.16, f"{rate*100:.2f}%", ha="center", va="center",
                    fontsize=9.5, color="#e8eef8" if rate > 0.55 else INK_SECONDARY)

    ax.set_xticks([0, 1], ["Prediksi\nNormal", "Prediksi\nLaundering"], fontsize=10)
    ax.set_yticks([0, 1], ["Aktual\nNormal", "Aktual\nLaundering"], fontsize=10)
    ax.set_title(title, fontsize=13, fontweight="bold", color=INK_PRIMARY, pad=14)
    ax.text(0.5, -0.30, subtitle, transform=ax.transAxes, ha="center",
            fontsize=10, color=INK_SECONDARY)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, colors=INK_SECONDARY)


def _draw_recall_bars(ax, v1: Dict, v2: Dict) -> None:
    """Bar chart recall per kelompok tipologi."""
    groups = ["Pola sintetis\n(disuntikkan)", "Tipologi asli SAML-D\n(kasus nyata)"]
    v1_vals = [
        v1["synthetic_caught"] / max(v1["synthetic_total"], 1) * 100,
        v1["rare_caught"] / max(v1["rare_total"], 1) * 100,
    ]
    v2_vals = [
        v2["synthetic_caught"] / max(v2["synthetic_total"], 1) * 100,
        v2["rare_caught"] / max(v2["rare_total"], 1) * 100,
    ]

    y = np.arange(len(groups))
    height = 0.32
    gap = 0.02  # jarak 2px antar bar, bukan garis pembatas

    ax.barh(y - height / 2 - gap, v1_vals, height, label="v1 (lama)", color=SERIES_1)
    ax.barh(y + height / 2 + gap, v2_vals, height, label="v2 (baru)", color=SERIES_2)

    for yy, val, cnt, tot in [
        (y[0] - height / 2 - gap, v1_vals[0], v1["synthetic_caught"], v1["synthetic_total"]),
        (y[0] + height / 2 + gap, v2_vals[0], v2["synthetic_caught"], v2["synthetic_total"]),
        (y[1] - height / 2 - gap, v1_vals[1], v1["rare_caught"], v1["rare_total"]),
        (y[1] + height / 2 + gap, v2_vals[1], v2["rare_caught"], v2["rare_total"]),
    ]:
        label = f"{val:.1f}%  ({cnt:,} dari {tot:,})"
        # Label bar panjang ditaruh di dalam bar agar tidak terpotong tepi kanan
        if val > 55:
            ax.text(val - 2, yy, label, va="center", ha="right",
                    fontsize=9.5, color="#ffffff")
        else:
            ax.text(val + 2, yy, label, va="center", ha="left",
                    fontsize=9.5, color=INK_SECONDARY)

    ax.set_yticks(y, groups, fontsize=10)
    ax.set_xlim(0, 104)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"], fontsize=9)
    ax.set_xlabel("Recall (proporsi laundering yang tertangkap)", fontsize=10, color=INK_SECONDARY)
    ax.set_title("Recall per kelompok tipologi", fontsize=13, fontweight="bold",
                 color=INK_PRIMARY, pad=34)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.yaxis.grid(False)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(length=0, colors=INK_SECONDARY)
    # Legenda di atas plot: di dalam area plot ia akan menimpa bar
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2,
              frameon=False, fontsize=10, labelcolor=INK_SECONDARY,
              handlelength=1.2, handleheight=1.0, columnspacing=1.4)


def render_figure(v1: Dict, v2: Dict, n_fp: int) -> None:
    """Render perbandingan: dua confusion matrix + recall per kelompok."""
    fig = plt.figure(figsize=(15.5, 5.6), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.5], wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    for ax in (ax1, ax2, ax3):
        ax.set_facecolor(SURFACE)

    _draw_confusion(
        ax1, v1["confusion_matrix"], "v1 — model lama",
        f"FN {v1['confusion_matrix']['fn']}  ·  FNR {v1['fnr']*100:.4f}%",
    )
    _draw_confusion(
        ax2, v2["confusion_matrix"], "v2 — setelah perbaikan",
        f"FN {v2['confusion_matrix']['fn']}  ·  FNR {v2['fnr']*100:.4f}%",
    )
    _draw_recall_bars(ax3, v1, v2)

    fig.suptitle(
        "Perbandingan model deteksi money laundering — test set 500.000 transaksi",
        fontsize=15, fontweight="bold", color=INK_PRIMARY, y=0.99,
    )
    fig.text(
        0.5, 0.915,
        f"Kedua model disetarakan pada {n_fp} false positive yang sama "
        f"(FPR {v1['fpr']*100:.4f}%), sehingga beban investigasi identik",
        ha="center", fontsize=10.5, color=INK_SECONDARY,
    )
    fig.subplots_adjust(top=0.80, bottom=0.17, left=0.06, right=0.97)

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Menghitung skor v1 ...")
    p1, d1 = load_v1_scores(device)
    print("Menghitung skor v2 ...")
    p2, d2 = load_v2_scores(device)

    labels = d2.y.numpy().astype(int)
    if not (d1.y.numpy().astype(int) == labels).all():
        raise RuntimeError("Urutan baris v1 dan v2 tidak sama — perbandingan tidak valid.")

    is_rare = d2.is_rare_positive.numpy()
    test = d2.test_mask.numpy()
    val = d2.val_mask.numpy()

    # Titik operasi utama: jumlah FP yang sama dengan titik operasi v1 saat ini.
    v1_threshold = json.load(open(C.METRICS_PATH, encoding="utf-8"))["optimal_threshold"]
    n_fp_v1 = int(((p1[test] >= v1_threshold) & (labels[test] == 0)).sum())
    fp_budget_01 = int((labels[test] == 0).sum() * 0.001)

    results = {}
    for point_name, n_fp in [
        (f"fp_{n_fp_v1}_titik_operasi_v1", n_fp_v1),
        ("fpr_0.1_persen", fp_budget_01),
    ]:
        m1 = metrics_at_fp_budget(p1[test], labels[test], is_rare[test], n_fp)
        m2 = metrics_at_fp_budget(p2[test], labels[test], is_rare[test], n_fp)
        results[point_name] = {"v1": m1, "v2": m2}

        print(f"\n=== Titik operasi: {n_fp} false positive ===")
        header = f"{'model':<6}{'TN':>10}{'FP':>7}{'FN':>7}{'TP':>9}{'FPR%':>10}{'FNR%':>10}{'langka':>12}{'F1':>10}"
        print(header)
        for name, m in [("v1", m1), ("v2", m2)]:
            cm = m["confusion_matrix"]
            print(f"{name:<6}{cm['tn']:>10,}{cm['fp']:>7,}{cm['fn']:>7,}{cm['tp']:>9,}"
                  f"{m['fpr']*100:>10.4f}{m['fnr']*100:>10.4f}"
                  f"{str(m['rare_caught']) + '/' + str(m['rare_total']):>12}{m['f1']:>10.5f}")

    # Metrik validasi sebagai bukti pemilihan model tidak mengintip test set
    val_v1 = metrics_at_fp_budget(p1[val], labels[val], is_rare[val],
                                  int((labels[val] == 0).sum() * 0.001))
    val_v2 = metrics_at_fp_budget(p2[val], labels[val], is_rare[val],
                                  int((labels[val] == 0).sum() * 0.001))
    results["validation_fpr_0.1_persen"] = {"v1": val_v1, "v2": val_v2}

    main_point = results[f"fp_{n_fp_v1}_titik_operasi_v1"]
    render_figure(main_point["v1"], main_point["v2"], n_fp_v1)

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nJSON  : {OUT_JSON}")
    print(f"Gambar: {OUT_PNG}")


if __name__ == "__main__":
    main()
