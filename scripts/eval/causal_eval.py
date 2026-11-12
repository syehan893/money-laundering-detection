"""
scripts/causal_eval.py — Evaluasi kausal (streaming) model v2
================================================================
Menjawab: berapa performa model bila dijalankan sebagaimana adanya di
produksi, yaitu saat menilai sebuah transaksi ia hanya boleh melihat
transaksi yang **sudah** terjadi?

Latar belakang
--------------
Training dan evaluasi standar pada GNN bersifat **transduktif**: message
passing dijalankan atas seluruh graf, termasuk edge periode test. Label test
memang tidak pernah dipakai, tetapi *struktur* periode test ikut membentuk
embedding node. Di produksi hal itu mustahil — saat transaksi masuk, edge
masa depan belum ada.

Skrip ini mengevaluasi secara kausal: periode test dipecah menjadi beberapa
jendela kronologis, lalu tiap jendela dinilai memakai graf yang hanya berisi
edge sebelum jendela itu. Hasilnya adalah batas bawah yang realistis.

Tiga angka yang dibandingkan:
  1. Transduktif  — graf penuh (angka yang lazim dilaporkan di literatur GNN).
  2. Kausal       — bertahap per jendela waktu (mendekati kondisi produksi).
  3. Beku         — graf hanya train+val (pesimistis: seolah tidak ada
                    pembaruan graf sama sekali selama periode test).

Output:
  - data/causal_eval.json

Usage:
    python scripts/causal_eval.py [--windows 16]
"""

import argparse
import json
import os
import sys
import warnings
from typing import Dict, List

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

import backend.config as C
from backend.ml.model import EdgeGATModelV2

OUT_JSON = C.resolve("causal_eval.json")


def load_model(artifacts: Dict, device: torch.device) -> EdgeGATModelV2:
    """Muat model v2 dengan konfigurasi dari backend.config."""
    model = EdgeGATModelV2(
        node_feat_dim=C.NODE_FEAT_DIM_V2,
        edge_numeric_dim=C.EDGE_NUMERIC_DIM_V2,
        cat_cardinalities=artifacts["cat_cardinalities"],
        cat_emb_dims=(C.CURRENCY_EMB_DIM, C.CURRENCY_EMB_DIM,
                      C.LOCATION_EMB_DIM, C.LOCATION_EMB_DIM, C.PAYMENT_TYPE_EMB_DIM),
        hidden_dim=C.HIDDEN_DIM_V2, num_heads=C.NUM_HEADS_V2, num_layers=C.NUM_LAYERS_V2,
        dropout=0.0, use_layer_norm=C.USE_LAYER_NORM_V2, use_residual=C.USE_RESIDUAL_V2,
    ).to(device)
    model.load_state_dict(torch.load(C.MODEL_V2_PATH, map_location=device, weights_only=True))
    model.eval()
    return model


@torch.no_grad()
def score_with_graph(
    model: EdgeGATModelV2,
    data,
    graph_edges: torch.Tensor,
    target_idx: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """
    Skor `target_idx` memakai `graph_edges` sebagai graf message passing.

    Memisahkan graf (konteks) dari target (yang dinilai) inilah yang membuat
    evaluasi kausal mungkin: konteks dibatasi pada masa lalu, target tetap
    edge periode test.
    """
    h = model.encode_nodes(data.x.to(device), graph_edges.to(device))
    logits = model.score_edges(
        h, data.edge_index.to(device), data.edge_attr.to(device),
        data.edge_cat.to(device), subset=target_idx.to(device), chunked=True,
    )
    return torch.sigmoid(logits.float()).cpu().numpy()


def metrics_at_fp_budget(
    probs: np.ndarray, labels: np.ndarray, is_rare: np.ndarray, rate: float = 0.001
) -> Dict:
    """Metrik pada anggaran FPR tertentu (default 0,1%)."""
    neg = np.sort(probs[labels == 0])[::-1]
    thr = float(neg[min(int(len(neg) * rate), len(neg) - 1)])
    pred = probs >= thr

    tp = int((pred & (labels == 1)).sum())
    tn = int((~pred & (labels == 0)).sum())
    fp = int((pred & (labels == 0)).sum())
    fn = int((~pred & (labels == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    rare_mask = (labels == 1) & is_rare
    synth_mask = (labels == 1) & ~is_rare
    return {
        "threshold": thr,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "fpr": fp / max(fp + tn, 1),
        "fnr": fn / max(fn + tp, 1),
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "rare_caught": int((pred & rare_mask).sum()),
        "rare_total": int(rare_mask.sum()),
        "rare_recall": int((pred & rare_mask).sum()) / max(int(rare_mask.sum()), 1),
        "synthetic_recall": int((pred & synth_mask).sum()) / max(int(synth_mask.sum()), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluasi kausal model v2.")
    parser.add_argument("--windows", type=int, default=16,
                        help="Jumlah jendela kronologis pada periode test (default 16).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(C.DATA_V2_PATH, weights_only=False)
    artifacts = joblib.load(C.ENCODERS_V2_PATH)
    model = load_model(artifacts, device)

    labels_all = data.y.numpy().astype(int)
    rare_all = data.is_rare_positive.numpy()
    test_idx = data.test_mask.nonzero(as_tuple=True)[0]
    past_mask = data.train_mask | data.val_mask

    labels = labels_all[test_idx.numpy()]
    is_rare = rare_all[test_idx.numpy()]

    print(f"Device: {device}")
    print(f"Edge test: {len(test_idx):,} | edge masa lalu (train+val): {int(past_mask.sum()):,}")
    print(f"Jendela kronologis: {args.windows}\n")

    results: Dict = {}

    # ── 1. Transduktif: graf penuh ──
    print("[1/3] Transduktif (graf penuh, termasuk edge periode test) ...")
    probs_trans = score_with_graph(model, data, data.edge_index, test_idx, device)
    results["transduktif"] = metrics_at_fp_budget(probs_trans, labels, is_rare)

    # ── 2. Kausal: bertahap per jendela ──
    # Baris pada periode test sudah terurut kronologis (lihat data_pipeline_v2),
    # sehingga pemotongan berdasarkan posisi setara dengan pemotongan waktu.
    print(f"[2/3] Kausal bertahap ({args.windows} jendela) ...")
    probs_causal = np.zeros(len(test_idx), dtype=np.float32)
    bounds = np.linspace(0, len(test_idx), args.windows + 1).astype(int)
    past_edges = data.edge_index[:, past_mask]

    for w in range(args.windows):
        lo, hi = bounds[w], bounds[w + 1]
        if lo == hi:
            continue
        # Konteks = seluruh edge masa lalu + edge periode test dari jendela sebelumnya
        if lo > 0:
            context = torch.cat([past_edges, data.edge_index[:, test_idx[:lo]]], dim=1)
        else:
            context = past_edges
        probs_causal[lo:hi] = score_with_graph(
            model, data, context, test_idx[lo:hi], device
        )
        print(f"      jendela {w + 1:>2}/{args.windows}: {hi - lo:>6,} transaksi | "
              f"konteks {context.shape[1]:>9,} edge")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results["kausal"] = metrics_at_fp_budget(probs_causal, labels, is_rare)

    # ── 3. Beku: graf hanya train+val ──
    print("[3/3] Graf beku (hanya train+val) ...")
    probs_frozen = score_with_graph(model, data, past_edges, test_idx, device)
    results["beku"] = metrics_at_fp_budget(probs_frozen, labels, is_rare)

    # ── Laporan ──
    print("\n" + "=" * 92)
    print("PERFORMA TEST SET PADA ANGGARAN FPR 0,1% — tiga cara message passing")
    print("=" * 92)
    print(f"{'skema':<34}{'FN':>7}{'FNR%':>10}{'F1':>10}{'recall langka':>18}{'recall sintetis':>18}")
    rows = [
        ("Transduktif (graf penuh)", "transduktif"),
        ("Kausal bertahap (produksi)", "kausal"),
        ("Graf beku (train+val saja)", "beku"),
    ]
    for label, key in rows:
        m = results[key]
        print(f"{label:<34}{m['confusion_matrix']['fn']:>7,}{m['fnr']*100:>10.4f}{m['f1']:>10.5f}"
              f"{m['rare_caught']:>11}/{m['rare_total']:<5} ({m['rare_recall']*100:4.1f}%)"
              f"{m['synthetic_recall']*100:>16.3f}%")

    t, c = results["transduktif"], results["kausal"]
    print(f"\n  Selisih transduktif -> kausal:")
    print(f"    F1            : {(t['f1'] - c['f1'])*100:+.3f} pp")
    print(f"    FNR           : {(c['fnr'] - t['fnr'])*100:+.4f} pp")
    print(f"    Recall langka : {t['rare_caught']} -> {c['rare_caught']} dari {t['rare_total']}")
    print("\n  Angka kausal adalah yang paling mendekati kondisi produksi dan sebaiknya")
    print("  itulah yang dilaporkan sebagai performa sistem.")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON: {OUT_JSON}")


if __name__ == "__main__":
    main()
