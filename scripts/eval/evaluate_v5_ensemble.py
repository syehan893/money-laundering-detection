"""
scripts/evaluate_v5_ensemble.py — Evaluasi ensemble v5 dan perbandingan dengan v4
=================================================================================
Menjalankan tiap model seed pada seluruh graf, merata-ratakan probabilitasnya,
lalu memilih ambang pada **validation** dan menerapkannya apa adanya ke test.

Ambang tidak boleh dipilih di test. Kalau dipilih di test, angka yang dilaporkan
adalah hasil terbaik yang bisa dicapai *setelah* melihat jawabannya, dan itu
bukan estimasi kinerja pada data baru.

Keluaran:
  - data/evaluation_v5.json
  - data/test_probs_v5_ensemble.npy

Usage:
    python scripts/evaluate_v5_ensemble.py
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

import backend.config as C
from backend.ml.model import EdgeGATModelV2
from backend.ml.train import infer_all_edges
from backend.ml.utils import get_device


def confusion(y: np.ndarray, pred: np.ndarray) -> dict:
    tp = int((pred & y).sum()); fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum()); tn = int((~pred & ~y).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / max(prec + rec, 1e-12),
            "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
            "fpr": fp / max(fp + tn, 1), "fnr": fn / max(fn + tp, 1)}


def pilih_ambang_seimbang(probs: np.ndarray, y: np.ndarray) -> float:
    """
    Ambang yang meminimalkan FP + FN.

    Dicari dengan menyisir seluruh nilai probabilitas unik yang ada, bukan grid
    seragam: seluruh kesalahan menumpuk di pita sempit, sehingga grid kasar
    melewatkan titik optimum.
    """
    order = np.argsort(probs)
    p_sorted, y_sorted = probs[order], y[order].astype(np.int64)
    n_pos = int(y_sorted.sum())
    # Pada tiap titik potong i: prediksi positif = elemen ke-i dan sesudahnya.
    fn_kum = np.concatenate([[0], np.cumsum(y_sorted)])          # positif di bawah ambang
    fp_kum = np.concatenate([[0], np.cumsum(1 - y_sorted)])      # negatif di bawah ambang
    n_neg = len(y_sorted) - n_pos
    total = fn_kum + (n_neg - fp_kum)                            # FN + FP
    best = int(np.argmin(total))
    if best >= len(p_sorted):
        return float(p_sorted[-1]) + 1e-6
    return float(p_sorted[best])


def main(paksa_cpu: bool = False) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    # --cpu dipakai saat training masih berjalan: inferensi graf penuh di GPU
    # yang sama akan berebut VRAM 8 GB dan menjatuhkan salah satu proses.
    device = torch.device("cpu") if paksa_cpu else get_device()
    data = torch.load(C.DATA_V5_PATH, weights_only=False)
    tensors = {k: getattr(data, k).to(device) for k in ("x", "edge_index", "edge_attr", "edge_cat")}
    y_all = data.y.cpu().numpy().astype(bool)
    val_m = data.val_mask.cpu().numpy()
    test_m = data.test_mask.cpu().numpy()

    cards = (data.edge_cat.max(dim=0).values + 1).tolist()
    n_typ = int(data.laundering_type.max()) + 1 if hasattr(data, "laundering_type") else 0

    probs_per_seed = []
    for seed in C.ENSEMBLE_SEEDS_V5:
        path = C.MODEL_V5_PATH if seed == C.SEED else C.MODEL_V5_PATH.replace(".pt", f"_seed{seed}.pt")
        if not os.path.exists(path):
            print(f"  [lewati] seed {seed}: {os.path.basename(path)} belum ada")
            continue
        ckpt = torch.load(path, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        model = EdgeGATModelV2(
            node_feat_dim=C.NODE_FEAT_DIM_V2, edge_numeric_dim=C.EDGE_NUMERIC_DIM_V5,
            cat_cardinalities=cards, hidden_dim=C.HIDDEN_DIM_V2, num_heads=2,
            num_layers=C.NUM_LAYERS_V2, dropout=C.DROPOUT_V2,
            use_layer_norm=C.USE_LAYER_NORM_V2, use_residual=C.USE_RESIDUAL_V2,
            use_edge_attention=True, num_typologies=n_typ,
            cat_embed_dropout=C.CAT_EMBED_DROPOUT_V5,
        ).to(device)
        model.load_state_dict(state)
        p = infer_all_edges(model, tensors, device, use_amp=(device.type == "cuda")).numpy()
        probs_per_seed.append(p)
        thr_s = pilih_ambang_seimbang(p[val_m], y_all[val_m])
        cm = confusion(y_all[test_m], p[test_m] >= thr_s)
        print(f"  seed {seed:>5}: FP {cm['fp']:>4}  FN {cm['fn']:>4}  total {cm['fp']+cm['fn']:>4}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not probs_per_seed:
        print("Tidak ada model v5 yang bisa dimuat.")
        return

    ens = np.mean(probs_per_seed, axis=0)
    thr = pilih_ambang_seimbang(ens[val_m], y_all[val_m])
    cm_val = confusion(y_all[val_m], ens[val_m] >= thr)
    cm = confusion(y_all[test_m], ens[test_m] >= thr)

    print(f"\n  ENSEMBLE ({len(probs_per_seed)} seed), ambang {thr:.6f} dipilih di validation")
    print(f"    validation : FP {cm_val['fp']:>4}  FN {cm_val['fn']:>4}")
    print(f"    test       : FP {cm['fp']:>4}  FN {cm['fn']:>4}  total {cm['fp']+cm['fn']:>4}")
    print(f"    FPR {cm['fpr']*100:.4f}%  FNR {cm['fnr']*100:.4f}%  F1 {cm['f1']:.4f}")

    # Pembanding v4 pada titik operasi seimbang yang sama-sama dipilih di validation
    pembanding = {}
    v4_probs = C.resolve("test_probs_v4.npy")
    if os.path.exists(v4_probs):
        with open(C.resolve("evaluation_v4.json"), encoding="utf-8") as f:
            v4 = json.load(f)
        pembanding["v4_titik_operasi_asli"] = v4["confusion_matrix"]

    out = {
        "versi": "v5",
        "n_seed": len(probs_per_seed),
        "threshold": thr,
        "threshold_dipilih_pada": "validation",
        "kriteria_ambang": "minimum FP+FN",
        "confusion_matrix": {k: cm[k] for k in ("tn", "fp", "fn", "tp")},
        "metrik": {k: cm[k] for k in ("precision", "recall", "f1", "accuracy", "fpr", "fnr")},
        "validation": {k: cm_val[k] for k in ("tn", "fp", "fn", "tp")},
        "pembanding": pembanding,
    }
    with open(C.resolve("evaluation_v5.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    np.save(C.resolve("test_probs_v5_ensemble.npy"), ens[test_m])
    print(f"\n  Tersimpan: data/evaluation_v5.json")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluasi ensemble v5.")
    parser.add_argument("--cpu", action="store_true",
                        help="Paksa inferensi di CPU (dipakai saat GPU sedang training).")
    main(parser.parse_args().cpu)
