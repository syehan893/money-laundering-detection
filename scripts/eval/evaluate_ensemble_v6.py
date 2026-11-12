"""
scripts/eval/evaluate_ensemble_v6.py — Evaluasi ensemble beragam & pemilihan anggota
=====================================================================================
Menemukan seluruh anggota ensemble v5 yang tersedia, menjalankan masing-masing
pada graf yang sama, lalu memilih kombinasi anggota terbaik **berdasarkan
validation** dan melaporkan hasilnya pada test.

Setiap anggota dapat memiliki hyperparameter berbeda (hidden_dim, dropout,
fn_cost). Konfigurasinya dibaca dari berkas metrik masing-masing, bukan
diasumsikan seragam -- memuat bobot dengan hidden_dim yang salah akan gagal
diam-diam atau melempar galat bentuk tensor.

Pemilihan kombinasi dilakukan di validation. Memilih kombinasi terbaik di test
sama saja dengan menyetel model pada data uji, dan angkanya tidak lagi menjadi
estimasi kinerja pada data baru.

Keluaran:
  - data/results/evaluation_v6.json
  - data/results/test_probs_v6_ensemble.npy

Usage:
    python scripts/eval/evaluate_ensemble_v6.py
    python scripts/eval/evaluate_ensemble_v6.py --cpu
"""

from __future__ import annotations

import argparse
import glob
import itertools
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
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    tn = int((~pred & ~y).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / max(prec + rec, 1e-12),
            "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
            "fpr": fp / max(fp + tn, 1), "fnr": fn / max(fn + tp, 1)}


def pilih_ambang_seimbang(probs: np.ndarray, y: np.ndarray) -> float:
    """Ambang yang meminimalkan FP + FN, disisir pada seluruh nilai unik."""
    order = np.argsort(probs)
    p_sorted, y_sorted = probs[order], y[order].astype(np.int64)
    n_pos = int(y_sorted.sum())
    n_neg = len(y_sorted) - n_pos
    fn_kum = np.concatenate([[0], np.cumsum(y_sorted)])
    fp_kum = np.concatenate([[0], np.cumsum(1 - y_sorted)])
    total = fn_kum + (n_neg - fp_kum)
    best = int(np.argmin(total))
    return float(p_sorted[best]) if best < len(p_sorted) else float(p_sorted[-1]) + 1e-6


def temukan_anggota() -> list:
    """Semua bobot model v5 beserta berkas metriknya."""
    pola = os.path.join(C.DATA_DIR, "models", "best_model_v5*.pt")
    anggota = []
    for mp in sorted(glob.glob(pola)):
        sufiks = os.path.basename(mp)[len("best_model_v5"):-len(".pt")]
        metrics = C.resolve(f"training_metrics_v5{sufiks}.json")
        anggota.append({"nama": sufiks.lstrip("_") or "seed42",
                        "model": mp, "metrics": metrics})
    return anggota


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    ap = argparse.ArgumentParser(description="Evaluasi ensemble beragam.")
    ap.add_argument("--cpu", action="store_true", help="Paksa inferensi di CPU.")
    args = ap.parse_args()

    device = torch.device("cpu") if args.cpu else get_device()
    data = torch.load(C.DATA_V5_PATH, weights_only=False)
    tensors = {k: getattr(data, k).to(device)
               for k in ("x", "edge_index", "edge_attr", "edge_cat")}
    y_all = data.y.cpu().numpy().astype(bool)
    val_m = data.val_mask.cpu().numpy()
    test_m = data.test_mask.cpu().numpy()
    cards = (data.edge_cat.max(dim=0).values + 1).tolist()
    n_typ = int(data.laundering_type.max()) + 1 if hasattr(data, "laundering_type") else 0

    print("=" * 78)
    print("  Anggota ensemble yang ditemukan")
    print("=" * 78)
    probs = {}
    for a in temukan_anggota():
        if not os.path.exists(a["metrics"]):
            print(f"  [lewati] {a['nama']}: berkas metrik tidak ada")
            continue
        with open(a["metrics"], encoding="utf-8") as f:
            hp = json.load(f).get("hyperparameters", {})
        # hidden_dim dan dropout dibaca per anggota; anggota "wide" memakai 96
        model = EdgeGATModelV2(
            node_feat_dim=C.NODE_FEAT_DIM_V2, edge_numeric_dim=C.EDGE_NUMERIC_DIM_V5,
            cat_cardinalities=cards, hidden_dim=hp.get("hidden_dim", C.HIDDEN_DIM_V2),
            num_heads=hp.get("num_heads", 2), num_layers=C.NUM_LAYERS_V2,
            dropout=hp.get("dropout", C.DROPOUT_V2),
            use_layer_norm=C.USE_LAYER_NORM_V2, use_residual=C.USE_RESIDUAL_V2,
            use_edge_attention=True, num_typologies=n_typ,
            cat_embed_dropout=C.CAT_EMBED_DROPOUT_V5,
        ).to(device)
        ckpt = torch.load(a["model"], map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        try:
            model.load_state_dict(state)
        except RuntimeError as e:
            print(f"  [lewati] {a['nama']}: bobot tidak cocok ({str(e)[:60]}...)")
            continue

        p = infer_all_edges(model, tensors, device,
                            use_amp=(device.type == "cuda")).numpy()
        probs[a["nama"]] = p
        thr = pilih_ambang_seimbang(p[val_m], y_all[val_m])
        cm = confusion(y_all[test_m], p[test_m] >= thr)
        print(f"  {a['nama']:<14} fn_cost={hp.get('fn_cost', '?'):<5} "
              f"hidden={hp.get('hidden_dim', '?'):<3} dropout={hp.get('dropout', '?'):<5} "
              f"| FP {cm['fp']:>4}  FN {cm['fn']:>4}  total {cm['fp'] + cm['fn']:>4}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if len(probs) < 2:
        print("\nAnggota kurang dari dua; ensemble tidak dibentuk.")
        return

    # ── Pilih kombinasi terbaik DI VALIDATION ──
    nama_semua = list(probs)
    print("\n" + "=" * 78)
    print("  Pemilihan kombinasi (dinilai di validation, bukan di test)")
    print("=" * 78)
    kandidat = []
    for r in range(2, len(nama_semua) + 1):
        for komb in itertools.combinations(nama_semua, r):
            ens = np.mean([probs[n] for n in komb], axis=0)
            thr = pilih_ambang_seimbang(ens[val_m], y_all[val_m])
            cv = confusion(y_all[val_m], ens[val_m] >= thr)
            kandidat.append({"anggota": list(komb), "thr": thr,
                             "val_total": cv["fp"] + cv["fn"],
                             "val_fp": cv["fp"], "val_fn": cv["fn"]})
    kandidat.sort(key=lambda d: d["val_total"])
    for k in kandidat[:5]:
        print(f"  val FP {k['val_fp']:>4} FN {k['val_fn']:>4} total {k['val_total']:>4}  "
              f"<- {' + '.join(k['anggota'])}")

    terbaik = kandidat[0]
    ens = np.mean([probs[n] for n in terbaik["anggota"]], axis=0)
    thr = terbaik["thr"]
    cm = confusion(y_all[test_m], ens[test_m] >= thr)
    cm_all = confusion(y_all[test_m],
                       np.mean(list(probs.values()), axis=0)[test_m] >=
                       pilih_ambang_seimbang(np.mean(list(probs.values()), axis=0)[val_m],
                                             y_all[val_m]))

    print("\n" + "=" * 78)
    print(f"  Kombinasi terpilih: {' + '.join(terbaik['anggota'])}")
    print(f"  Ambang {thr:.6f} (dipilih di validation)")
    print(f"    test  : FP {cm['fp']:>4}  FN {cm['fn']:>4}  total {cm['fp'] + cm['fn']:>4}")
    print(f"    FPR {cm['fpr'] * 100:.4f}%  FNR {cm['fnr'] * 100:.4f}%  F1 {cm['f1']:.4f}")
    print(f"  Sebagai pembanding, seluruh {len(probs)} anggota digabung: "
          f"FP {cm_all['fp']}  FN {cm_all['fn']}  total {cm_all['fp'] + cm_all['fn']}")

    out = {
        "versi": "v6", "anggota": terbaik["anggota"], "n_anggota": len(terbaik["anggota"]),
        "anggota_tersedia": nama_semua,
        "threshold": thr, "threshold_dipilih_pada": "validation",
        "kriteria_ambang": "minimum FP+FN",
        "seleksi_kombinasi_pada": "validation",
        "confusion_matrix": {k: cm[k] for k in ("tn", "fp", "fn", "tp")},
        "metrik": {k: cm[k] for k in ("precision", "recall", "f1", "accuracy", "fpr", "fnr")},
        "semua_anggota_digabung": {k: cm_all[k] for k in ("tn", "fp", "fn", "tp")},
    }
    with open(C.resolve("evaluation_v6.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    np.save(C.resolve("test_probs_v6_ensemble.npy"), ens[test_m])
    print("\n  Tersimpan: data/results/evaluation_v6.json")


if __name__ == "__main__":
    main()
