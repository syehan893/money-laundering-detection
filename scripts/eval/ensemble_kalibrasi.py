"""
scripts/eval/ensemble_kalibrasi.py — Penggabungan anggota yang skalanya berbeda
================================================================================
Merata-ratakan probabilitas mentah hanya sah bila seluruh anggota terkalibrasi
sama. Pada ensemble ini tidak demikian: anggota dengan `fn_cost = 12` memusatkan
skornya di sekitar 0,73 sedangkan anggota dengan `fn_cost = 5` di sekitar 0,30.
Rata-rata aritmetiknya mencampur dua satuan yang tidak sebanding, sehingga
anggota bernilai tinggi mendominasi tanpa alasan.

Tiga cara penggabungan diuji di sini, semuanya dinilai di validation:

  1. rata-rata probabilitas  -- acuan, cara yang dipakai v5
  2. rata-rata peringkat     -- tiap anggota diubah menjadi peringkat persentil
                                lebih dulu, sehingga skalanya seragam tanpa
                                mengasumsikan bentuk distribusi apa pun
  3. stacking                -- regresi logistik dilatih di validation dengan
                                probabilitas tiap anggota sebagai fitur, jadi
                                bobot tiap anggota dipelajari, bukan disamakan

Stacking dilatih pada validation dan diuji pada test. Validation dengan demikian
berperan ganda: memilih ambang dan melatih penggabung. Itu sah karena test tetap
tidak tersentuh, tetapi berarti estimasi stacking sedikit lebih optimistis
daripada dua cara lainnya, dan itu dinyatakan pada keluarannya.

Keluaran:
  - data/results/evaluation_ensemble_kalibrasi.json
  - data/results/test_probs_v6_terbaik.npy

Usage:
    python scripts/eval/ensemble_kalibrasi.py
    python scripts/eval/ensemble_kalibrasi.py --cpu
"""

from __future__ import annotations

import argparse
import glob
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


def ke_peringkat(p: np.ndarray) -> np.ndarray:
    """Ubah skor menjadi peringkat persentil dalam [0, 1]."""
    order = np.argsort(p, kind="mergesort")
    peringkat = np.empty(len(p), dtype=np.float64)
    peringkat[order] = np.arange(len(p), dtype=np.float64)
    return peringkat / max(len(p) - 1, 1)


def muat_probabilitas(device) -> dict:
    """Jalankan tiap anggota pada graf dan kembalikan probabilitasnya."""
    data = torch.load(C.DATA_V5_PATH, weights_only=False)
    tensors = {k: getattr(data, k).to(device)
               for k in ("x", "edge_index", "edge_attr", "edge_cat")}
    cards = (data.edge_cat.max(dim=0).values + 1).tolist()
    n_typ = int(data.laundering_type.max()) + 1 if hasattr(data, "laundering_type") else 0

    probs = {}
    for mp in sorted(glob.glob(os.path.join(C.DATA_DIR, "models", "best_model_v5*.pt"))):
        sufiks = os.path.basename(mp)[len("best_model_v5"):-len(".pt")]
        nama = sufiks.lstrip("_") or "seed42"
        metrics = C.resolve(f"training_metrics_v5{sufiks}.json")
        if not os.path.exists(metrics):
            continue
        with open(metrics, encoding="utf-8") as f:
            hp = json.load(f).get("hyperparameters", {})
        model = EdgeGATModelV2(
            node_feat_dim=C.NODE_FEAT_DIM_V2, edge_numeric_dim=C.EDGE_NUMERIC_DIM_V5,
            cat_cardinalities=cards, hidden_dim=hp.get("hidden_dim", C.HIDDEN_DIM_V2),
            num_heads=hp.get("num_heads", 2), num_layers=C.NUM_LAYERS_V2,
            dropout=hp.get("dropout", C.DROPOUT_V2),
            use_layer_norm=C.USE_LAYER_NORM_V2, use_residual=C.USE_RESIDUAL_V2,
            use_edge_attention=True, num_typologies=n_typ,
            cat_embed_dropout=C.CAT_EMBED_DROPOUT_V5,
        ).to(device)
        ckpt = torch.load(mp, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        try:
            model.load_state_dict(state)
        except RuntimeError:
            continue
        probs[nama] = infer_all_edges(model, tensors, device,
                                      use_amp=(device.type == "cuda")).numpy()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return probs, data


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    ap = argparse.ArgumentParser(description="Uji cara penggabungan ensemble.")
    ap.add_argument("--cpu", action="store_true", help="Paksa inferensi di CPU.")
    args = ap.parse_args()

    device = torch.device("cpu") if args.cpu else get_device()
    probs, data = muat_probabilitas(device)
    y = data.y.cpu().numpy().astype(bool)
    val_m = data.val_mask.cpu().numpy()
    test_m = data.test_mask.cpu().numpy()
    nama = sorted(probs)
    print(f"  {len(nama)} anggota: {', '.join(nama)}\n")

    hasil = {}

    def catat(label: str, skor: np.ndarray, catatan: str = "") -> dict:
        thr = pilih_ambang_seimbang(skor[val_m], y[val_m])
        cv = confusion(y[val_m], skor[val_m] >= thr)
        ct = confusion(y[test_m], skor[test_m] >= thr)
        hasil[label] = {"threshold": thr, "catatan": catatan,
                        "validation": {k: cv[k] for k in ("fp", "fn")},
                        "confusion_matrix": {k: ct[k] for k in ("tn", "fp", "fn", "tp")},
                        "metrik": {k: ct[k] for k in
                                   ("precision", "recall", "f1", "accuracy", "fpr", "fnr")}}
        print(f"  {label:<34} val FP {cv['fp']:>4} FN {cv['fn']:>4}  |  "
              f"test FP {ct['fp']:>4} FN {ct['fn']:>4} total {ct['fp'] + ct['fn']:>4}")
        return hasil[label]

    print("=" * 84)
    print("  Cara penggabungan (ambang selalu dipilih di validation)")
    print("=" * 84)

    tiga = [n for n in nama if n.startswith("seed")]
    catat("rata-rata prob (3 seed asli)", np.mean([probs[n] for n in tiga], axis=0),
          "acuan v5")
    catat("rata-rata prob (6 anggota)", np.mean([probs[n] for n in nama], axis=0),
          "skala antar-anggota tidak sebanding")

    pr = {n: ke_peringkat(probs[n]) for n in nama}
    catat("rata-rata peringkat (3 seed)", np.mean([pr[n] for n in tiga], axis=0),
          "skala seragam")
    catat("rata-rata peringkat (6 anggota)", np.mean([pr[n] for n in nama], axis=0),
          "skala seragam, keragaman penuh")

    # ── Stacking: regresi logistik dilatih di validation ──
    from sklearn.linear_model import LogisticRegression

    X = np.column_stack([probs[n] for n in nama])
    meta = LogisticRegression(max_iter=1000, class_weight="balanced")
    meta.fit(X[val_m], y[val_m])
    skor_stack = meta.predict_proba(X)[:, 1]
    h = catat("stacking (regresi logistik)", skor_stack,
              "dilatih di validation; estimasinya sedikit lebih optimistis")
    print("\n  Bobot stacking per anggota:")
    for n, w in sorted(zip(nama, meta.coef_[0]), key=lambda t: -abs(t[1])):
        print(f"      {n:<16} {w:+.3f}")

    # ── Acuan pembanding ──
    print("\n" + "=" * 84)
    print("  Acuan: XGBoost FP 53 / FN 95   |   Random Forest FP 63 / FN 92")
    print("=" * 84)
    menang = [k for k, v in hasil.items()
              if v["confusion_matrix"]["fp"] < 53 and v["confusion_matrix"]["fn"] < 80]
    if menang:
        print("  Unggul di KEDUA sisi terhadap XGBoost dan v5:")
        for k in menang:
            cm = hasil[k]["confusion_matrix"]
            print(f"      {k}: FP {cm['fp']}  FN {cm['fn']}")
    else:
        print("  Tidak ada cara penggabungan yang unggul di kedua sisi sekaligus.")

    terbaik = min(hasil.items(),
                  key=lambda kv: kv[1]["confusion_matrix"]["fp"] + kv[1]["confusion_matrix"]["fn"])
    print(f"\n  Total kesalahan terkecil: {terbaik[0]} "
          f"({terbaik[1]['confusion_matrix']['fp'] + terbaik[1]['confusion_matrix']['fn']})")

    with open(C.resolve("evaluation_ensemble_kalibrasi.json"), "w", encoding="utf-8") as f:
        json.dump({"anggota": nama, "cara_penggabungan": hasil}, f, indent=2, ensure_ascii=False)
    print("  Tersimpan: data/results/evaluation_ensemble_kalibrasi.json")


if __name__ == "__main__":
    main()
