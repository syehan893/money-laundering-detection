"""
scripts/baseline/train_baselines.py — Dua metode pembanding: Random Forest & XGBoost
=====================================================================================
Melatih dua model klasik pada **data, fitur, split, dan protokol ambang yang persis
sama** dengan GAT v5, agar selisih angkanya benar-benar mengukur sumbangan
pendekatan graf dan bukan perbedaan perlakuan data.

Apa yang disamakan
------------------
  Fitur      : 58 numerik + 5 kategorikal, diambil langsung dari processed_data_v5.pt
               -- tensor yang sama persis yang dibaca GAT, bukan dihitung ulang.
  Split      : train_mask / val_mask / test_mask dari graf yang sama (temporal).
  Ambang     : dipilih di validation dengan kriteria minimum FP+FN, lalu
               diterapkan apa adanya ke test. Sama seperti evaluate_v5_ensemble.py.
  Evaluasi   : confusion matrix tingkat transaksi dan tingkat akun.

Apa yang berbeda -- dan itulah yang diukur
-------------------------------------------
GAT meneruskan pesan antar simpul: skor sebuah transaksi ikut dipengaruhi
tetangga kedua akunnya. Random Forest dan XGBoost memperlakukan tiap transaksi
sebagai baris yang berdiri sendiri.

Perlu dicatat saat membaca hasilnya: ke-58 fitur itu SUDAH memuat ringkasan
struktur graf (motif, penutupan siklus, derajat, velocity tetangga). Jadi
pembanding di sini bukan "dengan graf" lawan "tanpa graf sama sekali",
melainkan "message passing" lawan "ringkasan graf sebagai kolom tabel".
Ini justru perbandingan yang paling ketat: keunggulan yang tersisa hanya bisa
berasal dari message passing itu sendiri.

Keluaran
--------
  logs/baseline_training.log
  data/results/baseline_random_forest.json
  data/results/baseline_xgboost.json
  data/results/baseline_comparison.json
  data/results/test_probs_random_forest.npy
  data/results/test_probs_xgboost.npy

Usage:
    python scripts/baseline/train_baselines.py
    python scripts/baseline/train_baselines.py --quick    # pohon lebih sedikit
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
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

LOG_PATH = os.path.join(PROJECT_ROOT, "logs", "baseline_training.log")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ─── Metrik ───────────────────────────────────────────────────────────────────
def confusion(y: np.ndarray, pred: np.ndarray) -> dict:
    """Confusion matrix beserta metrik turunannya."""
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
    """
    Ambang yang meminimalkan FP + FN, disisir pada seluruh nilai probabilitas unik.

    Identik dengan yang dipakai GAT v5 di scripts/eval/evaluate_v5_ensemble.py --
    membandingkan model dengan kriteria ambang yang berbeda tidak ada artinya.
    """
    order = np.argsort(probs)
    p_sorted, y_sorted = probs[order], y[order].astype(np.int64)
    n_pos = int(y_sorted.sum())
    n_neg = len(y_sorted) - n_pos
    fn_kum = np.concatenate([[0], np.cumsum(y_sorted)])
    fp_kum = np.concatenate([[0], np.cumsum(1 - y_sorted)])
    total = fn_kum + (n_neg - fp_kum)
    best = int(np.argmin(total))
    return float(p_sorted[best]) if best < len(p_sorted) else float(p_sorted[-1]) + 1e-6


# ─── Data ─────────────────────────────────────────────────────────────────────
def muat_fitur():
    """Ambil matriks fitur datar dari graf v5 -- tensor yang sama yang dibaca GAT."""
    logger.info("Memuat processed_data_v5.pt ...")
    data = torch.load(C.DATA_V5_PATH, weights_only=False)

    numerik = data.edge_attr.numpy().astype(np.float32)          # (E, 58)
    kategorikal = data.edge_cat.numpy().astype(np.float32)       # (E, 5)
    # Kode kategorikal dipakai apa adanya: model pohon memecah pada nilai ordinal
    # tanpa mengasumsikan urutannya bermakna, sehingga one-hot tidak diperlukan
    # dan hanya akan menggelembungkan 63 kolom menjadi ratusan.
    X = np.hstack([numerik, kategorikal])
    y = data.y.numpy().astype(bool)

    masks = {n: getattr(data, f"{n}_mask").numpy() for n in ("train", "val", "test")}
    logger.info(f"  Fitur : {X.shape[1]} kolom ({numerik.shape[1]} numerik + "
                f"{kategorikal.shape[1]} kategorikal)")
    for n, m in masks.items():
        logger.info(f"  {n:<5} : {int(m.sum()):>7,} baris, {int(y[m].sum()):>6,} laundering "
                    f"({y[m].mean()*100:.2f}%)")
    return X, y, masks


# ─── Model ────────────────────────────────────────────────────────────────────
def latih_random_forest(X, y, masks, quick: bool):
    from sklearn.ensemble import RandomForestClassifier

    n_tree = 100 if quick else 300
    logger.info(f"\n[Random Forest] {n_tree} pohon, kedalaman tak dibatasi, "
                f"class_weight balanced_subsample")
    model = RandomForestClassifier(
        n_estimators=n_tree, max_depth=None, min_samples_leaf=2,
        # Kelas laundering hanya 14,2%; tanpa pembobotan model condong ke kelas
        # mayoritas -- persis yang dicegah focal loss pada GAT.
        class_weight="balanced_subsample",
        n_jobs=-1, random_state=C.SEED, verbose=0,
    )
    t0 = time.time()
    model.fit(X[masks["train"]], y[masks["train"]])
    logger.info(f"  Selesai dalam {time.time() - t0:.1f} detik")
    return model


def latih_xgboost(X, y, masks, quick: bool):
    import xgboost as xgb

    n_tree = 200 if quick else 600
    pos = int(y[masks["train"]].sum())
    neg = int((~y[masks["train"]]).sum())
    logger.info(f"\n[XGBoost] {n_tree} pohon, scale_pos_weight={neg / max(pos, 1):.2f}")
    model = xgb.XGBClassifier(
        n_estimators=n_tree, max_depth=8, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=neg / max(pos, 1),
        tree_method="hist", eval_metric="aucpr",
        random_state=C.SEED, n_jobs=-1,
        early_stopping_rounds=30,
    )
    t0 = time.time()
    model.fit(X[masks["train"]], y[masks["train"]],
              eval_set=[(X[masks["val"]], y[masks["val"]])], verbose=False)
    logger.info(f"  Selesai dalam {time.time() - t0:.1f} detik "
                f"(berhenti di pohon ke-{model.best_iteration})")
    return model


# ─── Evaluasi ─────────────────────────────────────────────────────────────────
def evaluasi(nama: str, model, X, y, masks) -> dict:
    probs = model.predict_proba(X)[:, 1]
    thr = pilih_ambang_seimbang(probs[masks["val"]], y[masks["val"]])
    cm_val = confusion(y[masks["val"]], probs[masks["val"]] >= thr)
    cm = confusion(y[masks["test"]], probs[masks["test"]] >= thr)

    logger.info(f"  Ambang {thr:.6f} (dipilih di validation)")
    logger.info(f"    validation : FP {cm_val['fp']:>5}  FN {cm_val['fn']:>5}")
    logger.info(f"    test       : FP {cm['fp']:>5}  FN {cm['fn']:>5}  "
                f"total {cm['fp'] + cm['fn']:>5}")
    logger.info(f"    FPR {cm['fpr'] * 100:.4f}%  FNR {cm['fnr'] * 100:.4f}%  F1 {cm['f1']:.4f}")

    slug = nama.lower().replace(" ", "_")
    np.save(C.resolve(f"test_probs_{slug}.npy"), probs[masks["test"]])
    # Probabilitas seluruh graf disimpan juga: perbandingan pada titik operasi
    # yang disamakan harus dilakukan di validation, bukan di test.
    np.save(C.resolve(f"probs_{slug}_full.npy"), probs)
    hasil = {
        "metode": nama, "threshold": thr, "threshold_dipilih_pada": "validation",
        "kriteria_ambang": "minimum FP+FN",
        "confusion_matrix": {k: cm[k] for k in ("tn", "fp", "fn", "tp")},
        "metrik": {k: cm[k] for k in ("precision", "recall", "f1", "accuracy", "fpr", "fnr")},
        "validation": {k: cm_val[k] for k in ("tn", "fp", "fn", "tp")},
    }
    with open(C.resolve(f"baseline_{slug}.json"), "w", encoding="utf-8") as f:
        json.dump(hasil, f, indent=2, ensure_ascii=False)
    return hasil


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    ap = argparse.ArgumentParser(description="Latih dua metode pembanding.")
    ap.add_argument("--quick", action="store_true", help="Pohon lebih sedikit, untuk uji cepat.")
    args = ap.parse_args()

    logger.info("=" * 74)
    logger.info("  Metode pembanding: Random Forest & XGBoost")
    logger.info("  Data, fitur, split, dan protokol ambang identik dengan GAT v5")
    logger.info("=" * 74)

    X, y, masks = muat_fitur()

    hasil = {}
    hasil["Random Forest"] = evaluasi(
        "Random Forest", latih_random_forest(X, y, masks, args.quick), X, y, masks)
    hasil["XGBoost"] = evaluasi(
        "XGBoost", latih_xgboost(X, y, masks, args.quick), X, y, masks)

    # Sertakan GAT v5 sebagai acuan
    try:
        with open(C.resolve("evaluation_v5.json"), encoding="utf-8") as f:
            v5 = json.load(f)
        hasil["GAT v5 (ensemble)"] = {
            "metode": "GAT v5 (ensemble 3 seed)", "threshold": v5["threshold"],
            "threshold_dipilih_pada": "validation", "kriteria_ambang": "minimum FP+FN",
            "confusion_matrix": v5["confusion_matrix"], "metrik": v5["metrik"],
        }
    except FileNotFoundError:
        logger.warning("  evaluation_v5.json tidak ditemukan; GAT tidak disertakan.")

    logger.info("\n" + "=" * 74)
    logger.info(f"  {'metode':<26}{'FP':>7}{'FN':>7}{'total':>8}{'FPR':>10}{'FNR':>10}{'F1':>9}")
    for nama, h in hasil.items():
        cm, mt = h["confusion_matrix"], h["metrik"]
        logger.info(f"  {nama:<26}{cm['fp']:>7}{cm['fn']:>7}{cm['fp'] + cm['fn']:>8}"
                    f"{mt['fpr'] * 100:>9.4f}%{mt['fnr'] * 100:>9.4f}%{mt['f1']:>9.4f}")

    with open(C.resolve("baseline_comparison.json"), "w", encoding="utf-8") as f:
        json.dump({"perbandingan": hasil}, f, indent=2, ensure_ascii=False)
    logger.info(f"\n  Log    : {os.path.relpath(LOG_PATH, PROJECT_ROOT)}")
    logger.info("  Hasil  : data/results/baseline_comparison.json")


if __name__ == "__main__":
    main()
