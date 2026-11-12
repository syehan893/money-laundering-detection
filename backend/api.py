"""
api.py — REST API mandiri untuk menjalankan model secara lokal
=================================================================
Berbeda dari `backend/main.py` yang mensyaratkan MongoDB berjalan, modul ini
memuat model langsung dari berkas dan siap dipakai tanpa basis data. Cocok
untuk pengembangan lokal dan uji produksi.

Endpoint:
    GET  /api/health              status sistem & model yang dimuat
    GET  /api/model-info          versi, threshold, arsitektur, metrik pelatihan
    GET  /api/metrics             metrik & riwayat pelatihan lengkap
    POST /api/predict             prediksi dari JSON (satu atau banyak transaksi)
    POST /api/predict-csv         unggah berkas CSV -> prediksi + penilaian
    GET  /api/test-files          daftar berkas uji produksi yang tersedia
    POST /api/test-files/{name}   jalankan salah satu berkas uji bawaan

Menjalankan:
    python -m backend.api
    # atau
    uvicorn backend.api:app --reload --port 8000
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
from typing import Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backend.config as C
from backend.ml.inference import REQUIRED_COLUMNS, AMLPredictor, validate_csv

# v5 = ensemble tiga seed yang dilaporkan di naskah (threshold 0,7278).
MODEL_VERSION = os.getenv("AML_MODEL_VERSION", "v5")
TEST_FILE_DIR = os.path.join(C.DATA_DIR, "production_test")
MAX_UPLOAD_ROWS = 50_000

app = FastAPI(
    title="AML Guard API",
    description="Deteksi pencucian uang berbasis Graph Attention Network",
    version="5.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class _State:
    predictor: Optional[AMLPredictor] = None
    load_error: Optional[str] = None
    loaded_at: Optional[float] = None


state = _State()


@app.on_event("startup")
async def startup() -> None:
    """Muat model sekali saat server dinyalakan."""
    print(f"\nMemuat model AML (versi {MODEL_VERSION}) ...")
    started = time.time()
    try:
        state.predictor = AMLPredictor.load(version=MODEL_VERSION)
        state.loaded_at = time.time()
        print(f"  Model siap dalam {state.loaded_at - started:.1f} detik "
              f"(threshold {state.predictor.threshold:.4f}, perangkat {state.predictor.device})")
    except Exception as exc:  # noqa: BLE001 - ditampilkan lewat /api/health
        state.load_error = str(exc)
        print(f"  GAGAL memuat model: {exc}")
    print("Server siap.\n")


def _require_predictor() -> AMLPredictor:
    if state.predictor is None:
        raise HTTPException(503, f"Model belum dimuat. {state.load_error or ''}".strip())
    return state.predictor


# ─── Skema ────────────────────────────────────────────────────────────────────
class TransactionIn(BaseModel):
    """Satu transaksi yang akan dinilai."""

    Time: str = Field(..., examples=["14:30:00"])
    Date: str = Field(..., examples=["2022-11-05"])
    Sender_account: str
    Receiver_account: str
    Amount: float
    Payment_currency: str = "UK pounds"
    Received_currency: str = "UK pounds"
    Sender_bank_location: str = "UK"
    Receiver_bank_location: str = "UK"
    Payment_type: str = "Credit card"


class PredictRequest(BaseModel):
    transactions: List[TransactionIn]


# ─── Endpoint ─────────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health() -> Dict:
    """Status sistem dan model."""
    ready = state.predictor is not None
    return {
        "status": "ok" if ready else "model_not_loaded",
        "model_loaded": ready,
        "model_version": MODEL_VERSION,
        "threshold": state.predictor.threshold if ready else None,
        "device": str(state.predictor.device) if ready else None,
        "error": state.load_error,
    }


def _read_json(path: Optional[str]) -> Dict:
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _load_metrics() -> Dict:
    """
    Metrik model aktif.

    Untuk versi ensemble (v5), berkas training_metrics hanya memuat model seed
    42 pada threshold anggaran FPR. Angka uji yang benar -- dan yang dilaporkan
    di naskah -- adalah milik ensemble pada threshold yang dipilih di data
    validasi, sehingga bagian evaluasinya ditimpa dengan evaluation_v5.json.
    Riwayat pelatihan dan hyperparameter tetap dari model seed 42.
    """
    spec = AMLPredictor.VERSIONS[MODEL_VERSION]
    metrics = _read_json(getattr(C, spec["metrics"]))
    evaluation = _read_json(getattr(C, spec["evaluation"])) if spec.get("evaluation") else {}
    if evaluation:
        m = evaluation["metrik"]
        cm = evaluation["confusion_matrix"]
        metrics.update({
            "version": f"{MODEL_VERSION} ensemble",
            "optimal_threshold": evaluation["threshold"],
            "threshold_selection": evaluation.get("kriteria_ambang"),
            "test_metrics": {"precision": m["precision"], "recall": m["recall"], "f1": m["f1"]},
            "confusion_matrix": cm,
            "error_rates": {"fpr": m["fpr"], "fnr": m["fnr"], "accuracy": m["accuracy"],
                            "tpr": m["recall"], "tnr": 1 - m["fpr"]},
            "breakdown": {"rare_total": cm["tp"] + cm["fn"], "rare_caught": cm["tp"],
                          "rare_recall": m["recall"], "synthetic_total": 0,
                          "synthetic_caught": 0, "synthetic_recall": 0.0},
        })
        hp = dict(metrics.get("hyperparameters", {}))
        hp.pop("seed", None)
        hp["ensemble_seeds"] = ", ".join(str(x) for x in C.ENSEMBLE_SEEDS_V5)
        metrics["hyperparameters"] = hp
    return metrics


@app.get("/api/model-info")
async def model_info() -> Dict:
    """Ringkasan model: arsitektur, threshold, dan metrik uji."""
    predictor = _require_predictor()
    metrics = _load_metrics()
    spec = AMLPredictor.VERSIONS[MODEL_VERSION]
    per_pola = _read_json(getattr(C, spec["per_pola"])) if spec.get("per_pola") else {}

    return {
        "version": metrics.get("version", MODEL_VERSION),
        "threshold": predictor.threshold,
        "device": str(predictor.device),
        "graph": {
            "nodes": int(predictor.graph.x.shape[0]),
            "edges": int(predictor.graph.edge_index.shape[1]),
            "node_features": int(predictor.graph.x.shape[1]),
            "edge_features": int(predictor.graph.edge_attr.shape[1]),
        },
        "test_metrics": metrics.get("test_metrics"),
        "confusion_matrix": metrics.get("confusion_matrix"),
        "error_rates": metrics.get("error_rates"),
        "breakdown": metrics.get("breakdown"),
        "hyperparameters": metrics.get("hyperparameters"),
        "per_pola": per_pola.get("per_pola"),
        "risk_bands": {
            "Low": "P < 0,30", "Moderate": "0,30 <= P < 0,70",
            "High": "0,70 <= P < 0,90", "Critical": "P >= 0,90",
        },
    }


@app.get("/api/metrics")
async def metrics() -> Dict:
    """Metrik dan riwayat pelatihan lengkap."""
    metrics = _load_metrics()
    if not metrics:
        raise HTTPException(404, "Berkas metrik belum ada. Latih model terlebih dahulu.")
    return metrics


def _results_payload(results, elapsed: float, source: str) -> Dict:
    """Bentuk respons seragam untuk seluruh endpoint prediksi."""
    distribution: Dict[str, int] = {"Low": 0, "Moderate": 0, "High": 0, "Critical": 0}
    for r in results:
        distribution[r.risk_category] += 1

    return {
        "sumber": source,
        "total": len(results),
        "ditandai": sum(1 for r in results if r.flagged),
        "threshold": state.predictor.threshold if state.predictor else None,
        "waktu_proses_detik": round(elapsed, 3),
        "distribusi_risiko": distribution,
        "penilaian": AMLPredictor.score_results(results),
        "prediksi": [
            {
                "index": r.index,
                "sender": r.sender,
                "receiver": r.receiver,
                "amount": r.amount,
                "probability": r.probability,
                "risk_category": r.risk_category,
                "flagged": r.flagged,
                "actual_label": r.actual_label,
                "laundering_type": r.laundering_type,
            }
            for r in results
        ],
    }


@app.post("/api/predict")
async def predict(request: PredictRequest) -> Dict:
    """Prediksi risiko untuk satu atau beberapa transaksi dalam bentuk JSON."""
    predictor = _require_predictor()
    if not request.transactions:
        raise HTTPException(400, "Tidak ada transaksi yang dikirim.")

    df = pd.DataFrame([t.model_dump() for t in request.transactions])
    started = time.time()
    results = predictor.predict_dataframe(df)
    return _results_payload(results, time.time() - started, "input manual")


@app.post("/api/predict-csv")
async def predict_csv(file: UploadFile = File(...)) -> Dict:
    """
    Unggah berkas CSV lalu jalankan model atas seluruh barisnya.

    Kolom `Is_laundering` bersifat opsional. Bila ada, kolom itu diperlakukan
    sebagai kunci jawaban untuk menghitung confusion matrix -- bukan sebagai
    masukan model.
    """
    predictor = _require_predictor()

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Berkas harus berekstensi .csv")

    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"Berkas CSV tidak dapat dibaca: {exc}") from exc

    problems = validate_csv(df)
    if problems:
        raise HTTPException(400, {"pesan": "Berkas tidak valid", "masalah": problems,
                                  "kolom_wajib": REQUIRED_COLUMNS})
    if len(df) > MAX_UPLOAD_ROWS:
        raise HTTPException(413, f"Berkas terlalu besar: {len(df):,} baris "
                                 f"(maksimum {MAX_UPLOAD_ROWS:,}).")

    started = time.time()
    results = predictor.predict_dataframe(df)
    return _results_payload(results, time.time() - started, file.filename)


@app.get("/api/test-files")
async def test_files() -> Dict:
    """Daftar berkas uji produksi bawaan beserta deskripsinya."""
    manifest_path = os.path.join(TEST_FILE_DIR, "manifest.json")
    if not os.path.exists(manifest_path):
        return {"tersedia": False, "berkas": [],
                "petunjuk": "Jalankan `python scripts/make_production_test_csv.py`."}
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    return {"tersedia": True, **manifest}


@app.post("/api/test-files/{name}")
async def run_test_file(name: str) -> Dict:
    """Jalankan salah satu berkas uji bawaan tanpa perlu mengunggah."""
    predictor = _require_predictor()

    if "/" in name or "\\" in name or not name.endswith(".csv"):
        raise HTTPException(400, "Nama berkas tidak valid.")
    path = os.path.join(TEST_FILE_DIR, name)
    if not os.path.exists(path):
        raise HTTPException(404, f"Berkas uji tidak ditemukan: {name}")

    df = pd.read_csv(path)
    started = time.time()
    results = predictor.predict_dataframe(df)
    return _results_payload(results, time.time() - started, name)


@app.get("/api/graph-stats")
async def graph_stats() -> Dict:
    """Statistik struktur graf transaksi."""
    predictor = _require_predictor()
    n_nodes = int(predictor.graph.x.shape[0])
    n_edges = int(predictor.graph.edge_index.shape[1])
    return {
        "nodes": n_nodes,
        "edges": n_edges,
        "avg_degree": round(2 * n_edges / max(n_nodes, 1), 2),
        "density": round(n_edges / max(n_nodes * (n_nodes - 1), 1), 10),
        "node_features": int(predictor.graph.x.shape[1]),
        "edge_features": int(predictor.graph.edge_attr.shape[1]),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
