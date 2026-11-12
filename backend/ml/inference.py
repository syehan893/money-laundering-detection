"""
inference.py — Jalur inferensi produksi (model v4 dan ensemble v5)
============================================================
Menggantikan jalur prediksi lama pada `backend/main.py` yang memiliki empat
cacat (lihat docs/docs md/RINGKASAN_PROYEK.md §13):

  1. Urutan fitur edge berbeda dari saat pelatihan.
  2. `edge_scaler` tidak pernah diterapkan.
  3. Transaksi baru diberi `edge_index` nol, sehingga tidak terhubung ke akun
     yang sebenarnya dan konteks graf hilang.
  4. Struktur `encoders.pkl` diakses dengan kunci yang salah.

Modul ini memakai ulang fungsi rekayasa fitur yang sama persis dengan
`data_pipeline_v2`, sehingga transaksi yang diprediksi diperlakukan identik
dengan transaksi saat pelatihan.

Pemakaian:
    from backend.ml.inference import AMLPredictor
    predictor = AMLPredictor.load(version="v5")   # ensemble 3 seed
    hasil = predictor.predict_dataframe(df)
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import backend.config as C
from backend.ml.data_pipeline import (
    CATEGORICAL_COLS,
    _motif_features,
    _temporal_cycle_features,
    add_context_features,
    add_transaction_flags,
    add_velocity_features,
    build_edge_features,
)
from backend.ml.model import EdgeGATModelV2

RISK_BANDS = [(0.30, "Low"), (0.70, "Moderate"), (0.90, "High")]


def risk_category(prob: float) -> str:
    """Petakan probabilitas ke empat tingkat risiko."""
    for upper, name in RISK_BANDS:
        if prob < upper:
            return name
    return "Critical"


@dataclass
class PredictionResult:
    """Hasil prediksi satu transaksi."""

    index: int
    sender: str
    receiver: str
    amount: float
    probability: float
    risk_category: str
    flagged: bool
    actual_label: Optional[int] = None
    laundering_type: Optional[str] = None


class AMLPredictor:
    """Pembungkus model terlatih beserta seluruh artefak pra-pemrosesannya."""

    # v4 adalah versi yang dipakai. Checkpoint v2 dan v3 dibuat sebelum
    # arsitektur diubah (classifier dipecah menjadi trunk + kepala, fitur edge
    # bertambah dari 24 menjadi 50), sehingga tidak lagi dapat dimuat oleh kelas
    # model saat ini. Keduanya tetap terdaftar untuk keperluan penelusuran, namun
    # perlu dilatih ulang bila hendak dipakai.
    VERSIONS = {
        # v5: model yang dilaporkan di naskah -- ensemble tiga seed (42, 1337,
        # 2024) yang probabilitasnya dirata-ratakan, threshold 0,7278 dari
        # evaluasi ensemble pada data validasi.
        "v5": {"graph": "DATA_V5_PATH", "artifacts": "ENCODERS_V5_PATH",
               "model": "MODEL_V5_PATH", "ensemble": "MODEL_V5_ENSEMBLE_PATHS",
               "metrics": "METRICS_V5_PATH", "evaluation": "EVAL_V5_PATH",
               "per_pola": "PER_POLA_V5_PATH",
               "history": "CSV_V5_PATH", "heads": 2,
               "edge_attention": True, "multitask": True,
               "edge_numeric_dim": "EDGE_NUMERIC_DIM_V5"},
        "v4": {"graph": "DATA_V4_PATH", "artifacts": "ENCODERS_V4_PATH",
               "model": "MODEL_V4_PATH", "metrics": "METRICS_V4_PATH",
               "history": "CSV_V4_PATH", "heads": 2,
               "edge_attention": True, "multitask": True},
        "v3": {"graph": "DATA_V3_PATH", "artifacts": "ENCODERS_V3_PATH",
               "model": "MODEL_V3_PATH", "metrics": "METRICS_V3_PATH",
               "history": "CSV_V3_PATH", "heads": 2},
        "v2": {"graph": "DATA_V2_PATH", "artifacts": "ENCODERS_V2_PATH",
               "model": "MODEL_V2_PATH", "metrics": "METRICS_V2_PATH",
               "history": "CSV_PATH", "heads": None},
    }

    # Kolom riwayat yang dibutuhkan untuk menghitung fitur velocity & pass-through
    HISTORY_COLUMNS = [
        "Time", "Date", "Sender_account", "Receiver_account", "Amount",
        "Payment_currency", "Received_currency", "Sender_bank_location",
        "Receiver_bank_location", "Payment_type",
    ]

    def __init__(self, model, graph, artifacts, threshold: float, version: str, device,
                 history: Optional[pd.DataFrame] = None, extra_models=None,
                 edge_numeric_dim: Optional[int] = None):
        self.model = model
        # Seluruh anggota ensemble; satu model saja untuk versi non-ensemble.
        self.models = [model] + list(extra_models or [])
        self.edge_numeric_dim = edge_numeric_dim
        self.graph = graph
        self.artifacts = artifacts
        self.threshold = threshold
        self.version = version
        self.device = device
        self.node_map: Dict[str, int] = artifacts["node_map"]
        self.encoders = artifacts["encoders"]
        self.edge_scaler = artifacts["edge_scaler"]
        self.profiles = artifacts["account_profiles"]
        self._node_embeddings: Optional[List[torch.Tensor]] = None
        self.history = history
        # Indeks transaksi graf: memungkinkan fitur yang sudah dihitung saat
        # pembangunan graf dipakai ulang, alih-alih dihitung ulang dari subgraf.
        # Fitur motif dan penutupan siklus bersifat GLOBAL; menghitungnya dari
        # potongan riwayat menghasilkan nilai yang berbeda dan menurunkan akurasi
        # drastis (terukur: FNR 0,13% menjadi 23,9% pada satu jendela uji).
        self._graph_index: Dict[str, int] = {}
        if history is not None and hasattr(graph, "edge_attr"):
            keys = self._transaction_keys(history)
            if len(keys) == graph.edge_attr.shape[0]:
                self._graph_index = {k: i for i, k in enumerate(keys)}
        if history is not None:
            # Indeks akun -> baris riwayat, agar penyaringan per permintaan cepat
            self._hist_sender = history.groupby("Sender_account").indices
            self._hist_receiver = history.groupby("Receiver_account").indices
        else:
            self._hist_sender = {}
            self._hist_receiver = {}

    @staticmethod
    def _transaction_keys(frame: pd.DataFrame) -> List[str]:
        """Kunci unik per transaksi, dipakai mencocokkan permintaan dengan graf."""
        return (frame["Date"].astype(str) + "|" + frame["Time"].astype(str) + "|"
                + frame["Sender_account"].astype(str) + "|"
                + frame["Receiver_account"].astype(str) + "|"
                + frame["Amount"].astype(float).round(2).astype(str)).tolist()

    # ── Pemuatan ──
    @classmethod
    def load(cls, version: str = "v4", device: Optional[str] = None) -> "AMLPredictor":
        """Muat model, graf, dan artefak untuk versi tertentu."""
        if version not in cls.VERSIONS:
            raise ValueError(f"Versi tidak dikenal: {version}. Pilih {list(cls.VERSIONS)}.")
        spec = cls.VERSIONS[version]

        graph_path = getattr(C, spec["graph"])
        artifacts_path = getattr(C, spec["artifacts"])
        model_path = getattr(C, spec["model"])
        model_paths = list(getattr(C, spec["ensemble"])) if spec.get("ensemble") else [model_path]
        metrics_path = getattr(C, spec["metrics"])
        edge_numeric_dim = (getattr(C, spec["edge_numeric_dim"]) if spec.get("edge_numeric_dim")
                            else C.EDGE_NUMERIC_DIM_V2)

        checks = [(graph_path, "graf"), (artifacts_path, "artefak")]
        checks += [(p, "model") for p in model_paths]
        for path, label in checks:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Berkas {label} tidak ditemukan: {path}. "
                    f"Jalankan `python -m backend.ml.data_pipeline --version {version}` "
                    f"lalu `python -m backend.ml.train --version {version}`."
                )

        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        graph = torch.load(graph_path, weights_only=False, map_location="cpu")
        artifacts = joblib.load(artifacts_path)

        def _build_model(path: str) -> EdgeGATModelV2:
            m = EdgeGATModelV2(
                node_feat_dim=C.NODE_FEAT_DIM_V2,
                edge_numeric_dim=edge_numeric_dim,
                cat_cardinalities=artifacts["cat_cardinalities"],
                cat_emb_dims=(C.CURRENCY_EMB_DIM, C.CURRENCY_EMB_DIM,
                              C.LOCATION_EMB_DIM, C.LOCATION_EMB_DIM, C.PAYMENT_TYPE_EMB_DIM),
                hidden_dim=C.HIDDEN_DIM_V2,
                num_heads=spec["heads"] or C.NUM_HEADS_V2,
                num_layers=C.NUM_LAYERS_V2,
                dropout=0.0,
                use_layer_norm=C.USE_LAYER_NORM_V2,
                use_residual=C.USE_RESIDUAL_V2,
                use_edge_attention=spec.get("edge_attention", False),
                # Jumlah kelas diambil dari artefak, bukan ditebak: kolom
                # Laundering_type memuat pula varian "Normal_*" dari SAML-D asli.
                num_typologies=(len(artifacts.get("laundering_type_classes", []))
                                if spec.get("multitask") else 0),
            ).to(dev)
            m.load_state_dict(torch.load(path, map_location=dev, weights_only=True))
            m.eval()
            return m

        models = [_build_model(p) for p in model_paths]
        model = models[0]

        import json
        threshold = 0.5
        if os.path.exists(metrics_path):
            with open(metrics_path, encoding="utf-8") as f:
                threshold = float(json.load(f).get("optimal_threshold", 0.5))
        # Untuk ensemble, threshold yang berlaku adalah milik ensemble (dipilih
        # di data validasi), bukan threshold model seed 42 saja.
        eval_path = getattr(C, spec["evaluation"], None) if spec.get("evaluation") else None
        if eval_path and os.path.exists(eval_path):
            with open(eval_path, encoding="utf-8") as f:
                threshold = float(json.load(f).get("threshold", threshold))

        # Riwayat transaksi diperlukan karena fitur velocity dan pass-through
        # dihitung relatif terhadap transaksi akun yang sudah terjadi. Tanpa
        # riwayat, fitur-fitur itu bernilai nol dan akurasi anjlok drastis
        # (diuji: recall 7% tanpa riwayat vs 96% dengan riwayat).
        history = None
        history_path = getattr(C, spec["history"], None)
        if history_path and os.path.exists(history_path):
            history = pd.read_csv(history_path, usecols=cls.HISTORY_COLUMNS)
            history["Sender_account"] = history["Sender_account"].astype(str)
            history["Receiver_account"] = history["Receiver_account"].astype(str)
            # Urutkan sama seperti saat graf dibangun (train, val, test; tiap
            # bagian kronologis) agar baris riwayat sejajar dengan baris graf.
            n_rows = len(history)
            stamps = pd.to_datetime(
                history["Date"].astype(str) + " " + history["Time"].astype(str),
                format="%Y-%m-%d %H:%M:%S", errors="coerce")
            bounds = [0, int(n_rows * 0.408867),
                      int(n_rows * 0.408867) + int(n_rows * 0.182273), n_rows]
            parts = []
            for lo, hi in zip(bounds[:-1], bounds[1:]):
                chunk = history.iloc[lo:hi].copy()
                chunk["_dt"] = stamps.iloc[lo:hi].values
                parts.append(chunk.sort_values("_dt", kind="mergesort"))
            history = pd.concat(parts, ignore_index=True).drop(columns=["_dt"])

        return cls(model, graph, artifacts, threshold, version, dev, history=history,
                   extra_models=models[1:], edge_numeric_dim=edge_numeric_dim)

    # ── Embedding node ──
    def _ensure_node_embeddings(self) -> List[torch.Tensor]:
        """
        Hitung embedding node satu kali lalu simpan di memori.

        Message passing dijalankan atas graf historis (seluruh transaksi yang
        sudah tercatat). Transaksi yang diprediksi hanya memakai embedding ini,
        tidak mengubahnya -- sesuai kondisi produksi di mana graf diperbarui
        secara berkala, bukan pada tiap permintaan.
        """
        if self._node_embeddings is None:
            with torch.no_grad():
                self._node_embeddings = [
                    m.encode_nodes(
                        self.graph.x.to(self.device), self.graph.edge_index.to(self.device),
                        self.graph.edge_attr.to(self.device), self.graph.edge_cat.to(self.device),
                    )
                    for m in self.models
                ]
        return self._node_embeddings

    # ── Rekayasa fitur ──
    def _map_accounts(self, df: pd.DataFrame) -> torch.Tensor:
        """
        Petakan akun ke indeks node. Akun yang belum dikenal diarahkan ke node
        cadangan berfitur nol, bukan ke node 0 (yang merupakan akun nyata).
        """
        n_nodes = self.graph.x.shape[0]
        src = df["Sender_account"].astype(str).map(self.node_map).fillna(n_nodes).astype(int).values
        dst = df["Receiver_account"].astype(str).map(self.node_map).fillna(n_nodes).astype(int).values
        return torch.tensor(np.stack([src, dst]), dtype=torch.long)

    def _relevant_history(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ambil transaksi historis milik akun-akun yang muncul pada permintaan.

        Hanya akun terkait yang diambil, bukan seluruh 1,67 juta baris, agar
        perhitungan fitur tetap cepat namun konteksnya tetap benar.
        """
        if self.history is None:
            return pd.DataFrame(columns=self.HISTORY_COLUMNS)

        def rows_for(accs):
            found = []
            for acc in accs:
                for index in (self._hist_sender, self._hist_receiver):
                    hit = index.get(acc)
                    if hit is not None:
                        found.append(hit)
            return np.unique(np.concatenate(found)) if found else np.array([], dtype=np.int64)

        accounts = set(df["Sender_account"].astype(str)) | set(df["Receiver_account"].astype(str))
        rows = rows_for(accounts)
        if len(rows) == 0:
            return pd.DataFrame(columns=self.HISTORY_COLUMNS)

        # Lapis kedua: akun yang bersinggungan dengan akun-akun di atas.
        # Fitur motif dan penutupan siklus bersifat global -- jalur A -> B -> C -> A
        # melewati akun perantara yang tidak bertransaksi langsung dengan akun pada
        # permintaan. Tanpa lapis ini, jalur tersebut tidak terlihat dan skor
        # menyimpang dari hasil pelatihan (terukur: korelasi 0,915 menjadi ~1,0).
        neighbours = set(self.history["Sender_account"].values[rows]) |             set(self.history["Receiver_account"].values[rows])
        second = rows_for(neighbours - accounts)
        if len(second):
            rows = np.unique(np.concatenate([rows, second]))
        hist = self.history.iloc[rows].copy()

        # Buang baris riwayat yang identik dengan baris permintaan. Berkas uji
        # diambil dari dataset yang sama, sehingga tanpa penyaringan ini tiap
        # transaksi terhitung dua kali dan fitur velocity menggelembung --
        # diuji: FPR melonjak dari 0,1% menjadi 20%.
        def _key(frame: pd.DataFrame) -> pd.Series:
            return (frame["Date"].astype(str) + "|" + frame["Time"].astype(str) + "|"
                    + frame["Sender_account"].astype(str) + "|"
                    + frame["Receiver_account"].astype(str) + "|"
                    + frame["Amount"].astype(float).round(2).astype(str))

        request_keys = set(_key(df))
        return hist[~_key(hist).isin(request_keys)].copy()

    def _build_features(self, df: pd.DataFrame):
        """
        Hitung fitur persis seperti saat pelatihan, dengan memakai ulang fungsi
        `data_pipeline_v2`.

        Menyalin logika fitur ke modul ini pernah menyebabkan jalur inferensi
        melenceng diam-diam ketika pipeline berubah. Dengan memanggil fungsi yang
        sama, keduanya mustahil berbeda.
        """
        work = df.copy()
        work["Sender_account"] = work["Sender_account"].astype(str)
        work["Receiver_account"] = work["Receiver_account"].astype(str)
        work["Amount"] = work["Amount"].astype(float)
        work["Datetime"] = pd.to_datetime(
            work["Date"].astype(str) + " " + work["Time"].astype(str),
            format="%Y-%m-%d %H:%M:%S", errors="coerce",
        )
        work["Datetime"] = work["Datetime"].fillna(pd.Timestamp("2022-10-07"))
        work["_is_request"] = True

        # Riwayat akun terkait sebagai konteks: fitur velocity dan pass-through
        # dihitung relatif terhadap transaksi akun yang sudah terjadi.
        history = self._relevant_history(work)
        if len(history):
            history = history.copy()
            history["_is_request"] = False
            history["Datetime"] = pd.to_datetime(
                history["Date"].astype(str) + " " + history["Time"].astype(str),
                format="%Y-%m-%d %H:%M:%S", errors="coerce",
            )
            history = history.dropna(subset=["Datetime"])
            combined = pd.concat([history, work], ignore_index=True)
        else:
            combined = work

        combined = combined.sort_values("Datetime", kind="mergesort").reset_index(drop=True)

        # Indeks kategorikal; nilai yang belum pernah dilihat diarahkan ke 0
        for col in CATEGORICAL_COLS:
            classes = {v: i for i, v in enumerate(self.encoders[col].classes_)}
            combined[f"{col}_idx"] = combined[col].astype(str).map(classes).fillna(0).astype(int)

        combined = add_transaction_flags(combined)
        combined = add_velocity_features(combined)
        combined = add_context_features(combined)

        # Motif dan siklus dihitung atas graf gabungan riwayat + permintaan
        n_nodes = self.graph.x.shape[0] + 1
        src = combined["Sender_account"].map(self.node_map).fillna(n_nodes - 1).astype(int).values
        dst = combined["Receiver_account"].map(self.node_map).fillna(n_nodes - 1).astype(int).values
        motif = _motif_features(src, dst, n_nodes)
        t_sec = combined["Datetime"].values.astype("datetime64[s]").astype(np.int64)
        motif.update(_temporal_cycle_features(src, dst, t_sec, n_nodes))

        numeric_all, cat_all, _ = build_edge_features(
            combined, self.profiles, train_end=len(combined),
            motif=motif, scaler=self.edge_scaler,
        )

        # Pipeline kini selalu menghasilkan 58 fitur (v5). Model v4 dilatih
        # dengan 50 fitur pertama yang urutannya identik, jadi cukup dipotong.
        if self.edge_numeric_dim and numeric_all.shape[1] > self.edge_numeric_dim:
            numeric_all = numeric_all[:, : self.edge_numeric_dim]

        keep = combined["_is_request"].values
        request = combined[keep].reset_index(drop=True)
        return numeric_all[keep], cat_all[keep], request

    # ── Prediksi ──
    @torch.no_grad()
    def predict_dataframe(self, df: pd.DataFrame) -> List[PredictionResult]:
        """Prediksi seluruh baris DataFrame. Kolom label diabaikan bila ada."""
        if df.empty:
            return []

        # Transaksi yang sudah ada di graf memakai fitur hasil perhitungan penuh;
        # sisanya dihitung dari subgraf riwayat.
        keys = self._transaction_keys(df)
        known = np.array([self._graph_index.get(k, -1) for k in keys])
        n_known = int((known >= 0).sum())

        if n_known == len(df) and n_known > 0:
            numeric = self.graph.edge_attr[known]
            cat = self.graph.edge_cat[known]
            work = df.copy()
            work["Sender_account"] = work["Sender_account"].astype(str)
            work["Receiver_account"] = work["Receiver_account"].astype(str)
        else:
            numeric, cat, work = self._build_features(df)

        edge_index = self._map_accounts(work)

        # Ensemble: probabilitas tiap anggota dirata-ratakan, sama seperti
        # evaluasi pada naskah (scripts/eval/evaluate_v5_ensemble.py).
        member_probs = []
        for m, h in zip(self.models, self._ensure_node_embeddings()):
            # Node cadangan berfitur nol untuk akun yang belum dikenal
            h_ext = torch.cat([h, torch.zeros(1, h.shape[1], device=self.device)], dim=0)
            logits = m.score_edges(
                h_ext, edge_index.to(self.device),
                numeric.to(self.device), cat.to(self.device),
                subset=None, chunked=True,
            )
            member_probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        probs = np.mean(member_probs, axis=0)

        has_label = "Is_laundering" in df.columns
        has_type = "Laundering_type" in df.columns
        results = []
        for i, prob in enumerate(probs):
            p = float(prob)
            results.append(PredictionResult(
                index=i,
                sender=str(work["Sender_account"].iloc[i]),
                receiver=str(work["Receiver_account"].iloc[i]),
                amount=float(work["Amount"].iloc[i]),
                probability=round(p, 6),
                risk_category=risk_category(p),
                flagged=bool(p >= self.threshold),
                actual_label=int(df["Is_laundering"].iloc[i]) if has_label else None,
                laundering_type=str(df["Laundering_type"].iloc[i]) if has_type else None,
            ))
        return results

    @staticmethod
    def score_results(results: List[PredictionResult]) -> Dict:
        """Hitung confusion matrix dan metrik bila kunci jawaban tersedia."""
        labelled = [r for r in results if r.actual_label is not None]
        if not labelled:
            flagged = sum(r.flagged for r in results)
            return {"labeled": False, "total": len(results), "flagged": flagged}

        tp = sum(1 for r in labelled if r.flagged and r.actual_label == 1)
        fp = sum(1 for r in labelled if r.flagged and r.actual_label == 0)
        fn = sum(1 for r in labelled if not r.flagged and r.actual_label == 1)
        tn = sum(1 for r in labelled if not r.flagged and r.actual_label == 0)

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        by_type: Dict[str, Dict[str, int]] = {}
        for r in labelled:
            if r.actual_label == 1 and r.laundering_type:
                entry = by_type.setdefault(r.laundering_type, {"total": 0, "terdeteksi": 0})
                entry["total"] += 1
                entry["terdeteksi"] += int(r.flagged)

        return {
            "labeled": True,
            "total": len(labelled),
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(2 * precision * recall / max(precision + recall, 1e-12), 6),
            "accuracy": round((tp + tn) / max(len(labelled), 1), 6),
            "fpr": round(fp / max(fp + tn, 1), 6),
            "fnr": round(fn / max(fn + tp, 1), 6),
            "per_tipologi": by_type,
        }


REQUIRED_COLUMNS = [
    "Time", "Date", "Sender_account", "Receiver_account", "Amount",
    "Payment_currency", "Received_currency", "Sender_bank_location",
    "Receiver_bank_location", "Payment_type",
]


def validate_csv(df: pd.DataFrame) -> List[str]:
    """Periksa kelengkapan kolom CSV yang diunggah. Kembalikan daftar masalah."""
    problems = []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        problems.append(f"Kolom wajib tidak ada: {', '.join(missing)}")
    if len(df) == 0:
        problems.append("Berkas tidak berisi baris data.")
    if "Amount" in df.columns and not pd.api.types.is_numeric_dtype(df["Amount"]):
        try:
            pd.to_numeric(df["Amount"])
        except (ValueError, TypeError):
            problems.append("Kolom Amount memuat nilai yang bukan angka.")
    return problems


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Uji inferensi atas berkas CSV.")
    parser.add_argument("csv", help="Berkas CSV yang akan diprediksi.")
    parser.add_argument("--version", choices=["v2", "v3", "v4", "v5"], default="v5")
    args = parser.parse_args()

    predictor = AMLPredictor.load(version=args.version)
    frame = pd.read_csv(args.csv)
    issues = validate_csv(frame)
    if issues:
        print("Berkas tidak valid:")
        for issue in issues:
            print(f"  - {issue}")
        raise SystemExit(1)

    preds = predictor.predict_dataframe(frame)
    print(json.dumps(AMLPredictor.score_results(preds), indent=2, ensure_ascii=False))
