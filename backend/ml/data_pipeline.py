"""
data_pipeline_v2.py — Feature Engineering v2 (stationary + behavioural)
========================================================================
Membangun ulang graf PyG dengan fitur yang memperbaiki tiga kelemahan
pipeline v1 (lihat docs/docs md/RINGKASAN_PROYEK.md §13):

1. **Fitur temporal absolut dibuang.**
   v1 memakai `delta_t_minutes` (jarak dari transaksi paling awal) dan
   `temporal_weight = Amount / (delta_t + 1)`. Karena split bersifat
   temporal, kedua fitur ini tidak beririsan antara train dan test
   (delta_t train [-1.75, 1.72] vs test [3.39, 7.05]) dan temporal_weight
   kolaps menjadi konstanta di test (std 0.0087 vs 1.0 saat train).
   Penggantinya adalah fitur waktu **stasioner**: jam/hari siklikal dan
   jeda waktu relatif terhadap transaksi sebelumnya dari akun yang sama.

2. **Kategorikal tidak lagi di-scale sebagai angka ordinal.**
   v1 melakukan StandardScaler atas hasil LabelEncoder, sehingga
   "Euro=3" dan "Yen=7" punya jarak numerik yang tidak bermakna.
   v2 menyimpan indeks kategori apa adanya untuk dipakai sebagai
   input `nn.Embedding` pada model v2.

3. **Fitur perilaku (behavioural) ditambahkan.**
   Tipologi yang paling sering terlewat (Structuring, Cash_Withdrawal,
   Deposit-Send, Behavioural_Change) tidak dikenali dari struktur graf
   melainkan dari pola nominal dan kecepatan transaksi. v2 menambahkan
   velocity 1 jam/24 jam, deviasi nominal terhadap profil akun, penanda
   nominal bulat, dan penanda nominal tepat di bawah ambang pelaporan.

Semua statistik per-akun dihitung **hanya dari edge train** untuk mencegah
kebocoran data. Fitur velocity bersifat *backward-looking* (hanya melihat
transaksi yang sudah terjadi), sehingga aman dipakai lintas split.

Usage:
    python -m backend.ml.data_pipeline
"""

import os
import sys
import warnings
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch_geometric.data import Data

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.config import (
    CSV_PATH,
    DATA_V2_PATH,
    DATA_V3_PATH,
    DATA_V4_PATH,
    ENCODERS_V2_PATH,
    ENCODERS_V3_PATH,
    ENCODERS_V4_PATH,
    SYNTHETIC_LAUNDERING_TYPES,
    TEST_CSV_PATH,
    TEST_CSV_V3_PATH,
    TEST_CSV_V4_PATH,
    TRAIN_CSV_PATH,
    TRAIN_CSV_V3_PATH,
    TRAIN_CSV_V4_PATH,
    VAL_CSV_PATH,
    VAL_CSV_V3_PATH,
    VAL_CSV_V4_PATH,
    DATA_V5_PATH,
    ENCODERS_V5_PATH,
)

# Dipilih lewat argumen --version; v3 memakai dataset dengan seluruh tipologi terwakili.
PATHS = {
    "v2": {"train": TRAIN_CSV_PATH, "val": VAL_CSV_PATH, "test": TEST_CSV_PATH,
           "graph": DATA_V2_PATH, "artifacts": ENCODERS_V2_PATH},
    "v3": {"train": TRAIN_CSV_V3_PATH, "val": VAL_CSV_V3_PATH, "test": TEST_CSV_V3_PATH,
           "graph": DATA_V3_PATH, "artifacts": ENCODERS_V3_PATH},
    "v4": {"train": TRAIN_CSV_V4_PATH, "val": VAL_CSV_V4_PATH, "test": TEST_CSV_V4_PATH,
           "graph": DATA_V4_PATH, "artifacts": ENCODERS_V4_PATH},
    # v5 memakai CSV yang sama dengan v4 -- datanya tidak berubah. Yang berbeda
    # hanya fitur (tambahan konteks graf) sehingga grafnya dibangun terpisah.
    "v5": {"train": TRAIN_CSV_V4_PATH, "val": VAL_CSV_V4_PATH, "test": TEST_CSV_V4_PATH,
           "graph": DATA_V5_PATH, "artifacts": ENCODERS_V5_PATH},
}
ACTIVE = PATHS["v2"]

warnings.filterwarnings("ignore")

CATEGORICAL_COLS = [
    "Payment_currency",
    "Received_currency",
    "Sender_bank_location",
    "Receiver_bank_location",
    "Payment_type",
]

LABEL_COL = "Is_laundering"

# Ambang pelaporan yang lazim dipakai regulator (structuring bergerak tepat di bawahnya)
REPORTING_THRESHOLDS = [10_000.0, 15_000.0, 50_000.0]
NEAR_THRESHOLD_MARGIN = 0.10  # 10% di bawah ambang dianggap mencurigakan

NUM_NODE_FEATURES = 20
NUM_EDGE_NUMERIC = 58

# Batas atas fitur ketersediaan konteks; di atas ini tambahan derajat tidak
# menambah informasi, hanya menciptakan jalan pintas "akun sibuk = laundering".
CONTEXT_CAP = 20.0


# ─── 1. Load ──────────────────────────────────────────────────────────────────
def load_split_csvs() -> pd.DataFrame:
    """Muat train/val/test CSV, tandai asal split, urutkan kronologis per split."""
    print("[1/7] Memuat CSV per split ...")
    dfs = []
    for split_name, path in [
        ("train", ACTIVE["train"]),
        ("val", ACTIVE["val"]),
        ("test", ACTIVE["test"]),
    ]:
        df = pd.read_csv(path)
        df["_split"] = split_name
        dfs.append(df)
        print(f"      {split_name:>6}: {len(df):,} baris")

    df = pd.concat(dfs, ignore_index=True)
    df["Datetime"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )

    critical = ["Sender_account", "Receiver_account", "Amount", LABEL_COL]
    before = len(df)
    df = df.dropna(subset=critical).reset_index(drop=True)
    if before != len(df):
        print(f"      Dibuang {before - len(df)} baris dengan field kritis kosong.")

    df["Sender_account"] = df["Sender_account"].astype(str)
    df["Receiver_account"] = df["Receiver_account"].astype(str)
    df["Amount"] = df["Amount"].astype(float)
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    print(f"      Gabungan: {df.shape}")
    return df


# ─── 2. Encode kategorikal (indeks mentah untuk embedding) ───────────────────
def encode_categoricals(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """Label-encode kolom kategorikal. Indeks TIDAK di-scale (dipakai embedding)."""
    print("[2/7] Encoding kategorikal (indeks untuk embedding) ...")
    encoders: Dict[str, LabelEncoder] = {}

    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[f"{col}_idx"] = le.fit_transform(df[col].astype(str).fillna("UNKNOWN"))
        encoders[col] = le
        print(f"      {col}: {len(le.classes_)} kategori")

    le_lt = LabelEncoder()
    df["Laundering_type_idx"] = le_lt.fit_transform(df["Laundering_type"].astype(str).fillna("UNKNOWN"))
    encoders["Laundering_type"] = le_lt

    return df, encoders


# ─── 3. Fitur turunan per transaksi ───────────────────────────────────────────
def add_transaction_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Penanda level transaksi: lintas negara, beda mata uang, nominal bulat/near-threshold."""
    print("[3/7] Menghitung penanda per transaksi ...")
    df["is_cross_border"] = (
        df["Sender_bank_location"].astype(str) != df["Receiver_bank_location"].astype(str)
    ).astype(np.float32)
    df["is_currency_mismatch"] = (
        df["Payment_currency"].astype(str) != df["Received_currency"].astype(str)
    ).astype(np.float32)

    # Nominal pada dataset ini hampir selalu berdesimal; yang "terlalu rapi"
    # (bilangan bulat tanpa sen) hanya ~1% dan lazim pada transfer yang disusun manual.
    amount = df["Amount"].values
    df["is_round_amount"] = np.isclose(np.mod(amount, 1.0), 0.0).astype(np.float32)

    near = np.zeros(len(df), dtype=np.float32)
    for thr in REPORTING_THRESHOLDS:
        near = np.maximum(
            near, ((amount >= thr * (1 - NEAR_THRESHOLD_MARGIN)) & (amount < thr)).astype(np.float32)
        )
    df["is_near_threshold"] = near

    print(f"      Lintas negara     : {df['is_cross_border'].mean()*100:.2f}%")
    print(f"      Beda mata uang    : {df['is_currency_mismatch'].mean()*100:.2f}%")
    print(f"      Nominal bulat     : {df['is_round_amount'].mean()*100:.2f}%")
    print(f"      Dekat ambang lapor: {df['is_near_threshold'].mean()*100:.2f}%")
    return df


_KEY_SHIFT = 1 << 32  # t_sec (~1.7e9) selalu < 2^32, jadi aman sebagai offset per akun


def _per_account_window_stats(
    account_codes: np.ndarray,
    times_sec: np.ndarray,
    amounts: np.ndarray,
    windows_sec: Tuple[int, ...],
    forward: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Hitung jeda antar-transaksi dan statistik jendela per akun secara vektorial.

    Trik: bentuk kunci gabungan `code * 2^32 + t_sec` lalu urutkan. Karena
    `t_sec - window` tidak pernah keluar dari blok akun yang sama, satu
    `searchsorted` global setara dengan pencarian per akun — tanpa loop
    Python atas ratusan ribu akun.

    Seluruh statistik bersifat backward-looking: transaksi ke-i hanya
    melihat transaksi akun yang sama yang terjadi SEBELUMNYA.
    """
    n = len(times_sec)
    key = account_codes.astype(np.int64) * _KEY_SHIFT + times_sec
    order = np.argsort(key, kind="stable")
    key_sorted = key[order]
    amounts_sorted = amounts[order]

    idx = np.arange(n)
    is_group_start = np.empty(n, dtype=bool)
    is_group_start[0] = True
    np.not_equal(account_codes[order][1:], account_codes[order][:-1], out=is_group_start[1:])

    # Jeda ke transaksi sebelumnya dari akun yang sama
    t_sorted = key_sorted - account_codes[order].astype(np.int64) * _KEY_SHIFT
    gap = np.empty(n, dtype=np.float64)
    gap[0] = 7 * 24 * 3600
    gap[1:] = t_sorted[1:] - t_sorted[:-1]
    gap[is_group_start] = 7 * 24 * 3600  # transaksi pertama akun: jeda dianggap sangat lama

    cumsum = np.concatenate([[0.0], np.cumsum(amounts_sorted)])
    out: Dict[str, np.ndarray] = {}
    inv = np.empty(n, dtype=np.int64)
    inv[order] = idx

    for w in windows_sec:
        if forward:
            # Jendela MAJU: melihat transaksi akun yang sama SETELAH transaksi ini.
            # Dibutuhkan karena seluruh fitur velocity lain bersifat mundur, sehingga
            # transaksi PERTAMA dari sebuah pola ledakan tampak normal -- ledakannya
            # baru terjadi sesudahnya. Terukur: fitur mundur pada false negative
            # rata-rata 0,04 sedangkan fitur majunya 8,19 (transaksi normal 0,09).
            lo = np.searchsorted(key_sorted, key_sorted, side="right")
            hi = np.searchsorted(key_sorted, key_sorted + w, side="right")
        else:
            lo = np.searchsorted(key_sorted, key_sorted - w, side="left")
            hi = idx
        counts = (hi - lo).astype(np.float32)
        sums = (cumsum[hi] - cumsum[lo]).astype(np.float32)
        out[f"cnt_{w}"] = counts[inv]
        out[f"sum_{w}"] = sums[inv]

    out["gap_min"] = (gap / 60.0).astype(np.float32)[inv]
    return out


def _passthrough_stats(
    sender_codes: np.ndarray,
    receiver_codes: np.ndarray,
    times_sec: np.ndarray,
    amounts: np.ndarray,
    window_sec: int = 24 * 3600,
) -> Dict[str, np.ndarray]:
    """
    Hitung ciri *pass-through*: apakah dana yang masuk segera diteruskan keluar.

    Ini adalah inti dari seluruh tipologi layering (Cycle, Layered Fan-Out/In,
    Scatter-Gather, Bipartite bertingkat). Fitur velocity per akun tidak
    menangkapnya, karena yang menjadi ciri bukan banyaknya transaksi melainkan
    *keterhubungan* antara dana masuk dan dana keluar berikutnya.

    Untuk tiap transaksi u -> v pada waktu t dengan nominal A:
      - forward_sum : total dana yang dikirim v ke pihak lain pada (t, t+window]
      - forward_cnt : jumlah transaksi keluar v pada jendela tersebut
      - forward_gap : jeda sampai transaksi keluar pertama dari v
      - inflow_sum  : total dana yang diterima u pada [t-window, t)
      - returns     : jumlah transaksi balik v -> u pada (t, t+window]

    Seluruhnya kausal: hanya melihat transaksi pada jendela waktu setelah (atau
    sebelum) transaksi yang bersangkutan, tanpa memakai label apa pun.
    """
    n = len(times_sec)

    # Blok transaksi KELUAR, diurutkan per akun pengirim lalu waktu
    key_out = sender_codes.astype(np.int64) * _KEY_SHIFT + times_sec
    order_out = np.argsort(key_out, kind="stable")
    key_out_sorted = key_out[order_out]
    cum_out = np.concatenate([[0.0], np.cumsum(amounts[order_out])])
    t_out_sorted = times_sec[order_out]

    # Blok transaksi MASUK, diurutkan per akun penerima lalu waktu
    key_in = receiver_codes.astype(np.int64) * _KEY_SHIFT + times_sec
    order_in = np.argsort(key_in, kind="stable")
    key_in_sorted = key_in[order_in]
    cum_in = np.concatenate([[0.0], np.cumsum(amounts[order_in])])

    # Dana keluar dari penerima setelah transaksi ini
    base_v = receiver_codes.astype(np.int64) * _KEY_SHIFT
    lo = np.searchsorted(key_out_sorted, base_v + times_sec, side="right")
    hi = np.searchsorted(key_out_sorted, base_v + times_sec + window_sec, side="right")
    forward_sum = (cum_out[hi] - cum_out[lo]).astype(np.float32)
    forward_cnt = (hi - lo).astype(np.float32)

    # Jeda sampai transaksi keluar pertama dari penerima
    nxt = np.clip(lo, 0, n - 1)
    forward_gap = np.where(lo < hi, t_out_sorted[nxt] - times_sec, window_sec).astype(np.float64)
    forward_gap = np.maximum(forward_gap, 0.0) / 60.0

    # Dana masuk ke pengirim sebelum transaksi ini
    base_u = sender_codes.astype(np.int64) * _KEY_SHIFT
    lo_in = np.searchsorted(key_in_sorted, base_u + times_sec - window_sec, side="left")
    hi_in = np.searchsorted(key_in_sorted, base_u + times_sec, side="left")
    inflow_sum = (cum_in[hi_in] - cum_in[lo_in]).astype(np.float32)

    # Dana masuk ke pengirim SETELAH transaksi ini -- ciri siklus: uang yang
    # dikirim kembali lagi ke pengirimnya. Berbeda dari fitur motif yang hanya
    # menangkap siklus tiga langkah, ciri ini berlaku untuk siklus sepanjang apa
    # pun karena yang diperiksa adalah kembalinya dana, bukan panjang jalurnya.
    lo_back = np.searchsorted(key_in_sorted, base_u + times_sec, side="right")
    hi_back = np.searchsorted(key_in_sorted, base_u + times_sec + window_sec, side="right")
    return_sum = (cum_in[hi_back] - cum_in[lo_back]).astype(np.float32)
    return_cnt = (hi_back - lo_back).astype(np.float32)

    # Transaksi balik v -> u dalam jendela yang sama (indikasi siklus)
    pair_fwd = pd.factorize(
        pd.Series(sender_codes).astype(str) + ">" + pd.Series(receiver_codes).astype(str),
        sort=False,
    )
    pair_codes, pair_labels = pair_fwd[0], pair_fwd[1]
    label_to_code = {lab: i for i, lab in enumerate(pair_labels)}
    reverse_label = pd.Series(receiver_codes).astype(str) + ">" + pd.Series(sender_codes).astype(str)
    reverse_code = reverse_label.map(label_to_code).fillna(-1).values.astype(np.int64)

    key_pair = pair_codes.astype(np.int64) * _KEY_SHIFT + times_sec
    order_pair = np.argsort(key_pair, kind="stable")
    key_pair_sorted = key_pair[order_pair]

    has_reverse = reverse_code >= 0
    returns = np.zeros(n, dtype=np.float32)
    if has_reverse.any():
        base_r = reverse_code[has_reverse] * _KEY_SHIFT
        t_r = times_sec[has_reverse]
        lo_r = np.searchsorted(key_pair_sorted, base_r + t_r, side="right")
        hi_r = np.searchsorted(key_pair_sorted, base_r + t_r + window_sec, side="right")
        returns[has_reverse] = (hi_r - lo_r).astype(np.float32)

    return {
        "forward_sum": forward_sum,
        "forward_cnt": forward_cnt,
        "forward_gap_min": forward_gap.astype(np.float32),
        "inflow_sum": inflow_sum,
        "return_sum": return_sum,
        "return_cnt": return_cnt,
        "returns": returns,
    }


def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fitur kecepatan transaksi per akun (backward-looking, stasioner).

    Menangkap smurfing/structuring/rapid-burst: banyak transaksi kecil
    beruntun dari akun yang sama dalam jendela waktu pendek.
    """
    print("[4/7] Menghitung fitur velocity per akun ...")
    df = df.sort_values("Datetime", kind="mergesort").reset_index(drop=True)
    t_sec = df["Datetime"].values.astype("datetime64[s]").astype(np.int64)
    amounts = df["Amount"].values.astype(np.float64)

    # Kode akun harus berasal dari ruang yang SAMA untuk pengirim dan penerima.
    # Bila di-factorize terpisah, kode 5 pada kolom pengirim bukan akun yang sama
    # dengan kode 5 pada kolom penerima -- fitur pass-through akan mencari akun
    # yang keliru dan selalu menghasilkan nol.
    n_tx = len(df)
    unified = pd.factorize(
        pd.concat([df["Sender_account"], df["Receiver_account"]], ignore_index=True),
        sort=False,
    )[0]
    sender_codes = unified[:n_tx]
    receiver_codes = unified[n_tx:]
    # Pasangan pengirim->penerima sebagai satu entitas: membedakan mitra lama
    # (transfer rutin) dari mitra baru (ciri fan-out / scatter).
    pair_codes = pd.factorize(
        df["Sender_account"].astype(str) + ">" + df["Receiver_account"].astype(str), sort=False
    )[0]

    week = 7 * 24 * 3600
    s = _per_account_window_stats(sender_codes, t_sec, amounts, (3600, 24 * 3600, week))
    r = _per_account_window_stats(receiver_codes, t_sec, amounts, (3600, 24 * 3600))
    pair = _per_account_window_stats(pair_codes, t_sec, amounts, (week,))

    # Hitung transaksi near-threshold milik pengirim dalam 7 hari terakhir:
    # inti pola structuring -- memecah dana agar tiap transaksi tetap di bawah
    # ambang pelaporan. Satu transaksi tunggal tidak mencurigakan; pengulangannya
    # yang mencurigakan.
    near_flags = df["is_near_threshold"].values.astype(np.float64)
    s_near = _per_account_window_stats(sender_codes, t_sec, near_flags, (week,))

    df["sender_gap_min"] = s["gap_min"]
    df["sender_cnt_1h"] = s["cnt_3600"]
    df["sender_sum_1h"] = s["sum_3600"]
    df["sender_cnt_24h"] = s["cnt_86400"]
    df["sender_sum_24h"] = s["sum_86400"]
    df["sender_cnt_7d"] = s[f"cnt_{week}"]
    df["sender_near_thr_7d"] = s_near[f"sum_{week}"]
    df["receiver_gap_min"] = r["gap_min"]
    df["receiver_cnt_1h"] = r["cnt_3600"]
    df["receiver_cnt_24h"] = r["cnt_86400"]
    df["pair_gap_min"] = pair["gap_min"]
    df["pair_cnt_7d"] = pair[f"cnt_{week}"]

    # Jumlah MITRA BARU dalam jendela waktu -- tanda tangan pola fan.
    # Pola fan-out berarti mengirim ke banyak pihak BERBEDA dalam waktu singkat;
    # fitur `sender_cnt_1h` saja tidak membedakannya dari pembayaran berulang ke
    # pihak yang sama. Sebuah transaksi dianggap "mitra baru" bila pasangan
    # pengirim-penerima itu belum pernah bertransaksi dalam 7 hari terakhir.
    is_new_partner = (pair[f"cnt_{week}"] == 0).astype(np.float64)
    s_new = _per_account_window_stats(sender_codes, t_sec, is_new_partner, (3600, 24 * 3600))
    r_new = _per_account_window_stats(receiver_codes, t_sec, is_new_partner, (3600, 24 * 3600))
    df["sender_new_partner_1h"] = s_new["sum_3600"]
    df["sender_new_partner_24h"] = s_new["sum_86400"]
    df["receiver_new_partner_1h"] = r_new["sum_3600"]
    df["receiver_new_partner_24h"] = r_new["sum_86400"]
    print(f"      Maks mitra baru pengirim / 1 jam : {df['sender_new_partner_1h'].max():.0f}")
    print(f"      Maks mitra baru penerima / 1 jam : {df['receiver_new_partner_1h'].max():.0f}")

    # ── Jendela MAJU ──
    # Pola pencucian uang berupa ledakan transaksi: edge pertamanya tidak memiliki
    # pendahulu sehingga seluruh fitur mundur bernilai nol dan tampak normal.
    # Jendela maju melihat transaksi yang terjadi SESUDAHNYA, sehingga edge pertama
    # sebuah fan-out ikut mengetahui bahwa 29 transaksi lain menyusul.
    #
    # Konsekuensi operasional: penilaian tidak dapat dilakukan pada detik transaksi
    # terjadi, melainkan perlu jeda singkat (mis. satu jam) atau dijalankan per batch.
    # Hal ini lazim pada sistem transaction monitoring AML yang memang berjalan
    # berkala, bukan seketika.
    s_fwd = _per_account_window_stats(sender_codes, t_sec, amounts, (3600, 24 * 3600), forward=True)
    r_fwd = _per_account_window_stats(receiver_codes, t_sec, amounts, (3600, 24 * 3600), forward=True)
    s_fwd_new = _per_account_window_stats(sender_codes, t_sec, is_new_partner, (3600, 24 * 3600), forward=True)
    r_fwd_new = _per_account_window_stats(receiver_codes, t_sec, is_new_partner, (3600, 24 * 3600), forward=True)

    df["sender_cnt_fwd_1h"] = s_fwd["cnt_3600"]
    df["sender_cnt_fwd_24h"] = s_fwd["cnt_86400"]
    df["sender_sum_fwd_1h"] = s_fwd["sum_3600"]
    df["sender_new_fwd_1h"] = s_fwd_new["sum_3600"]
    df["sender_new_fwd_24h"] = s_fwd_new["sum_86400"]
    df["receiver_cnt_fwd_1h"] = r_fwd["cnt_3600"]
    df["receiver_cnt_fwd_24h"] = r_fwd["cnt_86400"]
    df["receiver_new_fwd_1h"] = r_fwd_new["sum_3600"]
    df["receiver_new_fwd_24h"] = r_fwd_new["sum_86400"]
    print(f"      Maks mitra baru pengirim (maju, 1 jam): {df['sender_new_fwd_1h'].max():.0f}")
    print(f"      Maks mitra baru penerima (maju, 1 jam): {df['receiver_new_fwd_1h'].max():.0f}")

    # Ciri pass-through: dana masuk yang langsung diteruskan keluar
    pt = _passthrough_stats(sender_codes, receiver_codes, t_sec, amounts)
    df["fwd_sum_24h"] = pt["forward_sum"]
    df["fwd_cnt_24h"] = pt["forward_cnt"]
    df["fwd_gap_min"] = pt["forward_gap_min"]
    df["inflow_sum_24h"] = pt["inflow_sum"]
    df["return_sum_24h"] = pt["return_sum"]
    df["return_cnt_24h"] = pt["return_cnt"]
    df["returns_24h"] = pt["returns"]
    print(f"      Median dana diteruskan penerima (24 jam): {np.median(pt['forward_sum']):,.0f}")
    print(f"      Transaksi dengan aliran balik   : {(pt['returns'] > 0).mean()*100:.2f}%")
    print(f"      Dana kembali ke pengirim (24 jam): {(pt['return_cnt'] > 0).mean()*100:.2f}%")

    print(f"      Median jeda transaksi pengirim : {np.median(df['sender_gap_min']):.1f} menit")
    print(f"      Maks transaksi pengirim / 1 jam: {df['sender_cnt_1h'].max():.0f}")
    print(f"      Maks transaksi pengirim / 24 jam: {df['sender_cnt_24h'].max():.0f}")
    return df


# ─── 4. Profil akun dari TRAIN saja ──────────────────────────────────────────

def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Seberapa banyak konteks yang dimiliki model untuk sebuah transaksi.

    Tanpa ini, fitur velocity bernilai nol punya dua arti yang tidak bisa
    dibedakan model: "akun ini tenang" dan "akun ini belum pernah terlihat".
    Keduanya diperlakukan sama, sehingga pada akun berderajat rendah model
    jatuh ke prior kategorikal -- terukur: FPR 0,455% dan FNR 96,3% pada
    derajat < 10, berbanding 0,027% dan 0,030% pada derajat >= 80.

    Fitur di sini menyatakan ketersediaan konteks secara eksplisit, agar model
    dapat mempelajari perilaku terpisah untuk kondisi minim-informasi alih-alih
    menerapkan prior wilayah padat.
    """
    print("      Fitur konteks graf ...")
    n = len(df)
    sender = df["Sender_account"].values
    receiver = df["Receiver_account"].values

    # Derajat total pada seluruh graf. Bersifat transduktif, sama seperti fitur
    # motif dan penutupan siklus yang sudah dipakai sejak v3.
    deg = pd.concat([df["Sender_account"], df["Receiver_account"]]).value_counts()
    df["sender_degree"] = df["Sender_account"].map(deg).fillna(0).values
    df["receiver_degree"] = df["Receiver_account"].map(deg).fillna(0).values

    # Cacah transaksi sebelumnya -- kausal, hanya melihat ke belakang.
    df["sender_n_prior"] = df.groupby("Sender_account").cumcount().values
    df["receiver_n_prior"] = df.groupby("Receiver_account").cumcount().values

    # Lama akun sudah teramati saat transaksi ini terjadi
    t = df["Datetime"].values.astype("datetime64[s]").astype(np.int64)
    first_seen = pd.Series(t).groupby(df["Sender_account"].values).transform("min").values
    df["sender_age_hours"] = np.maximum(t - first_seen, 0) / 3600.0

    return df

def build_account_profiles(train_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Agregasi profil akun HANYA dari transaksi train (anti-leakage)."""
    sender = train_df.groupby("Sender_account").agg(
        total_sent=("Amount", "sum"),
        cnt_sent=("Amount", "count"),
        mean_sent=("Amount", "mean"),
        std_sent=("Amount", "std"),
        max_sent=("Amount", "max"),
        uniq_receivers=("Receiver_account", "nunique"),
        foreign_sent=("is_currency_mismatch", "sum"),
        xborder_sent=("is_cross_border", "sum"),
        round_sent=("is_round_amount", "sum"),
        nearthr_sent=("is_near_threshold", "sum"),
        median_gap=("sender_gap_min", "median"),
    )
    sender["std_sent"] = sender["std_sent"].fillna(0.0)

    receiver = train_df.groupby("Receiver_account").agg(
        total_received=("Amount", "sum"),
        cnt_received=("Amount", "count"),
        mean_received=("Amount", "mean"),
        uniq_senders=("Sender_account", "nunique"),
        foreign_recv=("is_currency_mismatch", "sum"),
        xborder_recv=("is_cross_border", "sum"),
    )
    return {"sender": sender, "receiver": receiver}


def _motif_features(
    src: np.ndarray, dst: np.ndarray, num_nodes: int
) -> Dict[str, np.ndarray]:
    """
    Hitung ciri motif subgraf untuk tiap edge.

    Fitur velocity dan pass-through hanya melihat satu akun atau satu pasangan.
    Pola seperti Simple Cycle (A->B->C->A) dan Scatter-Gather (sumber dan target
    terhubung lewat banyak perantara) baru terlihat pada struktur dua langkah,
    sehingga perlu dihitung terpisah.

    Untuk tiap edge u -> v:
      - cycle_3hop  : banyaknya jalur v -> w -> u  (menutup siklus tiga langkah)
      - common_mid  : banyaknya jalur u -> w -> v  (ciri scatter-gather)
      - reciprocal  : apakah ada edge balik v -> u
      - jaccard     : kemiripan himpunan tetangga u dan v

    Dihitung dengan perkalian matriks ketetanggaan sparse; tidak memakai label
    apa pun sehingga aman dari kebocoran.
    """
    from scipy.sparse import csr_matrix

    print("      Menghitung fitur motif (perkalian matriks sparse) ...")
    n_edges = len(src)
    data = np.ones(n_edges, dtype=np.float32)
    A = csr_matrix((data, (src, dst)), shape=(num_nodes, num_nodes))
    A.sum_duplicates()
    A.data[:] = 1.0  # ketetanggaan biner: keberadaan hubungan, bukan frekuensinya

    # Jalur dua langkah. Simpul berderajat sangat tinggi dibatasi agar hasil
    # perkalian tidak meledak; pembatasan ini hanya menyangkut kecepatan hitung.
    out_deg = np.asarray(A.sum(axis=1)).ravel()
    cap = max(50.0, float(np.percentile(out_deg, 99.9)))
    keep = out_deg <= cap
    A_capped = A.multiply(keep.reshape(-1, 1)).tocsr()
    A2 = (A_capped @ A_capped).tocsr()
    print(f"        A^2 memiliki {A2.nnz:,} elemen bukan nol "
          f"(batas derajat {cap:.0f}, {int((~keep).sum()):,} simpul dibatasi)")

    def lookup(mat, rows, cols):
        """
        Ambil nilai mat[r, c] untuk sejuta pasangan sekaligus.

        Pasangan (baris, kolom) diubah menjadi satu kunci bilangan bulat
        `baris * num_nodes + kolom`, lalu dicari dengan searchsorted. Cara ini
        sepenuhnya vektorial -- iterasi Python atas 1,2 juta edge terlalu lambat.
        """
        coo = mat.tocoo()
        if coo.nnz == 0:
            return np.zeros(len(rows), dtype=np.float32)
        keys = coo.row.astype(np.int64) * num_nodes + coo.col.astype(np.int64)
        order = np.argsort(keys)
        keys_sorted, values_sorted = keys[order], coo.data[order]

        query = rows.astype(np.int64) * num_nodes + cols.astype(np.int64)
        pos = np.clip(np.searchsorted(keys_sorted, query), 0, len(keys_sorted) - 1)
        hit = keys_sorted[pos] == query
        return np.where(hit, values_sorted[pos], 0.0).astype(np.float32)

    cycle_3hop = lookup(A2, dst, src)   # v -> w -> u
    common_mid = lookup(A2, src, dst)   # u -> w -> v
    reciprocal = lookup(A, dst, src)    # v -> u

    # Jaccard tetangga: seberapa besar irisan lingkungan kedua akun
    A_und = (A + A.T).tocsr()
    A_und.data[:] = 1.0
    deg = np.asarray(A_und.sum(axis=1)).ravel()
    inter = lookup((A_und @ A_und).tocsr(), src, dst)
    union = deg[src] + deg[dst] - inter
    jaccard = np.where(union > 0, inter / np.maximum(union, 1.0), 0.0).astype(np.float32)

    print(f"        Edge dengan siklus 3 langkah : {(cycle_3hop > 0).mean()*100:.2f}%")
    print(f"        Edge dengan perantara bersama: {(common_mid > 0).mean()*100:.2f}%")
    print(f"        Edge dengan hubungan timbal balik: {(reciprocal > 0).mean()*100:.2f}%")

    return {
        "cycle_3hop": cycle_3hop,
        "common_mid": common_mid,
        "reciprocal": reciprocal,
        "jaccard": jaccard,
    }



def _temporal_cycle_features(
    src: np.ndarray, dst: np.ndarray, times_sec: np.ndarray, num_nodes: int,
    window_hours: int = 24, stride_hours: int = 12, max_hops: int = 7,
) -> Dict[str, np.ndarray]:
    """
    Deteksi penutupan siklus di dalam jendela waktu.

    Fitur motif `cycle_3hop` hanya menangkap siklus sepanjang tiga langkah,
    sedangkan pola Simple Cycle dapat melewati hingga delapan akun. Menghitung
    pangkat matriks ketetanggaan hingga A^7 pada graf penuh tidak mungkin karena
    jumlah elemen bukan nolnya meledak.

    Kuncinya: siklus pencucian uang bersifat **temporal-lokal** -- dana berputar
    dan kembali dalam hitungan jam, bukan minggu. Dengan memotong graf menjadi
    jendela 24 jam, tiap potongan hanya berisi puluhan ribu edge sehingga
    pangkat tinggi menjadi murah. Jendela dibuat bertumpang tindih (stride 12 jam)
    agar siklus yang melintasi batas jendela tetap tertangkap.

    Untuk tiap edge u -> v dihasilkan:
      - cycle_closes : 1 bila ada jalur v -> ... -> u di dalam jendela
      - cycle_length : panjang jalur balik terpendek (0 bila tidak ada)

    Tidak memakai label apa pun.
    """
    from scipy.sparse import csr_matrix

    print(f"      Mencari penutupan siklus per jendela {window_hours} jam "
          f"(hingga {max_hops} langkah) ...")
    n_edges = len(src)
    closes = np.zeros(n_edges, dtype=np.float32)
    length = np.zeros(n_edges, dtype=np.float32)

    window = window_hours * 3600
    stride = stride_hours * 3600
    t_min, t_max = int(times_sec.min()), int(times_sec.max())
    order = np.argsort(times_sec, kind="stable")
    t_sorted = times_sec[order]

    n_windows = 0
    for start in range(t_min, t_max + 1, stride):
        lo = np.searchsorted(t_sorted, start, side="left")
        hi = np.searchsorted(t_sorted, start + window, side="right")
        if hi - lo < 10:
            continue
        idx = order[lo:hi]
        n_windows += 1

        # Node dipetakan ulang ke indeks lokal agar matriks tetap kecil
        nodes = np.unique(np.concatenate([src[idx], dst[idx]]))
        remap = {int(g): i for i, g in enumerate(nodes)}
        local_src = np.fromiter((remap[int(v)] for v in src[idx]), dtype=np.int64, count=len(idx))
        local_dst = np.fromiter((remap[int(v)] for v in dst[idx]), dtype=np.int64, count=len(idx))
        n_local = len(nodes)

        A = csr_matrix((np.ones(len(idx), dtype=np.int8), (local_src, local_dst)),
                       shape=(n_local, n_local))
        A.sum_duplicates()
        A.data[:] = 1

        # Jalur balik v -> ... -> u dengan panjang 1..max_hops
        reach = A.copy()
        found = np.zeros(len(idx), dtype=bool)
        for hop in range(1, max_hops + 1):
            if reach.nnz == 0:
                break
            coo = reach.tocoo()
            keys = coo.row.astype(np.int64) * n_local + coo.col.astype(np.int64)
            keys.sort()
            query = local_dst * n_local + local_src      # apakah ada v -> ... -> u
            pos = np.clip(np.searchsorted(keys, query), 0, max(len(keys) - 1, 0))
            hit = (len(keys) > 0) & (keys[pos] == query) & (~found)
            if hit.any():
                global_idx = idx[hit]
                closes[global_idx] = 1.0
                length[global_idx] = hop + 1             # + edge u -> v itu sendiri
                found |= hit
            if found.all() or hop == max_hops:
                break
            reach = (reach @ A).tocsr()
            reach.data[:] = 1
            # Batasi pertumbuhan: jalur yang terlalu bercabang tidak informatif
            if reach.nnz > 40_000_000:
                break

    print(f"        {n_windows} jendela diperiksa | "
          f"edge yang menutup siklus: {closes.mean()*100:.2f}%")
    if closes.any():
        print(f"        Panjang siklus rata-rata: {length[closes > 0].mean():.1f} langkah")
    return {"cycle_closes": closes, "cycle_length": length}


def build_node_features(
    node_map: Dict[str, int], profiles: Dict[str, pd.DataFrame]
) -> Tuple[torch.Tensor, StandardScaler]:
    """
    Bangun 20 fitur per akun dari profil train.

    Perbedaan utama dari v1: arah kirim/terima dipisah (penting untuk
    membedakan fan-in vs fan-out), nominal di-log-scale, dan ditambahkan
    rasio nominal bulat / near-threshold serta ritme transaksi.
    """
    print(f"[5/7] Membangun fitur node ({NUM_NODE_FEATURES} dimensi) ...")
    n = len(node_map)
    f = np.zeros((n, NUM_NODE_FEATURES), dtype=np.float64)

    s = profiles["sender"]
    r = profiles["receiver"]
    s_idx = np.array([node_map.get(str(a), -1) for a in s.index])
    r_idx = np.array([node_map.get(str(a), -1) for a in r.index])
    s_ok = s_idx >= 0
    r_ok = r_idx >= 0

    sv = s.values[s_ok]
    si = s_idx[s_ok]
    (total_sent, cnt_sent, mean_sent, std_sent, max_sent, uniq_recv,
     foreign_sent, xborder_sent, round_sent, nearthr_sent, median_gap) = sv.T

    rv = r.values[r_ok]
    ri = r_idx[r_ok]
    (total_recv, cnt_recv, mean_recv, uniq_send, foreign_recv, xborder_recv) = rv.T

    f[si, 0] = np.log1p(total_sent)
    f[si, 2] = np.log1p(cnt_sent)
    f[si, 4] = np.log1p(uniq_recv)
    f[si, 8] = np.log1p(mean_sent)
    f[si, 9] = np.log1p(std_sent)
    f[si, 10] = np.log1p(max_sent)
    f[si, 12] = foreign_sent / np.maximum(cnt_sent, 1.0)
    f[si, 13] = xborder_sent / np.maximum(cnt_sent, 1.0)
    f[si, 16] = round_sent / np.maximum(cnt_sent, 1.0)
    f[si, 17] = nearthr_sent / np.maximum(cnt_sent, 1.0)
    f[si, 18] = np.log1p(np.nan_to_num(median_gap, nan=0.0))

    f[ri, 1] = np.log1p(total_recv)
    f[ri, 3] = np.log1p(cnt_recv)
    f[ri, 5] = np.log1p(uniq_send)
    f[ri, 11] = np.log1p(mean_recv)
    f[ri, 14] = foreign_recv / np.maximum(cnt_recv, 1.0)
    f[ri, 15] = xborder_recv / np.maximum(cnt_recv, 1.0)

    # Rasio arah — pembeda fan-in vs fan-out
    sent_amt = np.expm1(f[:, 0])
    recv_amt = np.expm1(f[:, 1])
    sent_cnt = np.expm1(f[:, 2])
    recv_cnt = np.expm1(f[:, 3])
    f[:, 6] = np.log1p(sent_amt / (recv_amt + 1.0))
    f[:, 7] = np.log1p(sent_cnt / (recv_cnt + 1.0))
    f[:, 19] = np.log1p(sent_cnt + recv_cnt)

    scaler = StandardScaler()
    f = scaler.fit_transform(f)
    print(f"      Matriks fitur node: ({n:,}, {NUM_NODE_FEATURES})")
    return torch.tensor(f, dtype=torch.float32), scaler


# ─── 5. Fitur edge ────────────────────────────────────────────────────────────
def build_edge_features(
    df: pd.DataFrame, profiles: Dict[str, pd.DataFrame], train_end: int,
    motif: Dict[str, np.ndarray] = None,
    scaler: StandardScaler = None,
) -> Tuple[torch.Tensor, torch.Tensor, StandardScaler]:
    """
    Bangun fitur edge: 16 numerik (di-scale) + 5 indeks kategorikal (mentah).

    Tidak ada fitur waktu absolut — hanya jam/hari siklikal dan jeda relatif.
    """
    print(f"[6/7] Membangun fitur edge ({NUM_EDGE_NUMERIC} numerik + 5 kategorikal) ...")
    amount = df["Amount"].values.astype(np.float64)
    log_amount = np.log1p(amount)

    s_prof = profiles["sender"]
    mapped_mean = df["Sender_account"].map(s_prof["mean_sent"])
    has_profile = mapped_mean.notna().values.astype(np.float32)
    sender_mean = mapped_mean.fillna(0.0).values
    sender_std = df["Sender_account"].map(s_prof["std_sent"]).fillna(0.0).values
    sender_max = df["Sender_account"].map(s_prof["max_sent"]).fillna(0.0).values

    # Selisih log, bukan rasio mentah: akun yang tidak muncul di train punya
    # profil nol, sehingga rasio mentah meledak sampai jutaan dan menjadi
    # NaN begitu masuk mixed precision. Selisih log tetap terbatas.
    dev_mean = log_amount - np.log1p(sender_mean)
    dev_max = log_amount - np.log1p(sender_max)
    dev_std = dev_mean / (np.log1p(sender_std) + 1.0)

    hour = df["Datetime"].dt.hour.values.astype(np.float64)
    dow = df["Datetime"].dt.dayofweek.values.astype(np.float64)

    numeric = np.column_stack([
        log_amount,                                             # 0  nominal
        dev_mean,                                               # 1  selisih log thd rata-rata pengirim
        dev_max,                                                # 2  selisih log thd nominal terbesar pengirim
        dev_std,                                                # 3  deviasi ternormalisasi sebaran pengirim
        has_profile,                                            # 4  pengirim punya riwayat di data train?
        df["is_round_amount"].values,                           # 5  nominal bulat
        df["is_near_threshold"].values,                         # 6  tepat di bawah ambang lapor
        np.sin(2 * np.pi * hour / 24.0),                        # 7  jam (siklikal)
        np.cos(2 * np.pi * hour / 24.0),                        # 8
        np.sin(2 * np.pi * dow / 7.0),                          # 9  hari (siklikal)
        np.cos(2 * np.pi * dow / 7.0),                          # 10
        np.log1p(df["sender_gap_min"].values),                  # 11 jeda dari tx pengirim sebelumnya
        np.log1p(df["receiver_gap_min"].values),                # 12 jeda dari tx penerima sebelumnya
        df["sender_cnt_1h"].values,                             # 13 velocity 1 jam
        np.log1p(df["sender_sum_1h"].values),                   # 14 nominal terkumpul 1 jam
        df["sender_cnt_24h"].values,                            # 15 velocity 24 jam
        df["receiver_cnt_1h"].values,                           # 16 velocity penerima 1 jam
        df["is_cross_border"].values,                           # 17 lintas negara
        # ── Konteks multi-transaksi (menyasar structuring & fan-out/in) ──
        df["sender_near_thr_7d"].values,                        # 18 transaksi near-threshold pengirim, 7 hari
        df["sender_cnt_7d"].values,                             # 19 jumlah transaksi pengirim, 7 hari
        log_amount - np.log1p(df["sender_sum_24h"].values),     # 20 porsi nominal thd volume 24 jam pengirim
        df["receiver_cnt_24h"].values,                          # 21 velocity penerima 24 jam
        np.log1p(df["pair_gap_min"].values),                    # 22 jeda sejak pasangan ini terakhir bertransaksi
        df["pair_cnt_7d"].values,                               # 23 frekuensi pasangan ini dalam 7 hari
        # ── Pass-through: inti tipologi layering ──
        np.log1p(df["fwd_sum_24h"].values) - log_amount,        # 24 rasio dana diteruskan penerima
        df["fwd_cnt_24h"].values,                               # 25 jumlah transaksi keluar penerima
        np.log1p(df["fwd_gap_min"].values),                     # 26 jeda sampai penerima meneruskan
        np.log1p(df["inflow_sum_24h"].values) - log_amount,     # 27 rasio dana masuk pengirim sebelumnya
        df["returns_24h"].values,                               # 28 aliran balik ke pengirim (siklus)
        # ── Motif subgraf: struktur dua langkah ──
        np.log1p(motif["cycle_3hop"]) if motif is not None else np.zeros(len(df)),   # 29 siklus 3 langkah
        np.log1p(motif["common_mid"]) if motif is not None else np.zeros(len(df)),   # 30 perantara bersama
        motif["reciprocal"] if motif is not None else np.zeros(len(df)),             # 31 hubungan timbal balik
        motif["jaccard"] if motif is not None else np.zeros(len(df)),                # 32 kemiripan tetangga
        # ── Kembalinya dana ke pengirim: ciri siklus panjang berapa pun ──
        np.log1p(df["return_sum_24h"].values) - log_amount,     # 33 rasio dana yang kembali
        df["return_cnt_24h"].values,                            # 34 jumlah transaksi masuk setelahnya
        # ── Penutupan siklus dalam jendela waktu ──
        motif["cycle_closes"] if motif is not None else np.zeros(len(df)),   # 35 siklus tertutup
        motif["cycle_length"] if motif is not None else np.zeros(len(df)),   # 36 panjang siklus
        # ── Mitra baru dalam jendela waktu: tanda tangan pola fan ──
        df["sender_new_partner_1h"].values,                     # 37 mitra baru pengirim, 1 jam
        df["sender_new_partner_24h"].values,                    # 38 mitra baru pengirim, 24 jam
        df["receiver_new_partner_1h"].values,                   # 39 mitra baru penerima, 1 jam
        df["receiver_new_partner_24h"].values,                  # 40 mitra baru penerima, 24 jam
        # ── Jendela MAJU: menangkap edge pertama sebuah pola ledakan ──
        df["sender_cnt_fwd_1h"].values,                         # 41 transaksi pengirim setelahnya, 1 jam
        df["sender_cnt_fwd_24h"].values,                        # 42 transaksi pengirim setelahnya, 24 jam
        np.log1p(df["sender_sum_fwd_1h"].values) - log_amount,  # 43 nominal pengirim setelahnya
        df["sender_new_fwd_1h"].values,                         # 44 mitra baru pengirim setelahnya, 1 jam
        df["sender_new_fwd_24h"].values,                        # 45 mitra baru pengirim setelahnya, 24 jam
        df["receiver_cnt_fwd_1h"].values,                       # 46 transaksi penerima setelahnya, 1 jam
        df["receiver_cnt_fwd_24h"].values,                      # 47 transaksi penerima setelahnya, 24 jam
        df["receiver_new_fwd_1h"].values,                       # 48 mitra baru penerima setelahnya, 1 jam
        df["receiver_new_fwd_24h"].values,                      # 49 mitra baru penerima setelahnya, 24 jam
        # ── Ketersediaan konteks: memberi tahu model kapan ia minim informasi ──
        # Semua dipagari di CONTEXT_CAP. Informasi yang dibutuhkan di sini
        # bersifat jenuh: beda derajat 1 dan 10 menentukan apakah GAT punya
        # tetangga untuk diagregasi, sedangkan beda 100 dan 500 tidak menambah
        # apa pun. Tanpa pagar, fitur ini menjadi jalan pintas "akun sibuk =
        # laundering" -- korelasinya dengan label 0,6152, melampaui fitur
        # terkuat yang sudah ada (0,4743), karena pola sintetis memang
        # membangun hub berderajat tinggi. Dipagari di 20 korelasinya turun ke
        # 0,4446 sementara daya pisah pada wilayah derajat < 20 tetap 0,62.
        np.log1p(np.minimum(df["sender_degree"].values, CONTEXT_CAP)),     # 50 derajat pengirim
        np.log1p(np.minimum(df["receiver_degree"].values, CONTEXT_CAP)),   # 51 derajat penerima
        np.log1p(np.minimum(df["sender_n_prior"].values, CONTEXT_CAP)),    # 52 tx pengirim sebelum ini
        np.log1p(np.minimum(df["receiver_n_prior"].values, CONTEXT_CAP)),  # 53 tx penerima sebelum ini
        (df["sender_n_prior"].values == 0).astype(np.float64),  # 54 transaksi pertama pengirim
        (df["receiver_n_prior"].values == 0).astype(np.float64),# 55 transaksi pertama penerima
        np.log1p(df["sender_age_hours"].values),                # 56 lama pengirim teramati (jam)
        np.log1p(np.minimum(np.minimum(df["sender_degree"].values,
                                       df["receiver_degree"].values),
                            CONTEXT_CAP)),                      # 57 konteks terlemah dari kedua sisi
    ]).astype(np.float32)

    # is_currency_mismatch sudah terwakili lewat pasangan embedding mata uang
    # Saat inferensi, scaler hasil pelatihan diteruskan agar transaksi baru
    # diperlakukan persis seperti saat model dilatih.
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(numeric[:train_end])
    numeric = scaler.transform(numeric).astype(np.float32)
    # Pagar pengaman terhadap outlier ekstrem agar aman di fp16
    numeric = np.clip(numeric, -10.0, 10.0).astype(np.float32)

    cat = df[[f"{c}_idx" for c in CATEGORICAL_COLS]].values.astype(np.int64)

    print(f"      Numerik  : {numeric.shape}")
    print(f"      Kategorikal: {cat.shape} (kardinalitas {cat.max(axis=0) + 1})")
    return torch.tensor(numeric), torch.tensor(cat), scaler


# ─── 6. Rakit graf ────────────────────────────────────────────────────────────
def build_graph(df: pd.DataFrame, encoders: Dict[str, LabelEncoder]) -> Tuple[Data, Dict]:
    """Rakit objek PyG Data lengkap dengan mask split dan metadata sampling."""
    print("[7/7] Merakit graf PyG ...")

    all_accounts = pd.concat([df["Sender_account"], df["Receiver_account"]]).unique()
    node_map = {str(a): i for i, a in enumerate(all_accounts)}

    src = df["Sender_account"].map(node_map).values
    dst = df["Receiver_account"].map(node_map).values
    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)

    train_mask = torch.tensor(df["_split"].values == "train", dtype=torch.bool)
    val_mask = torch.tensor(df["_split"].values == "val", dtype=torch.bool)
    test_mask = torch.tensor(df["_split"].values == "test", dtype=torch.bool)
    train_end = int(train_mask.sum())

    train_df = df[df["_split"] == "train"]
    profiles = build_account_profiles(train_df)

    node_features, node_scaler = build_node_features(node_map, profiles)
    motif = _motif_features(src, dst, len(node_map))
    t_sec = df["Datetime"].values.astype("datetime64[s]").astype(np.int64)
    motif.update(_temporal_cycle_features(src, dst, t_sec, len(node_map)))
    edge_numeric, edge_cat, edge_scaler = build_edge_features(df, profiles, train_end, motif=motif)

    y = torch.tensor(df[LABEL_COL].values, dtype=torch.float32)

    # Metadata untuk type-aware sampling (dipakai HANYA saat training, bukan fitur input)
    is_synthetic = df["Laundering_type"].isin(SYNTHETIC_LAUNDERING_TYPES).values
    is_rare_positive = ((df[LABEL_COL].values == 1) & (~is_synthetic))

    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_numeric, y=y)
    data.edge_cat = edge_cat
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.is_rare_positive = torch.tensor(is_rare_positive, dtype=torch.bool)
    # Kode tipologi per edge -- dipakai training untuk menyeimbangkan porsi tiap
    # tipologi dalam batch. Hanya metadata pelatihan, tidak pernah jadi fitur input.
    data.laundering_type = torch.tensor(df["Laundering_type_idx"].values, dtype=torch.long)
    data.num_nodes = len(node_map)

    artifacts = {
        "node_map": node_map,
        "node_scaler": node_scaler,
        "edge_scaler": edge_scaler,
        "encoders": encoders,
        "account_profiles": profiles,
        "cat_cardinalities": [int(edge_cat[:, i].max()) + 1 for i in range(edge_cat.shape[1])],
        "laundering_type_classes": list(encoders["Laundering_type"].classes_),
        "categorical_cols": CATEGORICAL_COLS,
        "num_node_features": NUM_NODE_FEATURES,
        "num_edge_numeric": NUM_EDGE_NUMERIC,
    }

    print(f"      Node      : {data.num_nodes:,}")
    print(f"      Edge      : {edge_index.shape[1]:,}")
    print(f"      x         : {tuple(node_features.shape)}")
    print(f"      edge_attr : {tuple(edge_numeric.shape)}")
    print(f"      edge_cat  : {tuple(edge_cat.shape)}")
    print(f"      Split     : train={int(train_mask.sum()):,} | val={int(val_mask.sum()):,} | test={int(test_mask.sum()):,}")
    print(f"      Positif langka (SAML-D asli): {int(data.is_rare_positive.sum()):,}")
    for name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        rare = int((data.is_rare_positive & mask).sum())
        pos = int(y[mask].sum())
        print(f"        {name:>5}: {pos:>7,} positif, {rare:>4,} di antaranya langka ({rare/max(pos,1)*100:.2f}%)")

    return data, artifacts


def main(version: str = "v2"):
    """Bangun graf PyG. `version` memilih dataset: v2 (asli) atau v3 (seluruh tipologi)."""
    global ACTIVE
    ACTIVE = PATHS[version]

    print("=" * 64)
    print(f"  AML Detection — Data Pipeline v2 (dataset {version})")
    print("=" * 64 + "\n")

    df = load_split_csvs()
    df, encoders = encode_categoricals(df)
    df = add_transaction_flags(df)
    df = add_velocity_features(df)
    df = add_context_features(df)

    # Urutkan ulang agar train mendahului val/test (mask + scaler mengandalkan urutan ini)
    split_order = {"train": 0, "val": 1, "test": 2}
    df["_order"] = df["_split"].map(split_order)
    df = df.sort_values(["_order", "Datetime"], kind="mergesort").reset_index(drop=True)

    data, artifacts = build_graph(df, encoders)

    torch.save(data, ACTIVE["graph"])
    joblib.dump(artifacts, ACTIVE["artifacts"])
    print(f"\n  Graf disimpan   : {ACTIVE['graph']}")
    print(f"  Artefak disimpan: {ACTIVE['artifacts']}")
    print(f"\nSelesai. Lanjut: python -m backend.ml.train --version {version}\n")
    return data, artifacts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bangun graf PyG dengan fitur v2.")
    parser.add_argument("--version", choices=["v2", "v3", "v4", "v5"], default="v2",
                        help="Dataset sumber: v2, v3, atau v4 (lima pola skripsi).")
    main(parser.parse_args().version)
