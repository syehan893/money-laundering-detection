"""
config.py — Centralized Configuration
=======================================
All paths, constants, and environment variables in one place.
"""

import os

# ─── Project Roots ────────────────────────────────────────────────────────────
# BASE_DIR = money-laundering-detection/ (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
DATA_DIR = os.path.join(BASE_DIR, "data")


# Isi data/ dipecah menurut peran agar mudah ditelusuri. Subfolder ini dicari
# otomatis oleh _resolve(), sehingga seluruh konstanta path di bawah tetap
# ditulis sebagai nama berkas polos dan tidak perlu diubah saat berkas berpindah.
DATA_SUBDIRS = ("raw", "processed", "models", "results")


def _kategori(filename: str) -> str:
    """Subfolder data/ tempat sebuah berkas baru seharusnya dibuat."""
    low = filename.lower()
    if low.endswith(".csv"):
        return "raw"                                    # sumber & split
    if low.startswith(("processed_data", "encoders")):
        return "processed"                              # graf & encoder hasil pipeline
    if low.startswith(("best_model", "best_sage")):
        return "models"                                 # bobot model terlatih
    return "results"                                    # metrik, probabilitas, laporan


def _resolve(filename: str) -> str:
    """
    Cari sebuah berkas data, atau tentukan di mana ia akan dibuat.

    Urutan pencarian: subfolder data/ menurut kategori, lalu data/ itu sendiri
    (berkas lama yang belum dipindah), lalu root proyek. Berkas yang belum ada
    diarahkan ke subfolder sesuai kategorinya.
    """
    kandidat = [os.path.join(DATA_DIR, sub, filename) for sub in DATA_SUBDIRS]
    kandidat += [os.path.join(DATA_DIR, filename), os.path.join(BASE_DIR, filename)]
    for path in kandidat:
        if os.path.exists(path):
            return path
    return os.path.join(DATA_DIR, _kategori(filename), filename)


# Alias publik. Modul lain memakai ini alih-alih menyusun path sendiri dengan
# os.path.join(DATA_DIR, ...), supaya berkas tetap ditemukan ke mana pun ia
# dipindah di dalam data/.
resolve = _resolve


# ─── Data Paths ───────────────────────────────────────────────────────────────
# Main CSV (used by populate_mongodb and fallback single-file pipeline)
CSV_PATH = _resolve("SAML-D_large.csv")
# Pre-split CSVs (used by data_pipeline when available)
TRAIN_CSV_PATH = _resolve("train.csv")
VAL_CSV_PATH = _resolve("val.csv")
TEST_CSV_PATH = _resolve("test.csv")
DATA_PATH = _resolve("processed_data.pt")
MODEL_PATH = _resolve("best_model.pt")
ENCODERS_PATH = _resolve("encoders.pkl")
METRICS_PATH = _resolve("training_metrics.json")

# ─── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42

# ─── Training Hyperparameters ─────────────────────────────────────────────────
EPOCHS = 150
LR = 0.0005
WEIGHT_DECAY = 1e-5
PATIENCE = 40
FOCAL_ALPHA = 0.92   # Increased from 0.90 to focus more on positive (laundering) class
FOCAL_GAMMA = 2.0

# ─── Cost-Sensitive Learning ──────────────────────────────────────────────────
# Higher FN cost strongly penalizes missed laundering (false negatives).
# FP cost penalizes false alarms (false positives).
FN_COST = 5.0
FP_COST = 1.0

# ─── Hard Example Mining ──────────────────────────────────────────────────────
HARD_MINING_ENABLED = True
HARD_MINING_START_EPOCH = 5      # Start mining after warmup
HARD_MINING_RATIO = 0.3          # Fraction of batch filled with hard examples

# ─── Label Smoothing ──────────────────────────────────────────────────────────
LABEL_SMOOTHING = 0.02

# ─── Threshold Optimization ───────────────────────────────────────────────────
# Target per-class error rates for constraint-based threshold selection.
MAX_FPR = 0.001   # False Positive Rate < 0.1%
MAX_FNR = 0.001   # False Negative Rate < 0.1%
THRESHOLD_GRID_STEP = 0.01

# ─── Warm Start / Resume ──────────────────────────────────────────────────────
# If True and a checkpoint exists at MODEL_PATH, load its weights before training.
# Useful for fine-tuning an already-good model toward stricter FPR/FNR targets.
WARM_START_FROM_CHECKPOINT = True

# ─── Data Split Sizes ─────────────────────────────────────────────────────────
# Override default 70/15/15 temporal split with absolute counts.
# Set to None to use default percentage-based split.
TRAIN_SIZE = 500_000   # 500K training edges
TEST_SIZE = 500_000    # 500K test edges
# Validation size is auto-calculated: total - train - test

# ─── Model Architecture ──────────────────────────────────────────────────────
NODE_FEAT_DIM = 7
EDGE_FEAT_DIM = 8
HIDDEN_DIM = 64
NUM_HEADS = 4          # Number of attention heads in first GAT layer
NUM_LAYERS = 2         # Number of GAT encoder layers (2 = original architecture)
DROPOUT_TRAIN = 0.3
DROPOUT_INFERENCE = 0.0
USE_LAYER_NORM = False  # LayerNorm between GAT layers (False = compatible with old checkpoint)
USE_RESIDUAL = False    # Residual connections between GAT layers


# ══════════════════════════════════════════════════════════════════════════════
#  VERSI 2 — Feature engineering stasioner + type-aware sampling
#  (modul v1 di atas sengaja tidak diubah agar hasil skripsi tetap reproducible)
# ══════════════════════════════════════════════════════════════════════════════

# ─── Path artefak v2 ──────────────────────────────────────────────────────────
DATA_V2_PATH = _resolve("processed_data_v2.pt")
MODEL_V2_PATH = _resolve("best_model_v2.pt")
ENCODERS_V2_PATH = _resolve("encoders_v2.pkl")
METRICS_V2_PATH = _resolve("training_metrics_v2.json")

# ─── Tipologi laundering sintetis ─────────────────────────────────────────────
# Pola yang disuntikkan oleh generate_large_dataset.py / augment_data.py.
# Semua tipe laundering DI LUAR daftar ini berasal dari SAML-D asli dan
# jumlahnya sangat sedikit (433 di train) -- kelompok inilah yang menyumbang
# hampir seluruh false negative pada model v1.
# Dipakai HANYA untuk bobot sampling saat training, tidak pernah jadi fitur input.
SYNTHETIC_LAUNDERING_TYPES = {
    "Smurfing",
    "Fan_Out_Fan_In",
    "Rapid_Burst",
    "Cross_Border_Layering",
    "Round_Tripping",
}

# ─── Arsitektur v2 ────────────────────────────────────────────────────────────
NODE_FEAT_DIM_V2 = 20      # 20 fitur akun (arah kirim/terima dipisah)
EDGE_NUMERIC_DIM_V2 = 50   # fitur stasioner + konteks multi-transaksi + pass-through
CURRENCY_EMB_DIM = 12      # dimensi embedding mata uang
LOCATION_EMB_DIM = 12      # dimensi embedding lokasi bank
PAYMENT_TYPE_EMB_DIM = 8   # dimensi embedding tipe pembayaran
HIDDEN_DIM_V2 = 64   # lebar GAT sama dengan v1 (terbukti muat di VRAM 8 GB)
NUM_HEADS_V2 = 4
NUM_LAYERS_V2 = 3          # GAT 3 layer (v1 memakai 2)
DROPOUT_V2 = 0.25
USE_LAYER_NORM_V2 = True   # LayerNorm lebih stabil untuk graf besar
USE_RESIDUAL_V2 = True

# ─── Training v2 ──────────────────────────────────────────────────────────────
EPOCHS_V2 = 300   # v4 belum konvergen pada 120 epoch
LR_V2 = 0.001
WEIGHT_DECAY_V2 = 1e-5
PATIENCE_V2 = 50
FOCAL_ALPHA_V2 = 0.90
FOCAL_GAMMA_V2 = 2.0
FN_COST_V2 = 12.0          # dinaikkan dari 5.0 -- false negative jauh lebih mahal
FP_COST_V2 = 1.0
LABEL_SMOOTHING_V2 = 0.01
OVERSAMPLE_RATIO_V2 = 4.0

# ─── Type-aware sampling (inti perbaikan v2) ──────────────────────────────────
# Porsi slot positif dalam tiap batch yang dijatahkan untuk laundering langka
# (tipologi SAML-D asli). Tanpa ini, 433 contoh langka tenggelam di antara
# 68.603 contoh sintetis dan praktis tidak pernah ikut mempengaruhi gradien.
RARE_POSITIVE_BATCH_RATIO = 0.35
RARE_POSITIVE_LOSS_WEIGHT = 3.0   # bobot loss tambahan untuk positif langka

# ─── Hard mining v2 ───────────────────────────────────────────────────────────
HARD_MINING_START_EPOCH_V2 = 3
HARD_MINING_RATIO_V2 = 0.35

# ─── Ukuran batch v2 ──────────────────────────────────────────────────────────
# Encoding node dilakukan atas graf penuh, tetapi gradien hanya disimpan untuk
# edge yang masuk batch -- inilah yang membuat model v2 (lebih besar) tetap
# muat di VRAM 8 GB. Beberapa step per epoch memberi lebih banyak update
# gradien dibanding v1 yang hanya 1 step per epoch.
POS_SLOTS_PER_STEP_V2 = 16_000
STEPS_PER_EPOCH_V2 = 4

# ─── Loss berbasis anggaran FPR (v2, iterasi 3) ───────────────────────────────
# Focal loss saja membuat positif yang sudah mencapai p~0,93 nyaris tidak
# menghasilkan gradien, sehingga mereka mandek tepat di bawah threshold.
# `budget_margin_loss` membandingkan skor positif terhadap skor negatif pada
# batas anggaran FPR, lalu mendorong selisihnya minimal sebesar MARGIN_VALUE_V2.
MARGIN_LOSS_WEIGHT_V2 = 1.0
MARGIN_VALUE_V2 = 2.0

# Hard mining top-K: yang menentukan letak threshold hanya beberapa ratus
# negatif berskor tertinggi, bukan 35% teratas seperti pada iterasi sebelumnya.
HARD_NEG_TOPK_V2 = 20_000
HARD_POS_TOPK_V2 = 20_000

# ══════════════════════════════════════════════════════════════════════════════
#  VERSI 3 — Dataset dengan seluruh tipologi terwakili
#  Enam belas tipologi yang sebelumnya hanya punya 5-216 contoh diaugmentasi
#  ke >=4.000 contoh (lihat backend/scripts/generate_missing_patterns.py).
# ══════════════════════════════════════════════════════════════════════════════
CSV_V3_PATH = _resolve("SAML-D_v3.csv")
TRAIN_CSV_V3_PATH = _resolve("train_v3.csv")
VAL_CSV_V3_PATH = _resolve("val_v3.csv")
TEST_CSV_V3_PATH = _resolve("test_v3.csv")
DATA_V3_PATH = _resolve("processed_data_v3.pt")
MODEL_V3_PATH = _resolve("best_model_v3.pt")
ENCODERS_V3_PATH = _resolve("encoders_v3.pkl")
METRICS_V3_PATH = _resolve("training_metrics_v3.json")

# Sampling seimbang per tipologi. Diuji pada dataset v3 dan hasilnya sedikit
# LEBIH BURUK (recall tipologi baru 59,1% vs 66,0%), karena porsi yang sama rata
# justru mengurangi perhatian pada tipologi yang sudah terdeteksi baik.
# Dibiarkan tersedia sebagai opsi, tidak aktif secara default.
TYPE_BALANCED_SAMPLING = False

# ══════════════════════════════════════════════════════════════════════════════
#  VERSI 4 — Lima pola sesuai definisi skripsi, atribut realistis
#  Dibangun dari SAML-D_original_backup.csv (dataset Kaggle asli). Atribut pola
#  laundering diambil dari distribusi transaksi normal, sehingga yang membedakan
#  hanya struktur dan perilaku -- bukan penanda atribut.
# ══════════════════════════════════════════════════════════════════════════════
CSV_V4_PATH = _resolve("SAML-D_v4.csv")
TRAIN_CSV_V4_PATH = _resolve("train_v4.csv")
VAL_CSV_V4_PATH = _resolve("val_v4.csv")
TEST_CSV_V4_PATH = _resolve("test_v4.csv")
DATA_V4_PATH = _resolve("processed_data_v4.pt")
MODEL_V4_PATH = _resolve("best_model_v4.pt")
ENCODERS_V4_PATH = _resolve("encoders_v4.pkl")
METRICS_V4_PATH = _resolve("training_metrics_v4.json")

# Lima pola yang menjadi lingkup penelitian
THESIS_PATTERNS = ["Fan_Out", "Fan_In", "Scatter_Gather", "Gather_Scatter", "Simple_Cycle"]

# ─── Teknik peningkatan model v4 ──────────────────────────────────────────────
USE_MOTIF_FEATURES = True    # fitur siklus & perantara bersama (menyasar Simple_Cycle)
USE_EDGE_ATTENTION = True    # alirkan fitur edge ke perhitungan atensi GAT
USE_MULTITASK = True         # kepala tambahan untuk memprediksi jenis pola
MULTITASK_WEIGHT = 0.3       # bobot loss tugas tambahan


# ─────────────────────────────────────────────────────────────────────────────
# Model v5 — menyasar kegagalan pada akun berkonteks-graf rendah
# ─────────────────────────────────────────────────────────────────────────────
# Diagnosis v4: 63% false positive dan 65% false negative terjadi pada transaksi
# berderajat < 20, padahal kelompok itu hanya 25,7% dari populasi. Pada akun yang
# nyaris tak terhubung GAT tidak punya tetangga untuk diagregasi, sehingga model
# bersandar pada prior kategorikal -- terlihat jelas pada Payment_type
# "Cash Deposit" yang menyumbang 323 dari 480 FP dengan FPR 3,24% (28x rata-rata).
#
# Tiga penangkalnya:
#   1. Delapan fitur konteks eksplisit, agar model tahu kapan ia minim informasi.
#   2. Dropout embedding kategorikal, agar prior kategorikal tidak mendominasi
#      saat graf tidak informatif.
#   3. Ensemble beberapa seed, karena seluruh kesalahan v4 berada di pita skor
#      ambigu 0,33-0,70 -- perataan antar-seed menajamkan pita itu.
CSV_V5_PATH = CSV_V4_PATH
TRAIN_CSV_V5_PATH = TRAIN_CSV_V4_PATH
VAL_CSV_V5_PATH = VAL_CSV_V4_PATH
TEST_CSV_V5_PATH = TEST_CSV_V4_PATH
DATA_V5_PATH = _resolve("processed_data_v5.pt")
MODEL_V5_PATH = _resolve("best_model_v5.pt")
ENCODERS_V5_PATH = _resolve("encoders_v5.pkl")
METRICS_V5_PATH = _resolve("training_metrics_v5.json")

EDGE_NUMERIC_DIM_V5 = 58   # 50 fitur v4 + 8 fitur ketersediaan konteks

# Dropout embedding kategorikal: seluruh vektor embedding sebuah kolom
# dinolkan dengan peluang ini saat training. Model dipaksa membaca struktur
# graf dan nominal, bukan menghafal "Cash Deposit cenderung mencurigakan".
CAT_EMBED_DROPOUT_V5 = 0.15

# Ensemble: seed berbeda, probabilitas dirata-ratakan.
ENSEMBLE_SEEDS_V5 = [42, 1337, 2024]

# Bobot ketiga anggota ensemble v5 (urutan sama dengan ENSEMBLE_SEEDS_V5) dan
# hasil evaluasi ensemble. Dipakai jalur inferensi/dashboard agar angka yang
# ditampilkan sama dengan yang dilaporkan di naskah (threshold 0,7278).
MODEL_V5_ENSEMBLE_PATHS = [
    MODEL_V5_PATH,
    _resolve("best_model_v5_seed1337.pt"),
    _resolve("best_model_v5_seed2024.pt"),
]
EVAL_V5_PATH = _resolve("evaluation_v5.json")
PER_POLA_V5_PATH = _resolve("per_pola_v5.json")
