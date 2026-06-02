# Panduan Training & Pengujian Ulang AML Guard

Dokumen ini menjelaskan langkah demi langkah cara melakukan **training ulang** dan **pengujian ulang** model deteksi money laundering dengan dataset berskala besar.

> **Target:** 500.000 data training + 500.000 data uji (dengan validasi otomatis dari sisa data).

---

## Daftar Isi

1. [Prasyarat](#prasyarat)
2. [Setup Awal](#setup-awal)
3. [Langkah 1: Jalankan MongoDB](#langkah-1-jalankan-mongodb)
4. [Langkah 2: Generate Dataset Besar](#langkah-2-generate-dataset-besar)
5. [Langkah 3: Preprocessing Data](#langkah-3-preprocessing-data)
6. [Langkah 4: Training Model](#langkah-4-training-model)
7. [Langkah 5: Evaluasi Hasil](#langkah-5-evaluasi-hasil)
8. [Langkah 6: Jalankan Backend](#langkah-6-jalankan-backend)
9. [Retrain dari Nol](#retrain-dari-nol)
10. [Troubleshooting](#troubleshooting)

---

## Prasyarat

### Hardware (Rekomendasi)
| Komponen | Minimum | Direkomendasikan |
|----------|---------|------------------|
| CPU | 4 core | 6+ core |
| RAM | 16 GB | 32 GB |
| GPU | - | NVIDIA dengan 6+ GB VRAM |
| Storage | 5 GB free | 10 GB free SSD |

> **Catatan:** Requirements.txt menggunakan `torch+cpu`. Jika Anda memiliki GPU NVIDIA, install PyTorch CUDA untuk training jauh lebih cepat.

### Software
- Python 3.11+
- Docker Desktop
- Git

---

## Setup Awal

### 1. Clone / Siapkan Project
```bash
cd money-laundering-detection
```

### 2. Buat Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Konfigurasi Environment
```bash
cp .env.example .env
```

Edit `.env` sesuai kebutuhan:
```
MONGO_USER=admin
MONGO_PASS=admin123
MONGO_DB=money-laundering-detection
MONGO_PORT=27017
MONGO_URL=mongodb://admin:admin123@localhost:27017/money-laundering-detection?authSource=admin

BACKEND_PORT=8000
FRONTEND_PORT=80
```

---

## Langkah 1: Jalankan MongoDB

Pilih salah satu opsi:

### Opsi A: MongoDB Standalone (Rekomendasi untuk Training)
```bash
docker compose -f docker-compose.mongodb.yml up -d
```

### Opsi B: Full Stack (MongoDB + Backend + Frontend)
```bash
docker compose up -d
```

Verifikasi MongoDB berjalan:
```bash
docker ps | grep mongo
```

---

## Langkah 2: Generate Dataset Besar

Dataset asli `SAML-D.csv` hanya berisi ~152.000 transaksi. Untuk mencapai target 500K training + 500K test (total 1 juta+), kita perlu generate data sintetis.

### Jalankan Generator
```bash
python -m backend.scripts.generate_large_dataset
```

**Apa yang terjadi:**
- Membaca `SAML-D.csv` (atau backup jika ada)
- Menghasilkan ~950.000 transaksi sintetis tambahan
  - ~900.000 transaksi normal (class 0)
  - ~50.000 transaksi money laundering (class 1)
- Output: `data/SAML-D_large.csv` (~1.100.000 total rows)

**Durasi:** 2-5 menit tergantung CPU.

**Output yang diharapkan:**
```
Total rows : 1,100,000
Normal     : 1,050,000  (95.45%)
Laundering : 50,000  (4.55%)
Imbalance  : 21.0:1

-- Split preview (500K train / 500K test) --
Train laundering: ~22,700
Test  laundering: ~22,700
```

### Konfigurasi Backend untuk Menggunakan Dataset Besar

Config sudah otomatis mengarah ke `SAML-D_large.csv`. Pipeline akan otomatis membaca `train.csv` / `val.csv` / `test.csv` jika tersedia.

---

## Langkah 3: Preprocessing Data

Build graph PyTorch Geometric dari CSV:

```bash
python -m backend.ml.data_pipeline
```

**Apa yang terjadi:**
1. Load dan clean CSV
2. Encode fitur kategorikal (currency, location, payment type)
3. Hitung temporal edge weights
4. Build node features (7 dimensi per akun)
5. Build edge features (8 dimensi per transaksi)
6. Split temporal: **500K train / sisanya val / 500K test**
7. Simpan `processed_data.pt` dan `encoders.pkl`

**Output yang diharapkan:**
```
Using pre-split files (train.csv / val.csv / test.csv)
Split: train=500,000 | val=100,000 | test=500,000

-- Split statistics --
Train:  500,000 edges  (Normal: 477,300 | Laundering: 22,700)
Val  :  100,000 edges  (Normal: 95,400 | Laundering: 4,600)
Test :  500,000 edges  (Normal: 477,300 | Laundering: 22,700)
```

**Durasi:** 3-10 menit tergantung RAM dan CPU.

---

## Langkah 4: Training Model

```bash
python -m backend.ml.train
```

**Parameter Training:**
| Parameter | Nilai |
|-----------|-------|
| Epochs | 150 (early stop patience=40) |
| Learning Rate | 0.0005 |
| Weight Decay | 1e-5 |
| Focal Alpha | 0.90 |
| Focal Gamma | 2.0 |
| Hidden Dim | 64 |
| Attention Heads | 4 |
| Dropout | 0.3 |
| Oversample Ratio | 5.0 |

**Optimasi:**
- Mixed Precision (AMP) jika GPU tersedia
- Balanced sampling per epoch
- Cosine Annealing Warm Restarts scheduler
- Early stopping berdasarkan validation F1

**Durasi Estimasi:**
- CPU-only: 2-6 jam
- GPU (RTX 3060): 15-30 menit

**Output yang disimpan:**
- `data/best_model.pt` — model terbaik
- `data/training_metrics.json` — metrik lengkap
- Data otomatis dipopulasikan ke MongoDB

---

## Langkah 5: Evaluasi Hasil

Setelah training selesai, cek metrik:

```bash
cat data/training_metrics.json | python -m json.tool
```

**Metrik yang perhatikan:**
- **Precision**: Seberapa akurat prediksi positive (laundering)
- **Recall**: Seberapa banyak laundering yang berhasil tertangkap
- **F1 Score**: Rata-rata harmonis precision dan recall
- **Optimal Threshold**: Threshold probabilitas terbaik (biasanya ~0.90)

**Contoh hasil yang baik:**
```json
{
  "test_metrics": {
    "precision": 0.95,
    "recall": 0.88,
    "f1": 0.91
  },
  "optimal_threshold": 0.90
}
```

### Confusion Matrix
```
                Predicted 0    Predicted 1
Actual 0 (Normal)     479          1
Actual 1 (Launder)      2         18
```

---

## Langkah 6: Jalankan Backend

Setelah training dan MongoDB terisi data:

```bash
python -m backend.main
```

Akses API di `http://localhost:8000/docs` (Swagger UI).

Test health:
```bash
curl http://localhost:8000/api/health
```

Test prediksi:
```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Sender_account": "800001000",
    "Receiver_account": "800002000",
    "Amount": 50000,
    "Payment_currency": "US Dollar",
    "Received_currency": "Euro",
    "Sender_bank_location": "US",
    "Receiver_bank_location": "Germany",
    "Payment_type": "Wire",
    "Date": "2024-01-15",
    "Time": "10:30:00"
  }'
```

---

## Retrain dari Nol

Jika Anda ingin melakukan training ulang dari awal (misal setelah mengubah hyperparameter atau dataset):

### 1. Hapus Artifacts Lama
```bash
rm data/processed_data.pt
rm data/best_model.pt
rm data/encoders.pkl
rm data/training_metrics.json
```

### 2. Reset MongoDB (Opsional)
```bash
docker compose -f docker-compose.mongodb.yml down -v
docker compose -f docker-compose.mongodb.yml up -d
```

### 3. Ulangi Langkah 2-4
```bash
# Generate data (jika belum)
python -m backend.scripts.generate_large_dataset

# Preprocess
python -m backend.ml.data_pipeline

# Train
python -m backend.ml.train
```

---

## Troubleshooting

### Training terlalu lambat
- **Solusi:** Install PyTorch dengan CUDA support:
  ```bash
  pip uninstall torch torchvision torchaudio torch-geometric
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  pip install torch-geometric
  ```

### Out of Memory (OOM)
- **Solusi:** Turunkan `batch_size` atau `edge_chunk_size` di `backend/ml/model.py`
- **Solusi:** Pastikan tidak ada aplikasi berat yang berjalan

### MongoDB connection failed
- Pastikan MongoDB container berjalan: `docker ps | grep mongo`
- Cek `.env` MONGO_URL mengarah ke `localhost:27017`

### Validation F1 sangat rendah (< 0.05)
- Ini normal di epoch awal karena class imbalance
- Model memerlukan 10-20 epoch untuk mulai belajar pola laundering
- Jika setelah 40 epoch masih rendah, coba:
  - Naikkan `oversample_ratio` di `train.py`
  - Turunkan `FOCAL_ALPHA` (misal 0.75)
  - Periksa kualitas synthetic data

### Test set terlalu besar (500K) menyebabkan evaluasi lambat
- Evaluasi test set 500K memang lebih lambat dibanding ukuran kecil
- Solusi: di `backend/ml/train.py`, evaluasi test bisa di-sampling atau di-chunk
- Namun untuk hasil yang akurat, evaluasi full 500K direkomendasikan

### File train/val/test tidak terdeteksi
- Pastikan `python -m backend.scripts.generate_large_dataset` sudah dijalankan
- Cek folder `data/` harus ada: `train.csv`, `val.csv`, `test.csv`
- Pipeline otomatis fallback ke `SAML-D_large.csv` jika file split tidak ada

### processed_data.pt tidak ditemukan
- Pastikan `data_pipeline.py` berhasil dijalankan
- Cek `backend/config.py` path sudah benar

---

## Ringkasan Perintah Cepat

```bash
# Setup
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env

# MongoDB
docker compose -f docker-compose.mongodb.yml up -d

# Generate + Preprocess + Train
python -m backend.scripts.generate_large_dataset
python -m backend.ml.data_pipeline
python -m backend.ml.train

# Jalankan backend
python -m backend.main
```

---

*Selamat training! Jika ada kendala, cek log dan pastikan setiap langkah menghasilkan output yang sesuai.*
