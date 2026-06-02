# Ketidakseimbangan Kelas (Class Imbalance) pada AML Detection

Dokumen ini menjelaskan teori, metode penanganan, perhitungan manual, serta metode pembanding yang relevan dengan masalah **class imbalance** pada project deteksi pencucian uang (AML) ini.

---

## 1. Teori: Apa itu Class Imbalance?

**Class imbalance** adalah kondisi di mana jumlah sampel antar kelas dalam dataset bersifat sangat tidak seimbang. Dalam konteks AML, transaksi normal jauh lebih banyak dibanding transaksi mencurigakan (laundering).

### Mengapa ini menjadi masalah?

| Masalah | Penjelasan |
|---|---|
| **Bias ke kelas mayoritas** | Model cenderung memprediksi semua sampel sebagai kelas mayoritas (normal) karena "aman" secara akurasi |
| **Metrik menipu** | Akurasi bisa tinggi (~86%) meski model sama sekali tidak menangkap laundering |
| **F1 rendah** | Precision dan Recall pada kelas minoritas (laundering) menjadi sangat rendah |
| **Boundary tidak optimal** | Decision boundary condong ke kelas mayoritas, sehingga sampel minoritas sulit terdeteksi |

### Situasi pada Dataset Ini

```
Total edges     : 1,222,901
Normal  (y = 0) : 1,049,078  → 85.79%
Fraud   (y = 1) :   173,823  → 14.21%
Imbalance Ratio : 6.04 : 1
```

Artinya, **setiap 1 transaksi laundering ada ~6 transaksi normal**.

---

## 2. Metode yang Digunakan dalam Project

Project ini menerapkan **tiga lapisan strategi** untuk menangani imbalance:

### A. Focal Loss

**Focal Loss** (Lin et al., 2017) adalah modifikasi dari Binary Cross Entropy (BCE) yang memberikan bobot lebih tinggi pada sampel yang sulit diklasifikasikan (hard examples), terutama dari kelas minoritas.

**Rumus:**

```
FL(pₜ) = -αₜ · (1 - pₜ)^γ · log(pₜ)
```

Dimana:
- `pₜ` = probabilitas prediksi untuk kelas yang benar
- `α` = balancing factor (bobot kelas)
- `γ` = focusing parameter (semakin tinggi, semakin fokus pada hard examples)

**Hyperparameter pada project:**
- `α = 0.75` (memberi bobot lebih pada kelas positif / laundering)
- `γ = 2.0` (menurunkan kontribusi easy examples secara kuadratik)

### B. Balanced Edge Sampling (Oversampling)

Selama training, batch disusun dengan **oversampling ratio = 5.0**, yang berarti:
- Setiap batch mencoba mengandung proporsi sampel minority yang lebih tinggi dari distribusi asli
- Ini membantu model "melihat" cukup banyak contoh laundering selama setiap epoch

### C. Threshold Optimization

Model dilatih dengan output probability, lalu threshold dituning setelah training:
- Default threshold BCE = **0.50**
- Threshold optimal project ini = **0.90** (berdasarkan validasi F1)
- Threshold tinggi memaksimalkan **Precision** pada kelas laundering, sehingga false positive rendah

---

## 3. Perhitungan Manual

### 3.1 Binary Cross Entropy (BCE) Biasa

Untuk satu sampel:

```
BCE = -[y · log(p) + (1 - y) · log(1 - p)]
```

**Contoh:**
Sampel laundering (y = 1) dengan prediksi model p = 0.95:
```
BCE = -[1 · log(0.95) + 0] = -(-0.0513) = 0.0513
```

Sampel laundering (y = 1) dengan prediksi model p = 0.20 (salah prediksi):
```
BCE = -[1 · log(0.20)] = -(-1.6094) = 1.6094
```

**Masalah:** Kedua sampel laundering sama-sama dianggap penting, padahal yang p = 0.95 sudah "mudah" dan tidak perlu dilatih lebih keras.

---

### 3.2 Focal Loss — Perhitungan Manual

Menggunakan rumus FL dengan α = 0.75, γ = 2.0.

**a) Sampel mudah — laundering dengan p = 0.95**

```
pₜ     = p = 0.95
(1-pₜ) = 0.05
(1-pₜ)^γ = 0.05² = 0.0025
αₜ     = 0.75 (karena y = 1)

FL = -0.75 · 0.0025 · log(0.95)
   = -0.75 · 0.0025 · (-0.0513)
   = 0.000096
```

Loss turun drastis dari **0.0513 → 0.000096** (hampir diabaikan).

**b) Sampel sulit — laundering dengan p = 0.20**

```
pₜ     = p = 0.20
(1-pₜ) = 0.80
(1-pₜ)^γ = 0.80² = 0.64
αₜ     = 0.75

FL = -0.75 · 0.64 · log(0.20)
   = -0.75 · 0.64 · (-1.6094)
   = 0.7725
```

Loss hanya turun sedikit dari **1.6094 → 0.7725**, sehingga model tetap fokus memperbaiki sampel ini.

**c) Sampel normal (y = 0) dengan p = 0.10**

```
pₜ     = 1 - p = 0.90
(1-pₜ) = 0.10
(1-pₜ)^γ = 0.10² = 0.01
αₜ     = 1 - 0.75 = 0.25 (karena y = 0)

FL = -0.25 · 0.01 · log(0.90)
   = -0.25 · 0.01 · (-0.1054)
   = 0.000263
```

Kelas mayoritas (normal) yang sudah benar diprediksi mendapat loss yang sangat kecil, sehingga tidak mendominasi gradient update.

---

### 3.3 Precision, Recall, dan F1-Score

Dari hasil evaluasi pada test set:

```
True Negative  (TN) = 429,447
False Positive (FP) =     187
False Negative (FN) =     296
True Positive  (TP) =  70,070
```

**Precision (Laundering):**
```
Precision = TP / (TP + FP)
          = 70,070 / (70,070 + 187)
          = 70,070 / 70,257
          = 0.9973  → 99.73%
```

**Recall (Laundering):**
```
Recall = TP / (TP + FN)
       = 70,070 / (70,070 + 296)
       = 70,070 / 70,366
       = 0.9958  → 99.58%
```

**F1-Score:**
```
F1 = 2 · (Precision · Recall) / (Precision + Recall)
   = 2 · (0.9973 · 0.9958) / (0.9973 + 0.9958)
   = 2 · 0.9931 / 1.9931
   = 0.9966  → 99.66%
```

---

## 4. Metode Pembanding

Berikut adalah teknik-teknik umum lain untuk menangani class imbalance yang **tidak** digunakan pada project ini, tetapi sering dipraktikkan:

### A. SMOTE (Synthetic Minority Over-sampling Technique)

| Aspek | Penjelasan |
|---|---|
| **Cara kerja** | Membuat sampel sintetis baru untuk kelas minoritas dengan interpolasi antara sampel minoritas yang ada dan k-nearest neighbors-nya |
| **Kelebihan** | Menambah variasi data tanpa duplikasi exact; mengurangi overfitting |
| **Kekurangan** | Risk generating noisy or unrealistic samples; tidak langsung compatible dengan graph/tabular data tanpa feature engineering |
| **Cocok untuk** | Tabular data klasik (credit scoring, customer churn) |

### B. Random Oversampling

| Aspek | Penjelasan |
|---|---|
| **Cara kerja** | Menduplikasi sampel dari kelas minoritas secara acak hingga jumlahnya seimbang |
| **Kelebihan** | Sangat sederhana dan cepat |
| **Kekurangan** | Mudah overfitting karena model melihat sampel yang sama berulang kali |
| **Cocok untuk** | Dataset kecil dan sederhana |

### C. Random Undersampling

| Aspek | Penjelasan |
|---|---|
| **Cara kerja** | Membuang sampel dari kelas mayoritas secara acak hingga seimbang |
| **Kelebihan** | Mengurangi waktu training |
| **Kekurangan** **Potensi kehilangan informasi penting** dari kelas mayoritas; bisa menurunkan akurasi overall |
| **Cocok untuk** | Dataset sangat besar di mana mayoritas redundan |

### D. Class Weighting (di Loss Function)

| Aspek | Penjelasan |
|---|---|
| **Cara kerja** | Memberikan bobot lebih besar pada loss kelas minoritas (misal `weight_pos = N_neg / N_pos`) |
| **Kelebihan** | Sederhana, tidak perlu mengubah data |
| **Kekurangan** | Tidak membedakan antara easy dan hard examples |
| **Perbandingan** | Focal Loss adalah evolusi dari class weighting — menambahkan mekanisme `(1-p)^γ` untuk membedakan easy vs hard |

### E. Cost-Sensitive Learning

| Aspek | Penjelasan |
|---|---|
| **Cara kerja** | Menetapkan cost matrix di mana misclassify minority class lebih mahal daripada majority |
| **Kelebihan** | Fleksibel sesuai dengan domain bisnis |
| **Kekurangan** | Sulit menentukan cost ratio yang optimal; seringkali diperlukan domain expert |
| **Cocok untuk** | Fraud detection, medical diagnosis di mana false negative sangat berbahaya |

### F. Ensemble Methods (BalancedBagging, EasyEnsemble)

| Aspek | Penjelasan |
|---|---|
| **Cara kerja** | Menggabungkan beberapa classifier yang masing-masing dilatih pada subset data yang lebih seimbang |
| **Kelebihan** | Stabil dan robust; mengurangi variance |
| **Kekurangan** | Lebih kompleks dan lambat; interpretabilitas menurun |
| **Cocok untuk** | Production systems yang memerlukan robustness tinggi |

---

## 5. Perbandingan Ringkas: Metode pada Project vs Pembanding

| Kriteria | **Focal Loss** (Project) | SMOTE | Random Oversampling | Class Weighting |
|---|---|---|---|---|
| **Mengubah data?** | ❌ Tidak | ✅ Ya | ✅ Ya | ❌ Tidak |
| **Fokus hard examples?** | ✅ Ya | ❌ Tidak | ❌ Tidak | ❌ Tidak |
| **Resiko overfitting?** | ⬜ Rendah | ⬜ Sedang | 🔴 Tinggi | ⬜ Rendah |
| **Kompleksitas** | 🟢 Rendah | 🟡 Sedang | 🟢 Rendah | 🟢 Rendah |
| **Training speed** | 🟢 Cepat | 🟡 Sedang | 🟢 Cepat | 🟢 Cepat |
| **Graph-compatible** | ✅ Ya | ⚠️ Sulit | ⚠️ Sulit | ✅ Ya |

> **Catatan:** Pada **Graph Neural Network** (GNN), teknik resampling seperti SMOTE atau Random Oversampling menjadi lebih sulit karena struktur graph (edge_index, node features) harus tetap konsisten. Oleh karena itu, **loss-based approach** seperti Focal Loss menjadi pilihan yang lebih natural dan efektif.

---

## 6. Kesimpulan

Project AML ini berhasil menangani class imbalance **6.04 : 1** dengan kombinasi:

1. **Focal Loss** (`α=0.75, γ=2.0`) — menekan kontribusi easy examples dan memfokuskan training pada hard examples
2. **Balanced Edge Sampling** (`oversample_ratio=5.0`) — memastikan batch training cukup "melihat" transaksi laundering
3. **Threshold Optimization** (`threshold=0.90`) — memaksimalkan precision pada kelas minoritas

Hasil akhir: **Precision = 99.73%**, **Recall = 99.58%**, **F1 = 99.66%** — menunjukkan bahwa meskipun data tidak seimbang, model tetap mampu mendeteksi laundering dengan sangat akurat dan minim false alarm.
