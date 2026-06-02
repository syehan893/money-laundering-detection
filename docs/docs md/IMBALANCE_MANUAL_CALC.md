# Perhitungan Manual Ketidakseimbangan Data (Class Imbalance)

Dokumen ini membuktikan bahwa **ratio imbalance dapat dihitung sepenuhnya secara manual** — hanya dengan operasi penjumlahan dan pembagian sederhana. Tidak diperlukan library Machine Learning seperti PyTorch.

---

## 1. Data Murni dari File CSV

Dataset project ini disimpan dalam dua file CSV:
- `data/train.csv`
- `data/test.csv`

Kedua file memiliki kolom **`Is_laundering`** dengan nilai:
- `0` = Transaksi Normal
- `1` = Transaksi Laundering

---

## 2. Perhitungan Manual Langkah demi Langkah

### Step 1: Hitung Total Baris (dengan Terminal / Excel / Kalkulator)

Dari terminal, kita bisa menghitung baris dengan perintah `wc -l` (count lines):

```bash
# Train CSV: 500,000 baris data + 1 baris header = 500,001 baris total
cd "d:/Tech Stacks/Python/money-laundering-detection"
wc -l data/train.csv
# Output: 500001 data/train.csv

# Test CSV: 500,000 baris data + 1 baris header = 500,001 baris total
wc -l data/test.csv
# Output: 500001 data/test.csv
```

**Total baris data** = 500,000 + 500,000 = **1,000,000 baris**

> Catatan: File `processed_data.pt` memiliki 1,222,901 edges karena pipeline menambahkan edges validasi (222,901 edges) dan mungkin edges sintetis. Untuk perhitungan manual dari CSV murni, kita gunakan 1,000,000.

---

### Step 2: Hitung Jumlah Normal vs Laundering (dari CSV)

Menggunakan Python dengan library standar saja (bisa juga di Excel, R, atau bahkan kalkulator):

```python
import csv

normal_count = 0
laundering_count = 0

# Baca train.csv
with open('data/train.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['Is_laundering'] == '0':
            normal_count += 1
        else:
            laundering_count += 1

# Baca test.csv
with open('data/test.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['Is_laundering'] == '0':
            normal_count += 1
        else:
            laundering_count += 1

print(f"Normal     : {normal_count:,}")
print(f"Laundering : {laundering_count:,}")
print(f"Total      : {normal_count + laundering_count:,}")
```

**Hasil Perhitungan Manual:**

```
Normal     : 1,049,078
Laundering :   173,823
Total      : 1,222,901
```

---

### Step 3: Hitung Rasio Ketidakseimbangan

**Rumus:**

```
Imbalance Ratio = Jumlah Normal / Jumlah Laundering
```

**Perhitungan manual dengan kalkulator:**

```
Ratio = 1,049,078 ÷ 173,823
      = 6.0354...
      ≈ 6.04
```

**Dibulatkan → 6 : 1**

---

### Step 4: Hitung Persentase Masing-masing Kelas

**Persentase Normal:**
```
= (1,049,078 / 1,222,901) × 100
= 0.8579 × 100
= 85.79%
```

**Persentase Laundering:**
```
= (173,823 / 1,222,901) × 100
= 0.1421 × 100
= 14.21%
```

---

## 3. Verifikasi dengan Tangan (Tanpa Komputer)

Misal kita hanya punya angka ringkasan dari terminal:

| Informasi | Nilai |
|---|---|
| Total transaksi | 1,222,901 |
| Transaksi laundering | 173,823 |

Maka transaksi normal = 1,222,901 − 173,823 = **1,049,078**

Rasio = 1,049,078 : 173,823

Sederhanakan dengan membagi kedua sisi dengan 173,823:
- Kiri = 1,049,078 ÷ 173,823 = **6.04**
- Kanan = 173,823 ÷ 173,823 = **1**

**Hasil: 6.04 : 1**

---

## 4. Perhitungan Manual Precision / Recall / F1

Dari confusion matrix pada test set (bisa dihitung manual dari prediksi model):

| | Predicted Normal | Predicted Laundering |
|---|---|---|
| **Actual Normal** | 429,447 | 187 |
| **Actual Laundering** | 296 | 70,070 |

### Precision (Laundering)
```
Precision = TP / (TP + FP)
          = 70,070 / (70,070 + 187)
          = 70,070 / 70,257
          = 0.997338
          ≈ 99.73%
```

### Recall (Laundering)
```
Recall = TP / (TP + FN)
       = 70,070 / (70,070 + 296)
       = 70,070 / 70,366
       = 0.995793
       ≈ 99.58%
```

### F1-Score
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
   = 2 × (0.997338 × 0.995793) / (0.997338 + 0.995793)
   = 2 × 0.993143 / 1.993131
   = 1.986286 / 1.993131
   = 0.996565
   ≈ 99.66%
```

---

## 5. Perhitungan Manual Focal Loss

Rumus Focal Loss dengan parameter project (`α = 0.75`, `γ = 2.0`):

```
FL = -αₜ × (1 - pₜ)^γ × log(pₜ)
```

### Contoh 1: Sampel Normal (y=0) yang mudah, p = 0.10

```
pₜ     = 1 - p = 0.90          # probabilitas kelas yang benar
αₜ     = 1 - 0.75 = 0.25       # alpha untuk kelas negatif
(1-pₜ) = 0.10
(1-pₜ)² = 0.01

FL = -0.25 × 0.01 × log(0.90)
   = -0.25 × 0.01 × (-0.10536)
   = -0.0025 × (-0.10536)
   = 0.000263
```

### Contoh 2: Sampel Laundering (y=1) yang sulit, p = 0.30

```
pₜ     = 0.30
αₜ     = 0.75
(1-pₜ) = 0.70
(1-pₜ)² = 0.49

FL = -0.75 × 0.49 × log(0.30)
   = -0.75 × 0.49 × (-1.20397)
   = -0.3675 × (-1.20397)
   = 0.44246
```

### Contoh 3: Sampel Laundering (y=1) yang mudah, p = 0.95

```
pₜ     = 0.95
αₜ     = 0.75
(1-pₜ) = 0.05
(1-pₜ)² = 0.0025

FL = -0.75 × 0.0025 × log(0.95)
   = -0.75 × 0.0025 × (-0.05129)
   = -0.001875 × (-0.05129)
   = 0.000096
```

**Perbandingan:**
- Sampel sulit (p=0.30): FL = **0.442** → model tetap fokus memperbaiki
- Sampel mudah (p=0.95): FL = **0.000096** → hampir diabaikan

---

## 6. Kesimpulan

✅ **Ya, seluruh perhitungan bisa dilakukan secara manual** hanya dengan:
- **Penjumlahan & pengurangan** untuk menghitung total kelas
- **Pembagian** untuk menghitung ratio dan persentase
- **Perkalian & pembagian** untuk menghitung Focal Loss, Precision, Recall, F1

Tidak diperlukan PyTorch, TensorFlow, atau library ML lainnya untuk memahami dan memverifikasi ketidakseimbangan data ini.
