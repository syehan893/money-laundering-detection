"""
scripts/make_production_test_csv.py — Berkas CSV untuk uji produksi
=====================================================================
Membuat lima berkas CSV kecil dari data uji yang sudah ada, masing-masing
menguji satu aspek berbeda dari model. Berkas-berkas ini diunggah lewat
halaman "Uji Produksi" pada dashboard, lalu hasil prediksinya dibandingkan
dengan label sebenarnya.

Seluruh baris diambil dari `data/test_v3.csv` — periode waktu yang TIDAK
pernah dilihat model saat pelatihan, sehingga hasilnya mencerminkan performa
pada data baru.

Berkas yang dihasilkan (di `data/production_test/`):

  1. 01_normal_only.csv        500 transaksi normal berurutan -> harapan: nyaris nol ditandai
  2. 02_mixed_realistic.csv    jendela waktu utuh apa adanya  -> harapan: mendekati metrik uji
  3. 03_pattern_fan.csv        300 pola fan-in/fan-out/mixer  -> harapan: recall tinggi
  4. 04_pattern_layering.csv   300 pola berlapis & siklus     -> harapan: recall sedang (pola tersulit)
  5. 05_edge_cases.csv         200 kasus ekstrem              -> harapan: tanpa error, skor wajar

CATATAN: berkas 1 dan 2 dirancang untuk menilai precision/recall/FPR karena
komposisinya alami. Berkas 3-5 sengaja diperkaya pola tertentu, sehingga yang
bermakna hanya **recall per tipologi**; angka FPR pada berkas tersebut tidak
sebanding dengan hasil evaluasi menyeluruh.

Setiap berkas memuat kolom `Is_laundering` sebagai kunci jawaban. Kolom itu
diabaikan saat prediksi dan hanya dipakai untuk menghitung skor pada laporan.

Usage:
    python scripts/make_production_test_csv.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

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

SEED = 7
OUT_DIR = os.path.join(C.DATA_DIR, "production_test")

# Nama pola pada dataset v4 (lima pola definisi skripsi); nama versi lama
# tetap disertakan agar skrip bekerja pada dataset mana pun.
FAN_TYPES = ["Fan_In", "Fan_Out", "Gather_Scatter", "Scatter_Gather",
             "Gather-Scatter", "Scatter-Gather", "Fan_Out_Fan_In"]
LAYERING_TYPES = ["Simple_Cycle", "Cycle", "Round_Tripping", "Layered_Fan_In",
                  "Layered_Fan_Out", "Bipartite", "Stacked Bipartite",
                  "Cross_Border_Layering"]


def _sample(df: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Ambil n baris acak tanpa pengembalian (atau seluruhnya bila kurang)."""
    if len(df) <= n:
        return df.copy()
    idx = rng.choice(len(df), size=n, replace=False)
    return df.iloc[idx].copy()


def main() -> None:
    rng = np.random.default_rng(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    # Pakai versi dataset terbaru yang tersedia
    for candidate in (C.TEST_CSV_V4_PATH, C.TEST_CSV_V3_PATH, C.TEST_CSV_PATH):
        if os.path.exists(candidate):
            source = candidate
            break
    print(f"Sumber: {source}")
    df = pd.read_csv(source)
    print(f"  {len(df):,} baris | {int(df.Is_laundering.sum()):,} laundering\n")

    normal = df[df.Is_laundering == 0]
    laundering = df[df.Is_laundering == 1]

    files = {}

    # ── 1. Hanya transaksi normal ──
    # Diambil dari satu jendela waktu berurutan, bukan acak: batch produksi
    # sungguhnya berupa rentang waktu, dan fitur velocity bergantung pada
    # transaksi bertetangga sehingga jendela berurutan lebih representatif.
    start = int(len(df) * 0.30)
    window = df.iloc[start:start + 4000]
    f1 = window[window.Is_laundering == 0].head(500).copy()
    files["01_normal_only.csv"] = (f1, "500 transaksi normal berurutan dari satu jendela waktu",
                                   "Nyaris tidak ada yang ditandai. Setiap penandaan adalah false positive.")

    # ── 2. Campuran realistis ──
    # Satu jendela waktu utuh apa adanya -- inilah yang paling menyerupai
    # kondisi produksi dan metriknya sebanding dengan hasil evaluasi uji.
    start2 = int(len(df) * 0.55)
    f2 = df.iloc[start2:start2 + 3500].copy().reset_index(drop=True)
    n_pos = int(f2.Is_laundering.sum())
    files["02_mixed_realistic.csv"] = (
        f2, f"{len(f2):,} transaksi berurutan apa adanya, {n_pos} di antaranya laundering "
            f"({n_pos/len(f2)*100:.1f}%)",
        "Jendela waktu utuh tanpa penyaringan. Metriknya sebanding dengan hasil evaluasi pada data uji.")

    # ── 3. Pola fan ──
    fan = laundering[laundering.Laundering_type.isin(FAN_TYPES)]
    f3 = pd.concat([_sample(fan, 240, rng), _sample(normal, 60, rng)])
    f3 = f3.sample(frac=1, random_state=SEED).reset_index(drop=True)
    files["03_pattern_fan.csv"] = (f3, "300 transaksi: 240 pola fan-in/fan-out/mixer + 60 normal",
                                   "Pola penyebaran dan pengumpulan dana. Recall diharapkan tinggi.")

    # ── 4. Pola layering ──
    lay = laundering[laundering.Laundering_type.isin(LAYERING_TYPES)]
    f4 = pd.concat([_sample(lay, 240, rng), _sample(normal, 60, rng)])
    f4 = f4.sample(frac=1, random_state=SEED).reset_index(drop=True)
    files["04_pattern_layering.csv"] = (f4, "300 transaksi: 240 pola berlapis & siklus + 60 normal",
                                        "Pola tersulit: tiap transaksi tampak biasa, yang mencurigakan adalah rangkaiannya. Recall diharapkan paling rendah.")

    # ── 5. Kasus ekstrem ──
    parts = []
    parts.append(df.nlargest(50, "Amount"))                     # nominal terbesar
    parts.append(df.nsmallest(50, "Amount"))                    # nominal terkecil
    cross = df[(df.Sender_bank_location != df.Receiver_bank_location) &
               (df.Payment_currency != df.Received_currency)]
    parts.append(_sample(cross, 50, rng))                       # lintas negara + beda mata uang
    near = df[(df.Amount >= 9000) & (df.Amount < 10000)]
    parts.append(_sample(near, 50, rng))                        # tepat di bawah ambang pelaporan
    f5 = pd.concat(parts).drop_duplicates().sample(frac=1, random_state=SEED).reset_index(drop=True)
    files["05_edge_cases.csv"] = (f5, f"{len(f5)} kasus ekstrem: nominal terbesar/terkecil, lintas negara, dekat ambang lapor",
                                  "Menguji ketahanan, bukan akurasi. Tidak boleh ada error; skor harus tetap masuk akal.")

    # ── Tulis berkas + manifes ──
    manifest = []
    print(f"{'berkas':<28}{'baris':>8}{'laundering':>12}{'rasio':>9}")
    for name, (data, desc, expectation) in files.items():
        path = os.path.join(OUT_DIR, name)
        data.to_csv(path, index=False)
        pos = int(data.Is_laundering.sum())
        manifest.append({
            "berkas": name,
            "deskripsi": desc,
            "harapan": expectation,
            "baris": len(data),
            "laundering": pos,
            "normal": len(data) - pos,
            "rasio_laundering_persen": round(pos / len(data) * 100, 2),
            "tipologi": data[data.Is_laundering == 1].Laundering_type.value_counts().to_dict(),
        })
        print(f"{name:<28}{len(data):>8,}{pos:>12,}{pos/len(data)*100:>8.1f}%")

    with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"sumber": os.path.basename(source), "seed": SEED, "berkas": manifest},
                  f, indent=2, ensure_ascii=False)

    print(f"\nBerkas tersimpan di: {OUT_DIR}")
    print("Manifes            : manifest.json")
    print("\nUnggah berkas-berkas ini lewat halaman 'Uji Produksi' pada dashboard.")


if __name__ == "__main__":
    main()
