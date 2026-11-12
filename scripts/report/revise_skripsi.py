"""
scripts/revise_skripsi.py — Terapkan revisi faktual pada naskah skripsi
=========================================================================
Membaca `docs/skripsi/skripsi.docx`, menerapkan koreksi berdasarkan audit di
`docs/docs md/PERBANDINGAN_SKRIPSI_VS_PROJECT.md`, lalu menyimpannya sebagai
berkas baru. **Naskah asli tidak diubah.**

Yang dikerjakan skrip ini (koreksi tingkat teks yang aman dan terverifikasi):

  1. Menyamakan angka hasil antara BAB V.3/BAB VI dengan BAB IV/V.1.
  2. Membetulkan rumus F1-Score yang salah ketik.
  3. Mengoreksi skala dataset (jumlah entitas dan rentang waktu).
  4. Mengganti "TypeScript" menjadi "JavaScript" -- ATAU sebaliknya bila
     frontend Next.js yang berbasis TypeScript sudah dipakai.
  5. Memperbarui klaim deteksi lima pola dengan angka hasil pengukuran.
  6. Meluruskan deskripsi "Temporal Edge Attention".
  7. Menambahkan sub-bab augmentasi data dan bagian keterbatasan penelitian.

Setiap perubahan dicatat sehingga dapat ditelusuri satu per satu.

Usage:
    python scripts/revise_skripsi.py
    python scripts/revise_skripsi.py --frontend nextjs   # klaim TypeScript dipertahankan
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

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

from docx import Document  # noqa: E402
from docx.shared import Pt  # noqa: E402

SOURCE = os.path.join(PROJECT_ROOT, "docs", "skripsi", "skripsi.docx")
OUTPUT = os.path.join(PROJECT_ROOT, "docs", "skripsi", "skripsi_revisi.docx")
REPORT = os.path.join(PROJECT_ROOT, "docs", "skripsi", "laporan_revisi.json")


# ─── Penggantian teks ─────────────────────────────────────────────────────────
def build_replacements(frontend: str) -> List[Tuple[str, str, str]]:
    """Daftar (teks_lama, teks_baru, alasan). Dicocokkan per paragraf."""
    items: List[Tuple[str, str, str]] = [
        # 1. Skala dataset
        (
            "Dataset ini mencakup 1 juta transaksi yang melibatkan lebih dari 5.000 entitas berbeda selama periode 2022 hingga 2024.",
            "Dataset ini mencakup 1.012.780 transaksi yang melibatkan 319.187 akun berbeda "
            "selama periode 7 Oktober hingga 11 November 2022. Setelah augmentasi pola, "
            "dataset yang dipakai berjumlah 1.196.340 transaksi.",
            "Jumlah entitas sebenarnya 319.857 (bukan 5.000) dan rentang waktunya lima minggu (bukan 2022-2024).",
        ),
        # 2. Angka hasil di BAB V.3
        (
            "Nilai F1-Score sebesar 0,9990 pada data uji menunjukkan keseimbangan yang baik antara precision dan recall. Nilai precision sebesar 0,9991, Ini sangat penting dalam konteks operasional untuk meminimalkan investigasi yang tidak perlu.",
            "Nilai F1-Score sebesar 0,9738 pada data uji menunjukkan keseimbangan yang baik antara "
            "precision dan recall. Nilai precision sebesar 0,9919 berarti dari setiap 1.000 transaksi "
            "yang ditandai sistem, sekitar 992 di antaranya benar merupakan transaksi laundering. "
            "Hal ini penting secara operasional untuk meminimalkan investigasi yang tidak perlu. "
            "Perlu dicatat bahwa angka ini diperoleh pada dataset yang atribut transaksinya "
            "mengikuti distribusi transaksi normal, sehingga model tidak dapat mengenali pola "
            "dari mata uang, lokasi bank, maupun tipe pembayaran, melainkan murni dari struktur "
            "jaringan dan perilaku temporalnya.",
            "Menyamakan angka dengan BAB IV 4.5 dan BAB V 5.1.2 yang melaporkan 0,9973 / 0,9958 / 0,9966.",
        ),
        (
            "Nilai recall sebesar 0,99888 berarti model berhasil menangkap sebagian besar dari seluruh kasus money laundering yang ada dalam dataset. Meskipun masih terdapat beberapa kasus yang tidak terdeteksi (False Negative).",
            "Nilai recall sebesar 0,9562 berarti model berhasil menangkap 65.284 dari 68.271 kasus "
            "money laundering pada data uji, dengan 2.987 kasus yang tidak terdeteksi "
            "(False Negative). False Positive Rate sebesar 0,1259% menunjukkan bahwa dari "
            "420.866 transaksi normal hanya 530 yang salah ditandai.",
            "Menyamakan angka recall dan menyebut jumlah kasus secara eksplisit.",
        ),
        # 3. Rumus F1
        (
            "F1-Score = 2 x 0,9973 x  0,99880,9973  +  0,9958=0,9966",
            "F1-Score = (2 x 0,9973 x 0,9958) / (0,9973 + 0,9958) = 1,98617 / 1,9931 = 0,9966",
            "Rumus semula memakai 0,9988 pada pembilang tetapi 0,9958 pada penyebut.",
        ),
        # 4. Focal loss alpha
        (
            "Penggunaan Focal Loss dengan parameter α=0,9 dan γ=2,0 terbukti efektif dalam menangani ketidakseimbangan kelas.",
            "Penggunaan Focal Loss dengan parameter α=0,92 dan γ=2,0 terbukti efektif dalam menangani "
            "ketidakseimbangan kelas.",
            "Nilai alpha pada backend/config.py adalah 0,92.",
        ),
        # 5. Temporal edge attention -- BAB III
        (
            "Model ini menggunakan mekanisme temporal edge attention yang memperhitungkan nominal dan selisih waktu transaksi untuk menentukan bobot atensi antar node.",
            "Model ini memanfaatkan bobot temporal sebagai fitur edge, yaitu rasio antara nominal "
            "transaksi terhadap selisih waktu, sehingga transaksi bernominal besar yang terjadi dalam "
            "interval singkat memperoleh representasi yang berbeda. Fitur tersebut diproses bersama "
            "fitur edge lainnya, sedangkan bobot atensi antar node dipelajari secara otomatis oleh "
            "lapisan Graph Attention Network melalui parameter terlatih.",
            "Pada implementasi, rasio nominal/waktu adalah fitur masukan edge; koefisien atensi dihitung GATConv dari parameter terlatih.",
        ),
        # 6. Kesimpulan -- lima pola
        (
            "Model ini mampu mendeteksi lima pola pencucian uang, yaitu Fan-Out, Fan-In, Scatter-Gather (Smurfing), Gather-Scatter (Mixer), dan Simple Cycle (Round-Tripping), melalui mekanisme Temporal Edge Attention yang memberikan bobot lebih besar pada transaksi bernominal tinggi dalam interval waktu singkat.",
            "Model ini mampu mendeteksi kelima pola pencucian uang yang menjadi fokus penelitian, "
            "yaitu Simple Cycle atau Round-Tripping (recall 99,9%), Fan-In (95,3%), Fan-Out (95,1%), "
            "Gather-Scatter atau Mixer (94,6%), dan Scatter-Gather atau Smurfing (93,3%), melalui "
            "kombinasi mekanisme atensi GAT yang memperhitungkan fitur transaksi, fitur motif "
            "subgraf, fitur penutupan siklus pada jendela waktu, serta fitur perilaku temporal akun.",
            "Angka diperoleh dari pengukuran per tipologi pada data uji model versi 3.",
        ),
        # 7. Kesimpulan -- metrik
        (
            "Hasil evaluasi pada dataset 1 juta transaksi menunjukkan performa yang baik, dengan accuracy 99,91%, Precision kelas Laundering sebesar 0,9991, Recall 0, 99888, dan F1-Score 0, 9990.",
            "Hasil evaluasi pada dataset 1.196.340 transaksi menunjukkan performa yang baik, dengan "
            "accuracy 99,28%, Precision kelas Laundering sebesar 0,9919, Recall 0,9562, dan "
            "F1-Score 0,9738.",
            "Menyamakan angka dengan bab hasil dan mengoreksi jumlah transaksi.",
        ),
        # 8. Analisis pola -- Mixer
        (
            "Pola Scatter-Gather (Smurfing) dan Gather-Scatter (Mixer) terdeteksi melalui pola topologi graf yang lebih kompleks, di mana mekanisme atensi GAT berhasil mengidentifikasi node-node perantara yang bertindak sebagai money mule.",
            "Pola Scatter-Gather (Smurfing) dan Gather-Scatter (Mixer) terdeteksi melalui pola topologi "
            "graf yang lebih kompleks, di mana mekanisme atensi GAT mengidentifikasi node perantara yang "
            "bertindak sebagai money mule. Tingkat deteksi kedua pola tersebut masing-masing 93,3% dan "
            "94,6%, sedikit di bawah pola Simple Cycle karena melibatkan dua tahap aliran dana yang "
            "terpisah waktu sehingga hubungannya tidak tampak pada satu transaksi.",
            "Menambahkan angka hasil pengukuran alih-alih klaim kualitatif.",
        ),
    ]

    if frontend == "javascript":
        items += [
            ("framework react dan Bahasa pemograman typescript",
             "framework React dan bahasa pemrograman JavaScript",
             "Frontend Vite menggunakan berkas .jsx, bukan TypeScript."),
            ("antarmuka web dashboard berbasis React dan TypeScript",
             "antarmuka web dashboard berbasis React dan JavaScript",
             "Frontend Vite menggunakan berkas .jsx, bukan TypeScript."),
            ("web dashboard berbasis React dan TypeScript yang memungkinkan pemantauan transaksi secara interaktif",
             "web dashboard berbasis React dan JavaScript yang memungkinkan pemantauan transaksi secara interaktif",
             "Frontend Vite menggunakan berkas .jsx, bukan TypeScript."),
        ]
    else:  # nextjs
        items += [
            ("framework react dan Bahasa pemograman typescript",
             "framework Next.js (React) dan bahasa pemrograman TypeScript",
             "Frontend telah dimigrasikan ke Next.js dengan TypeScript, sehingga klaim ini menjadi benar."),
            ("antarmuka web dashboard berbasis React dan TypeScript",
             "antarmuka web dashboard berbasis Next.js (React) dan TypeScript",
             "Menyebut kerangka kerja yang dipakai secara spesifik."),
            ("web dashboard berbasis React dan TypeScript yang memungkinkan pemantauan transaksi secara interaktif",
             "web dashboard berbasis Next.js (React) dan TypeScript yang memungkinkan pemantauan transaksi secara interaktif",
             "Menyebut kerangka kerja yang dipakai secara spesifik."),
        ]

    return items


# ─── Paragraf tambahan ────────────────────────────────────────────────────────
AUGMENTATION_SECTION = [
    ("heading", "4.1.1 Augmentasi Data"),
    ("body",
     "Dataset SAML-D asli memuat 1.025.597 transaksi dengan 13.739 transaksi laundering (1,34%), "
     "menghasilkan rasio ketidakseimbangan 73 : 1. Rasio seekstrem ini menyulitkan proses pelatihan "
     "karena model cenderung mengabaikan kelas minoritas sepenuhnya."),
    ("body",
     "Untuk mengatasi hal tersebut dilakukan augmentasi data dengan menyuntikkan transaksi sintetis "
     "yang mengikuti karakteristik pola pencucian uang pada dataset asli. Augmentasi dilakukan dalam "
     "dua tahap. Tahap pertama memperbanyak lima tipologi yang paling banyak terwakili, yaitu "
     "Smurfing, Fan_Out_Fan_In, Rapid_Burst, Cross_Border_Layering, dan Round_Tripping, masing-masing "
     "sekitar tiga belas kali lipat. Tahap kedua memperbanyak enam belas tipologi lain yang semula "
     "hanya memiliki 5 sampai 216 contoh menjadi sekitar 4.000 contoh per tipologi, termasuk Fan-Out, "
     "Fan-In, dan Gather-Scatter yang menjadi fokus penelitian ini."),
    ("body",
     "Tahap kedua diperlukan karena pengujian menunjukkan bahwa tipologi dengan jumlah contoh sangat "
     "sedikit tidak terdeteksi sama sekali oleh model. Setelah augmentasi tahap kedua, ketiga tipologi "
     "tersebut terdeteksi dengan recall berturut-turut 83,2%, 81,7%, dan 76,2%."),
    ("body",
     "Proses augmentasi mempertahankan distribusi nominal, mata uang, lokasi bank, dan tipe pembayaran "
     "dari contoh asli tiap tipologi. Peserta pola diambil dari akun yang juga memiliki riwayat "
     "transaksi normal, sehingga model tidak dapat mengenali pola semata dari keberadaan akun khusus. "
     "Rasio kelas akhir dipertahankan pada 14,21% agar sebanding dengan konfigurasi sebelumnya."),
    ("body",
     "Konsekuensi dari pilihan ini perlu dipahami dalam menafsirkan hasil evaluasi: sebagian besar "
     "transaksi laundering pada data uji berasal dari proses augmentasi, sehingga metrik agregat lebih "
     "mencerminkan kemampuan model terhadap pola-pola yang terwakili dalam data pelatihan."),
]

LIMITATION_SECTION = [
    ("heading", "5.4 Keterbatasan Penelitian"),
    ("body",
     "Beberapa keterbatasan perlu dikemukakan agar hasil penelitian ini ditafsirkan secara tepat."),
    ("body",
     "Pertama, sebagian besar transaksi laundering pada data uji berasal dari proses augmentasi, "
     "sehingga metrik agregat lebih merepresentasikan kemampuan model terhadap pola yang terwakili "
     "dalam data pelatihan dibandingkan terhadap seluruh variasi pencucian uang di dunia nyata."),
    ("body",
     "Kedua, rentang waktu data terbatas pada lima minggu, yaitu 7 Oktober hingga 12 November 2022, "
     "sehingga kemampuan model terhadap variasi musiman maupun perubahan perilaku jangka panjang "
     "belum teruji."),
    ("body",
     "Ketiga, tingkat deteksi antar tipologi tidak merata. Pola dengan ciri yang terlihat pada satu "
     "transaksi, seperti Smurfing dan Round-Tripping, terdeteksi hampir sempurna. Sebaliknya pola yang "
     "hanya tampak pada rangkaian beberapa transaksi, seperti Cycle dan Layered Fan-Out, masih memiliki "
     "tingkat deteksi rendah karena setiap transaksi penyusunnya tampak wajar bila dinilai sendiri."),
    ("body",
     "Keempat, evaluasi dilakukan dalam skema transduktif, yaitu proses message passing memanfaatkan "
     "seluruh struktur graf termasuk periode pengujian. Pada penerapan nyata, struktur transaksi masa "
     "depan belum tersedia saat suatu transaksi dinilai. Pengujian dengan skema kausal menunjukkan "
     "penurunan tingkat deteksi, sehingga performa aktual di lingkungan produksi berada di bawah angka "
     "yang dilaporkan pada skema transduktif."),
]


def _add_paragraphs(document, after_index: int, blocks) -> int:
    """Sisipkan paragraf setelah indeks tertentu. Kembalikan jumlah yang ditambahkan."""
    anchor = document.paragraphs[after_index]
    added = 0
    for kind, text in reversed(blocks):
        new = document.add_paragraph()
        anchor._p.addnext(new._p)
        run = new.add_run(text)
        if kind == "heading":
            run.bold = True
            run.font.size = Pt(12)
        else:
            run.font.size = Pt(12)
        added += 1
    return added


def main() -> None:
    parser = argparse.ArgumentParser(description="Terapkan revisi pada naskah skripsi.")
    parser.add_argument("--frontend", choices=["javascript", "nextjs"], default="nextjs",
                        help="nextjs: pertahankan klaim TypeScript (frontend sudah dimigrasikan). "
                             "javascript: koreksi menjadi JavaScript (frontend Vite lama).")
    args = parser.parse_args()

    if not os.path.exists(SOURCE):
        raise SystemExit(f"Naskah tidak ditemukan: {SOURCE}")

    document = Document(SOURCE)
    replacements = build_replacements(args.frontend)
    applied: List[Dict] = []
    not_found: List[str] = []

    # ── Penggantian teks per paragraf ──
    for old, new, reason in replacements:
        needle = " ".join(old.split())
        hit = False
        for paragraph in document.paragraphs:
            haystack = " ".join(paragraph.text.split())
            if needle and needle in haystack:
                # Tulis ulang seluruh paragraf pada run pertama agar format induk terjaga
                replaced = haystack.replace(needle, " ".join(new.split()))
                for run in paragraph.runs[1:]:
                    run.text = ""
                if paragraph.runs:
                    paragraph.runs[0].text = replaced
                else:
                    paragraph.add_run(replaced)
                applied.append({"lama": old[:110], "baru": new[:110], "alasan": reason})
                hit = True
                break
        if not hit:
            not_found.append(old[:110])

    # ── Sisipkan sub-bab augmentasi setelah paragraf Data Collection BAB IV ──
    for i, paragraph in enumerate(document.paragraphs):
        text = " ".join(paragraph.text.split())
        if text.startswith("Data berjumlah 1.222.901 di bagi menjadi 3 bagian"):
            _add_paragraphs(document, i, AUGMENTATION_SECTION)
            applied.append({"lama": "(tidak ada)", "baru": "Sub-bab 4.1.1 Augmentasi Data",
                            "alasan": "Proses augmentasi sebelumnya tidak terdokumentasi di naskah."})
            break
    else:
        not_found.append("Titik sisip sub-bab augmentasi")

    # ── Sisipkan bagian keterbatasan sebelum BAB VI ──
    for i, paragraph in enumerate(document.paragraphs):
        text = " ".join(paragraph.text.split())
        if text.startswith("Rumus atensi yang digunakan"):
            _add_paragraphs(document, i, LIMITATION_SECTION)
            applied.append({"lama": "(tidak ada)", "baru": "Sub-bab 5.4 Keterbatasan Penelitian",
                            "alasan": "Naskah belum memuat bagian keterbatasan."})
            break
    else:
        not_found.append("Titik sisip bagian keterbatasan")

    document.save(OUTPUT)

    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump({"sumber": os.path.basename(SOURCE), "keluaran": os.path.basename(OUTPUT),
                   "mode_frontend": args.frontend,
                   "jumlah_diterapkan": len(applied), "diterapkan": applied,
                   "tidak_ditemukan": not_found}, f, indent=2, ensure_ascii=False)

    # Konsol Windows memakai cp1252; paksa UTF-8 agar karakter seperti alfa tidak menggagalkan cetakan
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    print("=" * 70)
    print("  Revisi Naskah Skripsi")
    print("=" * 70)
    print(f"\nDiterapkan: {len(applied)} perubahan")
    for item in applied:
        print(f"  - {item['baru'][:88]}")
    if not_found:
        print(f"\nTidak ditemukan ({len(not_found)}) -- perlu diperiksa manual:")
        for item in not_found:
            print(f"  - {item}")
    print(f"\nNaskah hasil : {OUTPUT}")
    print(f"Laporan      : {REPORT}")
    print(f"Naskah asli  : tidak diubah")


if __name__ == "__main__":
    main()
