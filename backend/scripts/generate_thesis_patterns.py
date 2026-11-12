"""
generate_thesis_patterns.py — Dataset v4: lima pola sesuai definisi skripsi
=============================================================================
Membangun ulang dataset dengan tepat lima tipologi pencucian uang sebagaimana
didefinisikan pada BAB II naskah skripsi (mengacu Zola, 2026):

    1. Fan-Out                      simpul sentral menyebar ke banyak simpul berbeda
    2. Fan-In                       banyak simpul sumber mengalir ke satu simpul target
    3. Scatter-Gather (Smurfing)    fan-out lalu fan-in melalui perantara yang sama
    4. Gather-Scatter (Mixer)       satu perantara mengumpulkan lalu menyebarkan
    5. Simple Cycle (Round-Tripping) dana berputar dan kembali ke simpul awal

Perbedaan mendasar dari generator sebelumnya
---------------------------------------------
Generator lama membuat pola laundering dengan atribut yang mencolok: 91% lintas
negara, nominal besar, dan tipe pembayaran yang nyaris tidak muncul pada
transaksi normal (`Payment_type="Wire"` terukur 93,5% laundering). Akibatnya
sebagian "keberhasilan" model sebenarnya membaca atribut tersebut, bukan
mengenali pola.

Generator ini mengambil **seluruh atribut dari distribusi transaksi normal**:
mata uang, lokasi bank, tipe pembayaran, dan besaran nominal. Dengan demikian
satu-satunya yang membedakan transaksi laundering adalah:

  * **struktur** hubungan antar akun (fan, siklus, perantara), dan
  * **perilaku** temporal (transaksi beruntun rapat, nominal terpecah rata).

Inilah uji yang sesungguhnya bagi pendekatan berbasis graf: apakah model dapat
mengenali pola dari topologi dan perilaku, bukan dari penanda atribut.

Skrip ini juga MEMVERIFIKASI ketiadaan penanda atribut: setiap nilai kategorikal
diperiksa proporsi laundering-nya, dan peringatan dimunculkan bila ada nilai yang
menyimpang jauh dari proporsi dasar.

Usage:
    python -m backend.scripts.generate_thesis_patterns [--rows-per-pattern 34000]

Output:
    data/SAML-D_v4.csv, train_v4.csv, val_v4.csv, test_v4.csv
    data/augmentation_report_v4.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime, timedelta
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from backend.config import DATA_DIR, resolve

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# PENTING: memakai berkas cadangan, bukan SAML-D.csv. Berkas SAML-D.csv sudah
# ditimpa augmentasi terdahulu yang menyuntikkan mata uang "US Dollar",
# "Swiss Franc", "HK Dollar", dan "Singapore Dollar" -- seluruhnya 100% laundering,
# bahkan dengan kapitalisasi berbeda dari aslinya ("Swiss franc" vs "Swiss Franc").
# Nilai-nilai itu menjadi penanda label terselubung. Berkas cadangan adalah
# dataset Kaggle yang asli: 1.012.780 baris dengan 922 laundering (0,091%).
SOURCE_CSV = resolve("SAML-D_original_backup.csv")
OUTPUT_CSV = resolve("SAML-D_v4.csv")
REPORT_JSON = resolve("augmentation_report_v4.json")

SPLIT_FRACTIONS = {"train": 0.408867, "val": 0.182273, "test": 0.408860}

COLUMNS = [
    "Time", "Date", "Sender_account", "Receiver_account", "Amount",
    "Payment_currency", "Received_currency", "Sender_bank_location",
    "Receiver_bank_location", "Payment_type", "Is_laundering", "Laundering_type",
]

# Lima pola sesuai BAB II skripsi
PATTERNS = ["Fan_Out", "Fan_In", "Scatter_Gather", "Gather_Scatter", "Simple_Cycle"]

# Pemetaan tipologi SAML-D asli ke lima pola di atas. Baris laundering asli yang
# sesuai tetap dipertahankan; selebihnya dikeluarkan dari lingkup penelitian.
ORIGINAL_MAPPING = {
    "Fan_Out": "Fan_Out",
    "Layered_Fan_Out": "Fan_Out",
    "Fan_In": "Fan_In",
    "Layered_Fan_In": "Fan_In",
    "Smurfing": "Scatter_Gather",
    "Scatter-Gather": "Scatter_Gather",
    "Fan_Out_Fan_In": "Scatter_Gather",
    "Gather-Scatter": "Gather_Scatter",
    "Round_Tripping": "Simple_Cycle",
    "Cycle": "Simple_Cycle",
}


# ─── Distribusi atribut transaksi normal ──────────────────────────────────────
class NormalProfile:
    """
    Distribusi atribut yang diambil dari transaksi NORMAL.

    Seluruh transaksi laundering sintetis memakai profil ini, sehingga distribusi
    marginal mata uang, lokasi, tipe pembayaran, dan nominal pada kelas laundering
    tidak dapat dibedakan dari kelas normal. Yang membedakan hanya struktur dan
    perilaku temporalnya.
    """

    def __init__(self, normal: pd.DataFrame):
        self.rows = normal.reset_index(drop=True)
        self.n = len(self.rows)
        self.amounts = normal["Amount"].values
        # Kombinasi atribut diambil utuh per baris agar korelasi antar atribut
        # (mis. mata uang dengan lokasi bank) ikut terjaga
        self.attr_cols = ["Payment_currency", "Received_currency",
                          "Sender_bank_location", "Receiver_bank_location", "Payment_type"]
        self.attrs = normal[self.attr_cols].values

    def draw_attrs(self) -> Dict[str, str]:
        """Satu kombinasi atribut, diambil utuh dari satu transaksi normal."""
        row = self.attrs[np.random.randint(self.n)]
        return dict(zip(self.attr_cols, row))

    def draw_amount(self) -> float:
        """Nominal acak dari distribusi nominal transaksi normal."""
        return float(np.random.choice(self.amounts))


def _row(dt: datetime, sender: str, receiver: str, amount: float,
         attrs: Dict[str, str], pattern: str) -> Dict:
    return {
        "Time": dt.strftime("%H:%M:%S"),
        "Date": dt.strftime("%Y-%m-%d"),
        "Sender_account": sender,
        "Receiver_account": receiver,
        "Amount": round(max(float(amount), 1.0), 2),
        "Payment_currency": attrs["Payment_currency"],
        "Received_currency": attrs["Received_currency"],
        "Sender_bank_location": attrs["Sender_bank_location"],
        "Receiver_bank_location": attrs["Receiver_bank_location"],
        "Payment_type": attrs["Payment_type"],
        "Is_laundering": 1,
        "Laundering_type": pattern,
    }


# ─── Generator per pola ───────────────────────────────────────────────────────
# Tiap fungsi menghasilkan SATU instance pola. Nominal dan atribut berasal dari
# distribusi normal; yang menjadi ciri pola adalah topologi dan waktunya.

def gen_fan_out(base: datetime, pool: List[str], p: NormalProfile) -> List[Dict]:
    """Satu simpul sentral menyebar dana ke banyak simpul berbeda."""
    n = random.randint(12, 30)
    participants = random.sample(pool, k=min(n + 1, len(pool)))
    src, dsts = participants[0], participants[1:]
    # Dana dipecah rata -- ciri perilaku, bukan ciri atribut
    per = p.draw_amount()
    attrs = p.draw_attrs()
    return [
        _row(base + timedelta(seconds=i * random.randint(30, 180)),
             src, dst, per * random.uniform(0.95, 1.05), attrs, "Fan_Out")
        for i, dst in enumerate(dsts)
    ]


def gen_fan_in(base: datetime, pool: List[str], p: NormalProfile) -> List[Dict]:
    """Banyak simpul sumber mengalirkan dana ke satu simpul target."""
    n = random.randint(12, 30)
    participants = random.sample(pool, k=min(n + 1, len(pool)))
    dst, srcs = participants[0], participants[1:]
    per = p.draw_amount()
    attrs = p.draw_attrs()
    return [
        _row(base + timedelta(seconds=i * random.randint(30, 180)),
             src, dst, per * random.uniform(0.95, 1.05), attrs, "Fan_In")
        for i, src in enumerate(srcs)
    ]


def gen_scatter_gather(base: datetime, pool: List[str], p: NormalProfile) -> List[Dict]:
    """
    Smurfing: dana dipecah dari satu sumber ke banyak perantara,
    lalu dikumpulkan kembali ke satu target melalui perantara yang sama.
    """
    n = random.randint(8, 20)
    participants = random.sample(pool, k=min(n + 2, len(pool)))
    src, dst, mids = participants[0], participants[1], participants[2:]
    per = p.draw_amount()
    attrs_out, attrs_in = p.draw_attrs(), p.draw_attrs()

    rows = [
        _row(base + timedelta(seconds=i * random.randint(30, 150)),
             src, mid, per * random.uniform(0.95, 1.05), attrs_out, "Scatter_Gather")
        for i, mid in enumerate(mids)
    ]
    # Tahap pengumpulan: nominal sedikit menyusut (biaya perantara)
    later = base + timedelta(minutes=random.randint(30, 300))
    rows += [
        _row(later + timedelta(seconds=i * random.randint(30, 150)),
             mid, dst, per * random.uniform(0.88, 0.97), attrs_in, "Scatter_Gather")
        for i, mid in enumerate(mids)
    ]
    return rows


def gen_gather_scatter(base: datetime, pool: List[str], p: NormalProfile) -> List[Dict]:
    """Mixer: satu perantara mengumpulkan dana dari banyak sumber lalu menyebarkannya."""
    n_in = random.randint(8, 18)
    n_out = random.randint(8, 18)
    participants = random.sample(pool, k=min(n_in + n_out + 1, len(pool)))
    mid = participants[0]
    srcs = participants[1:1 + n_in]
    dsts = participants[1 + n_in:]
    per = p.draw_amount()
    attrs_in, attrs_out = p.draw_attrs(), p.draw_attrs()

    rows = [
        _row(base + timedelta(seconds=i * random.randint(30, 150)),
             src, mid, per * random.uniform(0.95, 1.05), attrs_in, "Gather_Scatter")
        for i, src in enumerate(srcs)
    ]
    later = base + timedelta(minutes=random.randint(30, 300))
    total_in = per * len(srcs)
    per_out = total_in / max(len(dsts), 1) * random.uniform(0.88, 0.97)
    rows += [
        _row(later + timedelta(seconds=i * random.randint(30, 150)),
             mid, dst, per_out * random.uniform(0.95, 1.05), attrs_out, "Gather_Scatter")
        for i, dst in enumerate(dsts)
    ]
    return rows


def gen_simple_cycle(base: datetime, pool: List[str], p: NormalProfile) -> List[Dict]:
    """Dana berputar melewati beberapa perantara lalu kembali ke simpul asal."""
    length = random.randint(3, 8)
    chain = random.sample(pool, k=min(length, len(pool)))
    chain.append(chain[0])           # tutup siklus
    amount = p.draw_amount()
    attrs = p.draw_attrs()
    return [
        _row(base + timedelta(minutes=i * random.randint(3, 25)),
             chain[i], chain[i + 1], amount * (0.97 ** i), attrs, "Simple_Cycle")
        for i in range(len(chain) - 1)
    ]


GENERATORS: Dict[str, Callable] = {
    "Fan_Out": gen_fan_out,
    "Fan_In": gen_fan_in,
    "Scatter_Gather": gen_scatter_gather,
    "Gather_Scatter": gen_gather_scatter,
    "Simple_Cycle": gen_simple_cycle,
}


# ─── Verifikasi ketiadaan penanda atribut ────────────────────────────────────
def verify_no_attribute_leak(df: pd.DataFrame) -> Dict:
    """
    Periksa apakah ada nilai kategorikal yang menjadi penanda label terselubung.

    Untuk tiap nilai, dihitung proporsi laundering-nya. Nilai yang menyimpang
    jauh dari proporsi dasar berarti model dapat menebak label dari atribut itu
    saja, tanpa memahami pola -- persis cacat yang ditemukan pada dataset
    sebelumnya (`Payment_type="Wire"` berproporsi 93,5% laundering).
    """
    base_rate = df["Is_laundering"].mean()
    findings = []
    for col in ["Payment_currency", "Received_currency", "Sender_bank_location",
                "Receiver_bank_location", "Payment_type"]:
        table = df.groupby(col)["Is_laundering"].agg(["mean", "count"])
        table = table[table["count"] >= 500]
        for value, row in table.iterrows():
            ratio = row["mean"] / base_rate if base_rate else 0
            if ratio > 1.5 or ratio < 0.5:
                findings.append({
                    "kolom": col, "nilai": str(value),
                    "proporsi_laundering": round(float(row["mean"]), 4),
                    "rasio_thd_dasar": round(float(ratio), 2),
                    "jumlah": int(row["count"]),
                })
    return {"proporsi_dasar": round(float(base_rate), 4),
            "penyimpangan": sorted(findings, key=lambda x: -abs(x["rasio_thd_dasar"] - 1))}


def generate_normal_rows(n: int, pool: List[str], profile: NormalProfile,
                         start: pd.Timestamp, span_days: int) -> pd.DataFrame:
    """Transaksi normal tambahan, dibangkitkan secara vektorial."""
    if n <= 0:
        return pd.DataFrame(columns=COLUMNS)
    idx = np.random.randint(0, profile.n, size=n)
    sample = profile.rows.iloc[idx]
    accounts = np.array(pool)
    senders = accounts[np.random.randint(0, len(accounts), size=n)]
    receivers = accounts[np.random.randint(0, len(accounts), size=n)]
    same = senders == receivers
    receivers[same] = accounts[(np.arange(same.sum()) + 1) % len(accounts)]

    minutes = np.random.randint(0, span_days * 24 * 60, size=n)
    stamps = pd.to_datetime(start) + pd.to_timedelta(minutes, unit="m")

    return pd.DataFrame({
        "Time": stamps.strftime("%H:%M:%S"),
        "Date": stamps.strftime("%Y-%m-%d"),
        "Sender_account": senders,
        "Receiver_account": receivers,
        "Amount": np.round(sample["Amount"].values * np.random.uniform(0.9, 1.1, size=n), 2),
        "Payment_currency": sample["Payment_currency"].values,
        "Received_currency": sample["Received_currency"].values,
        "Sender_bank_location": sample["Sender_bank_location"].values,
        "Receiver_bank_location": sample["Receiver_bank_location"].values,
        "Payment_type": sample["Payment_type"].values,
        "Is_laundering": 0,
        "Laundering_type": "Normal",
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Bangun dataset v4 dengan lima pola skripsi.")
    parser.add_argument("--rows-per-pattern", type=int, default=34_000,
                        help="Target jumlah baris laundering per pola (default 34.000).")
    parser.add_argument("--pool-size", type=int, default=8_000,
                        help="Jumlah akun yang terlibat dalam pola laundering (default 8.000).")
    parser.add_argument("--target-ratio", type=float, default=0.1421,
                        help="Proporsi kelas laundering yang dituju (default 0,1421).")
    parser.add_argument("--source", default=SOURCE_CSV,
                        help="Berkas sumber (default: SAML-D_original_backup.csv).")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    print("=" * 74)
    print("  Dataset v4 — Lima Pola sesuai Definisi Skripsi")
    print("=" * 74 + "\n")

    print(f"[1/6] Memuat dataset asli: {os.path.basename(args.source)}")
    df = pd.read_csv(args.source)
    df["Sender_account"] = df["Sender_account"].astype(str)
    df["Receiver_account"] = df["Receiver_account"].astype(str)
    print(f"      {len(df):,} baris | {int(df.Is_laundering.sum()):,} laundering "
          f"({df.Is_laundering.mean()*100:.2f}%)")

    dates = pd.to_datetime(df["Date"])
    start, end = dates.min(), dates.max()
    span_days = max(1, (end - start).days)
    print(f"      Rentang waktu: {start.date()} s/d {end.date()} ({span_days} hari)")

    # ── Pisahkan menurut lingkup ──
    print("\n[2/6] Menyaring menurut lingkup lima pola skripsi ...")
    normal = df[df.Is_laundering == 0].copy()
    laundering = df[df.Is_laundering == 1].copy()

    in_scope = laundering[laundering.Laundering_type.isin(ORIGINAL_MAPPING)].copy()
    out_scope = laundering[~laundering.Laundering_type.isin(ORIGINAL_MAPPING)]
    in_scope["Laundering_type"] = in_scope["Laundering_type"].map(ORIGINAL_MAPPING)

    print(f"      Transaksi normal            : {len(normal):,}")
    print(f"      Laundering dalam lingkup    : {len(in_scope):,} (dipertahankan, dilabeli ulang)")
    print(f"      Laundering di luar lingkup  : {len(out_scope):,} (dikeluarkan)")
    print("      Pemetaan label asli:")
    for old, new in sorted(ORIGINAL_MAPPING.items(), key=lambda kv: kv[1]):
        count = int((laundering.Laundering_type == old).sum())
        if count:
            print(f"        {old:<22} -> {new:<16} {count:>6,}")

    # ── Kumpulan akun ──
    print(f"\n[3/6] Menyiapkan kumpulan akun ...")
    active = pd.concat([normal["Sender_account"], normal["Receiver_account"]])
    counts = active.value_counts()
    pool = counts[counts >= 3].index.tolist()
    random.shuffle(pool)
    pool = pool[:args.pool_size]
    print(f"      {len(pool):,} akun dipakai, seluruhnya memiliki riwayat transaksi normal")

    profile = NormalProfile(normal)
    print(f"      Profil atribut diambil dari {profile.n:,} transaksi normal")

    # ── Bangkitkan pola ──
    print(f"\n[4/6] Membangkitkan lima pola (target {args.rows_per_pattern:,} baris/pola) ...")
    new_rows: List[Dict] = []
    report: Dict[str, Dict] = {}

    for pattern in PATTERNS:
        generator = GENERATORS[pattern]
        rows: List[Dict] = []
        instances = 0
        existing = int((in_scope.Laundering_type == pattern).sum())
        needed = max(0, args.rows_per_pattern - existing)

        while len(rows) < needed:
            offset = random.randint(0, span_days * 24 * 60)
            produced = generator(start.to_pydatetime() + timedelta(minutes=offset), pool, profile)
            if produced:
                rows.extend(produced)
                instances += 1
        rows = rows[:needed]
        new_rows.extend(rows)
        report[pattern] = {
            "asli_dipertahankan": existing,
            "sintetis_ditambahkan": len(rows),
            "total": existing + len(rows),
            "instance_pola": instances,
        }
        print(f"      {pattern:<18} asli {existing:>6,} + sintetis {len(rows):>7,} "
              f"= {existing + len(rows):>7,}  ({instances:,} instance)")

    df_new = pd.DataFrame(new_rows, columns=COLUMNS)

    # ── Transaksi normal pengimbang ──
    total_pos = len(in_scope) + len(df_new)
    target_total = total_pos / args.target_ratio
    n_extra_normal = int(max(0, target_total - len(normal) - total_pos))
    print(f"\n[5/6] Menambah {n_extra_normal:,} transaksi normal "
          f"(target rasio {args.target_ratio*100:.2f}%) ...")
    df_extra = generate_normal_rows(n_extra_normal, pool, profile, start, span_days)

    # ── Gabung + split temporal ──
    print("\n[6/6] Menggabungkan dan membagi secara temporal ...")
    combined = pd.concat([normal, in_scope, df_new, df_extra], ignore_index=True)
    combined["_dt"] = pd.to_datetime(
        combined["Date"].astype(str) + " " + combined["Time"].astype(str),
        format="%Y-%m-%d %H:%M:%S", errors="coerce",
    )
    combined = combined.dropna(subset=["_dt"]).sort_values("_dt", kind="mergesort")
    combined = combined.reset_index(drop=True).drop(columns=["_dt"])[COLUMNS]

    n = len(combined)
    n_train = int(n * SPLIT_FRACTIONS["train"])
    n_val = int(n * SPLIT_FRACTIONS["val"])
    splits = {
        "train": combined.iloc[:n_train],
        "val": combined.iloc[n_train:n_train + n_val],
        "test": combined.iloc[n_train + n_val:],
    }

    combined.to_csv(OUTPUT_CSV, index=False)
    for name, part in splits.items():
        part.to_csv(resolve(f"{name}_v4.csv"), index=False)

    print(f"\n      {'split':<8}{'baris':>12}{'laundering':>13}{'rasio':>9}")
    split_report = {}
    for name, part in splits.items():
        pos = int(part.Is_laundering.sum())
        split_report[name] = {"baris": len(part), "laundering": pos,
                              "rasio_persen": round(pos / len(part) * 100, 2)}
        print(f"      {name:<8}{len(part):>12,}{pos:>13,}{pos/len(part)*100:>8.2f}%")

    total = int(combined.Is_laundering.sum())
    print(f"\n      TOTAL: {n:,} baris | {total:,} laundering ({total/n*100:.2f}%) "
          f"| rasio {(n-total)/total:.2f} : 1")

    # ── Verifikasi ketiadaan penanda atribut ──
    print("\n" + "=" * 74)
    print("  VERIFIKASI: apakah ada atribut yang membocorkan label?")
    print("=" * 74)
    leak = verify_no_attribute_leak(combined)
    print(f"  Proporsi laundering dasar: {leak['proporsi_dasar']*100:.2f}%")
    if leak["penyimpangan"]:
        print(f"  Ditemukan {len(leak['penyimpangan'])} nilai yang menyimpang >1,5x dari dasar:")
        for f in leak["penyimpangan"][:10]:
            print(f"    {f['kolom']:<24} {f['nilai']:<18} "
                  f"{f['proporsi_laundering']*100:>6.2f}%  ({f['rasio_thd_dasar']}x, n={f['jumlah']:,})")
    else:
        print("  Tidak ada. Seluruh nilai kategorikal berproporsi laundering mendekati "
              "proporsi dasar,\n  sehingga label tidak dapat ditebak dari atribut saja.")

    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "sumber": os.path.basename(args.source),
            "keluaran": os.path.basename(OUTPUT_CSV),
            "pola": report,
            "laundering_luar_lingkup_dikeluarkan": len(out_scope),
            "normal_tambahan": len(df_extra),
            "split": split_report,
            "total_baris": n,
            "total_laundering": total,
            "rasio_persen": round(total / n * 100, 2),
            "verifikasi_atribut": leak,
            "seed": SEED,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n  Dataset : {OUTPUT_CSV}")
    print(f"  Laporan : {REPORT_JSON}")
    print("\nSelesai. Lanjut: python -m backend.ml.data_pipeline --version v4\n")


if __name__ == "__main__":
    main()
