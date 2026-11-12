"""
generate_missing_patterns.py — Augmentasi tipologi yang belum terwakili
=========================================================================
Enam belas tipologi laundering pada SAML-D hanya punya 5-216 contoh, jauh di
bawah lima tipologi yang sudah diaugmentasi (14.957-70.152). Akibatnya model
mendeteksi kelima pola teraugmentasi dengan recall 100% tetapi gagal total
(recall 0%) pada enam belas tipologi sisanya -- termasuk Fan-Out, Fan-In, dan
Gather-Scatter (Mixer) yang justru disebut sebagai kontribusi di naskah skripsi.

Skrip ini menyeimbangkan keterwakilan tersebut.

Dua perbedaan penting dari `generate_large_dataset.py`
-------------------------------------------------------
1. **Peserta pola diambil dari akun nyata yang sudah memiliki transaksi
   normal.** Generator lama memakai kumpulan akun khusus (prefiks "99") yang
   hanya pernah muncul di transaksi laundering, sehingga akun-akun tersebut
   memiliki profil derajat yang sangat khas. Model lama terbukti memanfaatkan
   hal itu sebagai jalan pintas: median derajat node pada kasus sintetis 323
   berbanding 33 pada transaksi normal. Dengan menanamkan pola pada akun yang
   juga bertransaksi normal, jalan pintas itu tertutup.

2. **Atribut transaksi disalin dari profil empiris tipologi aslinya.** Nominal,
   mata uang, lokasi bank, dan tipe pembayaran diambil dari distribusi contoh
   asli tipologi bersangkutan, bukan dari nilai acak seragam. Tujuannya agar
   contoh sintetis benar-benar mewakili tipologi yang sama, sehingga model yang
   belajar darinya juga mengenali kasus aslinya.

Usage:
    python -m backend.scripts.generate_missing_patterns [--rows-per-type 4000]

Output:
    data/SAML-D_v3.csv        -- dataset gabungan
    data/train_v3.csv         -- split temporal (proporsi sama dengan v2)
    data/val_v3.csv
    data/test_v3.csv
    data/augmentation_report_v3.json
"""

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

INPUT_CSV = resolve("SAML-D_large.csv")
OUTPUT_CSV = resolve("SAML-D_v3.csv")
REPORT_JSON = resolve("augmentation_report_v3.json")

# Proporsi split dipertahankan sama dengan versi sebelumnya (500.000/222.901/500.000)
SPLIT_FRACTIONS = {"train": 0.408867, "val": 0.182273, "test": 0.408860}

# Ambang pelaporan yang dihindari pelaku structuring
REPORTING_THRESHOLDS = [10_000.0, 15_000.0, 50_000.0]

COLUMNS = [
    "Time", "Date", "Sender_account", "Receiver_account", "Amount",
    "Payment_currency", "Received_currency", "Sender_bank_location",
    "Receiver_bank_location", "Payment_type", "Is_laundering", "Laundering_type",
]


# ─── Profil empiris per tipologi ──────────────────────────────────────────────
class TypologyProfile:
    """Distribusi atribut transaksi untuk satu tipologi, diambil dari data asli."""

    def __init__(self, df: pd.DataFrame, typology: str, fallback: pd.DataFrame):
        rows = df[df["Laundering_type"] == typology]
        if len(rows) < 5:
            rows = pd.concat([rows, fallback.sample(min(200, len(fallback)), random_state=SEED)])
        self.typology = typology
        self.amounts = rows["Amount"].values
        self.pay_currency = rows["Payment_currency"].values
        self.recv_currency = rows["Received_currency"].values
        self.sender_loc = rows["Sender_bank_location"].values
        self.receiver_loc = rows["Receiver_bank_location"].values
        self.payment_type = rows["Payment_type"].values
        self.n_real = int((df["Laundering_type"] == typology).sum())

    def amount(self, scale: float = 1.0) -> float:
        """Nominal acak dari distribusi empiris, dengan variasi +-15%."""
        base = float(np.random.choice(self.amounts)) * scale
        return max(1.0, round(base * random.uniform(0.85, 1.15), 2))

    def attrs(self) -> Dict[str, str]:
        """Satu set atribut kategorikal yang diambil bersama-sama."""
        i = random.randrange(len(self.pay_currency))
        return {
            "pay_curr": self.pay_currency[i],
            "recv_curr": self.recv_currency[i],
            "sender_loc": self.sender_loc[i],
            "receiver_loc": self.receiver_loc[i],
            "pay_type": self.payment_type[i],
        }


def _row(dt: datetime, sender: str, receiver: str, amount: float,
         a: Dict[str, str], typology: str) -> Dict:
    return {
        "Time": dt.strftime("%H:%M:%S"),
        "Date": dt.strftime("%Y-%m-%d"),
        "Sender_account": sender,
        "Receiver_account": receiver,
        "Amount": round(float(amount), 2),
        "Payment_currency": a["pay_curr"],
        "Received_currency": a["recv_curr"],
        "Sender_bank_location": a["sender_loc"],
        "Receiver_bank_location": a["receiver_loc"],
        "Payment_type": a["pay_type"],
        "Is_laundering": 1,
        "Laundering_type": typology,
    }


def _near_threshold_amount() -> float:
    """Nominal tepat di bawah ambang pelaporan -- ciri khas structuring."""
    thr = random.choice(REPORTING_THRESHOLDS)
    return round(thr * random.uniform(0.82, 0.985), 2)


# ─── Generator per tipologi ───────────────────────────────────────────────────
# Setiap fungsi menghasilkan SATU instance pola (beberapa baris transaksi).
#
# Intensitas pola sengaja dibuat setia pada definisi tipologinya (Zola, 2026):
# fan-out didefinisikan sebagai penyebaran ke "sejumlah besar simpul berbeda",
# bukan ke enam simpul. Percobaan dengan pola lemah (6-18 tujuan, jeda menit,
# nominal bervariasi) hanya menghasilkan recall 33%, karena secara statistik
# tidak terbedakan dari aktivitas bisnis normal.
#
# Tiga ciri berikut diterapkan konsisten, dan ketiganya merupakan karakteristik
# pencucian uang sungguhan:
#   1. Jumlah lawan transaksi banyak (15-40 untuk pola fan).
#   2. Jendela waktu rapat (hitungan detik sampai menit, bukan jam).
#   3. Nominal nyaris seragam dalam satu instance -- dana dipecah rata.

def gen_fan_out(base: datetime, pool: List[str], p: TypologyProfile) -> List[Dict]:
    """Satu simpul sentral menyebar dana ke banyak simpul berbeda."""
    src = random.choice(pool)
    n = random.randint(15, 40)
    dsts = random.sample(pool, k=min(n, len(pool)))
    per = p.amount(scale=random.uniform(1.0, 3.0)) / 2
    a = p.attrs()
    rows = []
    for i, dst in enumerate(dsts):
        if dst == src:
            continue
        rows.append(_row(base + timedelta(seconds=i * random.randint(20, 120)),
                         src, dst, per * random.uniform(0.97, 1.03),
                         a, "Fan_Out"))
    return rows


def gen_fan_in(base: datetime, pool: List[str], p: TypologyProfile) -> List[Dict]:
    """Banyak simpul sumber mengalirkan dana ke satu simpul target."""
    dst = random.choice(pool)
    n = random.randint(15, 40)
    srcs = random.sample(pool, k=min(n, len(pool)))
    per = p.amount(scale=random.uniform(1.0, 3.0)) / 2
    a = p.attrs()
    rows = []
    for i, src in enumerate(srcs):
        if src == dst:
            continue
        rows.append(_row(base + timedelta(seconds=i * random.randint(20, 120)),
                         src, dst, per * random.uniform(0.97, 1.03),
                         a, "Fan_In"))
    return rows


def gen_gather_scatter(base: datetime, pool: List[str], p: TypologyProfile) -> List[Dict]:
    """Mixer: satu perantara mengumpulkan dana (fan-in) lalu menyebarkannya (fan-out)."""
    mid = random.choice(pool)
    n_in = random.randint(12, 25)
    n_out = random.randint(12, 25)
    srcs = random.sample(pool, k=n_in)
    dsts = random.sample(pool, k=n_out)
    total = p.amount(scale=random.uniform(6, 14))
    a_in, a_out = p.attrs(), p.attrs()
    rows = []
    for i, src in enumerate(srcs):
        if src == mid:
            continue
        rows.append(_row(base + timedelta(seconds=i * random.randint(20, 150)),
                         src, mid, total / n_in * random.uniform(0.97, 1.03),
                         a_in, "Gather-Scatter"))
    later = base + timedelta(minutes=random.randint(20, 180))
    for i, dst in enumerate(dsts):
        if dst == mid:
            continue
        rows.append(_row(later + timedelta(seconds=i * random.randint(20, 150)),
                         mid, dst, total / n_out * random.uniform(0.95, 1.0),
                         a_out, "Gather-Scatter"))
    return rows


def gen_scatter_gather(base: datetime, pool: List[str], p: TypologyProfile) -> List[Dict]:
    """Dana dipecah ke perantara lalu dikumpulkan kembali ke satu target."""
    src = random.choice(pool)
    dst = random.choice(pool)
    n = random.randint(12, 25)
    mids = random.sample(pool, k=min(n, len(pool)))
    total = p.amount(scale=random.uniform(6, 14))
    a_out, a_in = p.attrs(), p.attrs()
    rows = []
    for i, mid in enumerate(mids):
        if mid in (src, dst):
            continue
        rows.append(_row(base + timedelta(seconds=i * random.randint(20, 150)),
                         src, mid, total / n * random.uniform(0.97, 1.03),
                         a_out, "Scatter-Gather"))
    later = base + timedelta(minutes=random.randint(20, 180))
    for i, mid in enumerate(mids):
        if mid in (src, dst):
            continue
        rows.append(_row(later + timedelta(seconds=i * random.randint(20, 150)),
                         mid, dst, total / n * random.uniform(0.95, 1.0),
                         a_in, "Scatter-Gather"))
    return rows


def gen_cycle(base: datetime, pool: List[str], p: TypologyProfile) -> List[Dict]:
    """Dana berputar melewati beberapa perantara lalu kembali ke simpul asal."""
    n = random.randint(3, 7)
    chain = random.sample(pool, k=n) + [None]
    chain[-1] = chain[0]
    amount = p.amount(scale=random.uniform(2, 6))
    a = p.attrs()
    rows = []
    for i in range(len(chain) - 1):
        rows.append(_row(base + timedelta(minutes=i * random.randint(2, 12)),
                         chain[i], chain[i + 1], amount * (0.98 ** i),
                         a, "Cycle"))
    return rows


def gen_structuring(base: datetime, pool: List[str], p: TypologyProfile) -> List[Dict]:
    """Dana dipecah menjadi banyak transaksi tepat di bawah ambang pelaporan."""
    src = random.choice(pool)
    n = random.randint(10, 25)
    dsts = random.sample(pool, k=min(6, len(pool)))
    thr = random.choice(REPORTING_THRESHOLDS)
    a = p.attrs()
    rows = []
    for i in range(n):
        dst = random.choice(dsts)
        if dst == src:
            continue
        # Seluruh transaksi menempel pada satu ambang yang sama -- ciri structuring
        rows.append(_row(base + timedelta(minutes=i * random.randint(20, 240)),
                         src, dst, round(thr * random.uniform(0.88, 0.985), 2),
                         a, "Structuring"))
    return rows


def gen_deposit_send(base: datetime, pool: List[str], p: TypologyProfile) -> List[Dict]:
    """Setoran tunai berturut-turut yang langsung diteruskan ke pihak lain."""
    acc = random.choice(pool)
    funder = random.choice(pool)
    dst = random.choice(pool)
    n = random.randint(8, 18)
    rows = []
    total = 0.0
    for i in range(n):
        amt = _near_threshold_amount()
        total += amt
        a = p.attrs()
        a["pay_type"] = "Cash Deposit"
        rows.append(_row(base + timedelta(minutes=i * random.randint(5, 45)),
                         funder, acc, amt, a, "Deposit-Send"))
    rows.append(_row(base + timedelta(hours=random.randint(2, 12)),
                     acc, dst, total * random.uniform(0.9, 0.98),
                     p.attrs(), "Deposit-Send"))
    return rows


def gen_cash_withdrawal(base: datetime, pool: List[str], p: TypologyProfile) -> List[Dict]:
    """Dana masuk lalu ditarik tunai secara bertahap."""
    acc = random.choice(pool)
    src = random.choice(pool)
    cash_out = random.choice(pool)
    amount = p.amount(scale=random.uniform(2, 5))
    rows = [_row(base, src, acc, amount, p.attrs(), "Cash_Withdrawal")]
    n = random.randint(8, 18)
    for i in range(n):
        a = p.attrs()
        a["pay_type"] = "Cash Withdrawal"
        rows.append(_row(base + timedelta(minutes=i * random.randint(15, 180)),
                         acc, cash_out, amount / n * random.uniform(0.8, 1.1),
                         a, "Cash_Withdrawal"))
    return rows


def _layered(base: datetime, pool: List[str], p: TypologyProfile,
             typology: str, inbound: bool) -> List[Dict]:
    """Pola fan bertingkat dua lapis (dipakai Layered_Fan_Out dan Layered_Fan_In)."""
    hub = random.choice(pool)
    n1 = random.randint(6, 12)
    level1 = random.sample(pool, k=n1)
    amount = p.amount(scale=random.uniform(4, 10))
    rows = []
    for i, mid in enumerate(level1):
        if mid == hub:
            continue
        src, dst = (hub, mid) if inbound is False else (mid, hub)
        rows.append(_row(base + timedelta(minutes=i * random.randint(2, 10)),
                         src, dst, amount / n1 * random.uniform(0.85, 1.15),
                         p.attrs(), typology))
        n2 = random.randint(4, 8)
        level2 = random.sample(pool, k=n2)
        later = base + timedelta(hours=random.randint(1, 10))
        for j, leaf in enumerate(level2):
            if leaf == mid:
                continue
            s2, d2 = (mid, leaf) if inbound is False else (leaf, mid)
            rows.append(_row(later + timedelta(minutes=j * random.randint(2, 10)),
                             s2, d2, amount / n1 / n2 * random.uniform(0.8, 1.2),
                             p.attrs(), typology))
    return rows


def gen_layered_fan_out(base, pool, p):
    return _layered(base, pool, p, "Layered_Fan_Out", inbound=False)


def gen_layered_fan_in(base, pool, p):
    return _layered(base, pool, p, "Layered_Fan_In", inbound=True)


def _bipartite(base: datetime, pool: List[str], p: TypologyProfile,
               typology: str, layers: int) -> List[Dict]:
    """Dua (atau lebih) kelompok akun yang saling terhubung padat."""
    rows = []
    group_a = random.sample(pool, k=random.randint(6, 12))
    for layer in range(layers):
        group_b = random.sample(pool, k=random.randint(6, 12))
        amount = p.amount(scale=random.uniform(1, 4))
        offset = base + timedelta(hours=layer * random.randint(2, 12))
        for i, src in enumerate(group_a):
            for j, dst in enumerate(group_b):
                if src == dst or random.random() > 0.8:
                    continue
                rows.append(_row(offset + timedelta(minutes=(i * 5 + j) * random.randint(1, 5)),
                                 src, dst, amount * random.uniform(0.7, 1.3),
                                 p.attrs(), typology))
        group_a = group_b
    return rows


def gen_bipartite(base, pool, p):
    return _bipartite(base, pool, p, "Bipartite", layers=1)


def gen_stacked_bipartite(base, pool, p):
    return _bipartite(base, pool, p, "Stacked Bipartite", layers=random.randint(2, 3))


def _behavioural_change(base: datetime, pool: List[str], p: TypologyProfile,
                        typology: str, burst: bool) -> List[Dict]:
    """
    Akun yang perilakunya berubah mendadak.

    Varian 1 (burst): frekuensi melonjak drastis dalam waktu singkat.
    Varian 2: nominal melonjak jauh di atas kebiasaan akun tersebut.
    """
    acc = random.choice(pool)
    partners = random.sample(pool, k=random.randint(3, 8))
    rows = []
    if burst:
        n = random.randint(25, 50)
        for i in range(n):
            dst = random.choice(partners)
            if dst == acc:
                continue
            rows.append(_row(base + timedelta(seconds=i * random.randint(30, 150)),
                             acc, dst, p.amount(scale=random.uniform(0.5, 1.5)),
                             p.attrs(), typology))
    else:
        n = random.randint(3, 7)
        for i in range(n):
            dst = random.choice(partners)
            if dst == acc:
                continue
            rows.append(_row(base + timedelta(hours=i * random.randint(1, 8)),
                             acc, dst, p.amount(scale=random.uniform(6, 15)),
                             p.attrs(), typology))
    return rows


def gen_behavioural_change_1(base, pool, p):
    return _behavioural_change(base, pool, p, "Behavioural_Change_1", burst=True)


def gen_behavioural_change_2(base, pool, p):
    return _behavioural_change(base, pool, p, "Behavioural_Change_2", burst=False)


def gen_single_large(base: datetime, pool: List[str], p: TypologyProfile) -> List[Dict]:
    """Satu transaksi bernominal sangat besar di luar kebiasaan."""
    src, dst = random.sample(pool, k=2)
    return [_row(base, src, dst, p.amount(scale=random.uniform(8, 25)),
                 p.attrs(), "Single_large")]


def gen_over_invoicing(base: datetime, pool: List[str], p: TypologyProfile) -> List[Dict]:
    """Transaksi lintas negara bernominal berlebihan dengan mata uang berbeda."""
    src, dst = random.sample(pool, k=2)
    a = p.attrs()
    if a["sender_loc"] == a["receiver_loc"]:
        a["receiver_loc"] = random.choice(["UAE", "Switzerland", "Singapore", "Hong Kong"])
    rows = []
    n = random.randint(1, 3)
    for i in range(n):
        rows.append(_row(base + timedelta(days=i * random.randint(1, 4)),
                         src, dst, p.amount(scale=random.uniform(5, 15)),
                         a, "Over-Invoicing"))
    return rows


GENERATORS: Dict[str, Callable] = {
    "Fan_Out": gen_fan_out,
    "Fan_In": gen_fan_in,
    "Gather-Scatter": gen_gather_scatter,
    "Scatter-Gather": gen_scatter_gather,
    "Cycle": gen_cycle,
    "Structuring": gen_structuring,
    "Deposit-Send": gen_deposit_send,
    "Cash_Withdrawal": gen_cash_withdrawal,
    "Layered_Fan_Out": gen_layered_fan_out,
    "Layered_Fan_In": gen_layered_fan_in,
    "Bipartite": gen_bipartite,
    "Stacked Bipartite": gen_stacked_bipartite,
    "Behavioural_Change_1": gen_behavioural_change_1,
    "Behavioural_Change_2": gen_behavioural_change_2,
    "Single_large": gen_single_large,
    "Over-Invoicing": gen_over_invoicing,
}


# ─── Transaksi normal pengimbang ──────────────────────────────────────────────
def generate_normal_rows(n: int, pool: List[str], df: pd.DataFrame,
                         start: datetime, span_days: int) -> pd.DataFrame:
    """
    Bangun transaksi normal secara vektorial dari distribusi empiris data asli.

    Dibutuhkan agar rasio kelas tetap sebanding dengan dataset sebelumnya
    setelah tipologi langka diaugmentasi.
    """
    normal = df[df["Is_laundering"] == 0]
    idx = np.random.randint(0, len(normal), size=n)
    sample = normal.iloc[idx]

    accounts = np.array(pool)
    senders = accounts[np.random.randint(0, len(accounts), size=n)]
    receivers = accounts[np.random.randint(0, len(accounts), size=n)]
    clash = senders == receivers
    receivers[clash] = accounts[(np.searchsorted(accounts, receivers[clash]) + 1) % len(accounts)]

    minutes = np.random.randint(0, span_days * 24 * 60, size=n)
    stamps = pd.to_datetime(start) + pd.to_timedelta(minutes, unit="m")

    amounts = sample["Amount"].values * np.random.uniform(0.85, 1.15, size=n)

    return pd.DataFrame({
        "Time": stamps.strftime("%H:%M:%S"),
        "Date": stamps.strftime("%Y-%m-%d"),
        "Sender_account": senders,
        "Receiver_account": receivers,
        "Amount": np.round(np.maximum(amounts, 1.0), 2),
        "Payment_currency": sample["Payment_currency"].values,
        "Received_currency": sample["Received_currency"].values,
        "Sender_bank_location": sample["Sender_bank_location"].values,
        "Receiver_bank_location": sample["Receiver_bank_location"].values,
        "Payment_type": sample["Payment_type"].values,
        "Is_laundering": 0,
        "Laundering_type": "Normal",
    })


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Augmentasi tipologi langka SAML-D.")
    parser.add_argument("--rows-per-type", type=int, default=4000,
                        help="Target jumlah baris laundering per tipologi (default 4000).")
    parser.add_argument("--keep-ratio", action="store_true", default=True,
                        help="Tambahkan transaksi normal agar rasio kelas tetap ~14%%.")
    parser.add_argument("--pool-size", type=int, default=5000,
                        help="Jumlah akun lama yang terlibat dalam pola laundering (default 5000).")
    parser.add_argument("--mule-accounts", type=int, default=3000,
                        help="Jumlah rekening mule baru yang dibuat (default 3000; 0 untuk menonaktifkan).")
    parser.add_argument("--mule-normal-tx", type=int, default=6,
                        help="Transaksi normal per rekening mule sebagai latar (default 6).")
    args = parser.parse_args()

    print("=" * 70)
    print("  Augmentasi Tipologi Laundering yang Belum Terwakili")
    print("=" * 70 + "\n")

    print(f"[1/5] Memuat {INPUT_CSV} ...")
    df = pd.read_csv(INPUT_CSV)
    df["Sender_account"] = df["Sender_account"].astype(str)
    df["Receiver_account"] = df["Receiver_account"].astype(str)
    print(f"      {len(df):,} baris | {int(df.Is_laundering.sum()):,} laundering "
          f"({df.Is_laundering.mean()*100:.2f}%)")

    dates = pd.to_datetime(df["Date"])
    start, end = dates.min(), dates.max()
    span_days = max(1, (end - start).days)
    print(f"      Rentang waktu: {start.date()} s/d {end.date()} ({span_days} hari)")

    # Kumpulan akun: hanya akun yang sudah memiliki transaksi NORMAL.
    # Inilah yang mencegah model memakai "akun khusus laundering" sebagai jalan pintas.
    print("\n[2/5] Menyiapkan kumpulan akun ...")
    normal_rows = df[df["Is_laundering"] == 0]
    active = pd.concat([normal_rows["Sender_account"], normal_rows["Receiver_account"]])
    counts = active.value_counts()
    # Kumpulan akun sengaja dibatasi menyerupai jaringan money mule: pelaku
    # nyata memakai kumpulan rekening yang sama berulang kali, bukan rekening
    # baru untuk tiap transaksi. Dengan kumpulan terlalu besar (mis. 60.000),
    # jejak struktural tiap pola menjadi terlalu encer untuk dikenali -- percobaan
    # dengan kumpulan sebesar itu hanya menghasilkan recall 25,7%.
    # Seluruh akun tetap memiliki riwayat transaksi normal, sehingga model tidak
    # dapat memakai "akun yang hanya muncul di pola laundering" sebagai pintasan.
    pool = counts[counts >= 3].index.tolist()
    random.shuffle(pool)
    pool = pool[:args.pool_size]
    print(f"      {len(pool):,} akun dipakai (semuanya punya riwayat transaksi normal)")
    involved = counts[pool].sum()
    print(f"      Akun tersebut memiliki {int(involved):,} transaksi normal sebagai latar")

    # Rekening mule: dibuka khusus untuk menampung dan meneruskan dana.
    # Ini fenomena nyata dalam pencucian uang -- rekening baru dengan riwayat
    # tipis yang dipakai beberapa minggu lalu ditinggalkan. Agar tidak menjadi
    # pintasan sempurna bagi model, tiap rekening mule tetap diberi sejumlah
    # kecil transaksi normal sebagai latar belakang.
    mule_normal_rows = []
    if args.mule_accounts > 0:
        mules = [f"88{i:08d}" for i in range(1, args.mule_accounts + 1)]
        print(f"      {len(mules):,} rekening mule baru dibuat "
              f"(+{args.mule_normal_tx} transaksi normal masing-masing)")
        normal_sample = df[df["Is_laundering"] == 0].sample(
            n=len(mules) * args.mule_normal_tx, replace=True, random_state=SEED
        ).reset_index(drop=True)
        for i, mule in enumerate(mules):
            for j in range(args.mule_normal_tx):
                src_row = normal_sample.iloc[i * args.mule_normal_tx + j]
                counterpart = random.choice(pool)
                dt = start.to_pydatetime() + timedelta(
                    minutes=random.randint(0, span_days * 24 * 60))
                sender, receiver = (mule, counterpart) if j % 2 == 0 else (counterpart, mule)
                mule_normal_rows.append({
                    "Time": dt.strftime("%H:%M:%S"), "Date": dt.strftime("%Y-%m-%d"),
                    "Sender_account": sender, "Receiver_account": receiver,
                    "Amount": round(float(src_row["Amount"]), 2),
                    "Payment_currency": src_row["Payment_currency"],
                    "Received_currency": src_row["Received_currency"],
                    "Sender_bank_location": src_row["Sender_bank_location"],
                    "Receiver_bank_location": src_row["Receiver_bank_location"],
                    "Payment_type": src_row["Payment_type"],
                    "Is_laundering": 0, "Laundering_type": "Normal",
                })
        pool = pool + mules
        random.shuffle(pool)
        print(f"      Kumpulan akhir: {len(pool):,} akun "
              f"({len(mules):,} mule + {len(pool)-len(mules):,} akun lama)")

    fallback = df[df["Is_laundering"] == 1]
    profiles = {t: TypologyProfile(df, t, fallback) for t in GENERATORS}

    print(f"\n[3/5] Membangkitkan pola (target {args.rows_per_type:,} baris/tipologi) ...")
    new_rows: List[Dict] = []
    report: Dict[str, Dict] = {}

    for typology, generator in GENERATORS.items():
        profile = profiles[typology]
        rows_for_type: List[Dict] = []
        instances = 0
        guard = 0
        while len(rows_for_type) < args.rows_per_type and guard < args.rows_per_type * 5:
            guard += 1
            offset_minutes = random.randint(0, span_days * 24 * 60)
            base = start.to_pydatetime() + timedelta(minutes=offset_minutes)
            produced = generator(base, pool, profile)
            if produced:
                rows_for_type.extend(produced)
                instances += 1
        rows_for_type = rows_for_type[:args.rows_per_type]
        new_rows.extend(rows_for_type)
        report[typology] = {
            "sebelum": profile.n_real,
            "ditambahkan": len(rows_for_type),
            "sesudah": profile.n_real + len(rows_for_type),
            "instance": instances,
        }
        print(f"      {typology:<22} {profile.n_real:>6,} -> {report[typology]['sesudah']:>7,} "
              f"({instances:,} instance pola)")

    df_new = pd.DataFrame(new_rows, columns=COLUMNS)
    print(f"\n      Total baris laundering baru: {len(df_new):,}")

    # Transaksi normal pengimbang
    if args.keep_ratio:
        current_ratio = df["Is_laundering"].mean()
        total_pos = int(df.Is_laundering.sum()) + len(df_new)
        target_total = total_pos / current_ratio
        n_normal = int(max(0, target_total - len(df) - len(df_new) - len(mule_normal_rows)))
        print(f"\n[4/5] Menambah {n_normal:,} transaksi normal agar rasio tetap "
              f"{current_ratio*100:.2f}% ...")
        df_normal = generate_normal_rows(n_normal, pool, df, start, span_days)
    else:
        print("\n[4/5] Melewati penambahan transaksi normal.")
        df_normal = pd.DataFrame(columns=COLUMNS)

    # Gabung + urutkan kronologis
    print("\n[5/5] Menggabungkan dan membagi ulang secara temporal ...")
    df_mule_normal = pd.DataFrame(mule_normal_rows, columns=COLUMNS) if mule_normal_rows \
        else pd.DataFrame(columns=COLUMNS)
    combined = pd.concat([df, df_new, df_mule_normal, df_normal], ignore_index=True)
    combined["_dt"] = pd.to_datetime(
        combined["Date"].astype(str) + " " + combined["Time"].astype(str),
        format="%Y-%m-%d %H:%M:%S", errors="coerce",
    )
    combined = combined.dropna(subset=["_dt"]).sort_values("_dt", kind="mergesort").reset_index(drop=True)
    combined = combined.drop(columns=["_dt"])[COLUMNS]

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
        path = resolve(f"{name}_v3.csv")
        part.to_csv(path, index=False)

    print(f"\n{'':6}{'split':<8}{'baris':>12}{'laundering':>14}{'rasio':>9}")
    split_report = {}
    for name, part in splits.items():
        pos = int(part.Is_laundering.sum())
        split_report[name] = {"baris": len(part), "laundering": pos,
                              "rasio_persen": round(pos / len(part) * 100, 2)}
        print(f"{'':6}{name:<8}{len(part):>12,}{pos:>14,}{pos/len(part)*100:>8.2f}%")

    total_pos = int(combined.Is_laundering.sum())
    print(f"\n      TOTAL: {n:,} baris | {total_pos:,} laundering "
          f"({total_pos/n*100:.2f}%) | rasio {(n-total_pos)/total_pos:.2f} : 1")

    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "sumber": os.path.basename(INPUT_CSV),
            "keluaran": os.path.basename(OUTPUT_CSV),
            "rows_per_type": args.rows_per_type,
            "per_tipologi": report,
            "normal_ditambahkan": len(df_normal),
            "rekening_mule": args.mule_accounts,
            "normal_latar_mule": len(mule_normal_rows),
            "split": split_report,
            "total_baris": n,
            "total_laundering": total_pos,
            "rasio_laundering_persen": round(total_pos / n * 100, 2),
            "seed": SEED,
        }, f, indent=2)

    print(f"\n      Dataset : {OUTPUT_CSV}")
    print(f"      Split   : data/train_v3.csv, val_v3.csv, test_v3.csv")
    print(f"      Laporan : {REPORT_JSON}")
    print("\nSelesai. Lanjut: python -m backend.ml.data_pipeline --version v3\n")


if __name__ == "__main__":
    main()
