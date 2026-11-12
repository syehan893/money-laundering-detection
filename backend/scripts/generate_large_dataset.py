"""
generate_large_dataset.py — Generate ~500K+ Transactions for Training
=====================================================================
Takes the original SAML-D.csv (~152K rows) and generates additional
synthetic transactions to reach the target scale:

    Target: 500,000 training + 500 test (plus validation buffer)
    Total generated: ~600,000 rows

Usage:
    python -m backend.scripts.generate_large_dataset

Output:
    data/SAML-D_large.csv
"""

import os
import random
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from backend.config import CSV_PATH, DATA_DIR, resolve

random.seed(42)
np.random.seed(42)

INPUT_CSV = CSV_PATH
OUTPUT_CSV = resolve("SAML-D_large.csv")
BACKUP_CSV = resolve("SAML-D_original_backup.csv")

# ─── Constants ────────────────────────────────────────────────────────────────
TARGET_TOTAL = 1_100_000  # total rows to generate (500K train + 500K test + 100K val buffer)

SYNTH_ACCOUNTS = [f"99{i:08d}" for i in range(1, 2001)]  # 2,000 synthetic accounts

HIGH_RISK_LOCATIONS = ["UAE", "Cayman Islands", "Switzerland", "Singapore", "Hong Kong"]
LOW_RISK_LOCATIONS = ["UK", "US", "Germany", "France", "Japan", "Australia", "Canada"]
ALL_LOCATIONS = HIGH_RISK_LOCATIONS + LOW_RISK_LOCATIONS

CURRENCIES = {
    "UK": "UK pounds", "US": "US Dollar", "UAE": "Dirham",
    "Cayman Islands": "US Dollar", "Switzerland": "Swiss Franc",
    "Singapore": "Singapore Dollar", "Hong Kong": "HK Dollar",
    "Germany": "Euro", "France": "Euro", "Japan": "Yen",
    "Australia": "Australian Dollar", "Canada": "Canadian Dollar",
}

PAYMENT_TYPES = ["Cheque", "Credit Card", "Wire", "Cash", "ACH", "Cross-border"]

# ─── Helpers ──────────────────────────────────────────────────────────────────

def random_time(base_date: datetime, delta_minutes: int = 0) -> tuple:
    dt = base_date + timedelta(minutes=delta_minutes)
    return dt.strftime("%H:%M:%S"), dt.strftime("%Y-%m-%d")


def _make_row(time_str, date_str, sender, receiver, amount, pay_curr, recv_curr,
              sender_loc, receiver_loc, pay_type, is_laundering, laund_type):
    return {
        "Time": time_str, "Date": date_str,
        "Sender_account": sender, "Receiver_account": receiver,
        "Amount": round(amount, 2), "Payment_currency": pay_curr,
        "Received_currency": recv_curr, "Sender_bank_location": sender_loc,
        "Receiver_bank_location": receiver_loc, "Payment_type": pay_type,
        "Is_laundering": is_laundering, "Laundering_type": laund_type,
    }


def sample_from_distribution(series: pd.Series, n: int) -> np.ndarray:
    """Sample n values from the empirical distribution of a pandas Series."""
    return np.random.choice(series.dropna().values, size=n, replace=True)


# ─── Normal Transaction Generators ────────────────────────────────────────────

def generate_normal_batch(base_date: datetime, normal_accounts: list,
                          amount_sampler, paytype_sampler,
                          loc_sampler, curr_sampler,
                          batch_size: int = 1000) -> list:
    """Generate a batch of normal (non-laundering) transactions."""
    rows = []
    for _ in range(batch_size):
        sender, receiver = random.sample(normal_accounts, k=2)
        amount = float(np.random.choice(amount_sampler))
        # Add small noise to amount
        amount = amount * random.uniform(0.9, 1.1)
        amount = max(1.0, amount)

        pay_type = random.choice(PAYMENT_TYPES)
        sender_loc = random.choice(ALL_LOCATIONS)
        receiver_loc = random.choice(ALL_LOCATIONS)
        pay_curr = CURRENCIES[sender_loc]
        recv_curr = CURRENCIES[receiver_loc]

        delta = random.randint(0, 1440)  # up to 1 day apart
        time_str, date_str = random_time(base_date, delta_minutes=delta)

        rows.append(_make_row(
            time_str, date_str, str(sender), str(receiver), round(amount, 2),
            pay_curr, recv_curr, sender_loc, receiver_loc, pay_type,
            0, "Normal",
        ))
    return rows


# ─── Laundering Pattern Generators ────────────────────────────────────────────

def generate_smurfing(base_date: datetime, account_pool: list, scale: int = 1) -> list:
    rows = []
    sender = random.choice(account_pool[:500])
    num_txs = random.randint(10, 25) * scale
    receivers = random.sample(account_pool[500:], k=min(num_txs, len(account_pool) - 500))
    sender_loc = random.choice(ALL_LOCATIONS)
    for i, recv in enumerate(receivers):
        recv_loc = random.choice(ALL_LOCATIONS)
        amount = round(random.uniform(500, 9500), 2)
        time_str, date_str = random_time(base_date, delta_minutes=i * random.randint(1, 5))
        rows.append(_make_row(
            time_str, date_str, sender, recv, amount,
            CURRENCIES[sender_loc], CURRENCIES[recv_loc],
            sender_loc, recv_loc, "Wire", 1, "Smurfing",
        ))
    return rows


def generate_round_tripping(base_date: datetime, account_pool: list, scale: int = 1) -> list:
    rows = []
    chain_length = random.randint(3, 7)
    accounts = random.sample(account_pool, k=chain_length)
    accounts.append(accounts[0])
    amount = round(random.uniform(20000, 120000), 2)
    locations = [random.choice(ALL_LOCATIONS) for _ in accounts]
    for i in range(len(accounts) - 1):
        tx_amount = round(amount * random.uniform(0.90, 0.98), 2)
        time_str, date_str = random_time(base_date, delta_minutes=i * random.randint(5, 30))
        rows.append(_make_row(
            time_str, date_str, accounts[i], accounts[i + 1], tx_amount,
            CURRENCIES[locations[i]], CURRENCIES[locations[i + 1]],
            locations[i], locations[i + 1], "Cross-border", 1, "Round_Tripping",
        ))
    return rows


def generate_cross_border_layering(base_date: datetime, account_pool: list, scale: int = 1) -> list:
    rows = []
    chain_length = random.randint(4, 10)
    accounts = random.sample(account_pool, k=chain_length)
    locations = []
    for _ in range(chain_length):
        locations.append(random.choice(HIGH_RISK_LOCATIONS) if random.random() < 0.7 else random.choice(LOW_RISK_LOCATIONS))
    amount = round(random.uniform(30000, 250000), 2)
    for i in range(len(accounts) - 1):
        tx_amount = round(amount * (0.96 ** i), 2)
        time_str, date_str = random_time(base_date, delta_minutes=i * random.randint(10, 60))
        rows.append(_make_row(
            time_str, date_str, accounts[i], accounts[i + 1], tx_amount,
            CURRENCIES[locations[i]], CURRENCIES[locations[i + 1]],
            locations[i], locations[i + 1], "Cross-border", 1, "Cross_Border_Layering",
        ))
    return rows


def generate_fan_out_fan_in(base_date: datetime, account_pool: list, scale: int = 1) -> list:
    rows = []
    hub_sender = random.choice(account_pool[:200])
    hub_receiver = random.choice(account_pool[200:400])
    num_intermediaries = random.randint(6, 15)
    intermediaries = random.sample(account_pool[400:], k=num_intermediaries)
    total_amount = round(random.uniform(60000, 300000), 2)
    per_amount = total_amount / len(intermediaries)
    hub_loc = random.choice(LOW_RISK_LOCATIONS)
    for i, mid in enumerate(intermediaries):
        mid_loc = random.choice(ALL_LOCATIONS)
        time_str, date_str = random_time(base_date, delta_minutes=i * random.randint(1, 5))
        rows.append(_make_row(
            time_str, date_str, hub_sender, mid,
            round(per_amount * random.uniform(0.85, 1.15), 2),
            CURRENCIES[hub_loc], CURRENCIES[mid_loc],
            hub_loc, mid_loc, "Wire", 1, "Fan_Out_Fan_In",
        ))
    collect_loc = random.choice(HIGH_RISK_LOCATIONS)
    for i, mid in enumerate(intermediaries):
        mid_loc = random.choice(ALL_LOCATIONS)
        time_str, date_str = random_time(
            base_date + timedelta(hours=random.randint(2, 12)),
            delta_minutes=i * random.randint(2, 8),
        )
        rows.append(_make_row(
            time_str, date_str, mid, hub_receiver,
            round(per_amount * random.uniform(0.80, 0.95), 2),
            CURRENCIES[mid_loc], CURRENCIES[collect_loc],
            mid_loc, collect_loc, "Cross-border", 1, "Fan_Out_Fan_In",
        ))
    return rows


def generate_rapid_burst(base_date: datetime, account_pool: list, scale: int = 1) -> list:
    rows = []
    sender = random.choice(account_pool[:300])
    num_txs = random.randint(8, 20) * scale
    receivers = random.sample(account_pool[300:], k=min(num_txs, len(account_pool) - 300))
    sender_loc = random.choice(HIGH_RISK_LOCATIONS)
    for i, recv in enumerate(receivers):
        recv_loc = random.choice(ALL_LOCATIONS)
        amount = round(random.uniform(15000, 80000), 2)
        time_str, date_str = random_time(base_date, delta_minutes=i * random.randint(1, 3))
        rows.append(_make_row(
            time_str, date_str, sender, recv, amount,
            CURRENCIES[sender_loc], CURRENCIES[recv_loc],
            sender_loc, recv_loc, "Wire", 1, "Rapid_Burst",
        ))
    return rows


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("  AML Large Dataset Generator")
    print("  Target: ~600,000 total transactions")
    print("=" * 62 + "\n")

    # Load original
    print("[1/5] Loading original CSV ...")
    if os.path.exists(BACKUP_CSV):
        df_orig = pd.read_csv(BACKUP_CSV)
    else:
        df_orig = pd.read_csv(INPUT_CSV)
    print(f"  Original rows: {len(df_orig):,}")

    # Backup if not exists
    if not os.path.exists(BACKUP_CSV):
        df_orig.to_csv(BACKUP_CSV, index=False)
        print("  ✓ Backup created")

    # Parse datetime
    df_orig["_dt"] = pd.to_datetime(df_orig["Date"].astype(str) + " " + df_orig["Time"].astype(str), errors="coerce")
    date_min = df_orig["_dt"].min()
    date_max = df_orig["_dt"].max()
    total_days = max((date_max - date_min).days, 1)
    df_orig.drop(columns=["_dt"], inplace=True)
    print(f"  Date range: {date_min.date()} → {date_max.date()} ({total_days} days)")

    # Combine original + synthetic accounts
    orig_accounts = pd.concat([df_orig["Sender_account"], df_orig["Receiver_account"]]).unique().tolist()
    all_accounts = orig_accounts + SYNTH_ACCOUNTS

    # Samplers from original distribution
    normal_amounts = df_orig[df_orig["Is_laundering"] == 0]["Amount"]
    normal_paytypes = df_orig[df_orig["Is_laundering"] == 0]["Payment_type"].dropna().unique().tolist()

    # ── Generate base dates spread across full range ──
    base_dates = [date_min + timedelta(days=random.randint(0, total_days), hours=random.randint(0, 23))
                  for _ in range(2000)]

    # ── Generate normal transactions ──
    needed_normal = TARGET_TOTAL - len(df_orig) - 50_000  # reserve ~50K for laundering
    print(f"\n[2/5] Generating ~{needed_normal:,} normal transactions ...")
    normal_rows = []
    batch_size = 5_000
    generated = 0
    while generated < needed_normal:
        base = random.choice(base_dates)
        batch = generate_normal_batch(
            base, all_accounts, normal_amounts.values, normal_paytypes,
            ALL_LOCATIONS, CURRENCIES, batch_size=min(batch_size, needed_normal - generated)
        )
        normal_rows.extend(batch)
        generated += len(batch)
        if generated % 50_000 == 0:
            print(f"  ... {generated:,} normal generated")
    print(f"  ✓ Normal transactions: {len(normal_rows):,}")

    # ── Generate laundering transactions ──
    print(f"\n[3/5] Generating laundering patterns ...")
    laundering_rows = []

    n_smurfing = 4_000
    n_round_trip = 3_000
    n_layering = 3_000
    n_fan_out = 2_000
    n_rapid_burst = 2_000

    for i in range(n_smurfing):
        laundering_rows.extend(generate_smurfing(random.choice(base_dates), SYNTH_ACCOUNTS))
    print(f"  ✓ Smurfing: {sum(1 for r in laundering_rows if r['Laundering_type']=='Smurfing'):,}")

    for i in range(n_round_trip):
        laundering_rows.extend(generate_round_tripping(random.choice(base_dates), SYNTH_ACCOUNTS))
    print(f"  ✓ Round-tripping: {sum(1 for r in laundering_rows if r['Laundering_type']=='Round_Tripping'):,}")

    for i in range(n_layering):
        laundering_rows.extend(generate_cross_border_layering(random.choice(base_dates), SYNTH_ACCOUNTS))
    print(f"  ✓ Cross-border layering: {sum(1 for r in laundering_rows if r['Laundering_type']=='Cross_Border_Layering'):,}")

    for i in range(n_fan_out):
        laundering_rows.extend(generate_fan_out_fan_in(random.choice(base_dates), SYNTH_ACCOUNTS))
    print(f"  ✓ Fan-out/fan-in: {sum(1 for r in laundering_rows if r['Laundering_type']=='Fan_Out_Fan_In'):,}")

    for i in range(n_rapid_burst):
        laundering_rows.extend(generate_rapid_burst(random.choice(base_dates), SYNTH_ACCOUNTS))
    print(f"  ✓ Rapid burst: {sum(1 for r in laundering_rows if r['Laundering_type']=='Rapid_Burst'):,}")

    print(f"  ✓ Total laundering: {len(laundering_rows):,}")

    # ── Merge everything ──
    print(f"\n[4/5] Merging datasets ...")
    df_normal = pd.DataFrame(normal_rows)
    df_laundering = pd.DataFrame(laundering_rows)
    df_combined = pd.concat([df_orig, df_normal, df_laundering], ignore_index=True)

    # Sort temporally
    df_combined["_sort_dt"] = pd.to_datetime(
        df_combined["Date"].astype(str) + " " + df_combined["Time"].astype(str), errors="coerce"
    )
    df_combined = df_combined.sort_values("_sort_dt").reset_index(drop=True)
    df_combined.drop(columns=["_sort_dt"], inplace=True)

    # Stats
    total = len(df_combined)
    pos = int(df_combined["Is_laundering"].sum())
    neg = total - pos
    print(f"  Total rows : {total:,}")
    print(f"  Normal     : {neg:,}  ({neg/total*100:.2f}%)")
    print(f"  Laundering : {pos:,}  ({pos/total*100:.2f}%)")
    print(f"  Imbalance  : {neg/max(pos,1):.1f}:1")

    # Temporal split preview
    df_combined["_dt"] = pd.to_datetime(
        df_combined["Date"].astype(str) + " " + df_combined["Time"].astype(str), errors="coerce"
    )
    train_end = 500_000
    test_end = train_end + 500_000
    if total >= train_end + 500_000:
        train_pos = df_combined.iloc[:train_end]["Is_laundering"].sum()
        test_pos = df_combined.iloc[train_end:test_end]["Is_laundering"].sum()
        print(f"\n  -- Split preview (500K train / 500K test) --")
        print(f"  Train laundering: {train_pos:,}")
        print(f"  Test  laundering: {test_pos:,}")
    df_combined.drop(columns=["_dt"], inplace=True)

    # ── Split into train / val / test files ──
    print(f"\n[5/5] Splitting into train / val / test files ...")

    TRAIN_SPLIT = 500_000
    TEST_SPLIT = 500_000
    VAL_SPLIT = total - TRAIN_SPLIT - TEST_SPLIT

    df_train = df_combined.iloc[:TRAIN_SPLIT].copy()
    df_val = df_combined.iloc[TRAIN_SPLIT:TRAIN_SPLIT + VAL_SPLIT].copy()
    df_test = df_combined.iloc[TRAIN_SPLIT + VAL_SPLIT:].copy()

    # Save split files
    train_path = resolve("train.csv")
    val_path = resolve("val.csv")
    test_path = resolve("test.csv")
    full_path = resolve("SAML-D_large.csv")

    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)
    df_combined.to_csv(full_path, index=False)

    print(f"  ✓ train.csv : {len(df_train):,} rows ({train_path})")
    print(f"  ✓ val.csv   : {len(df_val):,} rows ({val_path})")
    print(f"  ✓ test.csv  : {len(df_test):,} rows ({test_path})")
    print(f"  ✓ SAML-D_large.csv (full): {len(df_combined):,} rows ({full_path})")

    # Print split stats
    for name, df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        pos = int(df["Is_laundering"].sum())
        neg = len(df) - pos
        print(f"  {name:>6}: {len(df):>8,} rows | Normal: {neg:>8,} | Laundering: {pos:>6,}")

    print("\n" + "=" * 62)
    print("  NEXT STEPS:")
    print("  1. python -m backend.ml.legacy.data_pipeline")
    print("     (pipeline otomatis membaca train.csv + val.csv + test.csv)")
    print("  2. python -m backend.ml.legacy.train")
    print("  3. python -m backend.main")
    print("=" * 62)
    print("\n✅ Large dataset generation complete!\n")


if __name__ == "__main__":
    main()
