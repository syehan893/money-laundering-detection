"""
augment_data.py — Inject Synthetic Laundering Patterns
=======================================================
Adds synthetic money laundering transactions to SAML-D.csv with
distinctive AML patterns that the GAT model should classify as
High and Critical risk.

KEY DESIGN DECISIONS:
  - Synthetic data placed across the FULL date range so temporal split
    includes laundering in train/val/test proportionally.
  - Volume increased to ~5% positive class (from ~0.5%) so model can
    learn meaningful patterns despite class imbalance.
  - Patterns are STRONGLY differentiated from normal transactions:
    high amounts, cross-border, rapid bursts, currency mixing.

Patterns injected:
  1. Smurfing (structuring) — rapid small transactions below threshold
  2. Round-tripping — money circles back to origin through intermediaries
  3. Cross-border layering — multi-hop international transfers
  4. Fan-out / Fan-in — one account distributing to many, then re-collecting
  5. Rapid burst — many large transfers in very short time window

Usage:
    python -m backend.scripts.augment_data
"""

import os
import random
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from backend.config import CSV_PATH, DATA_DIR

INPUT_CSV = CSV_PATH
OUTPUT_CSV = CSV_PATH  # Overwrite original
BACKUP_CSV = os.path.join(DATA_DIR, "SAML-D_original_backup.csv")

# Seed for reproducibility
random.seed(42)
np.random.seed(42)

# ─── Synthetic Account IDs (clearly identifiable) ───────────────────────────
SYNTH_ACCOUNTS = [f"99{i:08d}" for i in range(1, 501)]  # 500 synthetic accounts

# ─── Currencies & Locations ──────────────────────────────────────────────────
HIGH_RISK_LOCATIONS = ["UAE", "Cayman Islands", "Switzerland", "Singapore", "Hong Kong"]
LOW_RISK_LOCATIONS = ["UK", "US", "Germany", "France", "Japan"]
ALL_LOCATIONS = HIGH_RISK_LOCATIONS + LOW_RISK_LOCATIONS

CURRENCIES = {
    "UK": "UK pounds",
    "US": "US Dollar",
    "UAE": "Dirham",
    "Cayman Islands": "US Dollar",
    "Switzerland": "Swiss Franc",
    "Singapore": "Singapore Dollar",
    "Hong Kong": "HK Dollar",
    "Germany": "Euro",
    "France": "Euro",
    "Japan": "Yen",
}


def random_time(base_date: datetime, delta_minutes: int = 0) -> tuple:
    """Generate a time string and date string."""
    dt = base_date + timedelta(minutes=delta_minutes)
    return dt.strftime("%H:%M:%S"), dt.strftime("%Y-%m-%d")


def _make_row(time_str, date_str, sender, receiver, amount, pay_curr, recv_curr,
              sender_loc, receiver_loc, pay_type, laund_type):
    """Helper to create a transaction row dict."""
    return {
        "Time": time_str, "Date": date_str,
        "Sender_account": sender, "Receiver_account": receiver,
        "Amount": round(amount, 2), "Payment_currency": pay_curr,
        "Received_currency": recv_curr, "Sender_bank_location": sender_loc,
        "Receiver_bank_location": receiver_loc, "Payment_type": pay_type,
        "Is_laundering": 1, "Laundering_type": laund_type,
    }


# ─── Pattern 1: Smurfing ─────────────────────────────────────────────────────
def generate_smurfing_pattern(base_date: datetime, account_pool: list) -> list:
    """Rapid small transactions below reporting threshold from one sender."""
    rows = []
    sender = random.choice(account_pool[:100])
    num_txs = random.randint(10, 25)  # Many small transactions
    receivers = random.sample(account_pool[100:400], k=num_txs)
    sender_loc = random.choice(ALL_LOCATIONS)

    for i, recv in enumerate(receivers):
        recv_loc = random.choice(ALL_LOCATIONS)
        # Small amounts below reporting threshold (typically $10K)
        amount = round(random.uniform(500, 9500), 2)
        # Very rapid — within minutes of each other
        time_str, date_str = random_time(base_date, delta_minutes=i * random.randint(1, 5))

        rows.append(_make_row(
            time_str, date_str, sender, recv, amount,
            CURRENCIES[sender_loc], CURRENCIES[recv_loc],
            sender_loc, recv_loc, "Wire", "Smurfing",
        ))
    return rows


# ─── Pattern 2: Round-tripping ───────────────────────────────────────────────
def generate_round_tripping(base_date: datetime, account_pool: list) -> list:
    """Money circles back to origin through intermediaries."""
    rows = []
    chain_length = random.randint(3, 7)
    accounts = random.sample(account_pool, k=chain_length)
    accounts.append(accounts[0])  # Close the cycle

    amount = round(random.uniform(20000, 120000), 2)
    locations = [random.choice(ALL_LOCATIONS) for _ in accounts]

    for i in range(len(accounts) - 1):
        sender, receiver = accounts[i], accounts[i + 1]
        sloc, rloc = locations[i], locations[i + 1]
        # Slight decrease per hop (fees)
        tx_amount = round(amount * random.uniform(0.90, 0.98), 2)
        time_str, date_str = random_time(base_date, delta_minutes=i * random.randint(5, 30))

        rows.append(_make_row(
            time_str, date_str, sender, receiver, tx_amount,
            CURRENCIES[sloc], CURRENCIES[rloc],
            sloc, rloc, "Cross-border", "Round_Tripping",
        ))
    return rows


# ─── Pattern 3: Cross-border Layering ────────────────────────────────────────
def generate_cross_border_layering(base_date: datetime, account_pool: list) -> list:
    """Multi-hop international transfers through many jurisdictions."""
    rows = []
    chain_length = random.randint(4, 10)
    accounts = random.sample(account_pool, k=chain_length)

    # Strongly prefer high-risk locations
    locations = []
    for _ in range(chain_length):
        if random.random() < 0.7:
            locations.append(random.choice(HIGH_RISK_LOCATIONS))
        else:
            locations.append(random.choice(LOW_RISK_LOCATIONS))

    amount = round(random.uniform(30000, 250000), 2)

    for i in range(len(accounts) - 1):
        sender, receiver = accounts[i], accounts[i + 1]
        sloc, rloc = locations[i], locations[i + 1]
        # Slight decrease per layer
        tx_amount = round(amount * (0.96 ** i), 2)
        time_str, date_str = random_time(base_date, delta_minutes=i * random.randint(10, 60))

        rows.append(_make_row(
            time_str, date_str, sender, receiver, tx_amount,
            CURRENCIES[sloc], CURRENCIES[rloc],
            sloc, rloc, "Cross-border", "Cross_Border_Layering",
        ))
    return rows


# ─── Pattern 4: Fan-out / Fan-in ─────────────────────────────────────────────
def generate_fan_out_fan_in(base_date: datetime, account_pool: list) -> list:
    """Hub distributes to many intermediaries, who re-collect to a single account."""
    rows = []
    hub_sender = random.choice(account_pool[:50])
    hub_receiver = random.choice(account_pool[50:100])
    num_intermediaries = random.randint(6, 15)
    intermediaries = random.sample(account_pool[100:], k=num_intermediaries)

    total_amount = round(random.uniform(60000, 300000), 2)
    per_amount = total_amount / len(intermediaries)

    hub_loc = random.choice(LOW_RISK_LOCATIONS)

    # Fan-out: hub → intermediaries
    for i, mid in enumerate(intermediaries):
        mid_loc = random.choice(ALL_LOCATIONS)
        time_str, date_str = random_time(base_date, delta_minutes=i * random.randint(1, 5))

        rows.append(_make_row(
            time_str, date_str, hub_sender, mid,
            round(per_amount * random.uniform(0.85, 1.15), 2),
            CURRENCIES[hub_loc], CURRENCIES[mid_loc],
            hub_loc, mid_loc, "Wire", "Fan_Out_Fan_In",
        ))

    # Fan-in: intermediaries → collection account (hours later)
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
            mid_loc, collect_loc, "Cross-border", "Fan_Out_Fan_In",
        ))

    return rows


# ─── Pattern 5: Rapid Burst ──────────────────────────────────────────────────
def generate_rapid_burst(base_date: datetime, account_pool: list) -> list:
    """Many large transfers in very short window — unusual velocity."""
    rows = []
    sender = random.choice(account_pool[:100])
    num_txs = random.randint(8, 20)
    receivers = random.sample(account_pool[100:], k=num_txs)
    sender_loc = random.choice(HIGH_RISK_LOCATIONS)

    for i, recv in enumerate(receivers):
        recv_loc = random.choice(ALL_LOCATIONS)
        # Large amounts
        amount = round(random.uniform(15000, 80000), 2)
        # All within a very short burst (1-2 minutes apart)
        time_str, date_str = random_time(base_date, delta_minutes=i * random.randint(1, 3))

        rows.append(_make_row(
            time_str, date_str, sender, recv, amount,
            CURRENCIES[sender_loc], CURRENCIES[recv_loc],
            sender_loc, recv_loc, "Wire", "Rapid_Burst",
        ))
    return rows


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 58)
    print("  AML Data Augmentation - Synthetic Laundering Patterns")
    print("=" * 58 + "\n")

    # Load original (or backup)
    print("[1/4] Loading CSV ...")
    if os.path.exists(BACKUP_CSV):
        print(f"  Loading from BACKUP: {BACKUP_CSV}")
        df = pd.read_csv(BACKUP_CSV)
    else:
        df = pd.read_csv(INPUT_CSV)

    original_shape = df.shape
    original_pos = int(df["Is_laundering"].sum())
    original_neg = len(df) - original_pos
    print(f"  Original: {original_shape[0]:,} rows | Class 1: {original_pos:,} | Class 0: {original_neg:,}")

    # Backup
    if not os.path.exists(BACKUP_CSV):
        print(f"\n[2/4] Creating backup → {BACKUP_CSV} ...")
        df.to_csv(BACKUP_CSV, index=False)
        print("  ✓ Backup created")
    else:
        print(f"\n[2/4] Backup already exists, skipping.")

    # ── Determine date range from actual data ──
    df["_dt"] = pd.to_datetime(df["Date"] + " " + df["Time"], errors="coerce")
    date_min = df["_dt"].min()
    date_max = df["_dt"].max()
    total_days = (date_max - date_min).days
    df.drop(columns=["_dt"], inplace=True)

    print(f"\n  Date range: {date_min} -> {date_max} ({total_days} days)")

    # Generate base dates SPREAD across the entire range
    # Ensures temporal split gets data in train/val/test
    base_dates = [date_min + timedelta(days=random.randint(0, total_days),
                                        hours=random.randint(0, 23))
                  for _ in range(500)]

    # ── Generate synthetic patterns ──
    print("\n[3/4] Generating synthetic laundering patterns ...")
    synthetic_rows = []

    # Increased volume for better learning
    n_smurfing = 300
    n_round_trip = 200
    n_layering = 200
    n_fan_out = 150
    n_rapid_burst = 150

    for i in range(n_smurfing):
        base = random.choice(base_dates)
        synthetic_rows.extend(generate_smurfing_pattern(base, SYNTH_ACCOUNTS))

    for i in range(n_round_trip):
        base = random.choice(base_dates)
        synthetic_rows.extend(generate_round_tripping(base, SYNTH_ACCOUNTS))

    for i in range(n_layering):
        base = random.choice(base_dates)
        synthetic_rows.extend(generate_cross_border_layering(base, SYNTH_ACCOUNTS))

    for i in range(n_fan_out):
        base = random.choice(base_dates)
        synthetic_rows.extend(generate_fan_out_fan_in(base, SYNTH_ACCOUNTS))

    for i in range(n_rapid_burst):
        base = random.choice(base_dates)
        synthetic_rows.extend(generate_rapid_burst(base, SYNTH_ACCOUNTS))

    synth_df = pd.DataFrame(synthetic_rows)

    # Print stats per pattern
    for lt in ["Smurfing", "Round_Tripping", "Cross_Border_Layering", "Fan_Out_Fan_In", "Rapid_Burst"]:
        count = sum(1 for r in synthetic_rows if r["Laundering_type"] == lt)
        print(f"  [done] {lt:<25}: {count:>6,} txs")

    print(f"  [done] Total synthetic        : {len(synth_df):>6,} transactions (all Is_laundering=1)")

    # ── Merge ──
    print("\n[4/4] Merging with original dataset ...")
    df_augmented = pd.concat([df, synth_df], ignore_index=True)

    # Sort by datetime to maintain temporal order (important for temporal split!)
    df_augmented["_sort_dt"] = pd.to_datetime(
        df_augmented["Date"] + " " + df_augmented["Time"], errors="coerce"
    )
    df_augmented = df_augmented.sort_values("_sort_dt").reset_index(drop=True)
    df_augmented.drop(columns=["_sort_dt"], inplace=True)

    new_pos = int(df_augmented["Is_laundering"].sum())
    new_neg = len(df_augmented) - new_pos
    print(f"  Augmented: {len(df_augmented):,} rows")
    print(f"  Class 0 (Normal)    : {new_neg:,}  ({new_neg/len(df_augmented)*100:.2f}%)")
    print(f"  Class 1 (Laundering): {new_pos:,}  ({new_pos/len(df_augmented)*100:.2f}%)")
    print(f"  Added laundering    : +{new_pos - original_pos:,} transactions")
    print(f"  New imbalance ratio : {new_neg/max(new_pos,1):.1f}:1")

    # Verify temporal distribution
    df_augmented["_dt"] = pd.to_datetime(
        df_augmented["Date"] + " " + df_augmented["Time"], errors="coerce"
    )
    n = len(df_augmented)
    t70 = int(0.7 * n)
    t85 = int(0.85 * n)
    train_pos = df_augmented.iloc[:t70]["Is_laundering"].sum()
    val_pos = df_augmented.iloc[t70:t85]["Is_laundering"].sum()
    test_pos = df_augmented.iloc[t85:]["Is_laundering"].sum()
    df_augmented.drop(columns=["_dt"], inplace=True)

    print(f"\n  -- Temporal split preview --")
    print(f"  Train laundering: {train_pos:,}")
    print(f"  Val   laundering: {val_pos:,}")
    print(f"  Test  laundering: {test_pos:,}")

    # Save
    df_augmented.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  [done] Saved augmented CSV to: {OUTPUT_CSV}")

    print("\n" + "=" * 58)
    print("  NEXT STEPS:")
    print("  1. python -m backend.ml.data_pipeline   (rebuild graph)")
    print("  2. python -m backend.ml.train           (retrain model)")
    print("=" * 58)
    print("\n[done] Data augmentation complete!\n")


if __name__ == "__main__":
    main()
