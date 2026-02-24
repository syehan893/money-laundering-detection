"""
augment_data.py — Inject Synthetic Laundering Patterns
=======================================================
Adds synthetic money laundering transactions to SAML-D.csv with
distinctive AML patterns that the GAT model should classify as
High and Critical risk.

Patterns injected:
  1. Smurfing (structuring) — rapid small transactions below threshold
  2. Round-tripping — money circles back to origin through intermediaries
  3. Cross-border layering — multi-hop international transfers
  4. Fan-out / Fan-in — one account distributing to many, then re-collecting

Usage:
    python augment_data.py
    python data_pipeline.py     # Re-build the graph
    python train.py             # Re-train the model
"""

import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(BASE_DIR, "SAML-D.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "SAML-D.csv")  # Overwrite original
BACKUP_CSV = os.path.join(BASE_DIR, "SAML-D_original_backup.csv")

# Seed for reproducibility
random.seed(42)
np.random.seed(42)

# ─── Synthetic Account IDs (clearly identifiable) ───────────────────────────
SYNTH_ACCOUNTS = [f"99{i:08d}" for i in range(1, 201)]  # 200 synthetic accounts

# ─── Currencies & Locations ──────────────────────────────────────────────────
HIGH_RISK_LOCATIONS = ["UAE", "Cayman Islands", "Switzerland", "Singapore", "Hong Kong"]
LOW_RISK_LOCATIONS = ["UK", "US", "Germany", "France", "Japan"]
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

LAUNDERING_TYPES = [
    "Smurfing",
    "Round_Tripping",
    "Cross_Border_Layering",
    "Fan_Out_Fan_In",
]


def random_time(base_date: datetime, delta_minutes: int = 0) -> tuple:
    """Generate a time string and date string."""
    dt = base_date + timedelta(minutes=delta_minutes)
    return dt.strftime("%H:%M:%S"), dt.strftime("%Y-%m-%d")


def generate_smurfing_pattern(base_date: datetime, account_pool: list) -> list:
    """
    Smurfing: One account sends many small rapid transactions to different
    recipients to avoid detection thresholds. All within 1-3 minutes.
    """
    rows = []
    sender = random.choice(account_pool[:50])
    receivers = random.sample(account_pool[50:150], k=random.randint(8, 15))
    sender_loc = random.choice(LOW_RISK_LOCATIONS)

    for i, recv in enumerate(receivers):
        recv_loc = random.choice(LOW_RISK_LOCATIONS + HIGH_RISK_LOCATIONS)
        amount = round(random.uniform(800, 4500), 2)  # Below typical thresholds
        time_str, date_str = random_time(base_date, delta_minutes=i)
        pay_curr = CURRENCIES[sender_loc]
        recv_curr = CURRENCIES[recv_loc]

        rows.append({
            "Time": time_str,
            "Date": date_str,
            "Sender_account": sender,
            "Receiver_account": recv,
            "Amount": amount,
            "Payment_currency": pay_curr,
            "Received_currency": recv_curr,
            "Sender_bank_location": sender_loc,
            "Receiver_bank_location": recv_loc,
            "Payment_type": "Wire",
            "Is_laundering": 1,
            "Laundering_type": "Smurfing",
        })
    return rows


def generate_round_tripping(base_date: datetime, account_pool: list) -> list:
    """
    Round-tripping: A → B → C → D → A. Money comes back to origin.
    High amounts, cross-border, within a short time.
    """
    rows = []
    chain_length = random.randint(3, 6)
    accounts = random.sample(account_pool, k=chain_length)
    accounts.append(accounts[0])  # Close the loop

    amount = round(random.uniform(15000, 80000), 2)
    locations = [random.choice(HIGH_RISK_LOCATIONS + LOW_RISK_LOCATIONS) for _ in accounts]

    for i in range(len(accounts) - 1):
        sender = accounts[i]
        receiver = accounts[i + 1]
        sloc = locations[i]
        rloc = locations[i + 1]
        # Slight amount variation to obscure
        tx_amount = round(amount * random.uniform(0.92, 1.0), 2)
        time_str, date_str = random_time(base_date, delta_minutes=i * random.randint(2, 10))

        rows.append({
            "Time": time_str,
            "Date": date_str,
            "Sender_account": sender,
            "Receiver_account": receiver,
            "Amount": tx_amount,
            "Payment_currency": CURRENCIES[sloc],
            "Received_currency": CURRENCIES[rloc],
            "Sender_bank_location": sloc,
            "Receiver_bank_location": rloc,
            "Payment_type": "Cross-border",
            "Is_laundering": 1,
            "Laundering_type": "Round_Tripping",
        })
    return rows


def generate_cross_border_layering(base_date: datetime, account_pool: list) -> list:
    """
    Cross-border layering: Funds hop through multiple countries
    with currency conversions to obscure the trail.
    """
    rows = []
    chain_length = random.randint(4, 8)
    accounts = random.sample(account_pool, k=chain_length)

    # Force multi-country hops
    locations = []
    for _ in range(chain_length):
        if random.random() < 0.6:
            locations.append(random.choice(HIGH_RISK_LOCATIONS))
        else:
            locations.append(random.choice(LOW_RISK_LOCATIONS))

    amount = round(random.uniform(25000, 150000), 2)

    for i in range(len(accounts) - 1):
        sender = accounts[i]
        receiver = accounts[i + 1]
        sloc = locations[i]
        rloc = locations[i + 1]
        # Amount decreases slightly at each hop (fees)
        tx_amount = round(amount * (0.97 ** i), 2)
        time_str, date_str = random_time(base_date, delta_minutes=i * random.randint(5, 30))

        rows.append({
            "Time": time_str,
            "Date": date_str,
            "Sender_account": sender,
            "Receiver_account": receiver,
            "Amount": tx_amount,
            "Payment_currency": CURRENCIES[sloc],
            "Received_currency": CURRENCIES[rloc],
            "Sender_bank_location": sloc,
            "Receiver_bank_location": rloc,
            "Payment_type": "Cross-border",
            "Is_laundering": 1,
            "Laundering_type": "Cross_Border_Layering",
        })
    return rows


def generate_fan_out_fan_in(base_date: datetime, account_pool: list) -> list:
    """
    Fan-out/Fan-in: One account sends to many intermediaries,
    who then all send to a single collection account.
    """
    rows = []
    hub_sender = random.choice(account_pool[:30])
    hub_receiver = random.choice(account_pool[30:60])
    intermediaries = random.sample(account_pool[60:], k=random.randint(5, 12))

    total_amount = round(random.uniform(50000, 200000), 2)
    per_amount = round(total_amount / len(intermediaries), 2)

    hub_loc = random.choice(LOW_RISK_LOCATIONS)

    # Fan-out: hub → intermediaries
    for i, mid in enumerate(intermediaries):
        mid_loc = random.choice(HIGH_RISK_LOCATIONS + LOW_RISK_LOCATIONS)
        time_str, date_str = random_time(base_date, delta_minutes=i * 2)

        rows.append({
            "Time": time_str,
            "Date": date_str,
            "Sender_account": hub_sender,
            "Receiver_account": mid,
            "Amount": round(per_amount * random.uniform(0.85, 1.15), 2),
            "Payment_currency": CURRENCIES[hub_loc],
            "Received_currency": CURRENCIES[mid_loc],
            "Sender_bank_location": hub_loc,
            "Receiver_bank_location": mid_loc,
            "Payment_type": "Wire",
            "Is_laundering": 1,
            "Laundering_type": "Fan_Out_Fan_In",
        })

    # Fan-in: intermediaries → collection account
    collect_loc = random.choice(HIGH_RISK_LOCATIONS)
    for i, mid in enumerate(intermediaries):
        mid_loc = random.choice(HIGH_RISK_LOCATIONS + LOW_RISK_LOCATIONS)
        time_str, date_str = random_time(
            base_date + timedelta(hours=random.randint(1, 6)),
            delta_minutes=i * 3,
        )

        rows.append({
            "Time": time_str,
            "Date": date_str,
            "Sender_account": mid,
            "Receiver_account": hub_receiver,
            "Amount": round(per_amount * random.uniform(0.80, 0.95), 2),
            "Payment_currency": CURRENCIES[mid_loc],
            "Received_currency": CURRENCIES[collect_loc],
            "Sender_bank_location": mid_loc,
            "Receiver_bank_location": collect_loc,
            "Payment_type": "Cross-border",
            "Is_laundering": 1,
            "Laundering_type": "Fan_Out_Fan_In",
        })

    return rows


def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  AML Data Augmentation — Synthetic Laundering Patterns  ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    # Load original
    print("[1/4] Loading original CSV ...")
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

    # Generate synthetic patterns
    print("\n[3/4] Generating synthetic laundering patterns ...")
    synthetic_rows = []
    base_dates = pd.date_range("2022-09-01", "2023-03-31", freq="D").tolist()

    # Pattern counts
    n_smurfing = 80
    n_round_trip = 60
    n_layering = 60
    n_fan_out = 40

    for i in range(n_smurfing):
        base = random.choice(base_dates) + timedelta(hours=random.randint(0, 23))
        synthetic_rows.extend(generate_smurfing_pattern(base, SYNTH_ACCOUNTS))

    for i in range(n_round_trip):
        base = random.choice(base_dates) + timedelta(hours=random.randint(0, 23))
        synthetic_rows.extend(generate_round_tripping(base, SYNTH_ACCOUNTS))

    for i in range(n_layering):
        base = random.choice(base_dates) + timedelta(hours=random.randint(0, 23))
        synthetic_rows.extend(generate_cross_border_layering(base, SYNTH_ACCOUNTS))

    for i in range(n_fan_out):
        base = random.choice(base_dates) + timedelta(hours=random.randint(0, 23))
        synthetic_rows.extend(generate_fan_out_fan_in(base, SYNTH_ACCOUNTS))

    synth_df = pd.DataFrame(synthetic_rows)
    print(f"  ✓ Smurfing       : {n_smurfing} patterns → {sum(1 for r in synthetic_rows if r['Laundering_type'] == 'Smurfing'):,} txs")
    print(f"  ✓ Round-tripping : {n_round_trip} patterns → {sum(1 for r in synthetic_rows if r['Laundering_type'] == 'Round_Tripping'):,} txs")
    print(f"  ✓ Cross-border   : {n_layering} patterns → {sum(1 for r in synthetic_rows if r['Laundering_type'] == 'Cross_Border_Layering'):,} txs")
    print(f"  ✓ Fan-out/Fan-in : {n_fan_out} patterns → {sum(1 for r in synthetic_rows if r['Laundering_type'] == 'Fan_Out_Fan_In'):,} txs")
    print(f"  ✓ Total synthetic: {len(synth_df):,} transactions (all Is_laundering=1)")

    # Merge
    print("\n[4/4] Merging with original dataset ...")
    df_augmented = pd.concat([df, synth_df], ignore_index=True)

    # Shuffle
    df_augmented = df_augmented.sample(frac=1, random_state=42).reset_index(drop=True)

    new_pos = int(df_augmented["Is_laundering"].sum())
    new_neg = len(df_augmented) - new_pos
    print(f"  Augmented: {len(df_augmented):,} rows")
    print(f"  Class 0 (Normal)    : {new_neg:,}  ({new_neg/len(df_augmented)*100:.2f}%)")
    print(f"  Class 1 (Laundering): {new_pos:,}  ({new_pos/len(df_augmented)*100:.2f}%)")
    print(f"  Added laundering    : +{new_pos - original_pos:,} transactions")
    print(f"  New imbalance ratio : {new_neg/max(new_pos,1):.1f}:1")

    # Save
    df_augmented.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  ✓ Saved augmented CSV to: {OUTPUT_CSV}")

    print("\n" + "=" * 58)
    print("  NEXT STEPS:")
    print("  1. python data_pipeline.py   (rebuild graph)")
    print("  2. python train.py           (retrain model)")
    print("=" * 58)
    print("\n✅ Data augmentation complete!\n")


if __name__ == "__main__":
    main()
