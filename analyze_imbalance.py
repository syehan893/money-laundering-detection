"""
analyze_imbalance.py — Summary Ketidakseimbangan Data AML
==========================================================

Script standalone untuk menghitung dan menvisualisasikan
class imbalance pada dataset AML. Dapat dijalankan langsung:

    python analyze_imbalance.py

Output:
    - Tabel summary di terminal
    - Visualisasi disimpan sebagai: aml_imbalance_summary.png
"""

import os
import sys
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_data():
    """Load data dari processed_data.pt jika tersedia, fallback ke CSV."""
    pt_path = os.path.join(os.path.dirname(__file__), "data", "processed_data.pt")
    csv_train = os.path.join(os.path.dirname(__file__), "data", "train.csv")
    csv_test = os.path.join(os.path.dirname(__file__), "data", "test.csv")

    if os.path.exists(pt_path):
        try:
            import torch
            data = torch.load(pt_path, map_location="cpu", weights_only=False)
            y = data.y.numpy()
            return y, "processed_data.pt"
        except Exception as e:
            print(f"[WARN] Gagal load .pt: {e}, mencoba CSV...")

    # Fallback ke CSV
    if os.path.exists(csv_train) and os.path.exists(csv_test):
        import pandas as pd
        train = pd.read_csv(csv_train)
        test = pd.read_csv(csv_test)
        df = pd.concat([train, test], ignore_index=True)
        y = df["Is_laundering"].values.astype(np.int64)
        return y, "CSV files"

    print("[ERROR] Tidak ditemukan data (processed_data.pt atau CSV)")
    sys.exit(1)


def compute_metrics(y):
    """Hitung semua metric imbalance."""
    total = len(y)
    normal = int((y == 0).sum())
    laundering = int((y == 1).sum())

    ratio = normal / laundering if laundering > 0 else float("inf")
    pct_normal = (normal / total) * 100
    pct_laundering = (laundering / total) * 100

    return {
        "total": total,
        "normal": normal,
        "laundering": laundering,
        "ratio": ratio,
        "pct_normal": pct_normal,
        "pct_laundering": pct_laundering,
    }


def print_summary(m):
    """Tampilkan tabel summary di terminal."""
    print("\n" + "=" * 60)
    print("  CLASS IMBALANCE SUMMARY — AML DETECTION")
    print("=" * 60)

    print(f"\n  {'Metric':<25} {'Value':>30}")
    print("  " + "-" * 56)
    print(f"  {'Total Samples':<25} {m['total']:>30,}")
    print(f"  {'Normal (y=0)':<25} {m['normal']:>30,}")
    print(f"  {'Laundering (y=1)':<25} {m['laundering']:>30,}")
    print("  " + "-" * 56)
    print(f"  {'% Normal':<25} {m['pct_normal']:>29.2f}%")
    print(f"  {'% Laundering':<25} {m['pct_laundering']:>29.2f}%")
    print(f"  {'Imbalance Ratio':<25} {'1 : ' + str(round(m['ratio'], 2)):>30}")
    print("=" * 60 + "\n")


def plot_summary(m, output_path="aml_imbalance_summary.png"):
    """Generate visualisasi summary dan simpan ke file."""
    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 3, wspace=0.35)

    # --- Subplot 1: Pie Chart ---
    ax1 = fig.add_subplot(gs[0, 0])
    sizes = [m["normal"], m["laundering"]]
    labels = [f"Normal\n{m['normal']:,}\n({m['pct_normal']:.1f}%)",
              f"Laundering\n{m['laundering']:,}\n({m['pct_laundering']:.1f}%)"]
    colors = ["#3498db", "#e74c3c"]
    explode = (0, 0.05)

    ax1.pie(sizes, labels=labels, colors=colors, explode=explode,
            autopct="", startangle=90, textprops={"fontsize": 10})
    ax1.set_title("Distribusi Kelas", fontsize=12, fontweight="bold")

    # --- Subplot 2: Bar Chart ---
    ax2 = fig.add_subplot(gs[0, 1])
    bars = ax2.bar(["Normal", "Laundering"], sizes, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("Jumlah Transaksi", fontsize=11)
    ax2.set_title("Perbandingan Jumlah", fontsize=12, fontweight="bold")
    ax2.grid(True, ls="--", alpha=0.3, axis="y")

    for bar, val in zip(bars, sizes):
        height = bar.get_height()
        ax2.annotate(f"{val:,}",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=11, fontweight="bold")

    # --- Subplot 3: Ratio Visualization ---
    ax3 = fig.add_subplot(gs[0, 2])
    ratio = m["ratio"]
    # Visualisasi: untuk setiap 1 laundering, berapa normal
    visual_normal = int(round(ratio))
    visual_laundering = 1

    # Buat stacked bar yang merepresentasikan ratio
    ax3.barh(["Ratio"], [visual_normal], color="#3498db", edgecolor="black", label="Normal")
    ax3.barh(["Ratio"], [visual_laundering], left=[visual_normal],
             color="#e74c3c", edgecolor="black", label="Laundering")

    ax3.set_xlim(0, visual_normal + 2)
    ax3.set_title(f"Imbalance Ratio\n1 : {ratio:.2f}", fontsize=12, fontweight="bold")
    ax3.legend(loc="upper right")
    ax3.set_xticks([])
    ax3.set_yticks([])

    # Annotate
    ax3.text(visual_normal / 2, 0, f"{visual_normal}x", ha="center", va="center",
             fontsize=14, fontweight="bold", color="white")
    ax3.text(visual_normal + 0.5, 0, "1x", ha="center", va="center",
             fontsize=14, fontweight="bold", color="white")

    fig.suptitle("AML Dataset — Class Imbalance Summary", fontsize=14, fontweight="bold", y=1.02)

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] Visualisasi disimpan: {output_path}")


def compare_with_splits():
    """Opsional: tampilkan perbandingan per split jika processed_data.pt tersedia."""
    pt_path = os.path.join(os.path.dirname(__file__), "data", "processed_data.pt")
    if not os.path.exists(pt_path):
        return

    try:
        import torch
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        y = data.y

        print("\n  PERBANDINGAN PER SPLIT:")
        print("  " + "-" * 56)
        print(f"  {'Split':<10} {'Total':>12} {'Normal':>12} {'Laundering':>12} {'Ratio':>8}")
        print("  " + "-" * 56)

        for name, mask in [("Train", data.train_mask), ("Val", data.val_mask), ("Test", data.test_mask)]:
            y_split = y[mask]
            n = int((y_split == 0).sum())
            l = int((y_split == 1).sum())
            t = len(y_split)
            r = n / l if l > 0 else 0
            print(f"  {name:<10} {t:>12,} {n:>12,} {l:>12,} {r:>7.2f}:1")

        print("  " + "-" * 56)
    except Exception as e:
        print(f"[WARN] Tidak bisa membaca split info: {e}")


def main():
    print("\n[1/3] Loading data ...")
    y, source = load_data()
    print(f"      [OK] Data loaded from: {source}")
    print(f"      [OK] Total samples: {len(y):,}")

    print("\n[2/3] Computing metrics ...")
    metrics = compute_metrics(y)

    print("\n[3/3] Generating visualization ...")
    out_path = os.path.join(os.path.dirname(__file__), "aml_imbalance_summary.png")
    plot_summary(metrics, out_path)

    print_summary(metrics)
    compare_with_splits()


if __name__ == "__main__":
    main()
