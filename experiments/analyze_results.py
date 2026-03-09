#!/usr/bin/env python3
"""
Analyze results from all 8 phases and produce 7 publication-quality figures.

Figures:
  1. Backend sensitivity     (plaintext vs file_oram vs ram_oram)
  2. Worker scaling           (epoch_time vs num_workers)
  3. Block size sweep         (epoch_time vs block_size)
  4. Model scaling            (slowdown_ratio vs model)
  5. Dataset scaling          (per_access_latency vs N, O(log N) ref)
  6. Batch size sweep         (epoch_time vs batch_size)
  7. Access leakage comparison (plaintext vs ORAM histogram)

Usage:
    python experiments/analyze_results.py                   # auto-detect
    python experiments/analyze_results.py --results-root results --output results/figures
"""

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS_ROOT = "results"
OUTPUT_DIR = "results/figures"


# ── helpers ────────────────────────────────────────────────────
def _load_csv(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_history(directory):
    p = os.path.join(directory, "history.json")
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f)


def _load_profile(directory):
    for fname in os.listdir(directory):
        if fname.endswith("_profile.json"):
            with open(os.path.join(directory, fname)) as f:
                return json.load(f)
    return {}


def _epoch_time(hist):
    """Average seconds per epoch from a history dict."""
    n = len(hist.get("epochs", []))
    t = hist.get("total_time", 0)
    return t / n if n > 0 else 0


def _ensure(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


# ── Figure 1: Backend sensitivity ──────────────────────────────
def fig_backend_sensitivity(root, out):
    """Plaintext vs file-ORAM vs RAM-ORAM epoch time."""
    data = []

    # Phase 0 baseline
    h = _load_history(os.path.join(root, "phase0", "baseline"))
    if h:
        data.append({"config": "Plaintext", "epoch_time": _epoch_time(h)})

    # Phase 0 file oram
    h = _load_history(os.path.join(root, "phase0", "oram_file"))
    if h:
        data.append({"config": "File ORAM", "epoch_time": _epoch_time(h)})

    # Phase 1 ram oram
    h = _load_history(os.path.join(root, "phase1", "oram_ram"))
    if h:
        data.append({"config": "RAM ORAM", "epoch_time": _epoch_time(h)})

    if not data:
        print("  fig_backend_sensitivity: no data")
        return

    df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["steelblue", "coral", "mediumseagreen"][:len(df)]
    bars = ax.bar(df["config"], df["epoch_time"], color=colors)
    for bar, val in zip(bars, df["epoch_time"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{val:.1f}s", ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Epoch Time (s)")
    ax.set_title("Backend Sensitivity")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out, "fig1_backend_sensitivity.png")
    _ensure(path)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Figure 2: Worker scaling ───────────────────────────────────
def fig_worker_scaling(root, out):
    """Epoch time vs num_workers."""
    data = []
    for nw in (0, 1, 2, 4):
        h = _load_history(os.path.join(root, "phase2", f"workers_{nw}"))
        if h:
            data.append({"num_workers": nw, "epoch_time": _epoch_time(h)})

    if not data:
        print("  fig_worker_scaling: no data")
        return

    df = pd.DataFrame(data).sort_values("num_workers")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df["num_workers"], df["epoch_time"], "o-", linewidth=2, color="coral")
    for _, row in df.iterrows():
        ax.annotate(f'{row["epoch_time"]:.1f}s',
                    (row["num_workers"], row["epoch_time"]),
                    textcoords="offset points", xytext=(0, 10), ha="center")
    ax.set_xlabel("Number of Workers")
    ax.set_ylabel("Epoch Time (s)")
    ax.set_title("Worker Scaling (Mediated Loader)")
    ax.set_xticks(df["num_workers"].tolist())
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out, "fig2_worker_scaling.png")
    _ensure(path)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Figure 3: Block size sweep ─────────────────────────────────
def fig_block_size(root, out):
    """Epoch time vs block size."""
    data = []
    for bs in (4096, 8192, 16384, 32768, 65536):
        h = _load_history(os.path.join(root, "phase3", f"block_{bs}"))
        if h:
            data.append({"block_size": bs, "epoch_time": _epoch_time(h)})

    if not data:
        print("  fig_block_size: no data")
        return

    df = pd.DataFrame(data).sort_values("block_size")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df["block_size"] / 1024, df["epoch_time"], "s-", linewidth=2, color="coral")
    for _, row in df.iterrows():
        ax.annotate(f'{row["epoch_time"]:.1f}s',
                    (row["block_size"]/1024, row["epoch_time"]),
                    textcoords="offset points", xytext=(0, 10), ha="center")
    ax.set_xlabel("Block Size (KB)")
    ax.set_ylabel("Epoch Time (s)")
    ax.set_title("ORAM Block Size Sweep")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out, "fig3_block_size.png")
    _ensure(path)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Figure 4: Model scaling ───────────────────────────────────
def fig_model_scaling(root, out):
    """Slowdown ratio per model."""
    models = ["resnet18", "resnet50", "efficientnet_b0"]
    data = []
    for m in models:
        h_bl = _load_history(os.path.join(root, "phase4", f"baseline_{m}"))
        h_or = _load_history(os.path.join(root, "phase4", f"oram_{m}"))
        if h_bl and h_or:
            t_bl = _epoch_time(h_bl)
            t_or = _epoch_time(h_or)
            ratio = t_or / t_bl if t_bl > 0 else 0
            data.append({"model": m, "plaintext": t_bl, "oram": t_or,
                         "slowdown": ratio})

    if not data:
        print("  fig_model_scaling: no data")
        return

    df = pd.DataFrame(data)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    x = np.arange(len(df))
    w = 0.35
    ax1.bar(x - w/2, df["plaintext"], w, label="Plaintext", color="steelblue")
    ax1.bar(x + w/2, df["oram"], w, label="ORAM", color="coral")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["model"])
    ax1.set_ylabel("Epoch Time (s)")
    ax1.set_title("Training Time by Model")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(df["model"], df["slowdown"], color="coral")
    for i, (_, row) in enumerate(df.iterrows()):
        ax2.text(i, row["slowdown"], f'{row["slowdown"]:.1f}x',
                 ha="center", va="bottom", fontweight="bold")
    ax2.set_ylabel("Slowdown Ratio (ORAM / Plaintext)")
    ax2.set_title("ORAM Overhead by Model")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out, "fig4_model_scaling.png")
    _ensure(path)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Figure 5: Dataset scaling ──────────────────────────────────
def fig_dataset_scaling(root, out):
    """Epoch time vs N with O(log N) reference."""
    data = []
    for n in (5000, 10000, 25000, 50000):
        h = _load_history(os.path.join(root, "phase5", f"n_{n}"))
        if h:
            data.append({"N": n, "epoch_time": _epoch_time(h)})

    if not data:
        print("  fig_dataset_scaling: no data")
        return

    df = pd.DataFrame(data).sort_values("N")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(df["N"], df["epoch_time"], "o-", linewidth=2, color="coral")
    ax1.set_xlabel("Dataset Size (N)")
    ax1.set_ylabel("Epoch Time (s)")
    ax1.set_title("ORAM Training Time vs Dataset Size")
    ax1.set_xscale("log")
    ax1.grid(True, alpha=0.3)

    # Per-sample time with O(log N) reference
    epochs = df.apply(lambda r: len(_load_history(
        os.path.join(root, "phase5", f"n_{int(r['N'])}")).get("epochs", [1])), axis=1)
    per_sample = df["epoch_time"] / df["N"]
    ax2.plot(df["N"], per_sample * 1000, "o-", linewidth=2, color="coral", label="Measured")
    ns = df["N"].values.astype(float)
    log_n = np.log2(ns)
    if len(per_sample) > 0 and per_sample.iloc[0] > 0:
        scale = (per_sample.iloc[0] * 1000) / log_n[0]
        ax2.plot(ns, log_n * scale, "--", color="gray", label="O(log N) reference")
    ax2.set_xlabel("Dataset Size (N)")
    ax2.set_ylabel("Time per Sample (ms)")
    ax2.set_title("Per-Sample ORAM Access Time")
    ax2.set_xscale("log")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out, "fig5_dataset_scaling.png")
    _ensure(path)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Figure 6: Batch size sweep ────────────────────────────────
def fig_batch_size(root, out):
    """Epoch time vs batch size for plaintext and ORAM."""
    bl_data, or_data = [], []
    batch_sizes = [32, 64, 128, 256, 512]
    for bs in batch_sizes:
        h = _load_history(os.path.join(root, "phase6", f"baseline_bs{bs}"))
        if h:
            bl_data.append({"batch_size": bs, "epoch_time": _epoch_time(h)})
        h = _load_history(os.path.join(root, "phase6", f"oram_bs{bs}"))
        if h:
            or_data.append({"batch_size": bs, "epoch_time": _epoch_time(h)})

    if not bl_data and not or_data:
        print("  fig_batch_size: no data")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    if bl_data:
        bl = pd.DataFrame(bl_data).sort_values("batch_size")
        ax1.plot(bl["batch_size"], bl["epoch_time"], "o-", label="Plaintext",
                 linewidth=2, color="steelblue")
    if or_data:
        od = pd.DataFrame(or_data).sort_values("batch_size")
        ax1.plot(od["batch_size"], od["epoch_time"], "s-", label="ORAM",
                 linewidth=2, color="coral")
    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("Epoch Time (s)")
    ax1.set_title("Training Time vs Batch Size")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Overhead ratio
    if bl_data and or_data:
        bl = pd.DataFrame(bl_data)
        od = pd.DataFrame(or_data)
        merged = bl.merge(od, on="batch_size", suffixes=("_bl", "_oram"))
        if not merged.empty:
            merged["ratio"] = merged["epoch_time_oram"] / merged["epoch_time_bl"]
            merged = merged.sort_values("batch_size")
            ax2.bar(merged["batch_size"].astype(str), merged["ratio"], color="coral")
            for i, (_, row) in enumerate(merged.iterrows()):
                ax2.text(i, row["ratio"], f'{row["ratio"]:.1f}x',
                         ha="center", va="bottom", fontweight="bold")
            ax2.set_xlabel("Batch Size")
            ax2.set_ylabel("Overhead Ratio")
            ax2.set_title("ORAM Overhead by Batch Size")
            ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out, "fig6_batch_size.png")
    _ensure(path)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Figure 7: Access leakage ──────────────────────────────────
def fig_leakage(root, out):
    """Plaintext vs ORAM access frequency histogram."""
    pt_path = os.path.join(root, "phase7", "plaintext_access_log.json")
    or_path = os.path.join(root, "phase7", "oram_access_log.json")

    if not os.path.exists(pt_path) or not os.path.exists(or_path):
        print("  fig_leakage: no data")
        return

    with open(pt_path) as f:
        pt = json.load(f)
    with open(or_path) as f:
        oram = json.load(f)

    pt_counts = pt.get("counts", {})
    oram_counts = oram.get("counts", {})

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    ax = axes[0]
    freqs = sorted(pt_counts.values(), reverse=True)
    ax.bar(range(len(freqs)), freqs, width=1.0, color="coral", edgecolor="none")
    ax.set_title("Plaintext Access Frequency")
    ax.set_xlabel("Sample rank")
    ax.set_ylabel("Access count")
    if freqs:
        ax.axhline(np.mean(freqs), ls="--", color="gray",
                    label=f"mean={np.mean(freqs):.1f}")
        ax.legend()

    ax = axes[1]
    freqs = sorted(oram_counts.values(), reverse=True)
    ax.bar(range(len(freqs)), freqs, width=1.0, color="steelblue", edgecolor="none")
    ax.set_title("ORAM Logical Access Frequency")
    ax.set_xlabel("Sample rank")
    if freqs:
        ax.axhline(np.mean(freqs), ls="--", color="gray",
                    label=f"mean={np.mean(freqs):.1f}")
        ax.legend()

    plt.suptitle("Access Pattern Leakage: Plaintext vs ORAM", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(out, "fig7_leakage.png")
    _ensure(path)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Summary table ─────────────────────────────────────────────
def generate_summary(root, out):
    """Generate a combined markdown summary from phase8."""
    rows = []
    for tag, label in [("plaintext", "Plaintext"),
                       ("oram_baseline", "Baseline ORAM"),
                       ("oram_optimized", "Optimized ORAM")]:
        h = _load_history(os.path.join(root, "phase8", tag))
        if h:
            rows.append({"System": label, "Epoch Time (s)": f"{_epoch_time(h):.1f}",
                          "Best Acc (%)": f"{h.get('best_acc', 0):.2f}"})

    if not rows:
        print("  summary: no phase8 data")
        return

    md = ["# ORAM Evaluation Summary\n",
          "| System | Epoch Time (s) | Best Acc (%) |",
          "|--------|----------------|--------------|"]
    for r in rows:
        md.append(f"| {r['System']} | {r['Epoch Time (s)']} | {r['Best Acc (%)']} |")
    md.append("")

    path = os.path.join(out, "summary.md")
    _ensure(path)
    with open(path, "w") as f:
        f.write("\n".join(md))
    print(f"  Saved: {path}")


# ── main ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Generate all figures from phase results")
    parser.add_argument("--results-root", type=str, default=RESULTS_ROOT)
    parser.add_argument("--output", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    root = args.results_root
    out = args.output
    os.makedirs(out, exist_ok=True)

    print("="*60)
    print("GENERATING FIGURES")
    print("="*60)

    fig_backend_sensitivity(root, out)
    fig_worker_scaling(root, out)
    fig_block_size(root, out)
    fig_model_scaling(root, out)
    fig_dataset_scaling(root, out)
    fig_batch_size(root, out)
    fig_leakage(root, out)
    generate_summary(root, out)

    print("\nDone. All figures saved to:", out)


if __name__ == "__main__":
    main()
