#!/usr/bin/env python3
"""
Parameter sweep experiments for ORAM overhead characterization.

Runs experiments varying:
1. Batch size: {32, 64, 128, 256} with fixed dataset size
2. Dataset size: {1000, 5000, 10000, 50000} with fixed batch size

Each configuration runs a small number of epochs (enough for stable
timing measurements). Results are saved per-configuration and a
combined summary is produced.

Usage:
    python experiments/run_sweep.py --sweep batch_size --epochs 3
    python experiments/run_sweep.py --sweep dataset_size --epochs 2
    python experiments/run_sweep.py --sweep all --epochs 3
"""

import argparse
import json
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.baseline_trainer import run_baseline_training
from src.oram_trainer import run_oram_training


BATCH_SIZES = [32, 64, 128, 256]
DATASET_SIZES = [1000, 5000, 10000, 50000]
DEFAULT_BATCH_SIZE = 128
DEFAULT_DATASET_SIZE = 5000  # Use 5k for sweep to keep runtime manageable


def run_batch_size_sweep(epochs: int, output_root: str, device: str = None):
    """Sweep over batch sizes for both baseline and ORAM."""
    print("\n" + "=" * 60)
    print("BATCH SIZE SWEEP")
    print("=" * 60)

    results = []

    for bs in BATCH_SIZES:
        # Baseline
        tag = f"baseline_bs{bs}"
        out_dir = os.path.join(output_root, "sweep_batch_size", tag)
        print(f"\n--- Baseline batch_size={bs}, epochs={epochs} ---")
        try:
            hist = run_baseline_training(
                num_epochs=epochs,
                batch_size=bs,
                output_dir=out_dir,
                device=device,
            )
            results.append({
                "mode": "baseline",
                "batch_size": bs,
                "epochs": epochs,
                "total_time": hist["total_time"],
                "best_acc": hist["best_acc"],
                "final_train_loss": hist["train_loss"][-1],
            })
        except Exception as exc:
            print(f"ERROR in baseline bs={bs}: {exc}")
            results.append({
                "mode": "baseline",
                "batch_size": bs,
                "error": str(exc),
            })

        # ORAM (use smaller dataset for tractability)
        tag = f"oram_bs{bs}"
        out_dir = os.path.join(output_root, "sweep_batch_size", tag)
        print(f"\n--- ORAM batch_size={bs}, epochs={epochs}, samples={DEFAULT_DATASET_SIZE} ---")
        try:
            hist = run_oram_training(
                num_epochs=epochs,
                batch_size=bs,
                output_dir=out_dir,
                device=device,
                num_samples=DEFAULT_DATASET_SIZE,
            )
            results.append({
                "mode": "oram",
                "batch_size": bs,
                "num_samples": DEFAULT_DATASET_SIZE,
                "epochs": epochs,
                "total_time": hist["total_time"],
                "best_acc": hist["best_acc"],
                "final_train_loss": hist["train_loss"][-1],
            })
        except Exception as exc:
            print(f"ERROR in oram bs={bs}: {exc}")
            results.append({
                "mode": "oram",
                "batch_size": bs,
                "error": str(exc),
            })

    # Save combined results
    summary_path = os.path.join(output_root, "sweep_batch_size", "sweep_summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBatch size sweep summary saved to: {summary_path}")

    return results


def run_dataset_size_sweep(epochs: int, output_root: str, device: str = None):
    """Sweep over dataset sizes for ORAM to measure O(log N) scaling."""
    print("\n" + "=" * 60)
    print("DATASET SIZE SWEEP (ORAM)")
    print("=" * 60)

    results = []

    for n in DATASET_SIZES:
        tag = f"oram_n{n}"
        out_dir = os.path.join(output_root, "sweep_dataset_size", tag)
        print(f"\n--- ORAM num_samples={n}, batch_size={DEFAULT_BATCH_SIZE}, epochs={epochs} ---")
        try:
            hist = run_oram_training(
                num_epochs=epochs,
                batch_size=DEFAULT_BATCH_SIZE,
                output_dir=out_dir,
                device=device,
                num_samples=n,
            )
            results.append({
                "mode": "oram",
                "num_samples": n,
                "batch_size": DEFAULT_BATCH_SIZE,
                "epochs": epochs,
                "total_time": hist["total_time"],
                "best_acc": hist["best_acc"],
                "final_train_loss": hist["train_loss"][-1],
            })
        except Exception as exc:
            print(f"ERROR in oram n={n}: {exc}")
            results.append({
                "mode": "oram",
                "num_samples": n,
                "error": str(exc),
            })

    # Save combined results
    summary_path = os.path.join(output_root, "sweep_dataset_size", "sweep_summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDataset size sweep summary saved to: {summary_path}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run parameter sweep experiments for ORAM overhead characterization"
    )
    parser.add_argument(
        "--sweep",
        type=str,
        choices=["batch_size", "dataset_size", "all"],
        default="all",
        help="Which sweep to run (default: all)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Epochs per sweep configuration (default: 3)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Root output directory (default: results)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, default: auto)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    start = time.time()

    if args.sweep in ("batch_size", "all"):
        run_batch_size_sweep(args.epochs, args.output_dir, args.device)

    if args.sweep in ("dataset_size", "all"):
        run_dataset_size_sweep(args.epochs, args.output_dir, args.device)

    elapsed = time.time() - start
    print(f"\nSweep complete. Total wall-clock time: {elapsed:.1f}s ({elapsed/3600:.2f}h)")


if __name__ == "__main__":
    main()
