#!/usr/bin/env python3
"""
Unified experiment runner for Phases 0-8.

Usage:
    python experiments/run_all_phases.py --phase 0        # baseline reproduction
    python experiments/run_all_phases.py --phase 1        # RAM backend
    python experiments/run_all_phases.py --phase all      # everything
    python experiments/run_all_phases.py --phase 0 --force  # re-run even if results exist
"""

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.baseline_trainer import run_baseline_training
from src.oram_trainer import run_oram_training
from src.oram_storage import DEFAULT_BLOCK_SIZE

RESULTS_ROOT = "results"


def _out(phase, tag=""):
    d = os.path.join(RESULTS_ROOT, f"phase{phase}", tag) if tag else os.path.join(RESULTS_ROOT, f"phase{phase}")
    return d


def _marker(phase, tag=""):
    return os.path.join(_out(phase, tag), "history.json")


def _done(phase, tag="", force=False):
    if force:
        return False
    return os.path.exists(_marker(phase, tag))


def _save_csv(rows, path):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV saved: {path}")


def _hist_row(hist, **extra):
    row = {
        "total_time": hist.get("total_time", 0),
        "best_acc": hist.get("best_acc", 0),
        "epochs": len(hist.get("epochs", [])),
    }
    row.update(extra)
    return row


# ── Phase 0 ────────────────────────────────────────────────────
def phase0(args):
    """Baseline reproduction: plaintext + file-ORAM."""
    print("\n" + "="*60)
    print("PHASE 0: Baseline Reproduction")
    print("="*60)
    rows = []

    # Plaintext baseline
    if not _done(0, "baseline", args.force):
        print("  Running plaintext baseline (3 epochs)...")
        h = run_baseline_training(num_epochs=3, batch_size=128,
                                  output_dir=_out(0, "baseline"), device=args.device)
        rows.append(_hist_row(h, mode="plaintext", backend="none"))
    else:
        print("  SKIP: plaintext baseline exists")

    # File-backed ORAM
    if not _done(0, "oram_file", args.force):
        print("  Running file-backed ORAM (2 epochs, 10k samples)...")
        h = run_oram_training(num_epochs=2, batch_size=128,
                              output_dir=_out(0, "oram_file"), device=args.device,
                              num_samples=10000, backend="file")
        rows.append(_hist_row(h, mode="oram", backend="file"))
    else:
        print("  SKIP: file ORAM exists")

    _save_csv(rows, os.path.join(_out(0), "phase0_results.csv"))


# ── Phase 1 ────────────────────────────────────────────────────
def phase1(args):
    """RAM-backed ORAM."""
    print("\n" + "="*60)
    print("PHASE 1: RAM Backend")
    print("="*60)
    rows = []

    for backend in ("file", "ram"):
        tag = f"oram_{backend}"
        if _done(1, tag, args.force):
            print(f"  SKIP: {tag} exists")
            continue
        print(f"  Running backend={backend} (2 epochs, 10k samples)...")
        h = run_oram_training(num_epochs=2, batch_size=128,
                              output_dir=_out(1, tag), device=args.device,
                              num_samples=10000, backend=backend)
        rows.append(_hist_row(h, backend=backend))

    _save_csv(rows, os.path.join(_out(1), "phase1_results.csv"))


# ── Phase 2 ────────────────────────────────────────────────────
def phase2(args):
    """Worker scaling."""
    print("\n" + "="*60)
    print("PHASE 2: Mediated Multi-Worker Loader")
    print("="*60)
    rows = []

    for nw in (0, 1, 2, 4):
        tag = f"workers_{nw}"
        if _done(2, tag, args.force):
            print(f"  SKIP: {tag} exists")
            continue
        print(f"  Running num_workers={nw} (2 epochs, 10k samples)...")
        h = run_oram_training(num_epochs=2, batch_size=128,
                              output_dir=_out(2, tag), device=args.device,
                              num_samples=10000, backend="file",
                              num_workers=nw)
        rows.append(_hist_row(h, num_workers=nw))

    _save_csv(rows, os.path.join(_out(2), "phase2_results.csv"))


# ── Phase 3 ────────────────────────────────────────────────────
def phase3(args):
    """Block size sweep."""
    print("\n" + "="*60)
    print("PHASE 3: Block Size Sweep")
    print("="*60)
    rows = []
    block_sizes = [4096, 8192, 16384, 32768, 65536]

    for bs in block_sizes:
        tag = f"block_{bs}"
        if _done(3, tag, args.force):
            print(f"  SKIP: {tag} exists")
            continue
        print(f"  Running block_size={bs} (2 epochs, 10k samples)...")
        h = run_oram_training(num_epochs=2, batch_size=128,
                              output_dir=_out(3, tag), device=args.device,
                              num_samples=10000, backend="file",
                              block_size=bs)
        rows.append(_hist_row(h, block_size=bs))

    _save_csv(rows, os.path.join(_out(3), "phase3_results.csv"))


# ── Phase 4 ────────────────────────────────────────────────────
def phase4(args):
    """Model scaling."""
    print("\n" + "="*60)
    print("PHASE 4: Model Scaling")
    print("="*60)
    rows = []
    models = ["resnet18", "resnet50", "efficientnet_b0"]

    for model in models:
        # Plaintext
        tag = f"baseline_{model}"
        if not _done(4, tag, args.force):
            print(f"  Running plaintext {model} (3 epochs)...")
            h = run_baseline_training(num_epochs=3, batch_size=128,
                                      output_dir=_out(4, tag), device=args.device,
                                      model_name=model)
            rows.append(_hist_row(h, mode="plaintext", model=model))
        else:
            print(f"  SKIP: {tag} exists")

        # ORAM
        tag = f"oram_{model}"
        if not _done(4, tag, args.force):
            print(f"  Running ORAM {model} (2 epochs, 5k samples)...")
            h = run_oram_training(num_epochs=2, batch_size=128,
                                  output_dir=_out(4, tag), device=args.device,
                                  num_samples=5000, model_name=model)
            rows.append(_hist_row(h, mode="oram", model=model))
        else:
            print(f"  SKIP: {tag} exists")

    _save_csv(rows, os.path.join(_out(4), "phase4_results.csv"))


# ── Phase 5 ────────────────────────────────────────────────────
def phase5(args):
    """Dataset size scaling."""
    print("\n" + "="*60)
    print("PHASE 5: Dataset Size Scaling")
    print("="*60)
    rows = []
    sizes = [5000, 10000, 25000, 50000]

    for n in sizes:
        tag = f"n_{n}"
        if _done(5, tag, args.force):
            print(f"  SKIP: {tag} exists")
            continue
        print(f"  Running N={n} (2 epochs)...")
        h = run_oram_training(num_epochs=2, batch_size=128,
                              output_dir=_out(5, tag), device=args.device,
                              num_samples=n)
        rows.append(_hist_row(h, num_samples=n))

    _save_csv(rows, os.path.join(_out(5), "phase5_results.csv"))


# ── Phase 6 ────────────────────────────────────────────────────
def phase6(args):
    """Batch size sweep."""
    print("\n" + "="*60)
    print("PHASE 6: Batch Size Sweep")
    print("="*60)
    rows = []
    batch_sizes = [32, 64, 128, 256, 512]

    for bs in batch_sizes:
        # Plaintext
        tag = f"baseline_bs{bs}"
        if not _done(6, tag, args.force):
            print(f"  Running plaintext batch_size={bs} (3 epochs)...")
            h = run_baseline_training(num_epochs=3, batch_size=bs,
                                      output_dir=_out(6, tag), device=args.device)
            rows.append(_hist_row(h, mode="plaintext", batch_size=bs))
        else:
            print(f"  SKIP: {tag} exists")

        # ORAM
        tag = f"oram_bs{bs}"
        if not _done(6, tag, args.force):
            print(f"  Running ORAM batch_size={bs} (2 epochs, 5k samples)...")
            h = run_oram_training(num_epochs=2, batch_size=bs,
                                  output_dir=_out(6, tag), device=args.device,
                                  num_samples=5000)
            rows.append(_hist_row(h, mode="oram", batch_size=bs))
        else:
            print(f"  SKIP: {tag} exists")

    _save_csv(rows, os.path.join(_out(6), "phase6_results.csv"))


# ── Phase 7 ────────────────────────────────────────────────────
def phase7(args):
    """Access pattern leakage demo."""
    print("\n" + "="*60)
    print("PHASE 7: Access Pattern Leakage Demo")
    print("="*60)
    marker = os.path.join(_out(7), "leakage_comparison.png")
    if os.path.exists(marker) and not args.force:
        print("  SKIP: leakage results exist")
        return

    exp_dir = os.path.dirname(os.path.abspath(__file__))
    leakage_script = os.path.join(exp_dir, "run_leakage_demo.py")
    import subprocess
    subprocess.check_call([
        sys.executable, leakage_script,
        "--num-samples", "5000",
        "--batch-size", "128",
        "--epochs", "3",
        "--output-dir", _out(7),
    ])


# ── Phase 8 ────────────────────────────────────────────────────
def phase8(args):
    """Combined optimization run."""
    print("\n" + "="*60)
    print("PHASE 8: Final Combined Optimization")
    print("="*60)
    rows = []

    # Plaintext reference
    tag = "plaintext"
    if not _done(8, tag, args.force):
        print("  Running plaintext reference (3 epochs)...")
        h = run_baseline_training(num_epochs=3, batch_size=128,
                                  output_dir=_out(8, tag), device=args.device)
        rows.append(_hist_row(h, config="plaintext"))
    else:
        print("  SKIP: plaintext exists")

    # Baseline ORAM (file, workers=0, 4KB blocks)
    tag = "oram_baseline"
    if not _done(8, tag, args.force):
        print("  Running baseline ORAM (2 epochs, 10k samples)...")
        h = run_oram_training(num_epochs=2, batch_size=128,
                              output_dir=_out(8, tag), device=args.device,
                              num_samples=10000, backend="file",
                              num_workers=0, block_size=4096)
        rows.append(_hist_row(h, config="oram_baseline"))
    else:
        print("  SKIP: oram_baseline exists")

    # Optimized ORAM (ram, workers=4, best batch size)
    tag = "oram_optimized"
    if not _done(8, tag, args.force):
        print("  Running optimized ORAM (2 epochs, 10k samples, ram, workers=4)...")
        h = run_oram_training(num_epochs=2, batch_size=256,
                              output_dir=_out(8, tag), device=args.device,
                              num_samples=10000, backend="ram",
                              num_workers=4, block_size=4096)
        rows.append(_hist_row(h, config="oram_optimized"))
    else:
        print("  SKIP: oram_optimized exists")

    _save_csv(rows, os.path.join(_out(8), "phase8_results.csv"))


PHASES = {
    "0": phase0,
    "1": phase1,
    "2": phase2,
    "3": phase3,
    "4": phase4,
    "5": phase5,
    "6": phase6,
    "7": phase7,
    "8": phase8,
}


def main():
    parser = argparse.ArgumentParser(description="Run ORAM evaluation phases 0-8")
    parser.add_argument("--phase", type=str, default="all",
                        help="Phase to run: 0-8 or 'all'")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if results exist")
    args = parser.parse_args()

    start = time.time()

    if args.phase == "all":
        for key in sorted(PHASES.keys()):
            PHASES[key](args)
    elif args.phase in PHASES:
        PHASES[args.phase](args)
    else:
        print(f"Unknown phase: {args.phase}. Use 0-8 or 'all'.")
        sys.exit(1)

    elapsed = time.time() - start
    print(f"\nDone. Wall-clock: {elapsed:.1f}s ({elapsed/3600:.2f}h)")


if __name__ == "__main__":
    main()
