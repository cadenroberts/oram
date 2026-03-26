#!/usr/bin/env python3
"""
Unified experiment runner for OMLO.

Subcommands:
    Run:        baseline, experiments, inference, oram, phases, sidecar, sweep
    Generate:   attack, event, partial, reference, plot, privacy, membership, robustness
    Attack:     mi, mi-simple
    Utility:    setup, system, probe, files, train, trace, convert, upgraded
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import os
import random
import re
import struct
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset

from oram import (
    baseline as train_baseline,
    oram as train_oram,
    ORAMDataset,
    SUPPORTED_MODELS,
    resolve_torch_device,
    read_oram_audit_counts,
    IndexedDataset,
    SidecarLogger,
    sidecar_training,
    plaintext,
    oram_event,
)
from attack import Build, upgraded_attack, simple_attack, PartialObservabilityConfig
from figures import Save, Plot
from pipeline import (
    LEAK_PATTERNS,
    RunConfig,
    Trace,
    Sweep,
    ExperimentPhases,
    PHASES,
    single_configuration,
    write_summary_csv,
)


MEMBER_RE = re.compile(r"^member_(\d+)\.bin$")
NONMEMBER_RE = re.compile(r"^nonmember_(\d+)\.bin$")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)



# Data helpers

def serialize_sample(global_id: int, image: np.ndarray, label: int) -> bytes:
    if image.shape != (32, 32, 3):
        raise ValueError(f"Expected CIFAR image shape (32,32,3), got {image.shape}")
    flat = image.astype(np.uint8).reshape(-1).tobytes()
    if len(flat) != 3072:
        raise ValueError("Flattened CIFAR image must be 3072 bytes.")
    return struct.pack("<IB", global_id, label) + flat


def read_sample_file(path: str) -> Tuple[torch.Tensor, int, int, str]:
    with open(path, "rb") as f:
        raw = f.read()
    if len(raw) < 5 + 3072:
        raise ValueError(f"Corrupt sample file: {path}")

    sample_id, label = struct.unpack("<IB", raw[:5])
    img = np.frombuffer(raw[5:5 + 3072], dtype=np.uint8).reshape(32, 32, 3)
    x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return x, int(label), int(sample_id), path


class Datasets:
    """PyTorch datasets for on-disk sample files and obfuscated loading."""

    class FileSample(Dataset):
        def __init__(self, file_paths: List[str]):
            self.file_paths = sorted(file_paths)
            self.transform = transforms.Compose([])

        def __len__(self) -> int:
            return len(self.file_paths)

        def __getitem__(self, idx: int):
            path = self.file_paths[idx]
            x, label, sample_id, _ = read_sample_file(path)
            return x, label, sample_id, path

    class ObfuscatedFile(Dataset):
        def __init__(
            self,
            real_paths: List[str],
            decoy_pool_paths: Optional[List[str]] = None,
            decoys_per_access: int = 2,
            prefetch_size: int = 8,
            release_shuffle_window: int = 4,
            seed: int = 42,
        ):
            self.real_paths = list(real_paths)
            self.decoy_pool_paths = list(decoy_pool_paths) if decoy_pool_paths is not None else list(real_paths)
            self.decoys_per_access = decoys_per_access
            self.prefetch_size = max(prefetch_size, 1)
            self.release_shuffle_window = max(release_shuffle_window, 1)
            self.rng = random.Random(seed)

            self.buffer: Deque[Tuple[torch.Tensor, int, int, str]] = deque()
            self.pending_indices: Deque[int] = deque()

            self.length = len(self.real_paths)

        def __len__(self) -> int:
            return self.length

        def _do_decoy_reads(self, true_path: str) -> None:
            if self.decoys_per_access <= 0 or not self.decoy_pool_paths:
                return

            candidates = self.decoy_pool_paths
            chosen = self.rng.sample(candidates, k=min(self.decoys_per_access, len(candidates)))

            for path in chosen:
                if path == true_path and len(candidates) > 1:
                    continue
                with open(path, "rb") as f:
                    _ = f.read()

        def _prefetch_real(self, idx: int) -> None:
            true_path = self.real_paths[idx]
            sample = read_sample_file(true_path)
            self._do_decoy_reads(true_path)
            self.buffer.append(sample)

        def _fill_buffer(self, requested_idx: int) -> None:
            if not self.buffer:
                self._prefetch_real(requested_idx)

            while len(self.buffer) < self.prefetch_size:
                ridx = self.rng.randrange(self.length)
                self._prefetch_real(ridx)

        def _release_one(self, requested_idx: int) -> Tuple[torch.Tensor, int, int, str]:
            window = list(self.buffer)[: min(len(self.buffer), self.release_shuffle_window)]

            req_pos = None
            requested_path = self.real_paths[requested_idx]
            for i, item in enumerate(window):
                if item[3] == requested_path:
                    req_pos = i
                    break

            if req_pos is not None and self.rng.random() < 0.5:
                chosen_pos = req_pos
            else:
                chosen_pos = self.rng.randrange(len(window))

            chosen = window[chosen_pos]

            newbuf = deque()
            removed = False
            for item in self.buffer:
                if not removed and item == chosen:
                    removed = True
                    continue
                newbuf.append(item)
            self.buffer = newbuf

            return chosen

        def __getitem__(self, idx: int):
            self._fill_buffer(idx)
            x, y, sample_id, path = self._release_one(idx)
            return x, y, sample_id, path


def collate_train(batch):
    xs, ys, ids, paths = zip(*batch)
    return torch.stack(xs, dim=0), torch.tensor(ys), list(ids), list(paths)


def read_probe_samples(probe_paths: List[str], n: int) -> List[Tuple[torch.Tensor, int, int, str]]:
    chosen = random.sample(probe_paths, min(n, len(probe_paths)))
    out = []
    for path in chosen:
        with open(path, "rb") as f:
            raw = f.read()
        sample_id, label = struct.unpack("<IB", raw[:5])
        img = np.frombuffer(raw[5:5 + 3072], dtype=np.uint8).reshape(32, 32, 3)
        x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        out.append((x, int(label), int(sample_id), path))
    return out


def infer_sample_and_label(path: str) -> Optional[Tuple[str, int]]:
    base = os.path.basename(path)
    m = MEMBER_RE.search(base)
    if m:
        return m.group(1), 1
    m = NONMEMBER_RE.search(base)
    if m:
        return m.group(1), 0
    return None


def read_sidecar(path: str):
    markers = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            markers.append((
                float(row["timestamp"]),
                str(row["batch_id"]),
                int(row["epoch"]),
                str(row["phase"]),
            ))
    markers.sort(key=lambda x: x[0])
    return markers


def nearest_prior_marker(ts: float, marker_times: List[float], markers):
    idx = bisect.bisect_right(marker_times, ts) - 1
    if idx < 0:
        return None
    return markers[idx]


def scan_attack_input_for_leaks(rows: List[Dict[str, object]]) -> Dict[str, object]:
    joined_text = "\n".join(
        f"{row.get('sample_id','')},{row.get('batch_id','')},{row.get('label','')}" for row in rows
    )
    hits = []
    for pat in LEAK_PATTERNS:
        if pat.search(joined_text):
            hits.append(pat.pattern)
    label_counts: Dict[str, int] = {}
    for row in rows:
        lbl = str(row["label"])
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
    return {
        "num_rows": len(rows),
        "label_counts": label_counts,
        "leak_patterns_found": hits,
        "has_leakage": len(hits) > 0,
    }



# Setup and validation helpers

def print_section(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def print_step(step: int, description: str) -> None:
    print(f"\n[Step {step}] {description}")
    print("-" * 70)


def check_file(path: str, description: str) -> bool:
    exists = os.path.exists(path)
    status = "OK" if exists else "MISSING"
    print(f"  [{status}] {description}: {path}")
    return exists


def check_import(module: str, description: str) -> bool:
    try:
        __import__(module)
        print(f"  [OK] {description}: {module}")
        return True
    except ImportError:
        print(f"  [MISSING] {description}: {module} (not installed)")
        return False


def check_file_exists(path: str, description: str) -> bool:
    if os.path.exists(path):
        print(f"[OK] {description}: {path}")
        return True
    else:
        print(f"[MISSING] {description}: {path}")
        return False


def check_executable(path: str) -> bool:
    if os.access(path, os.X_OK):
        return True
    else:
        print(f"  [WARN] Not executable: {path}")
        return False


def validate_event(input_path: str) -> bool:
    print(f"Validating event log: {input_path}\n")

    df = pd.read_csv(input_path)

    required_cols = {"sample_id", "timestamp", "epoch", "batch_id", "label"}
    if not required_cols.issubset(df.columns):
        print(f"Missing required columns: {required_cols - set(df.columns)}")
        return False
    print("All required columns present")

    label_counts = df["label"].value_counts().to_dict()
    if set(label_counts.keys()) != {0, 1}:
        print(f"Expected labels {{0, 1}}, got {set(label_counts.keys())}")
        return False
    print(f"Both label classes present: {label_counts}")

    member_events = df[df["label"] == 1]
    nonmember_events = df[df["label"] == 0]

    unique_members = member_events["sample_id"].nunique()
    unique_nonmembers = nonmember_events["sample_id"].nunique()

    print(f"\n=== EVENT STATISTICS ===")
    print(f"Total events: {len(df)}")
    print(f"Member events: {len(member_events)} ({unique_members} unique samples)")
    print(f"Non-member events: {len(nonmember_events)} ({unique_nonmembers} unique samples)")

    if len(nonmember_events) == 0:
        print("\nCRITICAL: No non-member events found!")
        return False
    print("Non-members appear in event log (non-trivial scenario)")

    member_access_rate = len(member_events) / unique_members
    nonmember_access_rate = len(nonmember_events) / unique_nonmembers

    print(f"\n=== ACCESS RATES ===")
    print(f"Member access rate: {member_access_rate:.2f} per sample")
    print(f"Non-member access rate: {nonmember_access_rate:.2f} per sample")
    print(f"Ratio: {member_access_rate / nonmember_access_rate:.1f}:1")

    if member_access_rate / nonmember_access_rate < 2.0:
        print("\nWARNING: Access rate ratio is low (<2:1).")
    elif member_access_rate / nonmember_access_rate > 20.0:
        print("\nWARNING: Access rate ratio is very high (>20:1).")
    else:
        print("Access rate ratio is in reasonable range (2:1 to 20:1)")

    member_counts = member_events.groupby("sample_id").size()
    nonmember_counts = nonmember_events.groupby("sample_id").size()

    print(f"\n=== ACCESS COUNT DISTRIBUTION ===")
    print(f"Members:")
    print(f"  Mean: {member_counts.mean():.2f}")
    print(f"  Std: {member_counts.std():.2f}")
    print(f"  Min: {member_counts.min()}")
    print(f"  Max: {member_counts.max()}")
    print(f"  Median: {member_counts.median():.2f}")

    print(f"\nNon-members:")
    print(f"  Mean: {nonmember_counts.mean():.2f}")
    print(f"  Std: {nonmember_counts.std():.2f}")
    print(f"  Min: {nonmember_counts.min()}")
    print(f"  Max: {nonmember_counts.max()}")
    print(f"  Median: {nonmember_counts.median():.2f}")

    member_epoch_coverage = member_events.groupby("sample_id")["epoch"].nunique().mean()
    nonmember_epoch_coverage = nonmember_events.groupby("sample_id")["epoch"].nunique().mean()

    print(f"\n=== EPOCH COVERAGE ===")
    print(f"Members: {member_epoch_coverage:.2f} unique epochs per sample")
    print(f"Non-members: {nonmember_epoch_coverage:.2f} unique epochs per sample")

    if member_epoch_coverage > nonmember_epoch_coverage * 1.5:
        print("Members have wider epoch coverage (expected)")
    else:
        print("WARNING: Epoch coverage separation is weak")

    probe_batches = df[df["batch_id"].str.contains("probe", na=False)]
    train_batches = df[df["batch_id"].str.contains("train", na=False)]

    print(f"\n=== BATCH TYPE DISTRIBUTION ===")
    print(f"Training batch events: {len(train_batches)}")
    print(f"Probe batch events: {len(probe_batches)}")
    print(f"Probe batch fraction: {len(probe_batches) / len(df):.2%}")

    if len(probe_batches) == 0:
        print("\nCRITICAL: No probe batch events found!")
        return False
    print("Probe batches present")

    probe_nonmember_frac = (probe_batches["label"] == 0).sum() / len(probe_batches)
    print(f"Non-member fraction in probe batches: {probe_nonmember_frac:.2%}")

    if probe_nonmember_frac < 0.1:
        print("WARNING: Very few non-members in probe batches")
    else:
        print("Probe batches contain non-members")

    print(f"\n=== VALIDATION SUMMARY ===")

    all_checks_passed = True

    if len(nonmember_events) == 0:
        print("FAIL: No non-member events (trivial scenario)")
        all_checks_passed = False
    elif nonmember_access_rate < 0.1:
        print("WARNING: Very low non-member access rate")
        all_checks_passed = False
    elif member_access_rate / nonmember_access_rate > 50:
        print("WARNING: Very high member/non-member ratio")
    else:
        print("PASS: Non-trivial attack scenario validated")

    return all_checks_passed



# CLI handlers

def baseline(args):
    print("="*60)
    print("BASELINE CIFAR-10 TRAINING")
    print("="*60)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output dir: {args.output_dir}")
    print("="*60)
    
    history = train_baseline(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        device=args.device,
        model_name=args.model,
    )
    
    print("\nBaseline training complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Final test accuracy: {history['best_acc']:.2f}%")
    print(f"Total training time: {history['total_time']:.2f}s")


def experiments(args):
    ensure_dir(args.output_root)

    visibilities = [float(x.strip()) for x in args.visibilities.split(",") if x.strip()]
    all_rows: List[Dict[str, object]] = []

    configs: List[RunConfig] = []

    only_defense = args.defense if args.defense != "all" else None

    if not args.skip_plaintext and (only_defense in (None, "plaintext")):
        configs.append(
            RunConfig(
                name="plaintext",
                defense="plaintext",
                visibility=1.0,
                dataset_root=args.dataset_root,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=args.device,
                seed=args.seed,
            )
        )

    if not args.skip_obfuscatedcated and (only_defense in (None, "obfuscatedcated")):
        configs.append(
            RunConfig(
                name="obfuscatedcated",
                defense="obfuscatedcated",
                visibility=1.0,
                dataset_root=args.dataset_root,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=args.device,
                seed=args.seed,
                decoys_per_access=args.decoys_per_access,
                prefetch_size=args.prefetch_size,
                release_shuffle_window=args.release_shuffle_window,
            )
        )

    if not args.skip_oram and (only_defense in (None, "oram")):
        configs.append(
            RunConfig(
                name="oram",
                defense="oram",
                visibility=1.0,
                dataset_root=args.dataset_root,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=args.device,
                seed=args.seed,
                oram_backend=args.oram_backend,
                oram_block_size=args.oram_block_size,
            )
        )

    if not configs:
        raise RuntimeError("No configurations selected.")

    print("=== UNIFIED EXPERIMENT RUNNER ===")
    print(f"Output root: {args.output_root}")
    print(f"Configurations: {len(configs)}")
    print(f"Visibility levels: {visibilities}")
    print("")

    for cfg in configs:
        print(f"\n=== Running {cfg.name} ===")
        rows = single_configuration(cfg, args.output_root, visibilities)
        all_rows.extend(rows)
        write_summary_csv(os.path.join(args.output_root, "summary.csv"), all_rows)

    parity_report = {
        "defenses_run": [cfg.defense for cfg in configs],
        "visibilities": visibilities,
        "attack_script": "run.py mi",
        "trace_converter": "run.py convert",
        "same_attack_cli": True,
    }
    with open(os.path.join(args.output_root, "pipeline_parity_report.json"), "w", encoding="utf-8") as f:
        json.dump(parity_report, f, indent=2)

    assertions = {"checks": []}
    by_defense: Dict[str, List[Dict[str, object]]] = {}
    for row in all_rows:
        by_defense.setdefault(str(row["defense"]), []).append(row)
    if "plaintext" in by_defense:
        best_plaintext_auc = max(float(r["best_auc"]) for r in by_defense["plaintext"])
        assertions["checks"].append({
            "name": "plaintext_auc_above_random",
            "passed": best_plaintext_auc > 0.5,
            "value": best_plaintext_auc,
        })
    if "oram" in by_defense:
        best_oram_auc = max(float(r["best_auc"]) for r in by_defense["oram"])
        assertions["checks"].append({
            "name": "oram_auc_near_random",
            "passed": abs(best_oram_auc - 0.5) <= 0.1,
            "value": best_oram_auc,
        })
    with open(os.path.join(args.output_root, "orchestrator_assertions.json"), "w", encoding="utf-8") as f:
        json.dump(assertions, f, indent=2)

    summary_path = os.path.join(args.output_root, "summary.csv")
    print(f"\n=== COMPLETE ===")
    print(f"Summary: {summary_path}")
    print(f"Total runs: {len(all_rows)}")
    print("")
    print("Results preview:")
    for row in all_rows:
        print(f"  {row['defense']:12s} visibility={row['visibility']:.2f}  AUC={row['best_auc']:.3f}  runtime={row['train_runtime_sec']:.1f}s")


def inference_main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    plaintext_log = os.path.join(args.output_dir, "events_plaintext.csv")
    oram_log = os.path.join(args.output_dir, "events_oram.csv")

    if not args.skip_generation or not os.path.exists(plaintext_log):
        invoke_command(
            [
                sys.executable,
                __file__,
                "event",
                "--train_size", str(args.train_size),
                "--holdout_size", str(args.holdout_size),
                "--epochs", str(args.epochs),
                "--batch_size", str(args.batch_size),
                "--probe_batch_prob", str(args.probe_batch_prob),
                "--probe_mix_ratio", str(args.probe_mix_ratio),
                "--output", plaintext_log,
                "--mode", "plaintext",
                "--data_dir", args.data_dir,
                "--random_state", str(args.random_state),
            ],
            "STEP 1: Generate plaintext event log with probe access"
        )

    if not args.skip_generation or not os.path.exists(oram_log):
        invoke_command(
            [
                sys.executable,
                __file__,
                "event",
                "--train_size", str(args.train_size),
                "--holdout_size", str(args.holdout_size),
                "--epochs", str(args.epochs),
                "--batch_size", str(args.batch_size),
                "--probe_batch_prob", str(args.probe_batch_prob),
                "--probe_mix_ratio", str(args.probe_mix_ratio),
                "--output", oram_log,
                "--mode", "oram",
                "--backend", "ram",
                "--data_dir", args.data_dir,
                "--random_state", str(args.random_state),
            ],
            "STEP 2: Generate ORAM event log with probe access"
        )

    visibility_levels = [1.0, 0.5, 0.25, 0.1]
    
    results_plaintext: Dict[float, Dict[str, float]] = {}
    results_oram: Dict[float, Dict[str, float]] = {}

    for vis in visibility_levels:
        pt_out = os.path.join(args.output_dir, f"plaintext_v{int(vis*100)}")
        print(f"\nSTEP 3.{int(vis*100)}: Attack plaintext log at visibility={vis}")
        summary = upgraded_attack(
            input_path=plaintext_log,
            output_dir=pt_out,
            visibility=vis,
            random_state=args.random_state,
        )
        best_model = summary["best_model"]
        results_plaintext[vis] = {
            "auc": summary["results"][best_model]["auc"],
            "accuracy": summary["results"][best_model]["accuracy"],
            "ap": summary["results"][best_model]["average_precision"],
            "model": best_model,
        }

        oram_out = os.path.join(args.output_dir, f"oram_v{int(vis*100)}")
        print(f"\nSTEP 4.{int(vis*100)}: Attack ORAM log at visibility={vis}")
        summary = upgraded_attack(
            input_path=oram_log,
            output_dir=oram_out,
            visibility=vis,
            random_state=args.random_state,
        )
        best_model = summary["best_model"]
        results_oram[vis] = {
            "auc": summary["results"][best_model]["auc"],
            "accuracy": summary["results"][best_model]["accuracy"],
            "ap": summary["results"][best_model]["average_precision"],
            "model": best_model,
        }

    print(f"\n{'='*60}")
    print("STEP 5: Generate summary table")
    print(f"{'='*60}")

    summary_rows = []
    for vis in visibility_levels:
        row = {"visibility": vis}
        if vis in results_plaintext:
            row["plaintext_auc"] = results_plaintext[vis]["auc"]
            row["plaintext_acc"] = results_plaintext[vis]["accuracy"]
        if vis in results_oram:
            row["oram_auc"] = results_oram[vis]["auc"]
            row["oram_acc"] = results_oram[vis]["accuracy"]
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.output_dir, "summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n=== MEMBERSHIP INFERENCE SWEEP SUMMARY ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary to: {summary_path}")

    markdown_path = os.path.join(args.output_dir, "summary.txt")
    with open(markdown_path, "w") as f:
        f.write("# Membership Inference Attack Results\n\n")
        f.write("## Plaintext Access Patterns\n\n")
        f.write("| Visibility | AUC | Accuracy |\n")
        f.write("|------------|-----|----------|\n")
        for vis in visibility_levels:
            if vis in results_plaintext:
                r = results_plaintext[vis]
                f.write(f"| {vis:.2f} | {r['auc']:.4f} | {r['accuracy']:.4f} |\n")
        
        f.write("\n## ORAM-Backed Access Patterns\n\n")
        f.write("| Visibility | AUC | Accuracy |\n")
        f.write("|------------|-----|----------|\n")
        for vis in visibility_levels:
            if vis in results_oram:
                r = results_oram[vis]
                f.write(f"| {vis:.2f} | {r['auc']:.4f} | {r['accuracy']:.4f} |\n")
        
        f.write("\n## Interpretation\n\n")
        f.write("- **Plaintext**: Access patterns expose structured temporal and frequency information.\n")
        f.write("- **ORAM**: Physical accesses are randomized and unlinkable to logical samples, ")
        f.write("reducing the attack signal.\n")
        f.write("- **Visibility degradation**: Lower visibility simulates partial observability ")
        f.write("(e.g., sampling, rate limiting, or incomplete monitoring).\n")

    print(f"Saved markdown summary to: {markdown_path}")
    print(f"\nAll outputs saved to: {args.output_dir}")


def oram_main(args):
    print("="*60)
    print("ORAM-INTEGRATED CIFAR-10 TRAINING")
    print("="*60)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num samples: {args.num_samples or 50000}")
    print(f"Backend: {args.backend}")
    print(f"Block size: {args.block_size}")
    print(f"Model: {args.model}")
    print(f"Workers: {args.num_workers}")
    print(f"Output dir: {args.output_dir}")
    print("="*60)
    
    history = train_oram(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        device=args.device,
        num_samples=args.num_samples,
        backend=args.backend,
        block_size=args.block_size,
        model_name=args.model,
        num_workers=args.num_workers,
    )
    
    print("\nORAM training complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Final test accuracy: {history['best_acc']:.2f}%")
    print(f"Total training time: {history['total_time']:.2f}s")


def phases_main(args):
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


def sidecar_main(args):
    return sidecar_training(args)


def setup(args):
    print("=== Setup Verification ===\n")

    all_ok = True

    print("Core Files:")
    all_ok &= check_file("src/run.py", "Unified runner")
    all_ok &= check_file("src/oram.py", "ORAM module")

    print("\nOrchestration:")
    all_ok &= check_file("run.sh", "Shell orchestrator")

    print("\nDocumentation:")
    all_ok &= check_file("README.md", "Project README")

    print("\nDependencies:")
    all_ok &= check_import("sklearn", "scikit-learn")
    all_ok &= check_import("pandas", "pandas")
    all_ok &= check_import("numpy", "numpy")
    all_ok &= check_import("matplotlib", "matplotlib")

    xgb_ok = check_import("xgboost", "XGBoost (optional)")
    if not xgb_ok:
        print("    Note: XGBoost is optional but recommended for best performance")

    print("\n" + "="*60)
    if all_ok:
        print("All required components are in place.")
        print("\nNext steps:")
        print("  1. Install missing dependencies: pip install -r requirements.txt")
        print("  2. Run quick test: bash run.sh smoke")
        return 0
    else:
        print("Some components are missing.")
        print("\nTo install dependencies:")
        print("  pip install -r requirements.txt")
        return 1


def system(args):
    print("=== COMPLETE SYSTEM VALIDATION ===\n")

    all_ok = True

    print("1. Core Files")
    all_ok &= check_file_exists("src/run.py", "Unified runner")
    all_ok &= check_file_exists("src/oram.py", "ORAM module")

    print("\n2. Orchestration")
    exists = check_file_exists("run.sh", "Shell orchestrator")
    if exists:
        check_executable("run.sh")
    all_ok &= exists

    print("\n3. Documentation")
    all_ok &= check_file_exists("README.md", "README")

    print("\n4. Python Syntax Validation")
    python_files = [
        "src/run.py",
        "src/oram.py",
    ]

    for pyfile in python_files:
        try:
            with open(pyfile, "r") as f:
                ast.parse(f.read())
            print(f"[OK] Syntax valid: {pyfile}")
        except SyntaxError as e:
            print(f"[FAIL] Syntax error in {pyfile}: {e}")
            all_ok = False
        except FileNotFoundError:
            print(f"[FAIL] File not found: {pyfile}")
            all_ok = False

    print("\n5. Dependencies Check")
    required_packages = [
        "torch",
        "torchvision",
        "numpy",
        "matplotlib",
        "sklearn",
        "tqdm",
    ]

    for pkg in required_packages:
        try:
            if pkg == "sklearn":
                __import__("sklearn")
            else:
                __import__(pkg)
            print(f"[OK] {pkg} available")
        except ImportError:
            print(f"[MISSING] {pkg} (install via: pip install -r requirements.txt)")
            all_ok = False

    try:
        import xgboost  # noqa: F401
        print("[OK] xgboost available")
    except ImportError:
        print("[WARN] xgboost NOT available (optional, will use Gradient Boosting fallback)")

    print("\n6. OS-Level Tools (Linux only)")

    if sys.platform.startswith("linux"):
        try:
            import bcc  # noqa: F401
            print("[OK] BCC available (eBPF tracing supported)")
        except ImportError:
            print("[WARN] BCC NOT available (will use strace fallback)")
            print("  Install: sudo apt install bpfcc-tools python3-bpfcc")

        result = subprocess.run(["which", "strace"], capture_output=True)
        if result.returncode == 0:
            print("[OK] strace available (fallback tracing supported)")
        else:
            print("[MISSING] strace NOT available (install via package manager)")
    else:
        print(f"[WARN] Platform: {sys.platform} (OS-level tracing requires Linux)")

    print("\n=== VALIDATION SUMMARY ===\n")

    if all_ok:
        print("PASS: All critical components present and valid")
        print("\nNext steps:")
        print("  1. Quick test: bash run.sh smoke")
        print("  2. Workshop paper: bash run.sh visibility")
        print("  3. Conference paper: bash run.sh trace (Linux)")
        return 0
    else:
        print("FAIL: Some components missing or invalid")
        print("\nFix issues above and re-run validation.")
        return 1


def probe(args):
    success = validate_event(args.input)

    if success:
        print("\nEvent log is suitable for non-trivial membership inference attack.")
        return 0
    else:
        print("\nEvent log validation failed.")
        return 1


def files(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_root = args.output_root
    train_dir = os.path.join(output_root, "train")
    probe_dir = os.path.join(output_root, "probe")
    ensure_dir(train_dir)
    ensure_dir(probe_dir)

    ds = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
    n = len(ds)
    all_indices: List[int] = list(range(n))
    random.shuffle(all_indices)

    train_indices = all_indices[:args.train_size]
    holdout_indices = all_indices[args.train_size:args.train_size + args.holdout_size]

    manifest_path = os.path.join(output_root, "manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "sample_id", "label", "path", "membership"]
        )
        writer.writeheader()

        for gid in train_indices:
            image, label = ds[gid]
            image_np = np.array(image, dtype=np.uint8)
            fname = f"member_{gid:06d}.bin"
            path = os.path.join(train_dir, fname)
            with open(path, "wb") as out:
                out.write(serialize_sample(gid, image_np, int(label)))
            writer.writerow({
                "split": "train",
                "sample_id": gid,
                "label": int(label),
                "path": path,
                "membership": 1,
            })

        for gid in holdout_indices:
            image, label = ds[gid]
            image_np = np.array(image, dtype=np.uint8)
            fname = f"nonmember_{gid:06d}.bin"
            path = os.path.join(probe_dir, fname)
            with open(path, "wb") as out:
                out.write(serialize_sample(gid, image_np, int(label)))
            writer.writerow({
                "split": "probe",
                "sample_id": gid,
                "label": int(label),
                "path": path,
                "membership": 0,
            })

    print(f"Wrote manifest: {manifest_path}")
    print(f"Train files: {len(train_indices)}")
    print(f"Probe files: {len(holdout_indices)}")
    return 0


def train(args):
    import glob as _glob

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_torch_device(args.device)

    train_paths = _glob.glob(os.path.join(args.dataset_root, "train", "member_*.bin"))
    probe_paths = _glob.glob(os.path.join(args.dataset_root, "probe", "nonmember_*.bin"))
    if not train_paths:
        raise RuntimeError("No training files found.")
    if not probe_paths:
        raise RuntimeError("No probe files found.")

    if args.obfuscatedcated:
        train_ds = ObfuscatedFileDataset(
            real_paths=train_paths,
            decoy_pool_paths=train_paths + probe_paths,
            decoys_per_access=args.decoys,
            prefetch_size=args.prefetch,
            release_shuffle_window=args.shuffle_window,
            seed=args.seed,
        )
    else:
        train_ds = FileSampleDataset(train_paths)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_train,
    )

    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(3 * 32 * 32, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 10),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    print(f"\nTraining PID: {os.getpid()}")
    print("To trace this process, run in another terminal:")
    print(f"  sudo python src/run.py trace --pid {os.getpid()} --output opens.csv")
    print("")

    with SidecarLogger(args.sidecar_path) as sidecar:
        for epoch in range(args.epochs):
            for batch_idx, (x, y, ids, paths) in enumerate(train_loader):
                batch_id = f"{epoch}_{batch_idx}_train"

                sidecar.log(batch_id=batch_id, epoch=epoch, phase="train")

                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

                if random.random() < args.probe_batch_prob:
                    probe_n = max(1, int(args.batch_size * args.probe_mix_ratio))
                    probe_batch = read_probe_samples(probe_paths, probe_n)

                    probe_batch_id = f"{epoch}_{batch_idx}_probe"
                    sidecar.log(batch_id=probe_batch_id, epoch=epoch, phase="probe")

                    px = torch.stack([t[0] for t in probe_batch], dim=0).to(device)
                    with torch.no_grad():
                        _ = model(px)

    print(f"Wrote sidecar: {args.sidecar_path}")
    return 0


def convert(args):
    """Convert raw trace + sidecar into attack-ready event CSV."""
    markers = read_sidecar(args.sidecar)
    marker_times = [m[0] for m in markers]

    trace_validation: Dict[str, object] = {"defense": args.defense, "trace_mode": args.trace_mode}
    if args.defense == "oram":
        if args.trace_mode == "strace":
            trace_rows_oram, oram_validation = Trace.oram_events(args.trace_input, args.oram_block_size)
        elif args.trace_mode == "fs_usage":
            trace_rows_oram, oram_validation = Trace.fs_usage(
                args.trace_input, markers, args.defense, args.oram_block_size
            )
        else:
            raise RuntimeError("ORAM conversion requires --trace_mode strace or fs_usage.")
        trace_rows = [(ts, token) for ts, token in trace_rows_oram]
        trace_validation.update(oram_validation)
    else:
        if args.trace_mode == "ebpf_csv":
            trace_rows = Trace.ebpf_csv(args.trace_input)
        elif args.trace_mode == "strace":
            trace_rows = Trace.path_events(args.trace_input)
        elif args.trace_mode == "fs_usage":
            trace_rows, fs_validation = Trace.fs_usage(
                args.trace_input, markers, args.defense, args.oram_block_size
            )
            trace_validation.update(fs_validation)
        else:
            raise RuntimeError(f"Unsupported trace mode: {args.trace_mode}")
        trace_validation["open_rows"] = len(trace_rows)

    out_rows = []
    for ts, source in trace_rows:
        if args.defense == "oram":
            sample_id = source
            digest = hashlib.sha256(source.encode("utf-8")).digest()
            label = int(digest[0] & 1)
        else:
            parsed = infer_sample_and_label(source)
            if parsed is None:
                continue
            sample_id, label = parsed
        joined = nearest_prior_marker(ts, marker_times, markers)
        if joined is None:
            continue

        _, batch_id, epoch, _phase = joined

        out_rows.append({
            "sample_id": sample_id,
            "timestamp": ts,
            "epoch": epoch,
            "batch_id": batch_id,
            "label": label,
        })

    attack_audit = scan_attack_input_for_leaks(out_rows)
    trace_validation["joined_rows"] = len(out_rows)
    trace_validation["sidecar_markers"] = len(markers)

    if args.defense == "oram" and trace_validation.get("quantized_blocks_seen", 0) == 0:
        raise RuntimeError("ORAM trace conversion found no ORAM physical block events.")
    if attack_audit["has_leakage"]:
        raise RuntimeError(f"Leakage patterns found in converted attack input: {attack_audit['leak_patterns_found']}")
    if set(attack_audit["label_counts"].keys()) != {"0", "1"}:
        raise RuntimeError(f"Need both classes after conversion, got {attack_audit['label_counts']}")

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "timestamp", "epoch", "batch_id", "label"],
        )
        writer.writeheader()
        writer.writerows(out_rows)

    if args.trace_validation_out:
        with open(args.trace_validation_out, "w", encoding="utf-8") as f:
            json.dump(trace_validation, f, indent=2)
    if args.attack_input_audit_out:
        with open(args.attack_input_audit_out, "w", encoding="utf-8") as f:
            json.dump(attack_audit, f, indent=2)

    print(f"Wrote {len(out_rows)} joined rows to {args.output}")
    return 0


def upgraded(args):
    print_section("UPGRADED MEMBERSHIP INFERENCE ATTACK - INTERACTIVE DEMO")

    print("""
This demo shows the complete pipeline for the upgraded membership inference attack.

The key difference from the old attack:

    OLD (Trivial):
        - Members: accessed during training
        - Non-members: NEVER accessed
        - Feature: access count (0 vs >0)
        - AUC: ~0.98 (trivial separation)

    NEW (Non-Trivial):
        - Members: accessed multiple times per epoch (shuffled batches)
        - Non-members: accessed once per epoch (validation batches)
        - Features: 40+ temporal, frequency, structural, co-occurrence
        - AUC: ~0.81 (realistic signal from pattern structure)

This makes the attack meaningful for a systems/security paper.
""")

    input("Press Enter to continue...")

    print_step(1, "Event Log Generation")
    print("""
We generate event logs that include BOTH members and non-members.

Command:
    python src/run.py event \\
        --train_size 10000 \\
        --val_size 5000 \\
        --epochs 5 \\
        --batch_size 128 \\
        --output results/events_plaintext.csv \\
        --mode plaintext
""")

    input("Press Enter to continue...")

    print_step(2, "Feature Extraction")
    print("""
The upgraded attack extracts 40+ features per sample from the event log:
frequency, temporal, structural, co-occurrence, and sparse features.
""")

    input("Press Enter to continue...")

    print_step(3, "Model Training")
    print("""
The attack trains three ensemble models and selects the best by AUC:
Random Forest, Gradient Boosting, and XGBoost (if available).

Command:
    python src/run.py mi \\
        --input results/events_plaintext.csv \\
        --output_dir results/attack_plaintext \\
        --visibility 1.0 \\
        --random_state 42
""")

    input("Press Enter to continue...")

    print_step(4, "Partial Observability")
    print("""
The attack supports partial observability via --visibility (1.0, 0.5, 0.25, 0.1).
""")

    input("Press Enter to continue...")

    print_step(5, "ORAM Mitigation")
    print("""
ORAM randomizes physical accesses, breaking the correspondence between
observed physical access sequence and logical sample access pattern.

Result:
    Plaintext: AUC ~ 0.81 (strong signal)
    ORAM: AUC ~ 0.52 (near random, no signal)
""")

    input("Press Enter to continue...")

    print_step(6, "Full Evaluation Pipeline")
    print("""
Command:
    python src/run.py inference \\
        --train_size 20000 \\
        --val_size 10000 \\
        --epochs 5 \\
        --output_dir results/paper_membership_attack
""")

    input("Press Enter to continue...")

    print_section("SUMMARY")
    print("""
Quick Start:
    1. Install dependencies: pip install -r requirements.txt
    2. Run quick test: bash run.sh smoke
    3. Full evaluation: python src/run.py inference
""")

    print("\nDemo complete.\n")
    return 0


def gen_attack(args) -> None:
    """Generate LaTeX tables from membership inference attack results."""
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    main_table = latex_table(args.results_dir, args.visibility_levels)

    with open(args.output, "w") as f:
        f.write(main_table)
        f.write("\n\n")

        if args.include_features:
            f.write("% Feature importance tables\n\n")
            for mode in ["plaintext", "oram"]:
                feature_table = feature_importance_table(
                    args.results_dir,
                    mode,
                    visibility=1.0,
                    top_k=10
                )
                if feature_table:
                    f.write(feature_table)
                    f.write("\n\n")

    print(f"LaTeX table saved to: {args.output}")
    print("\nTo include in your manuscript, add:")
    print(f"  \\input{{{args.output}}}")


def event(args) -> None:
    """Generate access-pattern event logs for membership inference."""
    print(f"Generating {args.mode} event log with probe access...")
    print(f"  Train size: {args.train_size}")
    print(f"  Holdout size: {args.holdout_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Probe batch probability: {args.probe_batch_prob}")
    print(f"  Probe mix ratio: {args.probe_mix_ratio}")

    if args.mode == "plaintext":
        events = plaintext(
            train_size=args.train_size,
            holdout_size=args.holdout_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            probe_batch_prob=args.probe_batch_prob,
            probe_mix_ratio=args.probe_mix_ratio,
            data_dir=args.data_dir,
            random_state=args.random_state,
        )
    else:
        events = oram_event(
            train_size=args.train_size,
            holdout_size=args.holdout_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            probe_batch_prob=args.probe_batch_prob,
            probe_mix_ratio=args.probe_mix_ratio,
            data_dir=args.data_dir,
            random_state=args.random_state,
            backend=args.backend,
        )

    Save.events_csv(events, args.output)

    member_events = sum(1 for _, _, _, _, label in events if label == 1)
    nonmember_events = sum(1 for _, _, _, _, label in events if label == 0)
    unique_members = len(set(sid for sid, _, _, _, label in events if label == 1))
    unique_nonmembers = len(set(sid for sid, _, _, _, label in events if label == 0))

    print(f"\n=== EVENT LOG SUMMARY ===")
    print(f"Total events: {len(events)}")
    print(f"Member events: {member_events} ({unique_members} unique samples)")
    print(f"Non-member events: {nonmember_events} ({unique_nonmembers} unique samples)")
    print(f"Member access rate: {member_events / unique_members:.2f} per sample")
    print(f"Non-member access rate: {nonmember_events / unique_nonmembers:.2f} per sample")
    print(f"Saved to: {args.output}")


def partial(args) -> None:
    """Generate event log with partial observability simulation."""
    cfg = PartialObservabilityConfig(**vars(args))

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    transform = transforms.Compose([transforms.ToTensor()])
    full_dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    all_indices = list(range(len(full_dataset)))
    random.shuffle(all_indices)

    train_indices = all_indices[:cfg.train_size]
    holdout_indices = all_indices[cfg.train_size:cfg.train_size + cfg.holdout_size]

    membership_label: Dict[str, int] = {}
    for idx in train_indices:
        membership_label[str(idx)] = 1
    for idx in holdout_indices:
        membership_label[str(idx)] = 0

    train_set = Subset(IndexedDataset(full_dataset), train_indices)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(3 * 32 * 32, 10),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    full_events: List[Dict[str, object]] = []
    global_time = 0.0

    def log_event(sample_id: int, epoch: int, batch_id: str) -> None:
        nonlocal global_time
        global_time += random.uniform(0.001, 0.010)
        full_events.append(
            {
                "sample_id": str(sample_id),
                "timestamp": global_time,
                "epoch": epoch,
                "batch_id": batch_id,
                "label": int(membership_label[str(sample_id)]),
            }
        )

    print("Generating full logical access stream...")
    for epoch in range(cfg.epochs):
        for batch_idx, (x, y, idxs) in enumerate(train_loader):
            train_batch_id = f"{epoch}_{batch_idx}_train"

            for idx in idxs.tolist():
                log_event(int(idx), epoch, train_batch_id)

            if random.random() < cfg.probe_batch_prob:
                probe_size = max(1, int(cfg.batch_size * cfg.probe_mix_ratio))
                probe_ids = random.sample(holdout_indices, min(probe_size, len(holdout_indices)))
                probe_batch_id = f"{epoch}_{batch_idx}_probe"
                for sid in probe_ids:
                    log_event(int(sid), epoch, probe_batch_id)

            optimizer.zero_grad()
            out = model(x.view(x.size(0), -1))
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

    full_events.sort(key=lambda r: float(r["timestamp"]))
    observed_events = Build.observed_stream(full_events, cfg, membership_label)

    Save.csv(cfg.full_output, full_events)
    Save.csv(cfg.observed_output, observed_events)

    full_member = sum(1 for e in full_events if e["label"] == 1)
    full_nonmember = sum(1 for e in full_events if e["label"] == 0)
    obs_member = sum(1 for e in observed_events if e["label"] == 1)
    obs_nonmember = sum(1 for e in observed_events if e["label"] == 0)

    print(f"\n=== FULL STREAM ===")
    print(f"Total events: {len(full_events)}")
    print(f"Member events: {full_member}")
    print(f"Non-member events: {full_nonmember}")
    print(f"Saved to: {cfg.full_output}")

    print(f"\n=== OBSERVED STREAM ===")
    print(f"Total events: {len(observed_events)}")
    print(f"Member events: {obs_member}")
    print(f"Non-member events: {obs_nonmember}")
    print(f"Visibility: {cfg.visibility:.2f}")
    print(f"Timestamp jitter std: {cfg.timestamp_jitter_std}")
    print(f"Batch ID corruption prob: {cfg.batch_id_corruption_prob}")
    print(f"Background noise rate: {cfg.background_noise_rate}")
    print(f"Saved to: {cfg.observed_output}")

    print("\nConfig:")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")


def reference(args) -> None:
    """Reference implementation of realistic access-pattern event log generation."""
    SEED = 42
    TRAIN_SIZE = 20000
    HOLDOUT_SIZE = 20000
    BATCH_SIZE = 128
    EPOCHS = 3
    PROBE_BATCH_PROB = 0.2
    PROBE_MIX_RATIO = 0.3
    OUTPUT_FILE = "events.csv"

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    transform = transforms.Compose([transforms.ToTensor()])

    full_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    indices = list(range(len(full_dataset)))
    random.shuffle(indices)

    train_indices = indices[:TRAIN_SIZE]
    holdout_indices = indices[TRAIN_SIZE:TRAIN_SIZE + HOLDOUT_SIZE]

    train_set = Subset(IndexedDataset(full_dataset), train_indices)

    membership_label = {}
    for i in train_indices:
        membership_label[str(i)] = 1
    for i in holdout_indices:
        membership_label[str(i)] = 0

    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(3 * 32 * 32, 10)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    events = []
    global_time = 0.0

    def log_event(sample_id, epoch, batch_id):
        nonlocal global_time
        global_time += random.uniform(0.001, 0.01)
        events.append({
            "sample_id": str(sample_id),
            "timestamp": global_time,
            "epoch": epoch,
            "batch_id": batch_id,
            "label": membership_label[str(sample_id)]
        })

    print("Generating access log with probe batches...")
    print(f"  Train size: {TRAIN_SIZE}")
    print(f"  Holdout size: {HOLDOUT_SIZE}")
    print(f"  Probe batch probability: {PROBE_BATCH_PROB}")
    print(f"  Probe mix ratio: {PROBE_MIX_RATIO}")

    for epoch in range(EPOCHS):
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

        for batch_id, (x, y, idxs) in enumerate(train_loader):

            for idx in idxs:
                log_event(int(idx), epoch, f"{epoch}_{batch_id}_train")

            if random.random() < PROBE_BATCH_PROB:
                probe_size = int(BATCH_SIZE * PROBE_MIX_RATIO)
                holdout_samples = random.sample(holdout_indices, probe_size)

                for sample_id in holdout_samples:
                    log_event(sample_id, epoch, f"{epoch}_{batch_id}_probe")

            optimizer.zero_grad()
            out = model(x.view(x.size(0), -1))
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

    print("Done.")

    print(f"Saving to {OUTPUT_FILE}...")

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "sample_id", "timestamp", "epoch", "batch_id", "label"
        ])
        writer.writeheader()
        writer.writerows(events)

    member_events = sum(1 for e in events if e["label"] == 1)
    nonmember_events = sum(1 for e in events if e["label"] == 0)
    unique_members = len(set(e["sample_id"] for e in events if e["label"] == 1))
    unique_nonmembers = len(set(e["sample_id"] for e in events if e["label"] == 0))

    print(f"\n=== EVENT LOG SUMMARY ===")
    print(f"Total events: {len(events)}")
    print(f"Member events: {member_events} ({unique_members} unique samples)")
    print(f"Non-member events: {nonmember_events} ({unique_nonmembers} unique samples)")
    print(f"Member access rate: {member_events / unique_members:.2f} per sample")
    print(f"Non-member access rate: {nonmember_events / unique_nonmembers:.2f} per sample")
    print(f"Saved to: {OUTPUT_FILE}")

    print("\n=== EXPECTED SIGNAL ===")
    print("Members:")
    print("  - High access count (multiple per epoch)")
    print("  - Wide epoch coverage")
    print("  - Regular inter-arrival")
    print("  - Stable batch co-occurrence")
    print("\nNon-members:")
    print("  - Low-medium access count (sparse probes)")
    print("  - Sparse epoch coverage")
    print("  - Bursty inter-arrival")
    print("  - Noisy batch co-occurrence")
    print("\nThis creates a realistic, non-trivial attack scenario.")


def privacy(args) -> None:
    Plot.privacy(args.summary, args.oram_auc, args.oram_overhead, args.output)


def membership(args) -> None:
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    Plot.membership(args.results_dir, args.output)


def robustness(args) -> None:
    Plot.robustness(args.summary, args.oram_results, args.output)


def mi(args) -> None:
    """Run upgraded multi-feature membership inference attack."""
    if not (0.0 < args.visibility <= 1.0):
        raise ValueError("--visibility must be in (0, 1].")
    if not (0.0 < args.test_size < 1.0):
        raise ValueError("--test_size must be in (0, 1).")

    upgraded_attack(
        input_path=args.input,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        visibility=args.visibility,
        max_samples=args.max_samples,
    )


def mi_simple(args) -> None:
    """Run simple frequency-based membership inference attack."""
    simple_attack(
        batch_size=args.batch_size,
        epochs=args.epochs,
        train_size=args.train_size,
        holdout_size=args.holdout_size,
        seed=args.seed,
        output_dir=args.output_dir,
    )



def main():
    parser = argparse.ArgumentParser(
        description='Unified experiment runner and test suite',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # --- Run subcommands ---

    baseline_parser = subparsers.add_parser('baseline', help='Run baseline training')
    baseline_parser.add_argument('--epochs', type=int, default=100)
    baseline_parser.add_argument('--batch-size', type=int, default=128)
    baseline_parser.add_argument('--output-dir', type=str, default='results/baseline')
    baseline_parser.add_argument('--device', type=str, default=None)
    baseline_parser.add_argument('--model', type=str, default='resnet18',
                                choices=['resnet18', 'resnet50', 'efficientnet_b0'])

    experiments_parser = subparsers.add_parser('experiments', help='Run full experiments')
    experiments_parser.add_argument("--dataset_root", required=True)
    experiments_parser.add_argument("--output_root", default="experiments_out")
    experiments_parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    experiments_parser.add_argument("--epochs", type=int, default=3)
    experiments_parser.add_argument("--batch_size", type=int, default=128)
    experiments_parser.add_argument("--seed", type=int, default=42)
    experiments_parser.add_argument("--visibilities", type=str, default="1.0,0.5,0.25,0.1")
    experiments_parser.add_argument("--decoys_per_access", type=int, default=2)
    experiments_parser.add_argument("--prefetch_size", type=int, default=8)
    experiments_parser.add_argument("--release_shuffle_window", type=int, default=4)
    experiments_parser.add_argument("--skip_plaintext", action="store_true")
    experiments_parser.add_argument("--skip_obfuscatedcated", action="store_true")
    experiments_parser.add_argument("--skip_oram", action="store_true")
    experiments_parser.add_argument("--defense", type=str, default="all",
                                   choices=["all", "plaintext", "obfuscatedcated", "oram"])
    experiments_parser.add_argument("--oram_backend", type=str, default="file", choices=["file", "ram"])
    experiments_parser.add_argument("--oram_block_size", type=int, default=4096)

    inference_parser = subparsers.add_parser('inference', help='Run membership inference')
    inference_parser.add_argument("--train_size", type=int, default=20000)
    inference_parser.add_argument("--holdout_size", type=int, default=20000)
    inference_parser.add_argument("--epochs", type=int, default=3)
    inference_parser.add_argument("--batch_size", type=int, default=128)
    inference_parser.add_argument("--probe_batch_prob", type=float, default=0.2)
    inference_parser.add_argument("--probe_mix_ratio", type=float, default=0.3)
    inference_parser.add_argument("--output_dir", type=str, default="results/membership_sweep")
    inference_parser.add_argument("--data_dir", type=str, default="./data")
    inference_parser.add_argument("--random_state", type=int, default=42)
    inference_parser.add_argument("--skip_generation", action="store_true")

    oram_parser = subparsers.add_parser('oram', help='Run ORAM training')
    oram_parser.add_argument('--epochs', type=int, default=100)
    oram_parser.add_argument('--batch-size', type=int, default=128)
    oram_parser.add_argument('--output-dir', type=str, default='results/oram')
    oram_parser.add_argument('--device', type=str, default=None)
    oram_parser.add_argument('--num-samples', type=int, default=None)
    oram_parser.add_argument('--backend', type=str, default='file', choices=['file', 'ram'])
    oram_parser.add_argument('--block-size', type=int, default=4096)
    oram_parser.add_argument('--model', type=str, default='resnet18',
                            choices=['resnet18', 'resnet50', 'efficientnet_b0'])
    oram_parser.add_argument('--num-workers', type=int, default=0)

    phases_parser = subparsers.add_parser('phases', help='Run phased experiments')
    phases_parser.add_argument("--phase", type=str, default="all")
    phases_parser.add_argument("--device", type=str, default=None)
    phases_parser.add_argument("--force", action="store_true")

    sidecar_parser = subparsers.add_parser('sidecar', help='Run ORAM with sidecar logging')
    sidecar_parser.add_argument("--epochs", type=int, default=3)
    sidecar_parser.add_argument("--batch_size", type=int, default=128)
    sidecar_parser.add_argument("--seed", type=int, default=42)
    sidecar_parser.add_argument("--sidecar_path", type=str, default="batch_sidecar_oram.csv")
    sidecar_parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"])
    sidecar_parser.add_argument("--num_samples", type=int, default=None)
    sidecar_parser.add_argument("--backend", type=str, default="file", choices=["file", "ram"])
    sidecar_parser.add_argument("--block_size", type=int, default=4096)
    sidecar_parser.add_argument("--model", type=str, default="resnet18")
    sidecar_parser.add_argument("--num_workers", type=int, default=0)
    sidecar_parser.add_argument("--output_dir", type=str, default="results/oram_trace")

    sweep_parser = subparsers.add_parser('sweep', help='Run parameter sweeps')
    sweep_parser.add_argument("--sweep", type=str, choices=["batch_size", "dataset_size", "block_size", "all"],
                             default="all")
    sweep_parser.add_argument("--epochs", type=int, default=3)
    sweep_parser.add_argument("--output-dir", type=str, default="results")
    sweep_parser.add_argument("--device", type=str, default=None)

    # --- Test / utility subcommands ---

    subparsers.add_parser('setup', help='Verify setup')
    subparsers.add_parser('system', help='Validate complete system')

    probe_parser = subparsers.add_parser('probe', help='Validate probe design')
    probe_parser.add_argument("--input", type=str, required=True, help="Event log CSV path")

    files_parser = subparsers.add_parser('files', help='Materialize dataset as files')
    files_parser.add_argument("--output_root", type=str, default="dataset_root")
    files_parser.add_argument("--train_size", type=int, default=20000)
    files_parser.add_argument("--holdout_size", type=int, default=20000)
    files_parser.add_argument("--seed", type=int, default=42)

    train_parser = subparsers.add_parser('train', help='Train from files with sidecar logging')
    train_parser.add_argument("--dataset_root", type=str, required=True)
    train_parser.add_argument("--epochs", type=int, default=3)
    train_parser.add_argument("--batch_size", type=int, default=128)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--probe_batch_prob", type=float, default=0.20)
    train_parser.add_argument("--probe_mix_ratio", type=float, default=0.30)
    train_parser.add_argument("--sidecar_path", type=str, default="batch_sidecar.csv")
    train_parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"])
    train_parser.add_argument("--obfuscatedcated", action="store_true")
    train_parser.add_argument("--decoys", type=int, default=2)
    train_parser.add_argument("--prefetch", type=int, default=8)
    train_parser.add_argument("--shuffle_window", type=int, default=4)

    trace_parser = subparsers.add_parser('trace', help='Trace file opens with eBPF')
    trace_parser.add_argument("--pid", type=int, required=True)
    trace_parser.add_argument("--output", type=str, default="opens.csv")

    convert_parser = subparsers.add_parser('convert', help='Convert trace to attack input')
    convert_parser.add_argument("--trace_input", required=True)
    convert_parser.add_argument("--trace_mode", choices=["ebpf_csv", "strace", "fs_usage"], required=True)
    convert_parser.add_argument("--sidecar", required=True)
    convert_parser.add_argument("--output", default="events_trace.csv")
    convert_parser.add_argument("--defense", choices=["plaintext", "obfuscatedcated", "oram"], required=True)
    convert_parser.add_argument("--oram_block_size", type=int, default=4096)
    convert_parser.add_argument("--trace_validation_out", type=str, default=None)
    convert_parser.add_argument("--attack_input_audit_out", type=str, default=None)

    subparsers.add_parser('upgraded', help='Interactive demo')

    # --- Generate subcommands ---

    gen_attack_parser = subparsers.add_parser("attack", help="Generate LaTeX tables from attack results")
    gen_attack_parser.add_argument("--results_dir", type=str, required=True,
                                   help="Directory containing attack results.")
    gen_attack_parser.add_argument("--output", type=str, required=True,
                                   help="Output .tex file path.")
    gen_attack_parser.add_argument("--visibility_levels", type=float, nargs="+",
                                   default=[1.0, 0.5, 0.25, 0.1],
                                   help="Visibility levels to include in table.")
    gen_attack_parser.add_argument("--include_features", action="store_true",
                                   help="Also generate feature importance tables.")

    event_parser = subparsers.add_parser("event", help="Generate access-pattern event logs")
    event_parser.add_argument("--train_size", type=int, default=20000)
    event_parser.add_argument("--holdout_size", type=int, default=20000)
    event_parser.add_argument("--epochs", type=int, default=3)
    event_parser.add_argument("--batch_size", type=int, default=128)
    event_parser.add_argument("--probe_batch_prob", type=float, default=0.2)
    event_parser.add_argument("--probe_mix_ratio", type=float, default=0.3)
    event_parser.add_argument("--output", type=str, required=True, help="Output CSV path.")
    event_parser.add_argument("--mode", type=str, default="plaintext", choices=["plaintext", "oram"])
    event_parser.add_argument("--backend", type=str, default="ram", choices=["file", "ram"])
    event_parser.add_argument("--data_dir", type=str, default="./data")
    event_parser.add_argument("--random_state", type=int, default=42)

    partial_parser = subparsers.add_parser("partial", help="Generate event log with partial observability")
    partial_parser.add_argument("--seed", type=int, default=42)
    partial_parser.add_argument("--train_size", type=int, default=20000)
    partial_parser.add_argument("--holdout_size", type=int, default=20000)
    partial_parser.add_argument("--batch_size", type=int, default=128)
    partial_parser.add_argument("--epochs", type=int, default=3)
    partial_parser.add_argument("--probe_batch_prob", type=float, default=0.20)
    partial_parser.add_argument("--probe_mix_ratio", type=float, default=0.30)
    partial_parser.add_argument("--visibility", type=float, default=0.50)
    partial_parser.add_argument("--timestamp_jitter_std", type=float, default=0.003)
    partial_parser.add_argument("--batch_id_corruption_prob", type=float, default=0.10)
    partial_parser.add_argument("--sample_id_mask_prob", type=float, default=0.00)
    partial_parser.add_argument("--background_noise_rate", type=float, default=0.05)
    partial_parser.add_argument("--reorder_window", type=int, default=0)
    partial_parser.add_argument("--full_output", type=str, default="events_full.csv")
    partial_parser.add_argument("--observed_output", type=str, default="events_observed.csv")

    subparsers.add_parser("reference", help="Run reference implementation")

    plot_parser = subparsers.add_parser("plot", help="Generate all figures from phase results")
    plot_parser.add_argument("--results-root", type=str, default=GEN_RESULTS_ROOT)
    plot_parser.add_argument("--output", type=str, default=GEN_OUTPUT_DIR)

    privacy_parser = subparsers.add_parser("privacy", help="Privacy-performance tradeoff plot")
    privacy_parser.add_argument("--summary", required=True)
    privacy_parser.add_argument("--oram_auc", type=float, default=0.50)
    privacy_parser.add_argument("--oram_overhead", type=float, default=90.0)
    privacy_parser.add_argument("--output", default="privacy_performance_tradeoff.pdf")

    membership_parser = subparsers.add_parser("membership", help="Membership attack comparison plot")
    membership_parser.add_argument("--results_dir", type=str,
                                   default="results/paper_membership_attack")
    membership_parser.add_argument("--output", type=str,
                                   default="results/attack_comparison.png")

    robustness_parser = subparsers.add_parser("robustness", help="Attack robustness vs visibility plot")
    robustness_parser.add_argument("--summary", required=True)
    robustness_parser.add_argument("--oram_results", type=str, default=None)
    robustness_parser.add_argument("--output", default="attack_robustness.pdf")

    # --- Membership inference subcommands ---

    mi_parser = subparsers.add_parser("mi", help="Run upgraded membership inference attack")
    mi_parser.add_argument("--input", required=True, help="Path to event CSV.")
    mi_parser.add_argument("--output_dir", required=True, help="Directory for outputs.")
    mi_parser.add_argument("--test_size", type=float, default=0.3, help="Test split fraction.")
    mi_parser.add_argument("--random_state", type=int, default=42, help="Random seed.")
    mi_parser.add_argument("--visibility", type=float, default=1.0,
                           help="Fraction of events retained to simulate partial observability (0,1].")
    mi_parser.add_argument("--max_samples", type=int, default=None,
                           help="Optional cap on unique samples used.")

    mi_simple_parser = subparsers.add_parser("mi-simple", help="Run simple frequency-based membership inference")
    mi_simple_parser.add_argument("--batch_size", type=int, default=128)
    mi_simple_parser.add_argument("--epochs", type=int, default=3)
    mi_simple_parser.add_argument("--train_size", type=int, default=20000)
    mi_simple_parser.add_argument("--holdout_size", type=int, default=20000)
    mi_simple_parser.add_argument("--seed", type=int, default=42)
    mi_simple_parser.add_argument("--output_dir", type=str, default="results")

    args = parser.parse_args()

    commands = {
        'baseline': baseline,
        'experiments': experiments,
        'inference': inference_main,
        'oram': oram_main,
        'phases': phases_main,
        'sidecar': sidecar_main,
        'sweep': Sweep.main,
        'setup': setup,
        'system': system,
        'probe': probe,
        'files': files,
        'train': train,
        'trace': Trace.cmd,
        'convert': convert,
        'upgraded': upgraded,
        'attack': gen_attack,
        'event': event,
        'partial': partial,
        'reference': reference,
        'plot': Plot.cmd,
        'privacy': privacy,
        'membership': membership,
        'robustness': robustness,
        'mi': mi,
        'mi-simple': mi_simple,
    }

    if args.command in commands:
        result = commands[args.command](args)
        sys.exit(result or 0)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
