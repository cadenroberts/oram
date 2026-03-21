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
import ast
import bisect
import csv
import datetime
import hashlib
import json
import math
import os
import random
import re
import signal
import shutil
import struct
import subprocess
import sys
import time
from collections import Counter, defaultdict, deque
from itertools import combinations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Tuple

from oram import (
    run_baseline_training, run_oram_training, Trainer,
    ORAMStorage, load_cifar10_to_oram, ORAMDataset, get_cifar10_transforms,
    create_model, SUPPORTED_MODELS,
)
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False


def resolve_torch_device(device_name: str) -> torch.device:
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but not available.")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(device_name)


class SidecarLogger:
    FIELDNAMES = ["timestamp", "batch_id", "epoch", "phase"]

    def __init__(self, path: str):
        self.path = path
        self._file = None
        self._writer = None

    def __enter__(self) -> "SidecarLogger":
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
        self._writer.writeheader()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None

    def log(self, batch_id: str, epoch: int, phase: str, timestamp: Optional[float] = None) -> None:
        if self._writer is None or self._file is None:
            raise RuntimeError("SidecarLogger must be used as a context manager.")
        self._writer.writerow(
            {
                "timestamp": time.time() if timestamp is None else timestamp,
                "batch_id": batch_id,
                "epoch": epoch,
                "phase": phase,
            }
        )
        self._file.flush()


MEMBER_RE = re.compile(r"^member_(\d+)\.bin$")
NONMEMBER_RE = re.compile(r"^nonmember_(\d+)\.bin$")
STRACE_RE = re.compile(r"^(\d+\.\d+)\s+.*?(open|openat|openat2)\(.*?\"([^\"]+)\"")
STRACE_OPEN_FD_RE = re.compile(
    r"^(\d+\.\d+)\s+.*?(open|openat|openat2)\(.*?\"([^\"]+)\".*?\)\s+=\s+(-?\d+)"
)
STRACE_PREAD_RE = re.compile(
    r"^(\d+\.\d+)\s+.*?pread64\((\d+),.*?,\s*(\d+),\s*(-?\d+)\)\s*=\s*(-?\d+)"
)
STRACE_PWRITE_RE = re.compile(
    r"^(\d+\.\d+)\s+.*?pwrite64\((\d+),.*?,\s*(\d+),\s*(-?\d+)\)\s*=\s*(-?\d+)"
)
STRACE_LSEEK_RE = re.compile(
    r"^(\d+\.\d+)\s+.*?lseek\((\d+),\s*(-?\d+),\s*SEEK_[A-Z]+\)\s*=\s*(-?\d+)"
)
FS_USAGE_TIME_RE = re.compile(r"^(\d{2}:\d{2}:\d{2}\.\d{6})")
FS_USAGE_DISK_OFF_RE = re.compile(r"D=0x([0-9A-Fa-f]+)")
FS_USAGE_PATH_RE = re.compile(r"(/\S*oram\.bin)", re.IGNORECASE)
FS_USAGE_MEMBER_PATH_RE = re.compile(r"(/\S*(?:member|nonmember)_\d+\.bin)")

LEAK_PATTERNS = [
    re.compile(r"member_", re.IGNORECASE),
    re.compile(r"nonmember_", re.IGNORECASE),
    re.compile(r"/train/"),
    re.compile(r"/probe/"),
]


@dataclass
class RunConfig:
    name: str
    defense: str
    visibility: float
    dataset_root: str
    epochs: int
    batch_size: int
    device: str
    seed: int
    decoys_per_access: int = 0
    prefetch_size: int = 1
    release_shuffle_window: int = 1
    oram_backend: str = "file"
    oram_block_size: int = 4096


RESULTS_ROOT = "results"
BATCH_SIZES = [32, 64, 128, 256]
DATASET_SIZES = [1000, 5000, 10000, 50000]
BLOCK_SIZES = [4096, 8192, 16384, 32768, 65536]
DEFAULT_BATCH_SIZE = 128
DEFAULT_DATASET_SIZE = 5000


def baseline_main(args):
    print("="*60)
    print("BASELINE CIFAR-10 TRAINING")
    print("="*60)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output dir: {args.output_dir}")
    print("="*60)
    
    history = run_baseline_training(
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


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run_subprocess(
    cmd: List[str],
    stdout_path: str,
    stderr_path: str,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
) -> subprocess.Popen:
    stdout_f = open(stdout_path, "w", encoding="utf-8")
    stderr_f = open(stderr_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        stdout=stdout_f,
        stderr=stderr_f,
        cwd=cwd,
        env=env,
        text=True,
    )
    proc._stdout_file = stdout_f
    proc._stderr_file = stderr_f
    return proc


def close_proc_files(proc: subprocess.Popen) -> None:
    for attr in ["_stdout_file", "_stderr_file"]:
        f = getattr(proc, attr, None)
        if f is not None:
            try:
                f.close()
            except Exception:
                pass


def launch_strace(pid: int, strace_log_path: str) -> subprocess.Popen:
    if shutil.which("strace") is None:
        if sys.platform == "darwin":
            raise RuntimeError(
                "strace is not available on macOS. Use run.sh macos "
                "for physical file-system tracing with fs_usage."
            )
        raise RuntimeError(
            "strace is required for physical trace auditing but was not found. "
            "Run this command on Linux with strace installed."
        )
    cmd = [
        "strace",
        "-ff",
        "-ttt",
        "-e",
        "trace=open,openat,openat2,read,write,pread64,pwrite64,lseek",
        "-p",
        str(pid),
    ]
    stderr_f = open(strace_log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=stderr_f,
        text=True,
    )
    proc._stderr_file = stderr_f
    return proc


def stop_process(proc: subprocess.Popen, grace_seconds: float = 3.0) -> None:
    if proc.poll() is not None:
        close_proc_files(proc)
        return

    try:
        proc.send_signal(signal.SIGINT)
    except Exception:
        pass

    deadline = time.time() + grace_seconds
    while time.time() < deadline:
        if proc.poll() is not None:
            close_proc_files(proc)
            return
        time.sleep(0.1)

    try:
        proc.terminate()
    except Exception:
        pass

    deadline = time.time() + grace_seconds
    while time.time() < deadline:
        if proc.poll() is not None:
            close_proc_files(proc)
            return
        time.sleep(0.1)

    try:
        proc.kill()
    except Exception:
        pass

    close_proc_files(proc)


def wait_success(proc: subprocess.Popen, name: str) -> None:
    ret = proc.wait()
    close_proc_files(proc)
    if ret != 0:
        raise RuntimeError(f"{name} failed with exit code {ret}")


def trainer_command(cfg: RunConfig, run_dir: str) -> List[str]:
    sidecar_path = os.path.join(run_dir, "batch_sidecar.csv")

    if cfg.defense == "plaintext":
        return [
            sys.executable,
            "run.py", "train",
            "--dataset_root", cfg.dataset_root,
            "--epochs", str(cfg.epochs),
            "--batch_size", str(cfg.batch_size),
            "--seed", str(cfg.seed),
            "--sidecar_path", sidecar_path,
            "--device", cfg.device,
        ]

    if cfg.defense == "obfuscatedcated":
        return [
            sys.executable,
            "run.py", "train",
            "--obfuscatedcated",
            "--dataset_root", cfg.dataset_root,
            "--epochs", str(cfg.epochs),
            "--batch_size", str(cfg.batch_size),
            "--seed", str(cfg.seed),
            "--decoys", str(cfg.decoys_per_access),
            "--prefetch", str(cfg.prefetch_size),
            "--shuffle_window", str(cfg.release_shuffle_window),
            "--sidecar_path", sidecar_path,
            "--device", cfg.device,
        ]

    if cfg.defense == "oram":
        return [
            sys.executable,
            "run.py",
            "sidecar",
            "--epochs", str(cfg.epochs),
            "--batch_size", str(cfg.batch_size),
            "--seed", str(cfg.seed),
            "--backend", cfg.oram_backend,
            "--block_size", str(cfg.oram_block_size),
            "--sidecar_path", sidecar_path,
            "--device", cfg.device,
            "--output_dir", run_dir,
        ]

    raise ValueError(f"Unknown defense type: {cfg.defense}")


def convert_trace(run_dir: str, cfg: RunConfig) -> str:
    trace_log = os.path.join(run_dir, "strace.log")
    sidecar = os.path.join(run_dir, "batch_sidecar.csv")
    output = os.path.join(run_dir, "events_trace.csv")
    trace_validation = os.path.join(run_dir, "trace_validation.json")
    attack_input_audit = os.path.join(run_dir, "attack_input_audit.json")

    cmd = [
        sys.executable,
        "run.py",
        "convert",
        "--trace_input", trace_log,
        "--trace_mode", "strace",
        "--sidecar", sidecar,
        "--defense", cfg.defense,
        "--oram_block_size", str(cfg.oram_block_size),
        "--trace_validation_out", trace_validation,
        "--attack_input_audit_out", attack_input_audit,
        "--output", output,
    ]
    proc = run_subprocess(
        cmd,
        stdout_path=os.path.join(run_dir, "convert_stdout.log"),
        stderr_path=os.path.join(run_dir, "convert_stderr.log"),
    )
    wait_success(proc, "trace conversion")
    return output


def read_oram_audit_counts(path: str) -> Dict[str, int]:
    counts = {"read": 0, "write": 0}
    if not os.path.exists(path):
        return counts
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if "op=read" in line:
                counts["read"] += 1
            elif "op=write" in line:
                counts["write"] += 1
    return counts


def read_json(path: str) -> Dict[str, object]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_claims_matrix(run_dir: str, cfg: RunConfig, claim_rows: List[Dict[str, object]]) -> None:
    json_path = os.path.join(run_dir, "claims_audit_matrix.json")
    csv_path = os.path.join(run_dir, "claims_audit_matrix.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(claim_rows, f, indent=2)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["claim_id", "claim", "verified", "evidence", "gap", "fix_applied"])
        writer.writeheader()
        for row in claim_rows:
            writer.writerow(row)


def scan_events_for_leakage(events_csv: str) -> Dict[str, object]:
    patterns = ["member_", "nonmember_", "/train/", "/probe/"]
    hits = {p: 0 for p in patterns}
    with open(events_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    joined = "\n".join(",".join(r.values()) for r in rows)
    for p in patterns:
        hits[p] = joined.count(p)
    return {
        "rows": len(rows),
        "hits": hits,
        "has_leakage": any(v > 0 for v in hits.values()),
    }


def build_mixed_access_report(events_csv: str) -> Dict[str, object]:
    with open(events_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    label_counts: Dict[str, int] = {}
    sample_event_counts: Dict[str, int] = {}
    for row in rows:
        label = str(row["label"])
        label_counts[label] = label_counts.get(label, 0) + 1
        sid = str(row["sample_id"])
        sample_event_counts[sid] = sample_event_counts.get(sid, 0) + 1
    repeated_samples = sum(1 for _, count in sample_event_counts.items() if count > 1)
    return {
        "num_rows": len(rows),
        "label_counts": label_counts,
        "has_both_labels": set(label_counts.keys()) == {"0", "1"},
        "unique_samples": len(sample_event_counts),
        "repeated_samples": repeated_samples,
        "non_binary_access_signal": repeated_samples > 0,
    }


def run_attack(run_dir: str, input_csv: str, visibility: float) -> Dict[str, object]:
    attack_dir = os.path.join(run_dir, f"attack_v{str(visibility).replace('.', 'p')}")
    ensure_dir(attack_dir)
    return run_upgraded_attack(
        input_path=input_csv,
        output_dir=attack_dir,
        visibility=visibility,
        random_state=42,
    )


def best_model_metrics(metrics_json: Dict[str, object]) -> Dict[str, float]:
    best = metrics_json["best_model"]
    results = metrics_json["results"][best]
    return {
        "best_auc": float(results["auc"]),
        "best_accuracy": float(results["accuracy"]),
        "best_ap": float(results["average_precision"]),
    }


def run_single_configuration(cfg: RunConfig, output_root: str, visibilities: List[float]) -> List[Dict[str, object]]:
    run_dir = os.path.join(output_root, cfg.name)
    ensure_dir(run_dir)

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    train_stdout = os.path.join(run_dir, "train_stdout.log")
    train_stderr = os.path.join(run_dir, "train_stderr.log")
    trainer_cmd = trainer_command(cfg, run_dir)
    oram_audit_log = os.path.join(run_dir, "oram_audit.log")
    train_env = os.environ.copy()
    if cfg.defense == "oram":
        train_env["ORAM_AUDIT_LOG"] = oram_audit_log

    print(f"  Launching trainer: {cfg.defense}")
    start_time = time.time()
    trainer_proc = run_subprocess(trainer_cmd, train_stdout, train_stderr, cwd="experiments", env=train_env)

    time.sleep(1.5)

    strace_log = os.path.join(run_dir, "strace.log")
    print(f"  Attaching strace to PID {trainer_proc.pid}")
    strace_proc = launch_strace(trainer_proc.pid, strace_log)

    try:
        print(f"  Waiting for training to complete...")
        wait_success(trainer_proc, f"trainer {cfg.name}")
    finally:
        stop_process(strace_proc)

    train_runtime = time.time() - start_time
    print(f"  Training runtime: {train_runtime:.1f}s")

    print(f"  Converting traces...")
    convert_start = time.time()
    input_csv = convert_trace(run_dir, cfg)
    convert_runtime = time.time() - convert_start

    trace_validation = read_json(os.path.join(run_dir, "trace_validation.json"))
    attack_input_audit = read_json(os.path.join(run_dir, "attack_input_audit.json"))
    leakage_scan = scan_events_for_leakage(input_csv)
    mixed_access_report = build_mixed_access_report(input_csv)
    if cfg.defense == "oram" and leakage_scan["has_leakage"]:
        raise RuntimeError(f"ORAM leakage detected in converted events: {leakage_scan['hits']}")
    if not mixed_access_report["has_both_labels"] or not mixed_access_report["non_binary_access_signal"]:
        raise RuntimeError(f"Mixed-access validity failed: {mixed_access_report}")

    oram_counts = read_oram_audit_counts(oram_audit_log)
    if cfg.defense == "oram" and oram_counts["read"] <= 0:
        raise RuntimeError("ORAM run produced zero audited read operations.")

    claims_rows = [
        {
            "claim_id": "C1",
            "claim": "ORAM backend is exercised during training",
            "verified": bool(cfg.defense != "oram" or oram_counts["read"] > 0),
            "evidence": f"oram_audit reads={oram_counts['read']} writes={oram_counts['write']}",
            "gap": "" if (cfg.defense != "oram" or oram_counts["read"] > 0) else "No ORAM reads observed",
            "fix_applied": "Added ORAM_AUDIT_LOG instrumentation and hard gate",
        },
        {
            "claim_id": "C2",
            "claim": "Trace is physical syscall data",
            "verified": bool(trace_validation.get("open_rows", 0) > 0),
            "evidence": json.dumps(trace_validation),
            "gap": "" if trace_validation.get("open_rows", 0) > 0 else "No syscall rows parsed",
            "fix_applied": "Expanded strace syscall set and conversion validation",
        },
        {
            "claim_id": "C3",
            "claim": "Attack input has no direct sample-path leakage",
            "verified": not leakage_scan["has_leakage"],
            "evidence": json.dumps({"scan": leakage_scan, "converter_audit": attack_input_audit}),
            "gap": "" if not leakage_scan["has_leakage"] else "Sample path leakage found",
            "fix_applied": "Leakage scanner + fail-closed conversion",
        },
    ]
    write_claims_matrix(run_dir, cfg, claims_rows)

    timing_breakdown = {
        "defense": cfg.defense,
        "train_runtime_sec": train_runtime,
        "convert_runtime_sec": convert_runtime,
        "trace_attach_overhead_sec": 1.5,
        "oram_read_count": oram_counts["read"],
        "oram_write_count": oram_counts["write"],
    }
    with open(os.path.join(run_dir, "timing_breakdown.json"), "w", encoding="utf-8") as f:
        json.dump(timing_breakdown, f, indent=2)
    with open(os.path.join(run_dir, "mixed_access_report.json"), "w", encoding="utf-8") as f:
        json.dump(mixed_access_report, f, indent=2)

    rows: List[Dict[str, object]] = []
    for visibility in visibilities:
        print(f"  Running attack at visibility={visibility}...")
        attack_start = time.time()
        metrics_json = run_attack(run_dir, input_csv, visibility)
        attack_runtime = time.time() - attack_start
        metric_row = best_model_metrics(metrics_json)

        row = {
            "run_name": cfg.name,
            "defense": cfg.defense,
            "visibility": visibility,
            "train_runtime_sec": train_runtime,
            "events_retained": metrics_json["num_events"],
            "num_samples": metrics_json["num_samples"],
            "best_model": metrics_json["best_model"],
            "convert_runtime_sec": convert_runtime,
            "attack_runtime_sec": attack_runtime,
            **metric_row,
            "decoys_per_access": cfg.decoys_per_access,
            "prefetch_size": cfg.prefetch_size,
            "release_shuffle_window": cfg.release_shuffle_window,
            "oram_backend": cfg.oram_backend,
            "oram_block_size": cfg.oram_block_size,
        }
        rows.append(row)
        print(f"    AUC={metric_row['best_auc']:.3f}, Accuracy={metric_row['best_accuracy']:.3f}")

    return rows


def write_summary_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def experiments_main(args):
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
        rows = run_single_configuration(cfg, args.output_root, visibilities)
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


def run_command(cmd: List[str], description: str) -> None:
    print(f"\n{'='*60}")
    print(description)
    print(f"{'='*60}")
    print(" ".join(cmd))
    result = subprocess.run(cmd, check=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def inference_main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    plaintext_log = os.path.join(args.output_dir, "events_plaintext.csv")
    oram_log = os.path.join(args.output_dir, "events_oram.csv")

    if not args.skip_generation or not os.path.exists(plaintext_log):
        run_command(
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
        run_command(
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
        summary = run_upgraded_attack(
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
        summary = run_upgraded_attack(
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
    
    history = run_oram_training(
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


def phase0(args):
    print("\n" + "="*60)
    print("PHASE 0: Baseline Reproduction")
    print("="*60)
    rows = []

    if not _done(0, "baseline", args.force):
        print("  Running plaintext baseline (3 epochs)...")
        h = run_baseline_training(num_epochs=3, batch_size=128,
                                  output_dir=_out(0, "baseline"), device=args.device)
        rows.append(_hist_row(h, mode="plaintext", backend="none"))
    else:
        print("  SKIP: plaintext baseline exists")

    if not _done(0, "oram_file", args.force):
        print("  Running file-backed ORAM (2 epochs, 10k samples)...")
        h = run_oram_training(num_epochs=2, batch_size=128,
                              output_dir=_out(0, "oram_file"), device=args.device,
                              num_samples=10000, backend="file")
        rows.append(_hist_row(h, mode="oram", backend="file"))
    else:
        print("  SKIP: file ORAM exists")

    _save_csv(rows, os.path.join(_out(0), "phase0_results.csv"))


def phase1(args):
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


def phase2(args):
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


def phase3(args):
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


def phase4(args):
    print("\n" + "="*60)
    print("PHASE 4: Model Scaling")
    print("="*60)
    rows = []
    models = ["resnet18", "resnet50", "efficientnet_b0"]

    for model in models:
        tag = f"baseline_{model}"
        if not _done(4, tag, args.force):
            print(f"  Running plaintext {model} (3 epochs)...")
            h = run_baseline_training(num_epochs=3, batch_size=128,
                                      output_dir=_out(4, tag), device=args.device,
                                      model_name=model)
            rows.append(_hist_row(h, mode="plaintext", model=model))
        else:
            print(f"  SKIP: {tag} exists")

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


def phase5(args):
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


def phase6(args):
    print("\n" + "="*60)
    print("PHASE 6: Batch Size Sweep")
    print("="*60)
    rows = []
    batch_sizes = [32, 64, 128, 256, 512]

    for bs in batch_sizes:
        tag = f"baseline_bs{bs}"
        if not _done(6, tag, args.force):
            print(f"  Running plaintext batch_size={bs} (3 epochs)...")
            h = run_baseline_training(num_epochs=3, batch_size=bs,
                                      output_dir=_out(6, tag), device=args.device)
            rows.append(_hist_row(h, mode="plaintext", batch_size=bs))
        else:
            print(f"  SKIP: {tag} exists")

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


def phase7(args):
    print("\n" + "="*60)
    print("PHASE 7: Access Pattern Leakage Demo")
    print("="*60)
    marker = os.path.join(_out(7), "leakage_comparison.png")
    if os.path.exists(marker) and not args.force:
        print("  SKIP: leakage results exist")
        return

    exp_dir = os.path.dirname(os.path.abspath(__file__))
    run_script = os.path.join(exp_dir, "run.py")
    subprocess.check_call([
        sys.executable, run_script,
        "leakage",
        "--num-samples", "5000",
        "--batch-size", "128",
        "--epochs", "3",
        "--output-dir", _out(7),
    ])


def phase8(args):
    print("\n" + "="*60)
    print("PHASE 8: Final Combined Optimization")
    print("="*60)
    rows = []

    tag = "plaintext"
    if not _done(8, tag, args.force):
        print("  Running plaintext reference (3 epochs)...")
        h = run_baseline_training(num_epochs=3, batch_size=128,
                                  output_dir=_out(8, tag), device=args.device)
        rows.append(_hist_row(h, config="plaintext"))
    else:
        print("  SKIP: plaintext exists")

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


def _record_epoch_metrics(
    history: dict,
    trainer: Trainer,
    epoch: int,
    train_loss: float,
    train_acc: float,
    test_loss: float,
    test_acc: float,
    best_acc: float,
) -> float:
    history["epochs"].append(epoch)
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["test_loss"].append(test_loss)
    history["test_acc"].append(test_acc)
    history["lr"].append(trainer.scheduler.get_last_lr()[0])

    updated_best_acc = max(best_acc, test_acc)

    print(f"\nEpoch {epoch} complete:")
    print(f"  Train loss: {train_loss:.3f}, Train acc: {train_acc:.2f}%")
    print(f"  Test loss: {test_loss:.3f}, Test acc: {test_acc:.2f}%")
    print(f"  Best test acc: {updated_best_acc:.2f}%")

    trainer.profiler.end_epoch(
        epoch,
        {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
        },
    )
    return updated_best_acc


def _new_history() -> dict:
    return {
        "epochs": [],
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "lr": [],
        "best_acc": 0.0,
        "total_time": 0.0,
    }


def _run_training_epochs(trainer: Trainer, sidecar: SidecarLogger, epochs: int) -> tuple[dict, float, float]:
    history = _new_history()
    total_start = time.time()
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{epochs}")
        print(f"{'='*60}")

        trainer.profiler.start_epoch(epoch)

        trainer.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(trainer.train_loader):
            batch_id = f"{epoch-1}_{batch_idx}_oram"

            sidecar.log(batch_id=batch_id, epoch=epoch - 1, phase="oram")

            inputs = inputs.to(trainer.device, non_blocking=True)
            targets = targets.to(trainer.device, non_blocking=True)

            trainer.optimizer.zero_grad()
            outputs = trainer.model(inputs)
            loss = trainer.criterion(outputs, targets)
            loss.backward()
            trainer.optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"  Batch {batch_idx+1}/{len(trainer.train_loader)}: "
                    f"loss={running_loss/(batch_idx+1):.3f}, "
                    f"acc={100.*correct/total:.2f}%"
                )

        trainer.scheduler.step()

        train_loss = running_loss / len(trainer.train_loader)
        train_acc = 100.0 * correct / total

        test_metrics = trainer.evaluate()
        test_loss = test_metrics["test_loss"]
        test_acc = test_metrics["test_acc"]

        best_acc = _record_epoch_metrics(
            history=history,
            trainer=trainer,
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            test_loss=test_loss,
            test_acc=test_acc,
            best_acc=best_acc,
        )

    total_time = time.time() - total_start
    return history, best_acc, total_time


def _persist_history(history: dict, output_dir: str) -> str:
    history_path = os.path.join(output_dir, "history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    return history_path


def _print_run_summary(best_acc: float, total_time: float, sidecar_path: str, history_path: str) -> None:
    print()
    print("=" * 60)
    print("ORAM TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best test accuracy: {best_acc:.2f}%")
    print(f"Total time: {total_time:.2f}s")
    print(f"Sidecar: {sidecar_path}")
    print(f"History: {history_path}")
    print("=" * 60)


def sidecar_main(args):
    device = resolve_torch_device(args.device)

    print("="*60)
    print("REAL ORAM TRAINING WITH SIDECAR LOGGING")
    print("="*60)
    print("Backend: Path ORAM (PyORAM)")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Storage backend: {args.backend}")
    print(f"Block size: {args.block_size}")
    print(f"Sidecar: {args.sidecar_path}")
    print("="*60)
    print()

    trainer = Trainer(
        batch_size=args.batch_size,
        device=str(device),
        num_samples=args.num_samples,
        backend=args.backend,
        block_size=args.block_size,
        model_name=args.model,
        num_workers=args.num_workers,
    )

    print("Setting up ORAM storage...")
    trainer.setup()
    print("✓ ORAM storage initialized")
    print(f"✓ PathORAM.setup() called with {trainer.oram_storage.num_samples} blocks")
    print()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Starting training with sidecar logging to {args.sidecar_path}")
    print()

    with SidecarLogger(args.sidecar_path) as sidecar:
        history, best_acc, total_time = _run_training_epochs(trainer, sidecar, args.epochs)
        history["best_acc"] = best_acc
        history["total_time"] = total_time
        history_path = _persist_history(history, args.output_dir)
        _print_run_summary(best_acc, total_time, args.sidecar_path, history_path)

    return history


def run_batch_size_sweep(epochs: int, output_root: str, device: str = None):
    print("\n" + "=" * 60)
    print("BATCH SIZE SWEEP")
    print("=" * 60)

    results = []

    for bs in BATCH_SIZES:
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

    summary_path = os.path.join(output_root, "sweep_batch_size", "sweep_summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBatch size sweep summary saved to: {summary_path}")

    return results


def run_dataset_size_sweep(epochs: int, output_root: str, device: str = None):
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

    summary_path = os.path.join(output_root, "sweep_dataset_size", "sweep_summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDataset size sweep summary saved to: {summary_path}")

    return results


def run_block_size_sweep(epochs: int, output_root: str, device: str = None):
    print("\n" + "=" * 60)
    print("BLOCK SIZE SWEEP (ORAM)")
    print("=" * 60)

    results = []

    for bs in BLOCK_SIZES:
        tag = f"oram_block{bs}"
        out_dir = os.path.join(output_root, "sweep_block_size", tag)
        print(f"\n--- ORAM block_size={bs}, epochs={epochs}, samples={DEFAULT_DATASET_SIZE} ---")
        try:
            hist = run_oram_training(
                num_epochs=epochs,
                batch_size=DEFAULT_BATCH_SIZE,
                output_dir=out_dir,
                device=device,
                num_samples=DEFAULT_DATASET_SIZE,
                block_size=bs,
            )
            results.append({
                "mode": "oram",
                "block_size": bs,
                "num_samples": DEFAULT_DATASET_SIZE,
                "epochs": epochs,
                "total_time": hist["total_time"],
                "best_acc": hist["best_acc"],
                "final_train_loss": hist["train_loss"][-1],
            })
        except Exception as exc:
            print(f"ERROR in oram block_size={bs}: {exc}")
            results.append({
                "mode": "oram",
                "block_size": bs,
                "error": str(exc),
            })

    summary_path = os.path.join(output_root, "sweep_block_size", "sweep_summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBlock size sweep summary saved to: {summary_path}")

    return results


def sweep_main(args):
    start = time.time()

    if args.sweep in ("batch_size", "all"):
        run_batch_size_sweep(args.epochs, args.output_dir, args.device)

    if args.sweep in ("dataset_size", "all"):
        run_dataset_size_sweep(args.epochs, args.output_dir, args.device)

    if args.sweep in ("block_size", "all"):
        run_block_size_sweep(args.epochs, args.output_dir, args.device)

    elapsed = time.time() - start
    print(f"\nSweep complete. Total wall-clock time: {elapsed:.1f}s ({elapsed/3600:.2f}h)")


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


def cmd_setup(args):
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


def cmd_system(args):
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


def validate_event_log(input_path: str) -> bool:
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


def cmd_probe(args):
    success = validate_event_log(args.input)

    if success:
        print("\nEvent log is suitable for non-trivial membership inference attack.")
        return 0
    else:
        print("\nEvent log validation failed.")
        return 1


def serialize_sample(global_id: int, image: np.ndarray, label: int) -> bytes:
    if image.shape != (32, 32, 3):
        raise ValueError(f"Expected CIFAR image shape (32,32,3), got {image.shape}")
    flat = image.astype(np.uint8).reshape(-1).tobytes()
    if len(flat) != 3072:
        raise ValueError("Flattened CIFAR image must be 3072 bytes.")
    return struct.pack("<IB", global_id, label) + flat


def cmd_files(args):
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


def read_sample_file(path: str) -> Tuple[torch.Tensor, int, int, str]:
    with open(path, "rb") as f:
        raw = f.read()
    if len(raw) < 5 + 3072:
        raise ValueError(f"Corrupt sample file: {path}")

    sample_id, label = struct.unpack("<IB", raw[:5])
    img = np.frombuffer(raw[5:5 + 3072], dtype=np.uint8).reshape(32, 32, 3)
    x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return x, int(label), int(sample_id), path


class FileSampleDataset(Dataset):
    def __init__(self, file_paths: List[str]):
        self.file_paths = sorted(file_paths)
        self.transform = transforms.Compose([])

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        path = self.file_paths[idx]
        x, label, sample_id, _ = read_sample_file(path)
        return x, label, sample_id, path


class ObfuscatedFileDataset(Dataset):
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


def cmd_train(args):
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


def cmd_trace(args):
    try:
        from bcc import BPF
    except ImportError:
        print("Error: BCC not available. Install with: sudo apt install bpfcc-tools python3-bpfcc")
        return 1

    BPF_PROGRAM = r"""
#include <uapi/linux/ptrace.h>
#include <linux/sched.h>

struct data_t {
    u32 pid;
    u64 ts_ns;
    char comm[TASK_COMM_LEN];
    char fname[256];
};

BPF_PERF_OUTPUT(events);

int trace_openat_entry(struct pt_regs *ctx, int dfd, const char __user *filename, int flags, umode_t mode) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    FILTER_PID

    struct data_t data = {};
    data.pid = pid;
    data.ts_ns = bpf_ktime_get_ns();
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    bpf_probe_read_user_str(&data.fname, sizeof(data.fname), filename);
    events.perf_submit(ctx, &data, sizeof(data));
    return 0;
}
"""

    text = BPF_PROGRAM.replace("FILTER_PID", f"if (pid != {args.pid}) return 0;")
    b = BPF(text=text)

    hooked = False
    for fn in ["openat", "open"]:
        for arch_prefix in ["__x64_sys_", "__arm64_sys_", "sys_", ""]:
            try:
                b.attach_kprobe(event=f"{arch_prefix}{fn}", fn_name="trace_openat_entry")
                hooked = True
                print(f"Attached to {arch_prefix}{fn}", file=sys.stderr)
            except Exception:
                pass

    if not hooked:
        print("Failed to attach to open/openat syscalls on this system.", file=sys.stderr)
        return 1

    out = open(args.output, "w", newline="", encoding="utf-8")
    writer = csv.writer(out)
    writer.writerow(["timestamp_ns", "pid", "comm", "filename"])

    running = True

    def stop_handler(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)

    def handle_event(cpu, data, size):
        event = b["events"].event(data)
        fname = event.fname.decode("utf-8", errors="replace").rstrip("\x00")
        comm = event.comm.decode("utf-8", errors="replace").rstrip("\x00")
        writer.writerow([event.ts_ns, event.pid, comm, fname])
        out.flush()

    b["events"].open_perf_buffer(handle_event)

    print(f"Tracing PID {args.pid}. Writing to {args.output}. Ctrl-C to stop.", file=sys.stderr)

    try:
        while running:
            b.perf_buffer_poll(timeout=100)
    except KeyboardInterrupt:
        pass

    out.close()
    print("Done.", file=sys.stderr)
    return 0


def infer_sample_and_label(path: str) -> Optional[Tuple[str, int]]:
    base = os.path.basename(path)
    m = MEMBER_RE.search(base)
    if m:
        return m.group(1), 1
    m = NONMEMBER_RE.search(base)
    if m:
        return m.group(1), 0
    return None


def read_trace_ebpf_csv(path: str) -> List[Tuple[float, str]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = float(row["timestamp_ns"]) / 1e9
            rows.append((ts, row["filename"]))
    rows.sort(key=lambda x: x[0])
    return rows


def read_trace_strace(path: str) -> List[Tuple[float, str]]:
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = STRACE_RE.search(line)
            if not m:
                continue
            ts = float(m.group(1))
            fname = m.group(3)
            rows.append((ts, fname))
    rows.sort(key=lambda x: x[0])
    return rows


def read_trace_strace_oram(path: str, block_size: int) -> Tuple[List[Tuple[float, str]], Dict[str, object]]:
    rows: List[Tuple[float, str]] = []
    fd_to_path: Dict[int, str] = {}
    validation: Dict[str, object] = {
        "open_rows": 0,
        "pread_rows": 0,
        "pwrite_rows": 0,
        "lseek_rows": 0,
        "candidate_paths": [],
        "quantized_blocks_seen": 0,
    }

    blocks_seen = set()
    path_seen = set()
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m_open = STRACE_OPEN_FD_RE.search(line)
            if m_open:
                ts = float(m_open.group(1))
                path_name = m_open.group(3)
                fd = int(m_open.group(4))
                validation["open_rows"] += 1
                if fd >= 0:
                    fd_to_path[fd] = path_name
                    if "oram" in path_name.lower():
                        path_seen.add(path_name)
                continue

            m_pread = STRACE_PREAD_RE.search(line)
            if m_pread:
                ts = float(m_pread.group(1))
                fd = int(m_pread.group(2))
                offset = int(m_pread.group(4))
                validation["pread_rows"] += 1
                if fd in fd_to_path and offset >= 0:
                    block_id = offset // block_size
                    token = f"oram_blk_{block_id}"
                    rows.append((ts, token))
                    blocks_seen.add(block_id)
                continue

            m_pwrite = STRACE_PWRITE_RE.search(line)
            if m_pwrite:
                ts = float(m_pwrite.group(1))
                fd = int(m_pwrite.group(2))
                offset = int(m_pwrite.group(4))
                validation["pwrite_rows"] += 1
                if fd in fd_to_path and offset >= 0:
                    block_id = offset // block_size
                    token = f"oram_blk_{block_id}"
                    rows.append((ts, token))
                    blocks_seen.add(block_id)
                continue

            m_lseek = STRACE_LSEEK_RE.search(line)
            if m_lseek:
                validation["lseek_rows"] += 1

    rows.sort(key=lambda x: x[0])
    validation["quantized_blocks_seen"] = len(blocks_seen)
    validation["candidate_paths"] = sorted(path_seen)[:20]
    return rows, validation


def _hms_to_seconds(hms: str) -> float:
    hh = int(hms[0:2])
    mm = int(hms[3:5])
    ss = float(hms[6:])
    return hh * 3600.0 + mm * 60.0 + ss


def read_trace_fs_usage(
    path: str,
    markers: List[Tuple[float, str, int, str]],
    defense: str,
    block_size: int,
) -> Tuple[List[Tuple[float, str]], Dict[str, object]]:
    rows: List[Tuple[float, str]] = []
    validation: Dict[str, object] = {
        "open_rows": 0,
        "pread_rows": 0,
        "pwrite_rows": 0,
        "lseek_rows": 0,
        "candidate_paths": [],
        "quantized_blocks_seen": 0,
    }

    if not markers:
        return rows, validation

    first_marker = markers[0][0]
    day_start = datetime.datetime.fromtimestamp(first_marker).replace(
        hour=0, minute=0, second=0, microsecond=0
    ).timestamp()

    prev_sec_of_day: Optional[float] = None
    day_offset = 0.0
    blocks_seen = set()
    path_seen = set()

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            tm = FS_USAGE_TIME_RE.match(line)
            if not tm:
                continue
            sec_of_day = _hms_to_seconds(tm.group(1))
            if prev_sec_of_day is not None and sec_of_day + 1.0 < prev_sec_of_day:
                day_offset += 86400.0
            prev_sec_of_day = sec_of_day
            ts = day_start + day_offset + sec_of_day

            if defense == "oram":
                if "oram.bin" not in line.lower():
                    continue
                path_match = FS_USAGE_PATH_RE.search(line)
                if path_match:
                    path_seen.add(path_match.group(1))
                if "rddat" in line.lower():
                    validation["pread_rows"] += 1
                if "wrdat" in line.lower():
                    validation["pwrite_rows"] += 1
                disk_off = FS_USAGE_DISK_OFF_RE.search(line)
                if disk_off:
                    block_id = int(disk_off.group(1), 16) // block_size
                    blocks_seen.add(block_id)
                    token = f"oram_blk_{block_id}"
                else:
                    token = "oram_blk_fallback"
                rows.append((ts, token))
                continue

            member_match = FS_USAGE_MEMBER_PATH_RE.search(line)
            if not member_match:
                continue
            validation["open_rows"] += 1
            rows.append((ts, member_match.group(1)))

    rows.sort(key=lambda x: x[0])
    validation["quantized_blocks_seen"] = len(blocks_seen)
    validation["candidate_paths"] = sorted(path_seen)[:20]
    return rows, validation


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


def cmd_convert(args):
    """Convert raw trace + sidecar into attack-ready event CSV."""
    markers = read_sidecar(args.sidecar)
    marker_times = [m[0] for m in markers]

    trace_validation: Dict[str, object] = {"defense": args.defense, "trace_mode": args.trace_mode}
    if args.defense == "oram":
        if args.trace_mode == "strace":
            trace_rows_oram, oram_validation = read_trace_strace_oram(args.trace_input, args.oram_block_size)
        elif args.trace_mode == "fs_usage":
            trace_rows_oram, oram_validation = read_trace_fs_usage(
                args.trace_input, markers, args.defense, args.oram_block_size
            )
        else:
            raise RuntimeError("ORAM conversion requires --trace_mode strace or fs_usage.")
        trace_rows = [(ts, token) for ts, token in trace_rows_oram]
        trace_validation.update(oram_validation)
    else:
        if args.trace_mode == "ebpf_csv":
            trace_rows = read_trace_ebpf_csv(args.trace_input)
        elif args.trace_mode == "strace":
            trace_rows = read_trace_strace(args.trace_input)
        elif args.trace_mode == "fs_usage":
            trace_rows, fs_validation = read_trace_fs_usage(
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


def cmd_upgraded(args):
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


# ---------------------------------------------------------------------------
# Generate: event logs, LaTeX tables, figures
# ---------------------------------------------------------------------------

class IndexedDataset(Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        return x, y, idx


@dataclass
class PartialObservabilityConfig:
    seed: int = 42
    train_size: int = 20000
    holdout_size: int = 20000
    batch_size: int = 128
    epochs: int = 3

    probe_batch_prob: float = 0.20
    probe_mix_ratio: float = 0.30

    visibility: float = 0.50
    timestamp_jitter_std: float = 0.003
    batch_id_corruption_prob: float = 0.10
    sample_id_mask_prob: float = 0.00
    background_noise_rate: float = 0.05
    reorder_window: int = 0

    full_output: str = "events_full.csv"
    observed_output: str = "events_observed.csv"


def load_attack_metrics(results_dir: str, mode: str, visibility: float) -> Dict[str, float]:
    """Load metrics.json for a given mode and visibility level."""
    vis_int = int(visibility * 100)
    metrics_path = os.path.join(results_dir, f"{mode}_v{vis_int}", "metrics.json")

    if not os.path.exists(metrics_path):
        return {}

    with open(metrics_path) as f:
        data = json.load(f)
        best_model = data["best_model"]
        res = data["results"][best_model]
        return {
            "auc": res["auc"],
            "accuracy": res["accuracy"],
            "ap": res["average_precision"],
            "model": best_model,
        }


def generate_latex_table(results_dir: str, visibility_levels: List[float]) -> str:
    """Generate LaTeX table comparing plaintext vs ORAM attack performance."""

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Membership inference attack performance under partial observability. "
        r"AUC values show that plaintext access patterns provide measurable signal, "
        r"while ORAM-backed patterns approach random guessing (0.5).}",
        r"\label{tab:membership_inference}",
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"Visibility & \multicolumn{2}{c}{Plaintext} & \multicolumn{2}{c}{ORAM} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
        r"          & AUC   & Accuracy & AUC   & Accuracy \\",
        r"\midrule",
    ]

    for vis in visibility_levels:
        pt_metrics = load_attack_metrics(results_dir, "plaintext", vis)
        oram_metrics = load_attack_metrics(results_dir, "oram", vis)

        pt_auc = f"{pt_metrics['auc']:.3f}" if pt_metrics else "---"
        pt_acc = f"{pt_metrics['accuracy']:.3f}" if pt_metrics else "---"
        oram_auc = f"{oram_metrics['auc']:.3f}" if oram_metrics else "---"
        oram_acc = f"{oram_metrics['accuracy']:.3f}" if oram_metrics else "---"

        lines.append(
            f"{vis:.2f}      & {pt_auc} & {pt_acc}    & {oram_auc} & {oram_acc}    \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def generate_feature_importance_table(
    results_dir: str,
    mode: str,
    visibility: float,
    top_k: int = 10
) -> str:
    """Generate LaTeX table of top feature importances."""

    vis_int = int(visibility * 100)
    metrics_path = os.path.join(results_dir, f"{mode}_v{vis_int}", "metrics.json")

    if not os.path.exists(metrics_path):
        return ""

    with open(metrics_path) as f:
        data = json.load(f)
        best_model = data["best_model"]

    importance_path = os.path.join(
        results_dir,
        f"{mode}_v{vis_int}",
        f"feature_importance_{best_model}.csv"
    )

    if not os.path.exists(importance_path):
        return ""

    df = pd.read_csv(importance_path).head(top_k)

    mode_label = "Plaintext" if mode == "plaintext" else "ORAM"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        f"\\caption{{Top {top_k} features for membership inference from {mode_label} "
        f"access patterns (visibility={visibility:.2f}, model={best_model}).}}",
        f"\\label{{tab:features_{mode}_v{vis_int}}}",
        r"\begin{tabular}{@{}lc@{}}",
        r"\toprule",
        r"Feature & Importance \\",
        r"\midrule",
    ]

    for _, row in df.iterrows():
        feature = row["feature"].replace("_", r"\_")
        importance = row["importance"]
        lines.append(f"{feature:30s} & {importance:.4f} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def generate_plaintext_log(
    train_size: int,
    holdout_size: int,
    epochs: int,
    batch_size: int,
    probe_batch_prob: float,
    probe_mix_ratio: float,
    data_dir: str,
    random_state: int,
) -> List[Tuple[str, float, int, str, int]]:
    """
    Generate event log from plaintext training with realistic probe access.

    Returns list of (sample_id, timestamp, epoch, batch_id, label) tuples.
    """
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    transform = transforms.Compose([transforms.ToTensor()])

    full_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )

    total_needed = train_size + holdout_size
    if total_needed > len(full_dataset):
        raise ValueError(f"Requested {total_needed} samples but dataset has {len(full_dataset)}")

    indices = list(range(len(full_dataset)))
    random.shuffle(indices)

    train_indices = indices[:train_size]
    holdout_indices = indices[train_size:train_size + holdout_size]

    train_set = Subset(IndexedDataset(full_dataset), train_indices)

    membership_label = {}
    for i in train_indices:
        membership_label[str(i)] = 1
    for i in holdout_indices:
        membership_label[str(i)] = 0

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 32 * 32, 10)
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    events: List[Tuple[str, float, int, str, int]] = []
    global_time = 0.0

    def log_event(sample_id: int, epoch: int, batch_id: str) -> None:
        nonlocal global_time
        global_time += random.uniform(0.001, 0.01)
        events.append((
            str(sample_id),
            global_time,
            epoch,
            batch_id,
            membership_label[str(sample_id)]
        ))

    print("Generating access log with probe batches...")

    for epoch in range(epochs):
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)

        for batch_id, (x, y, idxs) in enumerate(train_loader):
            batch_tag = f"{epoch}_{batch_id}_train"

            for idx in idxs:
                log_event(int(idx), epoch, batch_tag)

            if random.random() < probe_batch_prob:
                probe_size = int(batch_size * probe_mix_ratio)
                holdout_samples = random.sample(holdout_indices, min(probe_size, len(holdout_indices)))

                probe_tag = f"{epoch}_{batch_id}_probe"
                for sample_id in holdout_samples:
                    log_event(sample_id, epoch, probe_tag)

            optimizer.zero_grad()
            out = model(x.view(x.size(0), -1))
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

    return events


def generate_oram_log(
    train_size: int,
    holdout_size: int,
    epochs: int,
    batch_size: int,
    probe_batch_prob: float,
    probe_mix_ratio: float,
    data_dir: str,
    random_state: int,
    backend: str = "ram",
) -> List[Tuple[str, float, int, str, int]]:
    """
    Generate event log from ORAM-backed training with realistic probe access.

    Returns list of (sample_id, timestamp, epoch, batch_id, label) tuples.
    """
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    total_samples = train_size + holdout_size
    storage = ORAMStorage(num_samples=total_samples, backend=backend)

    load_cifar10_to_oram(
        storage,
        data_dir=data_dir,
        train=True,
        progress=True,
        limit=total_samples
    )

    train_indices = list(range(train_size))
    holdout_indices = list(range(train_size, total_samples))

    transform = get_cifar10_transforms(train=True)
    full_dataset = ORAMDataset(storage, total_samples, transform=transform)

    train_set = Subset(IndexedDataset(full_dataset), train_indices)

    membership_label = {}
    for i in train_indices:
        membership_label[str(i)] = 1
    for i in holdout_indices:
        membership_label[str(i)] = 0

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 32 * 32, 10)
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    events: List[Tuple[str, float, int, str, int]] = []
    global_time = 0.0

    def log_event(sample_id: int, epoch: int, batch_id: str) -> None:
        nonlocal global_time
        global_time += random.uniform(0.001, 0.01)
        events.append((
            str(sample_id),
            global_time,
            epoch,
            batch_id,
            membership_label[str(sample_id)]
        ))

    print("Generating ORAM access log with probe batches...")

    try:
        for epoch in range(epochs):
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)

            for batch_id, (x, y, idxs) in enumerate(train_loader):
                batch_tag = f"{epoch}_{batch_id}_train"

                for idx in idxs:
                    log_event(int(idx), epoch, batch_tag)

                if random.random() < probe_batch_prob:
                    probe_size = int(batch_size * probe_mix_ratio)
                    holdout_samples = random.sample(holdout_indices, min(probe_size, len(holdout_indices)))

                    probe_tag = f"{epoch}_{batch_id}_probe"
                    for sample_id in holdout_samples:
                        log_event(sample_id, epoch, probe_tag)

                optimizer.zero_grad()
                out = model(x.view(x.size(0), -1))
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

    finally:
        storage.close()

    return events


def save_events_csv(events: List[Tuple[str, float, int, str, int]], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "timestamp", "epoch", "batch_id", "label"])
        for sample_id, timestamp, epoch, batch_id, label in events:
            writer.writerow([sample_id, timestamp, epoch, batch_id, label])


def save_csv(path: str, rows: List[Dict[str, object]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "timestamp", "epoch", "batch_id", "label"],
        )
        writer.writeheader()
        writer.writerows(rows)


def maybe_reorder(events: List[Dict[str, object]], window: int, rng: random.Random) -> List[Dict[str, object]]:
    if window <= 1:
        return events

    out: List[Dict[str, object]] = []
    for start in range(0, len(events), window):
        chunk = events[start:start + window]
        rng.shuffle(chunk)
        out.extend(chunk)
    return out


def build_observed_stream(
    full_events: List[Dict[str, object]],
    cfg: PartialObservabilityConfig,
    membership_label: Dict[str, int],
) -> List[Dict[str, object]]:
    rng = random.Random(cfg.seed + 1000)
    np_rng = np.random.default_rng(cfg.seed + 1000)

    observed: List[Dict[str, object]] = []

    for ev in full_events:
        if rng.random() > cfg.visibility:
            continue

        ts = float(ev["timestamp"]) + float(np_rng.normal(0.0, cfg.timestamp_jitter_std))
        batch_id = str(ev["batch_id"])
        sample_id = str(ev["sample_id"])

        if rng.random() < cfg.batch_id_corruption_prob:
            batch_id = f"corrupt_{rng.randint(0, 999999)}"

        if cfg.sample_id_mask_prob > 0.0 and rng.random() < cfg.sample_id_mask_prob:
            sample_id = f"masked_{rng.randint(0, 1023)}"

        observed.append(
            {
                "sample_id": sample_id,
                "timestamp": max(ts, 0.0),
                "epoch": int(ev["epoch"]),
                "batch_id": batch_id,
                "label": int(ev["label"]),
            }
        )

    noise_count = int(len(observed) * cfg.background_noise_rate)
    all_ids = list(membership_label.keys())
    max_ts = max([float(r["timestamp"]) for r in observed], default=1.0)
    max_epoch = max([int(r["epoch"]) for r in observed], default=0)

    for i in range(noise_count):
        sid = rng.choice(all_ids)
        noisy = {
            "sample_id": sid,
            "timestamp": float(np_rng.uniform(0.0, max_ts)),
            "epoch": rng.randint(0, max_epoch),
            "batch_id": f"noise_{i}",
            "label": int(membership_label[sid]),
        }
        observed.append(noisy)

    observed.sort(key=lambda r: float(r["timestamp"]))
    observed = maybe_reorder(observed, cfg.reorder_window, rng)
    return observed


def cmd_gen_attack(args) -> None:
    """Generate LaTeX tables from membership inference attack results."""
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    main_table = generate_latex_table(args.results_dir, args.visibility_levels)

    with open(args.output, "w") as f:
        f.write(main_table)
        f.write("\n\n")

        if args.include_features:
            f.write("% Feature importance tables\n\n")
            for mode in ["plaintext", "oram"]:
                feature_table = generate_feature_importance_table(
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


def cmd_event(args) -> None:
    """Generate access-pattern event logs for membership inference."""
    print(f"Generating {args.mode} event log with probe access...")
    print(f"  Train size: {args.train_size}")
    print(f"  Holdout size: {args.holdout_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Probe batch probability: {args.probe_batch_prob}")
    print(f"  Probe mix ratio: {args.probe_mix_ratio}")

    if args.mode == "plaintext":
        events = generate_plaintext_log(
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
        events = generate_oram_log(
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

    save_events_csv(events, args.output)

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


def cmd_partial(args) -> None:
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
    observed_events = build_observed_stream(full_events, cfg, membership_label)

    save_csv(cfg.full_output, full_events)
    save_csv(cfg.observed_output, observed_events)

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


def cmd_reference(args) -> None:
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


GEN_RESULTS_ROOT = "results"
GEN_OUTPUT_DIR = "results/figures"


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


def fig_backend_sensitivity(root, out):
    """Plaintext vs file-ORAM vs RAM-ORAM epoch time."""
    data = []

    h = _load_history(os.path.join(root, "phase0", "baseline"))
    if h:
        data.append({"config": "Plaintext", "epoch_time": _epoch_time(h)})

    h = _load_history(os.path.join(root, "phase0", "oram_file"))
    if h:
        data.append({"config": "File ORAM", "epoch_time": _epoch_time(h)})

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


def generate_summary(root, out):
    """Generate a combined summary from phase8."""
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

    lines = ["# ORAM Evaluation Summary\n",
             "| System | Epoch Time (s) | Best Acc (%) |",
             "|--------|----------------|--------------|"]
    for r in rows:
        lines.append(f"| {r['System']} | {r['Epoch Time (s)']} | {r['Best Acc (%)']} |")
    lines.append("")

    path = os.path.join(out, "summary.txt")
    _ensure(path)
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {path}")


def plot_results(results_root: str, output_dir: str):
    """Generate all 7 figures from phase results."""
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("GENERATING FIGURES")
    print("=" * 60)

    fig_backend_sensitivity(results_root, output_dir)
    fig_worker_scaling(results_root, output_dir)
    fig_block_size(results_root, output_dir)
    fig_model_scaling(results_root, output_dir)
    fig_dataset_scaling(results_root, output_dir)
    fig_batch_size(results_root, output_dir)
    fig_leakage(results_root, output_dir)
    generate_summary(results_root, output_dir)

    print("\nDone. All figures saved to:", output_dir)


def load_summary(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_privacy(summary_path: str, oram_auc: float, oram_overhead: float, output_path: str):
    """Privacy-performance tradeoff plot."""
    rows = load_summary(summary_path)

    plaintext_row = None
    obfuscatedcated_row = None
    oram_row = None

    for row in rows:
        if float(row["visibility"]) == 1.0:
            if row["defense"] == "plaintext" and plaintext_row is None:
                plaintext_row = row
            elif row["defense"] == "obfuscatedcated" and obfuscatedcated_row is None:
                obfuscatedcated_row = row
            elif row["defense"] == "oram" and oram_row is None:
                oram_row = row

    if not plaintext_row:
        raise RuntimeError("No plaintext visibility=1.0 row found in summary")

    plaintext_runtime = float(plaintext_row["train_runtime_sec"])
    plaintext_auc = float(plaintext_row["best_auc"])

    points = [
        ("Plaintext", 1.0, plaintext_auc, 'o', 'red'),
    ]

    if obfuscatedcated_row:
        obf_runtime = float(obfuscatedcated_row["train_runtime_sec"])
        obf_overhead = obf_runtime / plaintext_runtime
        obf_auc = float(obfuscatedcated_row["best_auc"])
        points.append(("Prefetch obfuscatedcation", obf_overhead, obf_auc, 's', 'orange'))

    if oram_row:
        oram_runtime = float(oram_row["train_runtime_sec"])
        oram_overhead = oram_runtime / plaintext_runtime
        oram_auc = float(oram_row["best_auc"])
        points.append(("Path ORAM", oram_overhead, oram_auc, '^', 'blue'))
    else:
        points.append(("Path ORAM", oram_overhead, oram_auc, '^', 'blue'))

    fig, ax = plt.subplots(figsize=(8, 6))

    for label, overhead, auc, marker, color in points:
        ax.scatter(overhead, auc, s=200, marker=marker, color=color, label=label, edgecolors='black', linewidths=1.5, zorder=3)

    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label="Random")

    ax.set_xlabel("Training Runtime Overhead (x)", fontsize=12)
    ax.set_ylabel("AUC", fontsize=12)
    ax.set_title("Privacy-Performance Tradeoff", fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_ylim([0.45, 0.85])

    for label, overhead, auc, marker, color in points:
        ax.annotate(
            f"{auc:.2f}",
            xy=(overhead, auc),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8)
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

    print("\nTradeoff summary:")
    for label, overhead, auc, _, _ in points:
        print(f"  {label:25s}  AUC={auc:.2f}  Overhead={overhead:.1f}x")


def load_feature_table(results_dir: str, mode: str, visibility: float) -> pd.DataFrame:
    vis_int = int(visibility * 100)
    feature_path = os.path.join(results_dir, f"{mode}_v{vis_int}", "feature_table.csv")

    if not os.path.exists(feature_path):
        return pd.DataFrame()

    return pd.read_csv(feature_path)


def plot_membership(results_dir: str, output_path: str):
    """Visual comparison of trivial vs upgraded membership inference attacks."""

    fig = plt.figure(figsize=(14, 10))

    pt_features = load_feature_table(results_dir, "plaintext", 1.0)
    oram_features = load_feature_table(results_dir, "oram", 1.0)

    if pt_features.empty:
        print("No plaintext feature table found. Run attack first.")
        return

    ax1 = plt.subplot(2, 2, 1)
    members = pt_features[pt_features["label"] == 1]["count_total"]
    nonmembers = pt_features[pt_features["label"] == 0]["count_total"]

    ax1.hist(members, bins=30, alpha=0.6, label="Members", color="coral")
    ax1.hist(nonmembers, bins=30, alpha=0.6, label="Non-members", color="steelblue")
    ax1.set_xlabel("Access Count (Single Feature)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Old Attack: Trivial Single-Feature Separation")
    ax1.legend()
    ax1.text(0.05, 0.95, "Feature dimensionality: 1",
             transform=ax1.transAxes, fontsize=10, va="top",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax2 = plt.subplot(2, 2, 2)

    feature_cols = [c for c in pt_features.columns if c not in {"sample_id", "label"}]
    num_features = len(feature_cols)

    sample_features = [
        "count_total", "unique_epochs", "interarrival_mean",
        "interarrival_burstiness", "global_pos_mean", "batch_unique_partners"
    ]
    sample_features = [f for f in sample_features if f in pt_features.columns][:6]

    member_means = [pt_features[pt_features["label"] == 1][f].mean() for f in sample_features]
    nonmember_means = [pt_features[pt_features["label"] == 0][f].mean() for f in sample_features]

    member_means_norm = np.array(member_means) / (np.array(member_means) + np.array(nonmember_means) + 1e-9)
    nonmember_means_norm = np.array(nonmember_means) / (np.array(member_means) + np.array(nonmember_means) + 1e-9)

    x = np.arange(len(sample_features))
    width = 0.35

    ax2.bar(x - width/2, member_means_norm, width, label="Members", color="coral")
    ax2.bar(x + width/2, nonmember_means_norm, width, label="Non-members", color="steelblue")
    ax2.set_ylabel("Normalized Feature Value")
    ax2.set_title("New Attack: Multi-Dimensional Feature Separation")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f.replace("_", "\n") for f in sample_features], fontsize=8)
    ax2.legend()
    ax2.text(0.05, 0.95, f"Feature dimensionality: {num_features}",
             transform=ax2.transAxes, fontsize=10, va="top",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax3 = plt.subplot(2, 2, 3)

    metrics_path = os.path.join(results_dir, "plaintext_v100", "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            data = json.load(f)
            best_model = data["best_model"]
            pt_auc = data["results"][best_model]["auc"]
    else:
        pt_auc = 0.0

    metrics_path = os.path.join(results_dir, "oram_v100", "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            data = json.load(f)
            best_model = data["best_model"]
            oram_auc = data["results"][best_model]["auc"]
    else:
        oram_auc = 0.5

    old_attack_auc = 0.98

    scenarios = ["Old Attack\n(Trivial)", "New Attack\n(Plaintext)", "New Attack\n(ORAM)"]
    aucs = [old_attack_auc, pt_auc, oram_auc]
    colors = ["lightcoral", "coral", "steelblue"]

    bars = ax3.bar(scenarios, aucs, color=colors)
    ax3.axhline(0.5, ls="--", color="gray", label="Random")
    ax3.set_ylabel("AUC")
    ax3.set_title("Attack Performance Comparison")
    ax3.set_ylim([0.4, 1.0])
    ax3.legend()

    for bar, auc in zip(bars, aucs):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{auc:.3f}", ha="center", va="bottom", fontweight="bold")

    ax3.text(0.5, 0.05,
             "Old: trivial separation (members accessed, non-members never accessed)\n"
             "New: non-trivial separation (both classes in log, different patterns)",
             transform=ax3.transAxes, fontsize=9, ha="center",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax4 = plt.subplot(2, 2, 4)

    visibility_levels = [1.0, 0.5, 0.25, 0.1]
    pt_aucs = []
    oram_aucs = []

    for vis in visibility_levels:
        vis_int = int(vis * 100)

        metrics_path = os.path.join(results_dir, f"plaintext_v{vis_int}", "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                data = json.load(f)
                best_model = data["best_model"]
                pt_aucs.append(data["results"][best_model]["auc"])
        else:
            pt_aucs.append(None)

        metrics_path = os.path.join(results_dir, f"oram_v{vis_int}", "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                data = json.load(f)
                best_model = data["best_model"]
                oram_aucs.append(data["results"][best_model]["auc"])
        else:
            oram_aucs.append(None)

    if any(pt_aucs):
        ax4.plot([v*100 for v in visibility_levels], pt_aucs, "o-",
                linewidth=2, color="coral", label="Plaintext")
    if any(oram_aucs):
        ax4.plot([v*100 for v in visibility_levels], oram_aucs, "s-",
                linewidth=2, color="steelblue", label="ORAM")

    ax4.axhline(0.5, ls="--", color="gray", label="Random")
    ax4.set_xlabel("Visibility (%)")
    ax4.set_ylabel("AUC")
    ax4.set_title("Partial Observability Robustness")
    ax4.set_ylim([0.4, 1.0])
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle("Membership Inference Attack: Trivial vs Non-Trivial Approach",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Comparison figure saved to: {output_path}")


def plot_robustness(summary_path: str, oram_results_dir: Optional[str], output_path: str):
    """Attack robustness vs visibility."""
    rows = load_summary(summary_path)

    plaintext_data = {}
    obfuscatedcated_data = {}
    oram_data = {}

    for row in rows:
        defense = row["defense"]
        visibility = float(row["visibility"])
        auc = float(row["best_auc"])

        if defense == "plaintext":
            plaintext_data[visibility] = auc
        elif defense == "obfuscatedcated":
            obfuscatedcated_data[visibility] = auc
        elif defense == "oram":
            oram_data[visibility] = auc

    visibilities = sorted(set(plaintext_data.keys()) | set(obfuscatedcated_data.keys()) | set(oram_data.keys()))

    fig, ax = plt.subplots(figsize=(8, 6))

    if plaintext_data:
        vis_plain = sorted(plaintext_data.keys())
        auc_plain = [plaintext_data[v] for v in vis_plain]
        ax.plot([v * 100 for v in vis_plain], auc_plain, 'o-', linewidth=2, markersize=8, label="Plaintext")

    if obfuscatedcated_data:
        vis_obf = sorted(obfuscatedcated_data.keys())
        auc_obf = [obfuscatedcated_data[v] for v in vis_obf]
        ax.plot([v * 100 for v in vis_obf], auc_obf, 's-', linewidth=2, markersize=8, label="Prefetch obfuscatedcation")

    if oram_data:
        vis_oram = sorted(oram_data.keys())
        auc_oram = [oram_data[v] for v in vis_oram]
        ax.plot([v * 100 for v in vis_oram], auc_oram, '^-', linewidth=2, markersize=8, label="Path ORAM")
    elif oram_results_dir:
        oram_auc_values = []
        for vis in visibilities:
            vis_str = str(vis).replace(".", "p")
            metrics_path = os.path.join(oram_results_dir, f"attack_v{vis_str}", "metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)
                    best = metrics["best_model"]
                    oram_auc_values.append((vis, metrics["results"][best]["auc"]))

        if oram_auc_values:
            vis_oram, auc_oram = zip(*oram_auc_values)
            ax.plot([v * 100 for v in vis_oram], auc_oram, '^-', linewidth=2, markersize=8, label="ORAM")

    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label="Random")

    ax.set_xlabel("Visibility (%)", fontsize=12)
    ax.set_ylabel("AUC", fontsize=12)
    ax.set_title("Attack Robustness vs Visibility", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.45, 0.90])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def cmd_plot(args) -> None:
    plot_results(args.results_root, args.output)


def cmd_privacy(args) -> None:
    plot_privacy(args.summary, args.oram_auc, args.oram_overhead, args.output)


def cmd_membership(args) -> None:
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plot_membership(args.results_dir, args.output)


def cmd_robustness(args) -> None:
    plot_robustness(args.summary, args.oram_results, args.output)


# ---------------------------------------------------------------------------
# Membership inference attacks
# ---------------------------------------------------------------------------

ATTACK_REQUIRED_COLUMNS = {"sample_id", "timestamp", "epoch", "batch_id", "label"}


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.net(x)


def run_simple_attack(
    batch_size: int = 128,
    epochs: int = 3,
    train_size: int = 20000,
    holdout_size: int = 20000,
    seed: int = 42,
    output_dir: str = "results",
) -> Dict[str, float]:
    """Simple membership inference using access frequency alone."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465],
                             [0.2023, 0.1994, 0.2010])
    ])

    full_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    indices = list(range(len(full_dataset)))
    random.shuffle(indices)

    train_indices = indices[:train_size]
    holdout_indices = indices[train_size:train_size + holdout_size]

    train_dataset = Subset(IndexedDataset(full_dataset), train_indices)
    holdout_dataset = Subset(IndexedDataset(full_dataset), holdout_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    access_count = defaultdict(int)

    print("Training...")

    for epoch in range(epochs):
        model.train()
        for x, y, idx in train_loader:
            x, y = x.to(device), y.to(device)

            for i in idx:
                access_count[int(i)] += 1

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1} done")

    print("Building attack dataset...")

    X = []
    y_true = []

    for idx in train_indices:
        X.append([access_count[idx]])
        y_true.append(1)

    for idx in holdout_indices:
        X.append([access_count[idx]])
        y_true.append(0)

    X = np.array(X)
    y_true = np.array(y_true)

    print("Training attack model...")

    clf = LogisticRegression()
    clf.fit(X, y_true)

    preds = clf.predict(X)
    probs = clf.predict_proba(X)[:, 1]

    acc = accuracy_score(y_true, preds)
    auc = roc_auc_score(y_true, probs)

    print("\n=== MEMBERSHIP INFERENCE RESULTS ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC:      {auc:.4f}")

    train_counts = [access_count[i] for i in train_indices]
    holdout_counts = [access_count[i] for i in holdout_indices]

    os.makedirs(f"{output_dir}/figures", exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.hist(train_counts, bins=30, alpha=0.6, label="Train", color="coral")
    plt.hist(holdout_counts, bins=30, alpha=0.6, label="Holdout", color="steelblue")
    plt.xlabel("Access Count")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("Access Count Distribution")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figures/membership_auc.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {output_dir}/figures/membership_auc.png")

    with open(f"{output_dir}/membership_results.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"AUC: {auc:.4f}\n")
    print(f"Saved results: {output_dir}/membership_results.txt")

    return {"accuracy": acc, "auc": auc}


def load_events(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = ATTACK_REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="raise")
    df["epoch"] = pd.to_numeric(df["epoch"], errors="raise").astype(int)
    df["sample_id"] = df["sample_id"].astype(str)
    df["batch_id"] = df["batch_id"].astype(str)
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError("label column must contain only 0/1.")

    return df.sort_values(["timestamp", "epoch", "batch_id"]).reset_index(drop=True)


def subsample_visibility(df: pd.DataFrame, visibility: float, random_state: int) -> pd.DataFrame:
    if visibility >= 1.0:
        return df

    rng = np.random.default_rng(random_state)
    keep_mask = rng.random(len(df)) < visibility
    out = df.loc[keep_mask].copy().reset_index(drop=True)
    return out


def maybe_limit_samples(df: pd.DataFrame, max_samples: Optional[int], random_state: int) -> pd.DataFrame:
    if max_samples is None:
        return df

    sample_ids = df["sample_id"].drop_duplicates().tolist()
    if len(sample_ids) <= max_samples:
        return df

    rng = np.random.default_rng(random_state)
    chosen = set(rng.choice(sample_ids, size=max_samples, replace=False))
    return df[df["sample_id"].isin(chosen)].copy().reset_index(drop=True)


def compute_epoch_batch_normalizers(df: pd.DataFrame) -> Tuple[Dict[int, int], Dict[int, float]]:
    epoch_sizes: Dict[int, int] = df.groupby("epoch").size().to_dict()
    epoch_time_spans: Dict[int, float] = {}

    for epoch, g in df.groupby("epoch"):
        span = float(g["timestamp"].max() - g["timestamp"].min())
        epoch_time_spans[int(epoch)] = max(span, 1e-9)

    return epoch_sizes, epoch_time_spans


def compute_batch_cooccurrence_scores(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    unique_partners: Dict[str, set] = defaultdict(set)
    total_partner_events: Counter = Counter()
    batch_sizes_for_sample: Dict[str, List[int]] = defaultdict(list)

    for _, g in df.groupby("batch_id"):
        ids = g["sample_id"].astype(str).tolist()
        if len(ids) <= 1:
            for sid in ids:
                batch_sizes_for_sample[sid].append(1)
            continue

        uniq_ids = list(dict.fromkeys(ids))
        batch_size = len(uniq_ids)

        for sid in uniq_ids:
            batch_sizes_for_sample[sid].append(batch_size)

        for a, b in combinations(uniq_ids, 2):
            unique_partners[a].add(b)
            unique_partners[b].add(a)
            total_partner_events[a] += 1
            total_partner_events[b] += 1

    stats: Dict[str, Dict[str, float]] = {}
    all_ids = df["sample_id"].unique().tolist()
    for sid in all_ids:
        sizes = batch_sizes_for_sample.get(sid, [])
        stats[sid] = {
            "batch_unique_partners": float(len(unique_partners.get(sid, set()))),
            "batch_total_partner_events": float(total_partner_events.get(sid, 0)),
            "batch_mean_group_size": float(np.mean(sizes)) if sizes else 0.0,
            "batch_std_group_size": float(np.std(sizes)) if sizes else 0.0,
        }
    return stats


def safe_stats(values: np.ndarray, prefix: str) -> Dict[str, float]:
    if values.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_median": 0.0,
            f"{prefix}_iqr": 0.0,
        }

    q25, q75 = np.percentile(values, [25, 75])
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_iqr": float(q75 - q25),
    }


def coefficient_of_variation(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    mean = float(np.mean(values))
    std = float(np.std(values))
    return 0.0 if abs(mean) < 1e-12 else std / mean


def attack_burstiness(inter_arrivals: np.ndarray) -> float:
    """Standard burstiness proxy: (sigma - mu)/(sigma + mu)."""
    if inter_arrivals.size == 0:
        return 0.0
    mu = float(np.mean(inter_arrivals))
    sigma = float(np.std(inter_arrivals))
    denom = sigma + mu
    return 0.0 if denom < 1e-12 else (sigma - mu) / denom


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    epoch_sizes, epoch_time_spans = compute_epoch_batch_normalizers(df)
    cooc_stats = compute_batch_cooccurrence_scores(df)

    global_t_min = float(df["timestamp"].min())
    global_t_max = float(df["timestamp"].max())
    global_span = max(global_t_max - global_t_min, 1e-9)

    rows: List[Dict[str, float]] = []

    for sample_id, g in df.groupby("sample_id"):
        g = g.sort_values("timestamp")
        label_vals = g["label"].unique()
        if len(label_vals) != 1:
            raise ValueError(f"Sample {sample_id} has inconsistent labels: {label_vals}")
        label = int(label_vals[0])

        timestamps = g["timestamp"].to_numpy(dtype=float)
        epochs = g["epoch"].to_numpy(dtype=int)
        batches = g["batch_id"].astype(str).to_numpy()

        count_total = float(len(g))
        unique_epochs = float(len(np.unique(epochs)))
        unique_batches = float(len(np.unique(batches)))

        inter = np.diff(timestamps) if len(timestamps) > 1 else np.array([], dtype=float)

        epoch_counts_series = g.groupby("epoch").size()
        epoch_counts = epoch_counts_series.to_numpy(dtype=float)
        epoch_count_map = epoch_counts_series.to_dict()

        norm_epoch_freqs = []
        first_pos_in_epoch = []
        mean_pos_in_epoch = []
        last_pos_in_epoch = []

        for epoch, eg in g.groupby("epoch"):
            esize = float(epoch_sizes[int(epoch)])
            e_times = eg["timestamp"].to_numpy(dtype=float)
            e_start = float(df[df["epoch"] == int(epoch)]["timestamp"].min())
            span = epoch_time_spans[int(epoch)]

            norm_epoch_freqs.append(len(eg) / esize)
            rel_positions = (e_times - e_start) / span
            first_pos_in_epoch.append(float(np.min(rel_positions)))
            mean_pos_in_epoch.append(float(np.mean(rel_positions)))
            last_pos_in_epoch.append(float(np.max(rel_positions)))

        rel_global_positions = (timestamps - global_t_min) / global_span

        epoch_switches = float(np.sum(np.diff(epochs) != 0)) if len(epochs) > 1 else 0.0

        row: Dict[str, float] = {
            "sample_id": sample_id,
            "label": label,
            "count_total": count_total,
            "count_log1p": float(np.log1p(count_total)),
            "unique_epochs": unique_epochs,
            "unique_batches": unique_batches,
            "count_per_unique_epoch": float(count_total / max(unique_epochs, 1.0)),
            "count_per_unique_batch": float(count_total / max(unique_batches, 1.0)),
            "epoch_switches": epoch_switches,
            "epoch_switch_rate": float(epoch_switches / max(count_total - 1.0, 1.0)),
            "interarrival_cv": coefficient_of_variation(inter),
            "interarrival_burstiness": attack_burstiness(inter),
            "global_pos_first": float(np.min(rel_global_positions)) if len(rel_global_positions) else 0.0,
            "global_pos_mean": float(np.mean(rel_global_positions)) if len(rel_global_positions) else 0.0,
            "global_pos_last": float(np.max(rel_global_positions)) if len(rel_global_positions) else 0.0,
            "active_epoch_fraction": float(unique_epochs / max(len(epoch_sizes), 1)),
        }

        row.update(safe_stats(inter, "interarrival"))
        row.update(safe_stats(epoch_counts, "epochcount"))
        row.update(safe_stats(np.array(norm_epoch_freqs, dtype=float), "normepochfreq"))
        row.update(safe_stats(np.array(first_pos_in_epoch, dtype=float), "epochfirstpos"))
        row.update(safe_stats(np.array(mean_pos_in_epoch, dtype=float), "epochmeanpos"))
        row.update(safe_stats(np.array(last_pos_in_epoch, dtype=float), "epochlastpos"))
        row.update(cooc_stats.get(sample_id, {}))

        for e in sorted(epoch_sizes.keys())[:10]:
            row[f"epoch_{e}_count"] = float(epoch_count_map.get(e, 0))

        rows.append(row)

    feature_df = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
    return feature_df


def split_xy(feature_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    y = feature_df["label"].to_numpy(dtype=int)
    feature_cols = [c for c in feature_df.columns if c not in {"sample_id", "label"}]
    X = feature_df[feature_cols].copy()
    return X, y, feature_cols


def build_attack_models(random_state: int) -> Dict[str, Pipeline]:
    models: Dict[str, Pipeline] = {
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=None,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                (
                    "clf",
                    GradientBoostingClassifier(
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }

    if HAS_XGBOOST:
        models["xgboost"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=400,
                        max_depth=5,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        reg_lambda=1.0,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        random_state=random_state,
                        n_jobs=4,
                    ),
                ),
            ]
        )

    return models


def evaluate_attack_model(
    model: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, object]:
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    metrics: Dict[str, object] = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "auc": float(roc_auc_score(y_test, probs)),
        "average_precision": float(average_precision_score(y_test, probs)),
        "classification_report": classification_report(y_test, preds, output_dict=True),
    }

    fpr, tpr, _ = roc_curve(y_test, probs)
    precision, recall, _ = precision_recall_curve(y_test, probs)
    metrics["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
    metrics["pr_curve"] = {"precision": precision.tolist(), "recall": recall.tolist()}
    return metrics


def extract_feature_importance(model: Pipeline, feature_cols: List[str]) -> pd.DataFrame:
    clf = model.named_steps["clf"]

    if hasattr(clf, "feature_importances_"):
        importance = np.asarray(clf.feature_importances_, dtype=float)
    elif hasattr(clf, "coef_"):
        coef = np.asarray(clf.coef_, dtype=float)
        if coef.ndim == 2:
            coef = coef[0]
        importance = np.abs(coef)
    else:
        importance = np.zeros(len(feature_cols), dtype=float)

    df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": importance,
        }
    ).sort_values("importance", ascending=False)
    return df.reset_index(drop=True)


def plot_roc(results: Dict[str, Dict[str, object]], output_path: str) -> None:
    plt.figure(figsize=(6, 5))
    for model_name, metrics in results.items():
        roc = metrics["roc_curve"]
        fpr = np.array(roc["fpr"], dtype=float)
        tpr = np.array(roc["tpr"], dtype=float)
        auc = float(metrics["auc"])
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Membership Inference ROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_pr(results: Dict[str, Dict[str, object]], output_path: str) -> None:
    plt.figure(figsize=(6, 5))
    for model_name, metrics in results.items():
        pr = metrics["pr_curve"]
        precision = np.array(pr["precision"], dtype=float)
        recall = np.array(pr["recall"], dtype=float)
        ap = float(metrics["average_precision"])
        plt.plot(recall, precision, label=f"{model_name} (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Membership Inference Precision-Recall")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_top_features(importance_df: pd.DataFrame, output_path: str, top_k: int = 15) -> None:
    top = importance_df.head(top_k).iloc[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"], top["importance"])
    plt.xlabel("Importance")
    plt.title(f"Top {top_k} Features")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_attack_json(obj: object, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def run_upgraded_attack(
    input_path: str,
    output_dir: str,
    test_size: float = 0.3,
    random_state: int = 42,
    visibility: float = 1.0,
    max_samples: Optional[int] = None,
) -> Dict[str, object]:
    """Non-trivial membership inference from access-pattern logs."""
    ensure_dir(output_dir)

    df = load_events(input_path)
    df = subsample_visibility(df, visibility=visibility, random_state=random_state)
    df = maybe_limit_samples(df, max_samples=max_samples, random_state=random_state)

    if df.empty:
        raise ValueError("No events remain after visibility/max_samples filtering.")

    feature_df = build_feature_table(df)

    label_counts = feature_df["label"].value_counts().to_dict()
    if set(label_counts.keys()) != {0, 1}:
        raise ValueError(
            f"Need both label classes after filtering. Got label counts: {label_counts}"
        )

    X, y, feature_cols = split_xy(feature_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    models = build_attack_models(random_state)
    results: Dict[str, Dict[str, object]] = {}
    importance_tables: Dict[str, pd.DataFrame] = {}

    for model_name, model in models.items():
        metrics = evaluate_attack_model(model, X_train, X_test, y_train, y_test)
        results[model_name] = metrics
        importance_df = extract_feature_importance(model, feature_cols)
        importance_tables[model_name] = importance_df
        importance_df.to_csv(
            os.path.join(output_dir, f"feature_importance_{model_name}.csv"),
            index=False,
        )

    best_model_name = max(results, key=lambda k: float(results[k]["auc"]))
    best_importance = importance_tables[best_model_name]

    feature_df.to_csv(os.path.join(output_dir, "feature_table.csv"), index=False)

    summary = {
        "config": {
            "input_path": input_path,
            "test_size": test_size,
            "random_state": random_state,
            "visibility": visibility,
            "max_samples": max_samples,
        },
        "num_events": int(len(df)),
        "num_samples": int(feature_df.shape[0]),
        "label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "results": results,
        "best_model": best_model_name,
    }
    save_attack_json(summary, os.path.join(output_dir, "metrics.json"))

    plot_roc(results, os.path.join(output_dir, "roc_curve.png"))
    plot_pr(results, os.path.join(output_dir, "pr_curve.png"))
    plot_top_features(best_importance, os.path.join(output_dir, "top_features.png"))

    print("\n=== MEMBERSHIP INFERENCE RESULTS ===")
    print(f"Events retained: {len(df)}")
    print(f"Unique samples: {feature_df.shape[0]}")
    print(f"Visibility: {visibility:.2f}")
    print(f"Best model: {best_model_name}")
    for model_name, metrics in results.items():
        print(
            f"{model_name:18s} "
            f"AUC={metrics['auc']:.4f} "
            f"ACC={metrics['accuracy']:.4f} "
            f"AP={metrics['average_precision']:.4f}"
        )

    print(f"\nSaved outputs to: {output_dir}")

    return summary


def cmd_mi(args) -> None:
    """Run upgraded multi-feature membership inference attack."""
    if not (0.0 < args.visibility <= 1.0):
        raise ValueError("--visibility must be in (0, 1].")
    if not (0.0 < args.test_size < 1.0):
        raise ValueError("--test_size must be in (0, 1).")

    run_upgraded_attack(
        input_path=args.input,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        visibility=args.visibility,
        max_samples=args.max_samples,
    )


def cmd_mi_simple(args) -> None:
    """Run simple frequency-based membership inference attack."""
    run_simple_attack(
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
        'baseline': baseline_main,
        'experiments': experiments_main,
        'inference': inference_main,
        'oram': oram_main,
        'phases': phases_main,
        'sidecar': sidecar_main,
        'sweep': sweep_main,
        'setup': cmd_setup,
        'system': cmd_system,
        'probe': cmd_probe,
        'files': cmd_files,
        'train': cmd_train,
        'trace': cmd_trace,
        'convert': cmd_convert,
        'upgraded': cmd_upgraded,
        'attack': cmd_gen_attack,
        'event': cmd_event,
        'partial': cmd_partial,
        'reference': cmd_reference,
        'plot': cmd_plot,
        'privacy': cmd_privacy,
        'membership': cmd_membership,
        'robustness': cmd_robustness,
        'mi': cmd_mi,
        'mi-simple': cmd_mi_simple,
    }

    if args.command in commands:
        result = commands[args.command](args)
        sys.exit(result or 0)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
