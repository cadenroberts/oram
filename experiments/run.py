#!/usr/bin/env python3
"""
Unified experiment runner combining all run modules:
- baseline: Standard PyTorch training
- experiments: Full pipeline with trace collection and attacks
- inference: Membership inference evaluation
- oram: ORAM-integrated training
- phases: Phased evaluation (0-8)
- sidecar: ORAM training with sidecar logging
- sweep: Parameter sweep experiments

Usage:
    python experiments/run.py baseline --epochs 100 --batch-size 128
    python experiments/run.py experiments --dataset_root data --output_root results
    python experiments/run.py inference --train_size 20000 --epochs 3
    python experiments/run.py oram --epochs 100 --batch-size 128
    python experiments/run.py phases --phase 0
    python experiments/run.py sidecar --epochs 3 --sidecar_path batch.csv
    python experiments/run.py sweep --sweep all --epochs 3
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.baseline import run_baseline_training
from src.oram.trainer import run_oram_training, ORAMTrainer
from experiments.device import SidecarLogger, resolve_torch_device

import pandas as pd


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
                "strace is not available on macOS. Use scripts/test.sh macos "
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
            "test.py", "train",
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
            "test.py", "train",
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
        "test.py",
        "sidecar",
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

    cmd = [
        sys.executable,
        "attack.py",
        "--input", input_csv,
        "--output_dir", attack_dir,
        "--visibility", str(visibility),
        "--random_state", "42",
    ]
    proc = run_subprocess(
        cmd,
        stdout_path=os.path.join(attack_dir, "attack_stdout.log"),
        stderr_path=os.path.join(attack_dir, "attack_stderr.log"),
    )
    wait_success(proc, f"attack visibility={visibility}")

    metrics_path = os.path.join(attack_dir, "metrics.json")
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    return metrics


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
        "attack_script": "attack.py",
        "trace_converter": "test.py sidecar",
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
                "experiments/generate.py",
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
                "experiments/generate.py",
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
        run_command(
            [
                sys.executable,
                "experiments/attack.py",
                "--input", plaintext_log,
                "--output_dir", pt_out,
                "--visibility", str(vis),
                "--random_state", str(args.random_state),
            ],
            f"STEP 3.{int(vis*100)}: Attack plaintext log at visibility={vis}"
        )

        metrics_path = os.path.join(pt_out, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                metrics = json.load(f)
                best_model = metrics["best_model"]
                results_plaintext[vis] = {
                    "auc": metrics["results"][best_model]["auc"],
                    "accuracy": metrics["results"][best_model]["accuracy"],
                    "ap": metrics["results"][best_model]["average_precision"],
                    "model": best_model,
                }

        oram_out = os.path.join(args.output_dir, f"oram_v{int(vis*100)}")
        run_command(
            [
                sys.executable,
                "experiments/attack.py",
                "--input", oram_log,
                "--output_dir", oram_out,
                "--visibility", str(vis),
                "--random_state", str(args.random_state),
            ],
            f"STEP 4.{int(vis*100)}: Attack ORAM log at visibility={vis}"
        )

        metrics_path = os.path.join(oram_out, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                metrics = json.load(f)
                best_model = metrics["best_model"]
                results_oram[vis] = {
                    "auc": metrics["results"][best_model]["auc"],
                    "accuracy": metrics["results"][best_model]["accuracy"],
                    "ap": metrics["results"][best_model]["average_precision"],
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
    leakage_script = os.path.join(exp_dir, "test.py")
    subprocess.check_call([
        sys.executable, leakage_script,
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
    trainer: ORAMTrainer,
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


def _run_training_epochs(trainer: ORAMTrainer, sidecar: SidecarLogger, epochs: int) -> tuple[dict, float, float]:
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

    trainer = ORAMTrainer(
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


def main():
    parser = argparse.ArgumentParser(
        description='Unified experiment runner',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
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
    
    args = parser.parse_args()
    
    if args.command == 'baseline':
        baseline_main(args)
    elif args.command == 'experiments':
        experiments_main(args)
    elif args.command == 'inference':
        inference_main(args)
    elif args.command == 'oram':
        oram_main(args)
    elif args.command == 'phases':
        phases_main(args)
    elif args.command == 'sidecar':
        sidecar_main(args)
    elif args.command == 'sweep':
        sweep_main(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
