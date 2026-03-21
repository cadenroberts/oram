#!/usr/bin/env python3
"""
generate.py

Consolidated module for generating access-pattern event logs and LaTeX tables
for membership inference attack experiments.

Combines functionality from:
- attack.py: LaTeX table generation from attack results
- event.py: Access-pattern event log generation
- partial.py: Event log generation with partial observability
- reference.py: Reference implementation of event log generation

Usage:
    python experiments/generate.py attack \
        --results_dir results/paper_membership_attack \
        --output tables/membership_inference.tex

    python experiments/generate.py event \
        --train_size 20000 \
        --holdout_size 20000 \
        --epochs 3 \
        --batch_size 128 \
        --output events.csv \
        --mode plaintext

    python experiments/generate.py partial \
        --train_size 20000 \
        --holdout_size 20000 \
        --visibility 0.5 \
        --full_output events_full.csv \
        --observed_output events_observed.csv

    python experiments/generate.py reference
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms as transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.oram_storage import ORAMStorage, load_cifar10_to_oram
from src.oram.dataloader import ORAMDataset, get_cifar10_transforms


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
    
    import pandas as pd
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
    
    Scenario:
        - Training batches: members accessed via shuffled DataLoader
        - Probe batches: randomly inject non-members into some training batches
        - Both classes appear in log with different access patterns
    
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
    
    Scenario:
        - Training batches: members accessed via shuffled DataLoader
        - Probe batches: randomly inject non-members into some training batches
        - Both classes appear in log with different access patterns
        - ORAM randomizes physical accesses, breaking the signal
    
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


def cmd_attack(args) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate access-pattern event logs and LaTeX tables for membership inference experiments."
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    attack_parser = subparsers.add_parser("attack", help="Generate LaTeX tables from attack results")
    attack_parser.add_argument("--results_dir", type=str, required=True,
                              help="Directory containing attack results.")
    attack_parser.add_argument("--output", type=str, required=True,
                              help="Output .tex file path.")
    attack_parser.add_argument("--visibility_levels", type=float, nargs="+",
                              default=[1.0, 0.5, 0.25, 0.1],
                              help="Visibility levels to include in table.")
    attack_parser.add_argument("--include_features", action="store_true",
                              help="Also generate feature importance tables.")
    
    event_parser = subparsers.add_parser("event", help="Generate access-pattern event logs")
    event_parser.add_argument("--train_size", type=int, default=20000, help="Number of training samples (members).")
    event_parser.add_argument("--holdout_size", type=int, default=20000, help="Number of holdout samples (non-members).")
    event_parser.add_argument("--epochs", type=int, default=3, help="Number of epochs.")
    event_parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    event_parser.add_argument("--probe_batch_prob", type=float, default=0.2,
                             help="Probability that a training batch includes probe access (non-members).")
    event_parser.add_argument("--probe_mix_ratio", type=float, default=0.3,
                             help="Fraction of batch replaced with non-members during probe access.")
    event_parser.add_argument("--output", type=str, required=True, help="Output CSV path.")
    event_parser.add_argument("--mode", type=str, default="plaintext", choices=["plaintext", "oram"],
                             help="Generate plaintext or ORAM-backed log.")
    event_parser.add_argument("--backend", type=str, default="ram", choices=["file", "ram"],
                             help="ORAM backend (only used when mode=oram).")
    event_parser.add_argument("--data_dir", type=str, default="./data", help="CIFAR-10 data directory.")
    event_parser.add_argument("--random_state", type=int, default=42, help="Random seed.")
    
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
    
    reference_parser = subparsers.add_parser("reference", help="Run reference implementation")
    
    args = parser.parse_args()
    
    if args.command == "attack":
        cmd_attack(args)
    elif args.command == "event":
        cmd_event(args)
    elif args.command == "partial":
        cmd_partial(args)
    elif args.command == "reference":
        cmd_reference(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
