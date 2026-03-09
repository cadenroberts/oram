#!/usr/bin/env python3
"""
Access-pattern leakage demonstration.

Compares plaintext DataLoader access patterns (skewed, predictable)
with ORAM access patterns (uniform by construction).

Outputs:
  - plaintext_access_log.json   per-sample access counts
  - oram_access_log.json        per-sample access counts (logical)
  - leakage_comparison.png      side-by-side histograms
"""

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Sampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.oram_storage import ORAMStorage, load_cifar10_to_oram
from src.oram_dataloader import get_cifar10_transforms, ORAMDataset, ObliviousBatchSampler


class _LoggingSampler(Sampler):
    """Sampler that records every index it yields."""

    def __init__(self, num_samples, batch_size, shuffle=True, drop_last=False):
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.access_log: list = []

    def __iter__(self):
        indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(indices)

        batch = []
        for idx in indices:
            idx_int = int(idx)
            self.access_log.append(idx_int)
            batch.append(idx_int)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return self.num_samples // self.batch_size
        return (self.num_samples + self.batch_size - 1) // self.batch_size


def run_plaintext_logging(num_samples, batch_size, epochs, data_dir):
    """Log access indices from a standard shuffled DataLoader."""
    import torchvision
    import torchvision.transforms as transforms

    dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465],
                                 [0.2023, 0.1994, 0.2010]),
        ]),
    )
    if num_samples < len(dataset):
        dataset = torch.utils.data.Subset(dataset, list(range(num_samples)))

    sampler = _LoggingSampler(len(dataset), batch_size, shuffle=True, drop_last=True)

    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)

    for epoch in range(epochs):
        for _ in loader:
            pass

    return sampler.access_log


def run_oram_logging(num_samples, batch_size, epochs, data_dir, backend="file"):
    """Log logical access indices from ORAM-backed DataLoader."""
    storage = ORAMStorage(num_samples=num_samples, backend=backend)
    load_cifar10_to_oram(storage, data_dir=data_dir, train=True,
                         progress=True, limit=num_samples)

    transform = get_cifar10_transforms(train=True)
    dataset = ORAMDataset(storage, num_samples, transform=transform)
    sampler = _LoggingSampler(num_samples, batch_size, shuffle=True, drop_last=True)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)

    try:
        for epoch in range(epochs):
            for _ in loader:
                pass
    finally:
        storage.close()

    return sampler.access_log


def plot_leakage(plaintext_counts, oram_counts, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    ax = axes[0]
    freqs = sorted(plaintext_counts.values(), reverse=True)
    ax.bar(range(len(freqs)), freqs, width=1.0, color="coral", edgecolor="none")
    ax.set_title("Plaintext Access Frequency")
    ax.set_xlabel("Sample rank (sorted by frequency)")
    ax.set_ylabel("Access count")
    ax.axhline(np.mean(freqs), ls="--", color="gray", label=f"mean={np.mean(freqs):.1f}")
    ax.legend()

    ax = axes[1]
    freqs = sorted(oram_counts.values(), reverse=True)
    ax.bar(range(len(freqs)), freqs, width=1.0, color="steelblue", edgecolor="none")
    ax.set_title("ORAM Logical Access Frequency")
    ax.set_xlabel("Sample rank (sorted by frequency)")
    ax.axhline(np.mean(freqs), ls="--", color="gray", label=f"mean={np.mean(freqs):.1f}")
    ax.legend()

    plt.suptitle("Access Pattern Leakage: Plaintext vs ORAM", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Access pattern leakage demo")
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default="results/leakage")
    parser.add_argument("--backend", type=str, default="file", choices=["file", "ram"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=== Plaintext access logging ===")
    pt_log = run_plaintext_logging(args.num_samples, args.batch_size,
                                   args.epochs, "./data")
    pt_counts = dict(Counter(pt_log))

    print("=== ORAM access logging ===")
    oram_log = run_oram_logging(args.num_samples, args.batch_size,
                                args.epochs, "./data", args.backend)
    oram_counts = dict(Counter(oram_log))

    with open(os.path.join(args.output_dir, "plaintext_access_log.json"), "w") as f:
        json.dump({"counts": pt_counts, "total_accesses": len(pt_log)}, f, indent=2)
    with open(os.path.join(args.output_dir, "oram_access_log.json"), "w") as f:
        json.dump({"counts": oram_counts, "total_accesses": len(oram_log)}, f, indent=2)

    plot_leakage(pt_counts, oram_counts,
                 os.path.join(args.output_dir, "leakage_comparison.png"))

    pt_freqs = list(pt_counts.values())
    oram_freqs = list(oram_counts.values())
    print(f"\nPlaintext: std/mean = {np.std(pt_freqs)/np.mean(pt_freqs):.4f}")
    print(f"ORAM:      std/mean = {np.std(oram_freqs)/np.mean(oram_freqs):.4f}")
    print("(Lower ratio = more uniform = less leakage)")


if __name__ == "__main__":
    main()
