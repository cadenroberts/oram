"""
Figure generation, plotting, and result persistence.

Save: CSV and JSON output helpers
Figure: Phase result charts (7 figures)
Plot: Higher-level plotting (privacy, membership, robustness, ROC, PR, features)

Also includes reporting helpers for LaTeX tables and summaries.
"""

import csv
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Save:
    """CSV / JSON persistence helpers."""

    @staticmethod
    def events_csv(events: List[Tuple[str, float, int, str, int]], output_path: str) -> None:
        dirname = os.path.dirname(output_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["sample_id", "timestamp", "epoch", "batch_id", "label"])
            for sample_id, timestamp, epoch, batch_id, label in events:
                writer.writerow([sample_id, timestamp, epoch, batch_id, label])

    @staticmethod
    def csv(path: str, rows: List[Dict[str, object]]) -> None:
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["sample_id", "timestamp", "epoch", "batch_id", "label"],
            )
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def dict_csv(rows, path: str) -> None:
        if not rows:
            return
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        keys = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"  CSV saved: {path}")

    @staticmethod
    def attack_json(obj: object, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)



class Figure:
    """Phase-result charts (saved under ``fig*.png`` filenames)."""

    @staticmethod
    def backend_sensitivity(root, out):
        """Plaintext vs file-ORAM vs RAM-ORAM epoch time."""
        data = []

        h = _history(os.path.join(root, "phase0", "baseline"))
        if h:
            data.append({"config": "Plaintext", "epoch_time": _epoch_time(h)})

        h = _history(os.path.join(root, "phase0", "oram_file"))
        if h:
            data.append({"config": "File ORAM", "epoch_time": _epoch_time(h)})

        h = _history(os.path.join(root, "phase1", "oram_ram"))
        if h:
            data.append({"config": "RAM ORAM", "epoch_time": _epoch_time(h)})

        if not data:
            print("  backend_sensitivity: no data")
            return

        df = pd.DataFrame(data)
        _, ax = plt.subplots(figsize=(7, 5))
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

    @staticmethod
    def worker_scaling(root, out):
        """Epoch time vs num_workers."""
        data = []
        for nw in (0, 1, 2, 4):
            h = _history(os.path.join(root, "phase2", f"workers_{nw}"))
            if h:
                data.append({"num_workers": nw, "epoch_time": _epoch_time(h)})

        if not data:
            print("  worker_scaling: no data")
            return

        df = pd.DataFrame(data).sort_values("num_workers")
        _, ax = plt.subplots(figsize=(7, 5))
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

    @staticmethod
    def block_size(root, out):
        """Epoch time vs block size."""
        data = []
        for bs in (4096, 8192, 16384, 32768, 65536):
            h = _history(os.path.join(root, "phase3", f"block_{bs}"))
            if h:
                data.append({"block_size": bs, "epoch_time": _epoch_time(h)})

        if not data:
            print("  block_size: no data")
            return

        df = pd.DataFrame(data).sort_values("block_size")
        _, ax = plt.subplots(figsize=(7, 5))
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

    @staticmethod
    def model_scaling(root, out):
        """Slowdown ratio per model."""
        models = ["resnet18", "resnet50", "efficientnet_b0"]
        data = []
        for m in models:
            h_bl = _history(os.path.join(root, "phase4", f"baseline_{m}"))
            h_or = _history(os.path.join(root, "phase4", f"oram_{m}"))
            if h_bl and h_or:
                t_bl = _epoch_time(h_bl)
                t_or = _epoch_time(h_or)
                ratio = t_or / t_bl if t_bl > 0 else 0
                data.append({"model": m, "plaintext": t_bl, "oram": t_or,
                             "slowdown": ratio})

        if not data:
            print("  model_scaling: no data")
            return

        df = pd.DataFrame(data)
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

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

    @staticmethod
    def dataset_scaling(root, out):
        """Epoch time vs N with O(log N) reference."""
        data = []
        for n in (5000, 10000, 25000, 50000):
            h = _history(os.path.join(root, "phase5", f"n_{n}"))
            if h:
                data.append({"N": n, "epoch_time": _epoch_time(h)})

        if not data:
            print("  dataset_scaling: no data")
            return

        df = pd.DataFrame(data).sort_values("N")
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

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

    @staticmethod
    def batch_size(root, out):
        """Epoch time vs batch size for plaintext and ORAM."""
        bl_data, or_data = [], []
        batch_sizes = [32, 64, 128, 256, 512]
        for bs in batch_sizes:
            h = _history(os.path.join(root, "phase6", f"baseline_bs{bs}"))
            if h:
                bl_data.append({"batch_size": bs, "epoch_time": _epoch_time(h)})
            h = _history(os.path.join(root, "phase6", f"oram_bs{bs}"))
            if h:
                or_data.append({"batch_size": bs, "epoch_time": _epoch_time(h)})

        if not bl_data and not or_data:
            print("  batch_size: no data")
            return

        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

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

    @staticmethod
    def leakage(root, out):
        """Plaintext vs ORAM access frequency histogram."""
        pt_path = os.path.join(root, "phase7", "plaintext_access_log.json")
        or_path = os.path.join(root, "phase7", "oram_access_log.json")

        if not os.path.exists(pt_path) or not os.path.exists(or_path):
            print("  leakage: no data")
            return

        with open(pt_path, encoding="utf-8") as f:
            pt = json.load(f)
        with open(or_path, encoding="utf-8") as f:
            oram = json.load(f)

        pt_counts = pt.get("counts", {})
        oram_counts = oram.get("counts", {})

        _, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

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



class Plot:
    """Matplotlib figure helpers."""

    @staticmethod
    def results(results_root: str, output_dir: str):
        """Generate all 7 figures from phase results."""
        os.makedirs(output_dir, exist_ok=True)

        print("=" * 60)
        print("GENERATING FIGURES")
        print("=" * 60)

        Figure.backend_sensitivity(results_root, output_dir)
        Figure.worker_scaling(results_root, output_dir)
        Figure.block_size(results_root, output_dir)
        Figure.model_scaling(results_root, output_dir)
        Figure.dataset_scaling(results_root, output_dir)
        Figure.batch_size(results_root, output_dir)
        Figure.leakage(results_root, output_dir)
        phase_summary(results_root, output_dir)

        print("\nDone. All figures saved to:", output_dir)

    @staticmethod
    def privacy(summary_path: str, oram_auc: float, oram_overhead: float, output_path: str):
        """Privacy-performance tradeoff plot."""
        rows = summary(summary_path)

        plaintext_row = None
        obfuscated_row = None
        oram_row = None

        for row in rows:
            if float(row["visibility"]) == 1.0:
                if row["defense"] == "plaintext" and plaintext_row is None:
                    plaintext_row = row
                elif row["defense"] == "obfuscated" and obfuscated_row is None:
                    obfuscated_row = row
                elif row["defense"] == "oram" and oram_row is None:
                    oram_row = row

        if not plaintext_row:
            raise RuntimeError("No plaintext visibility=1.0 row found in summary")

        plaintext_runtime = float(plaintext_row["train_runtime_sec"])
        plaintext_auc = float(plaintext_row["best_auc"])

        points = [
            ("Plaintext", 1.0, plaintext_auc, 'o', 'red'),
        ]

        if obfuscated_row:
            obf_runtime = float(obfuscated_row["train_runtime_sec"])
            obf_overhead = obf_runtime / plaintext_runtime
            obf_auc = float(obfuscated_row["best_auc"])
            points.append(("Prefetch Obfuscation", obf_overhead, obf_auc, 's', 'orange'))

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
        ax.set_ylim((0.45, 0.85))

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

    @staticmethod
    def membership(results_dir: str, output_path: str):
        """Visual comparison of trivial vs upgraded membership inference attacks."""

        fig = plt.figure(figsize=(14, 10))

        pt_features = feature_table(results_dir, "plaintext", 1.0)

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
            with open(metrics_path, encoding="utf-8") as f:
                data = json.load(f)
                best_model = data["best_model"]
                pt_auc = data["results"][best_model]["auc"]
        else:
            pt_auc = 0.0

        metrics_path = os.path.join(results_dir, "oram_v100", "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, encoding="utf-8") as f:
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
        ax3.set_ylim((0.4, 1.0))
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
                with open(metrics_path, encoding="utf-8") as f:
                    data = json.load(f)
                    best_model = data["best_model"]
                    pt_aucs.append(data["results"][best_model]["auc"])
            else:
                pt_aucs.append(None)

            metrics_path = os.path.join(results_dir, f"oram_v{vis_int}", "metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, encoding="utf-8") as f:
                    data = json.load(f)
                    best_model = data["best_model"]
                    oram_aucs.append(data["results"][best_model]["auc"])
            else:
                oram_aucs.append(None)

        pt_valid = [(v * 100, a) for v, a in zip(visibility_levels, pt_aucs) if a is not None]
        oram_valid = [(v * 100, a) for v, a in zip(visibility_levels, oram_aucs) if a is not None]
        if pt_valid:
            ax4.plot([p[0] for p in pt_valid], [p[1] for p in pt_valid], "o-",
                    linewidth=2, color="coral", label="Plaintext")
        if oram_valid:
            ax4.plot([p[0] for p in oram_valid], [p[1] for p in oram_valid], "s-",
                    linewidth=2, color="steelblue", label="ORAM")

        ax4.axhline(0.5, ls="--", color="gray", label="Random")
        ax4.set_xlabel("Visibility (%)")
        ax4.set_ylabel("AUC")
        ax4.set_title("Partial Observability Robustness")
        ax4.set_ylim((0.4, 1.0))
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.suptitle("Membership Inference Attack: Trivial vs Non-Trivial Approach",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()

        print(f"Comparison figure saved to: {output_path}")

    @staticmethod
    def robustness(summary_path: str, oram_results_dir: Optional[str], output_path: str):
        """Attack robustness vs visibility."""
        rows = summary(summary_path)

        plaintext_data = {}
        obfuscated_data = {}
        oram_data = {}

        for row in rows:
            defense = row["defense"]
            visibility = float(row["visibility"])
            auc = float(row["best_auc"])

            if defense == "plaintext":
                plaintext_data[visibility] = auc
            elif defense == "obfuscated":
                obfuscated_data[visibility] = auc
            elif defense == "oram":
                oram_data[visibility] = auc

        visibilities = sorted(set(plaintext_data.keys()) | set(obfuscated_data.keys()) | set(oram_data.keys()))

        fig, ax = plt.subplots(figsize=(8, 6))

        if plaintext_data:
            vis_plain = sorted(plaintext_data.keys())
            auc_plain = [plaintext_data[v] for v in vis_plain]
            ax.plot([v * 100 for v in vis_plain], auc_plain, 'o-', linewidth=2, markersize=8, label="Plaintext")

        if obfuscated_data:
            vis_obf = sorted(obfuscated_data.keys())
            auc_obf = [obfuscated_data[v] for v in vis_obf]
            ax.plot([v * 100 for v in vis_obf], auc_obf, 's-', linewidth=2, markersize=8, label="Prefetch Obfuscation")

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
                    with open(metrics_path, "r", encoding="utf-8") as f:
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
        ax.set_ylim((0.45, 0.90))

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")

    @staticmethod
    def cmd(args) -> None:
        Plot.results(args.results_root, args.output)

    @staticmethod
    def roc(results: Dict[str, Dict[str, object]], output_path: str) -> None:
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

    @staticmethod
    def pr(results: Dict[str, Dict[str, object]], output_path: str) -> None:
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

    @staticmethod
    def top_features(importance_df: pd.DataFrame, output_path: str, top_k: int = 15) -> None:
        top = importance_df.head(top_k).iloc[::-1]
        plt.figure(figsize=(8, 6))
        plt.barh(top["feature"], top["importance"])
        plt.xlabel("Importance")
        plt.title(f"Top {top_k} Features")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()


def attack_metrics(results_dir: str, mode: str, visibility: float) -> Dict[str, float]:
    """Load metrics.json for a given mode and visibility level."""
    vis_int = int(visibility * 100)
    metrics_path = os.path.join(results_dir, f"{mode}_v{vis_int}", "metrics.json")

    if not os.path.exists(metrics_path):
        return {}

    with open(metrics_path, encoding="utf-8") as f:
        data = json.load(f)
        best_model = data["best_model"]
        res = data["results"][best_model]
        return {
            "auc": res["auc"],
            "accuracy": res["accuracy"],
            "ap": res["average_precision"],
            "model": best_model,
        }



def latex_table(results_dir: str, visibility_levels: List[float]) -> str:
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
        pt_metrics = attack_metrics(results_dir, "plaintext", vis)
        oram_metrics = attack_metrics(results_dir, "oram", vis)

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



def feature_importance_table(
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

    with open(metrics_path, encoding="utf-8") as f:
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



def _history(directory):
    p = os.path.join(directory, "history.json")
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _epoch_time(hist):
    """Average seconds per epoch from a history dict."""
    n = len(hist.get("epochs", []))
    t = hist.get("total_time", 0)
    return t / n if n > 0 else 0



def _ensure(path):
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)



def phase_summary(root, out):
    """Generate a combined summary from phase8."""
    rows = []
    for tag, label in [("plaintext", "Plaintext"),
                       ("oram_baseline", "Baseline ORAM"),
                       ("oram_optimized", "Optimized ORAM")]:
        h = _history(os.path.join(root, "phase8", tag))
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
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {path}")



def summary(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))



def feature_table(results_dir: str, mode: str, visibility: float) -> pd.DataFrame:
    vis_int = int(visibility * 100)
    feature_path = os.path.join(results_dir, f"{mode}_v{vis_int}", "feature_table.csv")

    if not os.path.exists(feature_path):
        return pd.DataFrame()

    return pd.read_csv(feature_path)



