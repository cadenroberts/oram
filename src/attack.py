"""
Membership inference attack infrastructure.

Build: Feature engineering and model pipelines
SimpleCNN: Small CNN for simple attacks
Functions: Feature extraction, model evaluation, attack orchestration

Includes upgraded_attack (full MI pipeline) and simple_attack (frequency-based).
"""

import csv
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
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

from figures import Save, Plot
from oram import IndexedDataset

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False


ATTACK_REQUIRED_COLUMNS = {"sample_id", "timestamp", "epoch", "batch_id", "label"}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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




class Build:
    """Construct derived tables, streams, and model pipelines."""

    @staticmethod
    def mixed_access_report(events_csv: str) -> Dict[str, object]:
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

    @staticmethod
    def observed_stream(
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

    @staticmethod
    def feature_table(df: pd.DataFrame) -> pd.DataFrame:
        epoch_sizes, epoch_time_spans, epoch_start_times = compute_epoch_batch_normalizers(df)
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
                e_start = epoch_start_times[int(epoch)]
                span = epoch_time_spans[int(epoch)]

                norm_epoch_freqs.append(len(eg) / esize)
                rel_positions = (e_times - e_start) / span
                first_pos_in_epoch.append(float(np.min(rel_positions)))
                mean_pos_in_epoch.append(float(np.mean(rel_positions)))
                last_pos_in_epoch.append(float(np.max(rel_positions)))

            rel_global_positions = (timestamps - global_t_min) / global_span

            epoch_switches = float(np.sum(np.diff(epochs) != 0)) if len(epochs) > 1 else 0.0

            row: Dict[str, object] = {
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

    @staticmethod
    def attack_models(random_state: int) -> Dict[str, Pipeline]:
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




def maybe_reorder(events: List[Dict[str, object]], window: int, rng: random.Random) -> List[Dict[str, object]]:
    if window <= 1:
        return events

    out: List[Dict[str, object]] = []
    for start in range(0, len(events), window):
        chunk = events[start:start + window]
        rng.shuffle(chunk)
        out.extend(chunk)
    return out




def simple_attack(
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

    X_train_atk, X_test_atk, y_train_atk, y_test_atk = train_test_split(
        X, y_true, test_size=0.3, random_state=seed, stratify=y_true
    )

    print("Training attack model...")

    clf = LogisticRegression()
    clf.fit(X_train_atk, y_train_atk)

    preds = clf.predict(X_test_atk)
    probs = clf.predict_proba(X_test_atk)[:, 1]

    acc = accuracy_score(y_test_atk, preds)
    auc = roc_auc_score(y_test_atk, probs)

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
    plt.savefig(f"{output_dir}/figures/access_count_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {output_dir}/figures/access_count_distribution.png")

    with open(f"{output_dir}/membership_results.txt", "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"AUC: {auc:.4f}\n")
    print(f"Saved results: {output_dir}/membership_results.txt")

    return {"accuracy": acc, "auc": auc}




def events(path: str) -> pd.DataFrame:
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




def compute_epoch_batch_normalizers(df: pd.DataFrame) -> Tuple[Dict[int, int], Dict[int, float], Dict[int, float]]:
    epoch_sizes: Dict[int, int] = df.groupby("epoch").size().to_dict()
    epoch_time_spans: Dict[int, float] = {}
    epoch_start_times: Dict[int, float] = {}

    for epoch, g in df.groupby("epoch"):
        t_min = float(g["timestamp"].min())
        t_max = float(g["timestamp"].max())
        epoch_time_spans[int(epoch)] = max(t_max - t_min, 1e-9)
        epoch_start_times[int(epoch)] = t_min

    return epoch_sizes, epoch_time_spans, epoch_start_times




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




def split_xy(feature_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    y = feature_df["label"].to_numpy(dtype=int)
    feature_cols = [c for c in feature_df.columns if c not in {"sample_id", "label"}]
    X = feature_df[feature_cols].copy()
    return X, y, feature_cols




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




def upgraded_attack(
    input_path: str,
    output_dir: str,
    test_size: float = 0.3,
    random_state: int = 42,
    visibility: float = 1.0,
    max_samples: Optional[int] = None,
) -> Dict[str, object]:
    """Non-trivial membership inference from access-pattern logs."""
    ensure_dir(output_dir)

    df = events(input_path)
    df = subsample_visibility(df, visibility=visibility, random_state=random_state)
    df = maybe_limit_samples(df, max_samples=max_samples, random_state=random_state)

    if df.empty:
        raise ValueError("No events remain after visibility/max_samples filtering.")

    feature_df = Build.feature_table(df)

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

    models = Build.attack_models(random_state)
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
    Save.attack_json(summary, os.path.join(output_dir, "metrics.json"))

    Plot.roc(results, os.path.join(output_dir, "roc_curve.png"))
    Plot.pr(results, os.path.join(output_dir, "pr_curve.png"))
    Plot.top_features(best_importance, os.path.join(output_dir, "top_features.png"))

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




