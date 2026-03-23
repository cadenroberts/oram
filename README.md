# OMLO — Privacy-Preserving ML Training System

OMLO is a privacy-preserving ML training system that integrates Path ORAM into GPU-accelerated PyTorch pipelines to eliminate data-dependent memory access patterns.

It introduces a data access abstraction layer that enforces oblivious access patterns while preserving GPU training throughput under constrained memory and bandwidth conditions.

## System Overview

OMLO inserts an ORAM-backed data access layer into the ML training pipeline:

```text
Dataset
   ↓
ORAM Layer (Path ORAM)
   ↓
DataLoader
   ↓
GPU Training (PyTorch)
   ↓
Evaluation + Attack Pipeline
```

This ensures memory access patterns are independent of input data, mitigating access-pattern leakage.

## Key Challenges

- ORAM introduces O(log N) access overhead → reduces effective training throughput
- GPU underutilization due to serialized memory access patterns
- Memory pressure from ORAM tree structures during training
- Balancing privacy guarantees vs training efficiency

## Design Decisions

- **Path ORAM** chosen for simplicity and well-defined access guarantees
- Integrated at the **data access layer** to preserve compatibility with PyTorch training loops
- Built an **attack pipeline (membership inference)** to empirically evaluate leakage

## Results

- Preserved 93% CIFAR-10 accuracy (ResNet-18) under full ORAM constraints
- Measured 1.5–2× GPU memory overhead from ORAM tree structures during training
- O(log N) access scaling confirmed across dataset sizes up to 50K samples
- Membership inference attack AUC reduced under ORAM vs plaintext access patterns

## Failure Modes

- GPU underutilization when ORAM access latency dominates compute time
- OOM under large ORAM tree sizes combined with full batch training
- Visibility sweep reveals partial observability still leaks access patterns at >60% coverage
- Serialized ORAM reads bottleneck multi-worker DataLoader throughput

## Structure

```
src/run.py          Unified runner (training, generation, attack, plotting)
src/oram.py         ORAM storage, dataloader, trainer
run.sh              Shell orchestrator (repo root)
reports/            Paper PDFs and LaTeX sources
```

## Quick Start

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
bash run.sh smoke
```

## Commands

### Training

```bash
python src/run.py baseline --epochs 5 --batch_size 128
python src/run.py oram --epochs 5 --batch_size 128
```

### Event Generation

```bash
python src/run.py event --train_size 10000 --epochs 5 --output events.csv --mode plaintext
```

### Membership Inference

```bash
python src/run.py mi --input events.csv --output_dir results/attack --visibility 1.0
```

### Plotting

```bash
python src/run.py privacy --summary results/summary.csv --output figures/privacy.pdf
python src/run.py robustness --summary results/summary.csv --output figures/robustness.pdf
```

### Full Pipelines

```bash
bash run.sh experiments    # all phases
bash run.sh visibility     # partial observability sweep
bash run.sh trace          # OS-level trace (Linux)
bash run.sh smoke          # quick test
```

## Notes

- Linux tracing requires `strace` or BCC/eBPF
- macOS tracing uses `fs_usage` (requires sudo)
- XGBoost optional but recommended for best attack performance
- ORAM uses PyORAM Path ORAM implementation
