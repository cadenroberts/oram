# OMLO: Oblivious ML-Ops

Path ORAM integration for PyTorch training workflows. Quantifies overhead of hiding data-dependent access patterns during ML training.

[[_TOC_]]

## Quick start

**Prerequisites:** Python 3.8+, 20GB disk, 4GB RAM

```bash
git clone <repo-url> && cd OMLO
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
./scripts/demo.sh
```

Demo runs 2 epochs each (baseline + ORAM), ~15 min CPU. Output: `results/demo/`.

**Full pipeline (~40 hr):** `./scripts/run_experiments.sh`

| Issue | Fix |
|-------|-----|
| `No module named 'pyoram'` | `pip install -r requirements.txt` |
| CUDA OOM | `--batch-size 64` or `--device cpu` |
| Lock file | `rm results/.experiment_lock` |

## Architecture

**Standard:** `CIFAR-10 → torchvision → DataLoader (4 workers) → GPU → ResNet-18`  
**ORAM:** `CIFAR-10 → ORAMStorage (AES 4KB blocks) → ORAMDataset → DataLoader (0 workers) → GPU → ResNet-18`

`num_workers=0` required (PyORAM not thread-safe). O(log N) blocks per access.

## Results

| Metric | Baseline | ORAM | Overhead |
|--------|----------|------|----------|
| Per-sample load | ~0.01 ms | ~5–15 ms | 500–1500× |
| Per-epoch wall time | ~30 s | ~45–90 min | 90–180× |
| Peak memory | ~1.2 GB | ~1.8–2.5 GB | 1.5–2× |

ORAM block I/O 60–70%, serialization 5–10%, compute 15–25%. Scaling: O(log N).

## Layout

```
src/           # oram_storage, oram_dataloader, oram_trainer, baseline_trainer
experiments/   # run_baseline, run_oram, run_sweep, analyze_results
scripts/       # demo.sh, run_experiments.sh
report.tex     # pdflatex report.tex (run twice)
```

## Reference

**Report:** `pdflatex report.tex` (twice for refs)

**Limitations:** Single model/dataset (ResNet-18, CIFAR-10), CPU-only, single-threaded ORAM, batch shuffle not oblivious.

**Future:** Concurrent ORAM (SONIC), oblivious shuffling, larger models, GPU crypto.
