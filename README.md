# OMLO: Oblivious ML-Ops

Path ORAM integration for PyTorch training. Experiments characterize storage-level access-pattern observability and ORAM overhead.

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
