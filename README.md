# OMLO: Oblivious ML-Ops

Path ORAM integration for PyTorch training workflows, with experiments that characterize storage-level access-pattern observability and ORAM overhead.

[[_TOC_]]

## Repository Layout

```text
src/                     Core ORAM + training implementations
experiments/run.py       Unified experiment runner (baseline, oram, phases, sweep, sidecar)
experiments/generate.py  Event log and LaTeX table generation
experiments/attack.py    Membership inference attack implementations
experiments/plot.py      Plotting utilities
experiments/test.py      Consolidated test suite
experiments/device.py    Device resolution and sidecar logger
scripts/run.sh           Experiment orchestration
scripts/test.sh          Unified test runner (smoke, attack, macos)
scripts/results.sh       Paper-ready result generation
README.md                Project documentation
```

## Quick Start

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run a fast smoke test:

```bash
bash scripts/test.sh smoke
```

## Main Pipelines

### End-to-end traced runner

```bash
python experiments/run.py experiments \
  --dataset_root dataset_root \
  --output_root experiments_out \
  --device cpu \
  --epochs 3 \
  --visibilities 1.0,0.5,0.25,0.1
```

### Plotting

```bash
python experiments/plot.py robustness \
  --summary experiments_out/summary.csv \
  --output figures/attack_robustness.pdf

python experiments/plot.py privacy \
  --summary experiments_out/summary.csv \
  --output figures/privacy_tradeoff.pdf
```

### Phase runner

```bash
python experiments/run.py phases --phase all
```

### Script-based orchestration

```bash
bash scripts/run.sh experiments
```

## Key Utilities

- Test ORAM integration: `python experiments/test.py sweep`
- Test attack setup: `python experiments/test.py setup`
- Test complete system: `python experiments/test.py system`

## Notes

- `experiments/run.py sidecar` uses the real ORAM trainer path through `src/oram/trainer.py` and `src/oram_storage.py`.
- Some scripts require Linux tooling (`strace`, optional BCC/eBPF) for OS-level tracing workflows.
- If `pyoram` is missing, install requirements and ensure the active environment is the project venv.
