# OMLO

## Purpose

Privacy-preserving ML training system that integrates Path ORAM into PyTorch data pipelines. Eliminates data-dependent memory access patterns during CIFAR-10 training. A membership inference attack pipeline empirically evaluates leakage under plaintext, obfuscated, and ORAM-backed access.

## Architecture

```mermaid
flowchart TB
    RunSh["run.sh<br/>CLI orchestrator"]
    RunPy["src/run.py<br/>Unified CLI"]
    Oram["src/oram.py<br/>ORAM storage + training"]
    Pipeline["src/pipeline.py<br/>Experiment orchestration"]
    Attack["src/attack.py<br/>Membership inference"]
    Figures["src/figures.py<br/>Plots + persistence"]
    Profiler["src/profiler.py<br/>Timing + memory"]
    TestCore["src/test_core.py<br/>24 unit tests"]
    CIFAR["data/<br/>CIFAR-10"]
    Results["results/<br/>Phase outputs + figures"]

    RunSh -->|subcommand dispatch| RunPy
    RunPy -->|baseline / oram training| Oram
    RunPy -->|phased benchmarks| Pipeline
    RunPy -->|mi attack| Attack
    RunPy -->|plot generation| Figures
    Oram -->|profiling| Profiler
    Oram -->|reads| CIFAR
    Pipeline -->|training| Oram
    Pipeline -->|attack| Attack
    Attack -->|plots| Figures
    TestCore -->|validates| Oram
    TestCore -->|validates| Attack
    TestCore -->|validates| Profiler
    RunSh --> Results
```

## Files

| File | Purpose |
|------|---------|
| `src/run.py` | Unified CLI: training, event generation, trace conversion, attack, plotting |
| `src/oram.py` | Path ORAM storage (PyORAM), ORAM-backed DataLoader, baseline/ORAM training loops |
| `src/pipeline.py` | Experiment orchestration, strace/eBPF/fs_usage trace parsing, phased benchmarks, sweeps |
| `src/attack.py` | Membership inference: 40+ feature engineering, sklearn model comparison, partial observability |
| `src/figures.py` | Phase-result plots (7 figures), CSV/JSON persistence, LaTeX table generation |
| `src/profiler.py` | Wall-clock timing, memory profiling, overhead breakdown, JSON report output |
| `src/test_core.py` | 24 unit tests covering ORAM storage, profiler, attack features, events, PyORAM API |
| `run.sh` | Shell orchestrator: venv setup, phased experiments, OS-level tracing, smoke tests |
| `requirements.txt` | Python dependencies (torch, pyoram, sklearn, matplotlib, xgboost, etc.) |
| `.gitlab-ci.yml` | CI: runs `./run.sh smoke` on Python 3.10 |
| `reports/` | Paper LaTeX source (LLNCS class) |

## Entry Points

| Command | Effect |
|---------|--------|
| `python src/run.py baseline` | Standard CIFAR-10 training (ResNet-18, SGD) |
| `python src/run.py oram` | ORAM-backed CIFAR-10 training |
| `python src/run.py event --mode plaintext` | Generate synthetic plaintext access-pattern event log |
| `python src/run.py event --mode oram` | Generate synthetic ORAM access-pattern event log |
| `python src/run.py mi --input X --output_dir Y` | Run membership inference attack on event log |
| `python src/run.py phases --phase all` | Run all 9 experiment phases (0-8) |
| `python src/run.py plot` | Generate all phase-result figures |
| `python src/run.py leakage` | Generate plaintext vs ORAM access frequency logs |
| `python src/run.py inference` | Full plaintext + ORAM attack sweep with summary |
| `bash run.sh smoke` | Unit tests + baseline + ORAM validation (2 epochs) |
| `bash run.sh experiments` | 5-step pipeline: baseline, ORAM, sweeps, plot |
| `bash run.sh pipeline` | All 9 phases + plot via `phases --phase all` |
| `bash run.sh trace` | OS-level trace capture (Linux, requires strace/eBPF) |
| `bash run.sh visibility` | Partial observability sweep (4 visibility levels) |
| `bash run.sh attack` | Plaintext vs ORAM attack comparison test |
| `bash run.sh results` | Paper-ready membership inference evaluation |
| `bash run.sh macos` | macOS physical-access audit with fs_usage |

## Verification

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cd src && python test_core.py
cd .. && bash run.sh smoke
```
