# Patchset Summary

## PHASE 0 — BASELINE SNAPSHOT

**Current branch:** main  
**Current HEAD:** bfecf86684862d2ed06be34447c2361de04e7bf3  
**Tracked file count:** 39 files (excluding .git, venv, __pycache__, results)

**Primary entry points:**
- `experiments/run_baseline.py` — Standard PyTorch training baseline
- `experiments/run_oram.py` — ORAM-integrated training
- `experiments/run_sweep.py` — Batch-size and dataset-size sweep experiments
- `experiments/analyze_results.py` — Overhead analysis and visualization
- `scripts/run_experiments.sh` — Master orchestrator for all phases

**How the project runs:**
1. Install dependencies: `pip install -r requirements.txt`
2. Run full experiment pipeline: `./scripts/run_experiments.sh`
3. Run individual phases: `./scripts/run_experiments.sh --phase N`
4. Run individual experiments: `python experiments/run_baseline.py`, `python experiments/run_oram.py`, etc.

**Current state:**
- Python package structure with `src/` modules and `experiments/` scripts
- ORAM storage layer wraps PyORAM Path ORAM for CIFAR-10 samples
- Profiler tracks overhead across six categories (I/O, crypto, shuffle, serialize, compute, memory)
- ResNet-18 model adapted for CIFAR-10 (32×32 images)
- Experiment orchestrator with 5 phases: baseline, ORAM, batch sweep, dataset sweep, analysis

---

## CHANGES

### Phase 1 — Technical Audit (Commit: 0a8f20b)
**Added:**
- `REPO_AUDIT.md` — Comprehensive repository audit covering purpose, entry points, dependencies, configuration, data flow, determinism risks, observability, test state, reproducibility, security, and ranked improvements (P0/P1/P2)
- `PATCHSET_SUMMARY.md` — This file

**Commit type:** Clarifying (insertions only)

### Phase 2 — Cleaning (Commit: 4b32200)
**Deleted:**
- `PROGRESS_REPORT.md` — Academic progress report (not core documentation)
- `.env.example` — Unused configuration file
- `mdc/oram.genesis.yaml` — Experimental MDC metadata coordination
- `experiments/mdc_snapshot.py` — Experimental MDC snapshot script
- `rebuild.sh` — Redundant setup script
- `scripts/install_cron.sh` — Operational infrastructure (not core)

**Commit type:** Cleaning (deletions only)

### Phase 3 — Documentation Rebuild (Commit: 5f9a072)
**Added:**
- `DESIGN_DECISIONS.md` — 10 ADR entries covering Path ORAM selection, block size, single-threaded DataLoader, profiler architecture, ResNet-18 adaptation, SGD optimizer, oblivious batch sampler, test set loader, experiment orchestrator, JSON export
- `EVAL.md` — Evaluation methodology with correctness definition, performance expectations, measurable commands, pass/fail criteria, theoretical comparison, optimization targets
- `DEMO.md` — Complete demo walkthrough with quick demo (15 min), full suite (40 hours), troubleshooting, expected outputs

**Modified:**
- `README.md` — Complete rewrite with neutral summary, what it does, architecture, design tradeoffs, evaluation, demo, repository layout, limitations/scope, references

**Commit type:** Refactoring (mixed changes)

### Phase 4 — Verification Implementation (Current)
**Added:**
- `scripts/demo.sh` — Non-interactive demo script that runs 2-epoch baseline + ORAM smoke test, verifies convergence, exits with DEMO_OK or non-zero

**Verification command:**
```bash
./scripts/demo.sh
```

**Expected output:**
```
DEMO_OK
```

**Status:** Script created and syntax-validated, but not executed (15-minute runtime)
