#!/usr/bin/env bash
#
# Master experiment orchestrator for the ORAM thesis project.
#
# Runs all experiments end-to-end:
#   Phase 1 - Baseline training (100 epochs, standard PyTorch)
#   Phase 2 - ORAM training (10 epochs, full 50k dataset)
#   Phase 3 - Batch-size sweep (3 epochs each, 4 batch sizes × baseline+ORAM)
#   Phase 4 - Dataset-size sweep (2 epochs each, 4 dataset sizes × ORAM)
#   Phase 5 - Analysis: plots, overhead report, figures
#
# Usage:
#   ./scripts/run_experiments.sh              # run everything
#   ./scripts/run_experiments.sh --phase 1    # run only phase 1
#   ./scripts/run_experiments.sh --phase 2    # run only phase 2
#   ...
#   ./scripts/run_experiments.sh --phase 5    # run only analysis
#
# Designed to be safe for cron: uses absolute paths, logs output,
# and skips phases whose results already exist (unless --force is set).

set -euo pipefail

# ── Resolve paths ──────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${PROJECT_ROOT}/venv"
PYTHON="${VENV}/bin/python"
LOG_DIR="${PROJECT_ROOT}/results/logs"
LOCK_FILE="${PROJECT_ROOT}/results/.experiment_lock"
TIMESTAMP="$(date +%Y%m%dT%H%M%S)"

# ── Parse arguments ────────────────────────────────────────────
PHASE=""
FORCE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase) PHASE="$2"; shift 2 ;;
        --force) FORCE=1; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Guard against concurrent runs ─────────────────────────────
if [ -f "$LOCK_FILE" ]; then
    LOCK_PID=$(cat "$LOCK_FILE" 2>/dev/null || echo "")
    if [ -n "$LOCK_PID" ] && kill -0 "$LOCK_PID" 2>/dev/null; then
        echo "Another experiment is running (PID $LOCK_PID). Exiting."
        exit 0
    fi
    rm -f "$LOCK_FILE"
fi
mkdir -p "$(dirname "$LOCK_FILE")"
echo $$ > "$LOCK_FILE"
trap 'rm -f "$LOCK_FILE"' EXIT

# ── Setup ──────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"
source "${VENV}/bin/activate"
cd "$PROJECT_ROOT"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
phase_done() { [ -f "$1" ] && [ "$FORCE" -eq 0 ]; }

log "Experiment run started. Timestamp: $TIMESTAMP"
log "Project root: $PROJECT_ROOT"
log "Python: $PYTHON"

# ── Phase 1: Baseline Training ────────────────────────────────
if [ -z "$PHASE" ] || [ "$PHASE" = "1" ]; then
    BASELINE_DIR="results/baseline"
    BASELINE_MARKER="${BASELINE_DIR}/history.json"
    if phase_done "$BASELINE_MARKER"; then
        log "Phase 1 SKIP: baseline results already exist at $BASELINE_MARKER"
    else
        log "Phase 1 START: Baseline training (100 epochs, bs=128)"
        "$PYTHON" experiments/run_baseline.py \
            --epochs 100 \
            --batch-size 128 \
            --output-dir "$BASELINE_DIR" \
            2>&1 | tee "${LOG_DIR}/baseline_${TIMESTAMP}.log"
        log "Phase 1 DONE"
    fi
fi

# ── Phase 2: ORAM Training ────────────────────────────────────
if [ -z "$PHASE" ] || [ "$PHASE" = "2" ]; then
    ORAM_DIR="results/oram"
    ORAM_MARKER="${ORAM_DIR}/history.json"
    if phase_done "$ORAM_MARKER"; then
        log "Phase 2 SKIP: ORAM results already exist at $ORAM_MARKER"
    else
        log "Phase 2 START: ORAM training (10 epochs, 50k samples, bs=128)"
        "$PYTHON" experiments/run_oram.py \
            --epochs 10 \
            --batch-size 128 \
            --output-dir "$ORAM_DIR" \
            2>&1 | tee "${LOG_DIR}/oram_${TIMESTAMP}.log"
        log "Phase 2 DONE"
    fi
fi

# ── Phase 3: Batch-Size Sweep ─────────────────────────────────
if [ -z "$PHASE" ] || [ "$PHASE" = "3" ]; then
    SWEEP_BS_MARKER="results/sweep_batch_size/sweep_summary.json"
    if phase_done "$SWEEP_BS_MARKER"; then
        log "Phase 3 SKIP: batch-size sweep results already exist"
    else
        log "Phase 3 START: Batch-size sweep (bs={32,64,128,256}, 3 epochs)"
        "$PYTHON" experiments/run_sweep.py \
            --sweep batch_size \
            --epochs 3 \
            --output-dir results \
            2>&1 | tee "${LOG_DIR}/sweep_bs_${TIMESTAMP}.log"
        log "Phase 3 DONE"
    fi
fi

# ── Phase 4: Dataset-Size Sweep ───────────────────────────────
if [ -z "$PHASE" ] || [ "$PHASE" = "4" ]; then
    SWEEP_DS_MARKER="results/sweep_dataset_size/sweep_summary.json"
    if phase_done "$SWEEP_DS_MARKER"; then
        log "Phase 4 SKIP: dataset-size sweep results already exist"
    else
        log "Phase 4 START: Dataset-size sweep (n={1k,5k,10k,50k}, 2 epochs)"
        "$PYTHON" experiments/run_sweep.py \
            --sweep dataset_size \
            --epochs 2 \
            --output-dir results \
            2>&1 | tee "${LOG_DIR}/sweep_ds_${TIMESTAMP}.log"
        log "Phase 4 DONE"
    fi
fi

# ── Phase 5: Analysis ─────────────────────────────────────────
if [ -z "$PHASE" ] || [ "$PHASE" = "5" ]; then
    log "Phase 5 START: Analysis and report generation"
    "$PYTHON" experiments/analyze_results.py \
        --baseline results/baseline \
        --oram results/oram \
        --output results/analysis \
        2>&1 | tee "${LOG_DIR}/analysis_${TIMESTAMP}.log"
    log "Phase 5 DONE"
fi

log "All requested phases complete."
