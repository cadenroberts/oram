#!/usr/bin/env bash
#
# Unified experiment runner for the ORAM thesis project.
#
# Subcommands:
#   experiments [--phase N] [--force]   Run training / sweep / analysis phases
#   pipeline                            Run all phases + analysis (phases.py path)
#   trace                               OS-level trace capture pipeline
#   visibility                          Visibility sweep with partial observability
#
# Usage:
#   ./scripts/run.sh experiments              # all 5 phases
#   ./scripts/run.sh experiments --phase 2    # only phase 2
#   ./scripts/run.sh experiments --force      # re-run even if results exist
#   ./scripts/run.sh pipeline                 # phases.py + plot
#   ./scripts/run.sh trace                    # real syscall trace pipeline
#   ./scripts/run.sh visibility               # visibility sweep

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${PROJECT_ROOT}/venv"
PYTHON="${VENV}/bin/python"
LOG_DIR="${PROJECT_ROOT}/results/logs"
LOCK_FILE="${PROJECT_ROOT}/results/.experiment_lock"
TIMESTAMP="$(date +%Y%m%dT%H%M%S)"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ── Subcommand dispatch ──────────────────────────────────────

usage() {
    echo "Usage: $0 <subcommand> [options]"
    echo ""
    echo "Subcommands:"
    echo "  experiments [--phase N] [--force]   Training / sweep / analysis phases"
    echo "  pipeline                            All phases via phases.py + plot"
    echo "  trace                               OS-level trace capture pipeline"
    echo "  visibility                          Visibility sweep (partial observability)"
    exit 1
}

[[ $# -lt 1 ]] && usage
SUBCMD="$1"; shift

# ── Helpers ───────────────────────────────────────────────────

acquire_lock() {
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
}

ensure_venv() {
    if [ ! -f "${VENV}/bin/activate" ]; then
        echo "Error: virtualenv not found at ${VENV}"
        exit 1
    fi
    source "${VENV}/bin/activate"
}

phase_done() { [ -f "$1" ] && [ "${FORCE:-0}" -eq 0 ]; }

# ══════════════════════════════════════════════════════════════
# experiments — phased training, sweeps, and analysis
# ══════════════════════════════════════════════════════════════

cmd_experiments() {
    local PHASE="" FORCE=0
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --phase) PHASE="$2"; shift 2 ;;
            --force) FORCE=1; shift ;;
            *) echo "Unknown argument: $1"; exit 1 ;;
        esac
    done

    acquire_lock
    mkdir -p "$LOG_DIR"
    ensure_venv
    cd "$PROJECT_ROOT"

    log "Experiment run started. Timestamp: $TIMESTAMP"
    log "Project root: $PROJECT_ROOT"
    log "Python: $PYTHON"

    # Phase 1: Baseline Training
    if [ -z "$PHASE" ] || [ "$PHASE" = "1" ]; then
        BASELINE_DIR="results/baseline"
        BASELINE_MARKER="${BASELINE_DIR}/history.json"
        if phase_done "$BASELINE_MARKER"; then
            log "Phase 1 SKIP: baseline results already exist at $BASELINE_MARKER"
        else
            log "Phase 1 START: Baseline training (100 epochs, bs=128)"
            "$PYTHON" experiments/run.py baseline \
                --epochs 100 \
                --batch-size 128 \
                --output-dir "$BASELINE_DIR" \
                2>&1 | tee "${LOG_DIR}/baseline_${TIMESTAMP}.log"
            log "Phase 1 DONE"
        fi
    fi

    # Phase 2: ORAM Training
    if [ -z "$PHASE" ] || [ "$PHASE" = "2" ]; then
        ORAM_DIR="results/oram"
        ORAM_MARKER="${ORAM_DIR}/history.json"
        if phase_done "$ORAM_MARKER"; then
            log "Phase 2 SKIP: ORAM results already exist at $ORAM_MARKER"
        else
            log "Phase 2 START: ORAM training (10 epochs, 50k samples, bs=128)"
            "$PYTHON" experiments/run.py sweep \
                --epochs 10 \
                --batch-size 128 \
                --output-dir "$ORAM_DIR" \
                2>&1 | tee "${LOG_DIR}/oram_${TIMESTAMP}.log"
            log "Phase 2 DONE"
        fi
    fi

    # Phase 3: Batch-Size Sweep
    if [ -z "$PHASE" ] || [ "$PHASE" = "3" ]; then
        SWEEP_BS_MARKER="results/sweep_batch_size/sweep_summary.json"
        if phase_done "$SWEEP_BS_MARKER"; then
            log "Phase 3 SKIP: batch-size sweep results already exist"
        else
            log "Phase 3 START: Batch-size sweep (bs={32,64,128,256}, 3 epochs)"
            "$PYTHON" experiments/run.py sweep \
                --sweep batch_size \
                --epochs 3 \
                --output-dir results \
                2>&1 | tee "${LOG_DIR}/sweep_bs_${TIMESTAMP}.log"
            log "Phase 3 DONE"
        fi
    fi

    # Phase 4: Dataset-Size Sweep
    if [ -z "$PHASE" ] || [ "$PHASE" = "4" ]; then
        SWEEP_DS_MARKER="results/sweep_dataset_size/sweep_summary.json"
        if phase_done "$SWEEP_DS_MARKER"; then
            log "Phase 4 SKIP: dataset-size sweep results already exist"
        else
            log "Phase 4 START: Dataset-size sweep (n={1k,5k,10k,50k}, 2 epochs)"
            "$PYTHON" experiments/run.py sweep \
                --sweep dataset_size \
                --epochs 2 \
                --output-dir results \
                2>&1 | tee "${LOG_DIR}/sweep_ds_${TIMESTAMP}.log"
            log "Phase 4 DONE"
        fi
    fi

    # Phase 5: Analysis
    if [ -z "$PHASE" ] || [ "$PHASE" = "5" ]; then
        log "Phase 5 START: Analysis and report generation"
        "$PYTHON" experiments/plot.py results \
            --results-root results \
            --output results/figures \
            2>&1 | tee "${LOG_DIR}/analysis_${TIMESTAMP}.log"
        log "Phase 5 DONE"
    fi

    log "All requested phases complete."
}

# ══════════════════════════════════════════════════════════════
# pipeline — phases.py all + plot
# ══════════════════════════════════════════════════════════════

cmd_pipeline() {
    ensure_venv
    cd "$PROJECT_ROOT"

    log "Pipeline started"

    log "Running all experiment phases"
    "$PYTHON" experiments/run.py phases --phase all

    log "Running analysis"
    "$PYTHON" experiments/plot.py results --results-root results --output results/figures

    log "Pipeline finished"
}

# ══════════════════════════════════════════════════════════════
# trace — OS-level trace capture pipeline
# ══════════════════════════════════════════════════════════════

cmd_trace() {
    cd "$PROJECT_ROOT"

    echo "=== OS-Level Trace Capture Pipeline ==="
    echo ""
    echo "Captures real file-open syscalls from the training process."
    echo ""
    echo "Steps:"
    echo "  1. Materialize dataset as per-sample files"
    echo "  2. Start training with file-backed dataset"
    echo "  3. Capture OS-level traces (eBPF or strace)"
    echo "  4. Convert traces to attack CSV"
    echo "  5. Run membership inference attack"
    echo ""

    local DATASET_ROOT="dataset_root"
    local TRAIN_SIZE=5000
    local HOLDOUT_SIZE=5000
    local EPOCHS=2
    local BATCH_SIZE=128
    local PROBE_BATCH_PROB=0.2
    local PROBE_MIX_RATIO=0.3
    local SIDECAR="batch_sidecar.csv"
    local TRACE_OUTPUT="opens.csv"
    local EVENTS_OUTPUT="events_trace.csv"
    local ATTACK_OUTPUT="results/os_trace_attack"

    echo "Configuration:"
    echo "  Dataset root: $DATASET_ROOT"
    echo "  Train size: $TRAIN_SIZE"
    echo "  Holdout size: $HOLDOUT_SIZE"
    echo "  Epochs: $EPOCHS"
    echo "  Batch size: $BATCH_SIZE"
    echo ""

    read -p "Continue? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi

    echo ""
    echo "=== STEP 1: Materialize Dataset ==="
    python3 experiments/test.py files \
        --output_dir "$DATASET_ROOT" \
        --train_size $TRAIN_SIZE \
        --holdout_size $HOLDOUT_SIZE \
        --seed 42

    echo ""
    echo "=== STEP 2: Check for eBPF/BCC ==="
    local USE_EBPF=0
    if python3 -c "import bcc" 2>/dev/null; then
        echo "BCC available, will use eBPF tracing"
        USE_EBPF=1
    else
        echo "BCC not available, will use strace fallback"
        echo "  To install BCC: sudo apt install bpfcc-tools python3-bpfcc"
    fi

    echo ""
    echo "=== STEP 3: Start Training (Background) ==="
    python3 experiments/test.py train \
        --dataset_root "$DATASET_ROOT" \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --probe_batch_prob $PROBE_BATCH_PROB \
        --probe_mix_ratio $PROBE_MIX_RATIO \
        --sidecar_path "$SIDECAR" &

    local TRAIN_PID=$!
    echo "Training PID: $TRAIN_PID"
    sleep 2

    echo ""
    echo "=== STEP 4: Capture Traces ==="
    local TRACE_PID
    if [ $USE_EBPF -eq 1 ]; then
        echo "Using eBPF trace capture..."
        sudo python3 experiments/test.py trace \
            --pid $TRAIN_PID \
            --output "$TRACE_OUTPUT" &
        TRACE_PID=$!
    else
        echo "Using strace fallback..."
        sudo strace -ff -ttt -e trace=open,openat,openat2 -p $TRAIN_PID 2> strace.log &
        TRACE_PID=$!
    fi
    echo "Trace PID: $TRACE_PID"

    echo ""
    echo "Waiting for training to complete..."
    wait $TRAIN_PID
    echo "Training complete."

    sleep 1
    sudo kill -INT $TRACE_PID 2>/dev/null || true
    wait $TRACE_PID 2>/dev/null || true

    echo ""
    echo "=== STEP 5: Convert Traces to Attack CSV ==="
    if [ $USE_EBPF -eq 1 ]; then
        python3 experiments/test.py sidecar \
            --trace_input "$TRACE_OUTPUT" \
            --trace_mode ebpf_csv \
            --sidecar "$SIDECAR" \
            --output "$EVENTS_OUTPUT"
    else
        python3 experiments/test.py sidecar \
            --trace_input strace.log \
            --trace_mode strace \
            --sidecar "$SIDECAR" \
            --output "$EVENTS_OUTPUT"
    fi

    echo ""
    echo "=== STEP 6: Validate Event Log ==="
    python3 experiments/test.py probe --input "$EVENTS_OUTPUT"

    echo ""
    echo "=== STEP 7: Run Membership Inference Attack ==="
    python3 experiments/attack.py \
        --input "$EVENTS_OUTPUT" \
        --output_dir "$ATTACK_OUTPUT" \
        --visibility 1.0 \
        --random_state 42

    echo ""
    echo "=== COMPLETE ==="
    echo "Results saved to: $ATTACK_OUTPUT"
    echo ""
    echo "Key files:"
    echo "  Dataset:       $DATASET_ROOT/"
    echo "  Batch sidecar: $SIDECAR"
    if [ $USE_EBPF -eq 1 ]; then
        echo "  eBPF trace:    $TRACE_OUTPUT"
    else
        echo "  strace log:    strace.log"
    fi
    echo "  Attack events: $EVENTS_OUTPUT"
    echo "  Attack results: $ATTACK_OUTPUT/"
}

# ══════════════════════════════════════════════════════════════
# visibility — visibility sweep with partial observability
# ══════════════════════════════════════════════════════════════

cmd_visibility() {
    cd "$PROJECT_ROOT"

    local TRAIN_SIZE=20000
    local HOLDOUT_SIZE=20000
    local EPOCHS=3
    local BATCH_SIZE=128
    local PROBE_BATCH_PROB=0.2
    local PROBE_MIX_RATIO=0.3
    local OUTPUT_DIR="results/visibility_sweep"
    local RANDOM_STATE=42

    echo "=== Visibility Sweep with Partial Observability ==="
    echo ""
    echo "Configuration:"
    echo "  Train size: $TRAIN_SIZE"
    echo "  Holdout size: $HOLDOUT_SIZE"
    echo "  Epochs: $EPOCHS"
    echo "  Batch size: $BATCH_SIZE"
    echo "  Probe batch prob: $PROBE_BATCH_PROB"
    echo "  Probe mix ratio: $PROBE_MIX_RATIO"
    echo "  Output directory: $OUTPUT_DIR"
    echo ""

    mkdir -p "$OUTPUT_DIR"

    # Phase 1: Generate event logs at multiple visibility levels
    echo "=== PHASE 1: Generate Event Logs ==="
    for VIS in 1.0 0.5 0.25 0.1; do
        VIS_INT=$(echo "$VIS * 100" | bc | cut -d. -f1)

        echo ""
        echo "--- Generating events at visibility=$VIS ---"
        python3 experiments/generate.py partial \
            --seed $RANDOM_STATE \
            --train_size $TRAIN_SIZE \
            --holdout_size $HOLDOUT_SIZE \
            --batch_size $BATCH_SIZE \
            --epochs $EPOCHS \
            --probe_batch_prob $PROBE_BATCH_PROB \
            --probe_mix_ratio $PROBE_MIX_RATIO \
            --visibility $VIS \
            --timestamp_jitter_std 0.003 \
            --batch_id_corruption_prob 0.10 \
            --background_noise_rate 0.05 \
            --full_output "$OUTPUT_DIR/events_full_v${VIS_INT}.csv" \
            --observed_output "$OUTPUT_DIR/events_observed_v${VIS_INT}.csv"
    done

    # Phase 2: Validate event logs
    echo ""
    echo "=== PHASE 2: Validate Event Logs ==="
    for VIS in 1.0 0.5 0.25 0.1; do
        VIS_INT=$(echo "$VIS * 100" | bc | cut -d. -f1)
        echo ""
        echo "--- Validating observed events at visibility=$VIS ---"
        python3 experiments/test.py probe \
            --input "$OUTPUT_DIR/events_observed_v${VIS_INT}.csv" || true
    done

    # Phase 3: Run attacks on observed streams
    echo ""
    echo "=== PHASE 3: Run Attacks on Observed Streams ==="
    for VIS in 1.0 0.5 0.25 0.1; do
        VIS_INT=$(echo "$VIS * 100" | bc | cut -d. -f1)
        echo ""
        echo "--- Attacking observed stream at visibility=$VIS ---"
        python3 experiments/attack.py \
            --input "$OUTPUT_DIR/events_observed_v${VIS_INT}.csv" \
            --output_dir "$OUTPUT_DIR/attack_v${VIS_INT}" \
            --visibility 1.0 \
            --random_state $RANDOM_STATE
    done

    # Phase 4: Summary
    echo ""
    echo "=== PHASE 4: Summary ==="
    python3 - <<'PYEOF'
import json, os

output_dir = "results/visibility_sweep"
visibility_levels = [1.0, 0.5, 0.25, 0.1]

print("\n=== VISIBILITY SWEEP SUMMARY ===\n")
print("| Visibility | AUC    | Accuracy | AP     | Model          |")
print("|------------|--------|----------|--------|----------------|")

for vis in visibility_levels:
    vis_int = int(vis * 100)
    metrics_path = os.path.join(output_dir, f"attack_v{vis_int}", "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            data = json.load(f)
            best = data["best_model"]
            res = data["results"][best]
            print(f"| {vis:.2f}       | {res['auc']:.4f} | {res['accuracy']:.4f}   | {res['average_precision']:.4f} | {best:14s} |")

print(f"\nResults saved to: {output_dir}")
PYEOF

    echo ""
    echo "=== Complete ==="
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Key files:"
    echo "  Full logs:       $OUTPUT_DIR/events_full_v*.csv"
    echo "  Observed logs:   $OUTPUT_DIR/events_observed_v*.csv"
    echo "  Attack results:  $OUTPUT_DIR/attack_v*/"
    echo "  Metrics:         $OUTPUT_DIR/attack_v*/metrics.json"
}

# ── Route subcommand ─────────────────────────────────────────

case "$SUBCMD" in
    experiments)  cmd_experiments "$@" ;;
    pipeline)     cmd_pipeline "$@" ;;
    trace)        cmd_trace "$@" ;;
    visibility)   cmd_visibility "$@" ;;
    *)            usage ;;
esac
