#!/usr/bin/env bash
#
# Unified script for the OMLO project.
#
# Subcommands:
#   experiments [--phase N] [--force]   Run training / sweep / analysis phases
#   pipeline                            Run all phases + analysis (phases.py path)
#   trace [--yes]                        OS-level trace capture pipeline
#   visibility                          Visibility sweep with partial observability
#   smoke                               Quick smoke test (2 epochs)
#   attack                              Membership inference attack pipeline test
#   macos [out_dir] [samples] [bs] [ep] macOS physical-access audit with fs_usage
#   results [--yes]                     Paper-ready membership inference results
#
# Usage:
#   ./run.sh experiments              # all 5 phases
#   ./run.sh experiments --phase 2    # only phase 2
#   ./run.sh experiments --force      # re-run even if results exist
#   ./run.sh pipeline                 # phases.py + plot
#   ./run.sh trace                    # real syscall trace pipeline (interactive)
#   ./run.sh trace --yes              # real syscall trace pipeline (non-interactive)
#   ./run.sh visibility               # visibility sweep
#   ./run.sh smoke                    # quick end-to-end verification
#   ./run.sh attack                   # attack pipeline test
#   ./run.sh macos                    # macOS fs_usage audit
#   ./run.sh results                  # paper-ready attack evaluation
#   ./run.sh results --yes            # skip confirmation prompt

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="${PROJECT_ROOT}/venv"
PYTHON="${VENV}/bin/python"
LOG_DIR="${PROJECT_ROOT}/results/logs"
LOCK_FILE="${PROJECT_ROOT}/results/.experiment_lock"
TIMESTAMP="$(date +%Y%m%dT%H%M%S)"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
fail() { log "ERROR: $*"; exit 1; }

# ── Subcommand dispatch ──────────────────────────────────────

usage() {
    echo "Usage: $0 <subcommand> [options]"
    echo ""
    echo "Subcommands:"
    echo "  experiments [--phase N] [--force]     Training / sweep / analysis phases"
    echo "  pipeline                              All phases via phases.py + plot"
    echo "  trace [--yes]                          OS-level trace capture pipeline"
    echo "  visibility                            Visibility sweep (partial observability)"
    echo "  smoke                                 Quick smoke test (2 epochs)"
    echo "  attack                                Membership inference attack test"
    echo "  macos [out] [samples] [bs] [epochs]   macOS fs_usage audit"
    echo "  results [--yes]                       Paper-ready membership inference results"
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
    if [ ! -d "$VENV" ]; then
        log "Virtual environment not found at $VENV"
        log "Creating virtual environment..."
        cd "$PROJECT_ROOT"
        python3 -m venv venv || fail "Failed to create virtual environment"
        log "Installing dependencies..."
        "$VENV/bin/pip" install -r requirements.txt || fail "Failed to install dependencies"
    fi
    if [ ! -f "$PYTHON" ]; then
        fail "Python binary not found at $PYTHON"
    fi
    source "${VENV}/bin/activate"
}

phase_done() { [ -f "$1" ] && [ "${FORCE:-0}" -eq 0 ]; }

# ══════════════════════════════════════════════════════════════
# experiments — phased training, sweeps, and analysis
# ══════════════════════════════════════════════════════════════

experiments() {
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

    if [ -z "$PHASE" ] || [ "$PHASE" = "1" ]; then
        BASELINE_DIR="results/baseline"
        BASELINE_MARKER="${BASELINE_DIR}/history.json"
        if phase_done "$BASELINE_MARKER"; then
            log "Phase 1 SKIP: baseline results already exist at $BASELINE_MARKER"
        else
            log "Phase 1 START: Baseline training (100 epochs, bs=128)"
            "$PYTHON" src/run.py baseline \
                --epochs 100 \
                --batch_size 128 \
                --output_dir "$BASELINE_DIR" \
                2>&1 | tee "${LOG_DIR}/baseline_${TIMESTAMP}.log"
            log "Phase 1 DONE"
        fi
    fi

    if [ -z "$PHASE" ] || [ "$PHASE" = "2" ]; then
        ORAM_DIR="results/oram"
        ORAM_MARKER="${ORAM_DIR}/history.json"
        if phase_done "$ORAM_MARKER"; then
            log "Phase 2 SKIP: ORAM results already exist at $ORAM_MARKER"
        else
            log "Phase 2 START: ORAM training (10 epochs, 50k samples, bs=128)"
            "$PYTHON" src/run.py oram \
                --epochs 10 \
                --batch_size 128 \
                --output_dir "$ORAM_DIR" \
                2>&1 | tee "${LOG_DIR}/oram_${TIMESTAMP}.log"
            log "Phase 2 DONE"
        fi
    fi

    if [ -z "$PHASE" ] || [ "$PHASE" = "3" ]; then
        SWEEP_BS_MARKER="results/sweep_batch_size/sweep_summary.json"
        if phase_done "$SWEEP_BS_MARKER"; then
            log "Phase 3 SKIP: batch-size sweep results already exist"
        else
            log "Phase 3 START: Batch-size sweep (bs={32,64,128,256}, 3 epochs)"
            "$PYTHON" src/run.py sweep \
                --sweep batch_size \
                --epochs 3 \
                --output_dir results \
                2>&1 | tee "${LOG_DIR}/sweep_bs_${TIMESTAMP}.log"
            log "Phase 3 DONE"
        fi
    fi

    if [ -z "$PHASE" ] || [ "$PHASE" = "4" ]; then
        SWEEP_DS_MARKER="results/sweep_dataset_size/sweep_summary.json"
        if phase_done "$SWEEP_DS_MARKER"; then
            log "Phase 4 SKIP: dataset-size sweep results already exist"
        else
            log "Phase 4 START: Dataset-size sweep (n={1k,5k,10k,50k}, 2 epochs)"
            "$PYTHON" src/run.py sweep \
                --sweep dataset_size \
                --epochs 2 \
                --output_dir results \
                2>&1 | tee "${LOG_DIR}/sweep_ds_${TIMESTAMP}.log"
            log "Phase 4 DONE"
        fi
    fi

    if [ -z "$PHASE" ] || [ "$PHASE" = "5" ]; then
        log "Phase 5 START: Analysis and report generation"
        "$PYTHON" src/run.py plot \
            --results_root results \
            --output results/figures \
            2>&1 | tee "${LOG_DIR}/analysis_${TIMESTAMP}.log"
        log "Phase 5 DONE"
    fi

    log "All requested phases complete."
}

# ══════════════════════════════════════════════════════════════
# pipeline — phases.py all + plot
# ══════════════════════════════════════════════════════════════

pipeline() {
    ensure_venv
    cd "$PROJECT_ROOT"

    log "Pipeline started"

    log "Running all experiment phases"
    "$PYTHON" src/run.py phases --phase all

    log "Running analysis"
    "$PYTHON" src/run.py plot --results_root results --output results/figures

    log "Pipeline finished"
}

# ══════════════════════════════════════════════════════════════
# trace — OS-level trace capture pipeline
# ══════════════════════════════════════════════════════════════

trace() {
    local SKIP_CONFIRM=0
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --yes|-y) SKIP_CONFIRM=1; shift ;;
            *) echo "Unknown argument: $1"; exit 1 ;;
        esac
    done

    cd "$PROJECT_ROOT"
    ensure_venv

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

    if [ "$SKIP_CONFIRM" -eq 0 ]; then
        read -p "Continue? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborted."
            exit 1
        fi
    fi

    echo ""
    echo "=== STEP 1: Materialize Dataset ==="
    $PYTHON src/run.py files \
        --output_root "$DATASET_ROOT" \
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
    $PYTHON src/run.py train \
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
        sudo $PYTHON src/run.py trace \
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
        $PYTHON src/run.py convert \
            --trace_input "$TRACE_OUTPUT" \
            --trace_mode ebpf_csv \
            --sidecar "$SIDECAR" \
            --defense plaintext \
            --output "$EVENTS_OUTPUT"
    else
        $PYTHON src/run.py convert \
            --trace_input strace.log \
            --trace_mode strace \
            --sidecar "$SIDECAR" \
            --defense plaintext \
            --output "$EVENTS_OUTPUT"
    fi

    echo ""
    echo "=== STEP 6: Validate Event Log ==="
    $PYTHON src/run.py probe --input "$EVENTS_OUTPUT"

    echo ""
    echo "=== STEP 7: Run Membership Inference Attack ==="
    $PYTHON src/run.py mi \
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

visibility() {
    cd "$PROJECT_ROOT"
    ensure_venv

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

    echo "=== PHASE 1: Generate Event Logs ==="
    for VIS in 1.0 0.5 0.25 0.1; do
        VIS_INT=$(awk -v v="$VIS" 'BEGIN {printf "%d", v * 100}')

        echo ""
        echo "--- Generating events at visibility=$VIS ---"
        $PYTHON src/run.py partial \
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

    echo ""
    echo "=== PHASE 2: Validate Event Logs ==="
    for VIS in 1.0 0.5 0.25 0.1; do
        VIS_INT=$(awk -v v="$VIS" 'BEGIN {printf "%d", v * 100}')
        echo ""
        echo "--- Validating observed events at visibility=$VIS ---"
        $PYTHON src/run.py probe \
            --input "$OUTPUT_DIR/events_observed_v${VIS_INT}.csv" || true
    done

    echo ""
    echo "=== PHASE 3: Run Attacks on Observed Streams ==="
    for VIS in 1.0 0.5 0.25 0.1; do
        VIS_INT=$(awk -v v="$VIS" 'BEGIN {printf "%d", v * 100}')
        echo ""
        echo "--- Attacking observed stream at visibility=$VIS ---"
        $PYTHON src/run.py mi \
            --input "$OUTPUT_DIR/events_observed_v${VIS_INT}.csv" \
            --output_dir "$OUTPUT_DIR/attack_v${VIS_INT}" \
            --visibility 1.0 \
            --random_state $RANDOM_STATE
    done

    echo ""
    echo "=== PHASE 4: Summary ==="
    $PYTHON - <<'PYEOF'
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

# ══════════════════════════════════════════════════════════════
# smoke — quick end-to-end verification
# ══════════════════════════════════════════════════════════════

smoke() {
    ensure_venv
    cd "$PROJECT_ROOT"

    log "Using Python: $PYTHON"
    "$PYTHON" --version

    log "Running unit tests..."
    cd "${PROJECT_ROOT}/src" && "$PYTHON" test_core.py || fail "Unit tests failed"
    cd "${PROJECT_ROOT}"

    DEMO_DIR="${PROJECT_ROOT}/results/demo"
    BASELINE_DIR="${DEMO_DIR}/baseline"
    ORAM_DIR="${DEMO_DIR}/oram"
    EPOCHS=2
    BATCH_SIZE=128
    NUM_SAMPLES=10000

    if [ -d "$DEMO_DIR" ]; then
        log "Removing previous demo results..."
        rm -rf "$DEMO_DIR"
    fi
    mkdir -p "$DEMO_DIR"

    log "Phase 1/3: Running baseline training ($EPOCHS epochs)..."
    "$PYTHON" src/run.py baseline \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --output_dir "$BASELINE_DIR" \
        || fail "Baseline training failed"

    [ -f "$BASELINE_DIR/history.json" ] || fail "Baseline history.json not created"
    [ -f "$BASELINE_DIR/baseline_profile.json" ] || fail "Baseline profile.json not created"
    log "Baseline training completed"

    log "Phase 2/3: Running ORAM training ($EPOCHS epochs, $NUM_SAMPLES samples)..."
    "$PYTHON" src/run.py oram \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --num_samples "$NUM_SAMPLES" \
        --output_dir "$ORAM_DIR" \
        || fail "ORAM training failed"

    [ -f "$ORAM_DIR/history.json" ] || fail "ORAM history.json not created"
    [ -f "$ORAM_DIR/oram_profile.json" ] || fail "ORAM profile.json not created"
    log "ORAM training completed"

    log "Phase 3/3: Verifying results..."
    "$PYTHON" - <<EOF || fail "Verification failed"
import json
import sys

with open('$BASELINE_DIR/history.json') as f:
    baseline = json.load(f)
with open('$ORAM_DIR/history.json') as f:
    oram = json.load(f)

baseline_acc = baseline['train_acc']
if not (baseline_acc[-1] > baseline_acc[0]):
    print(f"ERROR: Baseline not learning: {baseline_acc}")
    sys.exit(1)

oram_acc = oram['train_acc']
if not (oram_acc[-1] > oram_acc[0]):
    print(f"ERROR: ORAM not learning: {oram_acc}")
    sys.exit(1)

if oram_acc[-1] < baseline_acc[-1] * 0.7:
    print(f"WARNING: ORAM accuracy low: {oram_acc[-1]:.2f}% vs baseline {baseline_acc[-1]:.2f}%")
    print(f"This may be normal for $EPOCHS epochs with reduced dataset")

print("")
print("="*60)
print("SMOKE TEST SUMMARY")
print("="*60)
print(f"Baseline training accuracy: {baseline_acc}")
print(f"ORAM training accuracy:     {oram_acc}")
print(f"Baseline total time:        {baseline['total_time']:.2f}s")
print(f"ORAM total time:            {oram['total_time']:.2f}s")
print(f"Overhead ratio:             {oram['total_time'] / baseline['total_time']:.1f}x")
print("="*60)
EOF

    log "Verification passed"
    log ""
    log "Smoke test completed successfully!"
    log "Results saved to: $DEMO_DIR"

    echo "SMOKE_OK"
}

# ══════════════════════════════════════════════════════════════
# attack — membership inference attack pipeline test
# ══════════════════════════════════════════════════════════════

attack() {
    cd "$PROJECT_ROOT"
    ensure_venv

    echo "=== Testing Upgraded Membership Inference Attack ==="
    echo ""

    OUTPUT_DIR="results/test_attack"
    mkdir -p "$OUTPUT_DIR"

    echo "Step 1: Generate plaintext event log (small scale)..."
    $PYTHON src/run.py event \
        --train_size 1000 \
        --holdout_size 1000 \
        --epochs 3 \
        --batch_size 64 \
        --probe_batch_prob 0.2 \
        --probe_mix_ratio 0.3 \
        --output "$OUTPUT_DIR/events_plaintext_test.csv" \
        --mode plaintext \
        --random_state 42

    echo ""
    echo "Step 2: Validate plaintext event log..."
    $PYTHON src/run.py probe \
        --input "$OUTPUT_DIR/events_plaintext_test.csv"

    echo ""
    echo "Step 3: Generate ORAM event log (small scale)..."
    $PYTHON src/run.py event \
        --train_size 1000 \
        --holdout_size 1000 \
        --epochs 3 \
        --batch_size 64 \
        --probe_batch_prob 0.2 \
        --probe_mix_ratio 0.3 \
        --output "$OUTPUT_DIR/events_oram_test.csv" \
        --mode oram \
        --backend ram \
        --random_state 42

    echo ""
    echo "Step 4: Validate ORAM event log..."
    $PYTHON src/run.py probe \
        --input "$OUTPUT_DIR/events_oram_test.csv"

    echo ""
    echo "Step 5: Run attack on plaintext log..."
    $PYTHON src/run.py mi \
        --input "$OUTPUT_DIR/events_plaintext_test.csv" \
        --output_dir "$OUTPUT_DIR/attack_plaintext" \
        --visibility 1.0 \
        --random_state 42

    echo ""
    echo "Step 6: Run attack on ORAM log..."
    $PYTHON src/run.py mi \
        --input "$OUTPUT_DIR/events_oram_test.csv" \
        --output_dir "$OUTPUT_DIR/attack_oram" \
        --visibility 1.0 \
        --random_state 42

    echo ""
    echo "=== Test Complete ==="
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Check metrics:"
    echo "  Plaintext: $OUTPUT_DIR/attack_plaintext/metrics.json"
    echo "  ORAM:      $OUTPUT_DIR/attack_oram/metrics.json"
    echo ""
    echo "Expected behavior:"
    echo "  - Plaintext AUC should be 0.65-0.85 (non-trivial signal)"
    echo "  - ORAM AUC should be 0.48-0.53 (near random)"
    echo "  - Both logs should have non-member events (validated in steps 2 and 4)"
    echo ""
    echo "If plaintext AUC > 0.9, the scenario may be too trivial."
    echo "If non-member events = 0, probe batches are not working."
}

# ══════════════════════════════════════════════════════════════
# macos — physical-access audit with fs_usage
# ══════════════════════════════════════════════════════════════

macos() {
    local OUT_DIR="${1:-/tmp/oram_macos_audit}"
    local NUM_SAMPLES="${2:-128}"
    local BATCH_SIZE="${3:-32}"
    local EPOCHS="${4:-1}"

    ensure_venv

    local TRACE_LOG="${OUT_DIR}/trace.log"
    local ORAM_AUDIT_LOG="${OUT_DIR}/oram_audit.log"
    local SIDECAR_PATH="${OUT_DIR}/batch_sidecar.csv"
    local TRAIN_LOG="${OUT_DIR}/train_stdout.log"
    local EVENTS_CSV="${OUT_DIR}/events_trace.csv"
    local TRACE_VALIDATION_JSON="${OUT_DIR}/trace_validation.json"
    local ATTACK_INPUT_AUDIT_JSON="${OUT_DIR}/attack_input_audit.json"
    local ATTACK_OUT_DIR="${OUT_DIR}/attack_v1p0"

    mkdir -p "${OUT_DIR}"
    rm -f "${TRACE_LOG}" "${ORAM_AUDIT_LOG}" "${SIDECAR_PATH}" "${TRAIN_LOG}" "${EVENTS_CSV}" "${TRACE_VALIDATION_JSON}" "${ATTACK_INPUT_AUDIT_JSON}"

    echo "Starting fs_usage tracing (sudo required)..."
    sudo fs_usage -w -f filesys > "${TRACE_LOG}" 2>&1 &
    FS_PID=$!
    sleep 1

    cleanup() {
        if ps -p "${FS_PID}" >/dev/null 2>&1; then
            sudo kill "${FS_PID}" >/dev/null 2>&1 || true
        fi
    }
    trap cleanup EXIT

    echo "Running real ORAM training..."
    ORAM_AUDIT_LOG="${ORAM_AUDIT_LOG}" "$PYTHON" "${PROJECT_ROOT}/src/run.py" sidecar \
        --epochs "${EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --num_samples "${NUM_SAMPLES}" \
        --backend file \
        --block_size 4096 \
        --sidecar_path "${SIDECAR_PATH}" \
        --device cpu \
        --output_dir "${OUT_DIR}" | tee "${TRAIN_LOG}"

    cleanup
    trap - EXIT

    echo "Converting fs_usage trace to events CSV..."
    "$PYTHON" "${PROJECT_ROOT}/src/run.py" convert \
        --trace_input "${TRACE_LOG}" \
        --trace_mode fs_usage \
        --sidecar "${SIDECAR_PATH}" \
        --defense oram \
        --oram_block_size 4096 \
        --trace_validation_out "${TRACE_VALIDATION_JSON}" \
        --attack_input_audit_out "${ATTACK_INPUT_AUDIT_JSON}" \
        --output "${EVENTS_CSV}"

    echo "Running upgraded membership attack on converted trace..."
    "$PYTHON" "${PROJECT_ROOT}/src/run.py" mi \
        --input "${EVENTS_CSV}" \
        --output_dir "${ATTACK_OUT_DIR}" \
        --visibility 1.0 \
        --random_state 42

    echo ""
    echo "Audit artifacts:"
    echo "  trace log:      ${TRACE_LOG}"
    echo "  ORAM audit log: ${ORAM_AUDIT_LOG}"
    echo "  training log:   ${TRAIN_LOG}"
    echo "  events CSV:     ${EVENTS_CSV}"
    echo "  trace validate: ${TRACE_VALIDATION_JSON}"
    echo "  input audit:    ${ATTACK_INPUT_AUDIT_JSON}"
    echo "  attack metrics: ${ATTACK_OUT_DIR}/metrics.json"
}

# ══════════════════════════════════════════════════════════════
# results — paper-ready membership inference evaluation
# ══════════════════════════════════════════════════════════════

results() {
    cd "$PROJECT_ROOT"
    ensure_venv

    local SKIP_CONFIRM=0
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --yes|-y) SKIP_CONFIRM=1; shift ;;
            *) echo "Unknown argument: $1"; exit 1 ;;
        esac
    done

    local TRAIN_SIZE=20000
    local HOLDOUT_SIZE=20000
    local EPOCHS=3
    local BATCH_SIZE=128
    local PROBE_BATCH_PROB=0.2
    local PROBE_MIX_RATIO=0.3
    local OUTPUT_DIR="results/paper_membership_attack"
    local DATA_DIR="./data"
    local RANDOM_STATE=42

    echo "=== Generating Paper-Ready Membership Inference Results ==="
    echo ""
    echo "This script generates the full attack evaluation for the manuscript."
    echo "It will take significant time with larger dataset sizes."
    echo ""
    echo "Configuration:"
    echo "  Training samples (members): $TRAIN_SIZE"
    echo "  Holdout samples (non-members): $HOLDOUT_SIZE"
    echo "  Epochs: $EPOCHS"
    echo "  Batch size: $BATCH_SIZE"
    echo "  Probe batch probability: $PROBE_BATCH_PROB"
    echo "  Probe mix ratio: $PROBE_MIX_RATIO"
    echo "  Output directory: $OUTPUT_DIR"
    echo ""

    if [ "$SKIP_CONFIRM" -eq 0 ]; then
        read -p "Continue? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborted."
            exit 1
        fi
    fi

    mkdir -p "$OUTPUT_DIR"

    echo ""
    echo "=== PHASE 1: Generate Plaintext Event Log ==="
    $PYTHON src/run.py event \
        --train_size $TRAIN_SIZE \
        --holdout_size $HOLDOUT_SIZE \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --probe_batch_prob $PROBE_BATCH_PROB \
        --probe_mix_ratio $PROBE_MIX_RATIO \
        --output "$OUTPUT_DIR/events_plaintext.csv" \
        --mode plaintext \
        --data_dir "$DATA_DIR" \
        --random_state $RANDOM_STATE

    echo ""
    echo "=== PHASE 2: Generate ORAM Event Log ==="
    $PYTHON src/run.py event \
        --train_size $TRAIN_SIZE \
        --holdout_size $HOLDOUT_SIZE \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --probe_batch_prob $PROBE_BATCH_PROB \
        --probe_mix_ratio $PROBE_MIX_RATIO \
        --output "$OUTPUT_DIR/events_oram.csv" \
        --mode oram \
        --backend ram \
        --data_dir "$DATA_DIR" \
        --random_state $RANDOM_STATE

    echo ""
    echo "=== PHASE 3: Run Attacks at Multiple Visibility Levels ==="

    for VIS in 1.0 0.5 0.25 0.1; do
        VIS_INT=$(awk -v v="$VIS" 'BEGIN {printf "%d", v * 100}')

        echo ""
        echo "--- Plaintext @ visibility=$VIS ---"
        $PYTHON src/run.py mi \
            --input "$OUTPUT_DIR/events_plaintext.csv" \
            --output_dir "$OUTPUT_DIR/plaintext_v${VIS_INT}" \
            --visibility $VIS \
            --random_state $RANDOM_STATE

        echo ""
        echo "--- ORAM @ visibility=$VIS ---"
        $PYTHON src/run.py mi \
            --input "$OUTPUT_DIR/events_oram.csv" \
            --output_dir "$OUTPUT_DIR/oram_v${VIS_INT}" \
            --visibility $VIS \
            --random_state $RANDOM_STATE
    done

    echo ""
    echo "=== PHASE 4: Generate Summary Table ==="

    $PYTHON - <<'PYEOF'
import json
import os
import sys

output_dir = "results/paper_membership_attack"
visibility_levels = [1.0, 0.5, 0.25, 0.1]

print("\n=== MEMBERSHIP INFERENCE ATTACK SUMMARY ===\n")
print("Plaintext Access Patterns:")
print("| Visibility | AUC    | Accuracy | AP     | Model          |")
print("|------------|--------|----------|--------|----------------|")

for vis in visibility_levels:
    vis_int = int(vis * 100)
    metrics_path = os.path.join(output_dir, f"plaintext_v{vis_int}", "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            data = json.load(f)
            best = data["best_model"]
            res = data["results"][best]
            print(f"| {vis:.2f}       | {res['auc']:.4f} | {res['accuracy']:.4f}   | {res['average_precision']:.4f} | {best:14s} |")

print("\nORAM-Backed Access Patterns:")
print("| Visibility | AUC    | Accuracy | AP     | Model          |")
print("|------------|--------|----------|--------|----------------|")

for vis in visibility_levels:
    vis_int = int(vis * 100)
    metrics_path = os.path.join(output_dir, f"oram_v{vis_int}", "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            data = json.load(f)
            best = data["best_model"]
            res = data["results"][best]
            print(f"| {vis:.2f}       | {res['auc']:.4f} | {res['accuracy']:.4f}   | {res['average_precision']:.4f} | {best:14s} |")

print(f"\nAll results saved to: {output_dir}")
print("\nFor LaTeX table, see the generated summary files in each attack directory.")
PYEOF

    echo ""
    echo "=== Complete ==="
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Key files:"
    echo "  - Event logs: $OUTPUT_DIR/events_*.csv"
    echo "  - Attack results: $OUTPUT_DIR/{plaintext,oram}_v{100,50,25,10}/"
    echo "  - ROC curves: $OUTPUT_DIR/*/roc_curve.png"
    echo "  - Feature importance: $OUTPUT_DIR/*/feature_importance_*.csv"
}

# ── Route subcommand ─────────────────────────────────────────

case "$SUBCMD" in
    experiments)  experiments "$@" ;;
    pipeline)     pipeline "$@" ;;
    trace)        trace "$@" ;;
    visibility)   visibility "$@" ;;
    smoke)        smoke ;;
    attack)       attack ;;
    macos)        macos "$@" ;;
    results)      results "$@" ;;
    *)            usage ;;
esac
