#!/usr/bin/env bash
#
# Unified test runner for the OMLO project.
#
# Subcommands:
#   smoke    Quick smoke test (2 epochs) to verify system works
#   attack   Membership inference attack pipeline test
#   macos    macOS physical-access audit with fs_usage
#
# Usage:
#   ./scripts/test.sh smoke
#   ./scripts/test.sh attack
#   ./scripts/test.sh macos [output_dir] [num_samples] [batch_size] [epochs]

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${PROJECT_ROOT}/venv"
PYTHON="${VENV}/bin/python"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
fail() { log "ERROR: $*"; exit 1; }

usage() {
    echo "Usage: $0 <subcommand> [options]"
    echo ""
    echo "Subcommands:"
    echo "  smoke                                          Quick smoke test (2 epochs)"
    echo "  attack                                         Membership inference attack test"
    echo "  macos [out_dir] [samples] [batch_size] [epochs]  macOS fs_usage audit"
    exit 1
}

[[ $# -lt 1 ]] && usage
SUBCMD="$1"; shift

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
}

# ══════════════════════════════════════════════════════════════
# smoke — quick end-to-end verification
# ══════════════════════════════════════════════════════════════

cmd_smoke() {
    ensure_venv
    cd "$PROJECT_ROOT"

    log "Using Python: $PYTHON"
    "$PYTHON" --version

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
    "$PYTHON" experiments/run.py baseline \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --output-dir "$BASELINE_DIR" \
        || fail "Baseline training failed"

    [ -f "$BASELINE_DIR/history.json" ] || fail "Baseline history.json not created"
    [ -f "$BASELINE_DIR/baseline_profile.json" ] || fail "Baseline profile.json not created"
    log "Baseline training completed"

    log "Phase 2/3: Running ORAM training ($EPOCHS epochs, $NUM_SAMPLES samples)..."
    "$PYTHON" experiments/run.py sweep \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --num-samples "$NUM_SAMPLES" \
        --output-dir "$ORAM_DIR" \
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

cmd_attack() {
    cd "$PROJECT_ROOT"

    echo "=== Testing Upgraded Membership Inference Attack ==="
    echo ""

    OUTPUT_DIR="results/test_attack"
    mkdir -p "$OUTPUT_DIR"

    echo "Step 1: Generate plaintext event log (small scale)..."
    python3 experiments/generate.py event \
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
    python3 experiments/test.py probe \
        --input "$OUTPUT_DIR/events_plaintext_test.csv"

    echo ""
    echo "Step 3: Generate ORAM event log (small scale)..."
    python3 experiments/generate.py event \
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
    python3 experiments/test.py probe \
        --input "$OUTPUT_DIR/events_oram_test.csv"

    echo ""
    echo "Step 5: Run attack on plaintext log..."
    python3 experiments/attack.py \
        --input "$OUTPUT_DIR/events_plaintext_test.csv" \
        --output_dir "$OUTPUT_DIR/attack_plaintext" \
        --visibility 1.0 \
        --random_state 42

    echo ""
    echo "Step 6: Run attack on ORAM log..."
    python3 experiments/attack.py \
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

cmd_macos() {
    local OUT_DIR="${1:-/tmp/oram_macos_audit}"
    local NUM_SAMPLES="${2:-128}"
    local BATCH_SIZE="${3:-32}"
    local EPOCHS="${4:-1}"

    local VENV_PATH="${PROJECT_ROOT}/venv/bin/activate"
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

    if [[ ! -f "${VENV_PATH}" ]]; then
        echo "Missing venv at ${VENV_PATH}"
        exit 1
    fi

    source "${VENV_PATH}"

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
    ORAM_AUDIT_LOG="${ORAM_AUDIT_LOG}" python "${PROJECT_ROOT}/experiments/run.py" sidecar \
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
    python "${PROJECT_ROOT}/experiments/test.py" sidecar \
        --trace_input "${TRACE_LOG}" \
        --trace_mode fs_usage \
        --sidecar "${SIDECAR_PATH}" \
        --defense oram \
        --oram_block_size 4096 \
        --trace_validation_out "${TRACE_VALIDATION_JSON}" \
        --attack_input_audit_out "${ATTACK_INPUT_AUDIT_JSON}" \
        --output "${EVENTS_CSV}"

    echo "Running upgraded membership attack on converted trace..."
    python "${PROJECT_ROOT}/experiments/attack.py" \
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

# ── Route subcommand ─────────────────────────────────────────

case "$SUBCMD" in
    smoke)   cmd_smoke ;;
    attack)  cmd_attack ;;
    macos)   cmd_macos "$@" ;;
    *)       usage ;;
esac
