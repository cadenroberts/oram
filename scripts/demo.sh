#!/usr/bin/env bash
#
# Demo script for ORAM-integrated PyTorch training.
#
# Runs a quick smoke test (2 epochs each) to verify the system works.
# Non-interactive, exits non-zero on failure.
#
# Usage:
#   ./scripts/demo.sh
#
# Expected runtime: ~15 minutes on CPU, ~5 minutes on GPU

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="${PROJECT_ROOT}/venv"
PYTHON="${VENV}/bin/python"

# ── Output directories ─────────────────────────────────────────
DEMO_DIR="${PROJECT_ROOT}/results/demo"
BASELINE_DIR="${DEMO_DIR}/baseline"
ORAM_DIR="${DEMO_DIR}/oram"
ANALYSIS_DIR="${DEMO_DIR}/analysis"

# ── Configuration ──────────────────────────────────────────────
EPOCHS=2
BATCH_SIZE=128
NUM_SAMPLES=10000  # Reduced dataset for quick demo

# ── Helper functions ───────────────────────────────────────────
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
fail() { log "ERROR: $*"; exit 1; }

# ── Validate environment ───────────────────────────────────────
log "Validating environment..."

# Check Python virtual environment
if [ ! -d "$VENV" ]; then
    log "Virtual environment not found at $VENV"
    log "Creating virtual environment..."
    cd "$PROJECT_ROOT"
    python3 -m venv venv || fail "Failed to create virtual environment"
    log "Installing dependencies..."
    "$VENV/bin/pip" install -r requirements.txt || fail "Failed to install dependencies"
fi

# Activate virtual environment
if [ ! -f "$PYTHON" ]; then
    fail "Python binary not found at $PYTHON"
fi

log "Using Python: $PYTHON"
"$PYTHON" --version

# ── Clean up previous demo results ────────────────────────────
if [ -d "$DEMO_DIR" ]; then
    log "Removing previous demo results..."
    rm -rf "$DEMO_DIR"
fi

mkdir -p "$DEMO_DIR"
cd "$PROJECT_ROOT"

# ── Phase 1: Baseline training ─────────────────────────────────
log "Phase 1/3: Running baseline training ($EPOCHS epochs)..."
"$PYTHON" experiments/run_baseline.py \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --output-dir "$BASELINE_DIR" \
    || fail "Baseline training failed"

# Verify baseline outputs
[ -f "$BASELINE_DIR/history.json" ] || fail "Baseline history.json not created"
[ -f "$BASELINE_DIR/baseline_profile.json" ] || fail "Baseline profile.json not created"
log "✓ Baseline training completed"

# ── Phase 2: ORAM training ─────────────────────────────────────
log "Phase 2/3: Running ORAM training ($EPOCHS epochs, $NUM_SAMPLES samples)..."
"$PYTHON" experiments/run_oram.py \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --num-samples "$NUM_SAMPLES" \
    --output-dir "$ORAM_DIR" \
    || fail "ORAM training failed"

# Verify ORAM outputs
[ -f "$ORAM_DIR/history.json" ] || fail "ORAM history.json not created"
[ -f "$ORAM_DIR/oram_profile.json" ] || fail "ORAM profile.json not created"
log "✓ ORAM training completed"

# ── Phase 3: Analysis ──────────────────────────────────────────
log "Phase 3/3: Analyzing results..."
"$PYTHON" experiments/analyze_results.py \
    --baseline "$BASELINE_DIR" \
    --oram "$ORAM_DIR" \
    --output "$ANALYSIS_DIR" \
    || fail "Analysis failed"

# Verify analysis outputs
[ -f "$ANALYSIS_DIR/overhead_report.md" ] || fail "Overhead report not created"
log "✓ Analysis completed"

# ── Verification ───────────────────────────────────────────────
log "Verifying correctness..."

# Check training convergence
"$PYTHON" - <<EOF || fail "Verification failed"
import json
import sys

# Load baseline results
with open('$BASELINE_DIR/history.json') as f:
    baseline = json.load(f)

# Load ORAM results
with open('$ORAM_DIR/history.json') as f:
    oram = json.load(f)

# Verify baseline convergence
baseline_acc = baseline['train_acc']
if not (baseline_acc[-1] > baseline_acc[0]):
    print(f"ERROR: Baseline not learning: {baseline_acc}")
    sys.exit(1)

# Verify ORAM convergence
oram_acc = oram['train_acc']
if not (oram_acc[-1] > oram_acc[0]):
    print(f"ERROR: ORAM not learning: {oram_acc}")
    sys.exit(1)

# Verify ORAM accuracy is reasonable (within 30% of baseline after 2 epochs)
if oram_acc[-1] < baseline_acc[-1] * 0.7:
    print(f"WARNING: ORAM accuracy low: {oram_acc[-1]:.2f}% vs baseline {baseline_acc[-1]:.2f}%")
    print(f"This may be normal for {$EPOCHS} epochs with reduced dataset")

# Print summary
print("")
print("="*60)
print("DEMO SUMMARY")
print("="*60)
print(f"Baseline training accuracy: {baseline_acc}")
print(f"ORAM training accuracy:     {oram_acc}")
print(f"Baseline total time:        {baseline['total_time']:.2f}s")
print(f"ORAM total time:            {oram['total_time']:.2f}s")
print(f"Overhead ratio:             {oram['total_time'] / baseline['total_time']:.1f}x")
print("="*60)
EOF

log "✓ Verification passed"

# ── Success ────────────────────────────────────────────────────
log ""
log "Demo completed successfully!"
log "Results saved to: $DEMO_DIR"
log ""
log "Generated files:"
log "  - $BASELINE_DIR/history.json"
log "  - $ORAM_DIR/history.json"
log "  - $ANALYSIS_DIR/overhead_report.md"
log "  - $ANALYSIS_DIR/*.png (plots)"
log ""

echo "DEMO_OK"
exit 0
