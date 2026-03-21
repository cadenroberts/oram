#!/bin/bash
set -e

echo "=== Generating Paper-Ready Membership Inference Results ==="
echo ""
echo "This script generates the full attack evaluation for the manuscript."
echo "It will take significant time with larger dataset sizes."
echo ""

TRAIN_SIZE=20000
HOLDOUT_SIZE=20000
EPOCHS=3
BATCH_SIZE=128
PROBE_BATCH_PROB=0.2
PROBE_MIX_RATIO=0.3
OUTPUT_DIR="results/paper_membership_attack"
DATA_DIR="./data"
RANDOM_STATE=42

echo "Configuration:"
echo "  Training samples (members): $TRAIN_SIZE"
echo "  Holdout samples (non-members): $HOLDOUT_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Probe batch probability: $PROBE_BATCH_PROB"
echo "  Probe mix ratio: $PROBE_MIX_RATIO"
echo "  Output directory: $OUTPUT_DIR"
echo ""

read -p "Continue? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo ""
echo "=== PHASE 1: Generate Plaintext Event Log ==="
python3 experiments/generate.py event \
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
python3 experiments/generate.py event \
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
    VIS_INT=$(echo "$VIS * 100" | bc | cut -d. -f1)
    
    echo ""
    echo "--- Plaintext @ visibility=$VIS ---"
    python3 experiments/attack.py \
        --input "$OUTPUT_DIR/events_plaintext.csv" \
        --output_dir "$OUTPUT_DIR/plaintext_v${VIS_INT}" \
        --visibility $VIS \
        --random_state $RANDOM_STATE
    
    echo ""
    echo "--- ORAM @ visibility=$VIS ---"
    python3 experiments/attack.py \
        --input "$OUTPUT_DIR/events_oram.csv" \
        --output_dir "$OUTPUT_DIR/oram_v${VIS_INT}" \
        --visibility $VIS \
        --random_state $RANDOM_STATE
done

echo ""
echo "=== PHASE 4: Generate Summary Table ==="

python3 - <<'EOF'
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
EOF

echo ""
echo "=== Complete ==="
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Key files:"
echo "  - Event logs: $OUTPUT_DIR/events_*.csv"
echo "  - Attack results: $OUTPUT_DIR/{plaintext,oram}_v{100,50,25,10}/"
echo "  - ROC curves: $OUTPUT_DIR/*/roc_curve.png"
echo "  - Feature importance: $OUTPUT_DIR/*/feature_importance_*.csv"
