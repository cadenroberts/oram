#!/bin/bash
set -e
cd /Users/cwr/resume/repos/OMLO

echo "=== Pipeline started at $(date) ==="

echo "--- Running all experiment phases ---"
venv/bin/python experiments/run_all_phases.py --phase all

echo "--- Running analysis ---"
venv/bin/python experiments/analyze_results.py --results-root results --output results/figures

echo "=== Pipeline finished at $(date) ==="
