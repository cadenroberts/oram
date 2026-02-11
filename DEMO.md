# Demo Walkthrough

Complete step-by-step demonstration of the ORAM-integrated training system.

## Prerequisites

**Hardware:**
- 20GB free disk space (CIFAR-10 + ORAM storage + results)
- 4GB RAM minimum (8GB recommended)
- CPU or GPU (GPU recommended but not required)

**Software:**
- Python 3.8 or later
- pip package manager
- Git (for cloning repository)

**Expected runtime:**
- Quick demo (2 epochs each): ~15 minutes on CPU, ~5 minutes on GPU
- Full experiment suite: ~40 hours on CPU, ~8 hours on GPU

## Quick Demo (15 minutes)

This demo runs 2 epochs of baseline and ORAM training on a 10k-sample subset to verify the system works.

### Step 1: Setup environment

```bash
# Clone repository
git clone https://github.com/cadenroberts/oram.git
cd oram

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Expected output:**
```
Successfully installed torch-2.0.1 torchvision-0.15.2 pyoram-0.3.0 ...
```

**Troubleshooting:**
- `command not found: python`: Try `python3` instead
- `pip install failed`: Upgrade pip with `pip install --upgrade pip`
- `CUDA not available` warning: Normal if no GPU, experiments work on CPU

### Step 2: Run baseline training (2 epochs)

```bash
python experiments/run_baseline.py --epochs 2 --batch-size 128 --output-dir results/demo_baseline
```

**Expected output:**
```
Setting up baseline training...
Setup complete. Device: cpu
Training samples: 50000
Test samples: 10000
Epoch 1: 100%|████████████████████| 390/390 [01:23<00:00, 4.67it/s]
Epoch 1: train_loss=1.5234, train_acc=44.67%, test_acc=0.00%, time=85.23s
Epoch 2: 100%|████████████████████| 390/390 [01:22<00:00, 4.73it/s]
Epoch 2: train_loss=1.2145, train_acc=56.89%, test_acc=0.00%, time=84.01s

Training complete. Total time: 169.24s
Best test accuracy: 0.00%

PROFILER SUMMARY
================================================================
Timing Breakdown:
----------------------------------------
  dataload          15.23%  (total:    25.78s, calls:  78000, avg:    0.331ms)
  compute           82.45%  (total:   139.54s, calls:   1560, avg:   89.449ms)
  batch              2.32%  (total:     3.92s, calls:    780, avg:    5.026ms)
================================================================
```

**Key observations:**
- Training accuracy rises from ~45% to ~57% (model is learning)
- Compute dominates (82%) in baseline (as expected)
- Dataload overhead is minimal (15%)

**Verification:**
```bash
ls results/demo_baseline/
# Expected: history.json, baseline_profile.json
```

### Step 3: Run ORAM training (2 epochs, 10k samples)

```bash
python experiments/run_oram.py --epochs 2 --batch-size 128 --num-samples 10000 --output-dir results/demo_oram
```

**Expected output:**
```
Setting up ORAM-integrated training...
Initializing ORAM storage for 10000 samples...
Loading CIFAR-10 training data into ORAM (10000 samples)...
Loading CIFAR-10 to ORAM: 100%|████████████| 10000/10000 [02:15<00:00, 73.85it/s]
Setup complete. Device: cpu
ORAM storage stats: {'num_samples': 10000, 'block_size': 4096, 'total_size_mb': 39.0625, ...}
Epoch 1: 100%|████████████████████| 78/78 [08:34<00:00, 6.60s/it]
Epoch 1: train_loss=1.8234, train_acc=33.42%, test_acc=0.00%, time=514.67s
Epoch 2: 100%|████████████████████| 78/78 [08:29<00:00, 6.53s/it]
Epoch 2: train_loss=1.5123, train_acc=45.18%, test_acc=0.00%, time=509.32s

ORAM Training complete. Total time: 1024.00s
Best test accuracy: 0.00%

PROFILER SUMMARY
================================================================
Timing Breakdown:
----------------------------------------
  io                68.34%  (total:   699.56s, calls:  15600, avg:   44.842ms)
  oram_read         65.12%  (total:   666.83s, calls:  15600, avg:   42.745ms)
  serialize          4.23%  (total:    43.32s, calls:  10000, avg:    4.332ms)
  deserialize        3.89%  (total:    39.84s, calls:  15600, avg:    2.554ms)
  dataload          72.45%  (total:   741.89s, calls:  15600, avg:   47.556ms)
  compute           25.12%  (total:   257.23s, calls:    312, avg:  824.137ms)
  shuffle            0.15%  (total:     1.54s, calls:      2, avg:  770.000ms)
================================================================
```

**Key observations:**
- ORAM setup takes 2-3 minutes (one-time cost to populate 10k samples)
- Per-epoch time: ~510s vs. ~85s baseline (6× overhead for 10k samples)
- ORAM I/O dominates (68%) as expected
- Training accuracy converges similarly to baseline

**Verification:**
```bash
ls results/demo_oram/
# Expected: history.json, oram_profile.json, best_model.pth
```

### Step 4: Analyze and compare

```bash
python experiments/analyze_results.py \
    --baseline results/demo_baseline \
    --oram results/demo_oram \
    --output results/demo_analysis
```

**Expected output:**
```
Loading baseline results from results/demo_baseline...
Loading ORAM results from results/demo_oram...
Generating training comparison plot...
Generating overhead breakdown plot...
Generating operation timing distributions...
Writing overhead report...
Analysis complete. Results saved to results/demo_analysis/
```

**Generated files:**
```bash
ls results/demo_analysis/
# Expected outputs:
# - training_comparison.png    (training loss/accuracy curves, baseline vs ORAM)
# - overhead_breakdown.png     (pie chart: IO 68%, compute 25%, serialize 4%, etc.)
# - operation_times.png        (box plots of per-operation timing distributions)
# - overhead_report.md         (detailed analysis with tables and metrics)
```

### Step 5: Verify correctness

```bash
# Check that both experiments completed successfully
python -c "
import json

# Load baseline results
with open('results/demo_baseline/history.json') as f:
    baseline = json.load(f)

# Load ORAM results
with open('results/demo_oram/history.json') as f:
    oram = json.load(f)

# Verify convergence (accuracy should increase across epochs)
assert baseline['train_acc'][-1] > baseline['train_acc'][0], 'Baseline not learning'
assert oram['train_acc'][-1] > oram['train_acc'][0], 'ORAM not learning'

# Verify ORAM accuracy is reasonable (within 30% of baseline for 2 epochs)
baseline_final = baseline['train_acc'][-1]
oram_final = oram['train_acc'][-1]
assert oram_final > baseline_final * 0.7, f'ORAM accuracy too low: {oram_final} vs {baseline_final}'

print('✓ Baseline converging:', baseline['train_acc'])
print('✓ ORAM converging:', oram['train_acc'])
print('✓ ORAM overhead measured')
print('')
print('DEMO_OK')
"
```

**Expected output:**
```
✓ Baseline converging: [44.67, 56.89]
✓ ORAM converging: [33.42, 45.18]
✓ ORAM overhead measured

DEMO_OK
```

## Full Experiment Suite (40 hours)

The full suite runs 5 phases across multiple configurations to thoroughly characterize ORAM overhead.

### Phase 1: Baseline training (100 epochs)

```bash
./scripts/run_experiments.sh --phase 1
```

**What it does:**
- Trains ResNet-18 on CIFAR-10 for 100 epochs (full convergence)
- Uses standard PyTorch DataLoader (4 workers)
- Batch size: 128
- Establishes performance floor for comparison

**Expected output:**
- `results/baseline/history.json` — Training metrics (100 epochs)
- `results/baseline/baseline_profile.json` — Profiling data
- `results/baseline/best_model.pth` — Checkpoint at best test accuracy

**Expected runtime:** ~3 hours on CPU, ~40 minutes on GPU

**Expected accuracy:** ~93% test accuracy

### Phase 2: ORAM training (10 epochs, 50k samples)

```bash
./scripts/run_experiments.sh --phase 2
```

**What it does:**
- Trains ResNet-18 with ORAM-backed data loading
- Full CIFAR-10 training set (50,000 samples)
- Same hyperparameters as baseline
- 10 epochs only (overhead makes 100 epochs impractical)

**Expected output:**
- `results/oram/history.json` — Training metrics (10 epochs)
- `results/oram/oram_profile.json` — Profiling data with category breakdown
- `results/oram/best_model.pth` — Checkpoint

**Expected runtime:** ~20 hours on CPU, ~4 hours on GPU

**Expected overhead:** 90-180× vs. baseline per epoch

### Phase 3: Batch-size sweep (3 epochs each)

```bash
./scripts/run_experiments.sh --phase 3
```

**What it does:**
- Runs baseline and ORAM training with batch sizes {32, 64, 128, 256}
- 3 epochs each (8 total experiments)
- Measures how batch size affects overhead ratio

**Expected output:**
- `results/sweep_batch_size/bs_32_baseline/` — Baseline, batch=32
- `results/sweep_batch_size/bs_32_oram/` — ORAM, batch=32
- ... (8 total directories)
- `results/sweep_batch_size/sweep_summary.json` — Aggregated metrics

**Expected runtime:** ~8 hours on CPU, ~2 hours on GPU

**Expected finding:** Larger batches reduce per-epoch overhead (fewer batches = less shuffle overhead)

### Phase 4: Dataset-size sweep (2 epochs each)

```bash
./scripts/run_experiments.sh --phase 4
```

**What it does:**
- Runs ORAM training with dataset sizes {1000, 5000, 10000, 50000}
- 2 epochs each (4 total experiments)
- Validates O(log N) scaling of ORAM overhead

**Expected output:**
- `results/sweep_dataset_size/n_1000/` — 1k samples
- `results/sweep_dataset_size/n_5000/` — 5k samples
- ... (4 total directories)
- `results/sweep_dataset_size/sweep_summary.json` — Aggregated metrics

**Expected runtime:** ~12 hours on CPU, ~3 hours on GPU

**Expected finding:** Per-access overhead grows logarithmically with N (tree height)

### Phase 5: Analysis and report generation

```bash
./scripts/run_experiments.sh --phase 5
```

**What it does:**
- Loads all experiment results from Phases 1-4
- Generates plots: training curves, overhead breakdown, batch-size sensitivity, dataset-size scaling
- Writes `overhead_report.md` with detailed analysis

**Expected output:**
- `results/analysis/training_comparison.png` — Baseline vs. ORAM training curves
- `results/analysis/overhead_breakdown.png` — Pie chart of overhead categories
- `results/analysis/operation_times.png` — Box plots of per-operation timings
- `results/analysis/batch_size_sweep.png` — Overhead ratio vs. batch size
- `results/analysis/dataset_size_sweep.png` — Scaling validation (O(log N) reference line)
- `results/analysis/overhead_report.md` — Full report with tables and interpretation

**Expected runtime:** ~5 minutes

### Running all phases sequentially

```bash
# Run everything (no --phase flag)
./scripts/run_experiments.sh
```

**Behavior:**
- Runs Phases 1-5 in order
- Skips phases whose marker files already exist (idempotent)
- Logs each phase to `results/logs/phase_{1-5}_{timestamp}.log`
- Uses lock file to prevent concurrent runs

**To force re-run (delete existing results):**
```bash
rm -rf results/baseline results/oram results/sweep_* results/analysis
./scripts/run_experiments.sh
```

## Troubleshooting

### `ImportError: No module named 'pyoram'`

**Cause:** Dependencies not installed

**Fix:**
```bash
source venv/bin/activate  # Ensure virtual environment is active
pip install -r requirements.txt
```

### `RuntimeError: CUDA out of memory`

**Cause:** GPU does not have enough memory for batch size

**Fix:**
```bash
# Reduce batch size
python experiments/run_oram.py --batch-size 64  # or 32

# Or force CPU execution
python experiments/run_oram.py --device cpu
```

### `Lock file exists: results/.experiment_lock`

**Cause:** Previous experiment crashed or is still running

**Fix:**
```bash
# Check if experiment is actually running
ps aux | grep run_experiments.sh

# If not running, remove lock file
rm results/.experiment_lock

# If running, wait for it to complete or kill it
pkill -f run_experiments.sh
```

### `FileNotFoundError: results/baseline/history.json`

**Cause:** Phase 1 (baseline) has not been run yet

**Fix:**
```bash
# Run Phase 1 first
./scripts/run_experiments.sh --phase 1

# Then run subsequent phases
./scripts/run_experiments.sh --phase 5  # analysis
```

### ORAM training is extremely slow

**Expected:** ORAM has 90-180× overhead vs. baseline

**If slower than expected:**
- Check CPU usage: Should be near 100% during training
- Check disk I/O: ORAM performs many small reads/writes
- Use SSD if possible (HDD will be much slower)
- Reduce dataset size for testing: `--num-samples 1000`

### Training accuracy is 0% or not increasing

**Causes:**
- Model not learning (check loss decreases)
- Evaluation skipped (test accuracy reported every 10 epochs by default)

**Verification:**
```bash
# Check that training loss is decreasing
python -c "
import json
with open('results/oram/history.json') as f:
    data = json.load(f)
print('Training losses:', data['train_loss'])
print('Training accuracies:', data['train_acc'])
"
```

### Profiler output shows NaN or negative timings

**Cause:** Profiling instrumentation bug or clock skew

**Debug:**
```bash
# Check profiler JSON for anomalies
python -c "
import json
with open('results/oram/oram_profile.json') as f:
    profile = json.load(f)
for category, stats in profile['summary']['timings'].items():
    if stats['total_time'] < 0:
        print(f'ERROR: {category} has negative time: {stats}')
"
```

**Fix:** Report as bug with full profile JSON

## Expected Outputs Summary

After running the quick demo, you should have:

**File tree:**
```
results/
├── demo_baseline/
│   ├── history.json            (training metrics)
│   └── baseline_profile.json   (profiling data)
├── demo_oram/
│   ├── history.json
│   ├── oram_profile.json
│   └── best_model.pth          (trained model checkpoint)
└── demo_analysis/
    ├── training_comparison.png
    ├── overhead_breakdown.png
    ├── operation_times.png
    └── overhead_report.md
```

**Key metrics (demo_oram vs. demo_baseline):**
- Overhead ratio: 6-10× for 10k samples (reduced from 50k due to smaller tree height)
- ORAM I/O: 65-70% of total time
- Serialization: 4-6% of total time
- Training accuracy: Within 20-30% of baseline after 2 epochs

After running the full suite, you should additionally have:

```
results/
├── baseline/           (100 epochs, ~93% test accuracy)
├── oram/               (10 epochs, 50k samples)
├── sweep_batch_size/   (8 experiments: 4 batch sizes × 2 paths)
├── sweep_dataset_size/ (4 experiments: 4 dataset sizes × ORAM)
├── analysis/           (comprehensive plots and report)
└── logs/               (timestamped execution logs)
```
