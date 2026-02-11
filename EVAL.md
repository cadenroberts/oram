# Evaluation Methodology

## Correctness Definition

The ORAM-integrated training system is correct if:

1. **Model convergence:** ORAM training reaches ≥85% test accuracy on CIFAR-10 (within 10% of baseline ~93%)
2. **Overhead scaling:** Per-access ORAM cost scales logarithmically with dataset size N (within 20% of theoretical O(log N))
3. **Memory stability:** Peak memory usage does not grow unboundedly across epochs (no memory leaks)
4. **Profiler consistency:** Sum of per-category timings matches total epoch time (within 5% tolerance for measurement noise)

## Performance Expectations

### Baseline (standard PyTorch DataLoader)

**Expected performance (ResNet-18, CIFAR-10, batch_size=128):**
- Per-sample data load time: ~0.01ms (4 workers, parallel loading)
- Per-epoch wall time (50k samples): ~30s on CPU, ~10s on GPU
- Peak memory (RSS): ~1.2GB (model + optimizer + batch + data cache)
- Training accuracy (100 epochs): ~93%

**Theoretical bounds:**
- Batch loading: O(B) where B = batch size
- Epoch time: O(N/B × T_compute) where N = dataset size, T_compute = per-batch forward+backward time
- Memory: O(model_params + batch_size × sample_size)

### ORAM (Path ORAM-backed data loading)

**Expected performance (same config):**
- Per-sample data load time: ~5-15ms (O(log N) block accesses × block I/O + crypto + serialization)
- Per-epoch wall time: ~45-90min on CPU (90-180× overhead)
- Peak memory: ~1.8-2.5GB (ORAM tree structure + stash + position map)
- Training accuracy (100 epochs): ~93% (same convergence as baseline)

**Theoretical bounds:**
- ORAM bandwidth: O(log N) blocks per access (Path ORAM)
- Block I/O time: O(block_size) = O(4096 bytes)
- Crypto overhead: O(block_size) for AES encryption/decryption
- Tree height: ceil(log2(N)) = 16 for N=50k
- Expected blocks per access: ~16 for N=50k

### Overhead Breakdown

Expected time distribution during ORAM training:

| Category | Expected % | Measurement method |
|----------|-----------|-------------------|
| ORAM block I/O | 60-70% | `profiler.track('io')` |
| Serialization/deserialization | 5-10% | `profiler.track('serialize')` + `profiler.track('deserialize')` |
| Batch shuffling | <1% | `profiler.track('shuffle')` |
| Forward/backward compute | 15-25% | `profiler.track('compute')` |
| CPU→GPU transfer | 2-5% | `profiler.track('transfer')` |

## Measurable Commands

### Full experiment suite (40+ hours on CPU)

```bash
# Phase 1: Baseline training (100 epochs, 50k samples)
./scripts/run_experiments.sh --phase 1

# Phase 2: ORAM training (10 epochs, 50k samples)
./scripts/run_experiments.sh --phase 2

# Phase 3: Batch-size sweep (3 epochs, 4 batch sizes × 2 paths)
./scripts/run_experiments.sh --phase 3

# Phase 4: Dataset-size sweep (2 epochs, 4 dataset sizes × ORAM)
./scripts/run_experiments.sh --phase 4

# Phase 5: Analysis and report generation
./scripts/run_experiments.sh --phase 5
```

**Expected outputs:**
- `results/baseline/history.json` — Training metrics (100 epochs)
- `results/baseline/baseline_profile.json` — Profiling data
- `results/oram/history.json` — Training metrics (10 epochs)
- `results/oram/oram_profile.json` — Profiling data
- `results/sweep_batch_size/sweep_summary.json` — Batch-size sweep results
- `results/sweep_dataset_size/sweep_summary.json` — Dataset-size sweep results
- `results/analysis/*.png` — Plots (training curves, overhead breakdown, scaling)
- `results/analysis/overhead_report.md` — Detailed analysis

### Quick smoke test (15 minutes on CPU)

```bash
# Baseline (2 epochs)
python experiments/run_baseline.py --epochs 2 --batch-size 128 --output-dir results/smoke_baseline

# ORAM (2 epochs, 10k samples)
python experiments/run_oram.py --epochs 2 --batch-size 128 --num-samples 10000 --output-dir results/smoke_oram

# Verify non-zero accuracy
python -c "
import json
with open('results/smoke_baseline/history.json') as f:
    baseline = json.load(f)
with open('results/smoke_oram/history.json') as f:
    oram = json.load(f)
assert baseline['train_acc'][-1] > 40, f'Baseline acc too low: {baseline[\"train_acc\"][-1]}'
assert oram['train_acc'][-1] > 30, f'ORAM acc too low: {oram[\"train_acc\"][-1]}'
print('SMOKE_OK')
"
```

**Expected output:**
```
SMOKE_OK
```

## Pass/Fail Criteria

### Correctness (must pass)

1. **Model convergence (baseline):**
   - Test accuracy > 90% after 100 epochs
   - Training loss decreases monotonically (no divergence)

2. **Model convergence (ORAM):**
   - Test accuracy > 85% after 100 epochs (within 10% of baseline)
   - Training curves qualitatively similar to baseline (no anomalous spikes)

3. **Overhead scaling:**
   - Dataset-size sweep: overhead ratio follows O(log N) within 20% error
   - Measured tree height matches theoretical ceil(log2(N))

4. **Memory stability:**
   - Peak RSS does not grow >10% across epochs (no memory leaks)
   - Peak RSS scales with O(N × block_size) for ORAM storage

### Performance (must measure, not pass/fail)

5. **Overhead ratio:**
   - Per-sample ORAM load time / baseline load time (expected: 500-1500×)
   - Per-epoch ORAM time / baseline time (expected: 90-180×)

6. **Overhead breakdown:**
   - Identify dominant category (expected: ORAM block I/O at 60-70%)
   - Quantify serialization overhead (expected: 5-10%)

7. **Batch-size sensitivity:**
   - Measure how overhead ratio changes with batch size {32, 64, 128, 256}
   - Larger batches should reduce per-epoch overhead (fewer batches = fewer shuffle operations)

### Profiling (must validate)

8. **Category consistency:**
   - Sum of per-category timings ≈ total epoch time (within 5%)
   - No negative timings or NaN values

9. **Memory profiling:**
   - Peak RSS captured at end of each epoch
   - Memory samples recorded without crashes

## Theoretical Comparison

### Path ORAM bandwidth complexity

For N samples stored in a binary tree:
- Tree height: h = ceil(log2(N))
- Blocks per read: h (path from root to leaf)
- Blocks per write: h (write-back after eviction)
- Total blocks per access: 2h ≈ 2 log2(N)

**Measured vs. theoretical:**
- N=1,000 → h=10 → expected 20 blocks/access
- N=5,000 → h=13 → expected 26 blocks/access
- N=10,000 → h=14 → expected 28 blocks/access
- N=50,000 → h=16 → expected 32 blocks/access

**Validation:**
Compare measured per-access time against theoretical block count:
```
measured_time / theoretical_blocks ≈ constant (per-block I/O + crypto time)
```

### Expected overhead sources

**ORAM I/O dominance:**
- Each sample access: 32 blocks × (4KB read + 4KB write) = 256KB I/O
- Baseline sample access: 3KB read (image + label, cached in memory)
- I/O ratio: 256KB / 3KB ≈ 85× (before accounting for parallelism)

**Serialization overhead:**
- ORAM: struct.pack (3077 bytes) + zero padding (1019 bytes) = 4096 bytes
- Baseline: PyTorch tensor conversion (zero-copy)
- Expected serialization time: ~0.1ms per sample

**Lost parallelism:**
- Baseline: 4 workers × 128 batch size = 512 samples prefetched
- ORAM: 0 workers (single-threaded)
- Parallelism loss: 4× (additional overhead beyond ORAM I/O)

**Total expected overhead:**
```
ORAM_overhead = (I/O_ratio) × (serialization_overhead) × (parallelism_loss)
              ≈ 85 × 1.5 × 4
              ≈ 500-1500× (matches measured range)
```

## Optimization Targets

Based on expected overhead breakdown:

**P0 (highest impact):**
1. ORAM block I/O (60-70% of overhead)
   - Target: Concurrent ORAM (SONIC) to parallelize tree traversal
   - Expected improvement: 4-8× (multi-worker parallelism)

2. Serialization/deserialization (5-10% of overhead)
   - Target: Zero-copy memory mapping (mmap ORAM storage)
   - Expected improvement: 1.5-2× (eliminate struct.pack/unpack)

**P1 (medium impact):**
3. Memory pressure (1.5-2× memory overhead)
   - Target: Compact ORAM position map representation
   - Expected improvement: 1.2-1.5× (reduce cache misses)

**P2 (low impact):**
4. Batch shuffling (<1% of overhead)
   - Target: Oblivious shuffling (Bitonic sort)
   - Expected improvement: Negligible for performance, critical for security

## Reproducibility

To reproduce evaluation results:

1. Pin dependencies:
   ```bash
   pip freeze > requirements.lock
   pip install -r requirements.lock
   ```

2. Set seeds (requires code modification):
   ```python
   torch.manual_seed(42)
   np.random.seed(42)
   ```

3. Enable deterministic CUDA:
   ```python
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False
   ```

4. Use fixed ORAM storage path (avoid temp directories):
   ```bash
   python experiments/run_oram.py --oram-path /fixed/path/oram.bin
   ```

5. Document environment:
   ```bash
   python --version > results/python_version.txt
   uname -a > results/system_info.txt
   nvidia-smi > results/gpu_info.txt  # if GPU available
   ```

Current limitations:
- Dependencies pinned with `>=` (not exact versions)
- No seed management in experiment scripts
- ORAM storage uses temp directories (nondeterministic paths)
- No environment specification (Python version, CUDA version, OS)
