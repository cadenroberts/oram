# OMLO (Oblivious ML-Ops) Privacy-Preserving Machine Learning Infrastructure: Scalable Oblivious Computation for Enterprise AI Systems

I propose Oblivious ML-Ops (OMLO), a framework for privacy-preserving machine learning infrastructure that enables training and serving of ML models while hiding data-dependent access
patterns. This research extends recent work on oblivious computation and searchable encryption
to address metadata leakage in ML systems—a gap left open by federated learning and differential
privacy approaches, which protect data values but not access patterns. For CSE239A, I will characterize the overhead of applying ORAM to ML training workloads, establishing a baseline for the
optimizations proposed in the broader research vision.

# ORAM-Integrated PyTorch Training

Quantifies computational and I/O overhead when integrating Path ORAM into GPU-backed PyTorch training workflows.

## What it does

- Trains ResNet-18 on CIFAR-10 with standard PyTorch DataLoader (baseline)
- Trains ResNet-18 on CIFAR-10 with ORAM-backed data loading (oblivious access)
- Measures per-sample overhead across six categories: block I/O, AES encryption, tree reshuffling, serialization, forward/backward compute, memory
- Validates O(log N) scaling of Path ORAM against theoretical bounds
- Generates overhead breakdown charts and training curve comparisons

## Architecture

The system compares two data paths:

**Standard path:**
```
CIFAR-10 (disk) → torchvision.datasets → DataLoader (4 workers) → GPU → ResNet-18
```

**ORAM path:**
```
CIFAR-10 (disk) → ORAMStorage.write() → Path ORAM tree (AES-encrypted 4KB blocks)
                → ORAMStorage.read() → O(log N) block accesses → deserialize
                → DataLoader (0 workers, oblivious batch sampler) → GPU → ResNet-18
```

Key architectural constraints:
- ORAM requires single-threaded access (num_workers=0)
- Each sample access traverses O(log N) encrypted blocks with tree eviction
- Block size: 4096 bytes (3072-byte image + 1-byte label + 4-byte index + padding)

See [ARCHITECTURE.md](ARCHITECTURE.md) for full component diagram and data flow.

## Design tradeoffs

**Path ORAM selection:**
- Chosen for simplicity and mature PyORAM implementation
- O(log N) bandwidth per access is theoretical floor
- Tree-based structure requires sequential access (no parallelism)
- Alternative: Ring ORAM or concurrent ORAM schemes (future work)

**ResNet-18 model:**
- Standard CIFAR-10 architecture modifications: 3×3 conv1, no maxpool, 10-class output
- SGD optimizer (lr=0.1, momentum=0.9, weight_decay=5e-4)
- MultiStepLR schedule (decay at epochs 50, 75, 90)
- Same hyperparameters for baseline and ORAM (isolates data loading overhead)

**Profiling strategy:**
- Singleton profiler with context-manager-based tracking
- Categories: io, oram_read, oram_write, serialize, deserialize, shuffle, compute, transfer, batch, epoch
- Memory profiling via psutil (peak RSS, peak VMS)
- Per-batch and per-epoch metrics exported to JSON

**Single-threaded ORAM constraint:**
- PyORAM Path ORAM implementation requires exclusive access
- Eliminates PyTorch's multi-worker data loading advantage
- Future work: distributed ORAM with concurrent access (e.g., SONIC)

## Evaluation

**Correctness:**
- ORAM training should converge to similar accuracy as baseline (~93% on CIFAR-10)
- Overhead ratio should scale with O(log N) across dataset sizes {1k, 5k, 10k, 50k}
- Peak memory should increase proportionally to ORAM tree height

**Performance:**
- Per-sample data load time: baseline ~0.01ms, ORAM ~5-15ms (500-1500× overhead)
- Per-epoch wall time (batch=128): baseline ~30s, ORAM ~45-90min (90-180× overhead)
- Peak memory: baseline ~1.2GB, ORAM ~1.8-2.5GB (1.5-2× overhead)

**Commands:**
```bash
# Run full experiment pipeline (5 phases)
./scripts/run_experiments.sh

# Run individual experiments
python experiments/run_baseline.py --epochs 100 --batch-size 128
python experiments/run_oram.py --epochs 10 --batch-size 128
python experiments/run_sweep.py --sweep batch_size --epochs 3
python experiments/run_sweep.py --sweep dataset_size --epochs 2

# Regenerate analysis from existing data
python experiments/analyze_results.py --baseline results/baseline --oram results/oram --output results/analysis
```

**Pass/fail criteria:**
- Training accuracy must reach >90% for baseline, >85% for ORAM
- Overhead scaling must follow O(log N) within 20% variance
- No memory leaks (peak RSS stable across epochs)

See [EVAL.md](EVAL.md) for detailed evaluation methodology.

## Demo

**Prerequisites:**
- Python 3.8+
- 20GB free disk space (CIFAR-10 + ORAM storage + results)
- GPU recommended but not required (experiments validated on CPU)

**Quick demo (2 epochs each, ~15 minutes on CPU):**
```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run baseline (2 epochs)
python experiments/run_baseline.py --epochs 2 --batch-size 128 --output-dir results/demo_baseline

# Run ORAM (2 epochs, 10k samples)
python experiments/run_oram.py --epochs 2 --batch-size 128 --num-samples 10000 --output-dir results/demo_oram

# Compare results
python experiments/analyze_results.py --baseline results/demo_baseline --oram results/demo_oram --output results/demo_analysis

# Verify outputs
ls results/demo_analysis/
# Expected: training_comparison.png, overhead_breakdown.png, operation_times.png, overhead_report.md
```

**Full experiment suite (Phase 1-5, ~40 hours on CPU):**
```bash
./scripts/run_experiments.sh
```

**Troubleshooting:**
- `ImportError: No module named 'pyoram'`: Run `pip install -r requirements.txt`
- `CUDA out of memory`: Reduce batch size or use `--device cpu`
- `Lock file exists`: Another experiment is running or crashed. Remove `results/.experiment_lock`

See [DEMO.md](DEMO.md) for detailed demo walkthrough with expected outputs.

## Repository layout

```
oram/
├── src/                        Core library modules
│   ├── oram_storage.py         PyORAM wrapper for CIFAR-10 block storage
│   ├── oram_dataloader.py      ORAM-backed PyTorch Dataset + BatchSampler
│   ├── oram_trainer.py         ORAM-integrated training with profiling
│   ├── baseline_trainer.py     Standard training for comparison
│   └── profiler.py             Overhead measurement infrastructure
├── experiments/                Experiment entry points
│   ├── run_baseline.py         Baseline training experiments
│   ├── run_oram.py             ORAM training experiments
│   ├── run_sweep.py            Batch-size and dataset-size sweeps
│   └── analyze_results.py      Overhead breakdown analysis and plots
├── scripts/
│   └── run_experiments.sh      Master orchestrator (all 5 phases)
├── results/                    Experiment outputs (gitignored)
│   ├── baseline/               Baseline training results
│   ├── oram/                   ORAM training results
│   ├── sweep_batch_size/       Batch-size sweep results
│   ├── sweep_dataset_size/     Dataset-size sweep results
│   ├── analysis/               Generated plots and reports
│   └── logs/                   Execution logs
├── README.md                   This file
├── ARCHITECTURE.md             System architecture and component diagram
├── DESIGN_DECISIONS.md         Architecture decision records
├── EVAL.md                     Evaluation methodology and metrics
├── DEMO.md                     Demo walkthrough with expected outputs
├── REPO_AUDIT.md               Repository audit and improvement priorities
├── PATCHSET_SUMMARY.md         Documentation overhaul summary
├── requirements.txt            Python dependencies
├── .gitignore                  Gitignored files
└── sync.sh                     Git commit and push helper
```

## Limitations and scope

**Limitations:**
- CPU-only validation (GPU experiments pending hardware access)
- Single-machine execution (no distributed ORAM)
- PyORAM 0.3.0 Path ORAM only (no Ring ORAM or concurrent schemes)
- No checkpointing or resume capability for long-running experiments
- No real-time monitoring or dashboard (post-hoc analysis only)
- Test set uses standard DataLoader (not ORAM-backed) for faster evaluation

**Out of scope:**
- Oblivious shuffling optimization (placeholder implementation only)
- Oblivious gradient aggregation
- Federated learning integration
- Production deployment considerations
- Security analysis of PyORAM implementation
- Comparison with differential privacy or secure enclaves

**Future work:**
- Integrate concurrent ORAM schemes (SONIC, Pathos)
- Implement oblivious shuffling (Bitonic sort, oblivious random permutation)
- Extend to larger models (ResNet-50, Vision Transformers)
- Distributed ORAM across multiple machines
- GPU-accelerated cryptographic operations

## References

- Stefanov et al., "Path ORAM: An Extremely Simple Oblivious RAM Protocol" (CCS 2013)
- PyORAM: https://github.com/ghackebeil/PyORAM
- Talur, Demertzis, "SONIC: Concurrent Oblivious RAM" (USENIX Security 2026)
- Mavrogiannakis et al., "OBLIVIATOR" (USENIX Security 2025)
- Ngai et al., "Distributed & Scalable Oblivious Sorting and Shuffling" (IEEE S&P 2024)

## License

MIT
