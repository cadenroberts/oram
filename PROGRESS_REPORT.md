# CSE239A Progress Report

**Team Members:** Caden Roberts  
**Project Title:** Privacy-Preserving Machine Learning Infrastructure: Scalable Oblivious Computation for Enterprise AI Systems

---

## Problem Definition

Machine learning systems require access to large, sensitive training datasets, yet current privacy approaches — federated learning, differential privacy, standard encryption — protect data values while leaking substantial metadata through access patterns. An adversary observing which records a training pipeline reads, in what order, and how often, can infer dataset composition, model architecture, and convergence behavior without seeing plaintext data. This project characterizes the overhead of integrating Oblivious RAM (ORAM) with PyTorch training on CIFAR-10 to establish a quantitative baseline for where that overhead originates — I/O, cryptographic operations, or memory bandwidth — so that future optimizations (oblivious shuffling, oblivious gradient aggregation) can be targeted at the dominant cost centers.

---

## Progress Summary

### Literature Review (Weeks 1–2, Completed)

I reviewed the following publications that form the theoretical foundation of this work:

- **Stefanov et al., "Path ORAM" (CCS 2013):** The core ORAM scheme I integrated. Path ORAM stores data in a binary tree of encrypted buckets, achieving O(log N) bandwidth overhead per access with a small client-side stash. For N = 50,000 CIFAR-10 samples, this means approximately 16 block accesses per read — the theoretical floor my experiments measure against.
- **Talur & Demertzis, "SONIC" (USENIX Security 2026):** Concurrent ORAM with data-structure-level parallelism. Relevant to Phase 3 (Oblivious Model Serving) of the broader thesis, as it demonstrates that ORAM throughput can scale with concurrent access — a requirement for serving multiple inference requests simultaneously.
- **Mavrogiannakis et al., "OBLIVIATOR" (USENIX Security 2025):** Oblivious parallel operators for join-like computations. Their reported 5–20× overhead for oblivious operators informs my hypothesis that optimized oblivious training should incur 10–50× overhead (versus the potentially much worse naive integration I am measuring).
- **Ngai, Demertzis et al., "Distributed & Scalable Oblivious Sorting and Shuffling" (IEEE S&P 2024):** Directly applicable to oblivious batch shuffling, the primary optimization I will design post-baseline. Their distributed sorting primitive is a candidate building block for hiding which samples are assigned to each training batch.
- **Dauterman et al., "Snoopy" (SOSP 2021):** Scalable oblivious storage that separates the ORAM position map from data storage. Informed my design choice to use PyORAM's on-disk storage rather than keeping the full ORAM tree in memory.
- **Demertzis et al., "SEAL" (USENIX Security 2020):** The adjustable-leakage framework I adopt for my threat model — permitting leakage of dataset size, epoch count, and architecture metadata while hiding record-level access patterns.
- **Demertzis et al., "Searchable Encryption with Optimal Locality" (CRYPTO 2018):** Background for the Encrypted Feature Store thrust (future work, out of scope for CSE239A).

### Implementation Progress (Weeks 3–6)

The full ORAM-integrated training pipeline is implemented and operational. The codebase consists of five core modules:

1. **`src/oram_storage.py` — ORAM Storage Layer:** Wraps PyORAM's Path ORAM to store CIFAR-10 samples as encrypted 4KB blocks. Each block holds a 32×32×3 image (3,072 bytes), a 1-byte label, a 4-byte index, and padding. Provides `read()`, `write()`, and `batch_read()` operations, each instrumented with profiling hooks that track I/O time separately from serialization time.

2. **`src/oram_dataloader.py` — ORAM-Backed PyTorch DataLoader:** Implements a custom `ORAMDataset` (subclass of `torch.utils.data.Dataset`) whose `__getitem__` performs an ORAM read. Includes an `ObliviousBatchSampler` that generates shuffled batch indices (placeholder for future oblivious shuffling). Integrates standard CIFAR-10 augmentation (random crop, horizontal flip, normalization).

3. **`src/oram_trainer.py` — ORAM-Integrated Trainer:** Trains a CIFAR-10-adapted ResNet-18 (modified conv1 kernel, removed maxpool, 10-class output) using the ORAM DataLoader. Supports configurable sample counts for subset experiments and records per-batch, per-epoch, and per-category profiling data.

4. **`src/baseline_trainer.py` — Standard Baseline Trainer:** Identical model architecture and hyperparameters (SGD, LR=0.1, milestones at 50/75/90, weight_decay=5e-4), using the standard PyTorch DataLoader with 4 workers. Establishes the performance floor.

5. **`src/profiler.py` — Profiling Infrastructure:** Singleton profiler with context-manager-based time tracking across categories: `io`, `oram_read`, `oram_write`, `serialize`, `deserialize`, `shuffle`, `dataload`, `compute`, `transfer`, `batch`, `epoch`. Records memory usage via `psutil`. Outputs JSON for automated analysis.

### Experiment Infrastructure (Weeks 5–6)

I built a complete experiment automation suite:

- **`experiments/run_baseline.py`** and **`experiments/run_oram.py`:** CLI runners for individual baseline and ORAM training runs.
- **`experiments/run_sweep.py`:** Parameter sweep runner for batch-size variation ({32, 64, 128, 256}) and dataset-size scaling ({1k, 5k, 10k, 50k samples}) — the core experiments from Weeks 7–8 of the proposal.
- **`experiments/analyze_results.py`:** Automated analysis pipeline that generates training-curve comparisons, overhead breakdown pie/bar charts, per-operation timing distributions, batch-size sweep plots, and dataset-size scaling plots (with O(log N) theoretical reference). Produces a Markdown report.
- **`scripts/run_experiments.sh`:** Five-phase orchestrator (baseline → ORAM → batch-size sweep → dataset-size sweep → analysis) with lock-file concurrency guard, skip-if-exists logic, and structured logging.
- **`scripts/install_cron.sh`:** Installs cron entries for unattended experiment execution and nightly analysis re-generation.

### Preliminary Results

A 2-epoch baseline test run on CPU confirms the pipeline is functional:

| Metric | Value |
|--------|-------|
| Compute time (2 epochs) | 2,860.3s |
| Batches per epoch | 390 (50,000 samples / 128 batch size) |
| Avg batch time | 3,667ms |
| Peak memory (RSS) | 786 MB |
| Epoch 1 train accuracy | 23.3% |
| Epoch 2 train accuracy | 43.6% |

The model is learning (accuracy rose from 23% to 44% over 2 epochs), confirming correctness. A full 100-epoch baseline run is currently executing (launched February 10, 2026), followed by the ORAM and sweep experiments.

### Evaluation Against Proposed Timeline

| Proposed | Planned Activity | Actual Status |
|----------|-----------------|---------------|
| Weeks 1–2 | Literature review, identify ORAM library | **Completed.** Reviewed 7 papers. Selected PyORAM (Path ORAM implementation). |
| Weeks 3–4 | Set up CIFAR-10 baseline, integrate ORAM with DataLoader | **Completed.** Both baseline and ORAM-integrated trainers implemented and tested. |
| Weeks 5–6 | Debug integration, verify correctness | **Completed.** 2-epoch baseline validated (model converges). ORAM storage read/write verified. Experiment automation built. |
| Weeks 7–8 | Run overhead experiments (vary batch size, model complexity) | **In progress.** Experiment suite is automated and running. Full results expected within days. |
| Week 9 | Analyze overhead breakdown | **Partially ready.** Analysis scripts are written; awaiting experiment data. |
| Week 10 | Final report and presentation | **Not started.** |

**Assessment:** I am on schedule. Weeks 1–6 deliverables are complete. I built the experiment automation infrastructure during Weeks 5–6 (ahead of the Week 7 start for experiments), which means the overhead experiments are already running. The slight deviation is that experiments are running on CPU (no local GPU), making wall-clock times longer, but the profiling categories (I/O, crypto, compute, memory) are identical to what they would be on GPU — the overhead *ratios* between ORAM and baseline are the key measurement.

---

## Updated Timeline

| Week | Tasks |
|------|-------|
| **Week 7** (Feb 10–16) | Collect full baseline results (100 epochs). Collect ORAM training results (10 epochs, 50k samples). Begin batch-size sweep experiments. |
| **Week 8** (Feb 17–23) | Complete batch-size sweep ({32, 64, 128, 256}). Complete dataset-size scaling sweep ({1k, 5k, 10k, 50k}). Run analysis pipeline. Generate all figures. |
| **Week 9** (Feb 24–Mar 2) | Analyze overhead breakdown: identify dominant cost centers (I/O vs. crypto vs. memory). Compare measured overhead to theoretical O(log N) bounds. Draft results and discussion sections of report. |
| **Week 10** (Mar 3–9) | Write final report. Prepare presentation. Submit code and results. |

---

## TODO List

- [ ] Collect 100-epoch baseline training results (running, ETA ~40 hours on CPU)
- [ ] Collect 10-epoch ORAM training results on full 50k CIFAR-10
- [ ] Complete batch-size sweep: {32, 64, 128, 256} × {baseline, ORAM} × 3 epochs
- [ ] Complete dataset-size sweep: {1k, 5k, 10k, 50k} × ORAM × 2 epochs
- [ ] Run `analyze_results.py` to generate all figures and overhead report
- [ ] Compute theoretical O(log N) overhead bounds and compare to measured values
- [ ] Identify top-3 overhead categories and discuss optimization opportunities
- [ ] Write final report: introduction, methodology, results, theoretical comparison, optimization targets, conclusion
- [ ] Prepare presentation slides
- [ ] Push final results and report to repository

---

## Git Repository Link

**https://github.com/cadenroberts/oram**

---

## Challenges and Open Questions

1. **CPU-only training speed.** My local workstation does not have a CUDA GPU, so ResNet-18 training on CIFAR-10 takes approximately 20 minutes per epoch on CPU. The 100-epoch baseline run alone will take roughly 33 hours. This does not affect the validity of the overhead measurements — the I/O, crypto, and memory overhead ratios between ORAM and baseline training are architecture-independent — but it does constrain how many experiment configurations I can run within the timeline. I have mitigated this by automating all experiments with cron and using subset sizes (1k–10k samples) for the sweep experiments.

2. **PyORAM version compatibility.** The installed PyORAM version (0.2.1) predates the 0.3.0 version specified in my requirements. The Path ORAM API is compatible and all integration tests pass, but I am monitoring for any edge-case issues with large block counts (50k).

3. **ORAM initialization time.** Loading 50,000 CIFAR-10 samples into Path ORAM requires 50,000 individual encrypted block writes. At the ORAM's O(log N) write bandwidth, this initialization step alone is expected to take significant time. This is a one-time cost (not per-epoch), but it complicates the measurement of per-epoch overhead. I will report initialization time separately from training time.

4. **Isolating cryptographic overhead.** PyORAM performs encryption internally during block reads/writes. My profiler tracks total ORAM read/write time, but isolating the cryptographic component (AES encryption/decryption) from the tree-traversal component requires either instrumenting PyORAM internals or comparing against a "dummy ORAM" that does tree traversal without encryption. This remains an open design question for the analysis.

5. **Oblivious shuffling baseline.** My current `ObliviousBatchSampler` uses standard NumPy random shuffling as a placeholder. For the overhead breakdown, this means the "shuffle" category reflects standard shuffling cost, not oblivious shuffling. The comparison between this baseline shuffle cost and the ORAM data-access cost will indicate whether oblivious shuffling or oblivious storage access is the higher-priority optimization target.
