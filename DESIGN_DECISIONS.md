# Design Decisions

Architecture decision records for the ORAM-integrated training system.

---

## ADR-001: Path ORAM as baseline implementation

**Context:**
Multiple ORAM schemes exist with different performance characteristics: Path ORAM (O(log N) bandwidth, simple), Ring ORAM (lower constants), Oblivious Heaps (specialized for priority queues), and concurrent schemes like SONIC.

**Decision:**
Use PyORAM's Path ORAM implementation as the baseline.

**Rationale:**
- Path ORAM is well-studied with provable O(log N) bandwidth bounds
- PyORAM provides a mature, stable implementation
- Simple tree-based structure is easier to instrument and profile
- Establishes a performance floor that concurrent schemes can be compared against

**Consequences:**
- Sequential access only (no parallelism within ORAM operations)
- O(log N) tree traversal overhead per sample read
- Client stash size grows logarithmically with dataset size
- Single-threaded constraint (num_workers=0 in DataLoader)

**Alternatives considered:**
- Ring ORAM: Lower constants but more complex eviction logic
- SONIC: Concurrent access but requires distributed coordination
- Oblivious Heaps: Specialized for priority queues, not general data access

---

## ADR-002: 4KB block size for CIFAR-10 samples

**Context:**
PyORAM requires fixed-size blocks. CIFAR-10 images are 32×32×3 = 3072 bytes, plus 1-byte label and 4-byte index = 3077 bytes minimum.

**Decision:**
Use 4096-byte (4KB) blocks with zero padding.

**Rationale:**
- 4KB is the smallest power-of-two that fits the 3077-byte payload
- Aligns with PyORAM's block size requirements
- Matches common filesystem block sizes (efficient disk I/O)
- Minimal padding overhead (1019 bytes per block, ~25%)

**Consequences:**
- Disk storage overhead: 4096 bytes per sample vs. 3077 bytes (33% increase)
- Total storage for 50k samples: 200MB (vs. 150MB without padding)
- AES encryption operates on full 4KB blocks

**Alternatives considered:**
- 2KB blocks: Too small, cannot fit payload
- 8KB blocks: Wastes 5KB per sample (160% overhead)
- Variable-size blocks: Not supported by PyORAM Path ORAM

---

## ADR-003: Single-threaded DataLoader for ORAM

**Context:**
Standard PyTorch DataLoader uses multiple worker processes (default: 4) for parallel data loading. ORAM requires exclusive access to the storage file.

**Decision:**
Set `num_workers=0` for ORAM-backed DataLoader (single-threaded).

**Rationale:**
- PyORAM's Path ORAM implementation requires file-level locking
- Multiple processes accessing the same ORAM storage file would cause race conditions
- ORAM's cryptographic operations and tree updates are not thread-safe

**Consequences:**
- Eliminates PyTorch's multi-worker data loading advantage
- Baseline benefits from 4-worker parallelism, ORAM does not
- Overhead comparison includes both ORAM I/O cost and lost parallelism
- Per-batch data loading becomes a serialization bottleneck

**Alternatives considered:**
- Distributed ORAM: Requires multi-process coordination (out of scope)
- Per-worker ORAM instances: Wastes memory and violates oblivious access (each worker would leak access patterns)
- Thread-safe ORAM wrapper: Adds locking overhead without parallelism benefit

---

## ADR-004: Profiler as singleton with context-manager tracking

**Context:**
Overhead measurement requires instrumentation at multiple levels: ORAM I/O, serialization, shuffling, compute, memory. Need a centralized profiling API accessible across modules.

**Decision:**
Implement `Profiler` as a singleton class with context-manager-based time tracking.

**Rationale:**
- Singleton ensures global access without explicit parameter passing
- Context managers (`with profiler.track(category)`) provide clean, exception-safe timing
- Centralized state simplifies per-epoch and per-batch metric collection
- Thread-safe via lock (though single-threaded execution makes this mostly defensive)

**Consequences:**
- All profiling data aggregates in one instance
- Easy to disable profiling globally (`profiler.disable()`)
- Context managers integrate naturally with existing code structure
- JSON export is straightforward (single data structure)

**Alternatives considered:**
- Per-module profilers: More isolated but harder to aggregate metrics
- Decorator-based profiling: Less flexible for instrumenting PyTorch library code
- External profiling tools (cProfile, PyTorch Profiler): Less control over custom categories

---

## ADR-005: ResNet-18 adapted for CIFAR-10

**Context:**
CIFAR-10 images are 32×32 pixels (small compared to ImageNet's 224×224). Standard ResNet-18 has aggressive downsampling in early layers optimized for larger images.

**Decision:**
Modify ResNet-18 for CIFAR-10:
- Replace 7×7 stride-2 conv1 with 3×3 stride-1
- Remove maxpool after conv1
- Change final FC from 1000 → 10 classes

**Rationale:**
- Prevents excessive spatial downsampling that loses information on small images
- Standard CIFAR-10 adaptation documented in literature
- Same architecture used in baseline and ORAM (isolates data loading overhead)

**Consequences:**
- Model has same depth and parameter count as standard ResNet-18 (11.2M params)
- Converges to ~93% test accuracy on CIFAR-10 (standard for this architecture)
- Cannot directly compare with ImageNet-pretrained weights

**Alternatives considered:**
- ResNet-20/32/44 (custom CIFAR-10 architectures): Non-standard, harder to compare with literature
- Standard ResNet-18 without modifications: Poor accuracy due to over-downsampling

---

## ADR-006: SGD with MultiStepLR schedule

**Context:**
Need to choose optimizer and learning rate schedule for ResNet-18 training.

**Decision:**
Use SGD with:
- Initial learning rate: 0.1
- Momentum: 0.9
- Weight decay: 5e-4
- MultiStepLR milestones: [50, 75, 90] with gamma=0.1

**Rationale:**
- Standard CIFAR-10 training recipe from literature
- SGD with momentum generalizes better than Adam for vision tasks
- MultiStepLR schedule provides stable convergence
- Same hyperparameters for baseline and ORAM (isolates data loading overhead)

**Consequences:**
- Requires 100 epochs to converge (~93% accuracy)
- Early epochs are dominated by data loading overhead (not compute)
- Learning rate schedule is fixed (no adaptive scheduling)

**Alternatives considered:**
- Adam optimizer: Faster initial convergence but worse generalization
- Cosine annealing schedule: More complex, no clear benefit for this task
- Lower learning rate: Slower convergence without accuracy improvement

---

## ADR-007: Oblivious batch sampler as placeholder

**Context:**
Standard PyTorch DataLoader shuffles batch indices each epoch. True oblivious shuffling requires specialized algorithms (e.g., Bitonic sort, oblivious random permutation).

**Decision:**
Implement `ObliviousBatchSampler` using standard NumPy shuffle as a placeholder.

**Rationale:**
- Focus Phase 1 on ORAM storage overhead measurement
- Standard shuffling cost is negligible (<1% of total time)
- Future work can replace with true oblivious shuffling and measure the delta

**Consequences:**
- Profiler "shuffle" category reflects standard shuffle cost (not oblivious)
- Access pattern of batch index generation is not oblivious (leaked to OS/memory system)
- Overhead comparison focuses on ORAM I/O, not batch-level obliviousness

**Alternatives considered:**
- Implement oblivious shuffling immediately: Out of scope for Phase 1, would conflate two sources of overhead
- No shuffling: Poor training convergence (samples seen in same order each epoch)

---

## ADR-008: Test set uses standard DataLoader

**Context:**
Test set evaluation happens every 10 epochs. ORAM-backing the test set would double experiment runtime.

**Decision:**
Use standard PyTorch DataLoader (not ORAM-backed) for test set evaluation.

**Rationale:**
- Test set access patterns are less critical to hide (no training data leakage)
- Evaluation is already serialized (no batching parallelism needed)
- Speeds up experiments significantly (10× faster evaluation)

**Consequences:**
- Test set access patterns are visible to adversary
- Cannot measure ORAM overhead during evaluation
- Asymmetric data paths (ORAM for training, standard for testing)

**Alternatives considered:**
- ORAM-backed test set: More consistent but impractical for experiment velocity
- Skip evaluation: Loses accuracy tracking and convergence verification

---

## ADR-009: Experiment orchestrator with phase-based execution

**Context:**
Full experiment suite has 5 phases: baseline, ORAM, batch-size sweep, dataset-size sweep, analysis. Each phase produces artifacts needed by later phases.

**Decision:**
Implement `scripts/run_experiments.sh` as a Bash orchestrator with:
- Lock file to prevent concurrent runs
- Skip-if-exists logic (idempotent reruns)
- Per-phase logging to timestamped files
- Individual phase execution via `--phase N`

**Rationale:**
- Long-running experiments need resumability (40+ hours on CPU)
- Lock file prevents accidental overlapping runs (corrupted results)
- Phase-based execution allows iterative development (rerun analysis without retraining)
- Bash is portable and integrates with cron scheduling

**Consequences:**
- Lock file must be manually removed if experiment crashes
- No distributed execution (single machine only)
- Log files accumulate in `results/logs/` (manual cleanup needed)

**Alternatives considered:**
- Python-based orchestrator: More complex, harder to debug
- Makefile: Less flexible for conditional phase execution
- No orchestrator: Manual invocation error-prone for 5-phase pipeline

---

## ADR-010: JSON export for profiling data

**Context:**
Profiler collects timing and memory data across epochs and batches. Need a serialization format for automated analysis.

**Decision:**
Export profiling data as JSON with structure:
```json
{
  "summary": { "timings": {...}, "memory": {...} },
  "overhead_breakdown": { "io": 65.3, "compute": 20.1, ... },
  "epoch_data": [ { "epoch": 1, "timings": {...}, "metrics": {...} }, ... ],
  "batch_data": [ { "epoch": 1, "batch": 0, "loss": 2.3, ... }, ... ]
}
```

**Rationale:**
- JSON is human-readable and machine-parseable
- Python's `json` module provides stable serialization
- Analysis scripts can load JSON with `pandas.read_json()`
- Schema is self-documenting (no external format specification needed)

**Consequences:**
- Large JSON files for long experiments (100k+ batches)
- No schema validation (malformed JSON causes analysis failures)
- Not space-efficient (compared to binary formats like HDF5)

**Alternatives considered:**
- HDF5: More efficient but less human-readable
- CSV: Hierarchical data is awkward to represent
- Pickle: Python-specific, not portable across versions
