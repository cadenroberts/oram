# Architecture

## System overview

The experiment pipeline is designed to isolate and quantify the performance overhead introduced by ORAM relative to a standard PyTorch data path. I compare two training paths: a standard PyTorch DataLoader path and an ORAM-backed path. Both train the same ResNet-18 model on CIFAR-10. The ORAM path interposes a Path ORAM tree between the dataset and the training loop, adding encrypted block I/O and tree reshuffling to every sample access.

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃          Experiment Orchestrator (scripts/run_experiments.sh)         ┃
┃┌──────────┐┌─────────────┐┌──────────────┐┌─────────────┐┌───────────┐┃┌────────────────────────────────────────────────┐
┃│ Baseline ├┼▶ ORAM train ├┼▶ Batch Sweep ├┼▶ Data Sweep ├┼▶ Analysis ├╂┼▶ results/ # JSON profiles, logs, plots, report │
┃└──────────┘└─────────────┘└──────────────┘└─────────────┘└───────────┘┃└────────────────────────────────────────────────┘
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

```

## Data path: standard vs. ORAM

### Standard path

```
CIFAR-10 (disk) → torchvision.datasets → DataLoader (4 workers) → GPU
```

### ORAM path

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃       ORAMStorage # AES-enc 4KB blocks        ┃
┃┌─────────────────┐┌──────────────────────────┐┃┌──────────────┐┌───────────────────────────────────────────┐ 
┃│ CIFAR-10 # disk ├┼▶ load_cifar10_to_oram() ◀┼┃┼▶ struct.pack ├┼▶ ORAMStorage.read() # ST, O(log N) blocks ├────────────┐
┃└─────────────────┘└──────────────────────────┘┃└──────────────┘└───────────────────────────────────────────┘            │
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛                                                      ┌──────────────────┤
┌──────────────────┐┌─────────────────────────────────┐┌──────────────────────────────────────────────┐│┌────────────────┐│               
│ GPU # ResNet-18 ◀┼┤ ObliviousBatchSampler # seeded ◀┼┤ ORAMDataset.__getitem__() # num_workers = 0 ◀┼┴┤ struct.unpack ◀┼┘ 
└──────────────────┘└─────────────────────────────────┘└──────────────────────────────────────────────┘ └────────────────┘
```

## Profiling architecture

The `Profiler` singleton instruments every layer of the ORAM data path:

```
┌───────────────────────────────────────────────┐
│      Profiler (singleton, thread-safe)        │
│   ┌────────────┬──────────────────────────┐   │
│   │ io         │ ORAM block read/write    │   │
│   │ oram_read  │ PathORAM.read_block()    │   │
│   │ oram_write │ PathORAM.write_block()   │   │
│   │ serialize  │ image → block bytes      │   │
│   │ deserialize│ block bytes → image      │   │
│   │ shuffle    │ batch index generation   │   │
│   │ dataload   │ full __getitem__ cost    │   │
│   │ compute    │ forward + backward pass  │   │
│   │ transfer   │ CPU → GPU data transfer  │   │
│   │ batch      │ total per-batch time     │   │
│   │ epoch      │ total per-epoch time     │   │
│   │ setup      │ ORAM tree initialization │   │
│   └────────────┴──────────────────────────┘   │
│  Outputs:                                     │
│  • TimingStats (total, count, min, max, avg)  │
│  • MemoryStats (peak RSS, peak VMS)           │
│  • Per-epoch and per-batch metric records     │
│  • JSON export for analysis                   │
└───────────────────────────────────────────────┘
```

## Block format

Each CIFAR-10 sample occupies one 4KB ORAM block:

```
Offset  Size    Field
0       4       Sample index (uint32, little-endian)
4       1       Label (uint8, 0-9)
5       3072    Image pixels (32×32×3, uint8, row-major)
3077    1019    Zero padding
```

4KB block size is the smallest power-of-two that accommodates the 3077-byte payload while satisfying PyORAM's block alignment requirements.

## Model configuration

ResNet-18 adapted for CIFAR-10:
- `conv1`: 3×3 kernel, stride 1, padding 1 (replaces 7×7 stride-2 for 32×32 input)
- `maxpool`: replaced with `nn.Identity()`
- `fc`: 512 → 10 (CIFAR-10 classes)
- Optimizer: SGD (lr=0.1, momentum=0.9, weight_decay=5e-4)
- Schedule: MultiStepLR at epochs 50, 75, 90 (γ=0.1)
