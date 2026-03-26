"""
ORAM infrastructure: storage, data loading, and training.

Storage wraps PyORAM's Path ORAM to store CIFAR-10 samples as encrypted blocks.
DataLoader provides ORAM-backed and mediated loading modes for CIFAR-10.
Train runs full ORAM-integrated training with ResNet on CIFAR-10.

Also includes sidecar batch logging, ORAM audit parsing, device resolution,
and synthetic plaintext/ORAM access-pattern event generators for experiments.
"""

import csv
import os
import random
import time
import struct
import tempfile
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as tv_models
from torch.utils.data import Dataset, DataLoader, Sampler, Subset
from tqdm import tqdm
from pyoram.oblivious_storage.tree.path_oram import PathORAM

from profiler import Profiler, track, ProfilerContext



# ---------------------------------------------------------------------------
# Storage constants
# ---------------------------------------------------------------------------

CIFAR_IMAGE_SIZE = 32 * 32 * 3  # 3072 bytes
LABEL_SIZE = 1
METADATA_SIZE = 4
MIN_BLOCK_SIZE = METADATA_SIZE + LABEL_SIZE + CIFAR_IMAGE_SIZE  # 3077 bytes

DEFAULT_BLOCK_SIZE = 4096

VALID_BACKENDS = ("file", "ram")


# ---------------------------------------------------------------------------
# Record: sidecar logging, audit parsing, and synthetic event generation
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Storage constants
# ---------------------------------------------------------------------------

CIFAR_IMAGE_SIZE = 32 * 32 * 3  # 3072 bytes
LABEL_SIZE = 1
METADATA_SIZE = 4
MIN_BLOCK_SIZE = METADATA_SIZE + LABEL_SIZE + CIFAR_IMAGE_SIZE  # 3077 bytes

DEFAULT_BLOCK_SIZE = 4096

VALID_BACKENDS = ("file", "ram")


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------

@dataclass
class TimingStats:
    total_time: float = 0.0
    call_count: int = 0
    min_time: float = float('inf')
    max_time: float = 0.0
    samples: List[float] = field(default_factory=list)

    def record(self, duration: float, keep_samples: bool = False):
        self.total_time += duration
        self.call_count += 1
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
        if keep_samples:
            self.samples.append(duration)

    @property
    def avg_time(self) -> float:
        return self.total_time / self.call_count if self.call_count > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            'total_time': self.total_time,
            'call_count': self.call_count,
            'avg_time': self.avg_time,
            'min_time': self.min_time if self.min_time != float('inf') else 0.0,
            'max_time': self.max_time,
        }


@dataclass
class MemoryStats:
    peak_rss: int = 0
    peak_vms: int = 0
    samples: List[Dict[str, int]] = field(default_factory=list)

    def record(self):
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        self.peak_rss = max(self.peak_rss, mem_info.rss)
        self.peak_vms = max(self.peak_vms, mem_info.vms)
        self.samples.append({
            'rss': mem_info.rss,
            'vms': mem_info.vms,
            'timestamp': time.time()
        })

    def to_dict(self) -> dict:
        return {
            'peak_rss_mb': self.peak_rss / (1024 * 1024),
            'peak_vms_mb': self.peak_vms / (1024 * 1024),
            'sample_count': len(self.samples),
        }


class Profiler:
    """
    Central profiler for tracking ORAM overhead decomposition.

    Categories tracked:
    - io: ORAM block I/O operations
    - crypto: Encryption/decryption time
    - shuffle: Path ORAM shuffling/eviction
    - compute: Model forward/backward passes
    - dataload: Total data loading time
    - batch: Per-batch total time
    - epoch: Per-epoch total time
    """

    _instance: Optional['Profiler'] = None
    _lock = threading.Lock()

    def __init__(self):
        self.timings: Dict[str, TimingStats] = defaultdict(TimingStats)
        self.memory = MemoryStats()
        self.epoch_data: List[Dict] = []
        self.batch_data: List[Dict] = []
        self.current_epoch: int = 0
        self.current_batch: int = 0
        self._enabled = True
        self._keep_samples = False

    @classmethod
    def get_instance(cls) -> 'Profiler':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        with cls._lock:
            cls._instance = None

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    def set_keep_samples(self, keep: bool):
        self._keep_samples = keep

    @contextmanager
    def track(self, category: str):
        """
        Context manager for tracking time in a category.

        Usage:
            with profiler.track('io'):
                # I/O operations here
        """
        if not self._enabled:
            yield
            return

        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.timings[category].record(duration, self._keep_samples)

    def record_time(self, category: str, duration: float):
        if self._enabled:
            self.timings[category].record(duration, self._keep_samples)

    def record_memory(self):
        if self._enabled:
            self.memory.record()

    def start_epoch(self, epoch: int):
        self.current_epoch = epoch
        self.current_batch = 0

    def end_epoch(self, epoch: int, metrics: Optional[Dict] = None):
        epoch_record = {
            'epoch': epoch,
            'timings': {k: v.to_dict() for k, v in self.timings.items()},
            'memory': self.memory.to_dict(),
        }
        if metrics:
            epoch_record['metrics'] = metrics
        self.epoch_data.append(epoch_record)

    def record_batch(self, batch_idx: int, metrics: Optional[Dict] = None):
        self.current_batch = batch_idx
        if metrics:
            self.batch_data.append({
                'epoch': self.current_epoch,
                'batch': batch_idx,
                **metrics
            })

    def get_summary(self) -> Dict:
        return {
            'timings': {k: v.to_dict() for k, v in self.timings.items()},
            'memory': self.memory.to_dict(),
            'epochs': len(self.epoch_data),
            'total_batches': len(self.batch_data),
        }

    def get_overhead_breakdown(self) -> Dict[str, float]:
        """
        Calculate percentage breakdown of overhead by category.

        Returns dict mapping category -> percentage of total time.
        """
        total = sum(stats.total_time for stats in self.timings.values())
        if total == 0:
            return {}

        return {
            category: (stats.total_time / total) * 100
            for category, stats in self.timings.items()
        }

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        data = {
            'summary': self.get_summary(),
            'overhead_breakdown': self.get_overhead_breakdown(),
            'epoch_data': self.epoch_data,
            'batch_data': self.batch_data[-1000:],
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def print_summary(self):
        print("\n" + "=" * 60)
        print("PROFILER SUMMARY")
        print("=" * 60)

        print("\nTiming Breakdown:")
        print("-" * 40)
        breakdown = self.get_overhead_breakdown()
        for category, pct in sorted(breakdown.items(), key=lambda x: -x[1]):
            stats = self.timings[category]
            print(f"  {category:15} {pct:6.2f}%  "
                  f"(total: {stats.total_time:8.2f}s, "
                  f"calls: {stats.call_count:6d}, "
                  f"avg: {stats.avg_time * 1000:8.3f}ms)")

        print("\nMemory Usage:")
        print("-" * 40)
        print(f"  Peak RSS: {self.memory.peak_rss / (1024 * 1024):.2f} MB")
        print(f"  Peak VMS: {self.memory.peak_vms / (1024 * 1024):.2f} MB")

        print("=" * 60 + "\n")


class ProfilerContext:
    """
    Context manager for scoped profiling with automatic cleanup.

    Usage:
        with ProfilerContext('experiment_name') as profiler:
            # Run experiment
            ...
        # Profiler automatically saved and summarized
    """

    def __init__(self, name: str, output_dir: str = 'results',
                 keep_samples: bool = False):
        self.name = name
        self.output_dir = output_dir
        self.keep_samples = keep_samples
        self.profiler: Optional[Profiler] = None

    def __enter__(self) -> Profiler:
        Profiler.reset()
        self.profiler = Profiler.get_instance()
        self.profiler.set_keep_samples(self.keep_samples)
        self.profiler.record_memory()
        return self.profiler

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.profiler:
            self.profiler.record_memory()
            filepath = os.path.join(self.output_dir, f'{self.name}_profile.json')
            self.profiler.save(filepath)
            self.profiler.print_summary()
        return False


def get_profiler() -> Profiler:
    return Profiler.get_instance()


def track(category: str):
    return get_profiler().track(category)


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

class ORAMStorage:
    """
    ORAM-backed storage for CIFAR-10 dataset samples.

    Each sample (image + label) is stored as an encrypted block
    in a Path ORAM tree structure. Access patterns are hidden
    through the ORAM protocol.

    Attributes:
        num_samples: Number of samples stored
        block_size: Size of each ORAM block in bytes
        backend: Storage backend type ('file' or 'ram')
        oram: Underlying PyORAM PathORAM instance
    """

    def __init__(
        self,
        num_samples: int,
        storage_path: Optional[str] = None,
        backend: str = "file",
        block_size: int = DEFAULT_BLOCK_SIZE,
    ):
        if backend not in VALID_BACKENDS:
            raise ValueError(f"backend must be one of {VALID_BACKENDS}, got '{backend}'")
        if block_size < MIN_BLOCK_SIZE:
            raise ValueError(
                f"block_size {block_size} too small for CIFAR-10 "
                f"(minimum {MIN_BLOCK_SIZE} bytes per sample)")

        self.num_samples = num_samples
        self.block_size = block_size
        self.backend = backend
        self.profiler = get_profiler()

        if backend == "ram":
            self._temp_dir = None
            self.storage_path = storage_path or "oram_ram"
        else:
            if storage_path is None:
                self._temp_dir = tempfile.mkdtemp(prefix='oram_cifar_')
                storage_path = os.path.join(self._temp_dir, 'oram.bin')
            else:
                self._temp_dir = None
            self.storage_path = storage_path

        self._init_oram()

    def _init_oram(self):
        with self.profiler.track('oram_init'):
            self.oram = PathORAM.setup(
                storage_name=self.storage_path,
                block_size=self.block_size,
                block_count=self.num_samples,
                storage_type=self.backend if self.backend != "file" else "file",
            )

    def _audit_io(self, op: str, index: int) -> None:
        """Append ORAM I/O evidence when ORAM_AUDIT_LOG is set."""
        log_path = os.environ.get("ORAM_AUDIT_LOG")
        if not log_path:
            return
        include_index = os.environ.get("ORAM_AUDIT_INCLUDE_INDEX", "0") == "1"
        fields = [
            f"{time.time():.6f}",
            f"op={op}",
            f"backend={self.backend}",
            f"storage_path={self.storage_path}",
        ]
        if include_index:
            fields.append(f"index={index}")
        line = ",".join(fields) + "\n"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line)

    def _serialize_sample(self, image: np.ndarray, label: int, index: int) -> bytes:
        """
        Serialize a CIFAR-10 sample to bytes for ORAM storage.

        Format: [index (4 bytes)] [label (1 byte)] [image (3072 bytes)] [padding]
        """
        with self.profiler.track('serialize'):
            header = struct.pack('<IB', index, label)
            image_bytes = image.astype(np.uint8).tobytes()
            data = header + image_bytes
            padding = bytes(self.block_size - len(data))

            return data + padding

    def _deserialize_sample(self, data: bytes) -> Tuple[np.ndarray, int, int]:
        """
        Deserialize a CIFAR-10 sample from ORAM block.

        Returns:
            Tuple of (image, label, index)
        """
        with self.profiler.track('deserialize'):
            index, label = struct.unpack('<IB', data[:5])
            image_bytes = data[5:5 + CIFAR_IMAGE_SIZE]
            image = np.frombuffer(image_bytes, dtype=np.uint8)
            image = image.reshape(32, 32, 3)

            return image, label, index

    def write(self, index: int, image: np.ndarray, label: int):
        """
        Write a sample to ORAM storage.

        Args:
            index: Sample index (0 to num_samples-1)
            image: CIFAR-10 image array (32, 32, 3)
            label: Class label (0-9)
        """
        if index < 0 or index >= self.num_samples:
            raise ValueError(f"Index {index} out of range [0, {self.num_samples})")

        block_data = self._serialize_sample(image, label, index)
        with self.profiler.track('io'):
            with self.profiler.track('oram_write'):
                self.oram.write_block(index, block_data)
                self._audit_io("write", index)

    def read(self, index: int) -> Tuple[np.ndarray, int]:
        """
        Read a sample from ORAM storage.

        Args:
            index: Sample index to read

        Returns:
            Tuple of (image, label)
        """
        if index < 0 or index >= self.num_samples:
            raise ValueError(f"Index {index} out of range [0, {self.num_samples})")

        with self.profiler.track('io'):
            with self.profiler.track('oram_read'):
                block_data = self.oram.read_block(index)
                self._audit_io("read", index)

        image, label, stored_index = self._deserialize_sample(block_data)
        assert stored_index == index, f"Index mismatch: {stored_index} != {index}"

        return image, label

    def batch_read(self, indices: list) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read multiple samples from ORAM.

        Note: Each read is still an independent ORAM access.
        Future optimization: batch ORAM operations.

        Args:
            indices: List of sample indices to read

        Returns:
            Tuple of (images array, labels array)
        """
        images = []
        labels = []

        with self.profiler.track('batch_read'):
            for idx in indices:
                image, label = self.read(idx)
                images.append(image)
                labels.append(label)

        return np.stack(images), np.array(labels)

    def close(self):
        if hasattr(self, 'oram') and self.oram is not None:
            self.oram.close()

        if self._temp_dir is not None and os.path.exists(self._temp_dir):
            import shutil
            shutil.rmtree(self._temp_dir, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def get_stats(self) -> dict:
        return {
            'num_samples': self.num_samples,
            'block_size': self.block_size,
            'backend': self.backend,
            'total_size_mb': (self.num_samples * self.block_size) / (1024 * 1024),
            'storage_path': self.storage_path,
            'tree_height': self.oram.tree_height if hasattr(self.oram, 'tree_height') else None,
        }


def load_cifar10_to_oram(
    oram_storage: ORAMStorage,
    data_dir: str = './data',
    train: bool = True,
    progress: bool = True,
    limit: Optional[int] = None
) -> int:
    """
    Load CIFAR-10 dataset into ORAM storage.

    Args:
        oram_storage: ORAMStorage instance to populate
        data_dir: Directory containing/to download CIFAR-10
        train: Load training set (True) or test set (False)
        progress: Show progress bar
        limit: Optional limit on number of samples to load (for testing)

    Returns:
        Number of samples loaded
    """
    dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=train,
        download=True
    )

    images = dataset.data
    labels = dataset.targets

    num_samples = len(labels)
    if limit is not None:
        num_samples = min(num_samples, limit)

    assert num_samples <= oram_storage.num_samples, \
        f"ORAM storage too small: {oram_storage.num_samples} < {num_samples}"

    iterator = range(num_samples)
    if progress:
        iterator = tqdm(iterator, desc="Loading CIFAR-10 to ORAM")

    for i in iterator:
        oram_storage.write(i, images[i], labels[i])

    return num_samples


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------

class ORAMDataset(Dataset):
    """
    PyTorch Dataset backed by ORAM storage.

    Fetches each sample through ORAM read operations,
    hiding access patterns from potential adversaries.
    """

    def __init__(
        self,
        oram_storage: ORAMStorage,
        num_samples: int,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        self.oram_storage = oram_storage
        self.num_samples = num_samples
        self.transform = transform
        self.target_transform = target_transform
        self.profiler = get_profiler()

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        with self.profiler.track('dataload'):
            image, label = self.oram_storage.read(index)

            if self.transform is not None:
                image = torch.from_numpy(image.copy())
                image = image.permute(2, 0, 1).float() / 255.0
                image = self.transform(image)
            else:
                image = torch.from_numpy(image.copy())
                image = image.permute(2, 0, 1).float() / 255.0

            if self.target_transform is not None:
                label = self.target_transform(label)

        return image, label


class PrefetchedDataset(Dataset):
    """
    Dataset wrapping pre-fetched (image, label) numpy arrays.

    ORAM reads happen externally (main thread); this dataset only
    applies transforms, so DataLoader workers can parallelize them.
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray,
                 transform: Optional[Callable] = None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image = torch.from_numpy(self.images[index].copy())
        image = image.permute(2, 0, 1).float() / 255.0
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.labels[index])


class ObliviousBatchSampler(Sampler):
    """
    Batch sampler that generates indices in an oblivious manner.

    For the baseline, this just shuffles indices randomly.
    """

    def __init__(
        self,
        num_samples: int,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: Optional[int] = None
    ):
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rng = np.random.RandomState(seed)
        self.profiler = get_profiler()

    def __iter__(self):
        with self.profiler.track('shuffle'):
            indices = np.arange(self.num_samples)
            if self.shuffle:
                self.rng.shuffle(indices)

        batch = []
        for idx in indices:
            batch.append(int(idx))
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return self.num_samples // self.batch_size
        return (self.num_samples + self.batch_size - 1) // self.batch_size


def get_cifar10_transforms(train: bool = True) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            ),
        ])
    else:
        return transforms.Compose([
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            ),
        ])


def create_oram_dataloader(
    oram_storage: ORAMStorage,
    num_samples: int,
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 0,
    train: bool = True,
    seed: Optional[int] = None
) -> DataLoader:
    """
    Create a DataLoader backed by ORAM storage.

    If num_workers > 0, uses the mediated architecture: pre-fetches all
    samples from ORAM in the main thread once per epoch, then hands off
    to a standard DataLoader whose workers only run transforms.

    If num_workers == 0, uses the direct ORAM dataset (original path).
    """
    transform = get_cifar10_transforms(train=train)

    if num_workers > 0:
        return _MediatedORAMLoader(
            oram_storage=oram_storage,
            num_samples=num_samples,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            train=train,
            transform=transform,
            seed=seed,
        )

    dataset = ORAMDataset(
        oram_storage=oram_storage,
        num_samples=num_samples,
        transform=transform
    )

    batch_sampler = ObliviousBatchSampler(
        num_samples=num_samples,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=train,
        seed=seed
    )

    return DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        num_workers=0,
        pin_memory=True
    )


class _MediatedORAMLoader:
    """
    Mediated ORAM DataLoader.

    Each iteration (epoch):
      1. Main thread reads ALL samples from ORAM (serial, oblivious).
      2. Builds a PrefetchedDataset holding numpy arrays in RAM.
      3. Wraps it in a standard DataLoader with num_workers for transforms.

    This separates the single-threaded ORAM constraint from the
    parallelisable CPU transform work.
    """

    def __init__(self, oram_storage, num_samples, batch_size, shuffle,
                 num_workers, train, transform, seed):
        self.oram_storage = oram_storage
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.train = train
        self.transform = transform
        self.rng = np.random.RandomState(seed)
        self.profiler = get_profiler()

    def __len__(self) -> int:
        if self.train:
            return self.num_samples // self.batch_size
        return (self.num_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        with self.profiler.track('shuffle'):
            indices = np.arange(self.num_samples)
            if self.shuffle:
                self.rng.shuffle(indices)

        images = np.empty((self.num_samples, 32, 32, 3), dtype=np.uint8)
        labels = np.empty(self.num_samples, dtype=np.int64)

        with self.profiler.track('dataload'):
            for pos, idx in enumerate(indices):
                img, lbl = self.oram_storage.read(int(idx))
                images[pos] = img
                labels[pos] = lbl

        dataset = PrefetchedDataset(images, labels, self.transform)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=self.train,
        )
        yield from loader


def create_standard_dataloader(
    data_dir: str = './data',
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 4,
    train: bool = True
) -> Tuple[DataLoader, int]:
    """
    Create a standard (non-ORAM) CIFAR-10 DataLoader for comparison.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(32, padding=4) if train else transforms.Lambda(lambda x: x),
        transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        ),
    ])

    dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=train,
        download=True,
        transform=transform
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train
    )

    return dataloader, len(dataset)


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

class Train:
    """
    Unified CIFAR-10 trainer supporting both baseline and ORAM modes.

    When baseline=True, uses standard PyTorch DataLoader (no ORAM overhead).
    When baseline=False (default), uses ORAM-backed data loading with
    comprehensive profiling of overhead sources.
    """

    def __init__(
        self,
        baseline: bool = False,
        data_dir: str = './data',
        oram_storage_path: Optional[str] = None,
        batch_size: int = 128,
        learning_rate: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        device: Optional[str] = None,
        num_samples: Optional[int] = None,
        backend: str = "file",
        block_size: int = DEFAULT_BLOCK_SIZE,
        model_name: str = "resnet18",
        num_workers: int = 0,
    ):
        self.baseline = baseline
        self.data_dir = data_dir
        self.oram_storage_path = oram_storage_path
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.backend = backend
        self.block_size = block_size
        self.model_name = model_name
        self.num_workers = num_workers if not baseline else max(num_workers, 4)

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        self.profiler = get_profiler()

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None

        self.oram_storage = None
        self.train_loader = None
        self.test_loader = None
        self.num_train_samples = num_samples if num_samples is not None else 50000
        self.num_test_samples = 10000

    def setup(self):
        mode = "baseline" if self.baseline else "ORAM"
        print(f"Setting up {mode} training...")

        with self.profiler.track('setup'):
            if self.baseline:
                self._setup_baseline_loader()
            else:
                self._setup_oram_loader()

            self._setup_test_loader()
            self._setup_model()

        print(f"Setup complete. Device: {self.device}")
        if not self.baseline:
            print(f"ORAM storage stats: {self.oram_storage.get_stats()}")

    def _setup_baseline_loader(self):
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            ),
        ])

        train_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=train_transform
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

        print(f"Training samples: {len(train_dataset)}")

    def _setup_oram_loader(self):
        print(f"Initializing ORAM storage for {self.num_train_samples} samples "
              f"(backend={self.backend}, block_size={self.block_size})...")
        self.oram_storage = ORAMStorage(
            num_samples=self.num_train_samples,
            storage_path=self.oram_storage_path,
            backend=self.backend,
            block_size=self.block_size,
        )

        limit = self.num_train_samples if self.num_train_samples < 50000 else None
        print(f"Loading CIFAR-10 training data into ORAM ({self.num_train_samples} samples)...")
        load_cifar10_to_oram(
            self.oram_storage,
            data_dir=self.data_dir,
            train=True,
            progress=True,
            limit=limit
        )

        self.train_loader = create_oram_dataloader(
            oram_storage=self.oram_storage,
            num_samples=self.num_train_samples,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            train=True
        )

    def _setup_test_loader(self):
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            ),
        ])

        test_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=True,
            transform=test_transform
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        print(f"Test samples: {len(test_dataset)}")

    def _setup_model(self):
        self.model = create_model(self.model_name).to(self.device)

        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )

        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[50, 75, 90],
            gamma=0.1
        )

        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        self.profiler.start_epoch(epoch)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            batch_start = time.perf_counter()

            with self.profiler.track('dataload'):
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

            with self.profiler.track('compute'):
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            with self.profiler.track('compute'):
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            batch_time = time.perf_counter() - batch_start
            self.profiler.record_time('batch', batch_time)
            self.profiler.record_batch(batch_idx, {
                'loss': loss.item(),
                'batch_time': batch_time
            })

            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })

        self.scheduler.step()

        metrics = {
            'train_loss': running_loss / len(self.train_loader),
            'train_acc': 100. * correct / total,
            'lr': self.scheduler.get_last_lr()[0]
        }

        self.profiler.end_epoch(epoch, metrics)
        self.profiler.record_memory()

        return metrics

    def evaluate(self) -> Dict[str, float]:
        self.model.eval()

        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in tqdm(self.test_loader, desc="Evaluating"):
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return {
            'test_loss': test_loss / len(self.test_loader),
            'test_acc': 100. * correct / total
        }

    def train(
        self,
        num_epochs: int = 100,
        eval_every: int = 10,
        save_dir: str = 'results'
    ) -> Dict:
        """
        Run full training loop.

        Args:
            num_epochs: Number of epochs to train
            eval_every: Evaluate every N epochs
            save_dir: Directory to save results

        Returns:
            Training history dictionary
        """
        os.makedirs(save_dir, exist_ok=True)

        history = {
            'epochs': [],
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': [],
            'lr': [],
        }
        if not self.baseline and self.oram_storage is not None:
            history['oram_config'] = self.oram_storage.get_stats()

        best_acc = 0.0
        total_start = time.time()

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()

            train_metrics = self.train_epoch(epoch)

            if epoch % eval_every == 0 or epoch == num_epochs:
                test_metrics = self.evaluate()
            else:
                test_metrics = {'test_loss': 0, 'test_acc': 0}

            epoch_time = time.time() - epoch_start
            self.profiler.record_time('epoch', epoch_time)

            history['epochs'].append(epoch)
            history['train_loss'].append(train_metrics['train_loss'])
            history['train_acc'].append(train_metrics['train_acc'])
            history['test_loss'].append(test_metrics['test_loss'])
            history['test_acc'].append(test_metrics['test_acc'])
            history['lr'].append(train_metrics['lr'])

            print(f"Epoch {epoch}: train_loss={train_metrics['train_loss']:.4f}, "
                  f"train_acc={train_metrics['train_acc']:.2f}%, "
                  f"test_acc={test_metrics['test_acc']:.2f}%, "
                  f"time={epoch_time:.2f}s")

            if test_metrics['test_acc'] > best_acc:
                best_acc = test_metrics['test_acc']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_acc': best_acc,
                }, os.path.join(save_dir, 'best_model.pth'))

        total_time = time.time() - total_start

        history['total_time'] = total_time
        history['best_acc'] = best_acc

        with open(os.path.join(save_dir, 'history.json'), 'w') as f:
            json.dump(history, f, indent=2)

        mode = "Baseline" if self.baseline else "ORAM"
        print(f"\n{mode} training complete. Total time: {total_time:.2f}s")
        print(f"Best test accuracy: {best_acc:.2f}%")

        return history

    def cleanup(self):
        if self.oram_storage is not None:
            self.oram_storage.close()


ORAMTrainer = Train


def run_baseline_training(
    num_epochs: int = 100,
    batch_size: int = 128,
    output_dir: str = 'results/baseline',
    device: Optional[str] = None,
    model_name: str = "resnet18",
) -> Dict:
    """
    Convenience function to run baseline (non-ORAM) training.

    Returns:
        Training history
    """
    with ProfilerContext('baseline', output_dir=output_dir):
        trainer = Train(
            baseline=True,
            batch_size=batch_size,
            device=device,
            model_name=model_name,
        )
        trainer.setup()
        history = trainer.train(
            num_epochs=num_epochs,
            save_dir=output_dir
        )

    return history


def run_oram_training(
    num_epochs: int = 100,
    batch_size: int = 128,
    output_dir: str = 'results/oram',
    device: Optional[str] = None,
    num_samples: Optional[int] = None,
    backend: str = "file",
    block_size: int = DEFAULT_BLOCK_SIZE,
    model_name: str = "resnet18",
    num_workers: int = 0,
) -> Dict:
    """
    Convenience function to run ORAM-integrated training.

    Returns:
        Training history
    """
    with ProfilerContext('oram', output_dir=output_dir):
        trainer = Train(
            baseline=False,
            batch_size=batch_size,
            device=device,
            num_samples=num_samples,
            backend=backend,
            block_size=block_size,
            model_name=model_name,
            num_workers=num_workers,
        )
        try:
            trainer.setup()
            history = trainer.train(
                num_epochs=num_epochs,
                save_dir=output_dir
            )
        finally:
            trainer.cleanup()

    return history

# Backward-compatible aliases for run.py imports
baseline = Train.baseline
oram = Train.oram
ORAMDataset = Datasets.ORAM
IndexedDataset = Datasets.Indexed
resolve_torch_device = Train.resolve_torch_device
SUPPORTED_MODELS = Train.Models.SUPPORTED_MODELS
read_oram_audit_counts = Record.read_oram_audit_counts
SidecarLogger = Record.Sidecar
sidecar_training = Record.sidecar_training
plaintext = Record.plaintext
oram_event = Record.oram
