"""
ORAM infrastructure: storage, data loading, and training.

Storage wraps PyORAM's Path ORAM to store CIFAR-10 samples as encrypted blocks.
DataLoader provides ORAM-backed and mediated loading modes for CIFAR-10.
Train runs full ORAM-integrated training with ResNet on CIFAR-10.

Also includes sidecar batch logging, ORAM audit parsing, device resolution,
and synthetic plaintext/ORAM access-pattern event generators for experiments.
"""

import csv
import itertools
import os
import time
import struct
import tempfile
import json
import warnings
from typing import Dict, List, Optional, Tuple, Callable

warnings.filterwarnings("ignore", message="dtype.*align", module="torchvision")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm import tqdm
from pyoram.oblivious_storage.tree.path_oram import PathORAM

from profiler import Profiler, ProfilerContext



# ---------------------------------------------------------------------------
# Storage constants
# ---------------------------------------------------------------------------

CIFAR_IMAGE_SIZE = 32 * 32 * 3  # 3072 bytes
LABEL_SIZE = 1
METADATA_SIZE = 4
MIN_BLOCK_SIZE = METADATA_SIZE + LABEL_SIZE + CIFAR_IMAGE_SIZE  # 3077 bytes

DEFAULT_BLOCK_SIZE = 4096

VALID_BACKENDS = ("file", "ram")


SUPPORTED_MODELS = ["resnet18", "resnet50", "efficientnet_b0"]


def get_profiler() -> Profiler:
    return Profiler.instance()


def resolve_torch_device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_model(model_name: str) -> nn.Module:
    if model_name == "resnet18":
        model = torchvision.models.resnet18(num_classes=10)
    elif model_name == "resnet50":
        model = torchvision.models.resnet50(num_classes=10)
    elif model_name == "efficientnet_b0":
        model = torchvision.models.efficientnet_b0(num_classes=10)
    else:
        raise ValueError(f"Unsupported model: {model_name}. Choose from {SUPPORTED_MODELS}")
    return model


class IndexedDataset(Dataset):
    """Wraps a dataset to also return the index alongside (image, label)."""

    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        image, label = self.base_dataset[idx]
        return image, label, idx


class SidecarLogger:
    """CSV logger for batch-level sidecar timestamps."""

    def __init__(self, path: str):
        self.path = path
        self._file = None
        self._writer = None

    def __enter__(self):
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=["timestamp", "batch_id", "epoch", "phase"],
        )
        self._writer.writeheader()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file is not None:
            self._file.close()
        return False

    def _ensure_open(self) -> None:
        if self._writer is None:
            raise RuntimeError("SidecarLogger must be used as a context manager")

    def log(self, batch_id: str, epoch: int, phase: str) -> None:
        self._ensure_open()
        self._writer.writerow({
            "timestamp": f"{time.time():.6f}",
            "batch_id": batch_id,
            "epoch": epoch,
            "phase": phase,
        })

    def log_at(self, timestamp: float, batch_id: str, epoch: int, phase: str) -> None:
        self._ensure_open()
        self._writer.writerow({
            "timestamp": f"{timestamp:.6f}",
            "batch_id": batch_id,
            "epoch": epoch,
            "phase": phase,
        })


def read_oram_audit_counts(audit_log_path: str) -> Dict[str, int]:
    """Parse ORAM audit log and return read/write counts."""
    counts: Dict[str, int] = {"read": 0, "write": 0}
    if not os.path.exists(audit_log_path):
        return counts
    with open(audit_log_path, "r", encoding="utf-8") as f:
        for line in f:
            if "op=read" in line:
                counts["read"] += 1
            elif "op=write" in line:
                counts["write"] += 1
    return counts


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

        self._audit_file = None

        try:
            self._init_oram()
        except Exception:
            if self._temp_dir is not None and os.path.exists(self._temp_dir):
                import shutil
                shutil.rmtree(self._temp_dir, ignore_errors=True)
            raise

    def _init_oram(self):
        with self.profiler.track('oram_init'):
            self.oram = PathORAM.setup(
                storage_name=self.storage_path,
                block_size=self.block_size,
                block_count=self.num_samples,
                storage_type=self.backend,
            )

    def _audit_io(self, op: str, index: int) -> None:
        """Append ORAM I/O evidence when ORAM_AUDIT_LOG is set."""
        log_path = os.environ.get("ORAM_AUDIT_LOG")
        if not log_path:
            return
        if self._audit_file is None:
            self._audit_file = open(log_path, "a", encoding="utf-8")
        include_index = os.environ.get("ORAM_AUDIT_INCLUDE_INDEX", "0") == "1"
        fields = [
            f"{time.time():.6f}",
            f"op={op}",
            f"backend={self.backend}",
            f"storage_path={self.storage_path}",
        ]
        if include_index:
            fields.append(f"index={index}")
        self._audit_file.write(",".join(fields) + "\n")
        self._audit_file.flush()

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
        if stored_index != index:
            raise RuntimeError(f"Index mismatch: {stored_index} != {index}")

        return image, label

    def close(self):
        if self._audit_file is not None:
            self._audit_file.flush()
            self._audit_file.close()
            self._audit_file = None

        if hasattr(self, 'oram') and self.oram is not None:
            self.oram.close()
            self.oram = None

        if self._temp_dir is not None and os.path.exists(self._temp_dir):
            import shutil
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None

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

    if num_samples > oram_storage.num_samples:
        raise ValueError(f"ORAM storage too small: {oram_storage.num_samples} < {num_samples}")

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
        yield from DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=self.train,
        )


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
        seed: Optional[int] = None,
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
        self.num_workers = max(num_workers, 4) if baseline else num_workers
        self.seed = seed

        self.device = resolve_torch_device(device)

        self.profiler = get_profiler()

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None

        self.oram_storage = None
        self.train_loader = None
        self.test_loader = None
        self.num_train_samples = num_samples if num_samples is not None else 50000

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

        limit = self.num_train_samples
        print(f"Loading CIFAR-10 training data into ORAM ({self.num_train_samples} samples)...")
        actually_loaded = load_cifar10_to_oram(
            self.oram_storage,
            data_dir=self.data_dir,
            train=True,
            progress=True,
            limit=limit
        )
        self.num_train_samples = actually_loaded

        self.train_loader = create_oram_dataloader(
            oram_storage=self.oram_storage,
            num_samples=self.num_train_samples,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            train=True,
            seed=self.seed
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

            with self.profiler.track('transfer'):
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

        num_batches = max(len(self.train_loader), 1)
        metrics = {
            'train_loss': running_loss / num_batches,
            'train_acc': 100. * correct / max(total, 1),
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
            'test_loss': test_loss / max(len(self.test_loader), 1),
            'test_acc': 100. * correct / max(total, 1)
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
                test_metrics = {'test_loss': None, 'test_acc': None}

            epoch_time = time.time() - epoch_start
            self.profiler.record_time('epoch', epoch_time)

            history['epochs'].append(epoch)
            history['train_loss'].append(train_metrics['train_loss'])
            history['train_acc'].append(train_metrics['train_acc'])
            history['test_loss'].append(test_metrics['test_loss'])
            history['test_acc'].append(test_metrics['test_acc'])
            history['lr'].append(train_metrics['lr'])

            test_acc_str = f"{test_metrics['test_acc']:.2f}%" if test_metrics['test_acc'] is not None else "N/A"
            print(f"Epoch {epoch}: train_loss={train_metrics['train_loss']:.4f}, "
                  f"train_acc={train_metrics['train_acc']:.2f}%, "
                  f"test_acc={test_acc_str}, "
                  f"time={epoch_time:.2f}s")

            if test_metrics['test_acc'] is not None and test_metrics['test_acc'] > best_acc:
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

        with open(os.path.join(save_dir, 'history.json'), 'w', encoding='utf-8') as f:
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
        try:
            trainer.setup()
            history = trainer.train(
                num_epochs=num_epochs,
                save_dir=output_dir
            )
        finally:
            trainer.cleanup()

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

def _write_synthetic_sidecar(
    sidecar_path: str, epochs: int, num_batches: int, history: Dict
) -> None:
    total_time = history.get("total_time", num_batches * epochs * 0.1)
    batch_duration = total_time / max(num_batches * epochs, 1)
    training_start = time.time() - total_time

    with SidecarLogger(sidecar_path) as sc:
        for epoch, batch_idx in itertools.product(range(epochs), range(num_batches)):
            global_batch = epoch * num_batches + batch_idx
            synthetic_ts = training_start + global_batch * batch_duration
            sc.log_at(
                timestamp=synthetic_ts,
                batch_id=f"{epoch}_{batch_idx}_train",
                epoch=epoch,
                phase="train",
            )


def sidecar_training(args) -> Dict:
    """Run ORAM training with sidecar batch logging."""
    import random as _random
    _random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.environ["ORAM_AUDIT_LOG"] = os.path.join(args.output_dir, "oram_audit.csv")
    os.makedirs(args.output_dir, exist_ok=True)

    history = run_oram_training(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        device=args.device,
        num_samples=args.num_samples,
        backend=args.backend,
        block_size=args.block_size,
        model_name=args.model,
        num_workers=args.num_workers,
    )

    if sidecar_path := getattr(args, "sidecar_path", None):
        num_samples = args.num_samples or 50000
        num_batches = num_samples // args.batch_size
        _write_synthetic_sidecar(sidecar_path, args.epochs, num_batches, history)

    return history


def plaintext(
    train_size: int = 20000,
    holdout_size: int = 20000,
    epochs: int = 3,
    batch_size: int = 128,
    probe_batch_prob: float = 0.2,
    probe_mix_ratio: float = 0.3,
    data_dir: str = "./data",
    random_state: int = 42,
) -> List[Tuple[str, float, int, str, int]]:
    """Generate synthetic plaintext access-pattern event log."""
    import random as _random
    _random.seed(random_state)
    np.random.seed(random_state)

    ds = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True)
    n = len(ds)
    all_idx = list(range(n))
    _random.shuffle(all_idx)

    train_idx = all_idx[:train_size]
    holdout_idx = all_idx[train_size:train_size + holdout_size]

    events: List[Tuple[str, float, int, str, int]] = []
    global_time = 0.0

    for epoch in range(epochs):
        _random.shuffle(train_idx)
        for batch_start in range(0, len(train_idx), batch_size):
            batch = train_idx[batch_start:batch_start + batch_size]
            batch_id = f"{epoch}_{batch_start // batch_size}_train"
            for sid in batch:
                global_time += _random.uniform(0.001, 0.010)
                events.append((str(sid), global_time, epoch, batch_id, 1))

            if _random.random() < probe_batch_prob:
                probe_n = max(1, int(batch_size * probe_mix_ratio))
                probe_samples = _random.sample(holdout_idx, min(probe_n, len(holdout_idx)))
                probe_batch_id = f"{epoch}_{batch_start // batch_size}_probe"
                for sid in probe_samples:
                    global_time += _random.uniform(0.001, 0.010)
                    events.append((str(sid), global_time, epoch, probe_batch_id, 0))

    return events


def oram_event(
    train_size: int = 20000,
    holdout_size: int = 20000,
    epochs: int = 3,
    batch_size: int = 128,
    probe_batch_prob: float = 0.2,
    probe_mix_ratio: float = 0.3,
    data_dir: str = "./data",
    random_state: int = 42,
) -> List[Tuple[str, float, int, str, int]]:
    """Generate synthetic ORAM access-pattern event log.

    ORAM randomizes physical accesses, so the event log uses opaque
    block identifiers instead of sample IDs.
    """
    import random as _random
    _random.seed(random_state)
    np.random.seed(random_state)

    ds = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True)
    n = len(ds)
    all_idx = list(range(n))
    _random.shuffle(all_idx)

    train_idx = all_idx[:train_size]
    holdout_idx = all_idx[train_size:train_size + holdout_size]

    membership = {i: 1 for i in train_idx} | {i: 0 for i in holdout_idx}

    events: List[Tuple[str, float, int, str, int]] = []
    global_time = 0.0
    block_counter = 0

    for epoch in range(epochs):
        _random.shuffle(train_idx)
        for batch_start in range(0, len(train_idx), batch_size):
            batch = train_idx[batch_start:batch_start + batch_size]
            batch_id = f"{epoch}_{batch_start // batch_size}_train"
            for sid in batch:
                global_time += _random.uniform(0.001, 0.010)
                block_counter += 1
                oram_id = f"block_{block_counter}"
                events.append((oram_id, global_time, epoch, batch_id, membership.get(sid, 1)))

            if _random.random() < probe_batch_prob:
                probe_n = max(1, int(batch_size * probe_mix_ratio))
                probe_samples = _random.sample(holdout_idx, min(probe_n, len(holdout_idx)))
                probe_batch_id = f"{epoch}_{batch_start // batch_size}_probe"
                for sid in probe_samples:
                    global_time += _random.uniform(0.001, 0.010)
                    block_counter += 1
                    oram_id = f"block_{block_counter}"
                    events.append((oram_id, global_time, epoch, probe_batch_id, membership.get(sid, 0)))

    return events


# Backward-compatible aliases for run.py / pipeline.py imports
baseline = run_baseline_training
oram = run_oram_training
