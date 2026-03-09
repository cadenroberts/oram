"""
ORAM-backed PyTorch DataLoader for CIFAR-10.

Provides two loading modes:
  1. Direct ORAM DataLoader (num_workers=0), baseline
  2. Mediated loader, ORAM reads in main thread, transforms in workers
"""

import numpy as np
from typing import Optional, Tuple, Callable, List
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, IterableDataset
import torchvision.transforms as transforms

from .oram_storage import ORAMStorage
from .profiler import get_profiler


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
    import torchvision
    
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
