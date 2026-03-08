"""
ORAM-backed PyTorch DataLoader for CIFAR-10.

Provides a custom Dataset that fetches samples through ORAM storage,
enabling oblivious data access during ML training.
"""

import numpy as np
from typing import Optional, Tuple, Callable, List
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
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
        """
        Initialize ORAM-backed dataset.
        
        Args:
            oram_storage: ORAMStorage instance containing data
            num_samples: Number of samples in the dataset
            transform: Optional transform for images
            target_transform: Optional transform for labels
        """
        self.oram_storage = oram_storage
        self.num_samples = num_samples
        self.transform = transform
        self.target_transform = target_transform
        self.profiler = get_profiler()
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Fetch a sample through ORAM.
        
        This is the key integration point where ORAM overhead
        is incurred for each sample access.
        """
        with self.profiler.track('dataload'):
            # Read from ORAM (oblivious access)
            image, label = self.oram_storage.read(index)
            
            # Convert to PIL-like format for transforms
            # CIFAR-10 images are (32, 32, 3) uint8
            
            # Apply transforms
            if self.transform is not None:
                # Convert numpy to tensor first if needed
                image = torch.from_numpy(image.copy())
                # Permute to (C, H, W) for torchvision
                image = image.permute(2, 0, 1).float() / 255.0
                image = self.transform(image)
            else:
                image = torch.from_numpy(image.copy())
                image = image.permute(2, 0, 1).float() / 255.0
            
            if self.target_transform is not None:
                label = self.target_transform(label)
        
        return image, label


class ObliviousBatchSampler(Sampler):
    """
    Batch sampler that generates indices in an oblivious manner.
    
    For the baseline, this just shuffles indices randomly.
    Future work: integrate with oblivious shuffling protocols.
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
        
        # Yield batches
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
    """
    Get standard CIFAR-10 transforms.
    
    Note: Since we're loading raw numpy arrays from ORAM,
    we skip ToTensor() (done in __getitem__) and apply
    normalization and augmentation here.
    """
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
    num_workers: int = 0,  # Must be 0 for ORAM (single connection)
    train: bool = True,
    seed: Optional[int] = None
) -> DataLoader:
    """
    Create a DataLoader backed by ORAM storage.
    
    Args:
        oram_storage: ORAM storage containing CIFAR-10 data
        num_samples: Number of samples in storage
        batch_size: Batch size for training
        shuffle: Whether to shuffle each epoch
        num_workers: Number of worker processes (must be 0 for ORAM)
        train: Whether this is for training (affects transforms)
        seed: Random seed for shuffling
        
    Returns:
        PyTorch DataLoader with ORAM-backed dataset
    """
    if num_workers > 0:
        print("Warning: num_workers > 0 not supported with ORAM storage. Using 0.")
        num_workers = 0
    
    transform = get_cifar10_transforms(train=train)
    
    dataset = ORAMDataset(
        oram_storage=oram_storage,
        num_samples=num_samples,
        transform=transform
    )
    
    # Use custom batch sampler for oblivious shuffling
    batch_sampler = ObliviousBatchSampler(
        num_samples=num_samples,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=train,  # Drop last incomplete batch during training
        seed=seed
    )
    
    return DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=True
    )


def create_standard_dataloader(
    data_dir: str = './data',
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 4,
    train: bool = True
) -> Tuple[DataLoader, int]:
    """
    Create a standard (non-ORAM) CIFAR-10 DataLoader for comparison.
    
    Returns:
        Tuple of (DataLoader, num_samples)
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
