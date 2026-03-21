"""
ORAM Storage Layer for CIFAR-10 Dataset.

Wraps PyORAM's Path ORAM implementation to store CIFAR-10 samples
as encrypted blocks, enabling oblivious data access for ML training.
"""

import os
import tempfile
import struct
import time
from typing import Tuple, Optional
import numpy as np

from pyoram.oblivious_storage.tree.path_oram import PathORAM

from .oram.profiler import get_profiler


CIFAR_IMAGE_SIZE = 32 * 32 * 3  # 3072 bytes
LABEL_SIZE = 1
METADATA_SIZE = 4
MIN_BLOCK_SIZE = METADATA_SIZE + LABEL_SIZE + CIFAR_IMAGE_SIZE  # 3077 bytes

DEFAULT_BLOCK_SIZE = 4096

VALID_BACKENDS = ("file", "ram")


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
    import torchvision
    from tqdm import tqdm
    
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
