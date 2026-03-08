"""
Profiling infrastructure for ORAM overhead measurement.

Tracks time breakdown across:
- I/O operations (ORAM block reads/writes)
- Cryptographic operations (encryption/decryption)
- Shuffling operations (Path ORAM eviction)
- Memory usage
- Training compute time
"""

import time
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import defaultdict
import json
import os

import psutil


@dataclass
class TimingStats:
    """Statistics for a single timing category."""
    total_time: float = 0.0
    call_count: int = 0
    min_time: float = float('inf')
    max_time: float = 0.0
    samples: List[float] = field(default_factory=list)
    
    def record(self, duration: float, keep_samples: bool = False):
        """Record a timing measurement."""
        self.total_time += duration
        self.call_count += 1
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
        if keep_samples:
            self.samples.append(duration)
    
    @property
    def avg_time(self) -> float:
        """Average time per call."""
        return self.total_time / self.call_count if self.call_count > 0 else 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'total_time': self.total_time,
            'call_count': self.call_count,
            'avg_time': self.avg_time,
            'min_time': self.min_time if self.min_time != float('inf') else 0.0,
            'max_time': self.max_time,
        }


@dataclass 
class MemoryStats:
    """Memory usage statistics."""
    peak_rss: int = 0  # Peak resident set size in bytes
    peak_vms: int = 0  # Peak virtual memory size in bytes
    samples: List[Dict[str, int]] = field(default_factory=list)
    
    def record(self):
        """Record current memory usage."""
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
        """Convert to dictionary for serialization."""
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
    
    # Singleton instance for global access
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
        """Get or create singleton profiler instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset(cls):
        """Reset the singleton instance."""
        with cls._lock:
            cls._instance = None
    
    def enable(self):
        """Enable profiling."""
        self._enabled = True
        
    def disable(self):
        """Disable profiling."""
        self._enabled = False
        
    def set_keep_samples(self, keep: bool):
        """Set whether to keep individual timing samples."""
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
        """Directly record a timing measurement."""
        if self._enabled:
            self.timings[category].record(duration, self._keep_samples)
    
    def record_memory(self):
        """Record current memory usage."""
        if self._enabled:
            self.memory.record()
    
    def start_epoch(self, epoch: int):
        """Mark the start of a new epoch."""
        self.current_epoch = epoch
        self.current_batch = 0
        
    def end_epoch(self, epoch: int, metrics: Optional[Dict] = None):
        """Mark the end of an epoch and record metrics."""
        epoch_record = {
            'epoch': epoch,
            'timings': {k: v.to_dict() for k, v in self.timings.items()},
            'memory': self.memory.to_dict(),
        }
        if metrics:
            epoch_record['metrics'] = metrics
        self.epoch_data.append(epoch_record)
    
    def record_batch(self, batch_idx: int, metrics: Optional[Dict] = None):
        """Record per-batch metrics."""
        self.current_batch = batch_idx
        if metrics:
            self.batch_data.append({
                'epoch': self.current_epoch,
                'batch': batch_idx,
                **metrics
            })
    
    def get_summary(self) -> Dict:
        """Get a summary of all profiling data."""
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
        """Save profiling data to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        data = {
            'summary': self.get_summary(),
            'overhead_breakdown': self.get_overhead_breakdown(),
            'epoch_data': self.epoch_data,
            'batch_data': self.batch_data[-1000:],  # Keep last 1000 batches
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def print_summary(self):
        """Print a formatted summary to console."""
        print("\n" + "="*60)
        print("PROFILER SUMMARY")
        print("="*60)
        
        print("\nTiming Breakdown:")
        print("-"*40)
        breakdown = self.get_overhead_breakdown()
        for category, pct in sorted(breakdown.items(), key=lambda x: -x[1]):
            stats = self.timings[category]
            print(f"  {category:15} {pct:6.2f}%  "
                  f"(total: {stats.total_time:8.2f}s, "
                  f"calls: {stats.call_count:6d}, "
                  f"avg: {stats.avg_time*1000:8.3f}ms)")
        
        print("\nMemory Usage:")
        print("-"*40)
        print(f"  Peak RSS: {self.memory.peak_rss / (1024*1024):.2f} MB")
        print(f"  Peak VMS: {self.memory.peak_vms / (1024*1024):.2f} MB")
        
        print("="*60 + "\n")


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


# Convenience functions for global profiler access
def get_profiler() -> Profiler:
    """Get the global profiler instance."""
    return Profiler.get_instance()


def track(category: str):
    """Decorator/context manager for tracking time."""
    return get_profiler().track(category)
