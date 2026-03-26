"""
Profiler for tracking ORAM overhead decomposition.

Tracks timing categories (I/O, crypto, shuffle, compute, dataload, batch, epoch),
memory usage, and provides context manager for scoped profiling sessions.
"""

import json
import os
import time
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import defaultdict


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

    @dataclass
    class Timing:
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
            d = {
                'total_time': self.total_time,
                'call_count': self.call_count,
                'avg_time': self.avg_time,
                'min_time': self.min_time if self.min_time != float('inf') else 0.0,
                'max_time': self.max_time,
            }
            if self.samples:
                d['samples'] = self.samples
            return d

    @dataclass
    class Memory:
        peak_rss: int = 0
        peak_vms: int = 0
        samples: List[Dict[str, int]] = field(default_factory=list)

        def record(self):
            try:
                import psutil
            except ImportError:
                return
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

    _instance: Optional['Profiler'] = None
    _lock = threading.Lock()

    def __init__(self):
        self.timings: Dict[str, Profiler.Timing] = defaultdict(Profiler.Timing)
        self.memory = Profiler.Memory()
        self.epoch_data: List[Dict] = []
        self.batch_data: List[Dict] = []
        self.current_epoch: int = 0
        self.current_batch: int = 0
        self._enabled = True
        self._keep_samples = False

    @classmethod
    def instance(cls) -> 'Profiler':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def profiler(cls) -> 'Profiler':
        return cls.instance()

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

    @classmethod
    @contextmanager
    def track(cls, category: str):
        """Context manager for tracking time in a category (singleton-backed)."""
        prof = cls.instance()
        if not prof._enabled:
            yield
            return

        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            with cls._lock:
                prof.timings[category].record(duration, prof._keep_samples)

    def record_time(self, category: str, duration: float):
        if self._enabled:
            with Profiler._lock:
                self.timings[category].record(duration, self._keep_samples)

    def record_memory(self):
        if self._enabled:
            self.memory.record()

    def start_epoch(self, epoch: int):
        self.current_epoch = epoch
        self.current_batch = 0

    def end_epoch(self, epoch: int, metrics: Optional[Dict] = None):
        # NOTE: timings are cumulative across all epochs, not per-epoch deltas.
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

    def summary(self) -> Dict:
        return {
            'timings': {k: v.to_dict() for k, v in self.timings.items()},
            'memory': self.memory.to_dict(),
            'epochs': len(self.epoch_data),
            'total_batches': len(self.batch_data),
        }

    def overhead_breakdown(self) -> Dict[str, float]:
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
        dirname = os.path.dirname(filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        data = {
            'summary': self.summary(),
            'overhead_breakdown': self.overhead_breakdown(),
            'epoch_data': self.epoch_data,
            'batch_data': self.batch_data[-1000:],
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def print_summary(self):
        print("\n" + "=" * 60)
        print("PROFILER SUMMARY")
        print("=" * 60)

        print("\nTiming Breakdown:")
        print("-" * 40)
        breakdown = self.overhead_breakdown()
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

    class Context:
        """
        Scoped profiling context that creates a fresh Profiler instance and
        registers it as the active singleton for the duration of the ``with`` block.

        Usage:
            with Profiler.Context('experiment_name') as profiler:
                # Run experiment (Train uses Profiler.profiler() → same instance)
                ...
            # Profile JSON written and summary printed on exit
        """

        def __init__(self, name: str, output_dir: str = 'results',
                     keep_samples: bool = False):
            self.name = name
            self.output_dir = output_dir
            self.keep_samples = keep_samples
            self._profiler_instance = None

        def __enter__(self) -> 'Profiler':
            with Profiler._lock:
                Profiler._instance = None
                self._profiler_instance = Profiler()
                Profiler._instance = self._profiler_instance
            self._profiler_instance.set_keep_samples(self.keep_samples)
            self._profiler_instance.record_memory()
            return self._profiler_instance

        def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
            if self._profiler_instance:
                self._profiler_instance.record_memory()
                filepath = os.path.join(self.output_dir, f'{self.name}_profile.json')
                self._profiler_instance.save(filepath)
                self._profiler_instance.print_summary()
            return False


profiler = Profiler.profiler
track = Profiler.track
ProfilerContext = Profiler.Context
Timing = Profiler.Timing
Memory = Profiler.Memory
