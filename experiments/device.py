from __future__ import annotations

import csv
import time
from typing import Optional

import torch


def resolve_torch_device(device_name: str) -> torch.device:
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but not available.")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(device_name)


class SidecarLogger:
    FIELDNAMES = ["timestamp", "batch_id", "epoch", "phase"]

    def __init__(self, path: str):
        self.path = path
        self._file = None
        self._writer = None

    def __enter__(self) -> "SidecarLogger":
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
        self._writer.writeheader()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None

    def log(self, batch_id: str, epoch: int, phase: str, timestamp: Optional[float] = None) -> None:
        if self._writer is None or self._file is None:
            raise RuntimeError("SidecarLogger must be used as a context manager.")
        self._writer.writerow(
            {
                "timestamp": time.time() if timestamp is None else timestamp,
                "batch_id": batch_id,
                "epoch": epoch,
                "phase": phase,
            }
        )
        self._file.flush()
