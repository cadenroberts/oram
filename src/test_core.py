"""Minimal test suite for OMLO core functionality."""

import ast
import csv
import io
import json
import os
import sys
import tempfile

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_REPO_ROOT, "data")

import numpy as np
import torch

from profiler import Profiler, ProfilerContext
from oram import (
    ORAMStorage,
    ORAMDataset,
    IndexedDataset,
    SidecarLogger,
    resolve_torch_device,
    create_model,
    MIN_BLOCK_SIZE,
    DEFAULT_BLOCK_SIZE,
    SUPPORTED_MODELS,
    plaintext,
    oram_event,
)
from figures import Save
from attack import (
    safe_stats,
    coefficient_of_variation,
    attack_burstiness,
    maybe_reorder,
    events as parse_events,
    subsample_visibility,
    Build,
    PartialObservabilityConfig,
)
from pipeline import RunConfig, PHASES


def test_profiler():
    Profiler.reset()
    p = Profiler.instance()
    assert p is Profiler.instance()

    with Profiler.track("test_cat"):
        _ = sum(range(100))
    assert "test_cat" in p.timings
    assert p.timings["test_cat"].call_count == 1
    assert p.timings["test_cat"].total_time > 0

    summary = p.summary()
    assert "timings" in summary
    assert "memory" in summary

    breakdown = p.overhead_breakdown()
    assert "test_cat" in breakdown
    assert abs(sum(breakdown.values()) - 100.0) < 0.01

    Profiler.reset()
    print("[PASS] test_profiler")


def test_profiler_context():
    with tempfile.TemporaryDirectory() as tmpdir:
        with ProfilerContext("ctx_test", output_dir=tmpdir) as p:
            with Profiler.track("inner"):
                _ = list(range(50))
        profile_path = os.path.join(tmpdir, "ctx_test_profile.json")
        assert os.path.exists(profile_path)
        with open(profile_path) as f:
            data = json.load(f)
        assert "summary" in data
    Profiler.reset()
    print("[PASS] test_profiler_context")


def test_resolve_device():
    device = resolve_torch_device("cpu")
    assert device == torch.device("cpu")
    device = resolve_torch_device(None)
    assert isinstance(device, torch.device)
    print("[PASS] test_resolve_device")


def test_create_model():
    for name in SUPPORTED_MODELS:
        model = create_model(name)
        assert isinstance(model, torch.nn.Module)
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 10)
    try:
        create_model("nonexistent")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("[PASS] test_create_model")


def test_indexed_dataset():
    class FakeDS:
        def __len__(self):
            return 5
        def __getitem__(self, idx):
            return torch.zeros(3), idx
    ds = IndexedDataset(FakeDS())
    assert len(ds) == 5
    img, lbl, idx = ds[2]
    assert idx == 2
    print("[PASS] test_indexed_dataset")


def test_oram_storage_ram():
    Profiler.reset()
    n = 16
    with ORAMStorage(num_samples=n, backend="ram", block_size=DEFAULT_BLOCK_SIZE) as store:
        assert store.num_samples == n
        assert store.backend == "ram"
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        store.write(0, img, 5)
        read_img, read_label = store.read(0)
        assert read_label == 5
        assert read_img.shape == (32, 32, 3)
        np.testing.assert_array_equal(read_img, img)
    Profiler.reset()
    print("[PASS] test_oram_storage_ram")


def test_oram_storage_file():
    Profiler.reset()
    n = 8
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "oram_test.bin")
        with ORAMStorage(num_samples=n, storage_path=path, backend="file", block_size=DEFAULT_BLOCK_SIZE) as store:
            img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            store.write(3, img, 7)
            read_img, read_label = store.read(3)
            assert read_label == 7
            np.testing.assert_array_equal(read_img, img)
    Profiler.reset()
    print("[PASS] test_oram_storage_file")


def test_oram_storage_bounds():
    Profiler.reset()
    with ORAMStorage(num_samples=4, backend="ram") as store:
        try:
            store.read(-1)
            assert False
        except ValueError:
            pass
        try:
            store.read(4)
            assert False
        except ValueError:
            pass
        try:
            store.write(4, np.zeros((32, 32, 3), dtype=np.uint8), 0)
            assert False
        except ValueError:
            pass
    Profiler.reset()
    print("[PASS] test_oram_storage_bounds")


def test_oram_storage_block_size_validation():
    Profiler.reset()
    try:
        ORAMStorage(num_samples=4, backend="ram", block_size=10)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    try:
        ORAMStorage(num_samples=4, backend="invalid")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    Profiler.reset()
    print("[PASS] test_oram_storage_block_size_validation")


def test_sidecar_logger():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        path = f.name
    try:
        with SidecarLogger(path) as sc:
            sc.log(batch_id="0_0_train", epoch=0, phase="train")
            sc.log_at(timestamp=1234.5, batch_id="0_1_probe", epoch=0, phase="probe")
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["batch_id"] == "0_0_train"
        assert rows[1]["batch_id"] == "0_1_probe"
        assert float(rows[1]["timestamp"]) == 1234.5
    finally:
        os.unlink(path)
    print("[PASS] test_sidecar_logger")


def test_plaintext_event_generation():
    events = plaintext(
        train_size=100, holdout_size=50, epochs=2, batch_size=32,
        probe_batch_prob=0.5, probe_mix_ratio=0.3, random_state=42,
        data_dir=_DATA_DIR,
    )
    assert len(events) > 0
    members = [e for e in events if e[4] == 1]
    nonmembers = [e for e in events if e[4] == 0]
    assert members
    assert nonmembers
    for sid, ts, epoch, batch_id, label in events:
        assert isinstance(sid, str)
        assert isinstance(ts, float)
        assert ts > 0
        assert label in (0, 1)
    print("[PASS] test_plaintext_event_generation")


def test_oram_event_generation():
    events = oram_event(
        train_size=100, holdout_size=50, epochs=2, batch_size=32,
        probe_batch_prob=0.5, probe_mix_ratio=0.3, random_state=42,
        data_dir=_DATA_DIR,
    )
    assert len(events) > 0
    for sid, ts, epoch, batch_id, label in events:
        assert sid.startswith("block_")
    print("[PASS] test_oram_event_generation")


def test_save_events_csv():
    events = [
        ("s1", 0.001, 0, "0_0_train", 1),
        ("s2", 0.002, 0, "0_0_probe", 0),
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        path = f.name
    try:
        Save.events_csv(events, path)
        assert os.path.exists(path)
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["sample_id"] == "s1"
        assert rows[1]["label"] == "0"
    finally:
        os.unlink(path)
    print("[PASS] test_save_events_csv")


def test_safe_stats():
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = safe_stats(vals, "test")
    assert result["test_mean"] == 3.0
    assert result["test_min"] == 1.0
    assert result["test_max"] == 5.0

    empty_result = safe_stats(np.array([]), "empty")
    assert empty_result["empty_mean"] == 0.0
    print("[PASS] test_safe_stats")


def test_coefficient_of_variation():
    assert coefficient_of_variation(np.array([])) == 0.0
    vals = np.array([10.0, 10.0, 10.0])
    assert coefficient_of_variation(vals) == 0.0
    vals = np.array([1.0, 2.0, 3.0])
    assert coefficient_of_variation(vals) > 0
    print("[PASS] test_coefficient_of_variation")


def test_burstiness():
    assert attack_burstiness(np.array([])) == 0.0
    uniform = np.array([1.0, 1.0, 1.0])
    assert attack_burstiness(uniform) == -1.0
    print("[PASS] test_burstiness")


def test_maybe_reorder():
    import random
    events = [{"id": i} for i in range(10)]
    result = maybe_reorder(events, 0, random.Random(42))
    assert result == events
    result = maybe_reorder(events, 1, random.Random(42))
    assert result == events
    result = maybe_reorder(events, 3, random.Random(42))
    assert len(result) == len(events)
    assert {r["id"] for r in result} == set(range(10))
    print("[PASS] test_maybe_reorder")


def test_parse_events():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "timestamp", "epoch", "batch_id", "label"])
        writer.writerow(["s1", "0.001", "0", "0_0_train", "1"])
        writer.writerow(["s2", "0.002", "0", "0_0_probe", "0"])
        path = f.name
    try:
        df = parse_events(path)
        assert len(df) == 2
        assert set(df.columns) >= {"sample_id", "timestamp", "epoch", "batch_id", "label"}
    finally:
        os.unlink(path)
    print("[PASS] test_parse_events")


def test_subsample_visibility():
    import pandas as pd
    df = pd.DataFrame({
        "sample_id": [str(i) for i in range(1000)],
        "timestamp": np.arange(1000, dtype=float),
        "epoch": [0] * 1000,
        "batch_id": ["b"] * 1000,
        "label": [1] * 500 + [0] * 500,
    })
    full = subsample_visibility(df, 1.0, 42)
    assert len(full) == 1000
    half = subsample_visibility(df, 0.5, 42)
    assert 400 < len(half) < 600
    print("[PASS] test_subsample_visibility")


def test_run_config():
    cfg = RunConfig(
        name="test", defense="plaintext", visibility=1.0,
        dataset_root="/tmp", epochs=1, batch_size=32,
        device="cpu", seed=42,
    )
    assert cfg.name == "test"
    assert cfg.defense == "plaintext"
    assert cfg.oram_block_size == 4096
    print("[PASS] test_run_config")


def test_phases_registry():
    assert set(PHASES.keys()) == {"0", "1", "2", "3", "4", "5", "6", "7", "8"}
    for key, fn in PHASES.items():
        assert callable(fn)
    print("[PASS] test_phases_registry")


def test_python_syntax():
    src_dir = os.path.dirname(os.path.abspath(__file__))
    for fname in os.listdir(src_dir):
        if fname.endswith(".py"):
            fpath = os.path.join(src_dir, fname)
            with open(fpath) as f:
                ast.parse(f.read())
    print("[PASS] test_python_syntax")


def test_mixed_access_report():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "timestamp", "epoch", "batch_id", "label"])
        for i in range(10):
            writer.writerow([f"s{i}", str(i * 0.001), "0", "b0", "1"])
        for i in range(5):
            writer.writerow([f"h{i}", str(10 + i * 0.001), "0", "b1", "0"])
        path = f.name
    try:
        report = Build.mixed_access_report(path)
        assert report["has_both_labels"]
        assert report["num_rows"] == 15
        assert report["unique_samples"] == 15
    finally:
        os.unlink(path)
    print("[PASS] test_mixed_access_report")


def main():
    tests = [
        test_profiler,
        test_profiler_context,
        test_resolve_device,
        test_create_model,
        test_indexed_dataset,
        test_oram_storage_ram,
        test_oram_storage_file,
        test_oram_storage_bounds,
        test_oram_storage_block_size_validation,
        test_sidecar_logger,
        test_plaintext_event_generation,
        test_oram_event_generation,
        test_save_events_csv,
        test_safe_stats,
        test_coefficient_of_variation,
        test_burstiness,
        test_maybe_reorder,
        test_parse_events,
        test_subsample_visibility,
        test_run_config,
        test_phases_registry,
        test_python_syntax,
        test_mixed_access_report,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"RESULTS: {passed} passed, {failed} failed, {passed + failed} total")
    print(f"{'='*60}")
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
