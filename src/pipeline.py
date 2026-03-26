"""
Experiment pipeline orchestration.

RunConfig: Experiment configuration dataclass
Trace: OS-level trace capture and conversion
Sweep: Parameter sweep experiments
ExperimentPhases: Phased benchmark runner
Subprocess helpers: popen_logged, wait_success, etc.
Orchestration: trainer_command, single_configuration, etc.
"""

import csv
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

from oram import baseline as train_baseline, oram as train_oram, read_oram_audit_counts
from attack import Build, upgraded_attack
from figures import Save


MEMBER_RE = re.compile(r"^member_(\d+)\.bin$")
NONMEMBER_RE = re.compile(r"^nonmember_(\d+)\.bin$")
STRACE_RE = re.compile(r"^(\d+\.\d+)\s+.*?(open|openat|openat2)\(.*?\"([^\"]+)\"")
STRACE_OPEN_FD_RE = re.compile(
    r"^(\d+\.\d+)\s+.*?(open|openat|openat2)\(.*?\"([^\"]+)\".*?\)\s+=\s+(-?\d+)"
)
STRACE_PREAD_RE = re.compile(
    r"^(\d+\.\d+)\s+.*?pread64\((\d+),.*?,\s*(\d+),\s*(-?\d+)\)\s*=\s*(-?\d+)"
)
STRACE_PWRITE_RE = re.compile(
    r"^(\d+\.\d+)\s+.*?pwrite64\((\d+),.*?,\s*(\d+),\s*(-?\d+)\)\s*=\s*(-?\d+)"
)
STRACE_LSEEK_RE = re.compile(
    r"^(\d+\.\d+)\s+.*?lseek\((\d+),\s*(-?\d+),\s*SEEK_[A-Z]+\)\s*=\s*(-?\d+)"
)
FS_USAGE_TIME_RE = re.compile(r"^(\d{2}:\d{2}:\d{2}\.\d{6})")
FS_USAGE_DISK_OFF_RE = re.compile(r"D=0x([0-9A-Fa-f]+)")
FS_USAGE_PATH_RE = re.compile(r"(/\S*oram\.bin)", re.IGNORECASE)
FS_USAGE_MEMBER_PATH_RE = re.compile(r"(/\S*(?:member|nonmember)_\d+\.bin)")

LEAK_PATTERNS = [
    re.compile(r"member_", re.IGNORECASE),
    re.compile(r"nonmember_", re.IGNORECASE),
    re.compile(r"/train/"),
    re.compile(r"/probe/"),
]



@dataclass
class RunConfig:
    name: str
    defense: str
    visibility: float
    dataset_root: str
    epochs: int
    batch_size: int
    device: str
    seed: int
    decoys_per_access: int = 0
    prefetch_size: int = 1
    release_shuffle_window: int = 1
    oram_backend: str = "file"
    oram_block_size: int = 4096



RESULTS_ROOT = "results"
BATCH_SIZES = [32, 64, 128, 256]
DATASET_SIZES = [1000, 5000, 10000, 50000]
BLOCK_SIZES = [4096, 8192, 16384, 32768, 65536]
DEFAULT_BATCH_SIZE = 128
DEFAULT_DATASET_SIZE = 5000



def popen_logged(
    cmd: List[str],
    stdout_path: str,
    stderr_path: str,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
) -> subprocess.Popen:
    stdout_f = open(stdout_path, "w", encoding="utf-8")
    stderr_f = open(stderr_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        stdout=stdout_f,
        stderr=stderr_f,
        cwd=cwd,
        env=env,
        text=True,
    )
    proc._stdout_file = stdout_f
    proc._stderr_file = stderr_f
    return proc


def close_proc_files(proc: subprocess.Popen) -> None:
    for attr in ["_stdout_file", "_stderr_file"]:
        f = getattr(proc, attr, None)
        if f is not None:
            try:
                f.close()
            except Exception:
                pass


def stop_process(proc: subprocess.Popen, grace_seconds: float = 3.0) -> None:
    if proc.poll() is not None:
        close_proc_files(proc)
        return

    try:
        proc.send_signal(signal.SIGINT)
    except Exception:
        pass

    deadline = time.time() + grace_seconds
    while time.time() < deadline:
        if proc.poll() is not None:
            close_proc_files(proc)
            return
        time.sleep(0.1)

    try:
        proc.terminate()
    except Exception:
        pass

    deadline = time.time() + grace_seconds
    while time.time() < deadline:
        if proc.poll() is not None:
            close_proc_files(proc)
            return
        time.sleep(0.1)

    try:
        proc.kill()
    except Exception:
        pass

    close_proc_files(proc)


def wait_success(proc: subprocess.Popen, name: str) -> None:
    ret = proc.wait()
    close_proc_files(proc)
    if ret != 0:
        raise RuntimeError(f"{name} failed with exit code {ret}")



class Trace:
    """Syscall / fs trace capture, conversion, and parsing."""

    @staticmethod
    def attach(pid: int, strace_log_path: str) -> subprocess.Popen:
        if shutil.which("strace") is None:
            if sys.platform == "darwin":
                raise RuntimeError(
                    "strace is not available on macOS. Use run.sh macos "
                    "for physical file-system tracing with fs_usage."
                )
            raise RuntimeError(
                "strace is required for physical trace auditing but was not found. "
                "Run this command on Linux with strace installed."
            )
        cmd = [
            "strace",
            "-ff",
            "-ttt",
            "-e",
            "trace=open,openat,openat2,read,write,pread64,pwrite64,lseek",
            "-p",
            str(pid),
        ]
        stderr_f = open(strace_log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=stderr_f,
            text=True,
        )
        proc._stderr_file = stderr_f
        return proc

    @staticmethod
    def convert(run_dir: str, cfg: RunConfig) -> str:
        trace_log = os.path.join(run_dir, "strace.log")
        sidecar = os.path.join(run_dir, "batch_sidecar.csv")
        output = os.path.join(run_dir, "events_trace.csv")
        trace_validation = os.path.join(run_dir, "trace_validation.json")
        attack_input_audit = os.path.join(run_dir, "attack_input_audit.json")

        cmd = [
            sys.executable,
            "run.py",
            "convert",
            "--trace_input", trace_log,
            "--trace_mode", "strace",
            "--sidecar", sidecar,
            "--defense", cfg.defense,
            "--oram_block_size", str(cfg.oram_block_size),
            "--trace_validation_out", trace_validation,
            "--attack_input_audit_out", attack_input_audit,
            "--output", output,
        ]
        proc = popen_logged(
            cmd,
            stdout_path=os.path.join(run_dir, "convert_stdout.log"),
            stderr_path=os.path.join(run_dir, "convert_stderr.log"),
        )
        wait_success(proc, "trace conversion")
        return output

    @staticmethod
    def cmd(args):
        try:
            from bcc import BPF
        except ImportError:
            print("Error: BCC not available. Install with: sudo apt install bpfcc-tools python3-bpfcc")
            return 1

        BPF_PROGRAM = r"""
#include <uapi/linux/ptrace.h>
#include <linux/sched.h>

struct data_t {
    u32 pid;
    u64 ts_ns;
    char comm[TASK_COMM_LEN];
    char fname[256];
};

BPF_PERF_OUTPUT(events);

int trace_openat_entry(struct pt_regs *ctx, int dfd, const char __user *filename, int flags, umode_t mode) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    FILTER_PID

    struct data_t data = {};
    data.pid = pid;
    data.ts_ns = bpf_ktime_get_ns();
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    bpf_probe_read_user_str(&data.fname, sizeof(data.fname), filename);
    events.perf_submit(ctx, &data, sizeof(data));
    return 0;
}
"""

        text = BPF_PROGRAM.replace("FILTER_PID", f"if (pid != {args.pid}) return 0;")
        b = BPF(text=text)

        hooked = False
        for fn in ["openat", "open"]:
            for arch_prefix in ["__x64_sys_", "__arm64_sys_", "sys_", ""]:
                try:
                    b.attach_kprobe(event=f"{arch_prefix}{fn}", fn_name="trace_openat_entry")
                    hooked = True
                    print(f"Attached to {arch_prefix}{fn}", file=sys.stderr)
                except Exception:
                    pass

        if not hooked:
            print("Failed to attach to open/openat syscalls on this system.", file=sys.stderr)
            return 1

        out = open(args.output, "w", newline="", encoding="utf-8")
        writer = csv.writer(out)
        writer.writerow(["timestamp_ns", "pid", "comm", "filename"])

        running = True

        def stop_handler(signum, frame):
            nonlocal running
            running = False

        signal.signal(signal.SIGINT, stop_handler)
        signal.signal(signal.SIGTERM, stop_handler)

        def handle_event(cpu, data, size):
            event = b["events"].event(data)
            fname = event.fname.decode("utf-8", errors="replace").rstrip("\x00")
            comm = event.comm.decode("utf-8", errors="replace").rstrip("\x00")
            writer.writerow([event.ts_ns, event.pid, comm, fname])
            out.flush()

        b["events"].open_perf_buffer(handle_event)

        print(f"Tracing PID {args.pid}. Writing to {args.output}. Ctrl-C to stop.", file=sys.stderr)

        try:
            while running:
                b.perf_buffer_poll(timeout=100)
        except KeyboardInterrupt:
            pass

        out.close()
        print("Done.", file=sys.stderr)
        return 0

    @staticmethod
    def ebpf_csv(path: str) -> List[Tuple[float, str]]:
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts = float(row["timestamp_ns"]) / 1e9
                rows.append((ts, row["filename"]))
        rows.sort(key=lambda x: x[0])
        return rows

    @staticmethod
    def path_events(path: str) -> List[Tuple[float, str]]:
        rows = []
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                m = STRACE_RE.search(line)
                if not m:
                    continue
                ts = float(m.group(1))
                fname = m.group(3)
                rows.append((ts, fname))
        rows.sort(key=lambda x: x[0])
        return rows

    @staticmethod
    def oram_events(path: str, block_size: int) -> Tuple[List[Tuple[float, str]], Dict[str, object]]:
        rows: List[Tuple[float, str]] = []
        fd_to_path: Dict[int, str] = {}
        validation: Dict[str, object] = {
            "open_rows": 0,
            "pread_rows": 0,
            "pwrite_rows": 0,
            "lseek_rows": 0,
            "candidate_paths": [],
            "quantized_blocks_seen": 0,
        }

        blocks_seen = set()
        path_seen = set()
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                m_open = STRACE_OPEN_FD_RE.search(line)
                if m_open:
                    ts = float(m_open.group(1))
                    path_name = m_open.group(3)
                    fd = int(m_open.group(4))
                    validation["open_rows"] += 1
                    if fd >= 0:
                        fd_to_path[fd] = path_name
                        if "oram" in path_name.lower():
                            path_seen.add(path_name)
                    continue

                m_pread = STRACE_PREAD_RE.search(line)
                if m_pread:
                    ts = float(m_pread.group(1))
                    fd = int(m_pread.group(2))
                    offset = int(m_pread.group(4))
                    validation["pread_rows"] += 1
                    if fd in fd_to_path and offset >= 0:
                        block_id = offset // block_size
                        token = f"oram_blk_{block_id}"
                        rows.append((ts, token))
                        blocks_seen.add(block_id)
                    continue

                m_pwrite = STRACE_PWRITE_RE.search(line)
                if m_pwrite:
                    ts = float(m_pwrite.group(1))
                    fd = int(m_pwrite.group(2))
                    offset = int(m_pwrite.group(4))
                    validation["pwrite_rows"] += 1
                    if fd in fd_to_path and offset >= 0:
                        block_id = offset // block_size
                        token = f"oram_blk_{block_id}"
                        rows.append((ts, token))
                        blocks_seen.add(block_id)
                    continue

                m_lseek = STRACE_LSEEK_RE.search(line)
                if m_lseek:
                    validation["lseek_rows"] += 1

        rows.sort(key=lambda x: x[0])
        validation["quantized_blocks_seen"] = len(blocks_seen)
        validation["candidate_paths"] = sorted(path_seen)[:20]
        return rows, validation

    @staticmethod
    def fs_usage(
        path: str,
        markers: List[Tuple[float, str, int, str]],
        defense: str,
        block_size: int,
    ) -> Tuple[List[Tuple[float, str]], Dict[str, object]]:
        def hms_to_seconds(hms: str) -> float:
            hh = int(hms[0:2])
            mm = int(hms[3:5])
            ss = float(hms[6:])
            return hh * 3600.0 + mm * 60.0 + ss

        rows: List[Tuple[float, str]] = []
        validation: Dict[str, object] = {
            "open_rows": 0,
            "pread_rows": 0,
            "pwrite_rows": 0,
            "lseek_rows": 0,
            "candidate_paths": [],
            "quantized_blocks_seen": 0,
        }

        if not markers:
            return rows, validation

        first_marker = markers[0][0]
        day_start = datetime.datetime.fromtimestamp(first_marker).replace(
            hour=0, minute=0, second=0, microsecond=0
        ).timestamp()

        prev_sec_of_day: Optional[float] = None
        day_offset = 0.0
        blocks_seen = set()
        path_seen = set()

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                tm = FS_USAGE_TIME_RE.match(line)
                if not tm:
                    continue
                sec_of_day = hms_to_seconds(tm.group(1))
                if prev_sec_of_day is not None and sec_of_day + 1.0 < prev_sec_of_day:
                    day_offset += 86400.0
                prev_sec_of_day = sec_of_day
                ts = day_start + day_offset + sec_of_day

                if defense == "oram":
                    if "oram.bin" not in line.lower():
                        continue
                    path_match = FS_USAGE_PATH_RE.search(line)
                    if path_match:
                        path_seen.add(path_match.group(1))
                    if "rddat" in line.lower():
                        validation["pread_rows"] += 1
                    if "wrdat" in line.lower():
                        validation["pwrite_rows"] += 1
                    disk_off = FS_USAGE_DISK_OFF_RE.search(line)
                    if disk_off:
                        block_id = int(disk_off.group(1), 16) // block_size
                        blocks_seen.add(block_id)
                        token = f"oram_blk_{block_id}"
                    else:
                        token = "oram_blk_fallback"
                    rows.append((ts, token))
                    continue

                member_match = FS_USAGE_MEMBER_PATH_RE.search(line)
                if not member_match:
                    continue
                validation["open_rows"] += 1
                rows.append((ts, member_match.group(1)))

        rows.sort(key=lambda x: x[0])
        validation["quantized_blocks_seen"] = len(blocks_seen)
        validation["candidate_paths"] = sorted(path_seen)[:20]
        return rows, validation



class Sweep:
    """Parameter sweep experiments."""

    @staticmethod
    def batch_size(epochs: int, output_root: str, device: str = None):
        print("\n" + "=" * 60)
        print("BATCH SIZE SWEEP")
        print("=" * 60)

        results = []

        for bs in BATCH_SIZES:
            tag = f"baseline_bs{bs}"
            out_dir = os.path.join(output_root, "sweep_batch_size", tag)
            print(f"\n--- Baseline batch_size={bs}, epochs={epochs} ---")
            try:
                hist = train_baseline(
                    num_epochs=epochs,
                    batch_size=bs,
                    output_dir=out_dir,
                    device=device,
                )
                results.append({
                    "mode": "baseline",
                    "batch_size": bs,
                    "epochs": epochs,
                    "total_time": hist["total_time"],
                    "best_acc": hist["best_acc"],
                    "final_train_loss": hist["train_loss"][-1],
                })
            except Exception as exc:
                print(f"ERROR in baseline bs={bs}: {exc}")
                results.append({
                    "mode": "baseline",
                    "batch_size": bs,
                    "error": str(exc),
                })

            tag = f"oram_bs{bs}"
            out_dir = os.path.join(output_root, "sweep_batch_size", tag)
            print(f"\n--- ORAM batch_size={bs}, epochs={epochs}, samples={DEFAULT_DATASET_SIZE} ---")
            try:
                hist = train_oram(
                    num_epochs=epochs,
                    batch_size=bs,
                    output_dir=out_dir,
                    device=device,
                    num_samples=DEFAULT_DATASET_SIZE,
                )
                results.append({
                    "mode": "oram",
                    "batch_size": bs,
                    "num_samples": DEFAULT_DATASET_SIZE,
                    "epochs": epochs,
                    "total_time": hist["total_time"],
                    "best_acc": hist["best_acc"],
                    "final_train_loss": hist["train_loss"][-1],
                })
            except Exception as exc:
                print(f"ERROR in oram bs={bs}: {exc}")
                results.append({
                    "mode": "oram",
                    "batch_size": bs,
                    "error": str(exc),
                })

        summary_path = os.path.join(output_root, "sweep_batch_size", "sweep_summary.json")
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nBatch size sweep summary saved to: {summary_path}")

        return results

    @staticmethod
    def dataset_size(epochs: int, output_root: str, device: str = None):
        print("\n" + "=" * 60)
        print("DATASET SIZE SWEEP (ORAM)")
        print("=" * 60)

        results = []

        for n in DATASET_SIZES:
            tag = f"oram_n{n}"
            out_dir = os.path.join(output_root, "sweep_dataset_size", tag)
            print(f"\n--- ORAM num_samples={n}, batch_size={DEFAULT_BATCH_SIZE}, epochs={epochs} ---")
            try:
                hist = train_oram(
                    num_epochs=epochs,
                    batch_size=DEFAULT_BATCH_SIZE,
                    output_dir=out_dir,
                    device=device,
                    num_samples=n,
                )
                results.append({
                    "mode": "oram",
                    "num_samples": n,
                    "batch_size": DEFAULT_BATCH_SIZE,
                    "epochs": epochs,
                    "total_time": hist["total_time"],
                    "best_acc": hist["best_acc"],
                    "final_train_loss": hist["train_loss"][-1],
                })
            except Exception as exc:
                print(f"ERROR in oram n={n}: {exc}")
                results.append({
                    "mode": "oram",
                    "num_samples": n,
                    "error": str(exc),
                })

        summary_path = os.path.join(output_root, "sweep_dataset_size", "sweep_summary.json")
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nDataset size sweep summary saved to: {summary_path}")

        return results

    @staticmethod
    def block_size(epochs: int, output_root: str, device: str = None):
        print("\n" + "=" * 60)
        print("BLOCK SIZE SWEEP (ORAM)")
        print("=" * 60)

        results = []

        for bs in BLOCK_SIZES:
            tag = f"oram_block{bs}"
            out_dir = os.path.join(output_root, "sweep_block_size", tag)
            print(f"\n--- ORAM block_size={bs}, epochs={epochs}, samples={DEFAULT_DATASET_SIZE} ---")
            try:
                hist = train_oram(
                    num_epochs=epochs,
                    batch_size=DEFAULT_BATCH_SIZE,
                    output_dir=out_dir,
                    device=device,
                    num_samples=DEFAULT_DATASET_SIZE,
                    block_size=bs,
                )
                results.append({
                    "mode": "oram",
                    "block_size": bs,
                    "num_samples": DEFAULT_DATASET_SIZE,
                    "epochs": epochs,
                    "total_time": hist["total_time"],
                    "best_acc": hist["best_acc"],
                    "final_train_loss": hist["train_loss"][-1],
                })
            except Exception as exc:
                print(f"ERROR in oram block_size={bs}: {exc}")
                results.append({
                    "mode": "oram",
                    "block_size": bs,
                    "error": str(exc),
                })

        summary_path = os.path.join(output_root, "sweep_block_size", "sweep_summary.json")
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nBlock size sweep summary saved to: {summary_path}")

        return results

    @staticmethod
    def main(args):
        start = time.time()

        if args.sweep in ("batch_size", "all"):
            Sweep.batch_size(args.epochs, args.output_dir, args.device)

        if args.sweep in ("dataset_size", "all"):
            Sweep.dataset_size(args.epochs, args.output_dir, args.device)

        if args.sweep in ("block_size", "all"):
            Sweep.block_size(args.epochs, args.output_dir, args.device)

        elapsed = time.time() - start
        print(f"\nSweep complete. Total wall-clock time: {elapsed:.1f}s ({elapsed/3600:.2f}h)")



class ExperimentPhases:
    """Phased CIFAR/ORAM experiments. Methods `one` … `nine` map to phase indices 0 … 8."""

    def __init__(self, results_root: str = RESULTS_ROOT) -> None:
        self.results_root = results_root

    def _out(self, phase: int, tag: str = "") -> str:
        base = os.path.join(self.results_root, f"phase{phase}")
        return os.path.join(base, tag) if tag else base

    def _marker(self, phase: int, tag: str = "") -> str:
        return os.path.join(self._out(phase, tag), "history.json")

    def _done(self, phase: int, tag: str = "", force: bool = False) -> bool:
        if force:
            return False
        return os.path.exists(self._marker(phase, tag))

    def _hist_row(self, hist, **extra):
        row = {
            "total_time": hist.get("total_time", 0),
            "best_acc": hist.get("best_acc", 0),
            "epochs": len(hist.get("epochs", [])),
        }
        row.update(extra)
        return row

    def one(self, args):
        print("\n" + "="*60)
        print("PHASE 0: Baseline Reproduction")
        print("="*60)
        rows = []

        if not self._done(0, "baseline", args.force):
            print("  Running plaintext baseline (3 epochs)...")
            h = train_baseline(num_epochs=3, batch_size=128,
                                      output_dir=self._out(0, "baseline"), device=args.device)
            rows.append(self._hist_row(h, mode="plaintext", backend="none"))
        else:
            print("  SKIP: plaintext baseline exists")

        if not self._done(0, "oram_file", args.force):
            print("  Running file-backed ORAM (2 epochs, 10k samples)...")
            h = train_oram(num_epochs=2, batch_size=128,
                                  output_dir=self._out(0, "oram_file"), device=args.device,
                                  num_samples=10000, backend="file")
            rows.append(self._hist_row(h, mode="oram", backend="file"))
        else:
            print("  SKIP: file ORAM exists")

        Save.dict_csv(rows, os.path.join(self._out(0), "phase0_results.csv"))

    def two(self, args):
        print("\n" + "="*60)
        print("PHASE 1: RAM Backend")
        print("="*60)
        rows = []

        for backend in ("file", "ram"):
            tag = f"oram_{backend}"
            if self._done(1, tag, args.force):
                print(f"  SKIP: {tag} exists")
                continue
            print(f"  Running backend={backend} (2 epochs, 10k samples)...")
            h = train_oram(num_epochs=2, batch_size=128,
                                  output_dir=self._out(1, tag), device=args.device,
                                  num_samples=10000, backend=backend)
            rows.append(self._hist_row(h, backend=backend))

        Save.dict_csv(rows, os.path.join(self._out(1), "phase1_results.csv"))

    def three(self, args):
        print("\n" + "="*60)
        print("PHASE 2: Mediated Multi-Worker Loader")
        print("="*60)
        rows = []

        for nw in (0, 1, 2, 4):
            tag = f"workers_{nw}"
            if self._done(2, tag, args.force):
                print(f"  SKIP: {tag} exists")
                continue
            print(f"  Running num_workers={nw} (2 epochs, 10k samples)...")
            h = train_oram(num_epochs=2, batch_size=128,
                                  output_dir=self._out(2, tag), device=args.device,
                                  num_samples=10000, backend="file",
                                  num_workers=nw)
            rows.append(self._hist_row(h, num_workers=nw))

        Save.dict_csv(rows, os.path.join(self._out(2), "phase2_results.csv"))

    def four(self, args):
        print("\n" + "="*60)
        print("PHASE 3: Block Size Sweep")
        print("="*60)
        rows = []
        block_sizes = [4096, 8192, 16384, 32768, 65536]

        for bs in block_sizes:
            tag = f"block_{bs}"
            if self._done(3, tag, args.force):
                print(f"  SKIP: {tag} exists")
                continue
            print(f"  Running block_size={bs} (2 epochs, 10k samples)...")
            h = train_oram(num_epochs=2, batch_size=128,
                                  output_dir=self._out(3, tag), device=args.device,
                                  num_samples=10000, backend="file",
                                  block_size=bs)
            rows.append(self._hist_row(h, block_size=bs))

        Save.dict_csv(rows, os.path.join(self._out(3), "phase3_results.csv"))

    def five(self, args):
        print("\n" + "="*60)
        print("PHASE 4: Model Scaling")
        print("="*60)
        rows = []
        models = ["resnet18", "resnet50", "efficientnet_b0"]

        for model in models:
            tag = f"baseline_{model}"
            if not self._done(4, tag, args.force):
                print(f"  Running plaintext {model} (3 epochs)...")
                h = train_baseline(num_epochs=3, batch_size=128,
                                          output_dir=self._out(4, tag), device=args.device,
                                          model_name=model)
                rows.append(self._hist_row(h, mode="plaintext", model=model))
            else:
                print(f"  SKIP: {tag} exists")

            tag = f"oram_{model}"
            if not self._done(4, tag, args.force):
                print(f"  Running ORAM {model} (2 epochs, 5k samples)...")
                h = train_oram(num_epochs=2, batch_size=128,
                                      output_dir=self._out(4, tag), device=args.device,
                                      num_samples=5000, model_name=model)
                rows.append(self._hist_row(h, mode="oram", model=model))
            else:
                print(f"  SKIP: {tag} exists")

        Save.dict_csv(rows, os.path.join(self._out(4), "phase4_results.csv"))

    def six(self, args):
        print("\n" + "="*60)
        print("PHASE 5: Dataset Size Scaling")
        print("="*60)
        rows = []
        sizes = [5000, 10000, 25000, 50000]

        for n in sizes:
            tag = f"n_{n}"
            if self._done(5, tag, args.force):
                print(f"  SKIP: {tag} exists")
                continue
            print(f"  Running N={n} (2 epochs)...")
            h = train_oram(num_epochs=2, batch_size=128,
                                  output_dir=self._out(5, tag), device=args.device,
                                  num_samples=n)
            rows.append(self._hist_row(h, num_samples=n))

        Save.dict_csv(rows, os.path.join(self._out(5), "phase5_results.csv"))

    def seven(self, args):
        print("\n" + "="*60)
        print("PHASE 6: Batch Size Sweep")
        print("="*60)
        rows = []
        batch_sizes = [32, 64, 128, 256, 512]

        for bs in batch_sizes:
            tag = f"baseline_bs{bs}"
            if not self._done(6, tag, args.force):
                print(f"  Running plaintext batch_size={bs} (3 epochs)...")
                h = train_baseline(num_epochs=3, batch_size=bs,
                                          output_dir=self._out(6, tag), device=args.device)
                rows.append(self._hist_row(h, mode="plaintext", batch_size=bs))
            else:
                print(f"  SKIP: {tag} exists")

            tag = f"oram_bs{bs}"
            if not self._done(6, tag, args.force):
                print(f"  Running ORAM batch_size={bs} (2 epochs, 5k samples)...")
                h = train_oram(num_epochs=2, batch_size=bs,
                                      output_dir=self._out(6, tag), device=args.device,
                                      num_samples=5000)
                rows.append(self._hist_row(h, mode="oram", batch_size=bs))
            else:
                print(f"  SKIP: {tag} exists")

        Save.dict_csv(rows, os.path.join(self._out(6), "phase6_results.csv"))

    def eight(self, args):
        print("\n" + "="*60)
        print("PHASE 7: Access Pattern Leakage Demo")
        print("="*60)
        marker = os.path.join(self._out(7), "leakage_comparison.png")
        if os.path.exists(marker) and not args.force:
            print("  SKIP: leakage results exist")
            return

        exp_dir = os.path.dirname(os.path.abspath(__file__))
        run_script = os.path.join(exp_dir, "run.py")
        subprocess.check_call([
            sys.executable, run_script,
            "leakage",
            "--num-samples", "5000",
            "--batch-size", "128",
            "--epochs", "3",
            "--output-dir", self._out(7),
        ])

    def nine(self, args):
        print("\n" + "="*60)
        print("PHASE 8: Final Combined Optimization")
        print("="*60)
        rows = []

        tag = "plaintext"
        if not self._done(8, tag, args.force):
            print("  Running plaintext reference (3 epochs)...")
            h = train_baseline(num_epochs=3, batch_size=128,
                                      output_dir=self._out(8, tag), device=args.device)
            rows.append(self._hist_row(h, config="plaintext"))
        else:
            print("  SKIP: plaintext exists")

        tag = "oram_baseline"
        if not self._done(8, tag, args.force):
            print("  Running baseline ORAM (2 epochs, 10k samples)...")
            h = train_oram(num_epochs=2, batch_size=128,
                                  output_dir=self._out(8, tag), device=args.device,
                                  num_samples=10000, backend="file",
                                  num_workers=0, block_size=4096)
            rows.append(self._hist_row(h, config="oram_baseline"))
        else:
            print("  SKIP: oram_baseline exists")

        tag = "oram_optimized"
        if not self._done(8, tag, args.force):
            print("  Running optimized ORAM (2 epochs, 10k samples, ram, workers=4)...")
            h = train_oram(num_epochs=2, batch_size=256,
                                  output_dir=self._out(8, tag), device=args.device,
                                  num_samples=10000, backend="ram",
                                  num_workers=4, block_size=4096)
            rows.append(self._hist_row(h, config="oram_optimized"))
        else:
            print("  SKIP: oram_optimized exists")

        Save.dict_csv(rows, os.path.join(self._out(8), "phase8_results.csv"))


_EXPERIMENT_PHASES = ExperimentPhases()

PHASES = {
    "0": _EXPERIMENT_PHASES.one,
    "1": _EXPERIMENT_PHASES.two,
    "2": _EXPERIMENT_PHASES.three,
    "3": _EXPERIMENT_PHASES.four,
    "4": _EXPERIMENT_PHASES.five,
    "5": _EXPERIMENT_PHASES.six,
    "6": _EXPERIMENT_PHASES.seven,
    "7": _EXPERIMENT_PHASES.eight,
    "8": _EXPERIMENT_PHASES.nine,
}


def phases_main(args):
    start = time.time()

    if args.phase == "all":
        for key in sorted(PHASES.keys()):
            PHASES[key](args)
    elif args.phase in PHASES:
        PHASES[args.phase](args)
    else:
        print(f"Unknown phase: {args.phase}. Use 0-8 or 'all'.")
        sys.exit(1)

    elapsed = time.time() - start
    print(f"\nDone. Wall-clock: {elapsed:.1f}s ({elapsed/3600:.2f}h)")



def trainer_command(cfg: RunConfig, run_dir: str) -> List[str]:
    sidecar_path = os.path.join(run_dir, "batch_sidecar.csv")

    if cfg.defense == "plaintext":
        return [
            sys.executable,
            "run.py", "train",
            "--dataset_root", cfg.dataset_root,
            "--epochs", str(cfg.epochs),
            "--batch_size", str(cfg.batch_size),
            "--seed", str(cfg.seed),
            "--sidecar_path", sidecar_path,
            "--device", cfg.device,
        ]

    if cfg.defense == "obfuscatedcated":
        return [
            sys.executable,
            "run.py", "train",
            "--obfuscatedcated",
            "--dataset_root", cfg.dataset_root,
            "--epochs", str(cfg.epochs),
            "--batch_size", str(cfg.batch_size),
            "--seed", str(cfg.seed),
            "--decoys", str(cfg.decoys_per_access),
            "--prefetch", str(cfg.prefetch_size),
            "--shuffle_window", str(cfg.release_shuffle_window),
            "--sidecar_path", sidecar_path,
            "--device", cfg.device,
        ]

    if cfg.defense == "oram":
        return [
            sys.executable,
            "run.py",
            "sidecar",
            "--epochs", str(cfg.epochs),
            "--batch_size", str(cfg.batch_size),
            "--seed", str(cfg.seed),
            "--backend", cfg.oram_backend,
            "--block_size", str(cfg.oram_block_size),
            "--sidecar_path", sidecar_path,
            "--device", cfg.device,
            "--output_dir", run_dir,
        ]

    raise ValueError(f"Unknown defense type: {cfg.defense}")


def read_json(path: str) -> Dict[str, object]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_claims_matrix(run_dir: str, cfg: RunConfig, claim_rows: List[Dict[str, object]]) -> None:
    json_path = os.path.join(run_dir, "claims_audit_matrix.json")
    csv_path = os.path.join(run_dir, "claims_audit_matrix.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(claim_rows, f, indent=2)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["claim_id", "claim", "verified", "evidence", "gap", "fix_applied"])
        writer.writeheader()
        for row in claim_rows:
            writer.writerow(row)


def scan_events_for_leakage(events_csv: str) -> Dict[str, object]:
    patterns = ["member_", "nonmember_", "/train/", "/probe/"]
    hits = {p: 0 for p in patterns}
    with open(events_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    joined = "\n".join(",".join(r.values()) for r in rows)
    for p in patterns:
        hits[p] = joined.count(p)
    return {
        "rows": len(rows),
        "hits": hits,
        "has_leakage": any(v > 0 for v in hits.values()),
    }


def attack(run_dir: str, input_csv: str, visibility: float) -> Dict[str, object]:
    attack_dir = os.path.join(run_dir, f"attack_v{str(visibility).replace('.', 'p')}")
    ensure_dir(attack_dir)
    return upgraded_attack(
        input_path=input_csv,
        output_dir=attack_dir,
        visibility=visibility,
        random_state=42,
    )


def best_model_metrics(metrics_json: Dict[str, object]) -> Dict[str, float]:
    best = metrics_json["best_model"]
    results = metrics_json["results"][best]
    return {
        "best_auc": float(results["auc"]),
        "best_accuracy": float(results["accuracy"]),
        "best_ap": float(results["average_precision"]),
    }


def single_configuration(cfg: RunConfig, output_root: str, visibilities: List[float]) -> List[Dict[str, object]]:
    run_dir = os.path.join(output_root, cfg.name)
    ensure_dir(run_dir)

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    train_stdout = os.path.join(run_dir, "train_stdout.log")
    train_stderr = os.path.join(run_dir, "train_stderr.log")
    trainer_cmd = trainer_command(cfg, run_dir)
    oram_audit_log = os.path.join(run_dir, "oram_audit.log")
    train_env = os.environ.copy()
    if cfg.defense == "oram":
        train_env["ORAM_AUDIT_LOG"] = oram_audit_log

    print(f"  Launching trainer: {cfg.defense}")
    start_time = time.time()
    trainer_proc = popen_logged(trainer_cmd, train_stdout, train_stderr, cwd="experiments", env=train_env)

    time.sleep(1.5)

    strace_log = os.path.join(run_dir, "strace.log")
    print(f"  Attaching strace to PID {trainer_proc.pid}")
    strace_proc = Trace.attach(trainer_proc.pid, strace_log)

    try:
        print(f"  Waiting for training to complete...")
        wait_success(trainer_proc, f"trainer {cfg.name}")
    finally:
        stop_process(strace_proc)

    train_runtime = time.time() - start_time
    print(f"  Training runtime: {train_runtime:.1f}s")

    print(f"  Converting traces...")
    convert_start = time.time()
    input_csv = Trace.convert(run_dir, cfg)
    convert_runtime = time.time() - convert_start

    trace_validation = read_json(os.path.join(run_dir, "trace_validation.json"))
    attack_input_audit = read_json(os.path.join(run_dir, "attack_input_audit.json"))
    leakage_scan = scan_events_for_leakage(input_csv)
    mixed_access_report = Build.mixed_access_report(input_csv)
    if cfg.defense == "oram" and leakage_scan["has_leakage"]:
        raise RuntimeError(f"ORAM leakage detected in converted events: {leakage_scan['hits']}")
    if not mixed_access_report["has_both_labels"] or not mixed_access_report["non_binary_access_signal"]:
        raise RuntimeError(f"Mixed-access validity failed: {mixed_access_report}")

    oram_counts = read_oram_audit_counts(oram_audit_log)
    if cfg.defense == "oram" and oram_counts["read"] <= 0:
        raise RuntimeError("ORAM run produced zero audited read operations.")

    claims_rows = [
        {
            "claim_id": "C1",
            "claim": "ORAM backend is exercised during training",
            "verified": bool(cfg.defense != "oram" or oram_counts["read"] > 0),
            "evidence": f"oram_audit reads={oram_counts['read']} writes={oram_counts['write']}",
            "gap": "" if (cfg.defense != "oram" or oram_counts["read"] > 0) else "No ORAM reads observed",
            "fix_applied": "Added ORAM_AUDIT_LOG instrumentation and hard gate",
        },
        {
            "claim_id": "C2",
            "claim": "Trace is physical syscall data",
            "verified": bool(trace_validation.get("open_rows", 0) > 0),
            "evidence": json.dumps(trace_validation),
            "gap": "" if trace_validation.get("open_rows", 0) > 0 else "No syscall rows parsed",
            "fix_applied": "Expanded strace syscall set and conversion validation",
        },
        {
            "claim_id": "C3",
            "claim": "Attack input has no direct sample-path leakage",
            "verified": not leakage_scan["has_leakage"],
            "evidence": json.dumps({"scan": leakage_scan, "converter_audit": attack_input_audit}),
            "gap": "" if not leakage_scan["has_leakage"] else "Sample path leakage found",
            "fix_applied": "Leakage scanner + fail-closed conversion",
        },
    ]
    write_claims_matrix(run_dir, cfg, claims_rows)

    timing_breakdown = {
        "defense": cfg.defense,
        "train_runtime_sec": train_runtime,
        "convert_runtime_sec": convert_runtime,
        "trace_attach_overhead_sec": 1.5,
        "oram_read_count": oram_counts["read"],
        "oram_write_count": oram_counts["write"],
    }
    with open(os.path.join(run_dir, "timing_breakdown.json"), "w", encoding="utf-8") as f:
        json.dump(timing_breakdown, f, indent=2)
    with open(os.path.join(run_dir, "mixed_access_report.json"), "w", encoding="utf-8") as f:
        json.dump(mixed_access_report, f, indent=2)

    rows: List[Dict[str, object]] = []
    for visibility in visibilities:
        print(f"  Running attack at visibility={visibility}...")
        attack_start = time.time()
        metrics_json = attack(run_dir, input_csv, visibility)
        attack_runtime = time.time() - attack_start
        metric_row = best_model_metrics(metrics_json)

        row = {
            "run_name": cfg.name,
            "defense": cfg.defense,
            "visibility": visibility,
            "train_runtime_sec": train_runtime,
            "events_retained": metrics_json["num_events"],
            "num_samples": metrics_json["num_samples"],
            "best_model": metrics_json["best_model"],
            "convert_runtime_sec": convert_runtime,
            "attack_runtime_sec": attack_runtime,
            **metric_row,
            "decoys_per_access": cfg.decoys_per_access,
            "prefetch_size": cfg.prefetch_size,
            "release_shuffle_window": cfg.release_shuffle_window,
            "oram_backend": cfg.oram_backend,
            "oram_block_size": cfg.oram_block_size,
        }
        rows.append(row)
        print(f"    AUC={metric_row['best_auc']:.3f}, Accuracy={metric_row['best_accuracy']:.3f}")

    return rows


def write_summary_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


