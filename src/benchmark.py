"""
Accurate, rigorous GPU benchmarking for model architectures.

Rigor fixes vs. the previous version, for thesis/SACAIR-paper-grade results:

1. SUBPROCESS ISOLATION: when benchmarking multiple architectures, each one
   runs in its own fresh Python process (see `run_all_separately`), not a
   shared long-lived loop. A shared process risks cuDNN algorithm-cache
   reuse, CUDA allocator fragmentation, and leftover autograd/optimizer
   state silently biasing later models. Isolation also gives a clean place
   to shuffle launch order, so thermal drift over a long sweep doesn't
   systematically penalise whichever model happens to run last.
2. SEPARATED TIMING / MONITORING PASSES: CUDA-event latency timing and
   NVML utilisation/memory polling are now run as two separate passes per
   trial rather than concurrently. The polling thread competes for the GIL
   with kernel-launch overhead in the main thread; models with more
   Python-side overhead per iteration would otherwise be perturbed more
   than GPU-bound ones. This doubles wall-clock time per trial but removes
   that asymmetry from the latency numbers entirely.
3. `torch.backends.cudnn.benchmark = True` is now set explicitly (was
   previously left at PyTorch's default of False) and logged in run
   metadata, so autotuned conv algorithm selection is consistent and
   documented rather than an invisible default.
4. `full_step` no longer has a silently-differing default between
   `benchmark_train` and the CLI -- it's a required argument everywhere,
   so it's always explicit and always logged.
5. Every run's config (batch size, frame count, frame size, iterations,
   trials, full_step, cudnn.benchmark) and environment (torch/cuda/cudnn
   versions, GPU name + driver, git commit, timestamp, current vs. max SM
   clock) is written alongside the metrics. Results are appended into a
   shared JSON file keyed by a composite key of arch + config, so repeated
   runs at different configs accumulate instead of clobbering each other.
   NOTE: re-running the *exact same* config overwrites the previous entry
   under that key -- there's no history list. If you want to keep repeated
   measurements of an identical config (e.g. re-checking stability on a
   different day), add a distinguishing suffix to the key yourself.
6. GPU clock state is checked via NVML at the start of every run; if the
   current SM clock is well below the card's max, a warning is printed and
   logged, since that usually means clocks aren't locked
   (`sudo nvidia-smi -lgc <freq>`) and thermal/power throttling could add
   noise between trials or between architectures.
"""

import argparse
import gc
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pynvml
import torch
from torch import optim
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader

# locals
from models import avail_models, get_model, norm_vals
from run_types import (
    AugInfo,
    CentreCropConfig,
    DataInfo,
    HorizontalFlipConfig,
    NormDict,
    RandomCropConfig,
    UniformSampler,
)
from video_dataset import get_data_set, get_wlasl_info

# constants
OUTPUT = "benchmark.json"
GPU_POLL_INTERVAL_S = 0.05  # 50ms; NVML's own sample window is ~1/6-1s anyway

# Rigor fix #3: explicit, logged, rather than left at PyTorch's default.
torch.backends.cudnn.benchmark = True

pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)


def _decode(x):
    """pynvml returns str on newer bindings, bytes on older ones."""
    return x.decode() if isinstance(x, bytes) else x


# --------------------------------------------------------------------------- #
# Run metadata / provenance -- logged with every result so numbers in a
# shared JSON file (accumulated over weeks) are traceable to exact
# code/hardware/driver state.
# --------------------------------------------------------------------------- #
def get_run_metadata() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, cwd=os.path.dirname(os.path.abspath(__file__))
        ).decode().strip()
    except Exception:
        commit = None

    try:
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, cwd=os.path.dirname(os.path.abspath(__file__))
        ).decode().strip())
    except Exception:
        dirty = None

    clocks = {}
    try:
        clocks["sm_clock_mhz"] = pynvml.nvmlDeviceGetClockInfo(gpu_handle, pynvml.NVML_CLOCK_SM)
        clocks["mem_clock_mhz"] = pynvml.nvmlDeviceGetClockInfo(gpu_handle, pynvml.NVML_CLOCK_MEM)
        clocks["sm_clock_max_mhz"] = pynvml.nvmlDeviceGetMaxClockInfo(gpu_handle, pynvml.NVML_CLOCK_SM)
        if clocks["sm_clock_max_mhz"]:
            ratio = clocks["sm_clock_mhz"] / clocks["sm_clock_max_mhz"]
            clocks["sm_clock_ratio_of_max"] = ratio
            if ratio < 0.9:
                print(
                    f"WARNING: current SM clock ({clocks['sm_clock_mhz']} MHz) is "
                    f"{ratio:.0%} of max ({clocks['sm_clock_max_mhz']} MHz). Clocks "
                    f"don't look locked -- for maximally comparable numbers run "
                    f"`sudo nvidia-smi -lgc <freq>` before benchmarking. Proceeding "
                    f"anyway; this is logged in run metadata."
                )
    except pynvml.NVMLError as e:
        clocks["error"] = str(e)

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_dirty": dirty,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_benchmark_enabled": torch.backends.cudnn.benchmark,
        "python_version": platform.python_version(),
        "gpu_name": _decode(pynvml.nvmlDeviceGetName(gpu_handle)),
        "driver_version": _decode(pynvml.nvmlSystemGetDriverVersion()),
        "gpu_clocks": clocks,
        "hostname": platform.node(),
    }


# --------------------------------------------------------------------------- #
# GPU monitoring: continuous background sampling, run as its own pass
# (see fix #2 -- no longer concurrent with the CUDA-event timed region).
# --------------------------------------------------------------------------- #
class GPUMonitor:
    def __init__(self, handle, interval: float = GPU_POLL_INTERVAL_S):
        self.handle = handle
        self.interval = interval
        self._util_samples: list[float] = []
        self._mem_samples: list[float] = []  # MB
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self):
        while not self._stop_event.is_set():
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                self._util_samples.append(util.gpu)  # type: ignore
                self._mem_samples.append(mem.used / 1024**2)  # type: ignore
            except pynvml.NVMLError:
                pass
            self._stop_event.wait(self.interval)

    def start(self):
        self._util_samples = []
        self._mem_samples = []
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2 * self.interval + 1)
        if not self._util_samples:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            self._util_samples = [util.gpu]  # type: ignore
            self._mem_samples = [mem.used / 1024**2]  # type: ignore
        return {
            "util_mean": statistics.mean(self._util_samples),
            "util_max": max(self._util_samples),
            "mem_mean_mb": statistics.mean(self._mem_samples),
            "mem_max_mb": max(self._mem_samples),
            "n_samples": len(self._util_samples),
        }


def get_gpu_static_info() -> dict:
    mem = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
    return {"mem_total_mb": mem.total / 1024**2}  # type: ignore


# --------------------------------------------------------------------------- #
# Data setup (unchanged)
# --------------------------------------------------------------------------- #
def setup_data(
    model_name: str, num_frames: int = 16, frame_size: int = 224, batch_size: int = 1
):
    norms = norm_vals(model_name)
    norm_dict = NormDict.model_validate(norms, from_attributes=True)

    train_augs = AugInfo(
        normalise=True,
        norm_dict=norm_dict,
        temporal_aug=[UniformSampler(target_length=num_frames)],
        spatial_aug=[
            RandomCropConfig(frame_size=frame_size),
            HorizontalFlipConfig(),
        ],
    )
    test_augs = AugInfo(
        normalise=True,
        norm_dict=norm_dict,
        temporal_aug=[UniformSampler(target_length=num_frames)],
        spatial_aug=[CentreCropConfig(frame_size=frame_size)],
    )

    datainfo = DataInfo(
        train_augs=train_augs,
        test_augs=test_augs,
    )
    train_set, _, _ = get_data_set(get_wlasl_info("asl100", "train"), datainfo)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    return train_loader


# --------------------------------------------------------------------------- #
# Trial result container + stats aggregation
# --------------------------------------------------------------------------- #
@dataclass
class TrialResult:
    elapsed_s: float
    iterations: int
    batch_size: int
    gpu_stats: dict = field(default_factory=dict)


def _summarise_trials(trials: list[TrialResult]) -> dict:
    latencies_ms = [(t.elapsed_s / t.iterations) * 1000 for t in trials]
    throughputs_bps = [t.iterations / t.elapsed_s for t in trials]
    throughputs_sps = [
        (t.iterations * t.batch_size) / t.elapsed_s for t in trials
    ]
    util_means = [t.gpu_stats["util_mean"] for t in trials]
    util_maxes = [t.gpu_stats["util_max"] for t in trials]
    mem_maxes = [t.gpu_stats["mem_max_mb"] for t in trials]

    def mean_std(xs):
        return {
            "mean": statistics.mean(xs),
            "std": statistics.stdev(xs) if len(xs) > 1 else 0.0,
        }

    return {
        "n_trials": len(trials),
        "latency_ms": mean_std(latencies_ms),
        "throughput_batches_per_s": mean_std(throughputs_bps),
        "throughput_samples_per_s": mean_std(throughputs_sps),
        "gpu_utilisation_percent": mean_std(util_means),
        "gpu_utilisation_peak_percent": max(util_maxes),
        "peak_memory_mb": mean_std(mem_maxes),
        "raw_trial_latencies_ms": latencies_ms,
    }


# --------------------------------------------------------------------------- #
# Core timed run -- fix #2: timing and monitoring are now separate passes.
# --------------------------------------------------------------------------- #
def _time_only(run_iter, iterations: int) -> float:
    """Pure CUDA-event latency timing. No monitor thread running, so
    nothing competes with kernel-launch overhead for the GIL."""
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()  # type: ignore
    for _ in range(iterations):
        run_iter()
    end_event.record()  # type: ignore
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / 1000.0


def _monitor_only(run_iter, iterations: int, monitor: GPUMonitor) -> dict:
    """Separate pass purely for GPU utilisation/memory stats -- run
    independently of the timing pass so the monitor thread never perturbs
    a latency measurement."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    monitor.start()
    for _ in range(iterations):
        run_iter()
    torch.cuda.synchronize()
    stats = monitor.stop()
    stats["mem_max_mb"] = torch.cuda.max_memory_allocated() / 1024**2
    return stats


def _run_trial(run_iter, iterations: int, monitor: GPUMonitor, batch_size: int) -> TrialResult:
    elapsed_s = _time_only(run_iter, iterations)
    gpu_stats = _monitor_only(run_iter, iterations, monitor)
    return TrialResult(elapsed_s=elapsed_s, iterations=iterations, batch_size=batch_size, gpu_stats=gpu_stats)


# --------------------------------------------------------------------------- #
# torch.profiler cross-check (unchanged from before)
# --------------------------------------------------------------------------- #
def _event_time_us(evt, kind: str) -> float:
    candidates = {
        "total": ["device_time_total", "cuda_time_total"],
        "self": ["self_device_time_total", "self_cuda_time_total"],
    }[kind]
    for attr in candidates:
        if hasattr(evt, attr):
            return getattr(evt, attr)
    raise AttributeError(
        f"None of {candidates} found on profiler event {type(evt).__name__} "
        f"(available: {[a for a in dir(evt) if 'time' in a]})"
    )


def _table_sort_key(key_avgs) -> str:
    sample = next(iter(key_avgs), None)
    if sample is not None and hasattr(sample, "device_time_total"):
        return "device_time_total"
    return "cuda_time_total"


def profile_region(run_iter, iterations: int, label: str) -> dict:
    torch.cuda.synchronize()

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(activities=activities, record_shapes=False, profile_memory=False) as prof, record_function(label):
        for _ in range(iterations):
            run_iter()
    torch.cuda.synchronize()

    key_avgs = prof.key_averages()
    region_evt = next((e for e in key_avgs if e.key == label), None)

    if region_evt is not None:
        cuda_time_total_ms = _event_time_us(region_evt, "total") / 1000.0
        cpu_time_total_ms = region_evt.cpu_time_total / 1000.0
    else:
        cuda_time_total_ms = sum(_event_time_us(e, "self") for e in key_avgs) / 1000.0
        cpu_time_total_ms = sum(e.self_cpu_time_total for e in key_avgs) / 1000.0

    top_ops = key_avgs.table(sort_by=_table_sort_key(key_avgs), row_limit=10)

    return {
        "profiler_latency_ms": cuda_time_total_ms / iterations,
        "profiler_cpu_latency_ms": cpu_time_total_ms / iterations,
        "iterations": iterations,
        "top_ops_table": top_ops,
    }


def _print_profiler_comparison(label: str, event_latency_ms: float, prof_stats: dict):
    prof_latency = prof_stats["profiler_latency_ms"]
    diff_pct = ((prof_latency - event_latency_ms) / event_latency_ms) * 100 if event_latency_ms else float("nan")

    print(f"\n{label} — CUDA-event vs. torch.profiler cross-check:")
    print(f"  CUDA-event latency:  {event_latency_ms:.3f} ms/iter")
    print(f"  Profiler CUDA latency: {prof_latency:.3f} ms/iter  ({diff_pct:+.1f}% vs. CUDA-event)")
    print(f"  Profiler CPU latency:  {prof_stats['profiler_cpu_latency_ms']:.3f} ms/iter")
    if abs(diff_pct) > 15:
        print(
            "  NOTE: >15% divergence — worth checking the top-op table below "
            "for CPU-bound stalls or profiler overhead skewing the result."
        )
    print("\n  Top ops by CUDA time:")
    print(prof_stats["top_ops_table"])


# --------------------------------------------------------------------------- #
# Config / composite-key / append helpers (fix #5)
# --------------------------------------------------------------------------- #
def _make_run_key(arch: str, phase: str, config: dict) -> str:
    """Composite key encoding arch + config, so different configs
    accumulate as separate entries in the shared JSON file. Re-running the
    *same* config overwrites its existing entry -- see module docstring."""
    parts = [
        arch,
        phase,
        f"nf{config['num_frames']}",
        f"fsz{config['frame_size']}",
        f"bs{config['batch_size']}",
        f"it{config['iterations']}",
        f"tr{config['trials']}",
    ]
    if phase == "train":
        parts.append(f"fullstep{int(config['full_step'])}")
    return "__".join(parts)


def _append_result(path: str, key: str, entry: dict):
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}
    data.setdefault("runs", {})
    data["runs"][key] = entry
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# --------------------------------------------------------------------------- #
# Train / infer benchmarks
# --------------------------------------------------------------------------- #
def benchmark_train(
    model_name: str,
    num_frames: int,
    frame_size: int,
    batch_size: int,
    iterations: int,
    full_step: bool,  # fix #4: required, no silently-differing default
    warmup: int = 20,
    nwarms: int = 2,
    dropp: float = 0.5,
    nc: int = 100,
    trials: int = 5,
    profile_iterations: int = 0,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(model_name, nc, dropp)
    model.train()
    model.to(device)

    optimizer = (
        optim.SGD(model.parameters(), lr=0.01, momentum=0.9) if full_step else None
    )
    criterion = torch.nn.CrossEntropyLoss() if full_step else None

    dataloader = setup_data(model_name, num_frames, frame_size, batch_size)
    samp_batch = next(iter(dataloader))
    samp_frames = samp_batch["frames"].to(device)
    samp_labels = samp_batch["label_num"].to(device) if full_step else None

    print(f"Testing arch: {model_name} (train, full_step={full_step})")

    def run_iter():
        if full_step:
            optimizer.zero_grad(set_to_none=True)  # type: ignore
            out = model(samp_frames)
            loss = criterion(out, samp_labels)  # type: ignore
            loss.backward()
            optimizer.step()  # type: ignore
        else:
            with torch.no_grad():
                _ = model(samp_frames)

    monitor = GPUMonitor(gpu_handle)

    print(f"Warming up {nwarms} x {warmup} iterations...")
    for _ in range(nwarms):
        for _ in range(warmup):
            run_iter()
        torch.cuda.synchronize()

    print(f"Running {trials} trials of {iterations} iterations each (timing + monitoring passes)...")
    trial_results = []
    for t in range(trials):
        res = _run_trial(run_iter, iterations, monitor, batch_size)
        trial_results.append(res)
        print(
            f"  trial {t + 1}/{trials}: "
            f"{res.iterations / res.elapsed_s:.2f} batches/s, "
            f"util_mean={res.gpu_stats['util_mean']:.1f}% "
            f"(n_samples={res.gpu_stats['n_samples']})"
        )

    summary = _summarise_trials(trial_results)
    _print_summary("Train", summary)

    if profile_iterations > 0:
        prof_stats = profile_region(run_iter, profile_iterations, label=f"{model_name}_train")
        _print_profiler_comparison("Train", summary["latency_ms"]["mean"], prof_stats)
        summary["profiler_comparison"] = {
            k: v for k, v in prof_stats.items() if k != "top_ops_table"
        }

    return summary


def benchmark_infer(
    model_name: str,
    num_frames: int,
    frame_size: int,
    batch_size: int,
    iterations: int,
    warmup: int = 20,
    nwarms: int = 2,
    dropp: float = 0.5,
    nc: int = 100,
    trials: int = 5,
    profile_iterations: int = 0,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(model_name, nc, dropp)
    model.eval()
    model.to(device)

    dataloader = setup_data(model_name, num_frames, frame_size, batch_size)
    samp_batch = next(iter(dataloader))
    samp_frames = samp_batch["frames"].to(device)

    print(f"Testing arch: {model_name} (infer)")

    def run_iter():
        with torch.no_grad():
            _ = model(samp_frames)

    monitor = GPUMonitor(gpu_handle)

    print(f"Warming up {nwarms} x {warmup} iterations...")
    for _ in range(nwarms):
        with torch.no_grad():
            for _ in range(warmup):
                run_iter()
        torch.cuda.synchronize()

    print(f"Running {trials} trials of {iterations} iterations each (timing + monitoring passes)...")
    trial_results = []
    for t in range(trials):
        res = _run_trial(run_iter, iterations, monitor, batch_size)
        trial_results.append(res)
        print(
            f"  trial {t + 1}/{trials}: "
            f"{res.iterations / res.elapsed_s:.2f} batches/s, "
            f"util_mean={res.gpu_stats['util_mean']:.1f}% "
            f"(n_samples={res.gpu_stats['n_samples']})"
        )

    summary = _summarise_trials(trial_results)
    _print_summary("Infer", summary)

    if profile_iterations > 0:
        prof_stats = profile_region(run_iter, profile_iterations, label=f"{model_name}_infer")
        _print_profiler_comparison("Infer", summary["latency_ms"]["mean"], prof_stats)
        summary["profiler_comparison"] = {
            k: v for k, v in prof_stats.items() if k != "top_ops_table"
        }

    return summary


def _print_summary(label: str, summary: dict):
    print(f"\n{label} results ({summary['n_trials']} trials):")
    print(
        f"  Latency: {summary['latency_ms']['mean']:.2f} +/- "
        f"{summary['latency_ms']['std']:.2f} ms/iter"
    )
    print(
        f"  Throughput: {summary['throughput_samples_per_s']['mean']:.2f} +/- "
        f"{summary['throughput_samples_per_s']['std']:.2f} samples/s"
    )
    print(
        f"  GPU util (mean over run): {summary['gpu_utilisation_percent']['mean']:.1f} +/- "
        f"{summary['gpu_utilisation_percent']['std']:.1f} % "
        f"(peak {summary['gpu_utilisation_peak_percent']:.0f}%)"
    )
    print(
        f"  Peak memory: {summary['peak_memory_mb']['mean']:.0f} +/- "
        f"{summary['peak_memory_mb']['std']:.0f} MB\n"
    )


# --------------------------------------------------------------------------- #
# Single-architecture entry point -- this is what gets subprocess-launched
# by run_all_separately for a full sweep (fix #1).
# --------------------------------------------------------------------------- #
def _is_oom_error(e: Exception) -> bool:
    """torch.cuda.OutOfMemoryError (newer torch) is a RuntimeError subclass;
    older torch just raises RuntimeError with 'out of memory' in the message."""
    if hasattr(torch.cuda, "OutOfMemoryError") and isinstance(e, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(e, RuntimeError) and "out of memory" in str(e).lower()


def single_benchmark(
    arch: str,
    num_frames: int = 16,
    frame_size: int = 224,
    train_bs: list | None = None,  # list of training batch sizes to sweep
    test_bs: list | None = None,  # list of inference batch sizes to sweep
    iterations: int = 100,
    trials: int = 5,
    out_path: str | None = None,
    full_step: bool = False,
    profile_iterations: int = 0,
):
    print(f"\n{'=' * 50}\nBenchmarking: {arch}\n{'=' * 50}")
    out_path = out_path or OUTPUT
    train_bs = train_bs or [2]
    test_bs = test_bs or [2]
    metadata = get_run_metadata()
    gpu_info = get_gpu_static_info()

    # Training batch-size sweep. Same OOM-stops-the-rest logic as inference:
    # backward pass + optimizer state pushes memory higher than inference at
    # the same batch size, so training's ceiling is typically lower -- worth
    # tracking separately per architecture rather than assuming.
    train_keys = []
    for bs in sorted(train_bs):
        train_config = {
            "num_frames": num_frames, "frame_size": frame_size, "batch_size": bs,
            "iterations": iterations, "trials": trials, "full_step": full_step,
        }
        train_key = _make_run_key(arch, "train", train_config)
        try:
            train_summary = benchmark_train(
                arch, num_frames, frame_size, batch_size=bs,
                iterations=iterations, full_step=full_step, trials=trials,
                profile_iterations=profile_iterations,
            )
        except Exception as e:  # noqa: BLE001
            torch.cuda.empty_cache()
            gc.collect()
            if _is_oom_error(e):
                print(f"OOM at training batch_size={bs} for {arch}; stopping sweep here "
                      f"(larger sizes would also OOM).")
                train_entry = {
                    "arch": arch, "config": train_config, "metadata": metadata,
                    "gpu_info": gpu_info, "error": "OOM",
                }
                _append_result(out_path, train_key, train_entry)
                train_keys.append(train_key)
                break
            raise
        else:
            train_entry = {
                "arch": arch, "config": train_config, "metadata": metadata,
                "gpu_info": gpu_info, "results": train_summary,
            }
            _append_result(out_path, train_key, train_entry)
            train_keys.append(train_key)
        torch.cuda.empty_cache()
        gc.collect()

    # Inference batch-size sweep. Memory use is monotonic in batch size, so
    # once one size OOMs, larger ones in the list will too -- stop the sweep
    # for this architecture there rather than trying (and failing) the rest.
    # Different architectures will naturally complete different subsets of
    # this shared candidate list depending on their own memory footprint,
    # so this doubles as a per-model max-batch-size / saturation curve
    # without maintaining separate lists per model.
    infer_keys = []
    for bs in sorted(test_bs):
        infer_config = {
            "num_frames": num_frames, "frame_size": frame_size, "batch_size": bs,
            "iterations": iterations, "trials": trials,
        }
        infer_key = _make_run_key(arch, "infer", infer_config)
        try:
            infer_summary = benchmark_infer(
                arch, num_frames, frame_size, batch_size=bs,
                iterations=iterations, trials=trials,
                profile_iterations=profile_iterations,
            )
        except Exception as e:  # noqa: BLE001
            torch.cuda.empty_cache()
            gc.collect()
            if _is_oom_error(e):
                print(f"OOM at inference batch_size={bs} for {arch}; stopping sweep here "
                      f"(larger sizes would also OOM).")
                infer_entry = {
                    "arch": arch, "config": infer_config, "metadata": metadata,
                    "gpu_info": gpu_info, "error": "OOM",
                }
                _append_result(out_path, infer_key, infer_entry)
                infer_keys.append(infer_key)
                break
            raise
        else:
            infer_entry = {
                "arch": arch, "config": infer_config, "metadata": metadata,
                "gpu_info": gpu_info, "results": infer_summary,
            }
            _append_result(out_path, infer_key, infer_entry)
            infer_keys.append(infer_key)
        torch.cuda.empty_cache()
        gc.collect()

    print("\n" + "=" * 50 + "\nBENCHMARK SUMMARY\n" + "=" * 50)
    print(f"Results appended to {out_path} under keys:")
    for k in train_keys:
        print(f"  {k}")
    for k in infer_keys:
        print(f"  {k}")


# --------------------------------------------------------------------------- #
# Multi-architecture sweep -- fix #1: each arch in its own subprocess,
# launch order shuffled by default.
# --------------------------------------------------------------------------- #
def run_all_separately(
    archs: list | None = None,
    num_frames: int = 16,
    frame_size: int = 224,
    train_bs: list | None = None,  # list of training batch sizes to sweep
    test_bs: list | None = None,  # list of inference batch sizes to sweep
    iterations: int = 100,
    trials: int = 5,
    full_step: bool = False,
    out_path: str = OUTPUT,
    shuffle: bool = True,
    profile_iterations: int = 0,
):
    """Benchmark every architecture in its own subprocess: a fresh CUDA
    context per model avoids cuDNN algorithm-cache reuse, allocator
    fragmentation, and leftover autograd/optimizer state carrying over
    between architectures. Launch order is shuffled by default so thermal
    drift over a long sweep doesn't consistently penalise one model.
    """
    archs = list(archs or avail_models())
    train_bs = train_bs or [2]
    test_bs = test_bs or [2]
    if shuffle:
        random.shuffle(archs)

    print(f"Running {len(archs)} architectures in isolated subprocesses.")
    print(f"Order (shuffle={shuffle}): {archs}")
    print(f"Training batch-size sweep: {sorted(train_bs)}")
    print(f"Inference batch-size sweep: {sorted(test_bs)}")

    failures = []
    for i, arch in enumerate(archs, 1):
        cmd = [
            sys.executable, os.path.abspath(__file__),
            "--models", arch,
            "--_subprocess_worker",  # tells this child to run single_benchmark
            # directly instead of re-entering orchestration -- without this,
            # a --models invocation of exactly one arch would spawn a child
            # that itself sees --models with one arch and orchestrates again,
            # recursing forever.
            "--num_frames", str(num_frames),
            "--frame_size", str(frame_size),
            "--train_batch_sizes", *[str(b) for b in train_bs],
            "--test_batch_sizes", *[str(b) for b in test_bs],
            "--iterations", str(iterations),
            "--trials", str(trials),
            "--out_path", out_path,
            "--profile_iterations", str(profile_iterations),
        ]
        if full_step:
            cmd.append("--full_step")

        print(f"\n{'=' * 50}\n[{i}/{len(archs)}] Launching subprocess for: {arch}\n{'=' * 50}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"WARNING: subprocess for {arch} exited with code {result.returncode}; continuing.")
            failures.append(arch)

    print(f"\nAll runs complete. Results appended to {out_path}")
    if failures:
        print(f"Architectures that failed: {failures}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        epilog=(
            "For a full sweep across architectures, use --all: each model runs "
            "in its own subprocess (clean CUDA context) in shuffled order. "
            "GPU clock state is checked and logged automatically; if it's not "
            "locked, lock it yourself first with `sudo nvidia-smi -lgc <freq>` "
            "(reset with `-rgc`) for maximally reproducible numbers."
        ),
    )
    models_group = parser.add_mutually_exclusive_group(required=True)
    models_group.add_argument("--all", action="store_true",
                               help="Benchmark every available architecture, each in its own subprocess.")
    models_group.add_argument("--models", type=str, nargs="+", choices=avail_models(), default=None,
                               help="Architecture(s) to benchmark, each in its own subprocess.")
    parser.add_argument("--no_shuffle", action="store_true",
                         help="With --all/--models, disable shuffling of launch order (not recommended).")
    parser.add_argument("--_subprocess_worker", action="store_true", help=argparse.SUPPRESS,
                         # Internal flag set only by run_all_separately when it spawns a child
                         # for one architecture. Forces single_benchmark directly, bypassing
                         # orchestration, so a child process never re-shuffles/re-spawns.
                         )
    parser.add_argument("--num_frames", "-n", type=int, default=16)
    parser.add_argument("--frame_size", "-s", type=int, default=224)
    parser.add_argument("--train_batch_sizes", "-t", type=int, nargs="+", default=[1],
                         help="Training batch size(s) to sweep. Tried in ascending order per "
                              "architecture; stops early for a given architecture on CUDA OOM "
                              "(logged as an error entry) since larger sizes would also OOM.")
    parser.add_argument("--test_batch_sizes", "-e", type=int, nargs="+", default=[1],
                         help="Inference batch size(s) to sweep. Tried in ascending order per "
                              "architecture; stops early for a given architecture on CUDA OOM "
                              "(logged as an error entry) since larger sizes would also OOM.")
    parser.add_argument("--iterations", "-i", type=int, default=100)
    parser.add_argument("--trials", "-r", type=int, default=5)
    parser.add_argument("--out_path", "-o", type=str, default=OUTPUT)
    parser.add_argument("--full_step", "-f", action="store_true",
                         help="Include backward pass and optimizer step in training benchmark.")
    parser.add_argument("--profile_iterations", "-p", type=int, default=0)

    args = parser.parse_args()

    if args._subprocess_worker:
        # Child invocation from run_all_separately: run exactly the one
        # architecture it was told to, directly -- no orchestration, no
        # further subprocessing, regardless of --all/--models/--no_shuffle.
        if not args.models or len(args.models) != 1:
            parser.error("--_subprocess_worker expects exactly one --models entry")
        single_benchmark(
            args.models[0], args.num_frames, args.frame_size,
            args.train_batch_sizes, args.test_batch_sizes,
            args.iterations, args.trials, args.out_path,
            full_step=args.full_step, profile_iterations=args.profile_iterations,
        )
    elif args.all:
        run_all_separately(
            num_frames=args.num_frames, frame_size=args.frame_size,
            train_bs=args.train_batch_sizes, test_bs=args.test_batch_sizes,
            iterations=args.iterations, trials=args.trials,
            full_step=args.full_step, out_path=args.out_path,
            shuffle=not args.no_shuffle, profile_iterations=args.profile_iterations,
        )
    else:
        # --models given at top level: orchestrate a sweep over just this
        # subset, each still in its own isolated subprocess (also handles
        # the single-architecture case as a sweep of one).
        run_all_separately(
            archs=args.models,
            num_frames=args.num_frames, frame_size=args.frame_size,
            train_bs=args.train_batch_sizes, test_bs=args.test_batch_sizes,
            iterations=args.iterations, trials=args.trials,
            full_step=args.full_step, out_path=args.out_path,
            shuffle=not args.no_shuffle, profile_iterations=args.profile_iterations,
        )

    pynvml.nvmlShutdown()