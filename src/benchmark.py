"""
Accurate GPU benchmarking for model architectures.

Key fixes vs. the previous version (which used a single
`nvmlDeviceGetUtilizationRates()` snapshot call right after each loop):

1. GPU utilization/memory are sampled continuously on a background thread
   for the *entire* timed region and averaged. A single NVML snapshot only
   reflects the last ~1/6s-1s sample window (per NVIDIA's own docs), so it
   massively under/over-represents utilization over a multi-second run.
2. `torch.cuda.synchronize()` is called immediately before starting the
   CUDA event timer, not just after — otherwise leftover async work from
   warmup can leak into the "start" timestamp.
3. The whole benchmark (warmup + timed loop) is repeated over several
   independent trials, and results are reported as mean +/- std, so a
   single noisy run can't be mistaken for a stable number.
4. `optimizer.zero_grad(set_to_none=True)` avoids a memset every step,
   matching how training loops are actually written/timed in practice.
5. Peak memory is reset per-trial so trials don't contaminate each other.

Note on GPU clock throttling: for maximally reproducible numbers you'd
also want to lock the GPU clock (`sudo nvidia-smi -lgc <freq>`) so thermal
throttling doesn't add noise between trials. That requires root and is
environment-specific, so it's left as a manual step rather than baked in
here — see the README note printed at the bottom of --help.
"""

import argparse
import gc
import json
import statistics
import threading
from dataclasses import dataclass, field

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

pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)


# --------------------------------------------------------------------------- #
# GPU monitoring: continuous background sampling instead of a single snapshot
# --------------------------------------------------------------------------- #
class GPUMonitor:
    """Samples GPU utilization/memory on a background thread for as long as
    it's running, so stats reflect the whole measured region rather than
    whatever NVML's internal sample window happened to catch at one instant.
    """

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
            # Region was too short to get a single sample; fall back to one
            # instantaneous reading rather than reporting nothing.
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
# Data setup (unchanged from before)
# --------------------------------------------------------------------------- #
def setup_data(
    model_name: str, num_frames: int = 16, frame_size: int = 224, batch_size: int = 1
):
    norms = norm_vals(model_name)
    # Re-validate into run_types.NormDict explicitly rather than passing the
    # returned instance straight through. If this module and the rest of the
    # codebase import `models`/`run_types` via different paths (e.g. bare
    # `models` here vs `src.models` elsewhere), Python treats them as
    # separate classes with separate identities, so pydantic's isinstance
    # check on a foreign NormDict instance can fail even though the data is
    # identical. from_attributes=True reads the fields off the returned
    # object regardless of its concrete class.
    norm_dict = NormDict.model_validate(norms, from_attributes=True)

    # AugInfo (with strict_size=True, the default) requires at least one
    # temporal sampler and one spatial crop config to pass validation — it
    # derives target_length/frame_size from these, not from kwargs passed
    # directly to AugInfo/DataInfo. There's no `frame_size_strategy` field.
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

    # target_length/frame_size on DataInfo are derived from train_augs/test_augs
    # by its own validator — don't pass them in separately, they'd be ignored
    # or overwritten anyway.
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
# Core timed run — single trial. Called multiple times per benchmark.
# --------------------------------------------------------------------------- #
def _time_region(run_iter, iterations: int, monitor: GPUMonitor) -> TrialResult:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    monitor.start()
    start_event.record()  # type: ignore
    for _ in range(iterations):
        run_iter()
    end_event.record()  # type: ignore
    torch.cuda.synchronize()
    gpu_stats = monitor.stop()

    elapsed_s = start_event.elapsed_time(end_event) / 1000.0
    gpu_stats["mem_max_mb"] = torch.cuda.max_memory_allocated() / 1024**2
    return TrialResult(elapsed_s=elapsed_s, iterations=iterations, batch_size=-1, gpu_stats=gpu_stats)


# --------------------------------------------------------------------------- #
# torch.profiler cross-check
# --------------------------------------------------------------------------- #
def _event_time_us(evt, kind: str) -> float:
    """Get total/self device time in microseconds from a profiler event,
    tolerating the attribute rename across torch versions: older releases
    used `cuda_time_total`/`self_cuda_time_total`; newer ones (profiler is
    no longer CUDA-only) use `device_time_total`/`self_device_time_total`.
    Falls back to whichever attribute actually exists on this event.
    """
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
    """Pick whichever sort key torch.profiler's table() supports on this
    version — mirrors the cuda_time_total/device_time_total rename."""
    sample = next(iter(key_avgs), None)
    if sample is not None and hasattr(sample, "device_time_total"):
        return "device_time_total"
    return "cuda_time_total"


def profile_region(run_iter, iterations: int, label: str) -> dict:
    """Independently measure the same region with torch.profiler and compare
    against CUDA-event timing. The two methods measure different things —
    CUDA events time wall-clock GPU-stream duration; the profiler attributes
    time to individual ops via its own tracing — so agreement is a useful
    sanity check, and disagreement can point at CPU-bound stalls, dataloader
    gaps, or op-launch overhead that a bare CUDA-event total would hide.
    """
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
        # Fallback: sum top-level self time across all ops (may double-count
        # nested calls, but better than nothing if the record_function scope
        # wasn't captured as its own event on this torch version).
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


def benchmark_train(
    model_name: str,
    num_frames: int = 16,
    frame_size: int = 224,
    batch_size: int = 2,
    iterations: int = 200,
    warmup: int = 20,
    nwarms: int = 2,
    dropp: float = 0.5,
    nc: int = 100,
    full_step: bool = True,
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

    print(f"Running {trials} trials of {iterations} iterations each...")
    trial_results = []
    for t in range(trials):
        res = _time_region(run_iter, iterations, monitor)
        res.batch_size = batch_size
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
    num_frames: int = 16,
    frame_size: int = 224,
    batch_size: int = 2,
    iterations: int = 200,
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

    print(f"Running {trials} trials of {iterations} iterations each...")
    trial_results = []
    for t in range(trials):
        res = _time_region(run_iter, iterations, monitor)
        res.batch_size = batch_size
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


def full_benchmark(trials: int = 5, profile_iterations: int = 0):
    av_models = avail_models()
    results = {"gpu_info": get_gpu_static_info()}

    for arch in av_models:
        results[arch] = {}
        print(f"\n{'=' * 50}\nBenchmarking: {arch}\n{'=' * 50}")

        try:
            results[arch]["train"] = benchmark_train(
                arch, trials=trials, profile_iterations=profile_iterations
            )
            torch.cuda.empty_cache()
            gc.collect()

            results[arch]["infer"] = benchmark_infer(
                arch, trials=trials, profile_iterations=profile_iterations
            )
            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:  # noqa: BLE001
            print(f"ERROR benchmarking {arch}: {e}")
            results[arch]["error"] = str(e)
            torch.cuda.empty_cache()
            gc.collect()

    print("\n" + "=" * 50 + "\nBENCHMARK SUMMARY\n" + "=" * 50)
    print(json.dumps(results, indent=4))

    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {OUTPUT}")


def single_benchmark(
    arch: str,
    num_frames: int = 16,
    frame_size: int = 224,
    train_bs: int = 2,
    test_bs: int = 2,
    iterations: int = 100,
    trials: int = 5,
    out_path: str | None = None,
    full_step: bool = False,
    profile_iterations: int = 0,
):
    print(f"\n{'=' * 50}\nBenchmarking: {arch}\n{'=' * 50}")

    results = {"gpu_info": get_gpu_static_info()}
    results["train"] = benchmark_train(
        arch, num_frames, frame_size, batch_size=train_bs,
        iterations=iterations, full_step=full_step, trials=trials,
        profile_iterations=profile_iterations,
    )
    torch.cuda.empty_cache()
    gc.collect()

    results["infer"] = benchmark_infer(
        arch, num_frames, frame_size, batch_size=test_bs,
        iterations=iterations, trials=trials,
        profile_iterations=profile_iterations,
    )

    print("\n" + "=" * 50 + "\nBENCHMARK SUMMARY\n" + "=" * 50)
    print(json.dumps(results, indent=4))

    try:
        with open(OUTPUT, "r") as f:
            alldata = json.load(f)
    except FileNotFoundError:
        alldata = {}
    alldata[arch] = results

    if out_path:
        with open(out_path, "w") as f:
            json.dump(alldata, f, indent=4)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        epilog=(
            "Note: for maximally reproducible numbers, consider locking the "
            "GPU clock first (`sudo nvidia-smi -lgc <freq>`, reset with "
            "`-rgc`) to remove thermal-throttling noise between trials."
        ),
    )
    parser.add_argument("model", type=str, choices=avail_models())
    parser.add_argument(
        "--num_frames", "-n", type=int, default=16,
        help="Number of frames for dataset (default: %(default)s)",
    )
    parser.add_argument(
        "--frame_size", "-s", type=int, default=224,
        help="Frame size for dataset (default: %(default)s)",
    )
    parser.add_argument(
        "--train_batch_size", "-t", type=int, default=1,
        help="Train batch size for DataLoader (default: %(default)s)",
    )
    parser.add_argument(
        "--test_batch_size", "-e", type=int, default=1,
        help="Test batch size for DataLoader (default: %(default)s)",
    )
    parser.add_argument(
        "--iterations", "-i", type=int, default=100,
        help="Number of iterations per trial (default: %(default)s)",
    )
    parser.add_argument(
        "--trials", "-r", type=int, default=5,
        help="Number of independent timed trials to average over (default: %(default)s)",
    )
    parser.add_argument(
        "--out_path", "-o", type=str, default=OUTPUT,
        help="Path to output JSON (default: %(default)s)",
    )
    parser.add_argument(
        "--full_step", "-f", action="store_true",
        help="Include backward pass and optimizer step in training benchmark",
    )
    parser.add_argument(
        "--profile_iterations", "-p", type=int, default=0,
        help="If >0, also run this many iterations under torch.profiler and "
        "print a comparison against the CUDA-event timing (default: %(default)s, disabled)",
    )

    args = parser.parse_args()

    single_benchmark(
        args.model,
        args.num_frames,
        args.frame_size,
        args.train_batch_size,
        args.test_batch_size,
        args.iterations,
        args.trials,
        args.out_path,
        full_step=args.full_step,
        profile_iterations=args.profile_iterations,
    )

    pynvml.nvmlShutdown()