import torch.optim as optim
import torch
import pynvml
import json
import gc
from torch.utils.data import DataLoader
from typing import Optional

# locals
from models import norm_vals, get_model, avail_models
from video_dataset import get_wlasl_info, get_data_set

# from training import setup_data
from run_types import DataInfo, AugInfo, HorizontalFlipConfig

# constants
OUTPUT = "results/saicair/benchmark.json"


# Initialize NVML
pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)


def setup_data(
    model_name: str, num_frames: int = 16, frame_size: int = 224, batch_size: int = 1
):
    # TODO: load from config instead of hardcoding
    norms = norm_vals(model_name)

    train_augs = AugInfo(
        normalise=True,
        norm_dict=norms,
        frame_size_strategy="Random_crop",
        spatial_aug=[HorizontalFlipConfig()],
    )
    test_augs = AugInfo(
        normalise=True,
        norm_dict=norms,
        frame_size_strategy="Centre_crop",
    )

    datainfo = DataInfo(
        num_frames=num_frames,
        frame_size=frame_size,
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
    # test_set, _, _ = get_data_set(
    #     get_wlasl_info('asl100', 'test'),
    #     datainfo
    # )
    # test_loader = DataLoader(
    #     test_set,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=0,
    #     pin_memory=True,
    #     drop_last=False,
    # )
    return train_loader


def get_gpu_stats():
    """Get GPU utilization and memory usage"""
    util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
    mem = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
    return {
        "util": util.gpu,
        "mem_used": mem.used / 1024**2,  # type: ignore
        "mem_total": mem.total / 1024**2,  # type: ignore
        "mem_percent": (mem.used / mem.total) * 100,  # type: ignore
    }


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
):
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

    print(f"Testing arch: {model_name}")
    print(f"Full step (backward + optimizer): {full_step}")

    def run_iter():
        if full_step:
            optimizer.zero_grad()  # type: ignore
            out = model(samp_frames)
            loss = criterion(out, samp_labels)  # type: ignore
            loss.backward()
            optimizer.step()  # type: ignore
        else:
            _ = model(samp_frames)

    # Warm up
    print()
    print(f"Warming up {nwarms} times for {warmup} iterations: ")
    print()
    for i in range(nwarms):
        print(f"warm up: {i + 1} / {nwarms}")
        for _ in range(warmup):
            run_iter()
        torch.cuda.synchronize()

        stats = get_gpu_stats()
        print(f"  GPU Util: {stats['util']}%")
        print(
            f"  GPU Memory: {stats['mem_used']:.0f}/{stats['mem_total']:.0f} MB ({stats['mem_percent']:.1f}%)"
        )
        print()

    # Actual test
    print("Main test")
    print(f"Testing over {iterations} iterations")
    print()
    torch.cuda.reset_peak_memory_stats()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()  # type: ignore
    for _ in range(iterations):
        run_iter()
    end_event.record()  # type: ignore

    torch.cuda.synchronize()

    elapsed = start_event.elapsed_time(end_event) / 1000.0

    stats = get_gpu_stats()
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    # peak_mem = torch.cuda.max_memory_reserved() / 1024**2

    print("\nBenchmark Results:")
    print(f"  Total time: {elapsed:.4f}s")
    print(f"  Average Latency: {elapsed / iterations * 1000:.2f} ms/iter")
    print(f"  Throughput: {iterations / elapsed:.2f} batches/s")
    print(f"  Throughput: {(iterations * batch_size) / elapsed:.2f} samples/s")
    print(f"  GPU Utilization: {stats['util']}%")
    print(
        f"  GPU Memory: {stats['mem_used']:.0f}/{stats['mem_total']:.0f} MB ({stats['mem_percent']:.1f}%)"
    )
    print(f"  Peak GPU Memory: {peak_mem:.0f} MB")

    return {
        "throughput_batches_per_s": iterations / elapsed,
        "throughput_samples_per_s": (iterations * batch_size) / elapsed,
        "latency_ms": (elapsed / iterations) * 1000,
        "utilisation_percent": stats["util"],
        "memory_mb": stats["mem_used"],
        "peak_memory_mb": peak_mem,
    }


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
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(model_name, nc, dropp)

    dataloader = setup_data(model_name, num_frames, frame_size, batch_size)

    model.eval()
    model.to(device)

    samp_batch = next(iter(dataloader))
    samp_frames = samp_batch["frames"]
    samp_frames = samp_frames.to(device)

    print(f"Testing arch: {model_name}")

    # Warm up
    print()
    print(f"Warming up {nwarms} times for {warmup} iterations: ")
    print()
    for i in range(nwarms):
        print(f"warm up: {i + 1} / {nwarms}")
        with torch.no_grad():
            for j in range(warmup):
                _ = model(samp_frames)
        torch.cuda.synchronize()  # Ensure warmup completes

        stats = get_gpu_stats()
        print(f"  GPU Util: {stats['util']}%")
        print(
            f"  GPU Memory: {stats['mem_used']:.0f}/{stats['mem_total']:.0f} MB ({stats['mem_percent']:.1f}%)"
        )
        print()

    # Actual test with CUDA Events for accurate timing
    print("Main test")
    print(f"Testing over {iterations} iterations")
    print()
    torch.cuda.reset_peak_memory_stats()

    # Use CUDA Events for precise GPU timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        # Start recording
        start_event.record()  # type: ignore

        for i in range(iterations):
            _ = model(samp_frames)

        # End recording
        end_event.record()  # type: ignore

    # Wait for everything to finish
    torch.cuda.synchronize()

    # Get elapsed time in milliseconds
    elapsed = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds

    stats = get_gpu_stats()
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2

    print("\nBenchmark Results:")
    print(f"  Total time: {elapsed:.4f}s")
    print(f"  Average Latency: {elapsed / iterations * 1000:.2f} ms/iter")
    print(f"  Throughput: {iterations / elapsed:.2f} batches/s")
    print(f"  Throughput: {(iterations * batch_size) / elapsed:.2f} samples/s")
    print(f"  GPU Utilization: {stats['util']}%")
    print(
        f"  GPU Memory: {stats['mem_used']:.0f}/{stats['mem_total']:.0f} MB ({stats['mem_percent']:.1f}%)"
    )
    print(f"  Peak GPU Memory: {peak_mem:.0f} MB")

    return {
        "throughput_batches_per_s": iterations / elapsed,
        "throughput_samples_per_s": (iterations * batch_size) / elapsed,
        "latency_ms": (elapsed / iterations) * 1000,
        "utilisation_percent": stats["util"],
        "memory_mb": stats["mem_used"],
        "peak_memory_mb": peak_mem,
    }


def full_benchmark():
    av_models = avail_models()
    results = {}

    for arch in av_models:
        results[arch] = {}
        print(f"\n{'=' * 50}")
        print(f"Benchmarking: {arch}")
        print("=" * 50)

        try:
            print("\n>>> Training mode:")
            train_res = benchmark_train(arch)
            results[arch]["train"] = train_res

            # Clear GPU memory between train and inference
            torch.cuda.empty_cache()
            gc.collect()

            print()
            print("-" * 50)
            print()
            print(">>> Inference mode:")
            test_res = benchmark_infer(arch)
            results[arch]["infer"] = test_res

            # Clear GPU memory before next model
            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"ERROR benchmarking {arch}: {e}")
            results[arch]["error"] = str(e)
            # Still clean up memory on error
            torch.cuda.empty_cache()
            gc.collect()

        print("-" * 50)

    # Print summary
    print("\n" + "=" * 50)
    print("BENCHMARK SUMMARY")
    print("=" * 50)
    print(json.dumps(results, indent=4))

    # Save results
    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to {OUTPUT}")


def single_benchmark(
    arch: str,
    num_frames: int = 16,
    frame_size: int = 224,
    train_bs: int = 2,
    test_bs: int = 2,
    iter: int = 100,
    out_path: Optional[str] = None,
    full_step: bool = False,
):
    print(f"\n{'=' * 50}")
    print(f"Benchmarking: {arch}")
    print("=" * 50)

    results = {}

    print("\n>>> Training mode:")
    train_res = benchmark_train(
        arch,
        num_frames,
        frame_size,
        batch_size=train_bs,
        iterations=iter,
        full_step=full_step,
    )
    results["train"] = train_res

    torch.cuda.empty_cache()
    gc.collect()

    print()
    print("-" * 50)
    print()
    print(">>> Inference mode:")
    test_res = benchmark_infer(
        arch, num_frames, frame_size, batch_size=test_bs, iterations=iter
    )
    results["infer"] = test_res

    # Print summary
    print("\n" + "=" * 50)
    print("BENCHMARK SUMMARY")
    print("=" * 50)
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("model", type=str, choices=avail_models())
    parser.add_argument(
        "--num_frames", "-n", type=int, default=16, help="Number of frames for dataset"
    )
    parser.add_argument(
        "--frame_size", "-s", type=int, default=224, help="Frame size for dataset"
    )
    parser.add_argument(
        "--train_batch_size",
        "-t",
        type=int,
        default=1,
        help="Train batch size for DataLoader",
    )
    parser.add_argument(
        "--test_batch_size",
        "-e",
        type=int,
        default=1,
        help="Test batch size for DataLoader",
    )

    parser.add_argument(
        "--iterations",
        "-i",
        type=int,
        help="Number of iterations to benchmark",
        default=100,
    )
    parser.add_argument(
        "--out_path",
        "-o",
        type=str,
        help=f"Path to output, if not: {OUTPUT} ",
        default=OUTPUT,
    )
    parser.add_argument(
        "--full_step",
        "-f",
        action="store_true",
        help="Include backward pass and optimizer step in training benchmark",
    )

    args = parser.parse_args()

    single_benchmark(
        args.model,
        args.num_frames,
        args.frame_size,
        args.train_batch_size,
        args.test_batch_size,
        args.iterations,
        args.out_path,
        full_step=args.full_step,
    )

    pynvml.nvmlShutdown()
