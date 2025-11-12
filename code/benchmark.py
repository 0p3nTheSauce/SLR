from models import norm_vals, get_model, avail_models
from video_dataset import get_data_loader, get_wlasl_info
import torch
import time
import pynvml

from typing import cast

# Initialize NVML
pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

def get_gpu_stats():
    """Get GPU utilization and memory usage"""
    util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
    mem = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
    return {
        'util': util.gpu,
        'mem_used': mem.used / 1024**2,
        'mem_total': mem.total / 1024**2,
        'mem_percent': (mem.used / mem.total) * 100
    }

def benchmark_train(model_name: str, 
                    num_frames: int = 16,
                    frame_size: int = 224,
                    batch_size: int = 2,
                    iterations: int = 200,
                    warmup: int = 20,
                    nwarms: int = 2,
                    dropp: float=0.5,
                    nc: int = 100):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = get_model(model_name, nc, dropp)
    
    norms = norm_vals(model_name)
    dataloader, _, _, _ = get_data_loader(
        norms['mean'], 
        norms['std'], 
        frame_size,
        num_frames,
        get_wlasl_info('asl100', 'train'),
        batch_size=batch_size
    )
    
    model.train()
    model.to(device)
    
    samp_batch = next(iter(dataloader))
    # samp_frames = cast(torch.Tensor, samp_batch['frames'])
    samp_frames = samp_batch['frames']
    samp_frames = samp_frames.to(device) #move to GPU to avoid extra latency
    
    
    print(f"Testing arch: {model_name}")
    
    #warm up
    print()
    print(f"Warming up {nwarms} times for {warmup} iterations: ")
    print()
    for i in range(nwarms):
        print(f"warm up: {i+1} / {nwarms}")
        warmstart = time.perf_counter()
        # with torch.no_grad():
        for i in range(warmup):
            _ = model(samp_frames)
        torch.cuda.synchronize()  # Wait for GPU to finish
        elapsed = time.perf_counter() - warmstart
        
        stats = get_gpu_stats()
        # print(f"  Time: {elapsed:.4f}s ({elapsed/warmup*1000:.2f} ms/iter)")
        print(f"  Time: {elapsed:.4f}s ")
        print(f"  Latency: {elapsed/warmup*1000:.2f} ms/iter")
        print(f"  GPU Util: {stats['util']}%")
        print(f"  GPU Memory: {stats['mem_used']:.0f}/{stats['mem_total']:.0f} MB ({stats['mem_percent']:.1f}%)")
        print()
    
    #actual test
    print("Main test")
    print(f"Testing over {iterations} iterations")
    print()
    torch.cuda.reset_peak_memory_stats()
    main_start = time.perf_counter()
    # with torch.no_grad():
    for i in range(iterations):
        _ = model(samp_frames)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - main_start
    
    stats = get_gpu_stats()
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    
    print("\nBenchmark Results:")
    print(f"  Total time: {elapsed:.4f}s")
    print(f"  Average Latency: {elapsed/iterations*1000:.2f} ms/iter")
    print(f"  Throughput: {iterations/elapsed:.2f} batches/s")
    print(f"  Throughput: {(iterations*batch_size)/elapsed:.2f} samples/s")
    print(f"  GPU Utilization: {stats['util']}%")
    print(f"  GPU Memory: {stats['mem_used']:.0f}/{stats['mem_total']:.0f} MB ({stats['mem_percent']:.1f}%)")
    print(f"  Peak GPU Memory: {peak_mem:.0f} MB")
    
def benchmark_infer(model_name: str, 
                    num_frames: int = 16,
                    frame_size: int = 224,
                    batch_size: int = 2,
                    iterations: int = 200,
                    warmup: int = 20,
                    nwarms: int = 2,
                    dropp: float=0.5,
                    nc: int = 100):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = get_model(model_name, nc, dropp)
    
    norms = norm_vals(model_name)
    dataloader, _, _, _ = get_data_loader(
        norms['mean'], 
        norms['std'], 
        frame_size,
        num_frames,
        get_wlasl_info('asl100', 'train'),
        batch_size=batch_size
    )
    
    model.eval()
    model.to(device)
    
    samp_batch = next(iter(dataloader))
    # samp_frames = cast(torch.Tensor, samp_batch['frames'])
    samp_frames = samp_batch['frames']
    samp_frames = samp_frames.to(device) #move to GPU to avoid extra latency
    
    
    print(f"Testing arch: {model_name}")
    
    #warm up
    print()
    print(f"Warming up {nwarms} times for {warmup} iterations: ")
    print()
    for i in range(nwarms):
        print(f"warm up: {i+1} / {nwarms}")
        warmstart = time.perf_counter()
        with torch.no_grad():
            for i in range(warmup):
                _ = model(samp_frames)
        torch.cuda.synchronize()  # Wait for GPU to finish
        elapsed = time.perf_counter() - warmstart
        
        stats = get_gpu_stats()
        # print(f"  Time: {elapsed:.4f}s ({elapsed/warmup*1000:.2f} ms/iter)")
        print(f"  Time: {elapsed:.4f}s ")
        print(f"  Latency: {elapsed/warmup*1000:.2f} ms/iter")
        print(f"  GPU Util: {stats['util']}%")
        print(f"  GPU Memory: {stats['mem_used']:.0f}/{stats['mem_total']:.0f} MB ({stats['mem_percent']:.1f}%)")
        print()
    
    #actual test
    print("Main test")
    print(f"Testing over {iterations} iterations")
    torch.cuda.reset_peak_memory_stats()
    main_start = time.perf_counter()
    with torch.no_grad():
        for i in range(iterations):
            _ = model(samp_frames)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - main_start
    
    stats = get_gpu_stats()
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    
    print("\nBenchmark Results:")
    print(f"  Total time: {elapsed:.4f}s")
    print(f"  Average Latency: {elapsed/iterations*1000:.2f} ms/iter")
    print(f"  Throughput: {iterations/elapsed:.2f} batches/s")
    print(f"  Throughput: {(iterations*batch_size)/elapsed:.2f} samples/s")
    print(f"  GPU Utilization: {stats['util']}%")
    print(f"  GPU Memory: {stats['mem_used']:.0f}/{stats['mem_total']:.0f} MB ({stats['mem_percent']:.1f}%)")
    print(f"  Peak GPU Memory: {peak_mem:.0f} MB")
    
if __name__ == '__main__':
    
    print("Benchmarking Training:")
    benchmark_train('MViTv2_S')
    print()
    print('-'*20)
    print()
    print("Benchmarkign Inference: ")
    benchmark_infer('MViTv2_S')
    # import torch
    # print(f"PyTorch version: {torch.__version__}")
    # print(f"CUDA available: {torch.cuda.is_available()}")
    # print(f"CUDA version: {torch.version.cuda}")
    
pynvml.nvmlShutdown()