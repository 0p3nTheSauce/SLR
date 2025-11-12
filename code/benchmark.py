from models import norm_vals, get_model, avail_models
from video_dataset import get_data_loader, get_wlasl_info
import torch
import time
import pynvml
import json
import gc
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
    samp_frames = samp_batch['frames']
    samp_frames = samp_frames.to(device)
    
    print(f"Testing arch: {model_name}")
    
    # Warm up
    print()
    print(f"Warming up {nwarms} times for {warmup} iterations: ")
    print()
    for i in range(nwarms):
        print(f"warm up: {i+1} / {nwarms}")
        warmstart = time.perf_counter()
        for j in range(warmup):
            _ = model(samp_frames)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - warmstart
        
        stats = get_gpu_stats()
        print(f"  Time: {elapsed:.4f}s ")
        print(f"  Latency: {elapsed/warmup*1000:.2f} ms/iter")
        print(f"  GPU Util: {stats['util']}%")
        print(f"  GPU Memory: {stats['mem_used']:.0f}/{stats['mem_total']:.0f} MB ({stats['mem_percent']:.1f}%)")
        print()
    
    # Actual test
    print("Main test")
    print(f"Testing over {iterations} iterations")
    print()
    torch.cuda.reset_peak_memory_stats()
    main_start = time.perf_counter()
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
    
    return {
        'throughput_batches_per_s':  iterations/elapsed,
        'throughput_samples_per_s': (iterations*batch_size)/elapsed,
        'latency_ms': (elapsed/iterations)*1000,
        'utilisation_percent': stats['util'],
        'memory_mb': stats['mem_used'],
        'peak_memory_mb': peak_mem
    }
    
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
    samp_frames = samp_batch['frames']
    samp_frames = samp_frames.to(device)
    
    print(f"Testing arch: {model_name}")
    
    # Warm up
    print()
    print(f"Warming up {nwarms} times for {warmup} iterations: ")
    print()
    for i in range(nwarms):
        print(f"warm up: {i+1} / {nwarms}")
        warmstart = time.perf_counter()
        with torch.no_grad():
            for j in range(warmup):
                _ = model(samp_frames)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - warmstart
        
        stats = get_gpu_stats()
        print(f"  Time: {elapsed:.4f}s ")
        print(f"  Latency: {elapsed/warmup*1000:.2f} ms/iter")
        print(f"  GPU Util: {stats['util']}%")
        print(f"  GPU Memory: {stats['mem_used']:.0f}/{stats['mem_total']:.0f} MB ({stats['mem_percent']:.1f}%)")
        print()
    
    # Actual test
    print("Main test")
    print(f"Testing over {iterations} iterations")
    print()
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
    
    return {
        'throughput_batches_per_s':  iterations/elapsed,
        'throughput_samples_per_s': (iterations*batch_size)/elapsed,
        'latency_ms': (elapsed/iterations)*1000,
        'utilisation_percent': stats['util'],
        'memory_mb': stats['mem_used'],
        'peak_memory_mb': peak_mem
    }
    


def full_benchmark():
    av_models = avail_models()
    results = {}
    
    for arch in av_models:
        results[arch] = {}  
        print(f"\n{'='*50}")
        print(f"Benchmarking: {arch}")
        print('='*50)
        
        try:
            print("\n>>> Training mode:")
            train_res = benchmark_train(arch)
            results[arch]['train'] = train_res
            
            # Clear GPU memory between train and inference
            torch.cuda.empty_cache()
            gc.collect()
            
            print()
            print('-'*50)
            print()
            print(">>> Inference mode:")
            test_res = benchmark_infer(arch)
            results[arch]['infer'] = test_res
            
            # Clear GPU memory before next model
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"ERROR benchmarking {arch}: {e}")
            results[arch]['error'] = str(e)
            # Still clean up memory on error
            torch.cuda.empty_cache()
            gc.collect()
        
        print('-'*50)
    
    # Print summary
    print("\n" + "="*50)
    print("BENCHMARK SUMMARY")
    print("="*50)
    print(json.dumps(results, indent=4))
    
    # Save results
    with open('results/benchmark.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\nResults saved to results/benchmark.json")
    
def single_benchmark(arch:str):
    print(f"\n{'='*50}")
    print(f"Benchmarking: {arch}")
    print('='*50)
    
    results = {}
    
    print("\n>>> Training mode:")
    train_res = benchmark_train(arch)
    results['train'] = train_res
    
    torch.cuda.empty_cache()
    gc.collect()
    
    
    print()
    print('-'*50)
    print()
    print(">>> Inference mode:")
    test_res = benchmark_infer(arch)
    results['infer'] = test_res
    
    # Print summary
    print("\n" + "="*50)
    print("BENCHMARK SUMMARY")
    print("="*50)
    print(json.dumps(results, indent=4))
    

if __name__ == '__main__':
    single_benchmark('MViTv2_S')
    # benchmark_train("Swin3D_B")
    
    pynvml.nvmlShutdown()