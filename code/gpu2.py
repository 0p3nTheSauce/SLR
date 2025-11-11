import time
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import subprocess
import json

class InferenceBenchmark:
    def __init__(self, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.warmup_iters = 50
        self.num_iterations = 100
        
    def get_gpu_utilization(self) -> Dict:
        """Get current GPU utilization using nvidia-smi"""
        try:
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, check=True)
            
            if result.stdout.strip():
                values = result.stdout.strip().split(',')
                return {
                    'gpu_util': int(values[0]),
                    'memory_util': int(values[1]),
                    'memory_used': int(values[2]),
                    'memory_total': int(values[3])
                }
        except:
            pass
        return {}
    
    def monitor_gpu_during_inference(self, model: torch.nn.Module, dataloader, duration: int = 10):
        """Monitor GPU utilization during inference"""
        model.eval()
        max_utilization = {'gpu_util': 0, 'memory_util': 0, 'memory_used': 0}
        data_iter = iter(dataloader)
        
        end_time = time.time() + duration
        
        with torch.no_grad():
            while time.time() < end_time:
                try:
                    batch = next(data_iter)
                    inputs, targets = batch["frames"], batch["label_num"]
                    
                    if torch.is_tensor(inputs):
                        inputs = inputs.to(self.device)
                    
                    # Run inference
                    _ = model(inputs)
                    
                    # Check GPU utilization
                    util = self.get_gpu_utilization()
                    max_utilization['gpu_util'] = max(max_utilization['gpu_util'], util.get('gpu_util', 0))
                    max_utilization['memory_util'] = max(max_utilization['memory_util'], util.get('memory_util', 0))
                    max_utilization['memory_used'] = max(max_utilization['memory_used'], util.get('memory_used', 0))
                    
                except StopIteration:
                    data_iter = iter(dataloader)
                except Exception as e:
                    print(f"Error during GPU monitoring: {e}")
                    break
        
        return max_utilization

    def benchmark_inference(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        num_iterations: int = None,
        warmup_iters: int = None,
        monitor_gpu: bool = False,
        gpu_monitor_duration: int = 5
    ) -> Dict[str, float]:
        """
        Comprehensive inference benchmarking
        
        Args:
            model: PyTorch model
            dataloader: DataLoader providing input data
            num_iterations: Number of inference iterations
            warmup_iters: Number of warm-up iterations
            monitor_gpu: Whether to monitor GPU utilization
            gpu_monitor_duration: Duration for GPU monitoring in seconds
        
        Returns:
            Dictionary with benchmark results
        """
        # Setup parameters
        num_iterations = num_iterations or self.num_iterations
        warmup_iters = warmup_iters or self.warmup_iters
        
        model.eval()
        model.to(self.device)
        
        # Ensure model is in eval mode and gradients are disabled
        torch.set_grad_enabled(False)
        
        # Warm-up phase
        print(f"Running {warmup_iters} warm-up iterations...")
        self._run_inference_phase(model, dataloader, warmup_iters, "Warm-up")
        
        # Clear GPU cache
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Benchmarking phase
        print(f"Running {num_iterations} benchmark iterations...")
        latencies, batch_sizes = self._run_inference_phase(
            model, dataloader, num_iterations, "Benchmark", track_timing=True
        )
        
        # GPU monitoring (optional)
        gpu_stats = {}
        if monitor_gpu and self.device == 'cuda':
            print(f"Monitoring GPU utilization for {gpu_monitor_duration} seconds...")
            gpu_stats = self.monitor_gpu_during_inference(model, dataloader, gpu_monitor_duration)
        
        # Calculate statistics
        stats = self._calculate_statistics(latencies, batch_sizes)
        stats.update(gpu_stats)
        
        return stats
    
    def _run_inference_phase(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        num_iters: int,
        phase_name: str,
        track_timing: bool = False
    ) -> Tuple[List[float], List[int]]:
        """Run inference phase (warm-up or benchmark)"""
        latencies = []
        batch_sizes = []
        data_iter = iter(dataloader)
        
        with torch.no_grad():
            for i in range(num_iters):
                try:
                    # Get batch
                    batch = next(data_iter)
                    
                    inputs = batch['frames']
                    
                    # Move to device
                    
                    inputs = inputs.to(self.device)
                    
                    
                    # Get batch size
                    
                    batch_size = inputs.size(0)

                    
                    # Synchronize before timing
                    if self.device == 'cuda' and track_timing:
                        torch.cuda.synchronize()
                    
                    # Time inference
                    start_time = time.perf_counter()
                    
                    # Forward pass
                    outputs = model(inputs)
                    
                    # Synchronize after inference
                    if self.device == 'cuda' and track_timing:
                        torch.cuda.synchronize()
                    
                    end_time = time.perf_counter()
                    
                    if track_timing:
                        latency = (end_time - start_time) * 1000  # Convert to milliseconds
                        latencies.append(latency)
                        batch_sizes.append(batch_size)
                    
                    # Progress reporting
                    if (i + 1) % max(1, num_iters // 10) == 0:
                        print(f"{phase_name}: {i + 1}/{num_iters} iterations completed")
                        
                except StopIteration:
                    # Reset dataloader if we run out of data
                    data_iter = iter(dataloader)
                    continue
                except Exception as e:
                    print(f"Error during {phase_name} iteration {i}: {e}")
                    continue
        
        return latencies, batch_sizes
    
    def _calculate_statistics(self, latencies: List[float], batch_sizes: List[int]) -> Dict[str, float]:
        """Calculate performance statistics from latencies and batch sizes"""
        if not latencies:
            return {}
        
        latencies = np.array(latencies)
        batch_sizes = np.array(batch_sizes)
        
        total_items = np.sum(batch_sizes)
        total_time_ms = np.sum(latencies)
        
        # Throughput calculations
        throughput_items_per_sec = total_items / (total_time_ms / 1000)
        throughput_batches_per_sec = len(latencies) / (total_time_ms / 1000)
        
        # Latency statistics (in milliseconds)
        latency_stats = {
            'mean_latency_ms': np.mean(latencies),
            'median_latency_ms': np.median(latencies),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'std_latency_ms': np.std(latencies),
        }
        
        # Throughput statistics
        throughput_stats = {
            'throughput_items_per_sec': throughput_items_per_sec,
            'throughput_batches_per_sec': throughput_batches_per_sec,
            'mean_batch_size': np.mean(batch_sizes),
            'total_items_processed': total_items,
            'total_inference_time_ms': total_time_ms,
        }
        
        return {**latency_stats, **throughput_stats}

def benchmark_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = None,
    num_iterations: int = 100,
    warmup_iters: int = 50,
    monitor_gpu: bool = True
) -> Dict[str, float]:
    """
    Convenience function for quick benchmarking
    
    Args:
        model: PyTorch model to benchmark
        dataloader: DataLoader for inference data
        device: Device to run on ('cuda' or 'cpu')
        num_iterations: Number of benchmark iterations
        warmup_iters: Number of warm-up iterations
        monitor_gpu: Whether to monitor GPU utilization
    
    Returns:
        Dictionary with benchmark results
    """
    benchmark = InferenceBenchmark(device)
    return benchmark.benchmark_inference(
        model=model,
        dataloader=dataloader,
        num_iterations=num_iterations,
        warmup_iters=warmup_iters,
        monitor_gpu=monitor_gpu
    )

def print_benchmark_results(results: Dict[str, float]):
    """Print benchmark results in a formatted way"""
    print("\n" + "="*80)
    print("INFERENCE BENCHMARK RESULTS")
    print("="*80)
    
    print("\n📊 LATENCY STATISTICS (ms):")
    print(f"  Mean:        {results.get('mean_latency_ms', 0):.2f} ms")
    print(f"  Median:      {results.get('median_latency_ms', 0):.2f} ms")
    print(f"  Min:         {results.get('min_latency_ms', 0):.2f} ms")
    print(f"  Max:         {results.get('max_latency_ms', 0):.2f} ms")
    print(f"  P95:         {results.get('p95_latency_ms', 0):.2f} ms")
    print(f"  P99:         {results.get('p99_latency_ms', 0):.2f} ms")
    print(f"  Std Dev:     {results.get('std_latency_ms', 0):.2f} ms")
    
    print("\n🚀 THROUGHPUT STATISTICS:")
    print(f"  Throughput:  {results.get('throughput_items_per_sec', 0):.2f} items/sec")
    print(f"  Batch Rate:  {results.get('throughput_batches_per_sec', 0):.2f} batches/sec")
    print(f"  Avg Batch:   {results.get('mean_batch_size', 0):.1f} items")
    print(f"  Total:       {results.get('total_items_processed', 0):.0f} items")
    
    if 'gpu_util' in results:
        print("\n🎮 GPU UTILIZATION:")
        print(f"  Max GPU Util:     {results.get('gpu_util', 0)}%")
        print(f"  Max Memory Util:  {results.get('memory_util', 0)}%")
        print(f"  Max Memory Used:  {results.get('memory_used', 0)} MB")

# Example usage
if __name__ == "__main__":
    # Example with a dummy model and dataloader
    # import torch.nn as nn
    
    # class DummyModel(nn.Module):
    #     def __init__(self):
    #         super().__init__()
    #         self.layer = nn.Linear(1000, 1000)
            
    #     def forward(self, x):
    #         return self.layer(x)
    
    # # Create dummy data
    # dummy_loader = torch.utils.data.DataLoader(
    #     torch.randn(100, 3, 224, 224),  # 100 samples, 3 channels, 224x224
    #     batch_size=16
    # )
    
    # model = DummyModel()
    
    from models import avail_models, get_model, norm_vals
    from video_dataset import get_wlasl_info, get_data_loader
    avm = avail_models()
    idx = 7
    
    arch = avm[idx]
    normvals = norm_vals(arch)
    dloader, nc, _, _ = get_data_loader(
        normvals['mean'], 
        normvals['std'], 
        frame_size=224, 
        num_frames=16,
        set_info=get_wlasl_info('asl100', 'test'),
        batch_size=2)
    
    
   
    model = get_model(arch, nc, 0)
    
    # Run benchmark
    results = benchmark_model(
        model=model,
        dataloader=dloader,
        num_iterations=50,
        warmup_iters=10,
        monitor_gpu=True
    )
    print(f"{idx+1} / {len(avm)}")
    print(f"Benchmarking for: {arch}")
    print_benchmark_results(results)
    