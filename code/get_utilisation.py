#!/usr/bin/env python3
import subprocess
import time
import sys

def monitor_gpu(samples=60, interval=1):
    """
    Monitor GPU and calculate averages
    
    Args:
        samples: Number of samples to collect (default 60)
        interval: Seconds between samples (default 1)
    """
    gpu_utils = []
    mem_used = []
    mem_total = None
    
    print(f"Collecting {samples} samples (every {interval}s)...")
    print("GPU Util | Memory Used | Memory Total")
    print("-" * 45)
    
    try:
        for i in range(samples):
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, check=True
            )
            
            # Parse the output (handles single GPU)
            line = result.stdout.strip().split('\n')[0]
            util, used, total = map(float, line.split(','))
            
            gpu_utils.append(util)
            mem_used.append(used)
            if mem_total is None:
                mem_total = total
            
            # Print current values
            print(f"{util:6.1f}%  | {used:8.0f} MB | {total:8.0f} MB", end='')
            print(f"  [{i+1}/{samples}]", end='\r')
            sys.stdout.flush()
            
            if i < samples - 1:  # Don't sleep after last sample
                time.sleep(interval)
        
        print()  # New line after progress
        
    except KeyboardInterrupt:
        print("\n\nMonitoring interrupted by user")
        if not gpu_utils:
            return
    except Exception as e:
        print(f"\nError: {e}")
        return
    
    # Calculate statistics
    def median(values):
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        if n % 2 == 0:
            return (sorted_vals[n//2 - 1] + sorted_vals[n//2]) / 2
        else:
            return sorted_vals[n//2]
    
    print("\n" + "=" * 45)
    print("STATISTICS")
    print("=" * 45)
    print(f"Samples collected: {len(gpu_utils)}")
    print(f"\nGPU Utilization:")
    print(f"  Average: {sum(gpu_utils)/len(gpu_utils):6.1f}%")
    print(f"  Median:  {median(gpu_utils):6.1f}%")
    print(f"  Min:     {min(gpu_utils):6.1f}%")
    print(f"  Max:     {max(gpu_utils):6.1f}%")
    print(f"\nMemory Usage:")
    print(f"  Average: {sum(mem_used)/len(mem_used):8.0f} MB ({sum(mem_used)/len(mem_used)/mem_total*100:.1f}%)")
    print(f"  Median:  {median(mem_used):8.0f} MB ({median(mem_used)/mem_total*100:.1f}%)")
    print(f"  Min:     {min(mem_used):8.0f} MB ({min(mem_used)/mem_total*100:.1f}%)")
    print(f"  Max:     {max(mem_used):8.0f} MB ({max(mem_used)/mem_total*100:.1f}%)")
    print(f"  Total:   {mem_total:8.0f} MB")


if __name__ == "__main__":
    # Default: 60 samples (1 minute at 1 second intervals)
    samples = 60
    interval = 1
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        samples = int(sys.argv[1])
    if len(sys.argv) > 2:
        interval = float(sys.argv[2])
    
    print(f"GPU Monitor - Press Ctrl+C to stop early\n")
    monitor_gpu(samples=samples, interval=interval)