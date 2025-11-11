import subprocess
import time
import json
import argparse
from typing import Dict, List, Tuple

def get_gpu_utilization(interval: int = 1) -> Dict[str, List[Dict]]:
    """
    Get current GPU utilization using nvidia-smi
    
    Returns:
        Dictionary containing GPU utilization data
    """
    try:
        # Run nvidia-smi with JSON output
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, check=True)
        
        gpu_data = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = [part.strip() for part in line.split(',')]
                if len(parts) >= 6:
                    gpu_data.append({
                        'index': int(parts[0]),
                        'name': parts[1],
                        'gpu_util': int(parts[2]),
                        'memory_util': int(parts[3]),
                        'memory_used': int(parts[4]),
                        'memory_total': int(parts[5])
                    })
        
        return {'gpus': gpu_data}
    
    except subprocess.CalledProcessError as e:
        print(f"Error running nvidia-smi: {e}")
        return {'gpus': []}
    except FileNotFoundError:
        print("Error: nvidia-smi not found. Make sure NVIDIA drivers are installed.")
        return {'gpus': []}

def monitor_gpu_utilization(monitor_interval: int = 10, sample_interval: int = 1) -> Dict[str, Dict]:
    """
    Monitor GPU utilization over a specified interval and return maximum values
    
    Args:
        monitor_interval: Total monitoring time in seconds
        sample_interval: Time between samples in seconds
    
    Returns:
        Dictionary containing maximum utilization values for each GPU
    """
    if sample_interval <= 0:
        raise ValueError("Sample interval must be positive")
    
    max_utilization = {}
    sample_count = 0
    
    print(f"Monitoring GPU utilization for {monitor_interval} seconds...")
    print("Press Ctrl+C to stop early")
    
    start_time = time.time()
    
    try:
        while (time.time() - start_time) < monitor_interval:
            data = get_gpu_utilization()
            
            for gpu in data['gpus']:
                gpu_index = gpu['index']
                
                if gpu_index not in max_utilization:
                    max_utilization[gpu_index] = {
                        'name': gpu['name'],
                        'max_gpu_util': 0,
                        'max_memory_util': 0,
                        'max_memory_used': 0,
                        'memory_total': gpu['memory_total'],
                        'samples': 0
                    }
                
                # Update maximum values
                max_utilization[gpu_index]['max_gpu_util'] = max(
                    max_utilization[gpu_index]['max_gpu_util'], 
                    gpu['gpu_util']
                )
                max_utilization[gpu_index]['max_memory_util'] = max(
                    max_utilization[gpu_index]['max_memory_util'], 
                    gpu['memory_util']
                )
                max_utilization[gpu_index]['max_memory_used'] = max(
                    max_utilization[gpu_index]['max_memory_used'], 
                    gpu['memory_used']
                )
                max_utilization[gpu_index]['samples'] += 1
            
            sample_count += 1
            time.sleep(sample_interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring interrupted by user")
    
    # Calculate final statistics
    for gpu_index in max_utilization:
        max_utilization[gpu_index]['monitor_duration'] = time.time() - start_time
        max_utilization[gpu_index]['sample_count'] = max_utilization[gpu_index]['samples']
        del max_utilization[gpu_index]['samples']
    
    return max_utilization

def print_results(max_utilization: Dict[str, Dict]):
    """Print the monitoring results in a formatted way"""
    print("\n" + "="*80)
    print("GPU UTILIZATION MONITORING RESULTS")
    print("="*80)
    
    for gpu_index, stats in max_utilization.items():
        print(f"\nGPU {gpu_index} - {stats['name']}:")
        print(f"  Max GPU Utilization:     {stats['max_gpu_util']}%")
        print(f"  Max Memory Utilization:  {stats['max_memory_util']}%")
        print(f"  Max Memory Used:         {stats['max_memory_used']} MB / {stats['memory_total']} MB")
        print(f"  Monitoring Duration:     {stats['monitor_duration']:.1f} seconds")
        print(f"  Samples Collected:       {stats['sample_count']}")

def main():
    parser = argparse.ArgumentParser(description='Monitor GPU utilization and return maximum values')
    parser.add_argument('--interval', '-i', type=int, default=10,
                       help='Monitoring interval in seconds (default: 10)')
    parser.add_argument('--sample-interval', '-s', type=float, default=1.0,
                       help='Time between samples in seconds (default: 1.0)')
    
    args = parser.parse_args()
    
    if args.interval <= 0:
        print("Error: Monitoring interval must be positive")
        return
    
    if args.sample_interval <= 0:
        print("Error: Sample interval must be positive")
        return
    
    try:
        max_utilization = monitor_gpu_utilization(
            monitor_interval=args.interval,
            sample_interval=args.sample_interval
        )
        
        print_results(max_utilization)
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()