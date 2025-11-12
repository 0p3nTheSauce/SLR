from models import norm_vals, get_model, avail_models
from video_dataset import get_data_loader, get_wlasl_info
import torch
import time

from typing import cast

def benchmark_train(model_name: str, 
                    num_frames: int = 16,
                    frame_size: int = 224,
                    batch_size: int = 2,
                    iterations: int = 40,
                    warmup: int = 10,
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
    
    model.to(device)
    
    samp_batch = next(iter(dataloader))
    # samp_frames = cast(torch.Tensor, samp_batch['frames'])
    samp_frames = samp_batch['frames']
    samp_frames.to(device) #move to GPU to avoid extra latency
    
    
    print(f"Testing arch: {model_name}")
    print(f"Warming up {nwarms} times for {warmup} iterations: ")
    
    print()
    for i in range(nwarms):
        print(f"warm up: {i+1} / {nwarms}")
        warmstart = time.perf_counter()
        for i in range(warmup):
            _ = model(samp_frames)
        print(f"Time taken: {warmstart - time.perf_counter()}")
        print() 
            
    
if __name__ == '__main__':
    # benchmark_train('S3D')
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")