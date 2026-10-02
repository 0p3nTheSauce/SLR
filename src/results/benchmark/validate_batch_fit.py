"""Quick check: which models fit a *physical* training batch of 8 under the real
training setup (AdamW with two param groups, FP32), rather than the benchmark's
SGD step?

The benchmark (benchmark.py) uses SGD with momentum, which keeps one extra buffer
per parameter. AdamW keeps two (exp_avg, exp_avg_sq), so the real training step
needs more memory. This script finds the largest batch size that fits for each
model and frame count under AdamW, so we can say which models actually needed
gradient accumulation.

Usage (from the repo root, same env as training):
    python validate_batch_fit.py                  # all torchvision models, 16 and 32 frames
    python validate_batch_fit.py --models s3d r3d_18 --frames 32
    python validate_batch_fit.py --out batch_fit.json

Notes:
- Uses random input tensors, so data loading is excluded; host-side DataLoader
  overhead does not use GPU memory, so this should not change the result.
- Peak memory is torch.cuda.max_memory_allocated (tensors only). "reserved" is
  what the caching allocator holds, closer to what nvidia-smi shows.
- MViTv2-B (SlowFast) is not in torchvision; add it to MODELS via your own
  model factory if you want it covered too.
"""

import argparse
import gc
import json

import torch
import torch.nn as nn
import torchvision.models.video as tvm

MODELS = {
    "s3d": tvm.s3d,
    "r3d_18": tvm.r3d_18,
    "r2plus1d_18": tvm.r2plus1d_18,
    "swin3d_t": tvm.swin3d_t,
    "swin3d_s": tvm.swin3d_s,
    "swin3d_b": tvm.swin3d_b,
    "mvit_v1_b": tvm.mvit_v1_b,
    "mvit_v2_s": tvm.mvit_v2_s,
}
# MViT (torchvision) only accepts 16 frames without positional-embedding interpolation.
FIXED_FRAMES = {"mvit_v1_b": 16, "mvit_v2_s": 16}


def replace_head(model: nn.Module, num_classes: int) -> tuple[list, list]:
    """Swap the classifier for num_classes outputs; return (backbone, head) params."""
    head = None
    for name in ("fc", "head", "classifier"):
        mod = getattr(model, name, None)
        if mod is None:
            continue
        if isinstance(mod, nn.Linear):
            new = nn.Linear(mod.in_features, num_classes)
            setattr(model, name, new)
            head = new
        elif isinstance(mod, nn.Sequential):
            # last Linear (MViT head) or last Conv3d (S3D classifier)
            for i in reversed(range(len(mod))):
                layer = mod[i]
                if isinstance(layer, nn.Linear):
                    mod[i] = nn.Linear(layer.in_features, num_classes)
                    head = mod[i]
                    break
                if isinstance(layer, nn.Conv3d):
                    mod[i] = nn.Conv3d(layer.in_channels, num_classes, kernel_size=1, bias=True)
                    head = mod[i]
                    break
        if head is not None:
            break
    if head is None:
        raise RuntimeError(f"Could not find classifier on {type(model).__name__}")
    head_ids = {id(p) for p in head.parameters()}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    return backbone, list(head.parameters())


def try_step(name: str, frames: int, batch_size: int, num_classes: int, steps: int) -> dict:
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    model = MODELS[name](weights=None)
    backbone, head = replace_head(model, num_classes)
    model = model.cuda().train()
    opt = torch.optim.AdamW(
        [{"params": backbone, "lr": 1e-5}, {"params": head, "lr": 1e-3}], weight_decay=1e-4
    )
    loss_fn = nn.CrossEntropyLoss()
    x = torch.randn(batch_size, 3, frames, 224, 224, device="cuda")
    y = torch.randint(0, num_classes, (batch_size,), device="cuda")
    try:
        for _ in range(steps):  # >1 step so AdamW state is allocated
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
        torch.cuda.synchronize()
        result = {
            "status": "ok",
            "peak_alloc_gb": torch.cuda.max_memory_allocated() / 1024**3,
            "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
        }
    except torch.cuda.OutOfMemoryError:
        result = {"status": "OOM"}
    finally:
        del model, opt, x, y
        torch.cuda.empty_cache()
        gc.collect()
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--frames", nargs="+", type=int, default=[16, 32])
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--num-classes", type=int, default=100)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = False  # match training (deterministic settings)
    print(f"GPU: {torch.cuda.get_device_name()}, "
          f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    rows = []
    for name in args.models:
        frame_list = [FIXED_FRAMES[name]] if name in FIXED_FRAMES else args.frames
        for frames in frame_list:
            max_ok = 0
            for bs in sorted(args.batch_sizes):
                r = try_step(name, frames, bs, args.num_classes, args.steps)
                rows.append({"model": name, "frames": frames, "batch_size": bs, **r})
                extra = f"{r['peak_alloc_gb']:.2f} GB" if r["status"] == "ok" else ""
                print(f"{name:12s} {frames:3d}f bs={bs:<3d} {r['status']:4s} {extra}")
                if r["status"] != "ok":
                    break
                max_ok = bs
            fits8 = "yes" if max_ok >= 8 else f"no (max {max_ok})"
            print(f"  -> physical batch 8 fits: {fits8}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
