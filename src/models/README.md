# Models

Model architecture wrappers used by `training.py`/`testing.py` (`MODEL_NAME` argument — see
[the root README](../../README.md#usage)). `src/models/__init__.py`'s `get_model()` is the single
entry point that maps a `MODEL_NAME` string to a constructor.

## Table of Contents

- [Auto-downloaded weights (most models)](#auto-downloaded-weights-most-models)
- [Manually-downloaded weights (slowfast MViTv2 models)](#manually-downloaded-weights-slowfast-mvitv2-models)

## Auto-downloaded weights (most models)

Every model *except* the ones listed below is a thin wrapper around a `torchvision.models.video`
constructor called with its `*_Weights.KINETICS400*` enum
(`pytorch_r3d.py`, `pytorch_s3d.py`, `pytorch_swin3d.py`, `pytorch_mvit.py`, `pyvision_mvit.py` —
covering `MODEL_NAME`s `S3D`, `R3D_18`, `R(2+1)D_18`, `Swin3D_T`, `Swin3D_S`, `Swin3D_B`,
`MViTv2_S`, `MViTv2_S_e`, `MViTv1_B`). torchvision downloads these automatically the first time
each model is constructed, caching them under `~/.cache/torch/hub/checkpoints/`. Nothing needs to
be downloaded manually for these — just make sure the machine has internet access the first time
you construct one.

## Manually-downloaded weights (slowfast MViTv2 models)

`MODEL_NAME`s `MViTv2_S_16x4`, `MViTv2_S_16x4_e`, `MViTv2_B_32x3`, `MViTv2_B_32x3_r` (defined in
[`og_mvit.py`](og_mvit.py), built on the vendored `mvit/slowfast` reference implementation — see
its [license](mvit/SLOWFAST_LICENSE.md)) are **not** auto-downloaded. `get_model()` does not
forward a weights path for these — `og_mvit.py` hardcodes `pretrain_path` to two fixed paths, so
the checkpoint files must exist at those exact paths, with these exact filenames:

```bash
mkdir -p src/models/mvit/weights
cd src/models/mvit/weights
wget "https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/mvitv2/pysf_video_models/MViTv2_S_16x4_k400_f302660347.pyth"
wget "https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/mvitv2/pysf_video_models/MViTv2_B_32x3_k400_f304025456.pyth"
```

| `MODEL_NAME`(s) | Weights file | Config yaml |
| --- | --- | --- |
| `MViTv2_S_16x4`, `MViTv2_S_16x4_e` | `mvit/weights/MViTv2_S_16x4_k400_f302660347.pyth` (~400MB) | `mvit/configs/MVITv2_S_16x4.yaml` |
| `MViTv2_B_32x3`, `MViTv2_B_32x3_r` | `mvit/weights/MViTv2_B_32x3_k400_f304025456.pyth` (~615MB) | `mvit/configs/MVITv2_B_32x3.yaml` |

Both come from Meta's [SlowFast model zoo](https://github.com/facebookresearch/SlowFast/blob/main/MODEL_ZOO.md)
(the `MVITv2_S_16x4`/`MVITv2_B_32x3` Kinetics-400 rows). The paired config yamls under
`mvit/configs/` are small and tracked in git — they only need restoring if deleted, not
downloading.

You only need these weights on whichever machine actually constructs one of these four models
(i.e. the training server — see the [repo CLAUDE.md](../../CLAUDE.md#development-setup-two-machines)
for the local/server split this project uses). Constructing them without the weights in place
raises `FileNotFoundError`.
