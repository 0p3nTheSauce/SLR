# Conda environments

The repo uses two conda envs. Their specs live in the repo root.

- **`wlasl`** (`wlasl_gpu.yml`): the GPU env, used for training, testing, the Que server,
  preprocessing and the pytest suite. Last regenerated in commit `88023d1` (2026-09-14) from the
  live env with `conda env export --no-builds`, with the self-referential `slr==0.1.0` pip entry
  removed: `slr` is installed with `pip install -e .`, not as a pinned dependency (the convention
  set by commit `25aff3e`).
- **`wlasl_cpu`** (`wlasl_cpu.yml`): CPU-only, for non-MViT work (notebooks, plotting, the Que
  client). It deliberately omits the MViT/slowfast dependencies (`simplejson` etc.), so the pytest
  suite can't be collected under it. The yml had been deleted in commit `d287a94` while the env was
  still in use; it was regenerated from the live env on 2026-09-12.
- `pyproject.toml` declares the `slr` package (`src*`) with no pinned dependency list. The
  dependencies come from the conda env files.

## History: env cleanup after the detectron2 MViT removal (2026-09-12)

Commit `532a648` ("removed pretraining setup") deleted the detectron2-based image-classification
MViT code (`src/models/detectron_mvit.py`, `mvit/detectron2_cp/`, `mvit/image_classification/`,
plus `mvirted_mae.py`/`sep_mvit_bert.py`/`sepmay.py`). Investigation to prune the now-unused
`detectron2` dependency found:

- The still-active video MViT path (`og_mvit.py` → `mvit/slowfast/models/video_model_builder.py`)
  had an **unconditional** `from detectron2.layers import ROIAlign` in
  `mvit/slowfast/models/head_helper.py`, used only by `ResNetRoIHead` — an AVA-detection-style
  head that nothing in this project instantiates (confirmed: no reference to `ROIAlign`,
  `ResNetRoIHead`, `ContrastiveModel`, `MaskMViT`, `build_model`/`MODEL_REGISTRY`, or the PTV*
  classes anywhere outside the vendored `slowfast` dir itself). Fixed by moving that import inside
  `ResNetRoIHead.__init__` (lazy) so `detectron2` is only required if that unused head is ever
  built.
- Deleted further dead vendored files, confirmed unreferenced by tracing actual module imports
  during real model construction (`sys.modules` diff) plus a repo-wide grep for their symbols:
  `mvit/slowfast/models/{ptv_model_builder,contrastive,masked,losses,optimizer}.py` and
  `mvit/slowfast/utils/{ava_eval_helper,benchmark,bn_helper,lr_policy,meters,metrics,multigrid}.py`.
  Several of these were already broken/orphaned pre-refactor (e.g. `ava_eval_helper.py` imports a
  `slowfast.datasets` module that was never vendored here) — not just unused, but non-functional.
  `models/mvit/slowfast/models/__init__.py` no longer imports `ContrastiveModel`/`MaskMViT`/PTV*.
- Uninstalled `detectron2` and its now-orphaned dependency `pycocotools` from the live `wlasl` env
  (confirmed via `pip show --Required-by`: nothing else depended on either). Packages that
  *looked* detectron2-adjacent but are required by other tools in the env were **kept**: `cv2`/
  `matplotlib`/`six`/`pyparsing`/`kiwisolver`/`cycler`/`python-dateutil` (used directly by
  `preprocess.py`/`utils.py`/`visualise*.py`, and required by `seaborn`/`ultralytics`/`pandas`),
  and `fvcore`/`iopath`/`yacs`/`portalocker`/`simplejson`/`pytorchvideo`/`av` (still genuinely
  imported by the kept vendored files, e.g. `operators.py`/`batchnorm_helper.py` import from
  `pytorchvideo`). Verified `src.models` still imports and `MVITv2_S_16x4_basic` still constructs
  with `detectron2` fully uninstalled.
- Regenerated `wlasl_gpu.yml` from the live (now-cleaned) `wlasl` env via
  `conda env export --no-builds`, stripping the self-referential `slr==0.1.0` pip entry (matches
  the convention set by commit `25aff3e`, which deliberately excluded `slr` from the tracked yml
  since it's installed via `pip install -e .`, not as a pinned dep).
- Cleaned up orphaned `__pycache__` artifacts left over from the earlier file deletions
  (`mvit/detectron2_cp/__pycache__/*`, etc.) and removed the now-empty `detectron2_cp/` directory.

**One pre-existing, unrelated issue surfaced during this investigation, since fixed (2026-09-13):**

1. ~~`og_mvit.py`'s `CONF_PATH_16x4`/`CONF_PATH_32x3` point to
   `mvit/configs/MVITv2_S_16x4.yaml`/`MVITv2_B_32x3.yaml`, which no longer exist~~ — these two
   config files were actually deleted by the same commit (`532a648`, "removed pretraining setup")
   that did the detectron2-related pretraining removal, alongside the truly-unused 2D/test configs
   (`MVITv2_B_2D.yaml`, `MVITv2_L_40x3_test.yaml`, `MVITv2_S_2D.yaml`, `MVITv2_T_2D.yaml`) — but
   unlike those, `MVITv2_S_16x4.yaml`/`MVITv2_B_32x3.yaml` are still required by the active
   `MViTv2_S_16x4*`/`MViTv2_B_32x3*` classes in `og_mvit.py`. Restored both from `532a648^`.

**A package-drift issue found during this cleanup, since resolved (checked 2026-09-26):**

1. At the time, `nvidia-ml-py` (which provides `pynvml`, imported unconditionally by
   `src/benchmark.py`) was missing from the live `wlasl` env, as were `fairscale`, `fastapi`,
   `starlette`, `uvicorn`, `sweeps`, `ninja`, `jsonref` and `tomli-w`. The tracked yml had drifted
   from the real env before the cleanup started. All of these are now installed and listed in
   `wlasl_gpu.yml`, and `benchmark.py` imports when run as a script from `src/`.
