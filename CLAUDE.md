# CLAUDE.md

WIP research repo for training video classification architectures on the WLASL sign-language
dataset. See [README.md](README.md) for CLI usage of `training.py`/`testing.py` (args, output
directory layout, result schemas) — not repeated here.

## Development setup: two machines

The user works across two separate computers:

- **This machine (local)** — code-only. No training is run here, and it does not have model
  weights on disk (e.g. `src/models/mvit/weights/*.pyth` — gitignored, and genuinely absent
  locally). Don't expect weight files to exist here; a `FileNotFoundError` for a weights path is
  expected on this machine, not a bug to chase.
- **The training server** — connected to via SSH. This is where training/testing actually runs,
  and it has the MViTv2 (slowfast) pretrained weights already in place at the paths `og_mvit.py`
  expects (`src/models/mvit/weights/`).

This means code changes here can only be verified by import/construction checks and unit-level
testing, not by actually running training or loading pretrained weights — that has to happen on
the server.

## Repository structure

### `src/` top-level modules

- `configs.py` — Config loading/parsing (TOML/argparse/JSON); builds `RunInfo`/`WandbInfo` etc.
  from CLI args and config files. Central config plumbing for training/testing.
- `run_types.py` — Shared pydantic types/constants used across the codebase (`RunInfo`,
  `DataInfo`, path constants, enums). The shared schema module.
- `preprocess.py` — Preprocesses raw WLASL videos/annotations into the training dataset format
  (bbox cropping, YOLO-based steps, split/set JSON generation); defines the `Instance` model.
- `video_dataset.py` — PyTorch `Dataset` for loading WLASL video instances/labels.
- `video_transforms.py` — Video augmentation/transform pipeline (spatial/temporal crops,
  RandAugment, normalisation) built on `torchvision.transforms.v2`.
- `training.py` / `testing.py` — Core training loop and evaluation/inference; back the CLIs
  documented in the README.
- `stopping.py` — `EarlyStopper`: halts training on a monitored metric plateau.
- `sweeping.py` — Wandb hyperparameter-sweep helpers; used both by the Que `Worker` and as a
  standalone CLI entrypoint for `wandb agent`.
- `benchmark.py` — GPU benchmarking harness for model architectures (subprocess-isolated
  timing/NVML monitoring for thesis/paper-grade throughput & memory results).
- `stats.py` — Computes/summarises WLASL dataset statistics and generates LaTeX tables (legacy,
  tied to the original WLASL JSON format).
- `utils.py` — Grab-bag of shared utilities: GPU memory manager, video-frame loading, misc.
- `visualise.py` / `visualise2.py` — see [below](#srcvisualisepy---srcvisualise2py).
- `debug.py` — Ad-hoc manual debug scripts for the Que system; not a formal test suite (there is
  no `tests/` dir or pytest config anywhere in the repo).

### `src/` subdirectories

- `configfiles/` — TOML configs by split (`asl100`/`300`/`1000`/`2000`) plus experiment sets
  (AugComparison, RandAug, Satnac_2025, sweeps, debug, generic/generic2).
- `info/` — Dataset metadata: WLASL class lists (json/txt).
- `models/` — Model architecture wrappers (`classifiers.py`, `pytorch_mvit.py`, `pytorch_r3d.py`,
  `pytorch_s3d.py`, `pytorch_swin3d.py`, `og_mvit.py`/`pyvision_mvit.py` variants), plus `mvit/`
  vendoring a trimmed copy of the `slowfast` reference implementation (own license file — see
  README's Licence section). The detectron2-based image-classification MViT code
  (`detectron2_cp/`, `image_classification/`) was removed; see
  [below](#env-cleanup-after-the-detectron2-mvit-removal-2026-09-12) for what that meant for the
  environment.
- `que/` — The "Que" system: a client/server training-run queue/scheduler (`server.py`,
  `daemon.py`, `worker.py`, `core.py`, `shell.py` REPL, `setup.sh`/`unsetup.sh` install a systemd
  service + `que` shell command). Lets you queue and manage training runs (reorder, etc.). Server
  needs the `wlasl` conda env; client can use `wlasl` or `wlasl_cpu`. See
  [src/que/README.md](src/que/README.md).
- `runs/` — Training run output directory (checkpoints/logs per run, organised by split), as
  documented in the README.
- `results/` — Experiment outputs, one subdir per paper/venue/analysis. Directories *without* a
  `todo.txt` are the current-convention examples to follow (see below):
  - `satnac_2025` / `satnac_2025_refactor` — SATNAC 2025 result JSONs, notebooks, benchmark
    scripts (`_refactor` is the newer, non-todo version).
  - `satnac_2026` — per-model TOML configs + results notebook (no `todo.txt` — reference).
  - `aug_comparison` — augmentation comparison notebook + helpers (no `todo.txt` — reference).
  - `saicist` — per-instance analysis: correlation plots, over/underachiever JSONs, notebook.
  - `sacair_2026` — benchmark + results notebooks, CSV.
  - `stats` — dataset-stat notebooks.
  - `augmentation_demos` — illustrative notebooks, one per augmentation type.
  - `dataset_analysis` — exploratory notebooks (class viewer, worst/fewest instances, F1
    correlation).
  - `outputs` — final rendered `.tex`/`.pdf` figures/tables, organised by venue subfolder.

### Root env/config files

- `wlasl_gpu.yml` — conda env spec for the `wlasl` (GPU) env. Regenerated 2026-09-12; see
  [below](#env-cleanup-after-the-detectron2-mvit-removal-2026-09-12).
- `wlasl_cpu.yml` — conda env spec for the `wlasl_cpu` env. Had been deleted from the repo
  (commit `d287a94`) even though the README still referenced it and it remained the CPU env in
  active use; regenerated 2026-09-12 from the live `wlasl_cpu` env. **CPU-only, non-MViT work**
  (notebooks, plotting) — it does not carry the MViT/slowfast dependencies (matches `wlasl`'s
  scope decision below).
- `pyproject.toml` — minimal setuptools config declaring the `slr` package (`src*`), no pinned
  dependency list (deps come from the conda env files).

### Env cleanup after the detectron2 MViT removal (2026-09-12)

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

**Two pre-existing, unrelated issues surfaced during this investigation (not fixed — out of
scope for the env cleanup, flagging for awareness):**

1. `og_mvit.py`'s `CONF_PATH_16x4`/`CONF_PATH_32x3` point to
   `mvit/configs/MVITv2_S_16x4.yaml`/`MVITv2_B_32x3.yaml`, which no longer exist (only
   `MVIT_B_16x4_CONV.yaml`/`MVIT_B_32x3_CONV.yaml` remain under `mvit/configs/`). Constructing any
   of the `MViTv2_S_16x4*`/`MViTv2_B_32x3*` models currently raises `FileNotFoundError` — this
   predates the env cleanup (reproduced identically on the pre-cleanup code). Note this is about
   the **config yaml** (a small tracked file, deleted from git — a real bug on both machines), not
   the pretrained weights (which are expected to be absent on the local machine — see
   [Development setup](#development-setup-two-machines) above).
2. `src/benchmark.py` unconditionally `import pynvml` and calls `pynvml.nvmlInit()` at module
   level, but `nvidia-ml-py` (which provides `pynvml`) is not installed in the live `wlasl` env —
   `import src.benchmark` currently fails. Also not installed: `fairscale`, `fastapi`, `starlette`,
   `uvicorn`, `sweeps`, `ninja`, `jsonref`, `tomli-w` (all previously listed in the old
   `wlasl_gpu.yml`) — the tracked yml had drifted from the real env before this cleanup even
   started.

## Notebooks in `src/results` need updating

Several directories under `src/results` contain a `todo.txt` left as a reminder that the
notebooks there need work:

- `src/results/augmentation_demos/todo.txt`
- `src/results/dataset_analysis/todo.txt`
- `src/results/stats/todo.txt`
- `src/results/satnac_2025_refactor/todo.txt`
- `src/results/satnac_2025/todo.txt`
- `src/results/saicist/todo.txt`
- `src/results/sacair_2026/todo.txt`

For each notebook flagged this way:

1. **Strict typing.** Ruff must report zero warnings on the notebook.
2. **Fix broken code.** Before writing new code, check whether an equivalent notebook/script
   already exists in a `src/results` directory that does *not* have a `todo.txt` (e.g.
   `src/results/satnac_2026`, `src/results/aug_comparison`) — these represent the current
   working convention and should be used as reference/example code rather than reinventing
   the fix from scratch.
3. **Plan before implementing.** Since a convention likely already exists elsewhere in the
   repo, work out the approach first rather than writing new code immediately.
4. **Consistent styling.** All plotting must follow the conventions established in
   [`src/visualise2.py`](src/visualise2.py) (`set_thesis_style()`, `plot_bar_chart`,
   `plot_grouped_bar_chart`, `plot_loss_curves`, `save_fig`, the shared colour palette, etc.).

## `src/visualise.py` -> `src/visualise2.py`

[`src/visualise.py`](src/visualise.py) is being superseded by
[`src/visualise2.py`](src/visualise2.py). New/updated plotting code should target
`visualise2.py`'s conventions, not `visualise.py`. Before adding or changing a plotting function
there, read [`src/VISUALISE2_CONVENTIONS.md`](src/VISUALISE2_CONVENTIONS.md) — it documents the
signature/return/colour/styling conventions and the current function catalog, so they don't need
to be re-derived from the source each time.
