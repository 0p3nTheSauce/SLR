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

## Testing

The pytest suite lives in `tests/` (configured via `[tool.pytest.ini_options]` in
`pyproject.toml`), with subdirectories mirroring `src/` (e.g. `tests/que/`). Run it with the
**`wlasl` env**: `~/miniconda3/envs/wlasl/bin/python -m pytest`. It can't be collected under
`wlasl_cpu` — `tests/conftest.py` imports `src.models.og_mvit`, which needs MViT deps
(`simplejson` etc.) that `wlasl_cpu` deliberately omits.

- Tests needing real pretrained weights are marked `requires_weights` and are auto-skipped on this
  machine (see `conftest.py`), so the full suite should pass locally.
- A `Que` saves itself after every change, so tests must never build one on the default (live)
  runs path: use the `que`/`make_que` fixtures in `tests/que/conftest.py` (a `tmp_path` runs file;
  calling `make_que()` again simulates a restart), with runs from `tests/que/factories.py`.
  `tests/que/test_server.py`'s `start_server` fixture does the same for a real `ServerContext`.
- `setup_server_logging`/`setup_training_logging` (`src/que/core.py`) configure global loggers;
  tests calling them must restore those (see `TestLoggingSetup` in `tests/que/test_core.py`).
- The Que's live data and logs are in `src/que/state/` and `src/que/logs/` (see the "Files"
  section of `src/que/README.md`) — tests must never write there.
- `QueShell` can be built in tests with a fake server by stubbing `_show_banner`,
  `_setup_history` and `tmux_manager` (see the `harness` fixture in `tests/que/test_shell.py`) —
  this avoids touching the real `~/.que_shell_history`.

## Code quality bar & workflow expectations

The user is actively working to keep this a clean, strictly-typed, well-organised repo, and
cares a lot about code quality — not just "does it run." Hold new/edited code to this bar:

- **Strict typing, zero linter warnings.** Ruff (and type checking) should report nothing on
  code you touch. Prefer precise types (e.g. `Literal` over bare `str` for closed sets of names)
  over `Any`/untyped escape hatches. Run both from the repo root (from a subdirectory, ruff's
  isort misclassifies `src.` imports):
  - `ruff check <files>`
  - `pyright --pythonpath ~/miniconda3/envs/wlasl/bin/python <files>` — pyright is the checker
    behind VS Code's Pylance, installed locally as a uv tool (`uv tool install pyright`); there's
    no pyright config, so it runs in its default mode.
- **Documented, but not over-commented.** Public functions/classes get docstrings explaining
  non-obvious behaviour, inputs/outputs, and gotchas — but avoid restating what the code already
  says. This mirrors the top-level house style (see the "Default to writing no comments" rule),
  applied here specifically to docstrings on shared/reusable code.
- **Simple, single-purpose functions.** Don't overload one function with multiple responsibilities
  or too many branchy parameters; split when a function is trying to do more than one job.
- **Reuse over reinvention.** Before writing new code, check for an existing helper/convention
  elsewhere in the repo (e.g. `visualise2.py` plotting helpers, `FrameFetcher`/`FrameVisualiser`
  patterns) and use/extend it rather than duplicating logic.
- **Upgrade legacy/quick-and-dirty code when touched.** Parts of this repo were built under time
  pressure (quick solutions, skipped docs, suboptimal approaches) and never revisited. When you're
  working in or near such code, flag it for review rather than silently leaving it, and where it's
  in scope, take the opportunity to bring it up to current convention instead of just patching
  around it.

### TODO tracking — read and maintain these

The repo tracks outstanding work as plain-text/markdown TODO files scattered by area, rather than
one central list. **Check these before starting related work, and keep them current**: when you
complete or invalidate an item, remove/update it; when you spot a new problem area while doing
other work (dead code, drifted docs, a hacky workaround, a missing test), add a TODO for it in the
relevant file (or create one) instead of letting it go unrecorded. Known TODO files as of
2026-09-26:

- `src/TODO.md` — misc repo-wide TODOs.
- `src/que/todo` — Que system TODOs, broken out by subsystem (Que core, Shell, Daemon, Worker,
  Server).

Check for new ones with `find src -iname "todo*"` rather than trusting this list.

## Documentation: `src/info/`

Reference material used across the repo (dataset facts, repo-wide conventions, env history) lives
in [`src/info/`](src/info/), indexed by [`src/info/README.md`](src/info/README.md), which also
holds the rules for writing documents there. In short:

- **Say it once, link to it.** Notebooks and other docs point to the relevant `src/info/` file with
  a one-liner rather than copying its content. If you find the same information repeated, move it
  into `src/info/` and replace the copies with links.
- **Numbers need a source and a date,** so stale figures can be spotted.
- **Keep the index current** when adding, renaming or rescoping a document.
- Docs about a single package stay beside its code (`src/que/README.md`, `src/models/README.md`).

Most relevant: [`WLASL_info.md`](src/info/WLASL_info.md) (dataset fields, gotchas such as the real
gloss called `empty`, split/set naming, what preprocessing changed) and
[`plotting_conventions.md`](src/info/plotting_conventions.md) (read before writing plotting code).

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
- `debug.py` — Ad-hoc manual debug scripts (as is `que/debug.py`); not a formal test suite — see
  [Testing](#testing) for the real one.

### `src/` subdirectories

- `configfiles/` — TOML configs by split (`asl100`/`300`/`1000`/`2000`) plus experiment sets
  (AugComparison, RandAug, Satnac_2025, sweeps, debug, generic/generic2).
- `info/` — Repo-wide reference docs and the WLASL class lists; see
  [Documentation](#documentation-srcinfo).
- `models/` — Model architecture wrappers (`classifiers.py`, `pytorch_mvit.py`, `pytorch_r3d.py`,
  `pytorch_s3d.py`, `pytorch_swin3d.py`, `og_mvit.py`/`pyvision_mvit.py` variants), plus `mvit/`
  vendoring a trimmed copy of the `slowfast` reference implementation (own license file — see
  README's Licence section). The detectron2-based image-classification MViT code
  (`detectron2_cp/`, `image_classification/`) was removed; see
  [`src/info/environment.md`](src/info/environment.md) for what that meant for the environment.
- `que/` — The "Que" system: a client/server training-run queue/scheduler (`server.py`,
  `daemon.py`, `worker.py`, `core.py`, `shell.py` REPL, `setup.sh`/`unsetup.sh` install a systemd
  service + `que` shell command). Lets you queue and manage training runs (reorder, etc.). Server
  needs the `wlasl` conda env; client can use `wlasl` or `wlasl_cpu`. See
  [src/que/README.md](src/que/README.md).
- `runs/` — Training run output directory (checkpoints/logs per run, organised by split), as
  documented in the README.
- `results/` — Experiment outputs, one subdir per paper/venue/analysis. `satnac_2026` and
  `aug_comparison` are the reference examples of current convention:
  - `satnac_2025` / `satnac_2025_refactor` — SATNAC 2025 result JSONs, notebooks, benchmark
    scripts (`_refactor` is the newer version).
  - `satnac_2026` — per-model TOML configs + results notebook (reference).
  - `aug_comparison` — augmentation comparison notebook + helpers (reference).
  - `saicist` — per-instance analysis: correlation plots, over/underachiever JSONs, notebook.
  - `sacair_2026` — benchmark + results notebooks, CSV.
  - `benchmark` — `benchmark.ipynb`: every run in `all_benchmark.json` (all models), with
    the method, environment, coverage, summary and full tables in one file.
  - `stats` — dataset-stat notebooks.
  - `augmentation_demos` — illustrative notebooks, one per augmentation type.
  - `seed_comparison` — variance of one config across 10 seeds (see
    `configfiles/SeedComparison/`).
  - `sweeping` — `suggest_sweep.ipynb`: compares wandb sweeps' Que trials and suggests the next
    sweep's parameter ranges (helpers in `sweeping/helpers.py`).
  - `dataset_analysis` — exploratory notebooks (class viewer, worst/fewest instances, F1
    correlation).
  - `outputs` — final rendered `.tex`/`.pdf` figures/tables, organised by venue subfolder.

### Root env/config files

`wlasl_gpu.yml` / `wlasl_cpu.yml` (conda env specs) and `pyproject.toml` (the `slr` package, no
pinned deps). What each env is for, how the ymls are regenerated, and the history of the 2026-09-12
detectron2 cleanup: see [`src/info/environment.md`](src/info/environment.md).

## Updating notebooks in `src/results`

When bringing a notebook up to current convention:

1. **Strict typing.** Ruff must report zero warnings on the notebook.
2. **Fix broken code.** Before writing new code, check whether an equivalent notebook/script
   already exists elsewhere in `src/results` (e.g. `src/results/satnac_2026`,
   `src/results/aug_comparison`, which represent the current working convention), and use it as
   reference/example code rather than reinventing the fix from scratch.
3. **Plan before implementing.** Since a convention likely already exists elsewhere in the
   repo, work out the approach first rather than writing new code immediately.
4. **Consistent styling.** All plotting must follow
   [`src/info/plotting_conventions.md`](src/info/plotting_conventions.md).
5. **Link, don't repeat.** Replace copied reference material (e.g. dataset descriptions) with a
   one-liner pointing to `src/info/`.

## `src/visualise.py` -> `src/visualise2.py`

[`src/visualise.py`](src/visualise.py) is being superseded by
[`src/visualise2.py`](src/visualise2.py). New/updated plotting code should target
`visualise2.py`'s conventions, not `visualise.py`. Before adding or changing a plotting function
there, read [`src/info/plotting_conventions.md`](src/info/plotting_conventions.md) — it documents
the signature/return/colour/styling conventions (including legend placement — **the legend must
never occlude a bar/line/marker**, check the rendered figure, not just the code) and the current
function catalog, so they don't need to be re-derived from the source each time.
