# Plan: remove the speed sampler and the AugComparison experiments

Drafted 2026-10-06. Not started: for the thesis, the speed runs were only dropped from the
`results/aug_comparison` plots (see `src/TODO.md`). This branch holds the plan for the full removal.

## Why

`video_transforms.sample_speed_perturbed` doesn't change the speed of the whole sign. It picks a
speed `s` from `[speed_min, speed_max]`, takes `int(target_length * s)` *consecutive* frames from a
random start, and resamples them to `target_length`. Measured on 2026-10-06 (5000 draws each on a
58-frame clip, `target_length = 16`, as in the AugComparison configs):

| Config | Speed range | Frames covered |
|---|---|---|
| `speed01` | 0.9-1.1 | 14-17 |
| `speed02` | 0.8-1.2 | 12-19 |
| `speed03` | 0.7-1.3 | 11-20 |

The median asl100_cutoff_9 train clip (what these runs trained on) is 59 frames, so each training
sample is about a quarter of the sign at close to the native frame rate: effectively a short random
temporal crop. The configs also test with the `uniform` sampler (16 frames spread over the whole
clip), so train and test inputs differ too.

## What's affected (as of 2026-10-06)

- **Code:** `sample_speed_perturbed` and its `"speed"` case in `video_transforms.py`;
  `SpeedSampler` and its entries in `SAMPLER_TYPES` and the sampler union in `run_types.py`. No
  tests reference them.
- **Configs:** `configfiles/AugComparison/` (23 files, 3 of them speed), and
  `configfiles/asl100/S3D/exp071.toml` (S3D with the speed sampler, `target_length = 32`; wandb
  project `WLASL-100_cutoff_9`, run `plwk1pun`, saved to `runs/asl100_cutoff_9/S3D/exp071`).
- **Results:** `results/aug_comparison/` (`results.ipynb`, `helpers.py`); nothing else imports it.
- **The Que:** 44 AugComparison runs in `old_runs` (22 configs x 2, all MViTv2_S) plus exp071.
  7 have `type = "speed"`, so `Runs.json` won't validate once `SpeedSampler` is gone.
- **Docs:** `CLAUDE.md` (configfiles list; `aug_comparison` as a reference example, three places)
  and `src/info/plotting_conventions.md` (reference implementations). Point both at
  `satnac_2026` only.
- **Run dirs:** `runs/asl100_cutoff_9/MViTv2_S/exp022`-`exp045` (22 dirs, about 23 GB) and
  `runs/asl100_cutoff_9/S3D/exp071` (124 MB); no other run uses them.
- `visualise2.suggest_palette`/`CONTROL_COLORS` (grey "baseline"/"no_aug" bars) came from this
  experiment but are `plot_bar_chart`'s default palette, so they stay unless that's simplified.

## Steps

1. Remove the code, configs, results dir and doc references; ruff, pyright, pytest; commit.
2. One-off in `que/debug.py`: drop the AugComparison runs and exp071, writing
   `Runs_updated.json`. Validate (loads under the new types, exactly those runs gone, nothing
   else changed), then swap in with `Runs.json` copied to `old_ques/` first. Server stopped.
3. Run dirs, per the decision below.

## Open decisions

- Delete the run dirs (irreversible), or keep them on disk?
- Remove exp071 along with AugComparison (it used the speed sampler but isn't part of it)?
- Tag `main` before removing (e.g. `aug-comparison-final`), so the code is easy to find for
  reproducing earlier results?
