Misc TODOs:
- Repo-wide: audit blind `except Exception` blocks (training.py, que/worker.py, etc.) and
  replace with handling for specific exception types instead of blanket-catch + str(e).
  Concrete motivating case documented in src/que/todo (Worker section): a bare, message-less
  Exception from wandb's own hyperband/early-terminate thread-kill mechanism currently gets
  either silently swallowed or reported indistinguishably from a real crash, depending on
  which layer it surfaces at.
- `visualise2.plot_frame_grid` now exists as the convention-following (returns `(fig, axes)`,
  saved via `save_fig`) replacement for the legacy `utils.plt_display_grid` (which saves
  directly via its own `output` param, and doesn't mkdir the parent dir first -- unlike
  `save_fig`). `src/results/satnac_2026/eda_performers.ipynb` and `visualise2.FrameVisualiser`
  have been migrated; still on the old `utils.plt_display_grid` directly:
  `src/results/dataset_analysis/{view_dataset_by_admin_info,class_viewer}.ipynb`,
  `src/results/augmentation_demos/{autoaugment,cropping_norms,randaugment}.ipynb`,
  `src/results/bottom_worst_splits/{100_worst,100_fewest}.ipynb`. Migrate opportunistically
  when next touching one of these rather than as a standalone sweep.
- `CompRes` (src/run_types.py) doesn't record epochs trained or the epoch of the best val loss,
  so `results/sweeping/suggest_sweep.ipynb` can't show where hyperband/early stopping cut trials
  short from the Que alone (wandb's `Epoch` summary has it). Consider adding both.
- `CosineAnnealingLR` (training.get_scheduler) is periodic in PyTorch: after warmup + `tmax`
  epochs the LR climbs back up. S3D sweep 7 relies on patience to stop runs first. Consider
  ending training at warmup + `tmax`, or holding the LR at `eta_min` afterwards.
- `results/seed_comparison/results.ipynb` only covers the `S3D_13idpda6.toml` seed runs (its
  filters.py now pins `admin.config_path`). Extend it to compare `S3D_czopef0v.toml`'s seed runs
  once they finish.
- Preprocessing rework (branch `fix/preprocess-logging`) needs verifying against the real labels
  once nothing is training. A dry run on a *copy* of the cache already reproduced all six
  asl2000/asl2000_cutoff_9 label files exactly, without running YOLO. Still to do:
  (1) rerun `python -m preprocess all` (and `-lc 9`) so the real `instance_cache.json` migrates
  to the new format (296 frame-range resets rebuilt from the raw split) and each split gets a
  `preprocess_log.json`;
  (2) diff the regenerated `*_fixed_frange_bboxes.json` against the current ones (they should be
  identical) and delete the old per-stage logs (`cutoff_9_removed_short_samples_*.json`);
  (3) at some point, run once with `--no_cache` into a scratch output dir, to check that a clean
  YOLO rebuild matches the cached bboxes, and to get logs that don't rely on the cache migration.
- `preprocess.fix_bad_frame_range` accepts `end <= start + num_frames`, so an end frame past the
  end of the video is kept whenever start > 0. WLASL's `frame_start` is 1-indexed, but the code
  treats it as 0-indexed. Left unchanged on purpose, because changing it would change the labels
  that published results were trained on. Decide whether to fix it for future runs.
- `src/video_dataset.py:314-330` is dead commented-out code that calls the old
  `fix_bad_frame_range`/`fix_bad_bboxes`/`remove_short_samples` signatures. Delete it.
- `src/results/dataset_analysis/wlasl_stats_set_level.ipynb`'s gloss-merge and `find_missing`
  cells were a scratch investigation of the (now fixed) `reverse_preproc_format` bug. Clean them
  up or remove them.
