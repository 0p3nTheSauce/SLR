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
- `preprocess.fix_bad_frame_range` accepts `end <= start + num_frames`, so an end frame past the
  end of the video is kept whenever start > 0. WLASL's `frame_start` is 1-indexed, but the code
  treats it as 0-indexed. Left unchanged on purpose, because changing it would change the labels
  that published results were trained on. Decide whether to fix it for future runs.
- `src/video_dataset.py:314-330` is dead commented-out code that calls the old
  `fix_bad_frame_range`/`fix_bad_bboxes`/`remove_short_samples` signatures. Delete it.
- `src/benchmark.py` uses bare imports (`from models import ...`), so it only works when run as a
  script from `src/` and `import src.benchmark` fails. Switch to `src.`-prefixed imports like the
  rest of the repo, and check it still runs from `src/`.
