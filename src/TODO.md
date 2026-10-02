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
- **Regenerate the labels before the final round of results.** `preprocess.py` now stores
  0-based frame starts (cache version 2), fixing an off-by-one where every unreset instance
  skipped its first annotated frame, but the label files on disk still have the old 1-based
  starts so in-progress experiments (SATNAC, sweeps) stay comparable. When ready:
  `python -m src.preprocess all -ve` and `python -m src.preprocess all -ve -lc 9`. This is a full
  YOLO run, because the new `instance_cache_v2.json` starts empty. Then update the numbers in
  `src/info/WLASL_info.md` from the new logs, and delete the old `instance_cache.json` (it only
  matters for reproducing the old labels, together with the commit before this change). See
  `src/info/WLASL_info.md` ("What the rerun will change") for what to expect.
  Decision (2026-09-26): the final results should not remove any short clips, to match the
  original WLASL loader, which only skips videos under 9 frames, and no clip is that short. So train
  on the no-cutoff splits (`asl100`/.../`asl2000`), not `*_cutoff_9` (which after the rerun still
  removes `15144`, 9 frames). Switching the configs/`CUTOFF_9_NAMES` users over is part of this.
- The per-venue benchmark notebooks (`results/{satnac_2025,sacair_2026,satnac_2026}/benchmark.ipynb`)
  each copy-paste the same untyped `all_benchmark.json` loader and LaTeX formatting, differing
  only in which archs they keep. `results/benchmark/benchmark.ipynb` now has a typed loader
  (`load_runs`) covering every run; move it into a shared helper module and have the venue
  notebooks filter its output instead of re-parsing the JSON.
