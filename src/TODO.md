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
- `stats.reverse_preproc_format` uses the gloss name `"empty"` as its "slot not filled yet"
  sentinel, but WLASL has a real gloss called `empty`, so each new `empty` instance replaces the
  slot and only the last one per set survives (asl2000 loads as 21089 instead of 21095). It also
  builds slots with `[dict] * num_classes` (one shared dict) and sizes the list by the number of
  *distinct* labels, which raises IndexError for a set that is missing a class. Rewrite it to key
  by `label_num` with a real sentinel (or `defaultdict`), and add a test.
- `preprocess.py` remove-policy strings don't agree: the CLI passes `"reset"`, but
  `fix_bad_frame_range` checks `"reset_frames"` (so resets happen but are never logged) and
  `fix_bad_bboxes` checks `"reset_bbox"` (so `"reset"` with no detected person raises
  ValueError). Videos that can't be opened are always dropped, whatever the policy.
  `remove_short_samples` still runs with `--length_cutoff 0` and drops samples with <= 0 frames.
  Switch to one `Literal` for the policy.
