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
  `src/results/dataset_analysis/view_dataset_by_admin_info.ipynb`,
  `src/results/augmentation_demos/{autoaugment,cropping_norms,randaugment}.ipynb`,
  `src/results/bottom_worst_splits/{100_worst,100_fewest}.ipynb`. Migrate opportunistically
  when next touching one of these rather than as a standalone sweep.
- `CompRes` (src/run_types.py) doesn't record epochs trained or the epoch of the best val loss,
  so `results/sweeping/suggest_sweep.ipynb` can't show where hyperband/early stopping cut trials
  short from the Que alone (wandb's `Epoch` summary has it). It works around this by stashing
  each sweep's epochs from wandb's history. Consider adding both.
- `CosAnealInfo.hold_after_tmax` (training.cosine_then_hold) now holds the LR at `eta_min` after
  warmup + `tmax`; it defaults to PyTorch's periodic behaviour so older configs keep their meaning.
  Consider also ending training at warmup + `tmax` (plus a few epochs) when it's set, since with
  `eta_min = 0` the epochs until patience fires train nothing.
- With no warm-up, `training.get_scheduler` still wraps the main scheduler in a `SequentialLR`
  with milestone 0, which starts it one epoch late: epoch 0 runs on the identity placeholder, so
  the LR stays at its initial value for 2 epochs and the cosine trough lands at `tmax + 1` (see
  `tests/test_training.py`). Returning the main scheduler directly would fix it, but shifts every
  no-warm-up config (e.g. all of `configfiles/Satnac_2025`) by an epoch, so decide whether that's
  acceptable first.
- `results/seed_comparison/results.ipynb` only covers the `S3D_13idpda6.toml` seed runs (its
  filters.py now pins `admin.config_path`). Extend it to compare `S3D_czopef0v.toml`'s seed runs,
  which have finished. Their best val loss is already in `results/sweeping/suggest_sweep.ipynb`,
  stashed as `asl100_cutoff_9_S3D_seed_comparison_czopef0v_runs.json`.
- **Finish the switch to the 0-based labels.** The plain splits (`asl100`/.../`asl2000`) were
  regenerated on 2026-10-05, with the old 1-based ones kept as `*_1_indexed` (see
  `src/info/WLASL_info.md`, "The 0-based rerun"). Remaining:
  - Rerun `src/results/dataset_analysis/frame_ranges.ipynb` against the new labels: every label
    should "match annotation" and lose no frames, with the 296 resets unchanged.
  - Regenerate the dataset figures (`num_instance_splits.ipynb`, `wlasl_stats_set_level.ipynb`)
    from the plain splits.
  - Switch the configs/`CUTOFF_9_NAMES` users over to the plain splits (see decisions below).

  Decision (2026-09-26): the final results should not remove any short clips, to match the
  original WLASL loader, which only skips videos under 9 frames, and no clip is that short. So
  train on the no-cutoff splits, not `*_cutoff_9` (which after a 0-based rerun would still remove
  `15144`, 9 frames).
  Decision (2026-10-06), for the thesis: dataset figures describe the plain splits, i.e. no
  instance removed, but with the necessary fixes (corrected, 0-based frame ranges and our YOLO
  bboxes; WLASL's own bboxes are in the original videos' coordinates, so unusable). Results that
  would need retraining are not rerun, but get a footnote on the preprocessing differences
  (1-based starts, `cutoff_9`).
- **Keep the 1-based `*_cutoff_9` labels before regenerating them.** Every SATNAC/SACAIR
  experiment, and the paused sweep `510lhysg` (asl100_cutoff_9), trained on them. Once the sweep
  is done, repeat what was done for asl100: rename `labels/asl*_cutoff_9` and
  `src/runs/asl*_cutoff_9` to `*_cutoff_9_1_indexed`, add the names to `ONE_INDEXED_SPLITS`
  (`src/run_types.py`) and `SPLIT_NAME_MAP`, and move their Que runs with
  `que/debug.py`'s `move_to_one_indexed_splits` (it only handles the plain splits so far), writing
  `Runs_updated.json` and validating it before swapping it in. Only then run
  `python -m src.preprocess all -ve -lc 9`.
- **Release the 1-based labels**, as a GitHub release like v1.0 (`splits.zip`, `WLASL2000.zip`),
  so the earlier experiments can be reproduced: zip the `*_1_indexed`, (once renamed)
  `*_cutoff_9_1_indexed`, `asl100_bottom` and `asl100_worst` label dirs, and link them from the README's data setup with what they
  are for. After that, delete `labels/instance_cache.json`, which is only needed to rebuild those
  labels.
- The per-venue benchmark notebooks (`results/{satnac_2025,sacair_2026,satnac_2026}/benchmark.ipynb`)
  each copy-paste the same untyped `all_benchmark.json` loader and LaTeX formatting, differing
  only in which archs they keep. `results/benchmark/benchmark.ipynb` now has a typed loader
  (`load_runs`) covering every run; move it into a shared helper module and have the venue
  notebooks filter its output instead of re-parsing the JSON.
- `configfiles/asl100/MViTv2_B_32x3/exp013.toml` says it is "same as ... but with warm up", but
  it has no `[scheduler.warm_up]` block, and its Que run has no warmup either. Fix the comment, or
  rerun it with the warmup it was meant to have. No MViTv2_B_32x3 run so far has used warmup.
- **mp4 export for `visualise2` animations.** `animate_frames`/`animate_frames_topk` are only
  shown inline (`animation_html`); their docstrings point at `anim.save(path, dpi=fig.dpi)`,
  untested. Add a `save_animation` counterpart to `save_fig` (mkdir the parent, figure dpi, close
  the figure) and check `ffmpeg` is there: it's in `wlasl_gpu.yml` but not `wlasl_cpu.yml`
  (pillow, for ".gif", is in both).
- `visualise2.plot_frame_grid`'s `adapt=True` sizes cells from the frames' pixels with a
  hardcoded 5 in per 256 px, unlike the dpi-based `scale` of `animate_frames`/
  `_frame_panel_size`. Its height/width were also swapped (fixed). No tracked code uses it, so
  replace `adapt` with `scale` (or a FIGSIZE-fitted default, as `plot_frame_grid_topk` has).
- **WLASL gloss ambiguity (`before` / `former` / `past`).** See `src/info/WLASL_info.md` ("One
  sign, several glosses"). Check whether Boston University's revised WLASL gloss labels
  (https://www.bu.edu/asllrp/wlasl-alt-glosses.pdf) merge or split these classes, and consider scoring against them, or counting
  same-sign confusions as correct, when reporting per-class results.
- `asl100_bottom`/`asl100_worst` labels are still 1-based: they aren't made by `preprocess.py`, so
  the 0-based rerun doesn't touch them, unlike the plain splits (whose old labels were kept as
  `*_1_indexed`, see `src/info/WLASL_info.md`). Rebuild them from the new labels (and keep the
  old ones alongside, as for the plain splits) if they're used for final results.
