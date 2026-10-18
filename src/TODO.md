Misc TODOs:
- `training.py` keeps the final resume checkpoint (`checkpoint_NNN.pth`, ~3x `best.pth`) after a
  run finishes, though only `best.pth` is used afterwards; `python -m src.utils --only resume`
  clears them in bulk. Consider deleting it when a run completes successfully (keep it on a
  crash so the Que can recover the run).
- Runs predating `best.pth` (e.g. `runs/asl100_cutoff_9_1_indexed/S3D/exp008`, `exp009`,
  `runs/asl300_cutoff_9_1_indexed/{S3D,Resnet3D_18,Resnet2D_1D_18}/exp000`) only have a full
  resume checkpoint, which the runs cleaner deliberately skips. Converting each to a `best.pth`
  (its `model_state_dict`) would save ~2/3 of their size and make them loadable like the rest.
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
- **Remove the speed sampler and the AugComparison experiments.** `sample_speed_perturbed` only
  resamples a short window of consecutive frames, not the whole sign (see its docstring), so its
  runs are already left out of `results/aug_comparison` (and the thesis). The full removal (code,
  configs, results, Que runs, run dirs) is planned in `src/remove_speed_sampler_plan.md` on the
  `feat/remove-speed-sampler` branch, with three decisions still open.
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
- `visualise2.plot_frame_grid`'s `adapt=True` sizes cells from the frames' pixels with a
  hardcoded 5 in per 256 px, unlike the dpi-based `scale` of `animate_frames`/
  `_frame_panel_size`. Its height/width were also swapped (fixed). No tracked code uses it, so
  replace `adapt` with `scale` (or a FIGSIZE-fitted default, as `plot_frame_grid_topk` has).
- **WLASL gloss ambiguity (`before` / `former` / `past`).** See `src/info/WLASL_info.md` ("One
  sign, several glosses"). Check whether Boston University's revised WLASL gloss labels
  (https://www.bu.edu/asllrp/wlasl-alt-glosses.pdf) merge or split these classes, and consider scoring against them, or counting
  same-sign confusions as correct, when reporting per-class results.
- **Duplicate clips (see `src/info/WLASL_info.md`, "Duplicate clips").** 9.4% of WLASL-2000 are
  byte-identical copies, 1877 of them one clip under several glosses, and 184 test instances have a
  twin in train/val. Decide how the final results treat them (drop the conflicted instances from
  test, merge the glosses, or report both), and whether the train copies should be removed.
  Near-duplicates (re-encodes) are not found by the MD5 hashing: try perceptual hashing or
  frame-level matching. Compare the conflicting gloss pairs with the Boston University revised
  labels (the item above).
- **Cleaned dataset with variation-level classes (feasibility in
  `results/dataset_analysis/custom_dataset_feasibility.ipynb`).** Needs: a reviewed merge list (338
  class pairs share 2+ clips), a re-split rule, a per-dataset class list in place of the global
  `info/wlasl_class_list.json` (read by `stats.py`, `visualise.py`, `visualise2.py`), and a check that
  variation labels are not just the source site (guessing from gloss and source gets 81%, from the
  gloss alone 60%).

- `asl100_bottom`/`asl100_worst` labels are still 1-based: they aren't made by `preprocess.py`, so
  the 0-based rerun doesn't touch them, unlike the plain splits (whose old labels were kept as
  `*_1_indexed`, see `src/info/WLASL_info.md`). Rebuild them from the new labels (and keep the
  old ones alongside, as for the plain splits) if they're used for final results.
- `src/video_transforms.py` has 38 pre-existing ruff errors (as of 2026-10-06): old `typing`
  aliases (`List`/`Tuple`/`Optional`), unused `run_types` imports, mutable defaults in
  `get_transform`, bare `raise Exception`. Most are `ruff --fix`-able; do it when next working there
  (the speed-sampler removal, see above, touches this file).
- **Next up (as of 2026-10-15).** Thesis table: replace the stale `MViTv2-B*` stub entry (a test-selected
  pick) with val-loss-selected runs, and add the 5-trial sweep-1 ensemble (label it an ensemble) from
  `results/sweeping/stashed_results/asl100_cutoff_9_MViTv2_S_16x4_e_510lhysg_ensemble_.json`. Then the next
  sweep: draft `configfiles/sweeps/MViTv2_S_e/exp000` on plain `asl100` with a small positive `eta_min`
  (about 6 days for 20 trials). Optional GPU work for stronger thesis numbers: the sweep-1 best config on
  asl300 (about 16 h, initialised from trial `5mzfq88e`'s checkpoint), then asl1000 and asl2000.
  Clear the stale sweep (`510lhysg`, 20/20) in `src/que/state/Server.json` when the next one is set.
- **Two duplicate-clip analyses (as of 2026-10-18).** The two `main`s diverged and each did one:
  `results/dataset_analysis/find_duplicates.py` (exact + near-duplicate frame fingerprints, writes
  `stashed_results/duplicates_asl2000.json`, used by `satnac_2026/duplicate_accuracy.ipynb`) and
  `results/dataset_analysis/duplicates.py` (byte-identical MD5 copies, for `duplicate_clips.ipynb`).
  Compare their findings, then keep one module (or make one build on the other) and one stash.
- `results/dataset_analysis/duplicates.py` and `tests/results/test_duplicates.py` have 8 pyright errors
  (as of 2026-10-18): pandas groupby/`DataFrame(columns=list)` typing. Fix with explicit casts/types.
