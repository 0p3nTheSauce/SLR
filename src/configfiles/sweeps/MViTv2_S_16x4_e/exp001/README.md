# Sweep 1 : not created yet

Planned for 20 runs (see [Cost](#cost)).

This sweep follows [Sweep 0](../exp000/README.md) and carries over the setup of
[S3D sweep 7](../../S3D/exp007/README.md). The reasoning is in
[suggest_sweep.ipynb](../../../../results/sweeping/suggest_sweep.ipynb), section
"Conclusions (sweep 7 and the seed comparisons, as of 2026-10-02)".

MViTv2_S_16x4_e was chosen over MViTv2_B_32x3. It takes the same 32-frame input and is cheaper
(see [Cost](#cost)), and Sweep 0's best trial (val loss 0.590) beats every MViTv2_B_32x3 run
(best 0.630).

[base.py](./base.py) reuses S3D sweep 7's base config: S3D sweep 4's augmentation pipeline,
warmup, and a single `CosineAnnealingLR` cycle. Sweep 0 used the same augmentation but warm
restarts. Changes from Sweep 0:

- **No hyperband.** It stopped 18 of Sweep 0's 24 trials at epoch 15, and only the 6 it let
  finish got below 0.8. So Sweep 0's evidence is those 6 trials, and its ranges are their spread,
  padded.
- **A single cosine cycle (`tmax` 10-30) instead of warm restarts.** Sweep 0's finished trials
  used `t0` of 10-28, with the best at its lower bound, and most reached their best by epoch
  15-27.
- **Fixed what was flat in both Sweep 0 and S3D sweep 7**: `eps` (at 3e-6, the median of Sweep
  0's finished trials, all at or below 3.7e-5), `start_factor`, `hflip_p`, `max_wobble` and
  `num_ops`. This leaves 8 swept parameters for a small budget.
- **A lower `drop_p` range (0.1-0.6) than S3D's.** Sweep 0's finished trials used 0.19-0.51,
  while S3D sweep 7's top 10 used 0.46-0.77.
- **A backbone LR up to 6e-4.** MViTv2_B_32x3 diverged at 4e-4, but with no warmup. Sweep 0's
  finished trials reached 3.5e-4, with the best trial at that value.
- **The LR is held at `eta_min` after the cycle** (`hold_after_tmax`, set in base.py), so
  patience ends each trial soon after warmup + `tmax`. In S3D sweep 7, where PyTorch's periodic
  `CosineAnnealingLR` let the LR rise again, 13 of the 50 trials found their best only in a later
  cycle. They trained for a median of 140 epochs, against 56-61 for the others, and none of them
  was the best trial. `max_epoch` is 55 (warmup + `tmax` is at most 38, plus patience 15), which
  is only a backstop now.
- **Batch size 2 × 4 update steps**, as in Sweep 0.

## Cost

Sweep 0's trials took a median of 0.25 h per epoch (wandb `_runtime`, checked 2026-10-02), against
0.35 h for MViTv2_B_32x3. A typical trial here (warmup + `tmax` + patience ≈ 40 epochs) should
take about 10 h, and `max_epoch = 55` caps one at about 14 h. With the LR held at `eta_min` =
0, the epochs between the end of the cycle and the patience stop train nothing. 20 trials is roughly 8 days.
