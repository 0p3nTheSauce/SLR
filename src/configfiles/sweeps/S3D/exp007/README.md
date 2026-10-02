# Sweep 7 : u8yashg8

Created for 50 runs : [wandb](https://wandb.ai/ljgoodall2001-rhodes-university/Sweeps/sweeps/u8yashg8)

This sweep follows [Sweep 5](../exp005/README.md) and [Sweep 6](../exp006/README.md). The
reasoning is in [suggest_sweep.ipynb](../../../../results/sweeping/suggest_sweep.ipynb):

- Sweeps 5 and 6 (no augmentation, no scheduler) never got close to
  [Sweep 4](../exp004/README.md)'s augmented config. Sweep 5's best val loss was 1.186, while
  the [seed comparison](../../../SeedComparison/README.md) of Sweep 4's best config gives
  0.880 ± 0.039.
- Sweep 6 did not beat Sweep 5 because it barely sampled Sweep 5's best regime: large `eps`
  (> 1e-2) with a backbone LR around 1e-3.

Changes from Sweep 4, whose augmentation pipeline this reuses via [base.py](./base.py):

- **Scheduler**: warmup plus a single `CosineAnnealingLR` cycle (`tmax`) replaces
  `CosineAnnealingWarmRestarts`. Every seed-comparison run hit its best val loss at the end of
  the first cosine cycle, and the restart never paid off before patience stopped the run.
- **No hyperband**: early stopping (patience 20) only.
- **`eps` up to 1e-1** (Sweep 4 capped it at 1e-3), with the backbone LR widened to match, to
  test whether Sweep 5's large-`eps` regime still helps once there is warmup.
- **Ranges widened where Sweep 4's best trial sat near a bound**: backbone weight decay,
  `drop_p`, and RandAugment `magnitude`. `hflip_p` now starts at 0.1, since higher was better.

**Caveat**: PyTorch's `CosineAnnealingLR` is periodic, so after warmup + `tmax` epochs the LR
rises again. Patience 20 should stop a run before this matters (the best checkpoint is kept),
but training doesn't yet end at the bottom of the cycle.

