# MViTv2_S_e sweep 2 : flx99idx

Created 2026-10-22 for 20 trials on `asl100` (see [Running it](#running-it) and [Cost](#cost)), queued as Que_v2
group `sweep2-flx99idx` : [wandb](https://wandb.ai/ljgoodall2001-rhodes-university/Sweeps/sweeps/flx99idx)

This sweep reruns [MViTv2_S_16x4_e sweep 1](../../MViTv2_S_16x4_e/exp001/README.md) with both of
its caveats fixed (thesis run list job 5, 18 Oct 2026). Sweep 1's results and the reasoning for
these ranges are in
[suggest_sweep.ipynb](../../../../results/sweeping/suggest_sweep.ipynb), section
"Conclusions (MViTv2_S_16x4_e sweep 1, as of 2026-10-14)".

- **The torchvision `MViTv2_S_e` instead of PySlowFast's `MViTv2_S_16x4_e`.** Same 32-frame input,
  about 30% faster to train at the same memory, and accuracy at least as good in uncontrolled
  comparisons (`src/results/all_benchmark.json`, RTX 3060, 2026-09-12). It also needs no manual weight
  files. Sweep 1's ranges were tuned on the other implementation, so the ranges here are widened
  where its best sat at a bound.
- **`eta_min` is 1e-6, not 0.** [base.py](./base.py) still holds the LR at `eta_min` after
  warmup + `tmax`, so patience ends each trial soon after the cycle. In sweep 1 the LR was then
  exactly 0, and those epochs trained nothing. 1e-6 is below the smallest classifier LR, so the
  cosine always decays.
- **The corrected `asl100` split** (0-based frame starts, see
  [WLASL_info.md](../../../../info/WLASL_info.md)), the split the thesis's final results use,
  instead of `asl100_cutoff_9` (now `asl100_cutoff_9_1_indexed`). So its val losses aren't
  directly comparable with sweeps 0 and 1. Thesis job 1 trained sweep 1's best config on this
  split (job 6: 81.40 test top-1).

[base.py](./base.py) reuses sweep 1's base config unchanged. Changes to the ranges from sweep 1:

- **Backbone LR 1e-4 to 1.2e-3** (was 3e-5 to 6e-4). Sweep 1's best (5.1e-4) and its top 5 sat
  against the 6e-4 cap, and every trial below 2.5e-4 was poor.
- **Backbone weight decay 1e-3 to 0.3** (was 1e-5 to 0.1). Sweep 1's best (0.079) was near the cap.
- **`magnitude` 7 to 15** (was 3 to 15). The top 5 used 11-15.
- **`tmax` 15 to 30** (was 10 to 30). Only `tmax` 10 was poor.
- **`drop_p` 0.1 to 0.5** (was 0.1 to 0.6). This was flat in sweep 1.
- **Fixed `warmup_epochs` at 5** (sweep 1's top 15 all used 5) **and `classifier_weight_decay` at
  1e-4** (flat in sweep 1).
- **`classifier_init_lr` is still swept (1e-5 to 3e-3)**, although it was flat in sweep 1. Sweep 1's
  best paired a high backbone LR with a classifier LR 10× lower (5.2e-5). That config trained
  `asl100` and `asl300` directly from K400, but stayed at chance on `asl1000` (Que_v2 job 8,
  19 Oct 2026). A sweep on `asl100` can't catch that, so before using the best trial on the
  larger splits, check its LRs against the 1e-4 backbone / 1e-3 classifier that have trained
  them directly from K400, or chain from `asl100`.
- **`max_epoch` 50** (warmup 5 + `tmax` at most 30 + patience 15); only a backstop.

## Running it

The sweep runs on Que_v2, one job per trial, from the `sweep` template in [que.toml](../../../../../que.toml),
which runs [`src/sweep_trial.py`](../../../../sweep_trial.py). A job's first attempt asks wandb for
a trial and saves the trial's resolved config. Every attempt then trains that trial like a `train`
job, resuming it from its last epoch, so the jobs are preemptible. Trials go to
`src/runs/asl100/MViTv2_S_e/sweep_flx99idx/job<Que_v2 job id>/`.

    wandb sweep src/configfiles/sweeps/MViTv2_S_e/exp000/config.yaml   # prints the sweep id
    for i in $(seq 20); do que2 submit base sweep sweep_id=<id> --group sweep2-<id> --priority <p>; done

Then record the id at the top of this file, in `config.yaml` and in `sweep_meta.json`.
`suggest_sweep.ipynb` can't read these trials yet: it reads the v1 Que's records (see
`src/TODO.md`).

## Cost

Sweep 1 (MViTv2_S_16x4_e) took a median of 0.25 h per epoch, and its trials a median of 38
epochs. MViTv2_S_e trains about 30% faster (2.34 vs 1.74 samples/s at 32 frames, batch size 1), so
expect about 0.19 h per epoch, about 7-8 h per trial, and about 6 days for 20 trials. `max_epoch` 50
caps a trial at about 9.5 h.
