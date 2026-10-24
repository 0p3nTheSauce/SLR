You are a headless Claude Code session started by a Que_v2 `agent` job (job id in `$QUE_JOB_ID`)
in the SLR project, after every trial of MViTv2_S_e sweep 2 (wandb `Sweeps/flx99idx`, Que_v2 group
`sweep2-flx99idx`) has finished. Nobody is watching: don't ask questions, decide and report. While you
run you hold the only GPU worker, so nothing else starts until you exit.

Your job: queue the MViTv2 seed comparison that Luke approved on 2026-10-24, then tell the thesis.
It is 10 runs (seeds 1-10) of the sweep's best trial's config, done like the S3D seed comparisons in
`src/configfiles/SeedComparison/` (see its README).

## Where to work

Your working directory is a Que_v2 worktree of the SLR repo, pinned to an old commit. Do every
repo change, commit and `que2 submit` in the main checkout, `/home/luke/Code/SLR`, not here. Read
`/home/luke/Code/SLR/CLAUDE.md` first and follow it (code quality, `src/info/` docs, ruff/pyright
from the repo root). Python: `~/miniconda3/envs/wlasl/bin/python`. Training outputs are in
`/home/luke/Code/SLR/src/runs/`.

## Steps

1. **Check the checkout.** `git -C /home/luke/Code/SLR status` and branch. If it isn't on `main`,
   stop and report (step 8). If tracked files are modified, leave them alone: commit only your
   own files by path (`git commit -- <paths>`), and add `--allow-dirty` to `que2 submit` (the jobs
   run from HEAD, which will include your commit).

2. **Collect the trials.** Each finished trial is
   `src/runs/asl100/MViTv2_S_e/sweep_flx99idx/job<id>/run.json` (a `CompExpInfo`, see
   `src/run_types.py`). `resume.json` holds the resolved config of trials that didn't finish. Also
   look at `que2 list -g sweep2-flx99idx --json` for each job's final state. If fewer than 10
   trials have a `run.json`, stop and report: the sweep went wrong, and the seed study would rest
   on too few trials.

3. **Pick the best trial:** the lowest `results.best_val_loss` (the sweep's metric,
   `Best/Val_loss`). Note its wandb run id (`wandb.run_id`), best val loss, test top-1 (per
   instance), its swept values (backbone/classifier LR, backbone weight decay, `drop_p`, `tmax`,
   `magnitude`), the number of epochs it trained, and the epoch of its best val loss. The wandb run
   history has `Epoch` and `Loss/Val` (entity `ljgoodall2001-rhodes-university`, project `Sweeps`).

4. **Write the config** `src/configfiles/SeedComparison/MViTv2_S_e_<run id>.toml`, in the same TOML
   layout as `S3D_13idpda6.toml` and `src/configfiles/thesis_final/*.toml`, from the trial's resolved
   config. Keep everything as it is: training, optimizer, model_params, data (train/test augs),
   scheduler (including `hold_after_tmax` and `eta_min`) and stopping. Start it with a short comment
   saying where it comes from (sweep, trial id, Que_v2 job, best val loss) and how it's queued.
   **Verify it:** load it as `train_and_test.py` would (`build_admin` with a temporary
   `runs_path`, then `src.configs.load_config`), and check that its `model_dump()` equals the
   trial's config apart from `admin`. Don't go on until they match.

5. **Document it.** Add a row and a bullet for it to `src/configfiles/SeedComparison/README.md`,
   matching the existing ones (results to come). Say it's on the corrected `asl100` (0-based),
   unlike the S3D ones. Commit the config and README with a message that explains the why, ending
   with the `Co-Authored-By:` line your system prompt gives (or `Co-Authored-By: Claude
   <noreply@anthropic.com>` if none).

6. **Submit the 10 runs** from `/home/luke/Code/SLR`:

       QUE_SUBMITTER="SLR seed-comparison agent (job $QUE_JOB_ID)" que2 submit base train \
         model=MViTv2_S_e split=asl100 config=src/configfiles/SeedComparison/MViTv2_S_e_<run id>.toml \
         --seeds 1,2,3,4,5,6,7,8,9,10 --group seedcmp-MViTv2_S_e --priority 11 -p SeedComparison

   `-p SeedComparison` passes through to `src/train_and_test.py`, so the runs log to the wandb
   `SeedComparison` project like the S3D ones. Check with `que2 list -g seedcmp-MViTv2_S_e` that
   all 10 are queued, each pinned to your commit.

7. **Estimate the duration.** Use the best trial's epochs, and the hours per epoch from its Que_v2
   job (`que2 show <job id> --json`, attempts' `started_at`/`ended_at`, minus testing if you can
   tell). That gives hours per run, times 10. Each run also stops by early stopping
   (patience 15 after its best val loss), so give a range rather than one number.

8. **Report through the hub** (`~/claude-hub`, message format in its `README.md`; file name
   `inbox/<Project>/<YYYY-MM-DD HHMM> SLR <slug>.md`, with `from: SLR`):
   - To `Luke_MSc_Thesis`: what was queued (jobs, priority 11, wandb project `SeedComparison`),
     the best trial (id, best val loss, test top-1, swept values, epochs and best epoch), the
     config path and commit, and the estimate. Mention that the trial ran with the default seed
     42, so it isn't one of the 10, and that its val loss is the best of 20 trials, so expect the
     seed mean to be worse (selection bias). The thesis asked for these, on 2026-10-24: per
     instance and per class test top-1/5/10, test loss and best val loss, with mean, std and max
     deviation once the runs finish.
   - To `SLR`: a short note of what you did (or where you stopped and why).
   - Update `~/claude-hub/projects/SLR.md`: add a Recent bullet, and make its Focus line mention
     the seed comparison.
   - In `/home/luke/.claude/projects/-home-luke-Code-SLR/memory/thesis_run_list_2026_10_18.md`,
     mark the "MViTv2 seed comparison" step as queued (jobs, date, estimate).

   If you stop early at any step, still send the `SLR` message saying where you stopped and why,
   and send the thesis one saying it's delayed. Then exit.

Before you finish, check that nothing you changed is left uncommitted. Your last message should
summarise what you did; it ends up in the Que_v2 job log.
