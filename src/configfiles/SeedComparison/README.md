# Random seed comparison

Each seed comparison reruns one config 10 times, varying only the seed in the range [1,10]. They
are in this Wandb [project](https://wandb.ai/ljgoodall2001-rhodes-university/SeedComparison),
told apart by `admin.config_path`.

| Run id    | Split     | Sweep         | Test top1 acc | Test av loss | Best val loss | Config |
|-----------|-----------|---------------|---------------|--------------|---------------|--------|
| 13idpda6  | asl100    | [S3D sweep 4](../sweeps/S3D/exp004/)   | 75.581395     | 0.976105     | 0.936 | [S3D_13idpda6.toml](./S3D_13idpda6.toml) |
| czopef0v  | asl100_cutoff_9 | [S3D sweep 5](../sweeps/S3D/exp005/) | 64.341085 | 1.460382 | 1.186 | [S3D_czopef0v.toml](./S3D_czopef0v.toml) |

- `13idpda6` was the best `S3D` model by test average loss. It ran on the old `asl100` split,
  but its seed runs use `asl100_cutoff_9`. Results are in
  [seed_comparison](../../results/seed_comparison/results.ipynb).
- `czopef0v` is the best trial of sweep 5 (no augmentation, no scheduler). It checks whether
  that sweep's large-`eps` optimizer regime is reliably better than sweep 6's best, or just a
  lucky seed. Not run yet.
