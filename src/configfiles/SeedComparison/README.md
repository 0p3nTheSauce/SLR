# Random seed comparison

This seed sweep uses the current best `S3D` model in terms of test average loss, with an original metrics:


| Run id    | Split     | Sweep         | Test top1 acc | Test av loss |
|-----------|-----------|---------------|---------------|--------------|
| 13idpda6  | asl100    | [S3D sweep 4](../sweeps/S3D/exp004/)   | 75.581395     | 0.976105     |


They all used the same [config](./S3D_13idpda6.toml), but vary the seed in the range [1,10]. They are in this Wandb [project](https://wandb.ai/ljgoodall2001-rhodes-university/SeedComparison).