# AugComparison sweep

The following sweep was performed: 

| Type    | Strategy       | Parameter     | Values                       |
|---------|----------------|---------------|------------------------------|
| Spatial | Auto Augment   | `policy`      | `CIFAR10`, `SVHN`, `IMAGENET`|
| Spatial | Rand Augment   | `num_ops`     | 2, 3                         |
| Spatial | Rand Augment   | `magnitude`   | 6, 8, 10                     |
| Spatial | Resized Crop   | -             | -                            |
| Temporal| Chunked Sampler| -             | -                            |
| Temporal| Focal Sampler  | `mean`        | 0.5                          |
| Temporal| Focal Sampler  | `std`         | 0.15, 0.25, 0.35             |
| Temporal| Speed Sampler  | `deviation`   | 0.1, 0.2, 0.3                |
| Temporal| Indices        | `displacement`| 2, 4, 6                      |
| Control | No Augmentation| -             | -                            |
| Control | Baseline       | -             | -                            |

See the [results](../../results/aug_comparison/results.ipynb).