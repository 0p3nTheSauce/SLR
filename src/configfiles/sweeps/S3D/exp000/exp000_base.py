base_config = {
        "training": {"batch_size": None, "update_per_step": None},
        "optimizer": {
            "eps": None, "backbone_init_lr": None, "backbone_weight_decay": None,
            "classifier_init_lr": None, "classifier_weight_decay": None,
        },
        "model_params": {"drop_p": None},
        "scheduler": {"type": "CosineAnnealingWarmRestarts", "t0": None, "tmult": None, "eta_min": None},
        "data": {
            "train_augs": {
                "normalise": True,
                "temporal_aug": [{"type": "chunked", "max_wobble": None, "target_length": None}],
                "spatial_aug": [
                    {"type": "HORIZONTAL_FLIP", "p": None},
                    {"type": "Centre_crop", "frame_size": None},
                    {"type": "RANDAUGMENT", "num_ops": None, "magnitude": None,
                     "num_magnitude_bins": None, "interpolation": "bilinear"},
                ],
            },
            "test_augs": {
                "normalise": True,
                "temporal_aug": [{"type": "uniform", "target_length": None}],
                "spatial_aug": [{"type": "Centre_crop", "frame_size": None}],
            },
        },
        "stopping": {
            "max_epoch": None, "type": "early_stopper", "metric": "loss",
            "phase": "val", "mode": "min", "patience": None, "min_delta": None,
        },
    }