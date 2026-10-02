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

sweep_key_map = {
    # optimizer
    "backbone_init_lr":        "optimizer.backbone_init_lr",
    "classifier_init_lr":      "optimizer.classifier_init_lr",
    "backbone_weight_decay":   "optimizer.backbone_weight_decay",
    "classifier_weight_decay": "optimizer.classifier_weight_decay",
    "eps":                     "optimizer.eps",

    # model
    "drop_p":                  "model_params.drop_p",

    # scheduler (WarmRestartInfo)
    "t0":                      "scheduler.t0",
    "tmult":                   "scheduler.tmult",
    "eta_min":                 "scheduler.eta_min",
    "start_factor":            "scheduler.warm_up.start_factor",
    "end_factor":              "scheduler.warm_up.end_factor",
    "warmup_epochs":           "scheduler.warm_up.warmup_epochs",

    # training
    "batch_size":              "training.batch_size",
    "update_per_step":         "training.update_per_step",
    "max_epoch":               "training.max_epoch",

    # temporal aug -- now maps to both train and test
    "max_wobble":              "data.train_augs.temporal_aug.type:chunked.max_wobble",
    "target_length": [
        "data.train_augs.temporal_aug.type:chunked.target_length",
        "data.test_augs.temporal_aug.type:uniform.target_length"
    ],

    # spatial aug -- now maps to both train and test
    "hflip_p":                 "data.train_augs.spatial_aug.type:HORIZONTAL_FLIP.p",
    "frame_size": [
        "data.train_augs.spatial_aug.type:Centre_crop.frame_size",
        "data.test_augs.spatial_aug.type:Centre_crop.frame_size"
    ],
    "magnitude":               "data.train_augs.spatial_aug.type:RANDAUGMENT.magnitude",
    "num_ops":                 "data.train_augs.spatial_aug.type:RANDAUGMENT.num_ops",
    "num_magnitude_bins":      "data.train_augs.spatial_aug.type:RANDAUGMENT.num_magnitude_bins",

    # early stopping
    "patience":                "stopping.patience",
    "min_delta":               "stopping.min_delta",
}