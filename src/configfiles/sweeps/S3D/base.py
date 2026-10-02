base_config = {
        "training": {"batch_size": None, "update_per_step": None, "max_epoch": None,},
        "optimizer": {
            "eps": None, "backbone_init_lr": None, "backbone_weight_decay": None,
            "classifier_init_lr": None, "classifier_weight_decay": None,
        },
        "model_params": {"drop_p": None},
        "data": {
            "train_augs": {
                "normalise": True,
                "temporal_aug": [{"type": "uniform", "target_length": None}],
                "spatial_aug": [
                    {"type": "HORIZONTAL_FLIP", "p": None},
                    {"type": "Random_crop", "frame_size": None},
                ],
            },
            "test_augs": {
                "normalise": True,
                "temporal_aug": [{"type": "uniform", "target_length": None}],
                "spatial_aug": [{"type": "Centre_crop", "frame_size": None}],
            },
        },
        "stopping": {
            "type": "early_stopper", "metric": "loss",
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

    # training
    "batch_size":              "training.batch_size",
    "update_per_step":         "training.update_per_step",
    "max_epoch":               "training.max_epoch",

    # temporal aug -- now maps to both train and test
    "target_length": [
        "data.train_augs.temporal_aug.type:uniform.target_length",
        "data.test_augs.temporal_aug.type:uniform.target_length"
    ],

    # spatial aug -- now maps to both train and test
    "hflip_p":                 "data.train_augs.spatial_aug.type:HORIZONTAL_FLIP.p",
    "frame_size": [
        "data.train_augs.spatial_aug.type:Random_crop.frame_size",
        "data.test_augs.spatial_aug.type:Centre_crop.frame_size"
    ],

    # early stopping
    "patience":                "stopping.patience",
    "min_delta":               "stopping.min_delta",
}