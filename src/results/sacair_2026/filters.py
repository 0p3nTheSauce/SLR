sampler16 = {"target_length": 16, "max_wobble": 0, "type": "og", "randomise": False}
sampler32 = {"target_length": 32, "max_wobble": 0, "type": "og", "randomise": False}
samplers = [sampler16, sampler32]
randCrop = {"frame_size": 224, "type": "Random_crop"}
randHorizFlip = {"type": "HORIZONTAL_FLIP", "p": 0.5}
centreCrop = {"frame_size": 224, "type": "Centre_crop"}

model_params = {"drop_p": 0.5}
training = {"batch_size_equivalent": 8}
optimizer = {
    "eps": 1e-5,
    "backbone_init_lr": 1e-5,
    "backbone_weight_decay": 1e-3,
    "classifier_init_lr": 1e-3,
    "classifier_weight_decay": 1e-3,
}
stopping = {
    "type": "early_stopper", 
    "metric": "loss",
    "phase": "val",
    "mode": "min",
    "min_delta": 0.01,
    "patience": 15,
}

acc_cuttoff = 10
avail_models = ["MViTv2_S", "MViTv2_S_16x4", "MViTv2_B_32x3", "MViTv2_S_e",]

filters = {
    # "training": lambda x: x == training,
    # "optimizer": lambda x: x == optimizer,
    # "model_params": lambda x: x == model_params,
    # "data": {
    #     "train_augs": {
    #         "normalise": lambda x: x == True,
    #         "temporal_aug": lambda x: len(x) == 1 and x[0] in samplers,
    #         "spatial_aug": lambda x: (
    #             len(x) == 2 and randCrop in x and randHorizFlip in x
    #         ),
    #     },
    #     "test_augs": {
    #         "normalise": lambda x: x == True,
    #         "temporal_aug": lambda x: len(x) == 1 and x[0] in samplers,
    #         "spatial_aug": lambda x: len(x) == 1 and x[0] == centreCrop,
    #     },
    # },
    # "stopping": lambda x: x == stopping,
    "scheduler": lambda x: x is None,
    "results": {"best_val_acc": lambda x: x > acc_cuttoff},
    "admin": {"model": lambda x: x in avail_models},
}

drop_keys = []

# drop_keys = [
#     ["results", "test_shuff"],
#     ["results", "check_name"],
# ]



