from src.results import match
from src.run_types import CUTOFF_9_NAMES

sampler16 = {"target_length": 16, "max_wobble": 0, "type": "og", "randomise": False}
sampler32 = {"target_length": 32, "max_wobble": 0, "type": "og", "randomise": False}
samplers = [sampler16, sampler32]
randCrop = {"frame_size": 224, "type": "Random_crop"}
randHorizFlip = {"type": "HORIZONTAL_FLIP", "p": 0.5}
centreCrop = {"frame_size": 224, "type": "Centre_crop"}

model_params = {"drop_p": 0.5}
training = {"batch_size_equivalent": 8}
optimizer = {
    "eps": 1e-05,
    "backbone_init_lr": 0.0001,
    "backbone_weight_decay": 0.001,
    "classifier_init_lr": 0.001,
    "classifier_weight_decay": 0.001,
}
stopping = {
    "type": "early_stopper",
    "metric": "loss",
    "phase": "val",
    "mode": "min",
    "min_delta": 0.01,
    "patience": 50,
}
scheduler = {
    "type": "CosineAnnealingWarmRestarts",
    "t0": 10,
    "tmult": 1,
    "eta_min": 0.0,
    "warm_up": None,
}

data = (
    {
        "train_augs": {
            "normalise": lambda x: x == True,
            "temporal_aug": lambda x: (
                len(x) == 1 and any(match(x[0], s) for s in samplers)
            ),
            "spatial_aug": lambda x: (
                len(x) == 2
                and any(match(t, randCrop) for t in x)
                and any(match(t, randHorizFlip) for t in x)
            ),
        },
        "test_augs": {
            "normalise": lambda x: x == True,
            "temporal_aug": lambda x: (
                len(x) == 1 and any(match(x[0], s) for s in samplers)
            ),
            "spatial_aug": lambda x: len(x) == 1 and match(x[0], centreCrop),
        },
    },
)

acc_cuttoff = 10
# avail_models = [
#     "MViTv2_S",
#     "MViTv2_S_16x4",
#     "MViTv2_S_16x4_e",
#     "MViTv2_B_32x3",
#     "MViTv2_B_32x3_r",
#     "MViTv2_S_e",
# ]
# ignore_models = ['S3D','MViTv2_S_e', 'MViTv2_S']


filters = {
    "training": lambda x: match(x, training),
    "optimizer": lambda x: match(x, optimizer),
    "model_params": lambda x: match(x, model_params),
    "stopping": lambda x: match(x, stopping),
    "scheduler": lambda x: match(x, scheduler),
    "results": {"best_val_acc": lambda x: x > acc_cuttoff},
    "admin": {
        # "model": lambda x: x not in ignore_models,
        "split": lambda x: x in CUTOFF_9_NAMES},
}

drop_keys = [] #no drop keys means runs can be imported with typing

# drop_keys = [
#     ["results", "test_shuff"],
#     ["results", "check_name"],
# ]
