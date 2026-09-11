import json

from src.results import find_runs, match, same_augs

runs = find_runs({"admin": {"wandb": {"run_id" : lambda x : x == 'j8v7g110'}}})
if len(runs) >= 1:
    admins = [run.admin.model_dump() for run in runs]    
    raise ValueError(f'More than one spec run found: {json.dumps(admins, indent=4)}')
elif len(runs) == 0:
    raise ValueError('No runs found')


run = runs[0]

sampler16 = {"target_length": 16, "max_wobble": 0, "type": "og", "randomise": False}
sampler32 = {"target_length": 32, "max_wobble": 0, "type": "og", "randomise": False}

temp_aug16 = [sampler16]
temp_aug32 = [sampler32]
randCrop = {"frame_size": 224, "type": "Random_crop"}
randHorizFlip = {"type": "HORIZONTAL_FLIP", "p": 0.5}
centreCrop = {"frame_size": 224, "type": "Centre_crop"}
train_spatial = [randCrop, randHorizFlip]
test_spatial = [centreCrop]

training = {"batch_size_equivalent": 8}
stopping_50 = {
    "type": "early_stopper",
    "metric": "loss",
    "phase": "val",
    "mode": "min",
    "min_delta": 0.01,
    "patience": 50,
}

stopping_15 = {
    "type": "early_stopper",
    "metric": "loss",
    "phase": "val",
    "mode": "min",
    "min_delta": 0.01,
    "patience": 15,
}

train_augs = run.data.train_augs
if train_augs is None:
    spec_train_augs = lambda x: x is None
else:
    spec_train_augs = {
        "normalise": lambda x, train_augs=train_augs: x == train_augs.normalise,
        "strict_size": lambda x, train_augs=train_augs: x == train_augs.strict_size,
        "temporal_aug": lambda x: same_augs(x, temp_aug16) or same_augs(x, temp_aug32),
        "spatial_aug": lambda x: same_augs(x, train_spatial)
    }

test_augs = run.data.test_augs
if test_augs is None:
    spec_test_augs = lambda x: x is None
else:
    spec_test_augs = {
        "normalise": lambda x, test_augs=test_augs: x == test_augs.normalise,
        "strict_size": lambda x, test_augs=test_augs: x == test_augs.strict_size,
        "temporal_aug": lambda x: same_augs(x, temp_aug16) or same_augs(x, temp_aug32),
        "spatial_aug": lambda x: same_augs(x, test_spatial)
    }

acc_cuttoff = 10

include_splits = ["asl100_cutoff_9","asl300_cutoff_9", "asl1000_cutoff_9", "asl2000_cutoff_9"]

filters = {
    "training": lambda x: match(x, training),
    "optimizer": lambda x: match(x, run.optimizer),
    "model_params": lambda x: match(x, run.model_params),
    "stopping": lambda x: match(x, stopping_15) or match(x, stopping_50),
    "scheduler": lambda x: match(x, run.scheduler),
    "results": {"best_val_acc": lambda x: x > acc_cuttoff},
    "admin": {
        "split": lambda x: x in include_splits,
    },
    "data": {"train_augs": spec_train_augs, "test_augs": spec_test_augs},
}

drop_keys = []  # no drop keys means runs can be imported with typing

# drop_keys = [
#     ["results", "test_shuff"],
#     ["results", "check_name"],
# ]
