from src.results import find_runs, match, same_augs

run = find_runs({"admin": {"split": lambda x: x == "asl100_worst"}})[0]

# example_runs_p = Path('./wlasl_100_worst.json')
# assert example_runs_p.exists()
# run = load_runs(example_runs_p)[0]


train_augs = run.data.train_augs
if train_augs is None:
    spec_train_augs = lambda x: x is None
else:
    spec_train_augs = {
        "normalise": lambda x, train_augs=train_augs: x == train_augs.normalise,
        "temporal_aug": lambda x, train_augs=train_augs: same_augs(
            x, train_augs.temporal_aug
        ),
        "spatial_aug": lambda x, train_augs=train_augs: same_augs(
            x, train_augs.spatial_aug
        ),
    }

test_augs = run.data.test_augs
if test_augs is None:
    spec_test_augs = lambda x: x is None
else:
    spec_test_augs = {
        "normalise": lambda x, test_augs=test_augs: x == test_augs.normalise,
        "temporal_aug": lambda x, test_augs=test_augs: same_augs(
            x, test_augs.temporal_aug
        ),
        "spatial_aug": lambda x, test_augs=test_augs: same_augs(
            x, test_augs.spatial_aug
        ),
    }

acc_cuttoff = 10

include_splits = ["asl100_cutoff_9", "asl100_bottom", "asl100_worst"]

filters = {
    "training": lambda x: match(x, run.training),
    "optimizer": lambda x: match(x, run.optimizer),
    "model_params": lambda x: match(x, run.model_params),
    "stopping": lambda x: match(x, run.stopping),
    "scheduler": lambda x: match(x, run.scheduler),
    "results": {"best_val_acc": lambda x: x > acc_cuttoff},
    "admin": {
        "model": lambda x: x == run.admin.model,
        "split": lambda x: x in include_splits,
        "config_path" : lambda x: x == run.admin.config_path
    },
    "data": {"train_augs": spec_train_augs, "test_augs": spec_test_augs},
}

drop_keys = []  # no drop keys means runs can be imported with typing

# drop_keys = [
#     ["results", "test_shuff"],
#     ["results", "check_name"],
# ]
