frames = 32

key_set = [
    ['training', "batch_size_equivalent"],
    ["optimizer", "eps"],
    ["optimizer", "backbone_init_lr"],
    ["optimizer", "backbone_weight_decay"],
    ["optimizer", "classifier_init_lr"],
    ["optimizer", "classifier_weight_decay"],
    ["model_params", "drop_p"],
    ["data", "num_frames"],
    ["data", "frame_size"],
    ['scheduler'],
    # ['early_stopping', 'patience']
]

criterions = [
    lambda x: x == 8,
    lambda x: x == 1e-05,
    lambda x: x == 1e-4,
    lambda x: x == 1e-3,
    lambda x: x == 1e-3,
    lambda x: x == 1e-3,
    lambda x: x == 0.5,
    lambda x: x == frames,
    lambda x: x == 224,
    lambda x: x is None,
    # lambda x: x == 15
    
]

assert len(key_set) == len(criterions), f'key_set and criterions must be of the same length, but got {len(key_set)} and {len(criterions)} respectively'

out_suffix = f"frames_{frames}.json"