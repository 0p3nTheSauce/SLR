filters = {
    "wandb" : {"project" : lambda x: x == 'SeedComparison'}
}

drop_keys = [] #no drop keys means runs can be imported with typing

# drop_keys = [
#     ["results", "test_shuff"],
#     ["results", "check_name"],
# ]
