filters = {
    "wandb" : {"project" : lambda x: x == 'SeedComparison'},
    # The project also holds seed runs of other configs (e.g. S3D_czopef0v.toml)
    "admin" : {"config_path" : lambda x: x.endswith('S3D_13idpda6.toml')},
}

drop_keys = [] #no drop keys means runs can be imported with typing

# drop_keys = [
#     ["results", "test_shuff"],
#     ["results", "check_name"],
# ]
