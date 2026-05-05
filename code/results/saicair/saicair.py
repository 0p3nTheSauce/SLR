acc_cuttoff = 10
avail_models = ['MViTv2_S_16x4', 'MViTv2_B_32x3', 'S3D']

additional_modifications = {
    'results': {
        'best_val_acc': lambda x: x > acc_cuttoff
    },
    'admin': {
        'model': lambda x: x in avail_models
    }
}