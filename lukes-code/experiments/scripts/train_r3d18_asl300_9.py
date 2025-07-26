from configs import Config
from train import train_run_r3d18_1


if __name__ == '__main__':
  split = 'asl300'
  root = '../data/WLASL2000'
  labels=f'./preprocessed/labels/{split}'
  output=f'runs/{split}/r3d18_exp9'
  config_path = f'./configfiles/{split}.ini'
  configs = Config(config_path)
  train_run_r3d18_1(configs=configs, root=root, labels=labels, output=output)
  
  #TODO: move to correct output dir, currently in asl100, should be in asl300