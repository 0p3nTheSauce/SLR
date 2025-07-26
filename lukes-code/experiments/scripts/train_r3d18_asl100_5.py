from configs import Config
from train import train_run_r3d18_1


if __name__ == '__main__':
  root = '../data/WLASL2000'
  labels='./preprocessed/labels/asl100'
  output='runs/asl100/r3d18_exp5'
  config_path = './configfiles/asl100.ini'
  configs = Config(config_path)
  train_run_r3d18_1(configs=configs, root=root, labels=labels, output=output)
  
  