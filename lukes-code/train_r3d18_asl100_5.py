from configs import Config
from train import train_run_r3d18_1


if __name__ == '__main__':
  root = '../data/WLASL2000'
  labels='./preprocessed/labels/asl100'
  # output='runs/asl100/r3d18_exp5'   this changes slightly
  output='runs/asl100/r3d18_exp008'
  # config_path = './configfiles/asl100.ini'   and this
  config_path = './configfiles/asl100/r3d18_005.ini'   
  configs = Config(config_path) #yea i hope this is still compatible lol
  train_run_r3d18_1(configs=configs, root=root, labels=labels, output=output)
  
  