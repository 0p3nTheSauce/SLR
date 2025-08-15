import os
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import argparse
#local imports
from video_dataset import VideoDataset
from models.pytorch_swin3d import Swin3DBig_basic
import train
from utils import enum_dir
from configs import print_config, load_config
import wandb

def train_swin(wandb_run, load=None, save_every=5, recover=False, seed=None):
  config = wandb_run.config

  #swin specific mean and std from pytorch
  mean=[0.485, 0.456, 0.406]
  std=[0.229, 0.224, 0.225]
  
  swin3db_final = v2.Compose([
    v2.Lambda(lambda x: x.float() / 255.0),
    v2.Normalize(mean=mean, std=std),
    v2.Lambda(lambda x: x.permute(1,0,2,3)) 
  ])
  
  #setup dataset
  train_transforms = v2.Compose([v2.RandomCrop(config.data['frame_size']),
                                 v2.RandomHorizontalFlip(),
                                 swin3db_final])
  test_transforms = v2.Compose([v2.CenterCrop(config.data['frame_size']),
                                swin3db_final])
  
  train_instances = os.path.join(config.admin['labels'], 'train_instances_fixed_frange_bboxes_len.json')
  val_instances = os.path.join(config.admin['labels'],'val_instances_fixed_frange_bboxes_len.json' )
  train_classes = os.path.join(config.admin['labels'], 'train_classes_fixed_frange_bboxes_len.json')
  val_classes = os.path.join(config.admin['labels'],'val_classes_fixed_frange_bboxes_len.json' )
  
  dataset = VideoDataset(config.admin['root'],train_instances, train_classes,
    transforms=train_transforms, num_frames=config.data['num_frames'])
  dataloader = DataLoader(dataset, batch_size=config.training['batch_size'],
    shuffle=True, num_workers=2,pin_memory=True)
  num_classes = len(set(dataset.classes))
  
  val_dataset = VideoDataset(config.admin['root'], val_instances, val_classes,
    transforms=test_transforms, num_frames=config.data['num_frames'])
  val_dataloader = DataLoader(val_dataset,
    batch_size=config.training['batch_size'], shuffle=True, num_workers=2,pin_memory=False)
  val_classes = len(set(val_dataset.classes))
  assert num_classes == val_classes
  
  dataloaders = {'train': dataloader, 'val': val_dataloader}
  
  swin3db = Swin3DBig_basic(num_classes, config.model_params['drop_p'])
  
  train.train_loop(
    model=swin3db,
    dataloaders=dataloaders,
    wandb_run=wandb_run,
    load=load,
    save_every=save_every,
    recover=recover,
    seed=seed
  )
  
def main():
  model = 'Swin3D_B'
  splits_available = ['asl100', 'asl300']
  
  parser = argparse.ArgumentParser(description='Train a mvit model')
  
  #runs
  parser.add_argument('-e', '--experiment',type=int, help='Experiment number (e.g. 10)', required=True)
  parser.add_argument('-r', '--recover', action='store_true', help='Recover from last checkpoint')
  parser.add_argument('-ms', '--max_steps', type=int,help='gradient accumulation')
  parser.add_argument('-me', '--max_epoch', type=int,help='mixumum training epoch')
  parser.add_argument('-c' , '--config', help='path to config .ini file')
  
  #data
  parser.add_argument('-s', '--split',type=str, help='the class split (e.g. asl100)', required=True)
  parser.add_argument('-nf','--num_frames', type=int, help='video length')
  parser.add_argument('-fs', '--frame_size', type=int, help='width, height')
  parser.add_argument('-bs', '--batch_size', type=int,help='data_loader')
  parser.add_argument('-us', '--update_per_step', type=int, help='gradient accumulation')
  
  args = parser.parse_args()
  
  if args.split not in splits_available:
    raise ValueError(f"Sorry {args.split} not processed yet")
  
  exp_no = str(int(args.experiment)).zfill(3)
  
  args.model = model
  args.exp_no = exp_no
  args.root = '../data/WLASL/WLASL2000'
  args.labels = f'./preprocessed/labels/{args.split}'
  output = f'runs/{args.split}/{model}_exp{exp_no}'
  
  if not args.recover: #fresh run
    output = enum_dir(output, make=True)  
  
  save_path = f'{output}/checkpoints'
  if not args.recover:
    args.save_path = enum_dir(save_path, make=True) 
  
  # Set config path
  if args.config:
    args.config_path = args.config
  else:
    args.config_path = f'./configfiles/{args.split}/{model}_{exp_no}.ini'
  
  # Load config
  arg_dict = vars(args)
  config = load_config(arg_dict, verbose=True)
  
  # Create tags for wandb
  tags = [
      args.split,
      model,
      f"exp-{exp_no}"
  ]
  
  if args.recover:
    tags.append("recovered")
  
  print_config(config)
  
  
  proceed = input("Confirm: y/n: ")
  if proceed.lower() == 'y':
    run = wandb.init(
      entity='ljgoodall2001-rhodes-university',
      project='WLASL-SLR',
      name=f"{model}_{args.split}_exp{exp_no}",
      tags=tags,
      config=config      
    )
      
    # Start training
    train_swin(run, recover=args.recover)
  else:
    print("Training cancelled")

if __name__ =='__main__':
  main()
  
  