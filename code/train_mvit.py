import argparse
import torch 
import os

from utils import enum_dir
from torchvision.transforms import v2
# from  torchvision.models.video.resnet import  R3D_18_Weights #, r3d_18
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
# import models.pytorch_r3d as resnet_3d


#local imports
from video_dataset import VideoDataset
from configs import load_config, print_config
import numpy as np
import random
import wandb

def train(wandb_run, load=None, weights=None, save_every=5, recover=False):
  config = wandb_run.config
  
  #setup transforms
  mean=[0.45, 0.45, 0.45]
  std=[0.225, 0.225, 0.225]
  
  mvitv2s_final = v2.Compose([
    v2.Lambda(lambda x: x.float() / 255.0),
    v2.Normalize(mean=mean, std=std),
    v2.Lambda(lambda x: x.permute(1,0,2,3)) 
  ])
  
  #setup dataset
  train_transforms = v2.Compose([v2.RandomCrop(224),
                                 v2.RandomHorizontalFlip(),
                                 mvitv2s_final])
  test_transforms = v2.Compose([v2.CenterCrop(224),
                                mvitv2s_final])
  
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
  
  # mvitv2s = mvit_v2_s(MViT_V2_S_Weights.KINETICS400_V1, num_classes=num_classes)
  mvitv2s = mvit_v2_s(MViT_V2_S_Weights.KINETICS400_V1)
  
  #does not allow altering num_classes, so need to replace head
  original_linear = mvitv2s.head[1]  # The Linear layer
  in_features = original_linear.in_features
  mvitv2s.head = nn.Sequential(
    nn.Dropout(0.5, inplace=True),
    nn.Linear(in_features, num_classes)
  )
  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  mvitv2s.to(device)
  
  steps = 0
  epoch = 0 
  best_val_score=0
  
  optimizer = optim.AdamW(mvitv2s.parameters(), 
                          lr = config.optimizer['lr'],
                          eps = config.optimizer['eps'],
                          weight_decay= config.optimizer['weight_decay'])
  
  scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                   T_max=config.scheduler['tmax'],
                                                   eta_min=config.scheduler['eta_min'])
  
  loss_func = nn.CrossEntropyLoss()
  
  #check if we are continuing, if so set 'load'
  if recover: 
    fname = ''
    if os.path.exists(config.admin['save_path']):
      files = sorted(os.listdir(config.admin['save_path']))
      if len(files) > 0:
        fname = files[-1]
      else:
        raise ValueError(f"Directory: {config.admin['save_path']} is empty")
    else:
      raise ValueError(f"Could not find directory: {config.admin['save_path']}")
        #setup recovery
    
    load = os.path.join(config.admin['save_path'], fname)
  
  if load:
    if os.path.exists(load):
      checkpoint = torch.load(load, map_location=device)
      mvitv2s.load_state_dict(checkpoint['model_state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
      epoch = checkpoint['epoch'] + 1
      steps = checkpoint['steps']
      if 'best_val_score' in checkpoint:
        best_val_score = checkpoint['best_val_score']
      print(f"Resuming from epoch {epoch}, steps {steps}")
      print(f"Loaded model from {load}")
    else:
      cont = input(f"Checkpoint {load} does not exist, starting from scratch? [y]")
      if cont.lower() != 'y':
        return
      epoch = 0
      steps = 0
      
  #train it
  while steps < config.training['max_steps'] and epoch < config.training['max_epoch']:
    print(f"Step {steps}/{config.training['max_steps']}")
    print('-'*10)
    
    epoch += 1
    #training and validation stage
    for phase in ['train', 'val']:
      
      if phase == 'train':
        mvitv2s.train()
      else:
        mvitv2s.eval()

      #Reset matrics for this phase
      running_loss = 0.0
      running_corrects = 0
      total_samples = 0

      #for gradient accumulation  
      accumulated_loss = 0.0
      accumulated_steps = 0
      optimizer.zero_grad()
      
      for batch_idx, item in enumerate(dataloaders[phase]):
        data, target = item['frames'], item['label_num']
        data, target = data.to(device), target.to(device)
        batch_size = data.size(0)
        total_samples += batch_size
        
        if phase == 'train':
          model_output = mvitv2s(data)
        else:
          with torch.no_grad():
            model_output = mvitv2s(data)
        
        #Accumulate metrics
        loss = loss_func(model_output, target)        
        running_loss += loss.item() * batch_size  
        _, predicted = model_output.max(1)
        running_corrects += predicted.eq(target).sum().item()
        
        if phase == 'train':
          scaled_loss = loss / config.training['update_per_step']
          scaled_loss.backward()
          
          accumulated_loss += loss.item()
          accumulated_steps += 1
          
          if accumulated_steps == config.training['update_per_step']:
            optimizer.step()
            optimizer.zero_grad()
            steps += 1
            
            # Print progress every few steps
            if steps % 10 == 0:
              avg_acc_loss = accumulated_loss / accumulated_steps
              current_acc = 100.0 * running_corrects / total_samples
              print(f'Step {steps}: Accumulated Loss: {avg_acc_loss:.4f}, '
                    f'Current Accuracy: {current_acc:.2f}%')
              
              wandb_run.log({
                'Loss/Train_Step': avg_acc_loss,
                'Accuracy/Train_Step': current_acc,
                'Step': steps
              })
            
            # Reset accumulation
            accumulated_loss = 0.0
            accumulated_steps = 0
      
      #calculate  epoch metrics
      epoch_loss = running_loss / total_samples # Average loss per sample
      epoch_acc = 100.0 * running_corrects / total_samples
      
      print(f'{phase.upper()} - Epoch {epoch}:')
      print(f'  Loss: {epoch_loss:.4f}')
      print(f'  Accuracy: {epoch_acc:.2f}% ({running_corrects}/{total_samples})')
      
      wandb_run.log({
        f'Loss/{phase.capitalize()}': epoch_loss,
        f'Accuracy/{phase.capitalize()}': epoch_acc,
        'Epoch': epoch
      })
      
      # Validation specific logic
      if phase == 'val':
        # Save best model
        if epoch_acc > best_val_score:
          best_val_score = epoch_acc
          model_name = os.path.join(config.admin['save_path'], f'best.pth') 
          torch.save(mvitv2s.state_dict(), model_name)
          print(f'New best model saved: {model_name} (Acc: {epoch_acc:.2f}%)')
      
        # Step scheduler with validation loss
        scheduler.step() 
        
        print(f'Best validation accuracy so far: {best_val_score:.2f}%')
      
    # Save checkpoint
    if epoch % save_every == 0 or not (steps < config.training['max_steps'] 
                                       and epoch < config.training['max_epoch']):
        checkpoint_data = {
          'epoch': epoch,
          'steps': steps,
          'model_state_dict': mvitv2s.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'scheduler_state_dict': scheduler.state_dict(),
          'best_val_score': best_val_score
        }
        checkpoint_path = os.path.join(config.admin['save_path'], f'checkpoint_{str(epoch).zfill(3)}.pth')
        

        torch.save(checkpoint_data, checkpoint_path)
          
        print(f'Checkpoint saved: {checkpoint_path}')
        
  print('Finished training successfully')
  wandb_run.finish()
  
def main():
  splits_available = ['asl100', 'asl300']
  model = 'MViT_V2_S'
  
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
  config = load_config(arg_dict)
  
  # Create tags for wandb
  tags = [
      args.split,
      model,
      f"exp-{exp_no}"
  ]
  
  if args.recover:
      tags.append("recovered")
  
  # Print summary
  # title = f"""Training {model} on split {args.split}
  #             Experiment no: {exp_no}
  #             Raw videos at: {args.root}
  #             Labels at: {args.labels}
  #             Saving files to: {args.save_path}
  #             Recovering: {args.recover}
  #             Config: {args.config_path}
  #             """
  # print(title)
  
  # print("Available config keys:")
  # print("-" * 40)
  # for key, value in config.items():
  #     print(f"{key}: {value}")
  # print("-" * 40)
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
    train(run, recover=args.recover)
  else:
    print("Training cancelled")

if __name__ =='__main__':
  main()