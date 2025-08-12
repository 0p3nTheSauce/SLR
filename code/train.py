import argparse
import torch # type: ignore
import os
import tqdm   # type: ignore
from torch.utils.tensorboard import SummaryWriter # type: ignore
import json
import utils
from utils import enum_dir
from torchvision.transforms import v2
from  torchvision.models.video.resnet import  R3D_18_Weights #, r3d_18
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import models.pytorch_r3d as resnet_3d
#local imports
from video_dataset import VideoDataset
from configs import Config
import numpy as np
import random
import wandb



def train_loop_B(r3d18B,device, confB, dataloadersB, optimizerB, schedulerB,
                 loss_funcB, writerB, logsB, saveB, save_pathB,
                 save_every=5, max_epochs=400):
  r3d18B.to(device)
  steps=0
  epoch=0
  best_val_score=0

  while steps < confB.max_steps and epoch < max_epochs:
    print(f'Step {steps}/{confB.max_steps}')
    print('-'*10)
    
    epoch+=1
    #each epoch has training and validation stage
    for phase in ['train', 'val']:
      
      if phase == 'train':
        r3d18B.train()
      else:
        r3d18B.eval()
        
      #Reset matrics for this phase
      running_loss = 0.0
      running_corrects = 0
      total_samples = 0
      # num_batches = 0
      # tot_loc_loss = 0.0  #TODO once this gets working try the fancy loss
      # tot_cls_loss = 0.0
      
      #for gradient accumulation  
      accumulated_loss = 0.0
      accumulated_steps = 0
      optimizerB.zero_grad()
    
      #Iterate over data for this phase
      for batch_idx, item in enumerate(dataloadersB[phase]):
        data, target = item['frames'], item['label_num']
        data, target = data.to(device), target.to(device)
        batch_size = data.size(0)
        total_samples += batch_size
        # num_batches += 1
        
        #Forward pass
        if phase == 'train':
          model_output = r3d18B(data)
        else:
          with torch.no_grad():
            model_output = r3d18B(data)
            
        # Calculate loss
        loss = loss_funcB(model_output, target)

        #Accumulate metrics
        running_loss += loss.item() * batch_size  
        _, predicted = model_output.max(1)
        running_corrects += predicted.eq(target).sum().item()
        

        if phase == 'train':
          scaled_loss = loss / confB.update_per_step
          scaled_loss.backward()
          
          accumulated_loss += loss.item()
          accumulated_steps += 1
          
          if accumulated_steps == confB.update_per_step:
            optimizerB.step()
            optimizerB.zero_grad()
            steps += 1
            
            # Print progress every few steps
            if steps % 10 == 0:
              avg_acc_loss = accumulated_loss / accumulated_steps
              current_acc = 100.0 * running_corrects / total_samples
              print(f'Step {steps}: Accumulated Loss: {avg_acc_loss:.4f}, '
                    f'Current Accuracy: {current_acc:.2f}%')
              
              if logsB:
                writerB.add_scalar('Loss/Train_Step', avg_acc_loss, steps) 
                writerB.add_scalar('Accuracy/Train_Step', current_acc, steps) 
            
            # Reset accumulation
            accumulated_loss = 0.0
            accumulated_steps = 0
    
      #calculate  epoch metrics
      epoch_loss = running_loss / total_samples # Average loss per sample
      epoch_acc = 100.0 * running_corrects / total_samples

      print(f'{phase.upper()} - Epoch {epoch}:')
      print(f'  Loss: {epoch_loss:.4f}')
      print(f'  Accuracy: {epoch_acc:.2f}% ({running_corrects}/{total_samples})')
      try:
        for i, param_group in enumerate(optimizerB.param_groups):
          if logsB:
            writerB.add_scalar(f'LearningRate/Group_{i}', param_group['lr'], epoch) 
          print(f"Group {i} learning rate: {param_group['lr']}")
      except Exception as e:
        print(f'Failed to print all learning rates due to {e}')
        
      # Log epoch metrics
      if logsB:
        writerB.add_scalar(f'Loss/{phase.capitalize()}', epoch_loss, epoch) 
        writerB.add_scalar(f'Accuracy/{phase.capitalize()}', epoch_acc, epoch) 
      
      # Validation specific logic
      if phase == 'val':
          # Save best model
          if epoch_acc > best_val_score:
              best_val_score = epoch_acc
              model_name = os.path.join(save_pathB, f'best.pth') 
              torch.save(r3d18B.state_dict(), model_name)
              print(f'New best model saved: {model_name} (Acc: {epoch_acc:.2f}%)')
          
          # Step scheduler with validation loss
          schedulerB.step() 
          
          print(f'Best validation accuracy so far: {best_val_score:.2f}%')
      
      # Save checkpoint
    if saveB and (epoch % save_every == 0 or not (steps < confB.max_steps and epoch < 400)):
        checkpoint_data = {
            'epoch': epoch,
            'steps': steps,
            'model_state_dict': r3d18B.state_dict(),
            'optimizer_state_dict': optimizerB.state_dict(),
            'scheduler_state_dict': schedulerB.state_dict(),
            'best_val_score': best_val_score
        }
        checkpoint_path = os.path.join(save_pathB, f'checkpoint_{str(epoch).zfill(3)}.pth') 
        torch.save(checkpoint_data, checkpoint_path)
        print(f'Checkpoint saved: {checkpoint_path}')

  print('Finished training successfully')
  if logsB:
    writerB.close()


def set_seed(seed=42):
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  random.seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


def train_loop(model, datasets, wandb_run, load=None, weights=None, save_every=5,
                 recover=False):

  config = wandb_run.config
  
  train_transforms, test_transforms = transforms
  
  #setup data
  
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
  
  #model, metrics, optimizer, schedular, loss
  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)
  
  steps = 0
  epoch = 0 
  best_val_score=0
  
  optimizer = optim.AdamW(model.parameters(), 
                          lr = config.optimizer['lr'],
                          eps = config.optimizer['eps'],
                          weight_decay= config.optimizer['weight_decay'])
  
  scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                   T_max=config.scheduler['tmax'],
                                                   eta_min=config.scheduler['eta_min'])
  
  loss_func = nn.CrossEntropyLoss()
  
  #if we are continuing from last checkpoint, set 'load'
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
      model.load_state_dict(checkpoint['model_state_dict'])
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
        model.train()
      else:
        model.eval()

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
          model_output = model(data)
        else:
          with torch.no_grad():
            model_output = model(data)
        
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
          torch.save(model.state_dict(), model_name)
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
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'scheduler_state_dict': scheduler.state_dict(),
          'best_val_score': best_val_score
        }
        checkpoint_path = os.path.join(config.admin['save_path'], f'checkpoint_{str(epoch).zfill(3)}.pth')
        

        torch.save(checkpoint_data, checkpoint_path)
          
        print(f'Checkpoint saved: {checkpoint_path}')
        
  print('Finished training successfully')
  wandb_run.finish()