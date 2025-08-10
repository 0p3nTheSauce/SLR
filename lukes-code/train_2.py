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
        checkpoint_path = os.path.join(save_pathA, f'checkpoint_{str(epoch).zfill(3)}.pth') 
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

def train_big_swin(split, output, seed=420, conf):
  set_seed(seed)
  root = '../data/WLASL2000'
  labels = f'./preprocessed/labels/{split}'
  
  