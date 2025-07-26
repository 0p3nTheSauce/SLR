import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR, MultiStepLR

# from torchvision import transforms
from torchvision.transforms import v2
from  torchvision.models.video.resnet import r3d_18, R3D_18_Weights

import video_transforms as vt
from video_dataset import VideoDataset
import numpy as np
from torch.utils.data import DataLoader
from utils import enum_dir
#plan:
#implement 3d resnet, based on i3d
#actually, first train resnet 
from torch.utils.tensorboard import SummaryWriter # type: ignore
from configs import Config

def run(configs, root='../data/WLASL2000',labels='./preprocessed/labels/asl300',
        output='runs/exp_0', logs='logs', save='checkpoints', load=None,
        weights=R3D_18_Weights.DEFAULT, save_every=5):
  print(configs)
  
  base_mean = [0.43216, 0.394666, 0.37645]
  base_std = [0.22803, 0.22145, 0.216989]
  
  r3d18_final = v2.Compose([
    v2.Lambda(lambda x: x.float() / 255.0),
    # v2.Lambda(lambda x: vt.normalise(x, base_mean, base_std)),
    v2.Normalize(mean=base_mean, std=base_std),
    v2.Lambda(lambda x: x.permute(1,0,2,3)) 
  ])
  
  #setup dataset 
  train_transforms = v2.Compose([v2.RandomCrop(224),
                                 v2.RandomHorizontalFlip(),
                                 r3d18_final])
  test_transforms = v2.Compose([v2.CenterCrop(224),
                                r3d18_final])
  
  train_instances = os.path.join(labels, 'train_instances_fixed_frange_bboxes_len.json')
  val_instances = os.path.join(labels,'val_instances_fixed_frange_bboxes_len.json' )
  train_classes = os.path.join(labels, 'train_classes_fixed_frange_bboxes_len.json')
  val_classes = os.path.join(labels,'val_classes_fixed_frange_bboxes_len.json' )
  
  dataset = VideoDataset(root,train_instances, train_classes,
    transforms=train_transforms, num_frames=32)
  dataloader = DataLoader(dataset, batch_size=configs.batch_size,
    shuffle=True, num_workers=2,pin_memory=True)
  num_classes = len(set(dataset.classes))
  
  val_dataset = VideoDataset(root, val_instances, val_classes,
    transforms=test_transforms, num_frames=32)
  val_dataloader = torch.utils.data.DataLoader(val_dataset,
    batch_size=configs.batch_size, shuffle=True, num_workers=2,pin_memory=False)
  val_classes = len(set(val_dataset.classes))
  assert num_classes == val_classes
  
  dataloaders = {'train': dataloader, 'test': val_dataloader}
  datasets = {'train': dataset, 'test': val_dataset}
  
  #setup model
  r3d18 = r3d_18( weights=weights)

  in_features = r3d18.fc.in_features  # Store before replacement
    
  r3d18.fc = nn.Sequential(
    nn.Dropout(p=configs.drop_p),
    nn.Linear(in_features, num_classes)
  )
  
  for param in r3d18.parameters(): #freeze layers
    param.requires_grad = False
  
  for param in r3d18.fc.parameters():
    param.requires_grad = True #start with the classifier
  
  
  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  r3d18.to(device)
  
  lr = configs.init_lr
  weight_decay = configs.adam_weight_decay
  optimizer = optim.Adam(r3d18.parameters(), lr=lr, weight_decay=weight_decay)
  
  num_steps_per_update = configs.update_per_step #gradient accumulation
  steps=0
  epoch=0
  
  best_val_score=0

  # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
  #                                               patience=5, factor=0.3)
  # max_epochs = 200
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-5)
  loss_func = nn.CrossEntropyLoss()
  #setup recovery 
  if load:
    if os.path.exists(load):
      checkpoint = torch.load(load, map_location=device)
      r3d18.load_state_dict(checkpoint['model_state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      scheduler.load_state_dict(checkpoint['schedular_state_dict'])
      begin_epoch = checkpoint['epoch'] + 1
      print(f"Resuming from epoch {begin_epoch}")
      print(f"Loaded model from {load}")
    else:
      cont = input(f"Checkpoint {load} does not exist, starting from scratch? [y]")
      if cont.lower() != 'y':
        return
  
  #admin
  if output:
    if load is None: #fresh run, fresh folder
      output = enum_dir(output, make=True) 
    print(f"Output directory set to: {output}")
    
  if save:
    save_path = os.path.join(output, save)
    if load is None:
      save_path = enum_dir(save_path, make=True)
    print(f"Save directory set to: {save_path}")
  
  if logs:
    logs_path = os.path.join(output, logs)
    if load is None:
      logs_path = enum_dir(logs_path, make=True)
    print(f"Logs directory set to: {logs_path}")
    writer = SummaryWriter(logs_path) #watching loss
  
  #train it
  # pbar = tqdm.tqdm(range(begin_epoch, epochs), desc="Training R3D")
  while steps < configs.max_steps and epoch < 400:
    print(f'Step {steps}/{configs.max_steps}')
    print('-'*10)
    
    epoch+=1
    #each epoch has training and validation stage
    for phase in ['train', 'val']:
      
      if phase == 'train':
        r3d18.train()
      else:
        r3d18.eval()
        
      #Reset matrics for this phase
      running_loss = 0.0
      running_corrects = 0
      total_samples = 0
      num_batches = 0
      # tot_loc_loss = 0.0  #TODO once this gets working try the fancy loss
      # tot_cls_loss = 0.0
      
      #for gradient accumulation  
      accumulated_loss = 0.0
      accumulated_steps = 0
      optimizer.zero_grad()
    
      #Iterate over data for this phase
      for batch_idx, (data, target) in enumerate(dataloaders[phase]):
        data, target = data.to(device), target.to(device)
        batch_size = data.size(0)
        total_samples += batch_size
        num_batches += 1
        
        #Forward pass
        if phase == 'train':
          model_output = r3d18(data)
        else:
          with torch.no_grad():
            model_output = r3d18(data)
            
        # Calculate loss
        loss = loss_func(model_output, target)

        #Accumulate metrics
        running_loss += loss.item() * batch_size  
        _, predicted = model_output.max(1)
        running_corrects += predicted.eq(target).sum().item()
        

        if phase == 'train':
          scaled_loss = loss / num_steps_per_update
          scaled_loss.backward()
          
          accumulated_loss += loss.item()
          accumulated_steps += 1
          
          if accumulated_steps == num_steps_per_update:
            optimizer.step()
            optimizer.zero_grad()
            steps += 1
            
            # Print progress every few steps
            if steps % 10 == 0:
              avg_acc_loss = accumulated_loss / accumulated_steps
              current_acc = 100.0 * running_corrects / total_samples
              print(f'Step {steps}: Accumulated Loss: {avg_acc_loss:.4f}, '
                    f'Current Accuracy: {current_acc:.2f}%')
              
              if logs:
                writer.add_scalar('Loss/Train_Step', avg_acc_loss, steps)
                writer.add_scalar('Accuracy/Train_Step', current_acc, steps)
            
            # Reset accumulation
            accumulated_loss = 0.0
            accumulated_steps = 0
    
      #calculate  epoch metrics
      epoch_loss = running_loss / total_samples # Average loss per sample
      epoch_acc = 100.0 * running_corrects / total_samples

      print(f'{phase.upper()} - Epoch {epoch}:')
      print(f'  Loss: {epoch_loss:.4f}')
      print(f'  Accuracy: {epoch_acc:.2f}% ({running_corrects}/{total_samples})')
      
      # Log epoch metrics
      if logs:
        writer.add_scalar(f'Loss/{phase.capitalize()}', epoch_loss, epoch)
        writer.add_scalar(f'Accuracy/{phase.capitalize()}', epoch_acc, epoch)
      
      # Validation specific logic
      if phase == 'val':
          # Save best model
          if epoch_acc > best_val_score:
              best_val_score = epoch_acc
              model_name = os.path.join(save_path, f'best_asl{num_classes}.pth')
              torch.save(r3d18.state_dict(), model_name)
              print(f'New best model saved: {model_name} (Acc: {epoch_acc:.2f}%)')
          
          # Step scheduler with validation loss
          scheduler.step(epoch_loss)
          
          print(f'Best validation accuracy so far: {best_val_score:.2f}%')
      
     # Save checkpoint
    if save and (epoch % save_every == 0 or not (steps < configs.max_steps and epoch < 400)):
        checkpoint_data = {
            'epoch': epoch,
            'steps': steps,
            'model_state_dict': r3d18.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_score': best_val_score
        }
        checkpoint_path = os.path.join(save_path, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint_data, checkpoint_path)
        print(f'Checkpoint saved: {checkpoint_path}')
        
    
  print('Finished training successfully')

if __name__ == '__main__':
  # parser = argparse.ArgumentParser()
  # parser.add_argument('-save_model', type=str)
  # parser.add_argument('-root', type=str)
  # parser.add_argument('--num_class', typ=int)

  # args = parser.parse_args()

  # torch.manual_seed(0)
  # np.random.seed(0)

  # torch.backends.cudnn.deterministic = True   for determinism
  # torch.backends.cudnn.benchmark = False
  root = '../data/WLASL2000'
  labels='./preprocessed/labels/asl100'
  output='runs/asl100/r3d18_exp4'
  config_path = './configfiles/asl100.ini'
  configs = Config(config_path)
  run(configs=configs, root=root, labels=labels, output=output)