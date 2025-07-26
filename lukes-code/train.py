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

def setup_optimizer_scheduler(model, total_epochs=100): 
  # Separate learning rates - exactly what you want
  param_groups = [
    {
      'params': model.backbone.parameters(),
      'lr': 1e-5,  # Low LR for pretrained backbone
      'weight_decay': 1e-4
    },
    {
      'params': model.classifier.parameters(), 
      'lr': 1e-3,  # Higher LR for new classifier
      'weight_decay': 1e-4
    }
  ]
  
  optimizer = optim.AdamW(param_groups, betas=(0.9, 0.999))
  
  # Single scheduler affects both groups proportionally
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                         T_max=total_epochs,
                                                         eta_min=1e-6)
  
  return optimizer, scheduler

def freeze_layers(model, frozen_layers):
  """Freeze specified layers of the model"""
  for layer_name in frozen_layers:
    if hasattr(model.backbone, layer_name):
      layer = getattr(model.backbone, layer_name)
      for param in layer.parameters():
        param.requires_grad = False
      print(f"Frozen layer: {layer_name}")
    else:
      print(f"Warning: Layer {layer_name} not found")

def train_run_r3d18_1(configs, root='../data/WLASL2000',labels='./preprocessed/labels/asl300',
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
  
  dataloaders = {'train': dataloader, 'val': val_dataloader}
  datasets = {'train': dataset, 'val': val_dataset}
  
  #setup model
  # r3d18 = r3d_18( weights=weights)

  # in_features = r3d18.fc.in_features  # Store before replacement
    
  # r3d18.fc = nn.Sequential(
  #   nn.Dropout(p=configs.drop_p),
  #   nn.Linear(in_features, num_classes)
  # )
  r3d18 = resnet_3d.Resnet3D18_basic(num_classes=num_classes, drop_p=configs.drop_p,
                                     weights=weights)
  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  r3d18.to(device)
  
  num_steps_per_update = configs.update_per_step #gradient accumulation
  steps=0
  epoch=0
  
  best_val_score=0

  # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
  #                                               patience=5, factor=0.3)
  # lr = configs.init_lr
  # weight_decay = configs.adam_weight_decay
  # optimizer = optim.AdamW(r3d18.parameters(), lr=lr, weight_decay=weight_decay)
  # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
  #                                                        T_max=configs.t_max,
  #                                                        eta_min=configs.eta_min)
  param_groups = [
    {
      'params': r3d18.backbone.parameters(),
      'lr': 1e-5,  # Low LR for pretrained backbone
      'weight_decay': 1e-4
    },
    {
      'params': r3d18.classifier.parameters(), 
      'lr': 1e-3,  # Higher LR for new classifier
      'weight_decay': 1e-4
    }
  ]
  
  optimizer = optim.AdamW(param_groups, betas=(0.9, 0.999))
  
  # Single scheduler affects both groups proportionally
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                         T_max=100,
                                                         eta_min=1e-6)
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
                writer.add_scalar('Loss/Train_Step', avg_acc_loss, steps) # type: ignore
                writer.add_scalar('Accuracy/Train_Step', current_acc, steps) # type: ignore
            
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
        writer.add_scalar(f'Loss/{phase.capitalize()}', epoch_loss, epoch) # type: ignore
        writer.add_scalar(f'Accuracy/{phase.capitalize()}', epoch_acc, epoch) # type: ignore
      
      # Validation specific logic
      if phase == 'val':
          # Save best model
          if epoch_acc > best_val_score:
              best_val_score = epoch_acc
              model_name = os.path.join(save_path, f'best.pth') # type: ignore
              torch.save(r3d18.state_dict(), model_name)
              print(f'New best model saved: {model_name} (Acc: {epoch_acc:.2f}%)')
          
          # Step scheduler with validation loss
          scheduler.step(epoch_loss) # type: ignore
          
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
        checkpoint_path = os.path.join(save_path, f'checkpoint_{epoch}.pth') # type: ignore
        torch.save(checkpoint_data, checkpoint_path)
        print(f'Checkpoint saved: {checkpoint_path}')
        
    
  print('Finished training successfully')

  
def run_2(configs, root='../data/WLASL2000',labels='./preprocessed/labels/asl300',
        label_suffix='_fixed_frange_bboxes_len.json', output='runs/exp_0', logs='logs',
        save='checkpoints', load=None, save_every=5):
  # print(configs)
  
  train_transforms, test_transforms = configs.get_transforms()
  
  train_instances = os.path.join(labels, f'train_instances{label_suffix}')
  val_instances = os.path.join(labels, f'val_instances{label_suffix}' )
  train_classes = os.path.join(labels, f'train_classes{label_suffix}')
  val_classes = os.path.join(labels,f'val_classes{label_suffix}' )
  
  dataset = VideoDataset(root,train_instances, train_classes,
    transforms=train_transforms, num_frames=configs.num_frames)
  dataloader = DataLoader(dataset, batch_size=configs.batch_size,
    shuffle=True, num_workers=2,pin_memory=True)
  num_classes = len(set(dataset.classes))
  
  val_dataset = VideoDataset(root, val_instances, val_classes,
    transforms=test_transforms, num_frames=configs.num_frames)
  val_dataloader = torch.utils.data.DataLoader(val_dataset,
    batch_size=configs.batch_size, shuffle=True, num_workers=2,pin_memory=False)
  val_classes = len(set(val_dataset.classes))
  assert num_classes == val_classes
  
  dataloaders = {'train': dataloader, 'val': val_dataloader}
  datasets = {'train': dataset, 'val': val_dataset}
  
  model = configs.create_model() #this handles new fc, freezing, different learning rates

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)
  
  num_steps_per_update = configs.update_per_step #gradient accumulation
  steps=0
  epoch=0
  
  best_val_score=0

  param_groups = [ #TODO: this code is currently r3d18 specific, but could be more generalisable
    {
      'params': model.backbone.parameters(),
      'lr': configs.backbone_init_lr,  # Low LR for pretrained backbone
      'weight_decay': configs.backbone_weight_decay
    },
    {
      'params': model.classifier.parameters(), 
      'lr': configs.classifier_init_lr,  # Higher LR for new classifier
      'weight_decay': configs.classifier_weight_decay
    }
  ]
  
  optimizer = optim.AdamW(param_groups)
  
  # Single scheduler affects both groups proportionally
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                         T_max=configs.t_max,
                                                         eta_min=configs.eta_min)
  loss_func = nn.CrossEntropyLoss()
  #setup recovery 
  if load:
    if os.path.exists(load):
      checkpoint = torch.load(load, map_location=device)
      model.load_state_dict(checkpoint['model_state_dict'])
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
        model.train()
      else:
        model.eval()
        
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
          model_output = model(data)
        else:
          with torch.no_grad():
            model_output = model(data)
            
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
                writer.add_scalar('Loss/Train_Step', avg_acc_loss, steps) # type: ignore
                writer.add_scalar('Accuracy/Train_Step', current_acc, steps) # type: ignore
            
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
        writer.add_scalar(f'Loss/{phase.capitalize()}', epoch_loss, epoch) # type: ignore
        writer.add_scalar(f'Accuracy/{phase.capitalize()}', epoch_acc, epoch) # type: ignore
      
      # Validation specific logic
      if phase == 'val':
          # Save best model
          if epoch_acc > best_val_score:
              best_val_score = epoch_acc
              model_name = os.path.join(save_path, f'best.pth') # type: ignore
              torch.save(model.state_dict(), model_name)
              print(f'New best model saved: {model_name} (Acc: {epoch_acc:.2f}%)')
          
          # Step scheduler with validation loss
          scheduler.step(epoch_loss) # type: ignore
          
          print(f'Best validation accuracy so far: {best_val_score:.2f}%')
      
     # Save checkpoint
    if save and (epoch % save_every == 0 or not (steps < configs.max_steps and epoch < 400)):
        checkpoint_data = {
            'epoch': epoch,
            'steps': steps,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_score': best_val_score
        }
        checkpoint_path = os.path.join(save_path, f'checkpoint_{str(epoch).zfill(3)}.pth') # type: ignore
        torch.save(checkpoint_data, checkpoint_path)
        print(f'Checkpoint saved: {checkpoint_path}')
        
    
  print('Finished training successfully')

def main():
    # parser = argparse.ArgumentParser(description='Train a model')
    
    # parser.add_argument('model', help='model to use')
    # parser.add_argument('-o', '--output', help='Output file (default: stdout)')
    # parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    # parser.add_argument('-n', '--lines', type=int, default=10, help='Number of lines to process')
    
    # args = parser.parse_args()
    
    # print(f"Processing {args.input_file}")
    # if args.output:
    #     print(f"Output will go to {args.output}")
    # if args.verbose:
    #     print("Verbose mode enabled")
    # print(f"Processing {args.lines} lines")
    direc = '/home/luke/ExtraStorage/WLASL/lukes-code/runs/exp_enum'
    enum = enum_dir(direc)
    print(f"Enumerated directory: \n{enum}")
    
if __name__ == '__main__':
    main()