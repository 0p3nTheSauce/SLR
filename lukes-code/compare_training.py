# %% [markdown]
# # still trying to figure out why old training was better

# %%
from configs import Config
from train import train_run_r3d18_1, run_2
from torchvision.transforms import v2
import os
from video_dataset import VideoDataset
from torch.utils.data import DataLoader
import torch
import models.pytorch_r3d as resnet_3d
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
from utils import enum_dir
from torch.utils.tensorboard import SummaryWriter 

# %%
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed()

# %%
root = '../data/WLASL2000'
split = 'asl100'
labels = f'./preprocessed/labels/{split}'
output = f'runs/{split}/compr3d18_000'

conf_pathA = f'./configfiles/{split}/r3d18_005.ini'
conf_pathB = f'./configfiles/{split}/r3d18_007.ini'

confA = Config(conf_pathA)
confB = Config(conf_pathB)

print(confA)
print('-'*10)
print(confB)

# %% [markdown]
# ## Comparing transforms
# 
# ### train.train_run_r3d18_1

# %%
base_mean = [0.43216, 0.394666, 0.37645]
base_std = [0.22803, 0.22145, 0.216989]


r3d18_final = v2.Compose([
  v2.Lambda(lambda x: x.float() / 255.0),
  # v2.Lambda(lambda x: vt.normalise(x, base_mean, base_std)),
  v2.Normalize(mean=base_mean, std=base_std),
  v2.Lambda(lambda x: x.permute(1,0,2,3)) 
])

#setup dataset 
train_transformsA = v2.Compose([v2.RandomCrop(224),
                                v2.RandomHorizontalFlip(),
                                r3d18_final])
test_transformsA = v2.Compose([v2.CenterCrop(224),
                              r3d18_final])

# %%
print(train_transformsA)
print(test_transformsA)

# %% [markdown]
# ### train.run_2

# %%
train_transformsB, test_transformsB = confB.get_transforms()

# %%
print(train_transformsB)
print(test_transformsB)

# %%
A_objs = [ test_transformsA, train_transformsA]
B_objs = [ test_transformsB, train_transformsB]
for i, (obja, objb) in enumerate(zip(A_objs, B_objs)):
  print(i)
  assert(str(obja) == str(objb))

# %% [markdown]
# ## Setup data
# 

# %%
train_instances = os.path.join(labels, 'train_instances_fixed_frange_bboxes_len.json')
val_instances = os.path.join(labels,'val_instances_fixed_frange_bboxes_len.json' )
train_classes = os.path.join(labels, 'train_classes_fixed_frange_bboxes_len.json')
val_classes = os.path.join(labels,'val_classes_fixed_frange_bboxes_len.json' )
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
print(train_instances)
print(train_classes)
print(val_instances)
print(val_classes)
print(device)

# %% [markdown]
# ### train.train_run_r3d18_1
# 

# %%
train_setA = VideoDataset(root,train_instances, train_classes,
                          transforms=train_transformsA, num_frames=32)
train_loaderA = DataLoader(train_setA, batch_size=confA.batch_size,
    shuffle=True, num_workers=2,pin_memory=True)
num_classesA  = len(set(train_setA.classes))

val_setA = VideoDataset(root, val_instances, val_classes,
    transforms=test_transformsA, num_frames=32)
val_loaderA = DataLoader(val_setA,
    batch_size=confA.batch_size, shuffle=True, num_workers=2,pin_memory=False)
assert num_classesA == len(set(val_setA.classes))

dataloadersA = {'train': train_loaderA, 'val':val_loaderA}

# %%
print(train_setA)
print(train_loaderA)
print(num_classesA)
print(val_setA)
print(val_loaderA)

# %% [markdown]
# ### train.run_2

# %%
train_setB = VideoDataset(root,train_instances, train_classes,
                          transforms=train_transformsB, num_frames=confB.num_frames)
train_loaderB = DataLoader(train_setB, batch_size=confB.batch_size,
                          shuffle=True, num_workers=2,pin_memory=True)
num_classesB = len(set(train_setB.classes))

val_setB = VideoDataset(root, val_instances, val_classes,
    transforms=test_transformsB, num_frames=confB.num_frames)
val_loaderB = DataLoader(val_setB,
    batch_size=confB.batch_size, shuffle=True, num_workers=2,pin_memory=False)
val_classesB = len(set(val_setB.classes))
assert num_classesB == val_classesB 
assert num_classesB == confB.num_classes

dataloadersB = {'train': train_loaderB, 'val': val_loaderB}

# %%
print(train_setB)
print(train_loaderB)
print(num_classesB)
print(val_setB)
print(val_loaderB)

# %% [markdown]
# ## Setup model
# 
# ### train.train_run_r3d18_1

# %%
r3d18A = resnet_3d.Resnet3D18_basic(num_classes=num_classesA,
                                    drop_p=confA.drop_p)
print(r3d18A)

# %% [markdown]
# ### train.run_2

# %%
r3d18B = confB.create_model()
print(r3d18B)

# %%
assert(str(r3d18A) == str(r3d18B))

# %% [markdown]
# ## Setup optimizer
# 
# ### train.train_runr3d18_1 

# %%
param_groupsA = [
    {
      'params': r3d18A.backbone.parameters(),
      'lr': 1e-5,  # Low LR for pretrained backbone
      'weight_decay': 1e-4
    },
    {
      'params': r3d18A.classifier.parameters(), 
      'lr': 1e-3,  # Higher LR for new classifier
      'weight_decay': 1e-4
    }
  ]

optimizerA = optim.AdamW(param_groupsA, betas=(0.9, 0.999))

# %%
# print(param_groupsA)
print()
print(optimizerA)

# %% [markdown]
# ### train.run_2

# %%
param_groupsB = [ 
  {
    'params': r3d18B.backbone.parameters(),
    'lr': confB.backbone_init_lr,  # Low LR for pretrained backbone
    'weight_decay': confB.backbone_weight_decay
  },
  {
    'params': r3d18B.classifier.parameters(), 
    'lr': confB.classifier_init_lr,  # Higher LR for new classifier
    'weight_decay': confB.classifier_weight_decay
  }
]

# optimizer = optim.AdamW(param_groupsB, eps=confB.adam_eps) this was only done for exp11
optimizerB = optim.AdamW(param_groupsB) #this was for exp7

# %%
# print(param_groupsB)
print()
print(optimizerB)

# %%
assert(str(param_groupsA[0]) == str(param_groupsB[0]))

# %%
# assert(str(param_groupsA[1]) == str(param_groupsB[1]))

# %%
assert(str(param_groupsA[1]['lr']) == str(param_groupsB[1]['lr']))

# %%
assert(str(param_groupsA[1]['weight_decay']) == str(param_groupsB[1]['weight_decay']))

# %%
# assert(str(param_groupsA[1]['params']) == str(param_groupsB[1]['params']))

# %%
# print(param_groupsA[1]['params'])

# %%
# print(param_groupsB[1]['params'])

# %% [markdown]
# i suppose not too unusual that the classifier parameters don't match, but do they have the same shapes

# %%
assert(str(optimizerA) == str(optimizerB))

# %% [markdown]
# ## Setup scheduler & loss func
# 
# ### train.train_r3d18_1

# %%
schedulerA = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerA,
                                                        T_max=100,
                                                        eta_min=1e-6)
loss_funcA = nn.CrossEntropyLoss()

# %%
print(schedulerA)
print()
print(loss_funcA)

# %% [markdown]
# ### train.run_2

# %%
schedulerB = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerB,
                                                         T_max=confB.t_max,
                                                         eta_min=confB.eta_min)
loss_funcB = nn.CrossEntropyLoss()
  

# %%
print(schedulerB)
print()
print(loss_funcB)

# %% [markdown]
# ## Training loop:

# %%
load = None
# if output:
#   if load is None: #fresh run, fresh folder
#     output = enum_dir(output, make=True) 
#   print(f"Output directory set to: {output}")
print(f"Output directory set to: {output}")
save_every=5

# %%
saveA='checkpointsA'
saveB='checkpointsB'
# for save in [saveA, saveB]:
# # if save:
#   save_path = os.path.join(output, save)
#   if load is None:
#     save_path = enum_dir(save_path, make=True)
#   print(f"Save directory set to: {save_path}")
save_pathA = os.path.join(output, saveA)
save_pathB = os.path.join(output, saveB)
print(f"Save directory set to: {save_pathA}")
print(f"Save directory set to: {save_pathB}")

# %%
logsA = 'logsA'
logsB = 'logsB'
# if logs:
# for logs in [logsA, logsB]:
#   logs_path = os.path.join(output, logs)
#   if load is None:
#     logs_path = enum_dir(logs_path, make=True)
#   print(f"Logs directory set to: {logs_path}")
logs_pathA = os.path.join(output, logsA)
logs_pathB = os.path.join(output, logsB)
writerA = SummaryWriter(logs_pathA) 
writerB = SummaryWriter(logs_pathB) 
print(f"Logs directory set to: {logs_pathA}")
print(f"Logs directory set to: {logs_pathB}")


# %% [markdown]
# ### train.train_runr3d18_1

# %%
def train_loop_A(r3d18A, confA,optimizerA, loss_funcA, device,
                 dataloadersA, logsA, writerA, save_pathA, saveA,
                 schedulerA, save_every=5, max_epoch=400):
  r3d18A.to(device)
  steps=0
  epoch=0
  best_val_score=0
  num_steps_per_update = confA.update_per_step
  
  while steps < confA.max_steps and epoch < max_epoch:
    print(f'Step {steps}/{confA.max_steps}')
    print('-'*10)
    
    epoch+=1
    #each epoch has training and validation stage
    for phase in ['train', 'val']:
      
      if phase == 'train':
        r3d18A.train()
      else:
        r3d18A.eval()
        
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
      optimizerA.zero_grad()
    
      #Iterate over data for this phase
      # for batch_idx, (data, target) in enumerate(dataloaders[phase]):
      for batch_idx, item in enumerate(dataloadersA[phase]):
        data, target = item['frames'], item['label_num'] #for compatibility
        data, target = data.to(device), target.to(device)
        batch_size = data.size(0)
        total_samples += batch_size
        num_batches += 1
        
        #Forward pass
        if phase == 'train':
          model_output = r3d18A(data)
        else:
          with torch.no_grad():
            model_output = r3d18A(data)
            
        # Calculate loss
        loss = loss_funcA(model_output, target)

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
            optimizerA.step()
            optimizerA.zero_grad()
            steps += 1
            
            # Print progress every few steps
            if steps % 10 == 0:
              avg_acc_loss = accumulated_loss / accumulated_steps
              current_acc = 100.0 * running_corrects / total_samples
              print(f'Step {steps}: Accumulated Loss: {avg_acc_loss:.4f}, '
                    f'Current Accuracy: {current_acc:.2f}%')
              
              if logsA:
                writerA.add_scalar('Loss/Train_Step', avg_acc_loss, steps) 
                writerA.add_scalar('Accuracy/Train_Step', current_acc, steps)
            
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
      if logsA:
        writerA.add_scalar(f'Loss/{phase.capitalize()}', epoch_loss, epoch) 
        writerA.add_scalar(f'Accuracy/{phase.capitalize()}', epoch_acc, epoch) 
      
      # Validation specific logic
      if phase == 'val':
          # Save best model
          if epoch_acc > best_val_score:
              best_val_score = epoch_acc
              model_name = os.path.join(save_pathA, f'best.pth') 
              torch.save(r3d18A.state_dict(), model_name)
              print(f'New best model saved: {model_name} (Acc: {epoch_acc:.2f}%)')
          
          # Step scheduler with validation loss
          # scheduler.step(epoch_loss) # type: ignore
          schedulerA.step() 
          
          print(f'Best validation accuracy so far: {best_val_score:.2f}%')
      
      # Save checkpoint
    if saveA and (epoch % save_every == 0 or not (steps < confA.max_steps and epoch < 400)):
        checkpoint_data = {
            'epoch': epoch,
            'steps': steps,
            'model_state_dict': r3d18A.state_dict(),
            'optimizer_state_dict': optimizerA.state_dict(),
            'scheduler_state_dict': schedulerA.state_dict(),
            'best_val_score': best_val_score
        }
        checkpoint_path = os.path.join(save_pathA, f'checkpoint_{epoch}.pth') 
        torch.save(checkpoint_data, checkpoint_path)
        print(f'Checkpoint saved: {checkpoint_path}')
        
    
  print('Finished training successfully')
  

# %% [markdown]
# ### train.run_2

# %%
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

# %% [markdown]
# ## set to train

# %%
train_loop_A(r3d18A, confA, optimizerA, loss_funcA, device, dataloadersA,
             logsA, writerA, save_pathA, saveA, schedulerA, max_epoch=100) #lets not run for too long
train_loop_B(r3d18B, device, confB, dataloadersB, optimizerB, schedulerB,
             loss_funcB, writerB, logsB, saveB, save_pathB, max_epochs=100)


