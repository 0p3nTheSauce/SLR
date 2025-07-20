import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F

from PIL import Image
import torchvision.models.video as video_models
import torch.nn as nn
import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter # type: ignore
import json

from train import train_model_4
import video_dataset as Dataset
import video_transforms as vt

# from test import test_model

train_inst_path = './preprocessed/labels/asl300/train_instances_fixed_frange_bboxes_len.json'
train_clss_path = './preprocessed/labels/asl300/train_classes_fixed_frange_bboxes_len.json'
val_inst_path = './preprocessed/labels/asl300/val_instances_fixed_frange_bboxes_len.json'
val_clss_path = './preprocessed/labels/asl300/val_classes_fixed_frange_bboxes_len.json'
raw_path = '../data/WLASL2000'

base_mean = [0.43216, 0.394666, 0.37645]
base_std = [0.22803, 0.22145, 0.216989]
permute_fin = lambda x: x.permute(1,0,2,3)
base_transform = vt.get_base_transform()

rand_norm_aug =lambda x: \
  vt.RandomNormalizationAugmentation(x,
    base_mean=base_mean,
    base_std=base_std,
    mean_var=0.05,
    std_var=0.03,
    prob=0.5,                                 
  )

base_rand_norm = transforms.Compose([
  base_transform,
  transforms.Lambda(rand_norm_aug),
  transforms.Lambda(permute_fin)
])

train_set = Dataset.VideoDataset(
    root=raw_path,
    instances_path=train_inst_path,
    classes_path=train_clss_path,
    transform=base_rand_norm
)
val_set = Dataset.VideoDataset(
    root=raw_path,
    instances_path=val_inst_path,
    classes_path=val_clss_path,
    transform=base_rand_norm
)
print(f"Number of training samples: {len(train_set)}")
print(f"Number of training classes: {len(set(train_set.classes))}")
print(f"Number of validation samples: {len(val_set)}")
print(f"Number of validation classes: {len(set(val_set.classes))}")

# torch.manual_seed(42) #probably doesnt work because of numworkers
train_loader = DataLoader(
  train_set,
  batch_size=32, 
  shuffle=True,
  num_workers=2, #this was 4 but I previously had issues with the computer crashing (though this was with more data)
  drop_last=True
)

print(f'Train loader:\n{train_loader}')

val_loader = DataLoader(
  val_set,
  batch_size=32,
  shuffle=False,
  drop_last=False,
  num_workers=2
)
print('-----------------------')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print('-----------------------')
model = video_models.r3d_18(pretrained=True) #weights=R3D_18_Weights.KINETICS400_V1d
num_classes = 300
model.fc = nn.Linear(model.fc.in_features, num_classes)

for param in model.parameters():
  param.requires_grad = True
  
for layer_name in ['layer4', 'fc']:
  if hasattr(model, layer_name):
    for param in getattr(model, layer_name).parameters():
      param.requires_grad = True
      
for name, param in model.named_parameters():
  if param.requires_grad:
    print(f"Training parameter: {name}")
  else:
    print(f"Freezing parameter: {name}")
    
for name, module in model.named_modules():
  if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
    # Check if this BatchNorm is in a frozen layer
    is_in_frozen_layer = not any(unfreeze_layer in name for unfreeze_layer in ['layer4', 'fc'])
    
    if is_in_frozen_layer:
      module.eval()
      module.track_running_stats = False
      print(f"Set {name} to eval mode (frozen layer)")
      
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(trainable_params, lr=1e-4)  #this learning rate might be too high
#TODO : try this code
# optimizer = torch.optim.Adam([
#     {'params': model.layer4.parameters(), 'lr': 1e-4},
#     {'params': model.fc.parameters(), 'lr': 1e-3}  # Higher LR for new classifier
# ])
print(len(trainable_params), "trainable parameters")
loss_func = nn.CrossEntropyLoss() #TODO : try Contrastive loss

schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(
  optimizer,
  mode='min',
  factor=0.1,
  patience=30,
) 

train_losses, val_losses = train_model_4(
  model=model,
  train_loader=train_loader,
  optimizer=optimizer,
  loss_func=loss_func,
  epochs=100,
  val_loader=val_loader,
  scheduler=schedular,
  output='runs/asl300/r3d18_exp3'
)

#exp0 : with 
# schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(
#   optimizer,
#   mode='min',
#   factor=0.1,
#   patience=15,   NB
# ) 

# exp1 : better : without schedular

# exp2 : best? : with 
# schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(
#   optimizer,
#   mode='min',
#   factor=0.1,
#   patience=30,
# ) 

#exp3 : without
# transform0 = transforms.Compose([
#     transforms.Lambda(lambda x: Dataset.correct_num_frames(x, 16)),  # (T, C, H, W)
#     transforms.Lambda(lambda x: x.float() / 255.0),  # Convert to float and normalize to [0,1]
#     transforms.Lambda(lambda x: F.interpolate(x, size=(112, 112), mode='bilinear', align_corners=False)),  # Resize after normalization
#     transforms.Lambda(lambda x: Dataset.normalise(x,  mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])),  # Normalize per channel
#     transforms.Lambda(lambda x: x.permute(1, 0, 2, 3)),  # (T, C, H, W) -> (C, T, H, W)
# ]) #The transform that got the best result

#      : with
# base_transform = transforms.Compose([
#   transforms.Lambda(lambda x: vd.correct_num_frames(x, 16)),
#   transforms.Lambda(lambda x: x.float() / 255.0),
#   transforms.Lambda(lambda x: F.interpolate(x, size=(112, 112), 
#                     mode='bilinear', align_corners=False)),
# ])
# rand_norm_aug = lambda x: vd.RandomNormalizationAugmentation(x,
#   base_mean=base_mean,
#   base_std=base_std,
#   mean_var=0.05,
#   std_var=0.03,
#   prob=augment_prob
# )
# base_rand_norm = transforms.Compose([
#   base_transform, 
#   transforms.Lambda(rand_norm_aug)
# ])

# schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(
#   optimizer,
#   mode='min',
#   factor=0.1,
#   patience=30,
# ) 
