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
import gc

import train
import video_dataset as Dataset
import video_transforms as vt
import contrastive_losses as contr_l
import models
# from test import test_model

train_inst_path = './preprocessed/labels/asl300/train_instances_fixed_frange_bboxes_len.json'
train_clss_path = './preprocessed/labels/asl300/train_classes_fixed_frange_bboxes_len.json'
val_inst_path = './preprocessed/labels/asl300/val_instances_fixed_frange_bboxes_len.json'
val_clss_path = './preprocessed/labels/asl300/val_classes_fixed_frange_bboxes_len.json'
raw_path = '../data/WLASL2000'

base_mean = [0.43216, 0.394666, 0.37645]
base_std = [0.22803, 0.22145, 0.216989]
permute_fin = vt.get_swap_ct()
base_transform = vt.get_base()
regular_norm = vt.get_norm(base_mean, base_std)
rand_norm_aug = vt.get_rand_norm_aug(base_mean, base_std)

base_norm_fin = transforms.Compose([
  base_transform, 
  regular_norm,
  permute_fin
])

base_fin = transforms.Compose([
  base_transform,
  permute_fin
])


base_rand_norm_fin = transforms.Compose([
  base_transform,
  rand_norm_aug,
  permute_fin
])

# train_set = Dataset.VideoDataset(
#     root=raw_path,
#     instances_path=train_inst_path,
#     classes_path=train_clss_path,
#     transform=base_rand_norm_fin
# )

# contr_train_set = Dataset.ContrastiveVideoDataset(
contr_train_set = Dataset.SemiContrastiveVideoDataset(
  root=raw_path,
  instances_path=train_inst_path,  
  classes_path=train_clss_path,
  transform=base_norm_fin, 
  augmentation=base_fin #augmentation through lack of transform
)

# val_set = Dataset.VideoDataset(
#   root=raw_path,
#   instances_path=val_inst_path,
#   classes_path=val_clss_path,
#   transform=base_norm_fin, 
# )

# contr_val_set = Dataset.ContrastiveVideoDataset(
contr_val_set = Dataset.SemiContrastiveVideoDataset(
  root=raw_path,
  instances_path=val_inst_path,
  classes_path=val_clss_path,
  transform=base_norm_fin, 
  augmentation=base_fin #augmentation through lack of transform
)

# print(f"Number of training samples: {len(train_set)}")
# print(f"Number of training classes: {len(set(train_set.classes))}")
# print(f"Number of validation samples: {len(val_set)}")
# print(f"Number of validation classes: {len(set(val_set.classes))}")
print(f"Number of training samples: {len(contr_train_set)}")
print(f"Number of training classes: {len(set(contr_train_set.classes))}")
print(f"Number of validation samples: {len(contr_val_set)}")
print(f"Number of validation classes: {len(set(contr_val_set.classes))}")

# torch.manual_seed(42) #probably doesnt work because of numworkers
# train_loader = DataLoader(
#   train_set,
#   batch_size=32, 
#   shuffle=True,
#   num_workers=2, #this was 4 but I previously had issues with the computer crashing (though this was with more data)
#   drop_last=True
# )

contr_train_loader = DataLoader(
  contr_train_set,
  batch_size=32, 
  shuffle=True,
  num_workers=2, #this was 4 but I previously had issues with the computer crashing (though this was with more data)
  drop_last=True
  # collate_fn=Dataset.contrastive_collate_fn
)

# val_loader = DataLoader(
#   val_set,
#   batch_size=32,
#   shuffle=False,
#   drop_last=False,
#   num_workers=2,
#   drop_last=False
# )

contr_val_loader = DataLoader(
  contr_val_set,
  batch_size=32, 
  shuffle=False,
  num_workers=2, #this was 4 but I previously had issues with the computer crashing (though this was with more data)
  drop_last=False
  # collate_fn=Dataset.contrastive_collate_fn
)


print('-----------------------')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print('-----------------------')

model4, train_params4 = models.get_bas_r3d18()
optimizer = torch.optim.Adam(train_params4, lr=1e-4)
#TODO : try this code
# optimizer = torch.optim.Adam([
#     {'params': model.layer4.parameters(), 'lr': 1e-4},
#     {'params': model.fc.parameters(), 'lr': 1e-3}  # Higher LR for new classifier
# ])
print(len(train_params4), " trainable parameters")
loss_func = nn.CrossEntropyLoss() #TODO : try Contrastive loss

# schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(
#   optimizer,
#   mode='min',
#   factor=0.1,
#   patience=30,
# ) 

# train_losses, val_losses = train.train_model_4(
#   model=model4,
#   train_loader=train_loader,
#   optimizer=optimizer,
#   loss_func=loss_func,
#   epochs=100,
#   val_loader=val_loader,
#   # scheduler=schedular,
#   output='runs/asl300/r3d18_exp4'
# )

# model4 = model4.cpu()  # Move to CPU first (optional but safer)
# del model4
# torch.cuda.empty_cache()
# gc.collect()  # Python garbage collectio

model5, train_params5 = models.get_bas_r3d18()
optimizer = torch.optim.Adam(train_params4, lr=1e-4)
contr_loss_func = contr_l.InfoNCELoss()
super_loss_func = nn.CrossEntropyLoss()
# train_losses_cont, val_losses_cont = train.train_model_contrastive(
#   model=model5,
#   train_loader=contr_train_loader,
#   optimizer=optimizer,
#   loss_func=loss_func,
#   epochs=100,
#   val_loader=contr_val_loader,
#   output='runs/asl300/r3d18_exp7'
# )

train_losses_cont, val_losses_cont = train.train_model_semi_contrastive(
  model=model5,
  train_loader=contr_train_loader,
  optimizer=optimizer,
  contr_loss_func=contr_loss_func,
  super_loss_func=super_loss_func,
  epochs=100,
  val_loader=contr_val_loader,
  output='runs/asl300/r3d18_exp8'
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

#exp4 :
# exp3 actually had an error, was using base_rand_norm for validation as well
# so exp4 is essentially, the corrected exp3, except without schedular
#very similar results to exp3, but slightly betters


#exp 5:
#with the contrastive loss, no schedular
#we are getting heigher gpu memory utilisation ~ 10980MiB /  11264MiB
#compared to around ~7100MiB /  11264MiB (for non-contrastive)
#this loss is starting off awfully low
#exp 6: bot exp5 and exp6 has zero change in loss overtime, due to a fault in the 
#way the contrastive loss video dataset is applying transforms

#exp 7: train switch around and checking the augmentation function, but no luck

#exp 8 with semi supervised contrastive loss, otherwise same as 7