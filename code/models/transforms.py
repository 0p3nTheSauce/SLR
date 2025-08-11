import torch.nn as nn
import torch
import torchvision.models.video as video_models
from torchvision.models.video.resnet import BasicBlock, Bottleneck, Conv3DSimple,\
  Conv3DNoTemporal, Conv2Plus1D, VideoResNet, WeightsEnum, _ovewrite_named_param
from typing import Union, Callable, Sequence, Optional, Any
from  torchvision.models.video.resnet import r3d_18, R3D_18_Weights
from torchvision.transforms import v2

def basic_transforms(config):
  '''used by kinetics'''
  base_mean = [0.43216, 0.394666, 0.37645]
  base_std = [0.22803, 0.22145, 0.216989]
  kinetics_final = v2.Compose([
    v2.Lambda(lambda x: x.float() / 255.0),
    v2.Normalize(mean=base_mean, std=base_std),
    v2.Lambda(lambda x: x.permute(1,0,2,3)) 
  ])
  train_transforms = v2.Compose([
    v2.RandomCrop(config.frame_size),
    v2.RandomHorizontalFlip(),
    kinetics_final])
  test_transforms = v2.Compose([
    v2.CenterCrop(config.frame_size),
    kinetics_final])
  return train_transforms, test_transforms

def base_jitter_transforms(config):
  '''using Kinetics pretraining, plus
  some extra augmentation'''
  base_mean = [0.43216, 0.394666, 0.37645]
  base_std = [0.22803, 0.22145, 0.216989]
  kinetics_final = v2.Compose([
    v2.Lambda(lambda x: x.float() / 255.0),
    v2.Normalize(mean=base_mean, std=base_std),
    v2.Lambda(lambda x: x.permute(1,0,2,3)) 
  ])
  # b = (0.4, 2)
  # c = (0.4, 2)
  # s = (0.4, 2)
  # h = (0.5, 0.5)
  train_transforms = v2.Compose([
    v2.RandomCrop(config.frame_size),
    v2.RandomHorizontalFlip(),
    v2.ColorJitter(config.brightness, config.contrast,
                   config.saturation, config.hue),
    kinetics_final])
  test_transforms = v2.Compose([
    v2.CenterCrop(config.frame_size),
    kinetics_final])
  return train_transforms, test_transforms