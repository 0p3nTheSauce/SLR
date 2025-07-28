import torch.nn as nn
import torch
import torchvision.models.video as video_models
from torchvision.models.video.resnet import BasicBlock, Bottleneck, Conv3DSimple,\
  Conv3DNoTemporal, Conv2Plus1D, VideoResNet, WeightsEnum, _ovewrite_named_param
from typing import Union, Callable, Sequence, Optional, Any
from torchvision.models.video.s3d import s3d,  S3D_Weights
from torchvision.transforms import v2


class S3D_basic(nn.Module):
  def __init__(self, num_classes=100, drop_p=0.5,
               weights=S3D_Weights.KINETICS400_V1):
    super().__init__()
    self.num_classes = num_classes
    self.drop_p = drop_p

    #Load pretrained S3D
    s3d_model = s3d(weights=weights)
    
    #replace e