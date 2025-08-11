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
               weights_path=None):
    #TODO: look into using norm layers
    super().__init__()
    self.num_classes = num_classes
    self.drop_p = drop_p
    
    #Load pretrained S3D
    s3d_model = s3d(weights=S3D_Weights.KINETICS400_V1)
    
    self.backbone = nn.ModuleList([
      s3d_model.features,
      s3d_model.avgpool
    ])
    
    #replace head
    self.classifier = nn.Sequential(
      nn.Dropout(p=drop_p),
      nn.Conv3d(1024, num_classes, kernel_size=1, stride=1, bias=True)
    )
    
    self.features = self.backbone[0]
    self.avgpool = self.backbone[1]
    
    if weights_path:
      checkpoint = torch.load(weights_path, map_location='cpu')
      self.load_state_dict(checkpoint)
      print(f"Loaded pretrained weights from {weights_path}")
      
  def __str__(self):
    """Return string rep of model"""
    return f"""S3D basic implementation
  (num_classes={self.num_classes}, drop_p={self.drop_p})
  Model architecture:
    Backbone: {len(self.backbone)} layers
    Classifier: {self.classifier}"""
    
  def forward(self,x):
    """Forward pass"""
    x = self.features(x)
    x = self.avgpool(x)
    x = self.classifier(x)
    x = torch.mean(x, dim=(2,3,4))
    return x
  
  @classmethod
  def from_config(cls, config):
    """Create model instance from config objct"""
    instance = cls(
      num_classes=config.num_classes,
      drop_p=config.drop_p,
      weights_path=config.weights_path
    )
    if config.frozen:
      instance.freeze_layers(config.frozen)
    return instance
  
  def freeze_layers(self, frozen_layers):
    """Freeze specified layers of the model"""
    if not frozen_layers:
      print('Warning: no frozen layers')
      return
      
    for layer_name in frozen_layers:       
      if '.' in layer_name:
        parts = layer_name.split('.')
        top_layer_name = parts[0]
        nested_path = '.'.join(parts[1:])
      else:
        top_layer_name = layer_name
        nested_path = None

      if hasattr(self, top_layer_name):
        if nested_path:
          try:
            current_layer = self
            for part in layer_name.split('.'):
              current_layer = getattr(current_layer, part) #move down through modules
            layer_to_freeze = current_layer
            print(f"Frozen nested layer: {layer_name}")
          except AttributeError:
            print(f"Warning: Nested layer {layer_name} not found")
        else:
          layer_to_freeze = getattr(self, top_layer_name)
          print(f"Frozen layer: {layer_name}")
          
        for param in layer_to_freeze.parameters(): #type: ignore
          param.requires_grad = False

        # Handle BatchNorm and LayerNorm layers specifically
        def freeze_norm_layers(module):
          # Handle BatchNorm layers (if any)
          if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()
            for param in module.parameters():
              param.requires_grad = False
            if hasattr(module, 'track_running_stats'):
              module.track_running_stats = False
          
          # Handle LayerNorm layers (common in transformers)
          elif isinstance(module, nn.LayerNorm):
            module.eval()  # Set to eval mode
            for param in module.parameters():
              param.requires_grad = False
          
          # Handle GroupNorm if present
          elif isinstance(module, nn.GroupNorm):
            module.eval()
            for param in module.parameters():
              param.requires_grad = False
        
        # Apply normalization layer freezing recursively
        layer_to_freeze.apply(freeze_norm_layers)  #type: ignore
      else:
        available_layers = [name for name, _ in self.named_children()]
        print(f"Warning: Layer '{layer_name}' not found. Available layers: {available_layers}")
  
  @staticmethod
  def basic_transforms(size=224):
    '''used by kinetics'''
    base_mean = [0.43216, 0.394666, 0.37645]
    base_std = [0.22803, 0.22145, 0.216989]
    r3d18_final = v2.Compose([
      v2.Lambda(lambda x: x.float() / 255.0),
      v2.Normalize(mean=base_mean, std=base_std),
      v2.Lambda(lambda x: x.permute(1,0,2,3)) 
    ])
    train_transforms = v2.Compose([v2.RandomCrop(size),
                                  v2.RandomHorizontalFlip(),
                                  r3d18_final])
    test_transforms = v2.Compose([v2.CenterCrop(size),
                                  r3d18_final])
    return train_transforms, test_transforms
  