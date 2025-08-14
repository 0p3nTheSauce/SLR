import torch.nn as nn
import torch
import torchvision.models.video as video_models
from torchvision.models.video.resnet import BasicBlock, Bottleneck, Conv3DSimple,\
  Conv3DNoTemporal, Conv2Plus1D, VideoResNet, WeightsEnum, _ovewrite_named_param
from typing import Union, Callable, Sequence, Optional, Any
# from  torchvision.models.video.resnet import r3d_18, R3D_18_Weights
from torchvision.models.video import swin3d_t, swin3d_b, Swin3D_T_Weights, Swin3D_B_Weights
from torchvision.transforms import v2
# from .classifiers import AttentionClassifier

class Swin3DBig_basic(nn.Module):
  def __init__(self, num_classes=100, drop_p=0.5, 
               weights_path=None):
    super().__init__()
    self.num_classes = num_classes
    self.drop_p=drop_p
    
    swin3db = swin3d_b(weights=Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1)
    
    self.backbone = nn.ModuleList([
      swin3db.patch_embed,
      swin3db.pos_drop, 
      swin3db.features,
      swin3db.norm,
      swin3db.avgpool
    ])
    
    in_features = swin3db.head.in_features
    self.classifier = nn.Sequential(
        nn.Dropout(p=drop_p),
        nn.Linear(in_features, num_classes),
    )
    
    self.patch_embed = self.backbone[0]
    self.pos_drop = self.backbone[1] 
    self.features = self.backbone[2]
    self.norm = self.backbone[3]
    self.avgpool = self.backbone[4]
    self.head = self.classifier  # Alias
    
    if weights_path:
      checkpoint = torch.load(weights_path, map_location='cpu')
      self.load_state_dict(checkpoint)
      print(f"Loaded pretrained weights from {weights_path}")
      
  def __str__(self):
    """Return string representation of model"""
    return f"""Swin3D Tiny basic implementation
    (num_classes={self.num_classes}, drop_p={self.drop_p})
    Model architecture:
      Backbone: {len(self.backbone)} layers
      Classifier: {self.classifier}"""
    
  def forward(self, x):
    """Forward pass through the model"""
    x = self.patch_embed(x)
    x = self.pos_drop(x)
    x = self.features(x)
    x = self.norm(x)
    x = x.permute(0, 4, 1, 2, 3) # B, C, _T, _H, _W
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.head(x)  
    return x



class Swin3DTiny_basic(nn.Module):
  def __init__(self, num_classes=100, drop_p=0.3, 
               weights_path=None):
    super().__init__()
    self.num_classes = num_classes
    self.drop_p=drop_p
    
    swin3dt = swin3d_t(weights=Swin3D_T_Weights.KINETICS400_V1)
    
    self.backbone = nn.ModuleList([
      swin3dt.patch_embed,
      swin3dt.pos_drop, 
      swin3dt.features,
      swin3dt.norm,
      swin3dt.avgpool
    ])
    
    in_features = swin3dt.head.in_features
    self.classifier = nn.Sequential(
        nn.Dropout(p=drop_p),
        nn.Linear(in_features, num_classes),
    )
    
    self.patch_embed = self.backbone[0]
    self.pos_drop = self.backbone[1] 
    self.features = self.backbone[2]
    self.norm = self.backbone[3]
    self.avgpool = self.backbone[4]
    self.head = self.classifier  # Alias
    
    if weights_path:
      checkpoint = torch.load(weights_path, map_location='cpu')
      self.load_state_dict(checkpoint)
      print(f"Loaded pretrained weights from {weights_path}")
      
  def __str__(self):
    """Return string representation of model"""
    return f"""Swin3D Tiny basic implementation
    (num_classes={self.num_classes}, drop_p={self.drop_p})
    Model architecture:
      Backbone: {len(self.backbone)} layers
      Classifier: {self.classifier}"""
    
  def forward(self, x):
    """Forward pass through the model"""
    x = self.patch_embed(x)
    x = self.pos_drop(x)
    x = self.features(x)
    x = self.norm(x)
    x = x.permute(0, 4, 1, 2, 3) # B, C, _T, _H, _W
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.head(x)  
    return x
  
  @classmethod
  def from_config(cls, config):
    """Create model instance from config object"""
    instance = cls(
      num_classes=config.num_classes,
      drop_p=config.drop_p,
      weights_path=config.weights_path
    )
    if config.frozen:  # Only freeze if layers specified
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

      layer_to_freeze = None
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
        
        if layer_to_freeze is not None:
          for param in layer_to_freeze.parameters():
            param.requires_grad = False
          # Apply normalization layer freezing recursively
          layer_to_freeze.apply(freeze_norm_layers) 
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
        
if __name__ == '__main__':
  # swin3d = swin3d_t(weights=Swin3D_T_Weights.KINETICS400_V1)
  swin3d = Swin3DTiny_basic()
  input = torch.rand(1, 3, 16, 112, 112)
  # for name, module in swin3d.named_modules():
  #   print(f"{name}: {module}")
  # for param in swin3d.backbone.parameters():
  #   print(param)
  # for param in swin3d.classifier.parameters():
  #   print(param)
  output = swin3d(input)
  print(output.shape)
  # features = output.view(output.size(0), -1)
  # print(features)