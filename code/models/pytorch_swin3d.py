import torch.nn as nn
import torch
from torchvision.models.video import swin3d_t,swin3d_s, swin3d_b, Swin3D_T_Weights, Swin3D_S_Weights, Swin3D_B_Weights

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
    return f"""Swin3D Big basic implementation
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

class Swin3DSmall_basic(nn.Module):
  def __init__(self, num_classes=100, drop_p=0.5, 
               weights_path=None):
    super().__init__()
    self.num_classes = num_classes
    self.drop_p=drop_p
    
    swin3db = swin3d_b(weights=Swin3D_B_Weights.KINETICS400_V1)
    
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
    return f"""Swin3D Small basic implementation
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
  def __init__(self, num_classes=100, drop_p=0.5, 
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
    
    