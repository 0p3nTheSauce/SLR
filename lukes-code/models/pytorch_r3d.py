import torch.nn as nn
import torch
import torchvision.models.video as video_models
from torchvision.models.video.resnet import BasicBlock, Bottleneck, Conv3DSimple,\
  Conv3DNoTemporal, Conv2Plus1D, VideoResNet, WeightsEnum, _ovewrite_named_param
from typing import Union, Callable, Sequence, Optional, Any
from  torchvision.models.video.resnet import r3d_18, R3D_18_Weights
from torchvision.transforms import v2

#for reference, the r3d18 wrapper class from pytorch essentially does the following:
    # return _video_resnet(
    #     BasicBlock,
    #     [Conv3DSimple] * 4,
    #     [2, 2, 2, 2],
    #     BasicStem,
    #     weights,
    #     progress,
    #     **kwargs,
    # )
# and the different layers are as follows:
        # self.stem = stem()

        # self.layer1 = self._make_layer(block, conv_makers[0], 64, layers[0], stride=1)
        # self.layer2 = self._make_layer(block, conv_makers[1], 128, layers[1], stride=2)
        # self.layer3 = self._make_layer(block, conv_makers[2], 256, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, conv_makers[3], 512, layers[3], stride=2)

        # self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

def get_bas_r3d18_partlyfrozen(num_classes=300):
  model = video_models.r3d_18(pretrained=True) #weights=R3D_18_Weights.KINETICS400_V1d
  model.fc = nn.Linear(model.fc.in_features, num_classes)
  for param in model.parameters():
    param.requires_grad = False
    
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
  print(len(trainable_params), " trainable parameters")
  return model, trainable_params

def get_video_resnet(
    block: type[Union[BasicBlock, Bottleneck]],
    conv_makers: Sequence[type[Union[Conv3DSimple, Conv3DNoTemporal, Conv2Plus1D]]],
    layers: list[int],
    stem: Callable[..., nn.Module],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> VideoResNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = VideoResNet(block, conv_makers, layers, stem, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model

class Resnet3D18_basic(nn.Module):
  def __init__(self, num_classes=100, drop_p=0.5,
               weights=R3D_18_Weights.DEFAULT):
    super().__init__()
    self.num_classes = num_classes
    self.drop_p = drop_p
    
    # Load pretrained R3D-18
    r3d18 = r3d_18(weights=weights)
    
    # Replace the final fully connected layer
    in_features = r3d18.fc.in_features
    self.backbone = nn.Sequential(*list(r3d18.children())[:-1])  # All except fc
    self.classifier = nn.Sequential(
      nn.Dropout(p=drop_p),
      nn.Linear(in_features, num_classes)
    )
    
  def __str__(self):
    """Return string representation of the model"""
    return f"Resnet3D18_basic(num_classes={self.num_classes},\n\
      drop_p={self.drop_p})\n\
        Model architecture:\n\
          Backbone: {self.backbone}\n\
          Classifier: {self.classifier}"

  def forward(self, x):
    """Forward pass through the model"""
    # Extract features from backbone
    features = self.backbone(x)
    # Flatten if needed (R3D usually outputs [batch, features, 1, 1, 1])
    features = features.view(features.size(0), -1)
    # Classify
    return self.classifier(features)

  @classmethod
  def from_config(cls, config):
    """Create model instance from config object"""
    instance = cls(
      num_classes=config.num_classes,
      drop_p=config.drop_p
      # weights=config.weights #TODO add if needed
    )
    if config.frozen:  # Only freeze if layers specified
      instance.freeze_layers(config.frozen)
    return instance
  
  # def freeze_layers(self, frozen_layers):
  #   """Freeze specified layers of the model"""
  #   if not frozen_layers:
  #     print('Warning: no frozen layers')
  #     return
        
  #   for layer_name in frozen_layers:
  #     if hasattr(self.backbone, layer_name):
  #         layer = getattr(self.backbone, layer_name)
  #         for param in layer.parameters():
  #           param.requires_grad = False
  #         print(f"Frozen layer: {layer_name}")
  #     else:
  #         available_layers = [name for name, _ in self.backbone.named_children()]
  #         print(f"Warning: Layer '{layer_name}' not found. Available layers: {available_layers}")

  def freeze_layers(self, frozen_layers):
    """Freeze specified layers of the model"""
    if not frozen_layers:
        print('Warning: no frozen layers')
        return
    
    # Mapping from intuitive names to actual layer indices
    layer_mapping = {
        'stem': '0',
        'layer1': '1', 
        'layer2': '2',
        'layer3': '3',
        'layer4': '4',
        'avgpool': '5'
    }
    
    for layer_name in frozen_layers:
        # Convert intuitive name to actual layer name if needed
        actual_layer_name = layer_mapping.get(layer_name, layer_name)
        
        if hasattr(self.backbone, actual_layer_name): #type: ignore
            layer = getattr(self.backbone, actual_layer_name) #type: ignore
            for param in layer.parameters():
                param.requires_grad = False
            print(f"Frozen layer: {layer_name} -> {actual_layer_name}")
        else:
            available_layers = [name for name, _ in self.backbone.named_children()]
            print(f"Warning: Layer '{layer_name}' not found. Available layers: {available_layers}")
  
  @staticmethod
  def basic_transforms(size=224):
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

# Example usage:
if __name__ == "__main__":
  # Create model instance
  model = Resnet3D18_basic(num_classes=10, drop_p=0.5)
    
  # Print model info
  print(model)
    
  # Example inference (assuming input shape: [batch, channels, depth, height, width])
  dummy_input = torch.randn(1, 3, 16, 112, 112)  # Example video input
    
  # Forward pass
  output = model(dummy_input)
    
  print(f"Output shape: {output.shape}")
    
  # # Move to GPU if available
  # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  # model.to(device)
    
  # Set to training/eval mode
  model.train()  # or model.eval()