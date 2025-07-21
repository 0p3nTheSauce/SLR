import torch.nn as nn
import torchvision.models.video as video_models
def get_bas_r3d18(num_classes=300):
  model = video_models.r3d_18(pretrained=True) #weights=R3D_18_Weights.KINETICS400_V1d
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
  print(len(trainable_params), " trainable parameters")
  return model, trainable_params