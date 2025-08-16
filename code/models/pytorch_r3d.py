import torch.nn as nn
import torch
from  torchvision.models.video.resnet import r3d_18, r2plus1d_18, R3D_18_Weights, R2Plus1D_18_Weights

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


class Resnet3D_18_basic(nn.Module): #These did not have drop
  def __init__(self, num_classes=100, weights_path=None):
    super().__init__()
    self.num_classes = num_classes
    
    r3d18 = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
    
    self.backbone = nn.ModuleList([
      r3d18.stem,
      r3d18.layer1,
      r3d18.layer2,
      r3d18.layer3,
      r3d18.layer4,
      r3d18.avgpool
    ])
    
    self.stem = self.backbone[0]
    self.layer1 = self.backbone[1]
    self.layer2 = self.backbone[2]
    self.layer3 = self.backbone[3]
    self.layer4 = self.backbone[4]
    self.avgpool = self.backbone[5]
    in_features = r3d18.fc.in_features
    self.classifier = nn.Sequential(
      # nn.Dropout(drop_p, inplace=True),
      nn.Linear(in_features, num_classes),
    )
    
    if weights_path:
      checkpoint = torch.load(weights_path, map_location='cpu')
      self.load_state_dict(checkpoint)
      print(f"Loaded pretrained weights from {weights_path}")

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    #taken from pytorch video resnet
    x = self.stem(x)
    
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.avgpool(x)
    # Flatten the layer to fc
    x = x.flatten(1)
    x = self.classifier(x)
    return x
  
class Resnet2_plus1D_18_basic(nn.Module):
  def __init__(self, num_classes=100, weights_path=None):
    super().__init__()
    self.num_classes = num_classes
    
    r2plus1d18 = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
    
    self.backbone = nn.ModuleList([
      r2plus1d18.stem,
      r2plus1d18.layer1,
      r2plus1d18.layer2,
      r2plus1d18.layer3,
      r2plus1d18.layer4,
      r2plus1d18.avgpool
    ])
    
    self.stem = self.backbone[0]
    self.layer1 = self.backbone[1]
    self.layer2 = self.backbone[2]
    self.layer3 = self.backbone[3]
    self.layer4 = self.backbone[4]
    self.avgpool = self.backbone[5]
    
    in_features = r2plus1d18.fc.in_features
    self.classifier = nn.Sequential(
      # nn.Dropout(drop_p, inplace=True),
      nn.Linear(in_features, num_classes),
    )
    
    if weights_path:
      checkpoint = torch.load(weights_path, map_location='cpu')
      self.load_state_dict(checkpoint)
      print(f"Loaded pretrained weights from {weights_path}")
    
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    #taken from pytorch video resnet
    x = self.stem(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    # Flatten the layer to fc
    x = x.flatten(1)
    x = self.classifier(x)
    return x
    

    

# Example usage:
if __name__ == "__main__":
  # Create model instance

  pass