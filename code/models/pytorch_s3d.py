import torch.nn as nn
import torch
from torchvision.models.video.s3d import s3d,  S3D_Weights

#for reference from pytorch docs:
# self.classifier = nn.Sequential(
#             nn.Dropout(p=dropout),
#             nn.Conv3d(1024, num_classes, kernel_size=1, stride=1, bias=True),
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = self.avgpool(x)
#         x = self.classifier(x)
#         x = torch.mean(x, dim=(2, 3, 4))
#         return x


class S3D_basic(nn.Module):
  def __init__(self, num_classes=100, drop_p=0.5, weights_path=None):
    super().__init__()
    self.num_classes = num_classes
    self.drop_p = drop_p
    
    s3d_model = s3d(weights=S3D_Weights.KINETICS400_V1)
    
    self.backbone = nn.ModuleList([
      s3d_model.features,
      s3d_model.avgpool
    ])
    
    self.features = self.backbone[0]
    self.avgpool = self.backbone[1]
    
    self.classifier = nn.Sequential(
      nn.Dropout(p=drop_p),
      nn.Conv3d(1024, num_classes, kernel_size=1, stride=1, bias=True),
    )
    
    if weights_path:
      checkpoint = torch.load(weights_path, map_location='cpu')
      self.load_state_dict(checkpoint)
      print(f"Loaded pretrained weights from {weights_path}")
      
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.features(x)
    x = self.avgpool(x)
    x = self.classifier(x)
    x = torch.mean(x, dim=(2, 3, 4))
    return x