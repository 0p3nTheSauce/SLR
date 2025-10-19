from typing import Literal, Optional
import torch
#locals
from .pytorch_mvit import MViTv2S_basic, MViTv1B_basic
from .pytorch_swin3d import Swin3DBig_basic, Swin3DSmall_basic, Swin3DTiny_basic
from .pytorch_r3d import Resnet2_plus1D_18_basic, Resnet3D_18_basic
from .pytorch_s3d import S3D_basic


def get_model(model_name: 
    Literal["S3D_basic", "Resnet3D_18_basic", "Resnet2_plus1D_18_basic",
            "Swin3DTiny_basic", "Swin3DSmall_basic", "Swin3DBig_basic",
            "MViTv2S_basic", "MViTv1B_basic"],
    num_classes: int, 
    drop_p: Optional[float] = None
    ) -> torch.nn.Module:
    
   model_constructors = {
         "S3D_basic": S3D_basic,
            "Resnet3D_18_basic": Resnet3D_18_basic,
            "Resnet2_plus1D_18_basic": Resnet2_plus1D_18_basic,
            "Swin3DTiny_basic": Swin3DTiny_basic,
            "Swin3DSmall_basic": Swin3DSmall_basic,
            "Swin3DBig_basic": Swin3DBig_basic,
            "MViTv2S_basic": MViTv2S_basic,
            "MViTv1B_basic": MViTv1B_basic,
    }
   
   return model_constructors[model_name](num_classes=num_classes, drop_p=drop_p)

__all__ = [
    'get_model',
    'S3D_basic',
    'Resnet3D_18_basic',
    'Resnet2_plus1D_18_basic',
    'Swin3DTiny_basic',
    'Swin3DSmall_basic',
    'Swin3DBig_basic',
    'MViTv2S_basic',
    'MViTv1B_basic',
]