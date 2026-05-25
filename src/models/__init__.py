from typing import Optional, Tuple
import torch
import torch.nn as nn
from pydantic import BaseModel


# locals

from run_types import NormDict
#pytorch lib models
from .pytorch_mvit import MViTv2S_basic, MViTv2S_extended, MViTv1B_basic
from .pytorch_swin3d import Swin3DBig_basic, Swin3DSmall_basic, Swin3DTiny_basic
from .pytorch_r3d import Resnet2_plus1D_18_basic, Resnet3D_18_basic
from .pytorch_s3d import S3D_basic
# slowfast mvit
from .og_mvit import MVITv2_B_32x3_basic, MVITv2_S_16x4_basic
#custem seperable mvit
from .sep_mvit_bert import MVirTed_t_basic, MVirTed
from .mvirted_mae import SepMViTBERTMAE
from .detectron_mvit import MViT_2D_t


S3D = "S3D"
R3D_18 = "R3D_18"
R2plus1D_18 = "R(2+1)D_18"
Swin3D_T = "Swin3D_T"
Swin3D_S = "Swin3D_S"
Swin3D_B = "Swin3D_B"
MViTv2_S = "MViTv2_S"
MViTv2_S_e = "MViTv2_S_e"
MViTv1_B = "MViTv1_B"
MViTv2_S_16x4 = "MViTv2_S_16x4"
MViTv2_B_32x3 = "MViTv2_B_32x3"
MVirTed_t = "MVirTed_t"
MVirTed_t_MAE = MVirTed_t + '_MAE'


model_names = [
    S3D,
    R3D_18,
    R2plus1D_18,
    Swin3D_T,
    Swin3D_S,
    Swin3D_B,
    MViTv2_S,
    MViTv2_S_e,
    MViTv1_B,
    MViTv2_S_16x4,
    MViTv2_B_32x3,
    MVirTed_t,
    MVirTed_t_MAE
]


def get_model(model_name: str, num_classes: int, drop_p: Optional[float]) -> torch.nn.Module:
    """Get model by name.

    Args:
        model_name (str): One of S3D, R3D_18, R(2+1)D_18, Swin3D_T, Swin3D_S, Swin3D_B, MViTv2_S, MViTv1_B, MViTv2_S_e, MViTv2_S_16x4, MViTv2_B_32x3, MVirTed_t
        num_classes (int): Number of output classes.
        drop_p (float): Dropout in final classification layer.

    Raises:
            ValueError: If model_name is not recognized.

    Returns:
            torch.nn.Module: Model instance
    """
    model_constructors_dp = { #dropout has to be float at the moment
        S3D: S3D_basic,
        R3D_18: Resnet3D_18_basic,
        R2plus1D_18: Resnet2_plus1D_18_basic,
        Swin3D_T: Swin3DTiny_basic,
        Swin3D_S: Swin3DSmall_basic,
        Swin3D_B: Swin3DBig_basic,
        MViTv2_S: MViTv2S_basic,
        MViTv2_S_e: MViTv2S_extended,
        MViTv1_B: MViTv1B_basic,
        
    }

    model_constructors_opdp = { #optional dropout, defaults to original config
        MViTv2_S_16x4: MVITv2_S_16x4_basic,
        MViTv2_B_32x3: MVITv2_B_32x3_basic,
        MVirTed_t: MVirTed_t_basic,
        }

    if model_name not in model_constructors_dp and model_name not in model_constructors_opdp:
        raise ValueError(
            f"Model {model_name} not recognized. Available models: {list(model_constructors_dp.keys()) + list(model_constructors_opdp.keys())}"
        )

    if model_name in model_constructors_dp:
        return model_constructors_dp[model_name](num_classes=num_classes, drop_p=drop_p)
    else:
        return model_constructors_opdp[model_name](num_classes=num_classes, drop_p=drop_p)


# def get_mae_encoder(encoder_name: str, encoder: MVirTed, mask_ratio: float, embed_dim: int) -> torch.nn.Module:
#     """Map a name to an encoder"""
    
#     model_constructors = {
#         MVirTed_t: MVirTed    
#     }
    
#     if model_name not in model_constructors:
#         raise ValueError(
#             f"Model {model_name} not recognized. Available models: {', '.join(model_constructors.keys())}"
#         )

#     return model_constructors[model_name](encoder=encoder, mask_ratio=mask_ratio, embed_dim=embed_dim)


def get_mae_model(
    model_name: str,
    encoder: MVirTed,
    mask_ratio: float,
    embed_dim: int,
    
    
    ) -> torch.nn.Module:
    """Map a model name to Masked Auto Encoder Name"""
    
    model_constructors = {
        MVirTed_t_MAE: SepMViTBERTMAE        
    }
    
    if model_name not in model_constructors:
        raise ValueError(
            f"Model {model_name} not recognized. Available models: {', '.join(model_constructors.keys())}"
        )

    return model_constructors[model_name](encoder=encoder, mask_ratio=mask_ratio, embed_dim=embed_dim)




def avail_models() -> list[str]:
    """Get list of available model names.

    Returns:
            list[str]: List of available model names.
    """
    return model_names


def norm_vals(model_name: str) -> NormDict:
    """Get normalization values (mean and std) for a given model.

    Args:
            model_name (str): One of 'S3D', 'R3D_18', 'R(2+1)D_18', 'Swin3D_T', 'Swin3D_S', 'Swin3D_B', 'MViTv2_S', 'MViTv1_B'

    Raises:
            ValueError: If model_name is not recognized.

    Returns:
            Dict: Dictionary containing:
           - 'mean': Tuple of three floats
           - 'std': Tuple of three floats
    """

    norm_dict = {
        S3D: NormDict(
            mean=(0.43216, 0.394666, 0.37645), std=(0.22803, 0.22145, 0.216989)
        ),
        R3D_18: NormDict(
            mean=(0.43216, 0.394666, 0.37645), std=(0.22803, 0.22145, 0.216989)
        ),
        R2plus1D_18: NormDict(
            mean=(0.43216, 0.394666, 0.37645), std=(0.22803, 0.22145, 0.216989)
        ),
        Swin3D_T: NormDict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        Swin3D_S: NormDict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        Swin3D_B: NormDict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        
        #All mvits use same norm values since pretrained on kinetics400
        MViTv2_S: NormDict(mean=(0.45, 0.45, 0.45), std=(0.225, 0.225, 0.225)),
        MViTv2_S_e: NormDict(mean=(0.45, 0.45, 0.45), std=(0.225, 0.225, 0.225)),
        MViTv1_B: NormDict(mean=(0.45, 0.45, 0.45), std=(0.225, 0.225, 0.225)),
        #new mvit
        MViTv2_S_16x4: NormDict(mean=(0.45, 0.45, 0.45), std=(0.225, 0.225, 0.225)),
        MViTv2_B_32x3: NormDict(mean=(0.45, 0.45, 0.45), std=(0.225, 0.225, 0.225)),
        #seperable mvit
        MVirTed_t: NormDict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        MVirTed_t_MAE: NormDict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    }

    if model_name not in norm_dict:
        raise ValueError(
            f"Model {model_name} not recognized. Available models: {list(norm_dict.keys())}"
        )

    return norm_dict[model_name]

def extend_classifier(model: nn.Module, final_classes: int):
    """Extend the classifier to final_classes outputs, preserving existing weights."""
    old_layer = model.classifier[1]  # index 1 because of the Dropout at [0]

    if isinstance(old_layer, nn.Linear):
        old_out = old_layer.out_features
        in_features = old_layer.in_features
        old_weight = old_layer.weight  # [old_out, in_features]
        new_layer = nn.Linear(in_features, final_classes, bias=True)

    elif isinstance(old_layer, nn.Conv3d):
        old_out = old_layer.out_channels
        in_features = old_layer.in_channels
        old_weight = old_layer.weight  # [old_out, in_features, *kernel_size]
        new_layer = nn.Conv3d(
            in_features,
            final_classes,
            kernel_size=old_layer.kernel_size,  # type: ignore
            stride=old_layer.stride,  # type: ignore
            padding=old_layer.padding,  # type: ignore
            bias=True,
        )

    else:
        raise TypeError(f"Unexpected layer type: {type(old_layer)}")

    if old_layer.bias is None:
        raise ValueError("Expected old layer to have a bias, but bias is None")

    assert final_classes > old_out, (
        f"final_classes ({final_classes}) must be greater than current classes ({old_out})"
    )

    with torch.no_grad():
        new_layer.weight[:old_out] = old_weight
        new_layer.bias[:old_out] = old_layer.bias  # type: ignore
        # Rows beyond old_out are left as default kaiming/random init

    model.classifier[1] = new_layer
    return model

def get_num_parameters(model: nn.Module) -> int:
    """Get total number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())

__all__ = [
    "get_model",
    "extend_classifier",
    "S3D_basic",
    "Resnet3D_18_basic",
    "Resnet2_plus1D_18_basic",
    "Swin3DTiny_basic",
    "Swin3DSmall_basic",
    "Swin3DBig_basic",
    "MViTv2S_basic",
    "MViTv2S_extended",
    "MViTv1B_basic",
    "MVITv2_S_16x4_basic",
    "MVITv2_B_32x3_basic",
    "MVirTed_t_basic",
]
