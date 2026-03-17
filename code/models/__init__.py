from typing import  Tuple, TypedDict
import torch
import torch.nn as nn
# locals
from .pytorch_mvit import MViTv2S_basic, MViTv1B_basic
from .pytorch_swin3d import Swin3DBig_basic, Swin3DSmall_basic, Swin3DTiny_basic
from .pytorch_r3d import Resnet2_plus1D_18_basic, Resnet3D_18_basic
from .pytorch_s3d import S3D_basic


def get_model(model_name: str, num_classes: int, drop_p: float) -> torch.nn.Module:
    """Get model by name.

    Args:
            model_name (str): One of 'S3D', 'R3D_18', 'R(2+1)D_18', 'Swin3D_T', 'Swin3D_S', 'Swin3D_B', 'MViT_v2S', 'MViT_v1B'
            num_classes (int): Number of output classes.
            drop_p (float): Dropout in final classification layer.

    Raises:
            ValueError: If model_name is not recognized.

    Returns:
            torch.nn.Module: Model instance
    """
    model_constructors = {
        "S3D": S3D_basic,
        "R3D_18": Resnet3D_18_basic,
        "R(2+1)D_18": Resnet2_plus1D_18_basic,
        "Swin3D_T": Swin3DTiny_basic,
        "Swin3D_S": Swin3DSmall_basic,
        "Swin3D_B": Swin3DBig_basic,
        "MViTv2_S": MViTv2S_basic,
        "MViTv1_B": MViTv1B_basic,
    }

    if model_name not in model_constructors:
        raise ValueError(
            f"Model {model_name} not recognized. Available models: {list(model_constructors.keys())}"
        )

    return model_constructors[model_name](num_classes=num_classes, drop_p=drop_p)


def avail_models() -> list[str]:
    """Get list of available model names.

    Returns:
            list[str]: List of available model names.
    """
    return [
        "S3D",
        "R3D_18",
        "R(2+1)D_18",
        "Swin3D_T",
        "Swin3D_S",
        "Swin3D_B",
        "MViTv2_S",
        "MViTv1_B",
    ]


class NormDict(TypedDict):
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]


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
        "S3D": NormDict(
            mean=(0.43216, 0.394666, 0.37645), std=(0.22803, 0.22145, 0.216989)
        ),
        "R3D_18": NormDict(
            mean=(0.43216, 0.394666, 0.37645), std=(0.22803, 0.22145, 0.216989)
        ),
        "R(2+1)D_18": NormDict(
            mean=(0.43216, 0.394666, 0.37645), std=(0.22803, 0.22145, 0.216989)
        ),
        "Swin3D_T": NormDict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        "Swin3D_S": NormDict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        "Swin3D_B": NormDict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        "MViTv2_S": NormDict(mean=(0.45, 0.45, 0.45), std=(0.225, 0.225, 0.225)),
        "MViTv1_B": NormDict(mean=(0.45, 0.45, 0.45), std=(0.225, 0.225, 0.225)),
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
            in_features, final_classes,
            kernel_size=old_layer.kernel_size, #type: ignore
            stride=old_layer.stride, #type: ignore
            padding=old_layer.padding, #type: ignore
            bias=True,
        )

    else:
        raise TypeError(f"Unexpected layer type: {type(old_layer)}")

    if old_layer.bias is None:
        raise ValueError("Expected old layer to have a bias, but bias is None")

    assert final_classes > old_out, \
        f"final_classes ({final_classes}) must be greater than current classes ({old_out})"

    with torch.no_grad():
        new_layer.weight[:old_out] = old_weight
        new_layer.bias[:old_out]   = old_layer.bias #type: ignore
        # Rows beyond old_out are left as default kaiming/random init

    model.classifier[1] = new_layer
    return model


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
    "MViTv1B_basic",
]
