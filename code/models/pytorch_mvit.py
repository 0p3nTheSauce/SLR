from torchvision.models.video import (
    mvit_v2_s,
    mvit_v1_b,
    MViT_V2_S_Weights,
    MViT_V1_B_Weights,
)
import torch
import torch.nn as nn


def _unsqueeze(
    x: torch.Tensor, target_dim: int, expand_dim: int
) -> tuple[torch.Tensor, int]:
    tensor_dim = x.dim()
    if tensor_dim == target_dim - 1:
        x = x.unsqueeze(expand_dim)
    elif tensor_dim != target_dim:
        raise ValueError(f"Unsupported input dimension {x.shape}")
    return x, tensor_dim


class MViTv2S_basic(nn.Module):
    def __init__(self, num_classes=100, drop_p=0.5, weights_path=None):
        super().__init__()
        self.num_classes = num_classes
        self.drop_p = drop_p

        mvitv2s = mvit_v2_s(MViT_V2_S_Weights.KINETICS400_V1)

        self.backbone = nn.ModuleList(
            [  # stored in module list for parameter groups
                mvitv2s.conv_proj,
                mvitv2s.pos_encoding,
                mvitv2s.blocks,
                mvitv2s.norm,
            ]
        )

        self.conv_proj = self.backbone[0]
        self.pos_encoding = self.backbone[1]
        self.blocks = self.backbone[2]
        self.norm = self.backbone[3]

        original_linear = mvitv2s.head[1]  # The Linear layer
        in_features = original_linear.in_features

        self.classifier = nn.Sequential(
            nn.Dropout(drop_p),
            nn.Linear(in_features, num_classes),
        )

        if weights_path:
            checkpoint = torch.load(weights_path, map_location="cpu")
            self.load_state_dict(checkpoint)
            print(f"Loaded pretrained weights from {weights_path}")
        else:
            self._initialize_classifier()  # mvitspecific initialization

    def _initialize_classifier(self):
        """Initialize only the classifier weights, leaving backbone untouched"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert if necessary (B, C, H, W) -> (B, C, 1, H, W)
        x = _unsqueeze(x, 5, 2)[0]
        # patchify and reshape: (B, C, T, H, W) -> (B, embed_channels[0], T', H', W') -> (B, THW', embed_channels[0])
        x = self.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)

        # add positional encoding
        x = self.pos_encoding(x)

        # pass patches through the encoder
        thw = (self.pos_encoding.temporal_size,) + self.pos_encoding.spatial_size
        for block in self.blocks:  # type: ignore
            x, thw = block(x, thw)
        x = self.norm(x)

        # classifier "token" as used by standard language architectures
        x = x[:, 0]
        x = self.classifier(x)

        return x


class MViTv2S_extended(nn.Module):
    def __init__(self, num_classes=100, drop_p=0.5, weights_path=None):
        super().__init__()
        self.num_classes = num_classes
        self.drop_p = drop_p

        mvitv2s = mvit_v2_s(MViT_V2_S_Weights.KINETICS400_V1)

        self.backbone = nn.ModuleList(
            [  # stored in module list for parameter groups
                mvitv2s.conv_proj,
                mvitv2s.pos_encoding,
                mvitv2s.blocks,
                mvitv2s.norm,
            ]
        )

        self.conv_proj = self.backbone[0]
        self.pos_encoding = self.backbone[1]
        self.blocks = self.backbone[2]
        self.norm = self.backbone[3]

        original_linear = mvitv2s.head[1]  # The Linear layer
        in_features = original_linear.in_features

        self.classifier = nn.Sequential(
            nn.Dropout(drop_p),
            nn.Linear(in_features, num_classes),
        )

        if weights_path:
            checkpoint = torch.load(weights_path, map_location="cpu")
            self.load_state_dict(checkpoint)
            print(f"Loaded pretrained weights from {weights_path}")
        else:
            self._initialize_classifier()  # mvitspecific initialization

        # After building the model, interpolate temporal pos embed for 32 frames
        self._adapt_pos_encoding(num_frames=32)

    def _adapt_pos_encoding(self, num_frames: int):
        """Interpolate temporal positional embedding to match new frame count."""
        pe = self.pos_encoding
        temporal_stride = self.conv_proj.stride[0]  # typically 2
        new_temporal_size = num_frames // temporal_stride  # 32 // 2 = 16

        if pe.temporal_size == new_temporal_size:
            return  # nothing to do

        old_embed = pe.pos_embed.data  # [1, 1 + T*H*W, C]
        cls_token = old_embed[:, :1, :]
        spatial_embed = old_embed[:, 1:, :]

        old_T = pe.temporal_size
        H, W = pe.spatial_size
        C = old_embed.shape[-1]

        # Reshape to [1, C, T, H*W] and interpolate T
        spatial_embed = spatial_embed.reshape(1, old_T, H * W, C)
        spatial_embed = spatial_embed.permute(0, 3, 1, 2)  # [1, C, old_T, H*W]
        spatial_embed = torch.nn.functional.interpolate(
            spatial_embed,
            size=(new_temporal_size, H * W),
            mode="bilinear",
            align_corners=False,
        )
        spatial_embed = spatial_embed.permute(0, 2, 3, 1).reshape(
            1, new_temporal_size * H * W, C
        )

        pe.pos_embed = nn.Parameter(torch.cat([cls_token, spatial_embed], dim=1))
        pe.temporal_size = new_temporal_size

    def _initialize_classifier(self):
        """Initialize only the classifier weights, leaving backbone untouched"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


class MViTv1B_basic(nn.Module):
    def __init__(self, num_classes=100, drop_p=0.5, weights_path=None):
        super().__init__()
        self.num_classes = num_classes
        self.drop_p = drop_p

        mvitv1b = mvit_v1_b(MViT_V1_B_Weights.KINETICS400_V1)

        self.backbone = nn.ModuleList(
            [  # stored in module list for parameter groups
                mvitv1b.conv_proj,
                mvitv1b.pos_encoding,
                mvitv1b.blocks,
                mvitv1b.norm,
            ]
        )

        self.conv_proj = self.backbone[0]
        self.pos_encoding = self.backbone[1]
        self.blocks = self.backbone[2]
        self.norm = self.backbone[3]

        original_linear = mvitv1b.head[1]  # The Linear layer
        in_features = original_linear.in_features

        self.classifier = nn.Sequential(
            nn.Dropout(drop_p),
            nn.Linear(in_features, num_classes),
        )

        if weights_path:
            checkpoint = torch.load(weights_path, map_location="cpu")
            self.load_state_dict(checkpoint)
            print(f"Loaded pretrained weights from {weights_path}")
        else:
            self._initialize_classifier()  # mvitspecific initialization

    def _initialize_classifier(self):
        """Initialize only the classifier weights, leaving backbone untouched"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert if necessary (B, C, H, W) -> (B, C, 1, H, W)
        x = _unsqueeze(x, 5, 2)[0]
        # patchify and reshape: (B, C, T, H, W) -> (B, embed_channels[0], T', H', W') -> (B, THW', embed_channels[0])
        x = self.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)

        # add positional encoding
        x = self.pos_encoding(x)

        # pass patches through the encoder
        thw = (self.pos_encoding.temporal_size,) + self.pos_encoding.spatial_size
        for block in self.blocks:  # type: ignore
            x, thw = block(x, thw)
        x = self.norm(x)

        # classifier "token" as used by standard language architectures
        x = x[:, 0]
        x = self.classifier(x)

        return x
