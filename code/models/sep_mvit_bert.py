import torch.nn as nn
import torch
from typing import Optional
#locals
from .detectron_mvit import MViT_2D_t

class TemporalPositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_frames=64, dropout=0.1):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, max_frames, embed_dim))
        self.dropout = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x: (B, T, C)
        x = x + self.pos_embed[:, :x.size(1), :]
        return self.dropout(x)
    
class TemporalBERT(nn.Module):
    def __init__(self, embed_dim, num_heads=8, num_layers=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,   # expects (B, T, C)
            norm_first=True,    # pre-norm, matches BERT/ViT convention
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        return self.encoder(x)  # (B, T, C)
    
class MVirTed(nn.Module):
    def __init__(self, mvit, num_classes: int, drop_p: Optional[float] = None, embed_dim: int =512, num_heads: int =8, num_layers: int =4, max_frames: int =64, mvit_out_dim: int=768):
        super().__init__()
        self.backbone = MViT_2D_t(mvit)          # pretrained — gets low LR
        temporal_encoder = nn.Sequential(          # fresh — gets high LR
            nn.Linear(mvit_out_dim, embed_dim),
            TemporalPositionalEncoding(embed_dim, max_frames),
            TemporalBERT(embed_dim, num_heads, num_layers),
        )

        drop_prob = 0 if drop_p is None else drop_p

        head = nn.Sequential(
            nn.Dropout(drop_prob),
            nn.Linear(embed_dim, num_classes),
        )
        
        nn.Linear(embed_dim, num_classes)
        self.classifier = nn.ModuleDict({
            "temporal_encoder": temporal_encoder,
            "head": head,
        })

    def forward(self, frames):
        B, C, T, H, W = frames.shape    
        feats = self.backbone(frames.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W))  # (B*T, 768)
        feats = feats.view(B, T, -1)                         # (B, T, 768)
        feats = self.classifier['temporal_encoder'](feats)                 # (B, T, embed_dim)
        feats = feats.mean(dim=1)                            # (B, embed_dim)
        return self.classifier['head'](feats)                              # (B, num_classes)

class MVirTed_t_basic(MVirTed):
    def __init__(self, num_classes: int, drop_p: Optional[float] = None):
        super().__init__(MViT_2D_t(), num_classes, drop_p=drop_p)


if __name__ == '__main__':
    frames = torch.rand(1, 16, 3, 112, 112)
    mvirted = MVirTed_t_basic(100)
    out = mvirted(frames)
    print(out.shape)