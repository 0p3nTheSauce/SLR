import torch.nn as nn
import torch
from typing import Optional
#locals
from detectron_mvit import MViT_2D_t

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
    def __init__(self, mvit, num_classes,embed_dim=512, num_heads=8, num_layers=4, max_frames=64, mvit_out_dim=768):
        super().__init__()
        self.frame_encoder = MViT_2D_t(mvit)
        self.proj = nn.Linear(mvit_out_dim, embed_dim)  # if dims don't match
        self.pos_enc = TemporalPositionalEncoding(embed_dim, max_frames)
        self.temporal_encoder = TemporalBERT(embed_dim, num_heads, num_layers)
        self.cls_head = nn.Linear(embed_dim, num_classes)

    def forward(self, frames):
        B, T, C, H, W = frames.shape
        feats = self.frame_encoder(frames.view(B * T, C, H, W))  # (B*T, D)
        feats = feats.view(B, T, -1)                              # (B, T, D)
        feats = self.proj(feats)                                  # (B, T, embed_dim)
        feats = self.pos_enc(feats)
        feats = self.temporal_encoder(feats)                      # (B, T, embed_dim)
        return self.cls_head(feats.mean(dim=1))                   # (B, num_classes)


class MVirTed_t(MVirTed):
    def __init__(self, num_classes: int, drop_p: Optional[float] = None):
        super().__init__(MViT_2D_t(), num_classes)


if __name__ == '__main__':
    frames = torch.rand(1, 16, 3, 112, 112)
    mvirted = MVirTed_t(100)
    out = mvirted(frames)
    print(out.shape)