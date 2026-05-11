from .sep_mvit_bert import MVirTed, TemporalBERT
import torch
import torch.nn as nn
import torch.nn.functional as F

class SepMViTMAEDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads=4, num_layers=2):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.decoder = TemporalBERT(embed_dim, num_heads, num_layers)
        self.proj = nn.Linear(embed_dim, embed_dim)  # reconstruct in feature space

    def forward(self, feats, masked_ids, B, T):
        # replace zeros at masked positions with learned mask token
        mask_tokens = self.mask_token.expand(B, masked_ids.size(1), -1)
        feats = feats.clone()
        feats[torch.arange(B).unsqueeze(1), masked_ids] = mask_tokens
        decoded = self.decoder(feats)
        return self.proj(decoded)

class SepMViTBERTMAE(nn.Module):
    def __init__(self, encoder: MVirTed, mask_ratio=0.5, embed_dim=512):
        super().__init__()
        self.encoder = encoder
        self.decoder = SepMViTMAEDecoder(embed_dim)
        self.mask_ratio = mask_ratio

    def forward(self, frames):
        B, C, T, H, W = frames.shape

        # 1. Extract all frame features
        feats = self.encoder.backbone(
            frames.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        ).view(B, T, -1)                           # (B, T, 768)
        feats = self.encoder.temporal_encoder(feats)  # (B, T, embed_dim)

        # 2. Mask
        num_masked = int(T * self.mask_ratio)
        noise = torch.rand(B, T, device=frames.device).argsort(dim=1)
        masked_ids = noise[:, :num_masked]
        # visible_ids = noise[:, num_masked:]

        # targets before masking
        targets = feats.detach()

        # zero out masked positions
        masked_feats = feats.clone()
        masked_feats[torch.arange(B).unsqueeze(1), masked_ids] = 0.0

        # 3. Decode and compute loss only on masked positions
        preds = self.decoder(masked_feats, masked_ids, B, T)
        loss = F.mse_loss(
            preds[torch.arange(B).unsqueeze(1), masked_ids],
            targets[torch.arange(B).unsqueeze(1), masked_ids],
        )
        return loss