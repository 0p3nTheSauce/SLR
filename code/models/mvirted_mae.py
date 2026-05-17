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

    def forward(self, visible_feats, visible_ids, masked_ids, B, T):
        embed_dim = visible_feats.size(-1)
        # reconstruct full sequence with mask tokens at masked positions
        full_feats = torch.zeros(B, T, embed_dim, device=visible_feats.device)
        full_feats[torch.arange(B).unsqueeze(1), visible_ids] = visible_feats
        mask_tokens = self.mask_token.expand(B, masked_ids.size(1), -1)
        full_feats[torch.arange(B).unsqueeze(1), masked_ids] = mask_tokens
        decoded = self.decoder(full_feats)
        return self.proj(decoded)

class SepMViTBERTMAE(nn.Module):
    def __init__(self, encoder: MVirTed, mask_ratio=0.5, embed_dim=512):
        super().__init__()
        self.backbone = encoder.backbone
        self.classifier = nn.ModuleDict({
            "temporal_encoder": encoder.classifier['temporal_encoder'],
            "decoder": SepMViTMAEDecoder(embed_dim),
        })
        self.mask_ratio = mask_ratio

    def forward(self, frames):
        B, C, T, H, W = frames.shape

        # 1. Extract per-frame features (backbone is frame-independent)
        feats = self.backbone(
            frames.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        ).view(B, T, -1)  # (B, T, 768)

        # 2. Mask before temporal encoder
        num_masked = int(T * self.mask_ratio)
        noise = torch.rand(B, T, device=frames.device).argsort(dim=1)
        masked_ids = noise[:, :num_masked]
        visible_ids = noise[:, num_masked:]

        # targets are backbone features at masked positions
        targets = feats.detach()

        # 3. Pass only visible through temporal encoder
        visible_feats = feats[torch.arange(B).unsqueeze(1), visible_ids]  # (B, T_vis, 768)
        visible_feats = self.classifier['temporal_encoder'](visible_feats)  # (B, T_vis, embed_dim)

        # 4. Decode and compute loss on masked positions only
        preds = self.classifier['decoder'](visible_feats, visible_ids, masked_ids, B, T)
        loss = F.mse_loss(
            preds[torch.arange(B).unsqueeze(1), masked_ids],
            targets[torch.arange(B).unsqueeze(1), masked_ids],
        )
        return loss