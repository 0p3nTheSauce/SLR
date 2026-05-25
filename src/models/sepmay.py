
import torch.nn as nn
import torch
from typing import Optional
#locals
from .detectron_mvit import MViT_2D_t
from .sep_mvit_bert import TemporalBERT, TemporalPositionalEncoding




def main():
    frames = torch.rand(1, 16, 3, 224, 224)
    print(frames.shape)
    mvit2d = MViT_2D_t()
    B, T, C, H, W = frames.shape
    flat = frames.view(B * T, C, H, W)
    feats = mvit2d(flat)
    feats = feats.view(B, T, -1) 
    print(feats.shape)


if __name__ == '__main__':
    # main()
    T = 16
    embed_dim = 768
    mask_ratio = 0.5
    B = 1
    embed = torch.rand(B, T, embed_dim)
    
    num_masked = int(T * mask_ratio)
    noise = torch.rand(B,T)
    print(noise)
    noise = noise.argsort(dim=1)
    print(noise)
    masked_ids = noise[:, :num_masked]
    
    
    # print(masked_ids)
    print(masked_ids)
    
    
    targets = embed.detach()
    masked_feats = embed.clone()
    
    x = torch.arange(B).unsqueeze(1)
    masked_feats[x, masked_ids] = 0.0
    print(masked_feats)
    
    
    
    
    