import torch
import torch.nn as nn
from typing import Optional
import pickle

# locals
from .mvit.detectron2_cp.mvit2d import MViT
from .mvit.configs.mvit2d_conf import MViT_t_3x


MVITV2_2D_T = MViT_t_3x()


class MViT_2D_t(nn.Module):
    def __init__(
        self,
        num_classes: int = 100,
        drop_p: float = 0.5,
        pretrain_path: Optional[str] = None,
        out_features: Optional[list[str]] = ["scale5"],
        base_out_feature: str = "scale2",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.drop_p = drop_p

        if out_features:
            self.out_feats = out_features
        else:
            self.out_feats = MVITV2_2D_T.out_features

        self.out_scale = self.out_feats[-1]

        self.mvitv2_2d = MViT(
            embed_dim=MVITV2_2D_T.embed_dim,
            depth=MVITV2_2D_T.depth,
            num_heads=MVITV2_2D_T.num_heads,
            last_block_indexes=MVITV2_2D_T.last_block_indexes,
            residual_pooling=MVITV2_2D_T.residual_pooling,
            drop_path_rate=MVITV2_2D_T.drop_path_rate,
            norm_layer=MVITV2_2D_T.norm_layer,  # type: ignore
            out_features=self.out_feats,
        )

        if not pretrain_path:
            # MVITV2_2D_T.pretrain_path
            with open(MVITV2_2D_T.pretrain_path, "rb") as f:
                checkpoint = pickle.load(f, encoding="latin1")
            bbup_key = "backbone.bottom_up"
            backbone_bottom_up = {
                k.replace(bbup_key + ".", ""): torch.from_numpy(v)
                for k, v in checkpoint["model"].items()
                if bbup_key in k
            }
            if out_features is not None:
                scale_level = int(out_features[0][-1])
                base_scale_level = int(base_out_feature[-1])
                rem_scale_names = [
                    f"scale{i}_norm.{wb}"
                    for i in range(base_scale_level, scale_level)
                    for wb in ["weight", "bias"]
                ]
                for rem in rem_scale_names:
                    backbone_bottom_up.pop(rem)

            self.mvitv2_2d.load_state_dict(backbone_bottom_up)
            

    def forward(self, x):
        # x: (B, C, H, W) — single frame
        out = self.mvitv2_2d(x)[self.out_scale]  # (B, H', W', C')
        # out = out.mean(dim=(1, 2))    # global average pool → (B, C')
        out = out.mean(dim=(2, 3))
        return out



if __name__ == "__main__":
    frames = torch.rand(1, 16, 3, 224, 224)
    print(frames.shape)
    mvit2d = MViT_2D_t()
    B, T, C, H, W = frames.shape
    flat = frames.view(B * T, C, H, W)
    feats = mvit2d(flat)
    feats = feats.view(B, T, -1) 
    print(feats.shape)