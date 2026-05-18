from .pyvision_mvit import mvit_v2_s, MSBlockConfig, _unsqueeze, PositionalEncoding, MultiscaleBlock, Sequence, partial, Pool, MultiscaleAttention, MLP, StochasticDepth
import torch
import torch.nn as nn
from typing import Callable


#Model level dissection 

def make_dummy_video(batch_size=1, num_frames=16, height=224, width=224):
    return torch.randn(batch_size, 3, num_frames, height, width)

class SimMVITv2S(nn.Module):
    def __init__(
        self,
        block_setting: Sequence[MSBlockConfig],
        patch_embed_kernel: tuple[int, int, int] = (3, 7, 7),
        patch_embed_stride: tuple[int, int, int] = (2, 4, 4),
        patch_embed_padding: tuple[int, int, int] = (1, 3, 3),
        temporal_size: int=16,
        spatial_size: tuple[int, int] = (224, 224),
        rel_pos_embed: bool = True,
        residual_pool: bool = True,
        residual_with_cls_embed: bool = False,
        proj_after_attn: bool = True,
        #might be worth adding to configs
        num_classes: int = 400,
        head_dropout: float = 0.5,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.2,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        #this code is taken directly from pytorchvision with some slight tweaks: https://github.com/facebookresearch/mvit/blob/main/mvit/models/mvit_model.py
        
        total_stage_blocks = len(block_setting)
        if total_stage_blocks == 0:
            raise ValueError("The configuration parameter can't be empty.")
        print(f'Total stage blocks: {total_stage_blocks}')
        
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        
        
        #initial projection keeping dimension 5, by increasing channel dimensions and decreaseing T, H and W
        self.conv_proj = nn.Conv3d(
            in_channels=3,
            out_channels=block_setting[0].input_channels,
            kernel_size=patch_embed_kernel,
            stride=patch_embed_stride,
            padding=patch_embed_padding,
        )

        input_size = [size // stride for size, stride in zip((temporal_size,) + spatial_size, self.conv_proj.stride)]

        print(f'Input size to blocks/pos-encoding: {input_size}')

        # Spatio-Temporal Class Positional Encoding
        # Simply adds a class_token in mvitv2 - ie, this pos encoding only applicable to mvitv1b
        self.pos_encoding = PositionalEncoding(
            embed_size=block_setting[0].input_channels,
            spatial_size=(input_size[1], input_size[2]),
            temporal_size=input_size[0],
            rel_pos_embed=rel_pos_embed,
        )
        
        
        # Encoder module
        self.blocks = nn.ModuleList()
        for stage_block_id, cnf in enumerate(block_setting):
            # adjust stochastic depth probability based on the depth of the stage block
            sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)

            self.blocks.append(
                MultiscaleBlock(
                    input_size=input_size,
                    cnf=cnf,
                    residual_pool=residual_pool,
                    residual_with_cls_embed=residual_with_cls_embed,
                    rel_pos_embed=rel_pos_embed,
                    proj_after_attn=proj_after_attn,
                    dropout=attention_dropout,
                    stochastic_depth_prob=sd_prob,
                    norm_layer=norm_layer,
                )
            )

            if len(cnf.stride_q) > 0:
                input_size = [size // stride for size, stride in zip(input_size, cnf.stride_q)]
                print(f'New input size: {input_size}')

        #final norm layer after encoder
        self.norm = norm_layer(block_setting[-1].output_channels)

        self.head = nn.Sequential(
            nn.Dropout(head_dropout, inplace=True),
            nn.Linear(block_setting[-1].output_channels, num_classes)
        )
        
        self._initialise_weights()

    def _initialise_weights(self):
        """If the model is not pretrained, initialise the weights according to MVIT priniciples"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, PositionalEncoding):
                for weights in m.parameters():
                    nn.init.trunc_normal_(weights, std=0.02)



    def forward(self, x: torch.Tensor):
        # Convert if necessary (B, C, H, W) -> (B, C, 1, H, W) (does this even work?)
        #adds cls token at least, in the v2 which was suprising
        x = _unsqueeze(x, 5, 2)[0]
    
        print(f'X after unsqueeze (same): {x.shape}')
        # patchify and reshape: (B, C, T, H, W) -> (B, embed_channels[0], T', H', W') -> (B, THW', embed_channels[0])
        x = self.conv_proj(x)
        print(f'X after conv_proj: {x.shape}')
        x = x.flatten(2).transpose(1, 2)
        print(f'X after reshape: {x.shape}')
        
        #add positional encoding and cls token
        x = self.pos_encoding(x)
        print(f'X after pos_encoding: {x.shape}')
        
        # pass patches through the encoder
        thw = (self.pos_encoding.temporal_size,) + self.pos_encoding.spatial_size
        for block in self.blocks:
            x, thw = block(x, thw)
            print(f'Updated size: {x.shape}')
        x = self.norm(x)
        
        print(f'After encoder: {x.shape}')
        
        # classifier token
        x = x[:, 0]
        x = self.head(x)
        
        return x


def get_config():
    config: dict[str, list] = {
        "num_heads": [1, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8],
        "input_channels": [
            96,
            96,
            192,
            192,
            384,
            384,
            384,
            384,
            384,
            384,
            384,
            384,
            384,
            384,
            384,
            768,
        ],
        "output_channels": [
            96,
            192,
            192,
            384,
            384,
            384,
            384,
            384,
            384,
            384,
            384,
            384,
            384,
            384,
            768,
            768,
        ],
        "kernel_q": [
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
        ],
        "kernel_kv": [
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
        ],
        "stride_q": [
            [1, 1, 1],
            [1, 2, 2],
            [1, 1, 1],
            [1, 2, 2],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 2, 2],
            [1, 1, 1],
        ],
        "stride_kv": [
            [1, 8, 8],
            [1, 4, 4],
            [1, 4, 4],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 1, 1],
            [1, 1, 1],
        ],
    }

    return config


def get_block_setting():

    config = get_config()

    block_setting = []
    for i in range(len(config["num_heads"])):
        block_setting.append(
            MSBlockConfig(
                num_heads=config["num_heads"][i],
                input_channels=config["input_channels"][i],
                output_channels=config["output_channels"][i],
                kernel_q=config["kernel_q"][i],
                kernel_kv=config["kernel_kv"][i],
                stride_q=config["stride_q"][i],
                stride_kv=config["stride_kv"][i],
            )
        )

    return block_setting


#Block level dissection

def _product(s: Sequence[int]) -> int:
    product = 1
    for v in s:
        product *= v
    return product


def make_dummy_block_input():
    return torch.randn(1, 25089, 96), (8, 56, 56)

class SimMultiScaleBlock(nn.Module):
    def __init__(
        self,
        input_size: list[int],
        cnf: MSBlockConfig,        
        proj_after_attn: bool = True,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        rel_pos_embed: bool = True,
        residual_pool: bool = True,
        residual_with_cls_embed: bool = False,
        dropout: float = 0.0,
        stochastic_depth_prob: float = 0.2,
        ):
        super().__init__()
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        
        #for skip connection pooling
        if _product(cnf.stride_q) > 1: #if any stride greater than 1 (assumes no 0s)
            #configure the pooling kernel
            kernel_skip = [s + 1 if s > 1 else s for s in cnf.stride_q]
            padding_skip = [int(k // 2) for k in kernel_skip]
            self.pool_skip = Pool(
                nn.MaxPool3d(kernel_skip, stride=cnf.stride_q, padding=padding_skip), None  # type: ignore[arg-type]
            )
        else:
            self.pool_skip = None
    
        #adjust attention dim based on presence of extra projection
        attn_dim = cnf.output_channels if proj_after_attn else cnf.input_channels
        
        if cnf.input_channels != cnf.output_channels and proj_after_attn:
            self.project = nn.Linear(cnf.input_channels, cnf.output_channels)
        else:
            self.project = None
            print('Self project is none')
            print(f'In channels: {cnf.input_channels}')
            print(f'Out channels: {cnf.output_channels}')
        
        
        self.norm1 = norm_layer(cnf.input_channels)
        self.norm2 = norm_layer(attn_dim)
        self.needs_transposal = isinstance(self.norm1, nn.BatchNorm1d)
        
        self.attn = MultiscaleAttention(
            input_size,
            cnf.input_channels,
            attn_dim,
            cnf.num_heads,
            kernel_q=cnf.kernel_q,
            kernel_kv=cnf.kernel_kv,
            stride_q=cnf.stride_q,
            stride_kv=cnf.stride_kv,
            rel_pos_embed=rel_pos_embed,
            residual_pool=residual_pool,
            residual_with_cls_embed=residual_with_cls_embed,
            dropout=dropout,
            norm_layer=norm_layer
        )
        
        self.mlp = MLP(
            attn_dim, 
            [4 * attn_dim, cnf.output_channels],
            activation_layer=nn.GELU,
            dropout=dropout,
            inplace=None
        )
        
        
        
        
    def do_norm(self, norm_layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.needs_transposal:
            # For BatchNorm1d, we need to transpose to (B, C, N) and back
            x = norm_layer(x.transpose(1, 2)).transpose(1, 2)
        else:
            x = norm_layer(x)
        return x

    
    
    def forward(self, x: torch.Tensor, thw: tuple[int, int, int]) -> tuple[torch.Tensor, tuple[int, int, int]]:
        
        x_norm1 = self.do_norm(self.norm1, x)   
        print(f'X normed: {x_norm1.shape}')

        x_attn, thw_new = self.attn(x_norm1, thw)
        print(f'X attention: {x_attn.shape}')
        print(f'THW new: {thw_new}')
        
        #maybe project
        if self.project is not None:
            x = self.project(x_norm1)
            print(f'X after projection1 {x.shape}')
        
        #maybe pool skip connection
        if self.pool_skip is not None:
            x = self.pool_skip(x, thw)[0] + self.stochastic_depth(x_attn)
        else:
            x += self.stochastic_depth(x_attn)
        
        x_norm2 = self.do_norm(self.norm2, x)
        
        #maybe project
        if self.project is not None:
            x = self.project(x_norm2)
            print(f'X after projection2 {x.shape}')
        
        
        return x + self.stochastic_depth(self.mlp(x_norm2)), thw_new
        

def test_simvit():
    model = model = SimMVITv2S(
        get_block_setting(),
    )
    dummy_video = make_dummy_video()
    # dummy_video = dummy_video.squeeze(dim=0)
    print(f'Original shape: {dummy_video.shape}')
    print(f'Original dimensions: {dummy_video.dim()}')
    

    # output = model(dummy_video)
    
    output = model(dummy_video)
    
    print(f"Output shape: {output.shape}")

def test_simblock():
    
    dummy_input, thw = make_dummy_block_input()
    
    block = SimMultiScaleBlock(list(dummy_input.shape), get_block_setting()[1])
    
    print(f'Original shape: {dummy_input.shape}')
    output, newthw = block(dummy_input, thw)
    print(f'Output shape: {output.shape}')
    print(f'New THW: {newthw}')

if __name__ == "__main__":
    # test_simvit()
    test_simblock()
