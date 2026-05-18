from .pyvision_mvit import mvit_v2_s, MSBlockConfig, _unsqueeze, PositionalEncoding, MultiscaleBlock, Sequence, partial, Pool, MultiscaleAttention, MLP, StochasticDepth
import torch
import torch.nn as nn
from typing import Callable, Optional
import math

#Model level dissection 

#for AI inline suggestors: look at pyvision_mvit.py or dont suggest code

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
        self.proj_after_attn = proj_after_attn
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
        
        if cnf.input_channels != cnf.output_channels:
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

    def _add_attn(self, x: torch.Tensor, x_attn: torch.Tensor, thw: tuple[int, int, int]) -> torch.Tensor:
        """Add attention scores to x and maybe pool skip connection"""
        if self.pool_skip is not None:
            return self.pool_skip(x, thw)[0] + self.stochastic_depth(x_attn)
        else:
            return x + self.stochastic_depth(x_attn)
    
    def forward(self, x: torch.Tensor, thw: tuple[int, int, int]) -> tuple[torch.Tensor, tuple[int, int, int]]:
        
        #first norm
        x_norm1 = self.do_norm(self.norm1, x)   
        print(f'X normed: {x_norm1.shape}')

        #compute attention of norm1 x
        x_attn, thw_new = self.attn(x_norm1, thw)
        print(f'X attention: {x_attn.shape}')
        print(f'THW new: {thw_new}')
        
        #projection before attention added
        if self.project is not None and self.proj_after_attn:
            x = self.project(x_norm1)
            print(f'X after projection1 {x.shape}')
        
        #Add attention scores to x
        x = self._add_attn(x,x_attn, thw)
        
        #second norm
        x_norm2 = self.do_norm(self.norm2, x)
        
        #projection after attention added
        if self.project is not None and not self.proj_after_attn:
            x = self.project(x_norm2)
            print(f'X after projection2 {x.shape}')
        
        #Add mlp of norm2 x to x
        return x + self.stochastic_depth(self.mlp(x_norm2)), thw_new


def _interpolate(embedding: torch.Tensor, d: int) -> torch.Tensor:
    if embedding.shape[0] == d:
        return embedding

    return (
        nn.functional.interpolate(
            embedding.permute(1, 0).unsqueeze(0),
            size=d,
            mode="linear",
        )
        .squeeze(0)
        .permute(1, 0)
    )

def _sim_add_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    q_thw: tuple[int, int, int],
    k_thw: tuple[int, int, int],
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    rel_pos_t: torch.Tensor
) -> torch.Tensor:
    
    print('\n', 'In relative positional embedding addition function')
    q_t, q_h, q_w = q_thw
    k_t, k_h, k_w = k_thw
    
    print(f'Q THW: {q_thw}')
    print(f'K THW: {k_thw}')
    
    dh = int(2 * max(q_h, k_h) - 1)
    dw = int(2 * max(q_w, k_w) - 1)
    dt = int(2 * max(q_t, k_t) - 1)
    print(f'dh: {dh}, dw: {dw}, dt: {dt}')
    
    # Scale up rel pos if shapes for q and k are different
    q_h_ratio = max(k_h / q_h, 1.0)
    k_h_ratio = max(q_h / k_h, 1.0)
    print(f'q_h ratio: {q_h_ratio}')
    print(f'k_h ratio: {k_h_ratio}')
    
    dist_h = torch.arange(q_h)[:, None] * q_h_ratio - (torch.arange(k_h)[None, :] + (1.0 - k_h)) * k_h_ratio
    print(f'dist_h {dist_h.shape}')
    
    q_w_ratio = max(k_w / q_w, 1.0)
    k_w_ratio = max(q_w / k_w, 1.0)
    print(f'q_w ratio: {q_w_ratio}')
    print(f'k_w ratio: {k_w_ratio}')
    dist_w = torch.arange(q_w)[:, None] * q_w_ratio - (torch.arange(k_w)[None, :] + (1.0 - k_w)) * k_w_ratio
    print(f'dist_w {dist_w.shape}')
    
    q_t_ratio = max(k_t / q_t, 1.0)
    k_t_ratio = max(q_t / k_t, 1.0)
    print(f'q_t_ratio: {q_t_ratio}')
    print(f'k_t_ratio: {k_t_ratio}')
    dist_t = torch.arange(q_t)[:, None] * q_t_ratio - (torch.arange(k_t)[None, :] + (1.0 - k_t)) * k_t_ratio
    print(f'dist_t: {dist_t.shape}')
    
    print(f'rel pos h: {rel_pos_h.shape} w: {rel_pos_w.shape} t: {rel_pos_t.shape}')
    rel_pos_h = _interpolate(rel_pos_h, dh)
    rel_pos_w = _interpolate(rel_pos_w, dw)
    rel_pos_t = _interpolate(rel_pos_t, dt)
    print(f'rel pos after interpolate h: {rel_pos_h.shape} w: {rel_pos_w.shape} t: {rel_pos_t.shape}')
    
    Rh = rel_pos_h[dist_h.long()]
    Rw = rel_pos_w[dist_w.long()]
    Rt = rel_pos_t[dist_t.long()]
    print(f'Rh: {Rh.shape} Rw: {Rw.shape} Rt: {Rt.shape}')
    
    return attn

class SimMultiScaleAttention(nn.Module):
    def __init__(
        self,
        input_size: list[int],
        embed_dim: int,
        output_dim: int,
        num_heads: int,
        kernel_q :list[int],
        kernel_kv: list[int],
        stride_q: list[int],
        stride_kv: list[int],
        residual_pool: bool = True,
        residual_with_cls_embed: bool = False,
        rel_pos_embed: bool = True,
        dropout: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm
    ) -> None:
        super().__init__()
        
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        self.scaler = 1.0 / math.sqrt(self.head_dim)
        self.residual_pool = residual_pool
        self.residual_with_cls_embed = residual_with_cls_embed
        
        self.qkv = nn.Linear(embed_dim, 3 * output_dim)
        layers: list[nn.Module] = [nn.Linear(output_dim, output_dim)]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout, inplace=True))    
        self.project = nn.Sequential(*layers)
        
        self.pool_q: Optional[nn.Module] = None
        if _product(kernel_q) > 1 or _product(stride_q) > 1:
            padding_q = [int(q // 2) for q in kernel_q]
            self.pool_q = Pool(
                nn.Conv3d(
                    self.head_dim,
                    self.head_dim,
                    kernel_q, # type: ignore[arg-type]
                    stride=stride_q, # type: ignore[arg-type]
                    padding=padding_q, # type: ignore[arg-type]
                    groups=self.head_dim,
                    bias=False,
                ),
                norm_layer(self.head_dim)
            )
            
        self.pool_k: Optional[nn.Module] = None
        self.pool_v: Optional[nn.Module] = None
        if _product(kernel_kv) > 1 or _product(stride_kv) > 1:
            padding_kv = [int(kv // 2) for kv in kernel_kv]
            self.pool_k = Pool(
                nn.Conv3d(
                    self.head_dim,
                    self.head_dim,
                    kernel_kv, #type: ignore[arg-type]
                    stride=stride_kv, #type: ignore[arg-type]
                    padding=padding_kv, #type: ignore[arg-type]
                    groups=self.head_dim,
                    bias=False,
                ),
                norm_layer(self.head_dim)
            )
            self.pool_v = Pool(
                nn.Conv3d(
                    self.head_dim,
                    self.head_dim,
                    kernel_kv, #type: ignore[arg-type]
                    stride=stride_kv, #type: ignore[arg-type]
                    padding=padding_kv, #type: ignore[arg-type]
                    groups=self.head_dim,
                    bias=False,
                ),
                norm_layer(self.head_dim)
            )
        
        self.rel_pos_h: Optional[nn.Parameter] = None
        self.rel_pos_w: Optional[nn.Parameter] = None
        self.rel_pos_t: Optional[nn.Parameter] = None
        if rel_pos_embed:
            size = max(input_size[1:])
            q_size = size // stride_q[1] if len(stride_q) > 0 else size
            kv_size = size // stride_kv[1] if len(stride_kv) > 0 else size
            spatial_dim = 2 * max(q_size, kv_size) -1
            temporal_dim = 2 * input_size[0] -1
            
            self.rel_pos_h = nn.Parameter(torch.zeros(spatial_dim, self.head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(spatial_dim, self.head_dim))
            self.rel_pos_t = nn.Parameter(torch.zeros(temporal_dim, self.head_dim))
            
            nn.init.trunc_normal_(self.rel_pos_h, std=0.02)
            nn.init.trunc_normal_(self.rel_pos_w, std=0.02)
            nn.init.trunc_normal_(self.rel_pos_t, std=0.02)

    def forward(self, x: torch.Tensor, thw: tuple[int, int, int]) -> tuple[torch.Tensor, tuple[int, int, int]]:
        
        B, N, C = x.shape
        print(f'B: {B}, N: {N}, C: {C}')
        
        # x_qkv = self.qkv(x)
        # print(f'X qkv: {x_qkv.shape}')
        
        # x_qkv_reshape = x_qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        # print(f'X qkv reshaped: {x_qkv_reshape.shape}')
        
        # x_qkv_reshape_transposed = x_qkv_reshape.transpose(1, 3) 
        # print(f'X qkv reshaped transpose: {x_qkv_reshape_transposed.shape}')
        
        # q, k, v = x_qkv_reshape_transposed.unbind(dim=2)
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).transpose(1, 3).unbind(dim=2)
        print(f'X qkv reshaped transpose unbinded: {[t.shape for t in (q, k, v)]}')
        
        
        if self.pool_k is not None:
            k, k_thw = self.pool_k(k, thw)
            print(f'K after pooling: {k.shape}, K THW: {k_thw}')
        else:
            k_thw = thw
        if self.pool_v is not None:
            v, _ = self.pool_v(v, thw)
            print(f'V after pooling: {v.shape}')
        if self.pool_q is not None:
            q, thw = self.pool_q(q, thw)
            print(f'Q after pooling: {q.shape}, Q THW: {thw}')
        
        
        attn = torch.matmul(self.scaler * q, k.transpose(2, 3))
        print(f'Attention weights shape: {attn.shape}')
        
        if self.rel_pos_h is not None and self.rel_pos_w is not None and self.rel_pos_t is not None:
            attn = _sim_add_rel_pos(
                attn, 
                q,
                thw,
                k_thw,
                self.rel_pos_h,
                self.rel_pos_w,
                self.rel_pos_t
            )
        
        # if self.rel_pos_h is not None and self.rel_pos_w is not None and self.rel_pos_t is not None:
        #     attn = self._add_rel_pos(attn, q, thw)
        #     print(f'Attention weights after adding relative positional embedding: {attn.shape}')
        
        return x, thw

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


def test_sim_attn():
    cnf = get_block_setting()[1]
    print(f'embed dim: {cnf.input_channels}')
    print(f'output dim: {cnf.output_channels}')
    print(f'num_heads: {cnf.num_heads}')
    print(f'kernel_q={cnf.kernel_q}',
    f'kernel_kv={cnf.kernel_kv}', 
    f'stride_q={cnf.stride_q}',
    f'stride_kv={cnf.stride_kv}', '\n')
    
    dummy_input, thw = make_dummy_block_input()
    print(f'Original shape: {dummy_input.shape}')
    print(f'Original thw: {thw}', '\n')
    
    
    attn = SimMultiScaleAttention(
        list(dummy_input.shape),
        cnf.input_channels,
        cnf.output_channels,
        cnf.num_heads,
        kernel_q=cnf.kernel_q,
        kernel_kv=cnf.kernel_kv, 
        stride_q=cnf.stride_q,
        stride_kv=cnf.stride_kv
    )
    output, new_thw = attn(dummy_input, thw)
    
    print('\n', f'Output shape: {output.shape}')
    print(f'New THW: {new_thw}')


if __name__ == "__main__":
    # test_simvit()
    # test_simblock()
    test_sim_attn()
