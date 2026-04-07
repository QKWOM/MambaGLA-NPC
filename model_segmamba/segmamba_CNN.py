# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
import torch.nn as nn
import torch 
from functools import partial
import itertools
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from mamba.mamba_ssm.modules.mamba_simple import Mamba
import torch.nn.functional as F 
from .utils import UnetGridGatingSignal3
from .grid_attention_layer import GridAttentionBlock3D
from model_segmamba.kan import KANLinear
from model_segmamba.eaa import EfficientAdditiveAttention 
from typing import Optional, Tuple, Type
from mamba.mamba_ssm.modules.SS2D import VSSLayer
import math


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]

            return x

class MultiMamba(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2, num_slices=None):
        super(MultiMamba, self).__init__()
        dims = [2,3,4]
        self.permutations = [torch.tensor(x) for x in list(itertools.permutations(dims, len(dims)))]
        self.permutations = [
            self.permutations[0],
            self.permutations[1],
            self.permutations[2],
            self.permutations[4],
        ]
        self.mambas = nn.ModuleList([
            Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                bimamba_type="v2",
                nslices=num_slices,
            )
            for _ in range(len(self.permutations))
        ])
        self.permutations = torch.combinations(torch.tensor(dims), r=3)

    def forward(self, x):
        B, C, H, W, D = x.shape
        ys = []
        for permutation, mamba in zip(self.permutations,self.mambas):
            h = x.permute(0, 1, *permutation).reshape(B, C, H*W*D).permute(0, 2, 1)
            S1, S2, S3 = torch.tensor([H, W, D])[permutation-2]
            inv_permutation = list(torch.argsort(permutation)+2)
            out = mamba(h)
            out = out.permute(0, 2, 1)
            out = out.reshape(B, C, S1, S2, S3)
            out = out.permute(0, 1, *inv_permutation)
            ys.append(out)

        y = torch.stack(ys, dim=1).mean(dim=1)

        return y
    
class MambaLayer(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2, num_slices=None):
        super().__init__()
        self.dim = dim

        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                bimamba_type="v3",
                nslices=num_slices,
        )

        ###################
        # self.new_mamba = MultiMamba(dim)
        ####################

        # self.eaa = EfficientAdditiveAttention(in_dims=dim, token_dim=dim)
        # self.skip_scale = nn.Parameter(torch.ones(1))
    def forward(self, x):
        B, C = x.shape[:2]
        x_skip = x
        assert C == self.dim

        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)   #original
        # x_mamba = self.mamba(x_norm) + self.skip_scale * x_flat #Test26
        # x_mamba = self.eaa(x_norm)
        # x_mamba = self.norm(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)

        # x_mamba = self.new_mamba(x)
        # out = x_mamba

        out = out + x_skip
        
        
        return out
    
class MlpChannel(nn.Module):
    def __init__(self,hidden_size, mlp_dim, drop=0.5):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)
        # self.drop = nn.Dropout(drop)# random inactivation, the radio is 0.5


    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        x = self.fc2(x)
        # x = self.drop(x)
        return x

class GSC(nn.Module):
    def __init__(self, in_channles) -> None:
        super().__init__()

        self.proj = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm = nn.InstanceNorm3d(in_channles)
        self.nonliner = nn.ReLU()

        self.proj2 = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm2 = nn.InstanceNorm3d(in_channles)
        self.nonliner2 = nn.ReLU()

        self.proj3 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm3 = nn.InstanceNorm3d(in_channles)
        self.nonliner3 = nn.ReLU()

        self.proj4 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm4 = nn.InstanceNorm3d(in_channles)
        self.nonliner4 = nn.ReLU()

    def forward(self, x):

        x_residual = x 

        x1 = self.proj(x)
        x1 = self.norm(x1)
        x1 = self.nonliner(x1)

        x1 = self.proj2(x1)
        x1 = self.norm2(x1)
        x1 = self.nonliner2(x1)

        x2 = self.proj3(x)
        x2 = self.norm3(x2)
        x2 = self.nonliner3(x2)

        x = x1 + x2
        x = self.proj4(x)
        x = self.norm4(x)
        x = self.nonliner4(x)
        
        return x + x_residual


class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):  # 0.25
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        # print(D_features,D_hidden_features)
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)

    def forward(self, x):
        # x is (BT, HW+1, D)
        # print(x.shape)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x

class MLPBlock(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            mlp_dim: int,
            act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))

class PatchEmbed3D(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
            self,
            kernel_size: Tuple[int, int] = (16, 16, 16),
            stride: Tuple[int, int] = (16, 16, 16),
            padding: Tuple[int, int] = (0, 0, 0),
            in_chans: int = 1,
            embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C X Y Z -> B X Y Z C
        # x = x.permute(0, 2, 3, 4, 1)
        return x

class Block3D(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            act_layer: Type[nn.Module] = nn.GELU,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            window_size: int = 0,
            input_size: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size, window_size),
        )
        # self.patch_embed = PatchEmbed3D(
        #     kernel_size=(16, 16, 16),
        #     stride=(16, 16, 16),
        #     in_chans=dim,
        #     embed_dim=dim,
        # )
        
        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(x.shape)
        H1,W1,D1 = x.shape[2],x.shape[3],x.shape[4]
        x = F.interpolate(x, size=(H1//4,W1//4,D1//4), mode='trilinear', align_corners=False)
        # x = F.max_pool3d(x, kernel_size=4, stride=4)

        x = x.permute(0,4,2,3,1)   
        # print(x.shape)
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            D, H, W = x.shape[1], x.shape[2], x.shape[3]
            x, pad_dhw = window_partition3D(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition3D(x, self.window_size, pad_dhw, (D, H, W))

        x = shortcut + x
        # x = x + self.mlp(self.norm2(x)) 

        
        x = x.permute(0,4,2,3,1)
        x = F.interpolate(x, size=(H1,W1,D1), mode='trilinear', align_corners=False)
        # print(x.shape)
        return x

class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            input_size: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                    input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_d = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[2] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, D * H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, D * H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_d, self.rel_pos_h, self.rel_pos_w, (D, H, W), (D, H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, D, H, W, -1).permute(0, 2, 3, 4, 1, 5).reshape(B, D, H, W, -1)
        x = self.proj(x)

        return x

def window_partition3D(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, D, H, W, C = x.shape

    pad_d = (window_size - D % window_size) % window_size
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size

    if pad_h > 0 or pad_w > 0 or pad_d > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
    Hp, Wp, Dp = H + pad_h, W + pad_w, D + pad_d

    x = x.view(B, Dp // window_size, window_size, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size, window_size, window_size, C)
    return windows, (Dp, Hp, Wp)


def window_unpartition3D(
        windows: torch.Tensor, window_size: int, pad_dhw: Tuple[int, int, int], dhw: Tuple[int, int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Dp, Hp, Wp = pad_dhw
    D, H, W = dhw
    B = windows.shape[0] // (Dp * Hp * Wp // window_size // window_size // window_size)
    x = windows.view(B, Dp // window_size, Hp // window_size, Wp // window_size, window_size, window_size, window_size,
                     -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, Dp, Hp, Wp, -1)

    if Hp > H or Wp > W or Dp > D:
        x = x[:, :D, :H, :W, :].contiguous()
    return x

def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]

def add_decomposed_rel_pos(
        attn: torch.Tensor,
        q: torch.Tensor,
        rel_pos_d: torch.Tensor,
        rel_pos_h: torch.Tensor,
        rel_pos_w: torch.Tensor,
        q_size: Tuple[int, int, int],
        k_size: Tuple[int, int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_d, q_h, q_w = q_size
    k_d, k_h, k_w = k_size

    Rd = get_rel_pos(q_d, k_d, rel_pos_d)
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_d, q_h, q_w, dim)

    rel_d = torch.einsum("bdhwc,dkc->bdhwk", r_q, Rd)
    rel_h = torch.einsum("bdhwc,hkc->bdhwk", r_q, Rh)
    rel_w = torch.einsum("bdhwc,wkc->bdhwk", r_q, Rw)

    attn = (
            attn.view(B, q_d, q_h, q_w, k_d, k_h, k_w) + rel_d[:, :, :, :, None, None] + rel_h[:, :, :, None, :,
                                                                                         None] + rel_w[:, :, :, None,
                                                                                                 None, :]
    ).view(B, q_d * q_h * q_w, k_d * k_h * k_w)

    return attn

class MambaEncoder(nn.Module):
    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[48, 96, 192, 384], d_state=16,
                 drop_path_rate=0.1, layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
              nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
              nn.InstanceNorm3d(dims[0])
              )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                # LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.InstanceNorm3d(dims[i]),
                nn.Conv3d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        
        self.stage1 = nn.ModuleList()
        self.stage2 = nn.ModuleList()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.gscs = nn.ModuleList()
        num_slices_list = [64, 32, 16, 8]
        cur = 0
        for i in range(4):
            gsc = GSC(dims[i])

            # stage = nn.Sequential(
            #     *[MambaLayer(dim=dims[i], num_slices=num_slices_list[i]) for j in range(depths[i])]
            # )
            # stage = nn.Sequential(
            #     SS2D(d_model=dims[i]),
            # )

            stage = nn.Sequential(
                VSSLayer(
                    dim=dims[i],
                    depth=depths[i],
                    d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,  # 20240109
                    drop=0.0,
                    attn_drop=0,
                    drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                    norm_layer=nn.LayerNorm,
                    # downsample=PatchMerging2D if (i < self.num_layers - 1) else None,
                    downsample=None,
                )
            )

            # stage = nn.Sequential(
            #     # MambaLayer(dim=dims[i], num_slices=num_slices_list[i]),
            #     Block3D(
            #         dim=dims[i],
            #         num_heads=12,
            #         mlp_ratio=4,
            #         qkv_bias=True,
            #         norm_layer=nn.LayerNorm,
            #         act_layer=nn.GELU,
            #         use_rel_pos=True,
            #         rel_pos_zero_init=True,
            #         window_size=4,
            #         input_size=(num_slices_list[i] // 16, num_slices_list[i] // 16, num_slices_list[i] // 16),
            #     ),
            #     Block3D(
            #         dim=dims[i],
            #         num_heads=12,
            #         mlp_ratio=4,
            #         qkv_bias=True,
            #         norm_layer=nn.LayerNorm,
            #         act_layer=nn.GELU,
            #         use_rel_pos=True,
            #         rel_pos_zero_init=True,
            #         window_size=4,
            #         input_size=(num_slices_list[i] // 16, num_slices_list[i] // 16, num_slices_list[i] // 16),
            #     ),
            #     # MambaLayer(dim=dims[i], num_slices=num_slices_list[i]),
            # )

            # stage1 = nn.Sequential(
            #     MambaLayer(dim=dims[i], num_slices=num_slices_list[i]),  
            #     Block3D(
            #         dim=dims[i],
            #         num_heads=12,
            #         mlp_ratio=4,
            #         qkv_bias=True,
            #         norm_layer=nn.LayerNorm,
            #         act_layer=nn.GELU,
            #         use_rel_pos=True,
            #         rel_pos_zero_init=True,
            #         window_size=4,
            #         input_size=(num_slices_list[i] // 16, num_slices_list[i] // 16, num_slices_list[i] // 16),
            #     ),
            # )
            # stage2 = nn.Sequential(    
            #     Block3D(
            #         dim=dims[i],
            #         num_heads=12,
            #         mlp_ratio=4,
            #         qkv_bias=True,
            #         norm_layer=nn.LayerNorm,
            #         act_layer=nn.GELU,
            #         use_rel_pos=True,
            #         rel_pos_zero_init=True,
            #         window_size=4,
            #         input_size=(num_slices_list[i] // 16, num_slices_list[i] // 16, num_slices_list[i] // 16),
            #     ),
            #     MambaLayer(dim=dims[i], num_slices=num_slices_list[i]),
            # )

            self.stages.append(stage)

            # self.stage1.append(stage1)
            # self.stage2.append(stage2)

            self.gscs.append(gsc)
            cur += depths[i]

        self.out_indices = out_indices

        # self.fused = nn.ModuleList()
        # for i in range(4):
        #     k = nn.Conv3d(dims[i], dims[i], kernel_size=3, stride=1, padding=1)
        #     self.fused.append(k)

        self.mlps = nn.ModuleList()
        for i_layer in range(4):
            layer = nn.InstanceNorm3d(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
            self.mlps.append(MlpChannel(dims[i_layer], 2 * (dims[i_layer])))

        # self.input_Adapter = nn.ModuleList()
        # for i in range(4):
        #     input_Adapter = Adapter(num_slices_list[i])
        #     self.input_Adapter.append(input_Adapter)

        # self.Conv_Block = nn.ModuleList()
        # for i in range(4):
        #     ConvBlock = nn.Sequential(
        #         nn.BatchNorm3d(dims[i]),
        #         nn.Conv3d(in_channels=dims[i], out_channels=dims[i], kernel_size=3, stride=1, padding=1),
        #         nn.BatchNorm3d(dims[i]),
        #         nn.ReLU(),
        #         nn.Conv3d(in_channels=dims[i], out_channels=dims[i], kernel_size=3, stride=1, padding=1),
        #         nn.BatchNorm3d(dims[i]),
        #         nn.ReLU(),
        #         nn.Conv3d(in_channels=dims[i], out_channels=dims[i], kernel_size=1, stride=1),
        #         nn.ReLU()
        #     )
        #     self.Conv_Block.append(ConvBlock)

    def global_avg_pool_and_sigmoid(self,k):
        # k:(B,C,D,H,W)
        B, C, D, H, W = k.shape
        # pooled_tensor = k.mean(dim=(2, 3, 4), keepdim=True)
        pooled_tensor = F.max_pool3d(k, kernel_size=(D, H, W), stride=(D, H, W))
        sigmoid_tensor = torch.sigmoid(pooled_tensor)
        return sigmoid_tensor*k

    def forward_features(self, x):
        # x: (1,1,128,128,128) (1,48,64,64,64) (1,96,32,32,32) (1,192,16,16,16)
        outs = []
        for i in range(4):
            # SSM Block
            x = self.downsample_layers[i](x)
            # print(x.shape)
            x_input = x
            # x: (1,48,64,64,64) (1,96,32,32,32) (1,192,16,16,16) (1,384,8,8,8)
            # input_CNN, input_SSM = x.chunk(2,1)
            # x = input_SSM


            # cnn_x = self.input_Adapter[i](x)
            # cnn_x = x
            # output_CNN = self.Conv_Block[i](cnn_x)


            x = self.gscs[i](x)
            
            # output_CNN = output_CNN+x
            # print("gsc:", x.shape)
            
            x = self.stages[i](x)
            
            ##############
            # x = x.permute(0,4,1,2,3)
            ###############

            # print(x.shape)
            # gsc_input = x
            # x = self.stage1[i](x)
            # x+=gsc_input
            # x = self.stage2[i](x)

            x_2 = x 
            # x = x + output_CNN
            # x = self.global_avg_pool_and_sigmoid(x)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                x_out = self.mlps[i](x_out)
                ## resdual
                # x_out = x_out+x_2
                ##
                # x_out = x_out+output_CNN
                # x_out = self.global_avg_pool_and_sigmoid(x_out)
                # x_out = x_out+x_input
                outs.append(x_out)
                # x = x_out
            # outs.append(x)
        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x

class SegMamba(nn.Module):
    def __init__(
        self,
        in_chans=1,
        out_chans=2,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        hidden_size: int = 768,
        norm_name = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        spatial_dims=3,
        mode = 'grid_and_channel',
    ) -> None:
        super().__init__()

        # self.stages = nn.ModuleList()
        self.num_slices_list = [64, 32, 16, 8]
        # for i in range(4):
        #     stage = nn.Sequential(
        #         *[MambaLayer(dim=feat_size[i], num_slices=self.num_slices_list[i]) for j in range(depths[i])]
        #     )
        #     self.stages.append(stage)

        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value
        self.mode = mode
        self.spatial_dims = spatial_dims
        self.vit = MambaEncoder(in_chans, 
                                depths=depths,
                                dims=feat_size,
                                drop_path_rate=drop_path_rate,
                                layer_scale_init_value=layer_scale_init_value,
                              )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.in_chans,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.hidden_size,
            out_channels=self.feat_size[3],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=48, out_channels=self.out_chans)

        # self.center = UnetConv3(filters[3], filters[4], self.is_batchnorm)
        # self.gating4 = UnetGridGatingSignal3(768, 384, kernel_size=(1, 1, 1), is_batchnorm=True)
        # self.gating3 = UnetGridGatingSignal3(384, 192, kernel_size=(1, 1, 1), is_batchnorm=True)
        # self.gating2 = UnetGridGatingSignal3(192, 96, kernel_size=(1, 1, 1), is_batchnorm=True)
        # self.gating1 = UnetGridGatingSignal3(96, 48, kernel_size=(1, 1, 1), is_batchnorm=True)

        # attention blocks 
        # self.attentionblock1 = GridAttentionBlock3D(in_channels=feat_size[0], gating_channels=feat_size[0],
        #                                             inter_channels=feat_size[0], sub_sample_factor=(2, 2, 2),
        #                                             mode=mode)
        # self.attentionblock2 = GridAttentionBlock3D(in_channels=feat_size[1], gating_channels=feat_size[1],
        #                                             inter_channels=feat_size[1], sub_sample_factor=(2, 2, 2),
        #                                             mode=mode)
        # self.attentionblock3 = GridAttentionBlock3D(in_channels=feat_size[2], gating_channels=feat_size[2],
        #                                             inter_channels=feat_size[2], sub_sample_factor=(2, 2, 2),
        #                                             mode=mode)
        # self.attentionblock4 = GridAttentionBlock3D(in_channels=feat_size[3], gating_channels=feat_size[3],
        #                                             inter_channels=feat_size[3], sub_sample_factor=(2,2,2),
        #                                             mode=mode)

    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x
    
    def global_avg_pool_and_sigmoid(self,k):
        pooled_tensor = k.mean(dim=(2, 3, 4), keepdim=True)
        sigmoid_tensor = torch.sigmoid(pooled_tensor)
        return sigmoid_tensor*k
    def global_max_pool_and_sigmoid(self, k):
        # 对于最后三个维度进行最大池化
        pooled_tensor = k.max(dim=2, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        sigmoid_tensor = torch.sigmoid(pooled_tensor)
        return sigmoid_tensor * k





    def forward(self, x_in):
        outs = self.vit(x_in)
        # print(x_in.shape)
        enc1 = self.encoder1(x_in) # 48xDxHxW

        x2 = outs[0]    # 48xD/2xH/2xW/2
        enc2 = self.encoder2(x2) # 96xD/2xH/2xW/2

        x3 = outs[1]    # 96xD/4xH/4xW/4
        enc3 = self.encoder3(x3) # 192xD/4xH/4xW/4

        x4 = outs[2]    # 192xD/8xH/8xW/8
        enc4 = self.encoder4(x4) # 384xD/8xH/8xW/8

        x5 = outs[3]    # 384xD/16xH/16xW/16
        # print(x5.shape)
        enc_hidden = self.encoder5(x5) # 768xD/16xH/16xW/16
        # print(enc_hidden.shape,enc4.shape)
        # gating4 = self.gating4(enc_hidden) # (1,384,8,8,8)

        # g_enc4 = self.attentionblock4(enc4, gating4) # (1,384,16,16,16)
        # enc_hidden = self.global_avg_pool_and_sigmoid(enc_hidden)
        # dec3 = self.decoder5(enc_hidden, g_enc4) # (1,384,16,16,16)
        dec3 = self.decoder5(enc_hidden, enc4)
        
    
        # gating3 = self.gating3(dec3) # (1,192,16,16,16)

        # g_enc3 = self.attentionblock3(enc3, gating3)  # (1,192,32,32,32)
        # dec3 = self.global_avg_pool_and_sigmoid(dec3)
        # dec2 = self.decoder4(dec3, g_enc3) # (1,192,32,32,32)
        dec2 = self.decoder4(dec3, enc3)
        

        # gating2 = self.gating2(dec2) # (1,96,32,32,32)

        # g_enc2 = self.attentionblock2(enc2, gating2)  # (1,96,64,64,64)
        # dec2 = self.global_max_pool_and_sigmoid(dec2)
        # dec1 = self.decoder3(dec2, g_enc2) # (1,96,64,64,64)
        dec1 = self.decoder3(dec2, enc2)
        

        # gating1 = self.gating1(dec1) # (1,48,64,64,64)

        # g_enc1 = self.attentionblock1(enc1, gating1)  # (1,48,128,128,128)
        # dec1 = self.global_max_pool_and_sigmoid(dec1)
        # dec0 = self.decoder2(dec1, g_enc1) # (1,48,128,128,128)
        dec0 = self.decoder2(dec1, enc1)

        # dec0 = self.global_max_pool_and_sigmoid(dec0)
        out = self.decoder1(dec0)# (1,48,128,128,128)
        
        result = self.out(out)
        # print(result.shape)
        return result
    
if __name__ == '__main__':
    model = SegMamba().to("cuda")
    input = torch.ones(1,1,128,128,128).to("cuda")

    flops, params = profile(model, (input,))
    print('flops: ', flops, 'params: ', params)
    k = model(input)
    