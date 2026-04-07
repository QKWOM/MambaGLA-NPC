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
from ..utils import UnetGridGatingSignal3
from ..grid_attention_layer import GridAttentionBlock3D
from model_segmamba.kan import KANLinear
from model_segmamba.eaa import EfficientAdditiveAttention
from typing import Optional, Tuple, Type
from mamba.mamba_ssm.modules.SS2D_just_local import VSSLayer
from models.Transformer import Transformer
from timm.models.layers import trunc_normal_, DropPath

import math
######################################################
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size (wd, wh, ww)
    Returns:
        windows: (B*num_windows, window_size[0]*window_size[1]*window_size[2], C)
    """
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2],
               window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0] * window_size[1] * window_size[2],
                                                                  C)
    return windows


def window_reverse(windows, window_size, D, H, W):
    """
    Args:
        windows: (B*num_windows, window_size[0]*window_size[1]*window_size[2], C)
        window_size (tuple[int]): Window size (wd, wh, ww)
        D (int): Depth of image
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, D, H, W, C)
    """
    B = int(windows.shape[0] / (D * H * W / window_size[0] / window_size[1] / window_size[2]))
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2],
                     window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


class WindowAttention3D(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The temporal length, height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (wd, wh, ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))

        # Get relative position index for each token inside the window
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index[:N, :N].reshape(-1)].reshape(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock3D(nn.Module):
    """ Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size (wd, wh, ww).
        shift_size (tuple[int]): Shift size for SW-MSA (wd, wh, ww).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=(7, 7, 7), shift_size=(0, 0, 0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        # Ensure shift_size is smaller than window_size
        for i in range(3):
            if self.shift_size[i] >= self.window_size[i]:
                self.shift_size[i] = self.window_size[i] // 2

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # Compute attention mask for SW-MSA
        if any(i > 0 for i in self.shift_size):
            D, H, W = self.window_size
            img_mask = torch.zeros((1, D, H, W, 1))
            slices_d = [slice(0, -self.window_size[0]),
                        slice(-self.window_size[0], -self.shift_size[0]),
                        slice(-self.shift_size[0], None)]
            slices_h = [slice(0, -self.window_size[1]),
                        slice(-self.window_size[1], -self.shift_size[1]),
                        slice(-self.shift_size[1], None)]
            slices_w = [slice(0, -self.window_size[2]),
                        slice(-self.window_size[2], -self.shift_size[2]),
                        slice(-self.shift_size[2], None)]

            cnt = 0
            for d in slices_d:
                for h in slices_h:
                    for w in slices_w:
                        img_mask[:, d, h, w, :] = cnt
                        cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2])
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            self.register_buffer("attn_mask", attn_mask)
        else:
            self.attn_mask = None

    def forward(self, x):
        B, D, H, W, C = x.shape
        window_size = self.window_size

        # Padding if needed
        pad_d = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_h = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_w = (window_size[2] - W % window_size[2]) % window_size[2]
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
            _, Dp, Hp, Wp, _ = x.shape
        else:
            Dp, Hp, Wp = D, H, W

        # Cyclic shift
        if any(i > 0 for i in self.shift_size):
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]),
                                   dims=(1, 2, 3))
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(shifted_x, window_size)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # Merge windows
        attn_windows = attn_windows.view(-1, window_size[0], window_size[1], window_size[2], C)
        shifted_x = window_reverse(attn_windows, window_size, Dp, Hp, Wp)

        # Reverse cyclic shift
        if any(i > 0 for i in self.shift_size):
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]),
                           dims=(1, 2, 3))
        else:
            x = shifted_x

        # Remove padding
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x = x[:, :D, :H, :W, :].contiguous()

        # Residual connection
        x = x + self.drop_path(self.norm1(x))

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x


class SwinTransformerLayer3D(nn.Module):
    """ A basic Swin Transformer layer for 3D data.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (tuple[int]): Local window size. Default: (7,7,7)
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(self, dim, depth, num_heads, window_size=(7, 7, 7),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.depth = depth

        # Build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0 if (i % 2 == 0) else window_size[0] // 2,
                            0 if (i % 2 == 0) else window_size[1] // 2,
                            0 if (i % 2 == 0) else window_size[2] // 2),
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(dim)

    def forward(self, x):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
        # Convert to (B, D, H, W, C)
        x = x.permute(0, 2, 3, 4, 1)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        # Convert back to (B, C, D, H, W)
        x = x.permute(0, 4, 1, 2, 3)
        return x
######################################################
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
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]

            return x



class MlpChannel(nn.Module):
    def __init__(self, hidden_size, mlp_dim, drop=0.5):
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


class SwinTransformerEncoder(nn.Module):
    def  __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[48, 96, 192, 384], d_state=16,
                 drop_path_rate=0.0, layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3], window_size=(7,7,7)):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
            # nn.InstanceNorm3d(dims[0])
            nn.BatchNorm3d(dims[0], eps=1e-6)

        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                # LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                # nn.InstanceNorm3d(dims[i]),
                nn.BatchNorm3d(dims[i], eps=1e-6, affine=False),
                nn.Conv3d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # self.gscs = nn.ModuleList()
        num_slices_list = [64, 32, 16, 8]
        cur = 0

        self.adapter1_norm = nn.ModuleList()
        self.adapter1_linear_down = nn.ModuleList()
        self.adapter1_Conv = nn.ModuleList()
        self.adapter1_act = nn.ModuleList()
        self.adapter1_linear_up = nn.ModuleList()

        self.adapter2_norm = nn.ModuleList()
        self.adapter2_linear_down = nn.ModuleList()
        self.adapter2_Conv = nn.ModuleList()
        self.adapter2_act = nn.ModuleList()
        self.adapter2_linear_up = nn.ModuleList()

        for i in range(4):
            # gsc = GSC(dims[i])

            self.adapter1_norm.append(nn.InstanceNorm3d(dims[i]))
            self.adapter1_linear_down.append(nn.Linear(dims[i], dims[i] // 2, bias=False))
            self.adapter1_Conv.append(nn.Conv3d(dims[i] // 2, dims[i] // 2, kernel_size=(3, 1, 1), padding='same'))
            self.adapter1_act.append(nn.GELU())
            self.adapter1_linear_up.append(nn.Linear(dims[i] // 2, dims[i], bias=False))

            self.adapter2_norm.append(nn.InstanceNorm3d(dims[i]))
            self.adapter2_linear_down.append(nn.Linear(dims[i], dims[i] // 2, bias=False))
            self.adapter2_Conv.append(nn.Conv3d(dims[i] // 2, dims[i] // 2, kernel_size=(3, 1, 1), padding='same'))
            self.adapter2_act.append(nn.GELU())
            self.adapter2_linear_up.append(nn.Linear(dims[i] // 2, dims[i], bias=False))

            # Calculate number of heads (at least 1)
            num_heads = max(1, dims[i] // 16)
            # print(num_heads)

            stage = nn.Sequential(
                # nn.InstanceNorm3d(dims[i]),
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                SwinTransformerLayer3D(
                    dim=dims[i],
                    depth=depths[i],
                    num_heads=num_heads,
                    window_size=window_size,
                    drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                    drop=0.0,
                    attn_drop=0.0
                )
            )

            self.stages.append(stage)

            # self.gscs.append(gsc)
            cur += depths[i]

        self.out_indices = out_indices


        self.mlps = nn.ModuleList()
        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(4):
            # layer = nn.InstanceNorm3d(dims[i_layer])
            layer = norm_layer(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
            self.mlps.append(MlpChannel(dims[i_layer], 2 * (dims[i_layer])))


    def global_avg_pool_and_sigmoid(self, k):
        # k:(B,C,D,H,W)
        B, C, D, H, W = k.shape
        # pooled_tensor = k.mean(dim=(2, 3, 4), keepdim=True)
        pooled_tensor = F.max_pool3d(k, kernel_size=(D, H, W), stride=(D, H, W))
        sigmoid_tensor = torch.sigmoid(pooled_tensor)
        return sigmoid_tensor * k



    def forward_features(self, x):
        # x: (1,1,128,128,128) (1,48,64,64,64) (1,96,32,32,32) (1,192,16,16,16)
        outs = []
        for i in range(4):
            # SSM Block
            x = self.downsample_layers[i](x)
            # print(x.shape)
            x_input = x
            # x: (1,48,64,64,64) (1,96,32,32,32) (1,192,16,16,16) (1,384,8,8,8)

            # x = self.gscs[i](x)

            x_pre_adapter1 = x
            x = self.adapter1_norm[i](x)
            x = torch.permute(x, (0, 2, 3, 4, 1))
            x = self.adapter1_linear_down[i](x)
            x = torch.permute(x, (0, 4, 1, 2, 3))
            x = self.adapter1_Conv[i](x)
            x = self.adapter1_act[i](x)
            x = torch.permute(x, (0, 2, 3, 4, 1))
            x = self.adapter1_linear_up[i](x)
            x = torch.permute(x, (0, 4, 1, 2, 3))
            x+=x_pre_adapter1

            x = self.stages[i](x)

            x_pre_adapter2 = x
            x = self.adapter2_norm[i](x)
            x = torch.permute(x, (0, 2, 3, 4, 1))
            x = self.adapter2_linear_down[i](x)
            x = torch.permute(x, (0, 4, 1, 2, 3))
            x = self.adapter2_Conv[i](x)
            x = self.adapter2_act[i](x)
            x = torch.permute(x, (0, 2, 3, 4, 1))
            x = self.adapter2_linear_up[i](x)
            x = torch.permute(x, (0, 4, 1, 2, 3))
            x+=x_pre_adapter2

            x_2 = x

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                x_out = self.mlps[i](x_out)
                ## resdual
                # x_out = x_out+x_2
                ##
                # x_out = x_out+output_CNN
                x_out = self.global_avg_pool_and_sigmoid(x_out)
                # x_out = x_out+x_input
                outs.append(x_out)
                x = x_out
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
            norm_name="instance",
            conv_block: bool = True,
            res_block: bool = True,
            spatial_dims=3,
            mode='grid_and_channel',
    ) -> None:
        super().__init__()

        self.num_slices_list = [64, 32, 16, 8]


        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value
        self.mode = mode
        self.spatial_dims = spatial_dims
        self.vit = SwinTransformerEncoder(
            in_chans= self.in_chans,
            depths=self.depths,
            dims=self.feat_size,
            drop_path_rate=self.drop_path_rate,
            layer_scale_init_value=1e-6,
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
        self.gating4 = UnetGridGatingSignal3(768, 384, kernel_size=(1, 1, 1), is_batchnorm=True)
        self.gating3 = UnetGridGatingSignal3(384, 192, kernel_size=(1, 1, 1), is_batchnorm=True)
        self.gating2 = UnetGridGatingSignal3(192, 96, kernel_size=(1, 1, 1), is_batchnorm=True)
        self.gating1 = UnetGridGatingSignal3(96, 48, kernel_size=(1, 1, 1), is_batchnorm=True)

        # attention blocks
        self.attentionblock1 = GridAttentionBlock3D(in_channels=feat_size[0], gating_channels=feat_size[0],
                                                    inter_channels=feat_size[0], sub_sample_factor=(2, 2, 2),
                                                    mode=mode)
        self.attentionblock2 = GridAttentionBlock3D(in_channels=feat_size[1], gating_channels=feat_size[1],
                                                    inter_channels=feat_size[1], sub_sample_factor=(2, 2, 2),
                                                    mode=mode)
        self.attentionblock3 = GridAttentionBlock3D(in_channels=feat_size[2], gating_channels=feat_size[2],
                                                    inter_channels=feat_size[2], sub_sample_factor=(2, 2, 2),
                                                    mode=mode)
        self.attentionblock4 = GridAttentionBlock3D(in_channels=feat_size[3], gating_channels=feat_size[3],
                                                    inter_channels=feat_size[3], sub_sample_factor=(2,2,2),
                                                    mode=mode)

    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def global_avg_pool_and_sigmoid(self, k):
        pooled_tensor = k.mean(dim=(2, 3, 4), keepdim=True)
        sigmoid_tensor = torch.sigmoid(pooled_tensor)
        return sigmoid_tensor * k

    def global_max_pool_and_sigmoid(self, k):
        # 对于最后三个维度进行最大池化
        pooled_tensor = k.max(dim=2, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        sigmoid_tensor = torch.sigmoid(pooled_tensor)
        return sigmoid_tensor * k

    def forward(self, x_in):
        outs = self.vit(x_in)
        # print(x_in.shape)
        # print(len(outs))
        enc1 = self.encoder1(x_in)  # 48xDxHxW

        x2 = outs[0]  # 48xD/2xH/2xW/2
        enc2 = self.encoder2(x2)  # 96xD/2xH/2xW/2

        x3 = outs[1]  # 96xD/4xH/4xW/4
        enc3 = self.encoder3(x3)  # 192xD/4xH/4xW/4

        x4 = outs[2]  # 192xD/8xH/8xW/8
        enc4 = self.encoder4(x4)  # 384xD/8xH/8xW/8

        x5 = outs[3]  # 384xD/16xH/16xW/16
        # print(x5.shape)
        enc_hidden = self.encoder5(x5)  # 768xD/16xH/16xW/16
        # print(enc_hidden.shape,enc4.shape)
        gating4 = self.gating4(enc_hidden) # (1,384,8,8,8)

        g_enc4 = self.attentionblock4(enc4, gating4) # (1,384,16,16,16)
        # enc_hidden = self.global_avg_pool_and_sigmoid(enc_hidden)
        dec3 = self.decoder5(enc_hidden, g_enc4) # (1,384,16,16,16)


        gating3 = self.gating3(dec3) # (1,192,16,16,16)

        g_enc3 = self.attentionblock3(enc3, gating3)  # (1,192,32,32,32)
        # dec3 = self.global_avg_pool_and_sigmoid(dec3)
        dec2 = self.decoder4(dec3, g_enc3) # (1,192,32,32,32)


        gating2 = self.gating2(dec2) # (1,96,32,32,32)

        g_enc2 = self.attentionblock2(enc2, gating2)  # (1,96,64,64,64)
        # dec2 = self.global_max_pool_and_sigmoid(dec2)
        dec1 = self.decoder3(dec2, g_enc2) # (1,96,64,64,64)


        gating1 = self.gating1(dec1) # (1,48,64,64,64)

        g_enc1 = self.attentionblock1(enc1, gating1)  # (1,48,128,128,128)
        # dec1 = self.global_max_pool_and_sigmoid(dec1)
        dec0 = self.decoder2(dec1, g_enc1) # (1,48,128,128,128)


        # dec0 = self.global_max_pool_and_sigmoid(dec0)
        out = self.decoder1(dec0)  # (1,48,128,128,128)

        result = self.out(out)
        # print(result.shape)
        return result


if __name__ == '__main__':
    model = SegMamba().to("cuda")
    input = torch.ones(1, 1, 128, 128, 128).to("cuda")

    flops, params = profile(model, (input,))
    print('flops: ', flops, 'params: ', params)
    k = model(input)
