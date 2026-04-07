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
from mamba.mamba_ssm.modules.SS2D_new import VSSLayer
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


class MambaEncoder(nn.Module):
    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[24, 48, 96, 192], d_state=16,
                 drop_path_rate=0.0, layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
            nn.InstanceNorm3d(dims[0])
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                # LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.InstanceNorm3d(dims[i]),
                nn.Conv3d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.gscs = nn.ModuleList()
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

            stage = nn.Sequential(
                nn.InstanceNorm3d(dims[i]),
                VSSLayer(
                    dim=dims[i],
                    depth=depths[i],
                    d_state=math.ceil(dims[i] / 6) if d_state is None else d_state,  # 20240109
                    drop=0.0,
                    attn_drop=0,
                    drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                    norm_layer=nn.LayerNorm,
                    # downsample=PatchMerging2D if (i < self.num_layers - 1) else None,
                    downsample=None,
                )
            )

            self.stages.append(stage)

            # self.gscs.append(gsc)
            cur += depths[i]

        self.out_indices = out_indices


        self.mlps = nn.ModuleList()
        for i_layer in range(4):
            layer = nn.InstanceNorm3d(dims[i_layer])
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
            feat_size=[24, 48, 96, 192],
            drop_path_rate=0,
            layer_scale_init_value=1e-6,
            hidden_size: int = 384,
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
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feat_size[0], out_channels=self.out_chans)

        # self.center = UnetConv3(filters[3], filters[4], self.is_batchnorm)
        self.gating4 = UnetGridGatingSignal3(hidden_size, feat_size[3], kernel_size=(1, 1, 1), is_batchnorm=True)
        self.gating3 = UnetGridGatingSignal3(feat_size[3], feat_size[2], kernel_size=(1, 1, 1), is_batchnorm=True)
        self.gating2 = UnetGridGatingSignal3(feat_size[2], feat_size[1], kernel_size=(1, 1, 1), is_batchnorm=True)
        self.gating1 = UnetGridGatingSignal3(feat_size[1], feat_size[0], kernel_size=(1, 1, 1), is_batchnorm=True)

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
