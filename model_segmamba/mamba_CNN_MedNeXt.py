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

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from mamba.mamba_ssm.modules.mamba_simple import Mamba
import torch.nn.functional as F 
from .utils import UnetGridGatingSignal3
from .grid_attention_layer import GridAttentionBlock3D
from model_segmamba.kan import KANLinear
from MedNeXt.blocks import *

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
    
    def forward(self, x):
        B, C = x.shape[:2]
        x_skip = x
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)

        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        out = out + x_skip
        
        
        return out
    
class MlpChannel(nn.Module):
    def __init__(self,hidden_size, mlp_dim, ):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
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
        self.D_fc1 = KANLinear(D_features, D_hidden_features)
        self.D_fc2 = KANLinear(D_hidden_features, D_features)

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
def channel_shuffle(x: Tensor, groups: int) -> Tensor:

    batch_size, num_channels, Depth, height, width,  = x.size()
    channels_per_group = num_channels // groups

    # reshape
    # [batch_size, num_channels, height, width] -> [batch_size, groups, channels_per_group, height, width]
    x = x.view(batch_size, Depth, height, width, groups, channels_per_group)

    x = torch.transpose(x, 4, 5).contiguous()

    # flatten
    x = x.view(batch_size, Depth, height, width, -1)
    x = x.permute(0,4,1,2,3)
    return x
class MambaEncoder(nn.Module):
    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[48, 96, 192, 384],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]):
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
        self.gscs = nn.ModuleList()
        num_slices_list = [64, 32, 16, 8]
        cur = 0
        for i in range(4):
            gsc = GSC(dims[i])

            stage = nn.Sequential(
                *[MambaLayer(dim=dims[i], num_slices=num_slices_list[i]) for j in range(depths[i])]
            )

            self.stages.append(stage)
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

        self.Conv_Block = nn.ModuleList()
        for i in range(4):
            ConvBlock = nn.Sequential(
                nn.BatchNorm3d(dims[i]),
                nn.Conv3d(in_channels=dims[i], out_channels=dims[i], kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(dims[i]),
                nn.ReLU(),
                nn.Conv3d(in_channels=dims[i], out_channels=dims[i], kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(dims[i]),
                nn.ReLU(),
                nn.Conv3d(in_channels=dims[i], out_channels=dims[i], kernel_size=1, stride=1),
                nn.ReLU()
            )
            self.Conv_Block.append(ConvBlock)

    def global_avg_pool_and_sigmoid(self,k):
        pooled_tensor = k.mean(dim=(2, 3, 4), keepdim=True)
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
            output_CNN = self.Conv_Block[i](x)
            x = self.gscs[i](x)
            
            # output_CNN = output_CNN+x
            # print("gsc:", x.shape)
            x = self.stages[i](x) 
            # x = x + output_CNN
            # x = self.global_avg_pool_and_sigmoid(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                x_out = self.mlps[i](x_out)
                x_out = x_out+output_CNN+x_input
                x_out = self.global_avg_pool_and_sigmoid(x_out)
                # x_out = channel_shuffle(x_out,groups=2)
                # x_out = self.fused[i](x_out)
                outs.append(x_out)
                x = x_out
        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x

class SegMamba(nn.Module):
    def __init__(
        self,
        in_chans=1,
        out_chans=1,
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
        
        block_counts: list = [2,2,2,2,2,2,2,2,2,2,2],
        exp_r = 4,
    ) -> None:
        super().__init__()

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
        ###########################################
        self.enc_kernel_size = 7
        self.dec_kernel_size = 7
        self.do_res = False
        self.do_res_up_down = False
        self.norm_type = 'group'
        self.dim='3d'
        self.grn = False
        self.exp_r = [exp_r for i in range(len(block_counts))]
        ############################################

        self.stem = nn.Conv3d(1, 48, kernel_size=1)

        self.enc_block_0 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=self.in_chans,
                out_channels=self.in_chans,
                exp_r=self.exp_r[0],
                kernel_size=self.enc_kernel_size,
                do_res=self.do_res,
                norm_type=self.norm_type,
                dim=self.dim,
                grn=self.grn
                ) 
            for i in range(block_counts[0])]
        ) 
        # after down : chan->2 H,W,D->1/2
        self.down_0 = MedNeXtDownBlock(
            in_channels=self.in_chans,
            out_channels=self.feat_size[0],
            exp_r=self.exp_r[1],
            kernel_size=self.enc_kernel_size,
            do_res=self.do_res_up_down,
            norm_type=self.norm_type,
            dim=self.dim
        )

        self.enc_block_1 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=self.feat_size[0],
                out_channels=self.feat_size[0],
                exp_r=self.exp_r[1],
                kernel_size=self.enc_kernel_size,
                do_res=self.do_res,
                norm_type=self.norm_type,
                dim=self.dim,
                grn=self.grn
                ) 
            for i in range(block_counts[1])]
        ) 
        # after down : chan->2 H,W,D->1/2
        self.down_1 = MedNeXtDownBlock(
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[1],
            exp_r=self.exp_r[2],
            kernel_size=self.enc_kernel_size,
            do_res=self.do_res_up_down,
            norm_type=self.norm_type,
            dim=self.dim
        )

        self.enc_block_2 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=self.feat_size[1],
                out_channels=self.feat_size[1],
                exp_r=self.exp_r[2],
                kernel_size=self.enc_kernel_size,
                do_res=self.do_res,
                norm_type=self.norm_type,
                dim=self.dim,
                grn=self.grn
                ) 
            for i in range(block_counts[2])]
        ) 
        # after down : chan->2 H,W,D->1/2
        self.down_2 = MedNeXtDownBlock(
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[2],
            exp_r=self.exp_r[3],
            kernel_size=self.enc_kernel_size,
            do_res=self.do_res_up_down,
            norm_type=self.norm_type,
            dim=self.dim
        )

        self.enc_block_3 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=self.feat_size[2],
                out_channels=self.feat_size[2],
                exp_r=self.exp_r[3],
                kernel_size=self.enc_kernel_size,
                do_res=self.do_res,
                norm_type=self.norm_type,
                dim=self.dim,  
                grn=self.grn
                ) 
            for i in range(block_counts[3])]
        ) 
        # after down : chan->2 H,W,D->1/2
        self.down_3 = MedNeXtDownBlock(
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[3],
            exp_r=self.exp_r[4],
            kernel_size=self.enc_kernel_size,
            do_res=self.do_res_up_down,
            norm_type=self.norm_type,
            dim=self.dim
        )

        self.enc_block_4 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=self.feat_size[3],
                out_channels=self.feat_size[3],
                exp_r=self.exp_r[4],
                kernel_size=self.enc_kernel_size,
                do_res=self.do_res,
                norm_type=self.norm_type,
                dim=self.dim,  
                grn=self.grn
                ) 
            for i in range(block_counts[4])]
        ) 
        # after down : chan->2 H,W,D->1/2
        self.down_4 = MedNeXtDownBlock(
            in_channels=self.feat_size[3],
            out_channels=hidden_size,
            exp_r=self.exp_r[5],
            kernel_size=self.enc_kernel_size,
            do_res=self.do_res_up_down,
            norm_type=self.norm_type,
            dim=self.dim
        )

        self.bottleneck = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=hidden_size,
                out_channels=hidden_size,
                exp_r=self.exp_r[5],
                kernel_size=self.dec_kernel_size,
                do_res=self.do_res,
                norm_type=self.norm_type,
                dim=self.dim,
                grn=self.grn
                )
            for i in range(block_counts[5])]
        )

        self.up_4 = MedNeXtUpBlock(
            in_channels=hidden_size,
            out_channels=self.feat_size[3],
            exp_r=self.exp_r[6],
            kernel_size=self.dec_kernel_size,
            do_res=self.do_res_up_down,
            norm_type=self.norm_type,
            dim=self.dim,
            grn=self.grn
        )

        self.dec_block_4 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=self.feat_size[3],
                out_channels=self.feat_size[3],
                exp_r=self.exp_r[6],
                kernel_size=self.dec_kernel_size,
                do_res=self.do_res,
                norm_type=self.norm_type,
                dim=self.dim,
                grn=self.grn
                )
            for i in range(block_counts[6])]
        )

        self.up_3 = MedNeXtUpBlock(
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[2],
            exp_r=self.exp_r[7],
            kernel_size=self.dec_kernel_size,
            do_res=self.do_res_up_down,
            norm_type=self.norm_type,
            dim=self.dim,
            grn=self.grn
        )

        self.dec_block_3 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=self.feat_size[2],
                out_channels=self.feat_size[2],
                exp_r=self.exp_r[7],
                kernel_size=self.dec_kernel_size,
                do_res=self.do_res,
                norm_type=self.norm_type,
                dim=self.dim,
                grn=self.grn
                )
            for i in range(block_counts[7])]
        )
        self.up_2 = MedNeXtUpBlock(
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[1],
            exp_r=self.exp_r[8],
            kernel_size=self.dec_kernel_size,
            do_res=self.do_res_up_down,
            norm_type=self.norm_type,
            dim=self.dim,
            grn=self.grn
        )

        self.dec_block_2 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=self.feat_size[1],
                out_channels=self.feat_size[1],
                exp_r=self.exp_r[8],
                kernel_size=self.dec_kernel_size,
                do_res=self.do_res,
                norm_type=self.norm_type,
                dim=self.dim,
                grn=self.grn
                )
            for i in range(block_counts[8])]
        )
        self.up_1 = MedNeXtUpBlock(
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[0],
            exp_r=self.exp_r[9],
            kernel_size=self.dec_kernel_size,
            do_res=self.do_res_up_down,
            norm_type=self.norm_type,
            dim=self.dim,
            grn=self.grn
        )

        self.dec_block_1 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=self.feat_size[0],
                out_channels=self.feat_size[0],
                exp_r=self.exp_r[9],
                kernel_size=self.dec_kernel_size,
                do_res=self.do_res,
                norm_type=self.norm_type,
                dim=self.dim,
                grn=self.grn
                )
            for i in range(block_counts[9])]
        )

        self.up_0 = MedNeXtUpBlock(
            in_channels=self.feat_size[0],
            out_channels=self.in_chans,
            exp_r=self.exp_r[10],
            kernel_size=self.dec_kernel_size,
            do_res=self.do_res_up_down,
            norm_type=self.norm_type,
            dim=self.dim,
            grn=self.grn
        )

        self.dec_block_0 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=self.in_chans,
                out_channels=self.in_chans,
                exp_r=self.exp_r[10],
                kernel_size=self.dec_kernel_size,
                do_res=self.do_res,
                norm_type=self.norm_type,
                dim=self.dim,
                grn=self.grn
                )
            for i in range(block_counts[10])]
        )

        self.out_0 = OutBlock(in_channels=self.in_chans, n_classes=self.in_chans, dim=self.dim)


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
    
    def global_avg_pool_and_sigmoid(self,k):
        pooled_tensor = k.mean(dim=(2, 3, 4), keepdim=True)
        sigmoid_tensor = torch.sigmoid(pooled_tensor)
        return sigmoid_tensor*k
    def forward(self, x_in):
        # x_input = self.stem(x_in)
        
        outs = self.vit(x_in)
        # print(x_in.shape)
        enc1 = self.enc_block_0(x_in) 
        f_enc1 = self.down_0(enc1) # 48xD/2xH/2xW/2

        x2 = outs[0]    # 48xD/2xH/2xW/2
        enc2 = self.enc_block_1(x2) 
        f_enc2 = self.down_1(enc2) # 96xD/4xH/4xW/4

        x3 = outs[1]    # 96xD/4xH/4xW/4
        enc3 = self.enc_block_2(x3) 
        f_enc3 = self.down_2(enc3) # 192xD/8xH/8xW/8

        x4 = outs[2]    # 192xD/8xH/8xW/8
        enc4 = self.enc_block_3(x4) 
        f_enc4 = self.down_3(enc4) # 384xD/16xH/16xW/16

        x5 = outs[3] # 384xD/16xH/16xW/16
        enc5 = self.enc_block_4(x5)
        f_enc5 = self.down_4(enc5) # 768xD/32xH/32xW/32
        enc_hidden = self.bottleneck(f_enc5) # 768xD/32xH/32xW/32

        x_up_4 = self.up_4(enc_hidden) # 384xD/16xH/16xW/16
        dec_x = enc5 + x_up_4 # 384xD/16xH/16xW/16
        x = self.dec_block_4(dec_x) # 384xD/16xH/16xW/16

        x_up_3 = self.up_3(x) # 192xD/8xH/8xW/8
        dec_x = enc4 + x_up_3 # 192xD/8xH/8xW/8
        x = self.dec_block_3(dec_x) # 192xD/8xH/8xW/8

        x_up_2 = self.up_2(x) # 96xD/4xH/4xW/4
        dec_x = enc3 + x_up_2  # 96xD/4xH/4xW/4
        x = self.dec_block_2(dec_x) # 96xD/4xH/4xW/4

        x_up_1 = self.up_1(x) # 48xD/2xH/2xW/2
        dec_x = enc2 + x_up_1  # 48xD/2xH/2xW/2
        x = self.dec_block_1(dec_x) # 48xD/2xH/2xW/2

        x_up_0 = self.up_0(x) # 1xDxHxW
        # dec_x = enc1 + x_up_0 # 1xDxHxW
        # x = self.dec_block_0(dec_x) # 1xDxHxW

        # x = self.out_0(x)
                
        return x_up_0
    
if __name__ == '__main__':
    model = SegMamba().to("cuda")
    input = torch.ones(1,1,128,128,128).to("cuda")
    
    k = model(input)
    