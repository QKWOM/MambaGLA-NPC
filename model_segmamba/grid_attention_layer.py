# -*-coding:utf-8-*-
import torch
from torch import nn
from torch.nn import functional as F
from models.networks_other import init_weights
from typing import Type
from model_segmamba.kan import KANLinear
import math

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class EfficientCon(nn.Module):
    """更高效的卷积模块，适用于医学图像"""

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(EfficientCon, self).__init__()
        self.out_channels = out_planes

        # 使用更高效的深度可分离卷积
        if kernel_size == 1:
            self.conv = nn.Conv3d(in_planes, out_planes, 1, stride=stride,
                                  padding=padding, bias=bias)
        else:
            self.conv = nn.Sequential(
                nn.Conv3d(in_planes, in_planes, kernel_size, stride=stride,
                          padding=padding, dilation=dilation, groups=in_planes, bias=bias),
                nn.Conv3d(in_planes, out_planes, 1, bias=bias)
            )

        self.bn = nn.BatchNorm3d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return self.relu(x)


class EfficientChannelAttention(nn.Module):
    """高效的通道注意力模块，参考ECA-Net思想"""

    def __init__(self, channels, gamma=2, b=1):
        super(EfficientChannelAttention, self).__init__()
        self.channels = channels
        # 自适应卷积核大小
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k_size = max(t if t % 2 else t + 1, 3)

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 平均池化分支
        y_avg = self.avg_pool(x)
        y_max = self.max_pool(x)

        # 合并并应用1D卷积
        y = y_avg + y_max
        y = y.squeeze(-1).squeeze(-1).transpose(-1, -2)  # [B, C, 1] -> [B, 1, C]
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1).unsqueeze(-1)

        return self.sigmoid(y) * x


class EnhancedSpatialAttention(nn.Module):
    """增强的空间注意力模块，包含多尺度统计信息"""

    def __init__(self, in_channels, reduction=4):
        super(EnhancedSpatialAttention, self).__init__()
        # 多尺度特征：均值、最大值、标准差
        self.conv = nn.Sequential(
            EfficientCon(3, in_channels // reduction, 3, padding=1),
            nn.ReLU(inplace=False),
            EfficientCon(in_channels // reduction, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 计算多尺度统计特征
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        std_out = torch.std(x, dim=1, keepdim=True)

        # 拼接多尺度特征
        spatial_feat = torch.cat([avg_out, max_out, std_out], dim=1)
        return self.conv(spatial_feat) * x

class BasicCon(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicCon, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm3d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
###################################################### EM-Net

class FFParser_n(nn.Module):
    def __init__(self, dim, h=128, w=239, d=65):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(dim, h, w, d, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=None):
        B, C, H, W, D = x.shape
        assert H == W, "height and width are not equal"
        if spatial_size is None:
            a = b = H
        else:
            a, b = spatial_size

        # x = x.view(B, a, b, C)
        x = x.to(torch.float32)
        x = torch.fft.rfftn(x, dim=(2, 3, 4), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfftn(x, s=(H, W, D), dim=(2, 3, 4), norm='ortho')

        x = x.reshape(B, C, H, W, D)

        return x

class Spectral_Layer(nn.Module):
    def __init__(self, dim, stage=1, in_shape=[128, 128, 128]):
        super().__init__()
        self.dim = dim

        self.h = in_shape[0] // 2**(stage+1)
        self.w = in_shape[1] // 2**(stage+1)
        self.d = in_shape[2] // 2**(stage+2) + 1

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MlpChannel(hidden_size=dim, mlp_dim=dim//2)
        self.ffp_module = FFParser_n(dim, h=self.h, w=self.w, d=self.d)

    def forward(self, x):
        B, C = x.shape[:2]
        # B, C, DIM1, DIM2, DIM3
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        # print(x.shape,'shape')

        x_reshape = x.reshape(B, C, n_tokens).transpose(-1, -2)
        norm1_x = self.norm1(x_reshape)
        norm1_x = norm1_x.reshape(B, C, *img_dims)
        x_fft = self.ffp_module(norm1_x)
        # print(x_fft.shape, 'xfft')
        norm2_x_fft = self.norm2(x_fft.reshape(B, C, n_tokens).transpose(-1, -2))
        x_spatial = self.mlp(norm2_x_fft.transpose(-1, -2).reshape(B, C, *img_dims))
        out_all = x + x_spatial
        new_out = out_all.transpose(-1, -2).reshape(B, C, *img_dims)
        return new_out
####################################################   ##

class _GridAttentionBlockND(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels=None, dimension=3, mode='concatenation',
                 sub_sample_factor=(2,2,2)):
        super(_GridAttentionBlockND, self).__init__()

        assert dimension in [2, 3]
        assert mode in ['concatenation', 'concatenation_debug', 'concatenation_residual', 'grid_and_channel', 'enhanced_grid_and_channel']

        # Downsampling rate for the input featuremap
        if isinstance(sub_sample_factor, tuple): self.sub_sample_factor = sub_sample_factor
        elif isinstance(sub_sample_factor, list): self.sub_sample_factor = tuple(sub_sample_factor)
        else: self.sub_sample_factor = tuple([sub_sample_factor]) * dimension

        if dimension == 3:
            conv_nd = nn.Conv3d
            bn = nn.BatchNorm3d
            self.upsample_mode = 'trilinear'

        # Default parameter set
        self.mode = mode
        self.dimension = dimension
        self.sub_sample_kernel_size = self.sub_sample_factor

        # Number of channels (pixel dimensions)
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels
        
        self.mlp_ratio = 16

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1


        self.mlp1 = nn.Sequential(
            Flatten(),
            nn.Linear(in_channels, in_channels // self.mlp_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // self.mlp_ratio, in_channels),
            )

        self.spatial = BasicCon(2, 1, 3, stride=1, padding=1, relu=False)


        self.phi = conv_nd(in_channels=self.gating_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0, bias=True)
 

        # Initialise weights
        # for m in self.children():
        #     init_weights(m, init_type='kaiming')

        # Define the operation
        if mode == 'grid_and_channel':
            self.operation_function = self.grid_and_channel
        elif mode == 'concatenation':
            self.operation_function = self._concatenation
        elif mode == 'enhanced_grid_and_channel':
            self.operation_function = self.enhanced_grid_and_channel

        # 高效通道注意力
        self.channel_attention = EfficientChannelAttention(in_channels)

        # 增强空间注意力
        self.spatial_attention = EnhancedSpatialAttention(in_channels)

        # 智能融合权重
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, g):
        '''
        :param x: (b, c, t, h, w)
        :param g: (b, g_d)
        :return:
        '''

        output = self.operation_function(x, g)
        return output

    def enhanced_grid_and_channel(self, x, g):
        """改进的网格和通道注意力"""
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # 更精确的上采样
        gating = F.interpolate(self.phi(g), size=x.size()[2:],
                               mode=self.upsample_mode, align_corners=False)

        # 残差连接 + 激活
        fused_features = F.relu(x + gating, inplace=True)

        # 顺序应用注意力机制 (CBAM风格)
        # 1. 先通道注意力
        channel_refined = self.channel_attention(fused_features)

        # 2. 后空间注意力
        spatial_refined = self.spatial_attention(channel_refined)

        # 3. 门控残差连接
        output = self.fusion_weight * spatial_refined + (1 - self.fusion_weight) * x

        return output

    def grid_and_channel(self, x, g):
        input_size = x.size()
        batch_size = input_size[0] 
        assert batch_size == g.size(0)
        gating = F.upsample(self.phi(g), size=x.size()[2:], mode=self.upsample_mode)
        fix = F.relu(x+gating, inplace=True)
        
        B, C, D, H, W = fix.shape
        # max_pool_c = nn.MaxPool3d(kernel_size=(D,H,W), stride=(D,H,W))
        max_out_c = F.max_pool3d(fix, kernel_size=(D, H, W), stride=(D, H, W))
        # avg_pool_c = nn.AvgPool3d(kernel_size=(D,H,W), stride=(D,H,W))
        avg_out_c = F.avg_pool3d(fix, kernel_size=(D, H, W), stride=(D, H, W))

        # channel_plot = self.mlp1(max_out_c+avg_out_c)

        channel_plot = self.mlp1(max_out_c) + self.mlp1(avg_out_c)

        channel_plot = channel_plot.view(B,C,1,1,1)

        # print(channel_plot.shape)
        
        spatial_plot = torch.cat( (torch.max(fix,1)[0].unsqueeze(1), torch.mean(fix,1).unsqueeze(1)), dim=1 )
        spatial_plot = self.spatial(spatial_plot)
        # print(spatial_plot.shape)

        c_g = torch.sigmoid(channel_plot)
        s_g = torch.sigmoid(spatial_plot)
        
        out = c_g*fix+s_g*fix
        # out = c_g*fix
        return out

    def _concatenation(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
        f = F.relu(theta_x + phi_g, inplace=True)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = F.sigmoid(self.psi(f))

        # upsample the attentions and multiply
        sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y

    def _concatenation_debug(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
        f = F.softplus(theta_x + phi_g)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = F.sigmoid(self.psi(f))

        # upsample the attentions and multiply
        sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f


    def _concatenation_residual(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
        f = F.relu(theta_x + phi_g, inplace=True)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        f = self.psi(f).view(batch_size, 1, -1)
        sigm_psi_f = F.softmax(f, dim=2).view(batch_size, 1, *theta_x.size()[2:])

        # upsample the attentions and multiply
        sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f





class GridAttentionBlock3D(_GridAttentionBlockND):
    def __init__(self, in_channels, gating_channels, inter_channels=None, mode='concatenation',
                 sub_sample_factor=(2,2,2)):
        super(GridAttentionBlock3D, self).__init__(in_channels,
                                                   inter_channels=inter_channels,
                                                   gating_channels=gating_channels,
                                                   dimension=3, mode=mode,
                                                   sub_sample_factor=sub_sample_factor,
                                                   )

