import torch
from typing import Callable, Optional
from torch import nn
import torch.nn.functional as F


class RegionSpecificLoss(nn.Module):
    def __init__(self,
                 device: None,
                 apply_nonlin: Optional[Callable] = None,
                 smooth: float = 1e-5,
                 num_region_per_axis: tuple = (16, 16, 16),
                 do_bg: bool = False,
                 batch_dice: bool = True,
                 alpha: float = 0.3,
                 beta: float = 0.4,
                 gamma: float = 0.5):
        """
        改进的区域特定损失函数，专为3D医学图像分割设计

        参数:
        apply_nonlin: 应用于网络输出的非线性函数 (如sigmoid)
        smooth: 平滑因子防止除零
        num_region_per_axis: 每个轴上的区域划分数量 (D, H, W)
        do_bg: 是否计算背景损失
        batch_dice: 是否在批次维度计算Dice
        alpha: Tversky损失的α参数 (控制FP惩罚)
        beta: Tversky损失的β参数 (控制FN惩罚)
        gamma: 区域权重调节因子
        """
        super(RegionSpecificLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.do_bg = do_bg
        self.apply_nonlin = apply_nonlin
        self.batch_dice = batch_dice
        self.dim = len(num_region_per_axis)
        self.device = device
        # 验证输入维度
        assert self.dim == 3, "该损失函数专为3D数据设计"
        self.pool = nn.AdaptiveAvgPool3d(num_region_per_axis)

        # 区域权重矩阵 (关注中心区域)
        self.register_buffer('region_weights', self.create_region_weights(num_region_per_axis))

    def create_region_weights(self, region_size):
        """创建区域权重矩阵，中心区域权重更高"""
        d, h, w = region_size
        weights = torch.ones(1, 1, d, h, w)

        # 创建中心区域掩码
        d_center = d // 4
        h_center = h // 4
        w_center = w // 4

        center_mask = torch.zeros_like(weights)
        center_mask[:, :, d // 2 - d_center:d // 2 + d_center,
        h // 2 - h_center:h // 2 + h_center,
        w // 2 - w_center:w // 2 + w_center] = 1.0

        # 组合权重: 中心区域权重=1.5, 边缘区域权重=0.8
        weights = weights * 0.8 + center_mask * 0.7
        return weights

    def forward(self, x: torch.Tensor, y: torch.Tensor, loss_mask: Optional[torch.Tensor] = None):
        """
        前向传播计算损失

        参数:
        x: 网络预测 (B, C, D, H, W) 值在[0,1]范围内
        y: 真实标签 (B, 1, D, H, W) 值0或1
        loss_mask: 可选损失掩码 (B, 1, D, H, W)
        """
        shp_x, shp_y = x.shape, y.shape

        # 验证输入维度
        assert shp_x == shp_y, f"预测和标签形状不匹配: {shp_x} vs {shp_y}"
        assert self.dim == (len(shp_x) - 2), "区域大小必须与数据维度匹配"

        # 应用非线性激活
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        # 获取区域统计量
        region_tp, region_fp, region_fn = self.get_region_tp_fp_fn(x, y, loss_mask)

        # 计算自适应权重
        alpha = self.alpha + self.gamma * (region_fp + self.smooth) / (region_fp + region_fn + self.smooth)
        beta = self.beta + self.gamma * (region_fn + self.smooth) / (region_fp + region_fn + self.smooth)

        # 计算区域Tversky损失
        region_tversky = (region_tp + self.smooth) / (region_tp + alpha * region_fp + beta * region_fn + self.smooth)

        # 应用区域权重
        region_tversky = region_tversky * self.region_weights.to(self.device)

        # 聚合区域损失
        if self.batch_dice:
            region_tversky = region_tversky.mean(dim=(0, 2, 3, 4))
        else:
            region_tversky = region_tversky.mean(dim=(2, 3, 4))

        # 处理背景损失
        if not self.do_bg:
            region_tversky = region_tversky[:, 1] if region_tversky.dim() > 1 else region_tversky

        # 最终损失值
        return 1 - region_tversky.mean()

    def get_region_tp_fp_fn(self, net_output: torch.Tensor, gt: torch.Tensor,
                            mask: Optional[torch.Tensor] = None):
        """
        计算每个区域的真正例(TP)、假正例(FP)、假负例(FN)
        """
        # 二值分割处理 (单通道)
        if net_output.shape[1] == 1:
            # 二值情况: 直接使用预测和标签
            tp = net_output * gt
            fp = net_output * (1 - gt)
            fn = (1 - net_output) * gt
        else:
            # 多类情况: 使用one-hot编码
            y_onehot = torch.zeros_like(net_output)
            y_onehot.scatter_(1, gt.long(), 1)

            tp = net_output * y_onehot
            fp = net_output * (1 - y_onehot)
            fn = (1 - net_output) * y_onehot

        # 应用损失掩码 (如果有)
        if mask is not None:
            tp *= mask
            fp *= mask
            fn *= mask

        # 区域特定池化
        region_tp = self.pool(tp)
        region_fp = self.pool(fp)
        region_fn = self.pool(fn)

        return region_tp, region_fp, region_fn