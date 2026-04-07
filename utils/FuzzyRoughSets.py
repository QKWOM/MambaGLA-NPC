import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt as edt


class FuzzyRoughLoss(nn.Module):
    def __init__(self, alpha=1, eps=1e-6, **kwargs):
        super(FuzzyRoughLoss, self).__init__()
        self.alpha = alpha
        self.eps = eps  # 防止除以零

    @torch.no_grad()
    def distance_field(self, img: torch.Tensor) -> torch.Tensor:
        """
        计算3D距离场，支持5D输入张量 (batch, channel, depth, height, width)
        """
        # 转换到CPU numpy处理
        np_img = img.cpu().numpy()
        field = np.zeros_like(np_img, dtype=np.float32)

        # 处理每个batch和每个通道
        for b in range(np_img.shape[0]):
            for c in range(np_img.shape[1]):
                # 获取当前3D体积
                volume = np_img[b, c]

                # 创建二值掩码
                fg_mask = volume > 0.5

                if np.any(fg_mask):
                    bg_mask = ~fg_mask

                    # 计算前景和背景的距离变换
                    fg_dist = edt(fg_mask)
                    bg_dist = edt(bg_mask)

                    # 组合距离场
                    combined_dist = fg_dist + bg_dist
                    field[b, c] = 1 - np.exp(-(combined_dist ** 2) / self.alpha)
                else:
                    # 如果没有前景，设为1
                    field[b, c] = 1.0

        return torch.from_numpy(field).float()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 验证输入维度 (batch, channel, depth, height, width)
        assert pred.dim() == 5, f"Expected 5D input (got {pred.dim()}D)"
        assert pred.shape == target.shape, f"Shape mismatch: pred {pred.shape}, target {target.shape}"

        # 计算目标距离场
        target_dt = self.distance_field(target).to(pred.device)

        # 计算预测误差
        pred_error = (pred - target) ** 2

        # 应用距离场权重
        weighted_error = pred_error * target_dt

        # 计算有效元素数量（非零权重）
        valid_elements = (target_dt > self.eps).sum()

        # 防止除以零
        if valid_elements == 0:
            return torch.tensor(0.0, device=pred.device)

        # 计算加权平均损失
        loss = weighted_error.sum() / valid_elements

        return loss