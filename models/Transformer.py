import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Transformer(nn.Module):
    """经典 Transformer 编码器，处理 3D 特征图"""

    def __init__(self, dim, depth, num_heads=8, mlp_ratio=4.,
                 drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.):
        super().__init__()
        self.dim = dim
        self.depth = depth

        # Transformer 层
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                # drop_path=drop_path_rate * (i / depth))
                drop_path=0)
            for i in range(depth)])

        # 归一化层
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (B, C, D, H, W)
        B, C, D, H, W = x.shape

        # 将 3D 特征图转换为序列 (B, D*H*W, C)
        x = x.permute(0, 2, 3, 4, 1).reshape(B, D * H * W, C)

        # 通过 Transformer 层
        for layer in self.layers:
            x = layer(x)

        # 归一化
        x = self.norm(x)

        # 将序列转换回 3D 特征图 (B, C, D, H, W)
        x = x.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3)
        return x


class TransformerBlock(nn.Module):
    """经典 Transformer 块"""

    def __init__(self, dim, num_heads, mlp_ratio=4.,
                 drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        # 自注意力层
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=True
        )

        # 前馈网络
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

        # DropPath
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # 自注意力
        attn_output, _ = self.attn(
            query=self.norm1(x),
            key=self.norm1(x),
            value=self.norm1(x)
        )
        x = x + self.drop_path(attn_output)

        # 前馈网络
        mlp_output = self.mlp(self.norm2(x))
        x = x + self.drop_path(mlp_output)
        return x


class DropPath(nn.Module):
    """随机深度路径（用于正则化）"""

    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor