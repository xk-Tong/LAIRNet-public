"""轻量级基础网络层辅助定义（免除对 timm 库的外部依赖）。"""

import torch
from torch import Tensor, nn
from torch.nn.init import trunc_normal_


def to_2tuple(value):
    """作用：将整数尺寸转成二元组；参数为 int/tuple；返回 (H,W)。用于 patch 元信息。"""
    return value if isinstance(value, tuple) else (value, value)


class DropPath(nn.Module):
    """随机深度（Stochastic Depth）正则化层；推理阶段（eval 模式）表现为恒等映射。"""

    def __init__(self, drop_prob: float = 0.0):
        """作用：构建随机深度层；drop_prob 为 [0,1) 丢弃概率；返回 Module。"""
        super().__init__()
        if not 0 <= drop_prob < 1:
            raise ValueError("drop_prob must be in [0,1).")
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        """作用：按样本丢弃 residual 分支；x 为 (B,...) float；返回同 shape。用于 RSSG。"""
        if not self.training or self.drop_prob == 0:
            return x
        keep = 1-self.drop_prob
        mask_shape = (x.shape[0],) + (1,)*(x.ndim-1)
        mask = x.new_empty(mask_shape).bernoulli_(keep)  # (B,1,...)，沿其余维广播。
        return x*mask/keep
