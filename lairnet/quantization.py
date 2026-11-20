"""前向 8-bit 硬量化与 Sigmoid 软量化算子。"""

import math

import torch
from torch import Tensor


def quantize_8bit(x: Tensor, mode: str = "hard", sharpness: float = 100.0) -> Tensor:
    """作用：将浮点图像映射到 8-bit 网格；x 为 float [0,1]，保持 shape/device。

    参数 mode：none 原样返回；hard 采用 half-up 四舍五入；soft 采用 Sigmoid 软量化近似。
    sharpness 为 soft 模式的正有限锐度系数 k，默认 100。返回浮点 [0,1]（none 保留原值）。
    使用场景：宿主分支 IWT 重构后的图像量化。
    """
    if mode == "none":
        return x
    if mode not in ("hard", "soft"):
        raise ValueError("mode must be 'none', 'hard' or 'soft'.")
    if not math.isfinite(sharpness) or sharpness <= 0:
        raise ValueError("sharpness must be positive and finite.")
    scaled = x.clamp(0, 1)*255
    if mode == "hard":
        return torch.floor(scaled+0.5)/255
    base = torch.floor(scaled)
    # Q_soft(x,k)=floor(x)+sigmoid(k*(x-floor(x)-0.5))，在 0..255 域计算。
    return ((base+torch.sigmoid(sharpness*(scaled-base-0.5)))/255).clamp(0, 1)
