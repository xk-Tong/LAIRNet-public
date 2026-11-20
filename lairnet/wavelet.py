"""2D Haar 小波正逆变换算子。B/C/H/W 分别表示 batch/channel/height/width。"""

import torch
from torch import Tensor


def check_image(x: Tensor, channels: int | None = None, even: bool = False) -> None:
    """作用：校验图像张量接口；参数 x 为浮点 (B,C,H,W)，channels/even 指定通道与偶数尺寸约束。

    返回：None；不满足约束时抛出 TypeError/ValueError。
    """
    if not isinstance(x, Tensor) or not x.is_floating_point():
        raise TypeError("Expected a floating-point torch.Tensor.")
    if x.ndim != 4 or min(x.shape) <= 0:
        raise ValueError("Expected nonempty (B,C,H,W).")
    if channels is not None and x.shape[1] != channels:
        raise ValueError(f"Expected {channels} channels; got {x.shape[1]}.")
    if even and (x.shape[2] % 2 or x.shape[3] % 2):
        raise ValueError("Haar DWT requires even height and width.")


def dwt(x: Tensor) -> Tensor:
    """作用：2D Haar DWT；参数 x 为 (B,C,H,W) float，H/W 必须为偶数。

    返回：(B,4C,H/2,W/2)，子带顺序为 LL/HL/LH/HH；保留 dtype/device。用于嵌入和提取。
    """
    check_image(x, even=True)
    a = x[:, :, 0::2, 0::2] / 2  # (B,C,H/2,W/2)
    b = x[:, :, 1::2, 0::2] / 2  # (B,C,H/2,W/2)
    c = x[:, :, 0::2, 1::2] / 2  # (B,C,H/2,W/2)
    d = x[:, :, 1::2, 1::2] / 2  # (B,C,H/2,W/2)
    return torch.cat((a+b+c+d, -a-b+c+d, -a+b-c+d, a-b-c+d), dim=1)  # (B,4C,H/2,W/2)


def iwt(x: Tensor) -> Tensor:
    """作用：逆 2D Haar DWT；参数 x 为 (B,4C,h,w) float，子带顺序与 dwt 一致。

    返回：(B,C,2h,2w)，保持 dtype/device；用于重构含水印图像和恢复水印图像。
    """
    check_image(x)
    if x.shape[1] % 4:
        raise ValueError("IWT channel count must be divisible by four.")
    ll, hl, lh, hh = x.chunk(4, dim=1)  # 四个 (B,C,h,w)
    out = x.new_empty(x.shape[0], x.shape[1] // 4, 2*x.shape[2], 2*x.shape[3])
    out[:, :, 0::2, 0::2] = (ll-hl-lh+hh) / 2
    out[:, :, 1::2, 0::2] = (ll-hl+lh-hh) / 2
    out[:, :, 0::2, 1::2] = (ll+hl-lh-hh) / 2
    out[:, :, 1::2, 1::2] = (ll+hl+lh+hh) / 2
    return out
