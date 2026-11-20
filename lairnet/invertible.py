"""空间注意力引导的仿射耦合层与可逆 1×1 通道混合卷积。"""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .wavelet import check_image


class SpatialAttention(nn.Module):
    """沿 channel 聚合均值/最大值，用 1×1 convolution 生成空间 mask。"""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """作用：空间加权；x 为 (B,C,H,W) float；返回同 shape 特征。用于耦合子网末端。"""
        stats = torch.cat((x.mean(1, keepdim=True), x.amax(1, keepdim=True)), 1)  # (B,2,H,W)
        mask = torch.sigmoid(self.conv(stats))  # (B,1,H,W)
        return x * mask  # broadcasting：(B,C,H,W) * (B,1,H,W)


class AttentionDenseSubnet(nn.Module):
    """DenseNet 风格变换子网；每个中间层读取输入及之前各层的输出。"""

    def __init__(self, channels: int, growth: int = 32, layers: int = 4):
        """作用：构建 ψ/φ 耦合子网；参数 channels 为输入/输出通道，growth/layers 控制宽度与深度。

        返回：Module；用于每个 affine coupling 的特征变换。
        """
        super().__init__()
        if min(channels, growth, layers) < 1:
            raise ValueError("channels, growth and layers must be positive.")
        # ModuleList 注册各子层，使 .to(device) 与 state_dict 自动包含它们。
        self.layers = nn.ModuleList([
            nn.Conv2d(channels + i*growth, growth, 3, padding=1) for i in range(layers)
        ])
        self.project = nn.Conv2d(channels + layers*growth, channels, 3, padding=1)
        self.attention = SpatialAttention()
        # 零初始化末端输出层，使耦合层初始表现为恒等变换，保障深层网络数值稳定。
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, x: Tensor) -> Tensor:
        """作用：计算 ψ 或 φ；x 为 (B,C,H,W) float；返回同 shape。用于耦合正反两向。"""
        features = [x]
        for layer in self.layers:
            joined = torch.cat(features, dim=1)  # (B,C+i*growth,H,W)
            features.append(F.leaky_relu(layer(joined), negative_slope=0.01))
        joined = torch.cat(features, dim=1)  # (B,C+layers*growth,H,W)
        return self.attention(self.project(joined))


class Invertible1x1Conv(nn.Module):
    """W = P L U：固定 P，L 的对角线为 1，U 的对角线为 sign*exp(log_abs)。"""

    def __init__(self, channels: int):
        """作用：构建可逆通道混合；channels 为正整数通道数；返回 Module。

        使用场景：每个耦合层之后。LU 参数化保证参数更新过程中矩阵始终可逆。
        """
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive.")
        q, _ = torch.linalg.qr(torch.randn(channels, channels))
        p, lower, upper = torch.linalg.lu(q)
        # register_buffer 随 .to() 和 state_dict 移动/保存，但不作为可学习 Parameter。
        self.register_buffer("permutation", p)
        self.register_buffer("sign", upper.diag().sign())
        self.register_buffer("identity", torch.eye(channels))
        self.lower = nn.Parameter(lower.tril(-1))
        self.upper = nn.Parameter(upper.triu(1))
        self.log_abs_diag = nn.Parameter(upper.diag().abs().log())

    def matrix(self) -> Tensor:
        """作用：从 LU 参数构造混合矩阵 W；返回 (C,C) float。用于前向卷积和逆向线性求解。"""
        lower = self.lower.tril(-1) + self.identity
        upper = self.upper.triu(1) + torch.diag(self.sign * self.log_abs_diag.exp())
        return self.permutation @ lower @ upper

    def forward(self, x: Tensor, reverse: bool = False) -> Tensor:
        """作用：通道混合或其逆；x 为 (B,C,H,W)，reverse=True 表示求解 W x=y。

        返回同 shape；不改变空间维度。
        """
        check_image(x, channels=self.identity.shape[0])
        weight = self.matrix()
        if reverse:
            b, c, h, w = x.shape
            rhs = x.permute(1, 0, 2, 3).reshape(c, -1)  # (C,B*H*W)
            solved = torch.linalg.solve(weight, rhs)  # (C,B*H*W)
            return solved.reshape(c, b, h, w).permute(1, 0, 2, 3).contiguous()  # (B,C,H,W)
        return F.conv2d(x, weight[:, :, None, None])  # weight: (C,C,1,1)


class InvertibleBlock(nn.Module):
    """先以 watermark 条件更新 host，再用可逆 1×1 卷积混合两支。"""

    def __init__(self, branch_channels: int = 12, growth: int = 32, clamp: float = 2.0):
        """作用：构建单个可逆嵌入块；branch_channels=4*RGB，growth 为子网宽度。

        clamp>0 限制 log-scale 到 (-clamp,clamp)；返回 Module；用于正反共享映射。
        """
        super().__init__()
        if not math.isfinite(clamp) or clamp <= 0:
            raise ValueError("clamp must be positive and finite.")
        self.branch_channels = branch_channels
        self.clamp = clamp
        self.psi = AttentionDenseSubnet(branch_channels, growth)
        self.phi = AttentionDenseSubnet(branch_channels, growth)
        self.mix = Invertible1x1Conv(2*branch_channels)

    def forward(self, x: Tensor, reverse: bool = False) -> Tensor:
        """作用：可逆块前向或逆向变换；x 为 (B,2C,H,W) float，前 C 个通道为 host，后 C 个为 watermark。

        返回同 shape。reverse=True 时先逆混合，再逆 affine coupling。
        """
        check_image(x, 2*self.branch_channels)
        if reverse:
            x = self.mix(x, reverse=True)
        host, secret = x.chunk(2, dim=1)  # 两个 (B,C,H,W)
        log_scale = self.clamp * (2*torch.sigmoid(self.psi(secret)) - 1)
        shift = self.phi(secret)
        # 正向 h'=h*exp(α(ψ(s)))+φ(s)；逆向 h=(h'-φ(s))/exp(α(ψ(s)))。
        host = (host-shift)*torch.exp(-log_scale) if reverse else host*log_scale.exp()+shift
        out = torch.cat((host, secret), dim=1)  # (B,2C,H,W)
        return out if reverse else self.mix(out)


class InvertibleCore(nn.Module):
    """由多个 InvertibleBlock 串联组成的可逆主干网络（默认 16 层）。"""

    def __init__(self, blocks: int = 16, growth: int = 32, clamp: float = 2.0):
        """作用：构建 RGB 小波域 INN 主干；blocks/growth 为正整数，clamp 为缩放界。

        返回 Module；用于正向水印嵌入与逆向恢复。
        """
        super().__init__()
        if blocks < 1:
            raise ValueError("blocks must be positive.")
        self.blocks = nn.ModuleList([InvertibleBlock(12, growth, clamp) for _ in range(blocks)])

    def forward(self, x: Tensor, reverse: bool = False) -> Tensor:
        """作用：堆叠可逆映射；x 为 (B,24,h,w) float；返回同 shape。逆向按相反顺序遍历块。"""
        for block in reversed(self.blocks) if reverse else self.blocks:
            x = block(x, reverse=reverse)
        return x
