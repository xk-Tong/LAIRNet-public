"""提取端图像域 Dense2Net 多尺度密集连接前增强网络。"""

import torch
from torch import Tensor, nn

from .wavelet import check_image


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    """作用：交错通道；x 为 (B,C,H,W)，groups 整除 C；返回同 shape。

    使用场景：Dense2Net 多尺度分支拆分前，让不同分组交换特征。
    """
    b, c, h, w = x.shape
    if groups < 1 or c % groups:
        raise ValueError("groups must be positive and divide channels.")
    grouped = x.reshape(b, groups, c//groups, h, w)  # (B,groups,C/groups,H,W)
    return grouped.transpose(1, 2).reshape(b, c, h, w)  # (B,C,H,W)


class Dense2Unit(nn.Module):
    """通道压缩→shuffle→分组旁路/递进卷积→融合；输入经 dense concatenation 聚合。"""

    def __init__(self, in_channels: int, growth: int = 32, scale: int = 4, expansion: int = 4, compression: int = 4):
        """作用：构建单个多尺度密集特征单元；in_channels 为密集输入，growth 为新增通道。

        scale>=2 为分组数，expansion 为瓶颈扩张倍数，compression 为处理分支压缩率。
        growth*expansion 须整除 scale，且每组通道须整除 compression。
        返回 Module；用于前增强网络特征提取。
        """
        super().__init__()
        width = growth*expansion
        if min(in_channels, growth, expansion, compression) < 1 or scale < 2 or width % scale or (width//scale) % compression:
            raise ValueError("Require positive dimensions, scale>=2 and divisible bottleneck width.")
        self.scale = scale
        group_width = width//scale
        branch_width = group_width//compression
        self.reduce = nn.Conv2d(in_channels, width, 1)
        self.branches = nn.ModuleList([
            nn.Conv2d(group_width if i == 0 else group_width+branch_width,
                      branch_width, 3, padding=1) for i in range(scale-1)
        ])
        self.fuse = nn.Conv2d(group_width+(scale-1)*branch_width, growth, 1)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x: Tensor) -> Tensor:
        """作用：递进多尺度特征提取；x 为 (B,C,H,W) float；返回 (B,growth,H,W)。

        使用场景：每个 dense 单元；第一组直通，第三组起拼接上一压缩分支。
        """
        features = channel_shuffle(self.act(self.reduce(x)), self.scale)
        parts = features.chunk(self.scale, dim=1)  # scale 个 (B,width/scale,H,W)
        outputs = [parts[0]]
        for i, conv in enumerate(self.branches, start=1):
            # 压缩后通道数不同，递进连接采用 concat 而非逐元素相加。
            branch = parts[i] if i == 1 else torch.cat((parts[i], outputs[-1]), dim=1)
            # branch: (B,group_width,H,W) 或 (B,group_width+branch_width,H,W)
            outputs.append(self.act(conv(branch)))
        return self.fuse(torch.cat(outputs, dim=1))  # (B,growth,H,W)


class Dense2NetEnhancer(nn.Module):
    """保持 RGB 空间尺寸的 Dense2Net 前增强器。"""

    def __init__(self, width: int = 64, growth: int = 32, units: int = 3, scale: int = 4, compression: int = 4):
        """作用：构建前增强网络；width/growth/units 为正整数，scale/compression 控制分组与压缩率。

        返回 Module；用于提取端 DWT 变换之前的图像特征增强。
        """
        super().__init__()
        if min(width, growth, units) < 1:
            raise ValueError("width, growth and units must be positive.")
        self.stem = nn.Conv2d(3, width, 3, padding=1)
        self.units = nn.ModuleList([Dense2Unit(width+i*growth, growth, scale, compression=compression) for i in range(units)])
        self.fuse = nn.Conv2d(width+units*growth, width, 1)
        self.output = nn.Conv2d(width, 3, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x: Tensor) -> Tensor:
        """作用：抑制含水印图像失真；x 为 (B,3,H,W) float [0,1]；返回同 shape 浮点特征。

        使用场景：逆向 INN 之前，采用全局残差连接与零初始化输出层。
        """
        check_image(x, 3)
        features = [self.stem(x)]
        for unit in self.units:
            joined = torch.cat(features, dim=1)  # (B,width+i*growth,H,W)
            features.append(unit(joined))
        joined = torch.cat(features, dim=1)  # (B,width+units*growth,H,W)
        return x + self.output(self.fuse(joined))
