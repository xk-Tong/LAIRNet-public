"""LAIRNet 顶层网络定义：串联水印嵌入与盲提取完整流程。"""

import torch
from torch import Tensor, nn

from .enhancement import Dense2NetEnhancer
from .invertible import InvertibleCore
from .quantization import quantize_8bit
from .wavelet import check_image, dwt, iwt


class LAIRNet(nn.Module):
    """RGB host+watermark→INN→container；Dense2Net→inverse INN→MambaIR→watermark。"""

    def __init__(
        self, blocks: int = 16, growth: int = 32, clamp: float = 2.0,
        pre_width: int = 64, pre_growth: int = 32, pre_units: int = 3, pre_scale: int = 4,
        pre_compression: int = 4,
        mamba_embed_dim: int = 96, mamba_depths: tuple[int, ...] = (6, 6, 6, 6),
        mamba_state: int = 16,
    ):
        """作用：初始化完整 LAIRNet 架构；各参数均为网络结构超参数。

        blocks=16 为可逆块数量；growth/clamp 控制耦合子网宽度与缩放限幅；
        pre_* 配置 Dense2Net 前增强网络；mamba_* 配置 MambaIR 后恢复网络（embed_dim>=30，depths/state>0）。
        返回 Module；完整网络使用 float32，输入张量需在同一 device。
        """
        super().__init__()
        if mamba_embed_dim < 30 or not mamba_depths or min(mamba_depths) < 1 or mamba_state < 1:
            raise ValueError("Require mamba_embed_dim>=30, nonempty positive depths and positive state size.")
        # Lazy import：仅查看 INN 时不必导入 einops / MambaIR。
        from .third_party.mambair import MambaIR
        self.inn = InvertibleCore(blocks, growth, clamp)
        self.pre_enhance = Dense2NetEnhancer(pre_width, pre_growth, pre_units, pre_scale, pre_compression)
        self.post_restore = MambaIR(
            in_chans=3, embed_dim=mamba_embed_dim, depths=mamba_depths, d_state=mamba_state,
            upscale=1, upsampler="", img_range=1.0, drop_path_rate=0.0, use_checkpoint=False,
        )

    def embed(
        self, host: Tensor, watermark: Tensor, quantization: str = "hard", sharpness: float = 100.0,
    ) -> tuple[Tensor, Tensor]:
        """作用：水印嵌入；host/watermark 为等 shape (B,3,H,W) float32 [0,1]，H/W 为偶数。

        参数 quantization=hard/soft/none，sharpness 为 soft 模式的锐度系数 k。
        返回：(container (B,3,H,W), latent (B,12,H/2,W/2))。
        使用场景：正向嵌入推理；盲提取阶段无需传入该 latent，而是采样独立标准高斯潜变量。
        """
        check_image(host, 3, even=True)
        check_image(watermark, 3, even=True)
        if host.shape != watermark.shape or host.dtype != watermark.dtype or host.device != watermark.device:
            raise ValueError("host and watermark must share shape, dtype and device.")
        features = torch.cat((dwt(host), dwt(watermark)), dim=1)  # (B,24,H/2,W/2)
        encoded = self.inn(features)
        carrier, latent = encoded.chunk(2, dim=1)  # 两个 (B,12,H/2,W/2)
        return quantize_8bit(iwt(carrier), quantization, sharpness), latent

    def extract(self, container: Tensor, latent: Tensor | None = None) -> Tensor:
        """作用：水印盲提取；container 为 (B,3,H,W) float32，H/W 为偶数。

        latent 可为同 device/dtype 的 (B,12,H/2,W/2)；省略时默认采样标准正态分布 N(0,1)。
        返回 (B,3,H,W) float32 恢复水印张量。
        使用场景：从含水印图像中盲提取并恢复水印图像。
        """
        check_image(container, 3, even=True)
        expected_shape = (container.shape[0], 12, container.shape[2]//2, container.shape[3]//2)
        if latent is not None and (latent.shape != expected_shape or latent.dtype != container.dtype or latent.device != container.device):
            raise ValueError("latent must match the carrier wavelet shape, dtype and device.")
        carrier = dwt(self.pre_enhance(container))
        if latent is None:
            latent = torch.randn_like(carrier)
        decoded = self.inn(torch.cat((carrier, latent), dim=1), reverse=True)  # (B,24,H/2,W/2)
        return self.post_restore(iwt(decoded[:, 12:]))  # (B,3,H,W)

    def forward(self, host: Tensor, watermark: Tensor) -> tuple[Tensor, Tensor]:
        """作用：embed 的默认 hard-quantization 接口；参数为两张等 shape RGB 浮点图。

        返回 (container, latent)；model(...) 会执行 PyTorch hooks，推荐用于正向调用。
        """
        return self.embed(host, watermark)
