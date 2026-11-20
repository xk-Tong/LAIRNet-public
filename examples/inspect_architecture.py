"""运行 LAIRNet 网络结构检查：输出各模块参数量、张量维度与可逆重构误差。"""

import argparse

import torch

from lairnet import LAIRNet, dwt, iwt


def main():
    """作用：解析 --device/--small 参数，打印各子模块参数量、张量 shape 与可逆主干重构误差。

    返回：None；命令行入口。--small 使用轻量级配置以便在 CPU 上快速执行验证。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--small", action="store_true", help="使用轻量级配置快速检查。")
    args = parser.parse_args()
    torch.manual_seed(17)
    torch.set_num_threads(1)
    options = dict(blocks=2, growth=4, pre_width=8, pre_growth=4, pre_units=2,
                   mamba_embed_dim=30, mamba_depths=(1,), mamba_state=4) if args.small else {}
    model = LAIRNet(**options).to(args.device).eval()
    host = torch.rand(1, 3, 8, 10, device=args.device)
    watermark = torch.rand_like(host)
    # eval 切换模块行为；inference_mode 关闭 autograd 记录，两者作用不同。
    with torch.inference_mode():
        raw_container, latent = model.embed(host, watermark, quantization="none")
        inverse = model.inn(torch.cat((dwt(raw_container), latent), dim=1), reverse=True)
        exact_watermark = iwt(inverse[:, 12:])  # (1,3,8,10)，保留 latent 的可逆映射验证。
        blind_output = model.extract(raw_container)  # (1,3,8,10)，采样独立标准高斯潜变量。
    print("Configuration:", "lightweight (2 blocks)" if args.small else "default (16 blocks)")
    print("PyTorch:", torch.__version__, "device:", args.device)
    for name, module in (("inn", model.inn), ("pre_enhance", model.pre_enhance), ("post_restore", model.post_restore)):
        print(f"{name} parameters: {sum(p.numel() for p in module.parameters()):,}")
    print("container:", tuple(raw_container.shape), "latent:", tuple(latent.shape))
    print("blind extraction:", tuple(blind_output.shape), "finite:", bool(torch.isfinite(blind_output).all()))
    print("Core-only, unquantized max reconstruction error:", float((exact_watermark-watermark).abs().max()))


if __name__ == "__main__":
    main()
