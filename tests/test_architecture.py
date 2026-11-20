"""LAIRNet 网络结构与数值可逆性单元测试。"""

import unittest

import torch
from torch import nn

from lairnet import LAIRNet, Invertible1x1Conv, InvertibleCore, dwt, iwt, quantize_8bit
from lairnet.enhancement import Dense2Unit
from lairnet.third_party.mambair.scan import selective_scan_reference


class ArchitectureTests(unittest.TestCase):
    """验证频域约定、非平凡映射的可逆性、SSM 递推和完整网络调用链。"""

    def setUp(self):
        """作用：固定随机种子和 CPU 线程数；返回 None；在每个测试用例执行前调用。"""
        torch.manual_seed(17)
        torch.set_num_threads(1)

    def test_haar_convention_and_roundtrip(self):
        """作用：用已知 2×2 图像核对子带符号，并检查非方形图像逆变换；无外部输入。"""
        x = torch.tensor([[[[1., 2.], [3., 4.]]]])
        torch.testing.assert_close(dwt(x).flatten(), torch.tensor([5., 1., 2., 0.]))
        for shape in ((2, 3, 8, 10), (1, 1, 2, 6)):
            image = torch.randn(shape, dtype=torch.float64)
            torch.testing.assert_close(iwt(dwt(image)), image, atol=1e-12, rtol=1e-12)
        with self.assertRaises(ValueError):
            dwt(torch.zeros(1, 3, 7, 8))
        with self.assertRaises(ValueError):
            iwt(torch.zeros(1, 5, 3, 4))

    def test_mixing_after_nonorthogonal_parameter_change(self):
        """作用：扰动 LU 后验证真逆，专门覆盖把 W.T 误当通用逆矩阵的错误。"""
        layer = Invertible1x1Conv(8).double()
        with torch.no_grad():
            layer.log_abs_diag.add_(0.2)
            layer.lower.add_(torch.randn_like(layer.lower)*0.03)
        weight = layer.matrix()
        self.assertFalse(torch.allclose(weight.T@weight, torch.eye(8, dtype=torch.float64)))
        x = torch.randn(2, 8, 3, 5, dtype=torch.float64)
        torch.testing.assert_close(layer(layer(x), reverse=True), x, atol=1e-11, rtol=1e-11)

    def test_nonzero_coupling_roundtrip(self):
        """作用：在非零耦合参数下验证多层堆叠 INN 的正向与逆向精确可逆性。"""
        core = InvertibleCore(blocks=3, growth=4).double()
        with torch.no_grad():
            for block in core.blocks:
                for subnet in (block.psi, block.phi):
                    nn.init.normal_(subnet.project.weight, std=0.01)
                    nn.init.normal_(subnet.project.bias, std=0.01)
                block.mix.log_abs_diag.add_(0.04)
        x = torch.randn(2, 24, 4, 5, dtype=torch.float64)
        with torch.inference_mode():
            encoded = core(x)
            recovered = core(encoded, reverse=True)
        torch.testing.assert_close(recovered, x, atol=1e-10, rtol=1e-10)

    def test_quantization(self):
        """作用：验证 8-bit 硬量化舍入、Sigmoid 软量化近似及输出值域范围。"""
        x = torch.tensor([-1., 0., 0.5/255, 1.2/255, 1.8/255, 1., 2.])
        hard = quantize_8bit(x, "hard")
        torch.testing.assert_close(hard*255, torch.tensor([0., 0., 1., 1., 2., 255., 255.]))
        soft = quantize_8bit(x, "soft", sharpness=100)
        torch.testing.assert_close(soft[[0, 1, 3, 4, 5, 6]], hard[[0, 1, 3, 4, 5, 6]], atol=1e-7, rtol=0)
        self.assertIs(quantize_8bit(x, "none"), x)
        with self.assertRaises(ValueError):
            quantize_8bit(x, "soft", sharpness=0)

    def test_dense2_compressed_branches(self):
        """作用：验证 Dense2Unit 直通分支、通道压缩分支及递进拼接输出的特征维度一致性。"""
        unit = Dense2Unit(in_channels=16, growth=8, scale=4, compression=4)
        shapes = []
        handles = [branch.register_forward_hook(lambda module, args, out: shapes.append((args[0].shape[1], out.shape[1])))
                   for branch in unit.branches]
        with torch.inference_mode():
            out = unit(torch.rand(2, 16, 5, 7))
        for handle in handles:
            handle.remove()
        self.assertEqual(shapes, [(8, 2), (10, 2), (10, 2)])
        self.assertEqual(out.shape, (2, 8, 5, 7))

    def test_selective_scan_against_closed_form(self):
        """作用：用常系数递推的几何级数闭式解验证 scan，覆盖 grouped B/C 和 D/z。"""
        u = torch.ones(1, 4, 5)
        delta = torch.full_like(u, 0.2)
        a = torch.full((4, 1), -0.5)
        b = torch.tensor([2., 3.]).view(1, 2, 1, 1).expand(1, 2, 1, 5)
        c = torch.tensor([4., 5.]).view(1, 2, 1, 1).expand(1, 2, 1, 5)
        d = torch.full((4,), 0.7)
        z = torch.zeros_like(u)+0.9
        actual, last = selective_scan_reference(u, delta, a, b, c, d, z=z, return_last_state=True)
        rho = torch.exp(torch.tensor(-0.1))
        steps = torch.arange(1, 6)
        states = 0.2*torch.tensor([2., 2., 3., 3.])[:, None]*(1-rho**steps)/(1-rho)
        expected = (states*torch.tensor([4., 4., 5., 5.])[:, None]+0.7)*torch.nn.functional.silu(torch.tensor(0.9))
        torch.testing.assert_close(actual[0], expected)
        torch.testing.assert_close(last[0, :, 0], states[:, -1])

    def test_full_pipeline_and_core_only_roundtrip(self):
        """作用：检查所有增强模块实际执行，并独立验证未量化且保留 latent 时的 INN 逆。

        使用轻量级配置的随机张量验证完整前向与逆向调用链，返回 None。
        """
        model = LAIRNet(blocks=2, growth=4, pre_width=8, pre_growth=4, pre_units=2,
                        mamba_embed_dim=30, mamba_depths=(1,), mamba_state=4).eval()
        host, secret = torch.rand(1, 3, 8, 10), torch.rand(1, 3, 8, 10)
        calls = []
        handles = [module.register_forward_hook(lambda module, args, output, name=name: calls.append(name))
                   for name, module in (("pre", model.pre_enhance), ("post", model.post_restore))]
        with torch.inference_mode():
            carrier, latent = model.embed(host, secret, quantization="none")
            inverse = model.inn(torch.cat((dwt(carrier), latent), 1), reverse=True)
            torch.testing.assert_close(iwt(inverse[:, :12]), host, atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(iwt(inverse[:, 12:]), secret, atol=2e-5, rtol=2e-5)
            recovered = model.extract(carrier)
        for handle in handles:
            handle.remove()
        self.assertEqual(calls, ["pre", "post"])
        self.assertEqual(recovered.shape, secret.shape)
        self.assertEqual(latent.shape, (1, 12, 4, 5))
        self.assertTrue(torch.isfinite(recovered).all())
        with self.assertRaises(ValueError):
            model.embed(host, secret[:, :, :6])
        with self.assertRaises(ValueError):
            model.extract(carrier, latent[:, :4])

    def test_default_architecture(self):
        """作用：验证默认 16 层可逆块、Dense2Net 前增强与 MambaIR 后恢复组成的完整模型前向与提取流程。"""
        model = LAIRNet().eval()
        self.assertEqual(len(model.inn.blocks), 16)
        with torch.inference_mode():
            container, _ = model(torch.rand(1, 3, 4, 6), torch.rand(1, 3, 4, 6))
            restored = model.extract(container)
        self.assertEqual(restored.shape, (1, 3, 4, 6))
        self.assertTrue(torch.isfinite(restored).all())


if __name__ == "__main__":
    unittest.main()
