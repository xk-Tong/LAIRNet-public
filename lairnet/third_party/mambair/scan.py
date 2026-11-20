"""实数域选择性状态空间扫描（Selective Scan）递推实现与后端自动分发。

状态更新方程：h_t = exp(delta_t * A) * h_(t-1) + delta_t * B_t * u_t；
输出方程：y_t = sum(C_t * h_t) + D * u_t。支持 CUDA 加速算子与纯 PyTorch 备选实现。
"""

import warnings

import torch
from torch import Tensor
from torch.nn import functional as F

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _cuda_scan
except (ImportError, OSError):
    _cuda_scan = None

_warned_reference = False


def selective_scan_reference(
    u: Tensor, delta: Tensor, A: Tensor, B: Tensor, C: Tensor,
    D: Tensor | None = None, z: Tensor | None = None,
    delta_bias: Tensor | None = None, delta_softplus: bool = False,
    return_last_state: bool = False,
):
    """作用：基于纯 PyTorch 的显式 SSM 序列递推实现，支持在 CPU 或未安装 mamba-ssm 的环境中运行。

    参数：u/delta=(batch,channels,length)，A=(channels,state)，
    B/C=(batch,groups,state,length)，D/delta_bias=(channels,)，z 同 u。
    仅支持实数，groups 整除 channels；内部用 float32，返回 dtype 与 u 相同。
    返回：y 同 u；return_last_state=True 时返回 (y,h_last)。用于 MambaIR SS2D。
    """
    if u.ndim != 3 or delta.shape != u.shape or u.shape[-1] < 1:
        raise ValueError("u and delta must have shape (batch,channels,length), length>0.")
    batch, channels, length = u.shape
    if A.ndim != 2 or A.shape[0] != channels or A.is_complex():
        raise ValueError("A must be real with shape (channels,state).")
    if B.ndim != 4 or C.shape != B.shape or B.shape[0] != batch or B.shape[2:] != (A.shape[1], length):
        raise ValueError("B and C must have shape (batch,groups,state,length).")
    groups = B.shape[1]
    if groups < 1 or channels % groups:
        raise ValueError("groups must divide channels.")
    input_dtype = u.dtype
    u, delta, A, B, C = u.float(), delta.float(), A.float(), B.float(), C.float()
    if delta_bias is not None:
        delta = delta+delta_bias.float()[None, :, None]  # (batch,channels,length)
    if delta_softplus:
        delta = F.softplus(delta)
    state = u.new_zeros(batch, channels, A.shape[1])
    outputs = []
    for t in range(length):
        # 按组共享 B/C；只展开当前 t，避免预先物化整个 (batch,channels,state,length)。
        b_t = B[:, :, :, t].repeat_interleave(channels//groups, dim=1)  # (batch,channels,state)
        c_t = C[:, :, :, t].repeat_interleave(channels//groups, dim=1)  # (batch,channels,state)
        dt = delta[:, :, t, None]  # (batch,channels,1)
        state = torch.exp(dt*A[None, :, :])*state + dt*b_t*u[:, :, t, None]
        outputs.append((state*c_t).sum(dim=-1))
    out = torch.stack(outputs, dim=-1)  # (batch,channels,length)
    if D is not None:
        out = out+D.float()[None, :, None]*u
    if z is not None:
        out = out*F.silu(z.float())
    out = out.to(input_dtype)
    return (out, state) if return_last_state else out


def selective_scan_fn(*args, **kwargs):
    """作用：选择后端；参数/返回值与 selective_scan_reference 相同。

    CUDA tensor 且已安装 mamba-ssm 时用其 kernel，否则运行 reference 并提示一次。
    kernel 运行错误直接传播，不静默掩盖。用于所有 SS2D 扫描。
    """
    global _warned_reference
    if args[0].is_cuda and _cuda_scan is not None:
        return _cuda_scan(*args, **kwargs)
    if not _warned_reference:
        warnings.warn("mamba-ssm CUDA kernel is unavailable on current device/environment; falling back to PyTorch selective-scan implementation.", RuntimeWarning, stacklevel=2)
        _warned_reference = True
    return selective_scan_reference(*args, **kwargs)
