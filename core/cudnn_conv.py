"""General fp16-accumulate convolutions through cuDNN graphs.

PyTorch always asks cuDNN for fp32 accumulation; on GeForce cards that runs the tensor cores at half rate
(RTX 5060 Ti: ~47 vs ~80 TFLOPS). Here every fp16 Conv2d/Conv3d (groups 1, dilation 1) runs as a plain cuDNN
conv_fprop graph with compute_data_type=HALF, NHWC / NDHWC in and out.

General = any input size, no per-size compilation or tuning: plans that need runtime compilation (NVRTC) are
deselected, the first precompiled plan in cuDNN's heuristic order is used (no timing), ~ms per new geometry.
No fused epilogue (those only exist as runtime-compiled kernels on sm_120): bias is added by PyTorch.
Engines 0 (reference, very slow) and 8 (crashes on sm_120) are skipped.
FS_CONV_VERIFY=1 checks every new plan against an fp32 reference (debug).
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import cudnn
except ImportError:  # optional: pip install nvidia-cudnn-frontend
    cudnn = None

VERIFY = os.environ.get("FS_CONV_VERIFY") == "1"
_handle = None
_plans = {}
_ws = [None]
stats = {"cudnn": 0, "torch": 0, "built": 0}


def _get_handle():
    global _handle
    if _handle is None:
        _handle = cudnn.create_handle()
    return _handle


def _out_size(x, w, padding, stride):
    return [(x.shape[2 + i] + 2 * padding[i] - w.shape[2 + i]) // stride[i] + 1 for i in range(w.dim() - 2)]


def _cl_stride(shape):
    """channels-last strides for [N, C, *spatial]."""
    n, c, *sp = shape
    st = [0] * len(shape)
    acc = c
    st[1] = 1
    for i in range(len(sp) - 1, -1, -1):
        st[2 + i] = acc
        acc *= sp[i]
    st[0] = acc
    return st


def _build(x, w, padding, stride):
    H16 = cudnn.data_type.HALF
    nd = w.dim() - 2
    out_sp = _out_size(x, w, padding, stride)
    yshape = [x.shape[0], w.shape[0], *out_sp]
    h = _get_handle()
    try:
        g = cudnn.pygraph(io_data_type=H16, intermediate_data_type=H16, compute_data_type=H16, handle=h)
        X = g.tensor(name="X", dim=list(x.shape), stride=list(x.stride()), data_type=H16)
        W = g.tensor(name="W", dim=list(w.shape), stride=list(w.stride()), data_type=H16)
        Y = g.conv_fprop(image=X, weight=W, padding=list(padding), stride=list(stride), dilation=[1] * nd,
                         compute_data_type=H16)
        Y.set_output(True).set_data_type(H16).set_dim(yshape).set_stride(_cl_stride(yshape))
        g.validate()
        g.build_operation_graph()
        g.deselect_behavior_notes([cudnn.behavior_note.RUNTIME_COMPILATION])
        g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        g.check_support()
    except Exception:  # noqa: BLE001 - no half-compute graph for this geometry
        return None
    for i in range(g.get_execution_plan_count()):
        if g.get_plan_name_at_index(i).split("_")[0] in ("eng0", "eng8"):
            continue
        try:
            g.build_plan_at_index(i)
        except Exception:  # noqa: BLE001
            continue
        if VERIFY and not _verify(g, X, W, Y, i, x, w, yshape, padding, stride):
            continue
        stats["built"] += 1
        return g, X, W, Y, i, g.get_workspace_size_plan_at_index(i), yshape
    return None


def _verify(g, X, W, Y, i, x, w, yshape, padding, stride):
    xr = (torch.rand(x.shape, device=x.device) * 2 - 1).half().contiguous(memory_format=_mf(x.dim()))
    y = torch.full(yshape, float("nan"), device=x.device, dtype=torch.half).contiguous(memory_format=_mf(x.dim()))
    ws = torch.empty(max(g.get_workspace_size_plan_at_index(i), 1), device=x.device, dtype=torch.uint8)
    g.execute_plan_at_index({X: xr, W: w, Y: y}, ws, i, handle=_get_handle())
    conv = F.conv2d if x.dim() == 4 else F.conv3d
    ref = conv(xr.float(), w.float(), None, stride, padding)
    dev = float((y.float() - ref).abs().max() / (ref.abs().max() + 1e-3))
    ok = dev == dev and dev < 0.05
    if not ok:
        print(f"[cudnn_conv] rejected {g.get_plan_name_at_index(i)} x{tuple(x.shape)} w{tuple(w.shape)}: {dev:.3g}")
    return ok


def _mf(dim):
    return torch.channels_last if dim == 4 else torch.channels_last_3d


def conv_half_acc(x, w, b, padding, stride):
    """fp16 conv with fp16 accumulation; x any layout (made channels-last), w channels-last fp16. Returns channels-last."""
    mf = _mf(x.dim())
    x = x.contiguous(memory_format=mf)
    key = (tuple(x.shape), tuple(w.shape), tuple(padding), tuple(stride), x.device.index)
    e = _plans.get(key, False)
    if e is False:
        e = _plans[key] = _build(x, w, padding, stride)
    if e is None:
        stats["torch"] += 1
        conv = F.conv2d if x.dim() == 4 else F.conv3d
        return conv(x, w, b, stride, padding).contiguous(memory_format=mf)
    stats["cudnn"] += 1
    g, X, W, Y, i, ws_size, yshape = e
    y = torch.empty(yshape, device=x.device, dtype=x.dtype, memory_format=mf)
    ws = _ws[0]
    if ws is None or ws.numel() < ws_size:
        ws = _ws[0] = torch.empty(max(ws_size, 1), device=x.device, dtype=torch.uint8)
    h = _get_handle()
    cudnn.set_stream(handle=h, stream=torch.cuda.current_stream().cuda_stream)
    g.execute_plan_at_index({X: x, W: w, Y: y}, ws, i, handle=h)
    if b is not None:
        y.add_(b.view(1, -1, *([1] * (x.dim() - 2))))
    return y


def _forward(self, x):
    return conv_half_acc(x, self.weight, self.bias, list(self.padding), list(self.stride))


def patch_convs(module, mark_only=False, skip=()):
    """Route every fp16 nn.Conv2d / nn.Conv3d (groups 1, dilation 1, zero padding) of `module` through conv_half_acc.
    mark_only: only set m._hacc (the caller's forward uses conv_half_acc); output channels are then zero-padded to a
    multiple of 8 (tensor-core friendly; the caller drops the extra outputs)."""
    assert cudnn is not None, "pip install nvidia-cudnn-frontend"
    n = 0
    for m in module.modules():
        if id(m) in skip:
            continue
        if type(m) in (nn.Conv2d, nn.Conv3d) and m.groups == 1 and all(d == 1 for d in m.dilation) \
                and m.padding_mode == "zeros" and not isinstance(m.padding, str) and m.weight.dtype == torch.half:
            w = m.weight.data
            if mark_only and w.shape[0] % 8:
                pad = torch.zeros((8 - w.shape[0] % 8,) + tuple(w.shape[1:]), dtype=w.dtype, device=w.device)
                w = torch.cat([w, pad], 0)
                if getattr(m, "_bias32", None) is not None:
                    m._bias32 = torch.cat([m._bias32, m._bias32.new_zeros(pad.shape[0])]).contiguous()
            m.weight.data = w.contiguous(memory_format=_mf(w.dim()))
            if mark_only:
                m._hacc = True
            else:
                m.forward = _forward.__get__(m)
                m._hacc = True
            n += 1
    return n
