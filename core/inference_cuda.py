"""Inference-only fast path for FoundationStereo with custom CUDA kernels (the plain-PyTorch path is core/inference.py).
Any input size, no per-size compile / autotune.

    from core.inference_cuda import prepare_inference, forward_fast, Opts
    prepare_inference(model, Opts(hacc=True, gacc=True, kern=True, fused=True, c3d=True))         # once: BN folded into conv weights, fp16 weights, NHWC, fp16-acc convs
    disp = forward_fast(model, L, R, iters=32)  # L, R: (1,3,H,W) 0..255, padded to /32

Changes vs upstream forward (math-preserving unless noted):
  - pure fp16 weights / activations (no autocast re-casts each iteration); disparity state and the softmax /
    regression / final upsampling stay fp32; BatchNorm folded into the (fp32) conv weights before the fp16 cast
  - channels_last(_3d) everywhere; InstanceNorm / LayerNorm-free layout-preserving norms
  - cost volume built, filtered (corr_stem, hourglass full-resolution tail, classifier) in H-strips with halos:
    exact (zero padding only at true borders), peak memory ~ one full-resolution 28-ch volume
  - geometry volume = hourglass output itself (NDHWC, fp16), level 1 by pairwise D-pooling; all-pairs correlation
    fp16; lookup = gather + linear interpolation (== grid_sample align_corners=True, zero padding), no host sync
    (upstream asserts torch.unique(...) per lookup = 4 syncs / iteration)
  - motion-encoder input channel order matched by permuting convc1's weights (no permute copy)
  - upsampling mask head only after the last iteration
  - features of the two images one after the other (per-sample norms -> same math, half the activation memory)
  - optional: fp16-accumulate cuDNN convs (hacc), fp16-accumulate GEMMs (gacc)
  - memory (12 MP fits in 16 GB): full volumes allocated channels-last directly (no full-volume copy), the corr_stem volume
    freed before the hourglass mid section, buffers released between phases (second ViT map, geometry / GRU buffers
    before the upsampling), geometry lookup + convc1 in row bands (FusedGRU.BAND_MB); outputs bit-identical
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

from core import submodule, extractor, update
from Utils import get_resize_keep_aspect_ratio


@dataclass
class Opts:
    half: bool = True          # pure fp16 weights/activations (else fp32 weights + autocast like upstream)
    cl: bool = True            # channels_last(_3d)
    hacc: bool = False         # fp16-accumulate cuDNN convs (core/cudnn_conv.py)
    gacc: bool = False         # fp16-accumulate cuBLAS GEMMs (torch.backends.cuda.matmul.allow_fp16_accumulation)
    strip_mb: float = 192.0    # H-strip size of the full-resolution volume work (MB per 28-ch fp16 strip)
    dino_amp: bool = False     # keep the ViT in fp32 weights + autocast (if pure fp16 ViT were not accurate enough)
    seq_feat: bool = True      # features one image at a time
    kern: bool = False         # conv + bias + activation (+ residual) epilogues as one custom kernel (core/kernels.py)
    fused: bool = False        # GRU loop: CUTLASS convs with fused GRU epilogues, no concatenations (FusedGRU; needs kern)
    c3d: bool = False          # full-resolution 3D convs (corr_stem, ...) as CUTLASS fp16-acc implicit GEMMs with fused
                               # epilogues on 32-channel (zero-padded) strips (needs kern, hwd layout)


# ------------------------------------------------------------------------------------------------ preparation
def _fold(conv, bn):
    """BN(conv(x)) -> conv'(x), in fp32 (one fp16 rounding of the folded weight later)."""
    s = bn.weight.detach().float() * torch.rsqrt(bn.running_var.float() + bn.eps)
    t = bn.bias.detach().float() - bn.running_mean.float() * s
    w = conv.weight.data.float()
    if isinstance(conv, (nn.ConvTranspose2d, nn.ConvTranspose3d)):
        shp = (1, -1) + (1,) * (w.dim() - 2)          # weight (Cin, Cout/groups, k...)
    else:
        shp = (-1,) + (1,) * (w.dim() - 1)
    conv.weight.data = w * s.view(shp)
    b = conv.bias.data.float() if conv.bias is not None else torch.zeros_like(s)
    conv.bias = nn.Parameter(b * s + t)


_BN = (nn.BatchNorm2d, nn.BatchNorm3d)
_CONV = (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)


def fold_batchnorms(model):
    n = 0
    for mod in model.modules():
        if isinstance(mod, nn.Sequential):
            ch = list(mod._modules.items())
            for (k0, a), (k1, b) in zip(ch, ch[1:]):
                if isinstance(a, _CONV) and isinstance(b, _BN):
                    _fold(a, b); mod._modules[k1] = nn.Identity(); n += 1
        for cname, bname in (("conv", "bn"), ("conv1", "bn1"), ("conv2", "bn2"), ("conv1", "norm1"), ("conv2", "norm2")):
            c, b = getattr(mod, cname, None), getattr(mod, bname, None)
            if isinstance(c, _CONV) and isinstance(b, _BN):
                _fold(c, b); setattr(mod, bname, nn.Identity()); n += 1
    # ResidualBlock(norm_fn='batch').norm3 lives in .downsample (already folded via the Sequential rule)
    for mod in model.modules():
        if isinstance(mod, extractor.ResidualBlock) and isinstance(getattr(mod, "norm3", None), _BN):
            mod.norm3 = nn.Identity()
    return n


def _instance_norm_nhwc(self, x):
    # == F.instance_norm (affine=False, no running stats), statistics in fp32, keeps the memory layout
    dims = tuple(range(2, x.dim()))
    var, mean = torch.var_mean(x.float(), dim=dims, keepdim=True, unbiased=False)
    return ((x.float() - mean) * torch.rsqrt(var + self.eps)).to(x.dtype)


def prepare_inference(model, o: Opts = Opts()):
    model.eval()
    model._opts = o
    nf = fold_batchnorms(model)
    for m in model.modules():
        if isinstance(m, (nn.InstanceNorm2d, nn.InstanceNorm3d)) and not m.affine:
            m.forward = _instance_norm_nhwc.__get__(m)
    # motion encoder: our geometry feature order is [geo0 (k,c) | corr0 (k) | geo1 (k,c) | corr1 (k)];
    # upstream's is [geo0 (c,k) | corr0 (k) | geo1 (c,k) | corr1 (k)] -> permute convc1's input channels once
    C, K, L = 28, 2 * model.args.corr_radius + 1, model.args.corr_levels
    perm, base = [], 0
    for _ in range(L):
        perm += [base + c * K + k for k in range(K) for c in range(C)]
        base += C * K
        perm += [base + k for k in range(K)]
        base += K
    if not o.kern:  # (the geo_lookup kernel writes upstream's order directly)
        cc1 = model.update_block.encoder.convc1
        cc1.weight.data = cc1.weight.data[:, perm].contiguous()
    dino = model.feature.dino
    if o.half:
        model.half()
        if o.dino_amp:
            dino.float()
    if o.cl:
        for m in model.modules():
            if isinstance(m, _CONV) and m.weight.dim() in (4, 5):
                m.weight.data = m.weight.data.contiguous(
                    memory_format=torch.channels_last if m.weight.dim() == 4 else torch.channels_last_3d)
    nh = 0
    if o.hacc and o.half:
        from core import cudnn_conv
        # not the ViT (patch embed only) -- keep its convs on torch
        skip = {id(m) for m in dino.modules()} | {id(m) for m in model.modules() if isinstance(m, nn.Conv3d)}
        nh = cudnn_conv.patch_convs(model, skip=skip)
    if o.kern and o.half:
        cc1 = model.update_block.encoder.convc1
        cin = cc1.weight.shape[1]
        if cin % 8:
            w = torch.zeros((cc1.weight.shape[0], (cin + 7) // 8 * 8) + tuple(cc1.weight.shape[2:]), device=cc1.weight.device, dtype=cc1.weight.dtype)
            w[:, :cin] = cc1.weight.data
            cc1.weight.data = w.contiguous(memory_format=torch.channels_last)
            cc1.in_channels = w.shape[1]
        model._hwd = True
        _to_hwd(model)
        patch_kernels(model)
        for m in model.modules():   # remaining plain convs: bias through the epilogue kernel too
            if type(m) in (nn.Conv2d, nn.Conv3d) and id(m) not in {id(x) for x in dino.modules()}:
                m.forward = (lambda self, x: conv_ep(self, x)).__get__(m)
    model._dx = torch.linspace(-model.args.corr_radius, model.args.corr_radius, 2 * model.args.corr_radius + 1,
                               device=next(model.parameters()).device)
    return dict(folded_bn=nf, hacc_convs=nh)



# ------------------------------------------------------------------------------------------------ fused epilogues
ACT = {"none": 0, "relu": 1, "leaky": 2, "tanh": 3, "sigmoid": 4, "gelu": 6}
_K = [None]


def _kern():
    if _K[0] is None:
        from core.kernels import Kernels
        _K[0] = Kernels.get()
    return _K[0]


def epilogue(y, t=None, act=0, r=None, s=None):
    """In place: y = act(y * s + t + r); y dense channels-last fp16; s, t fp32 [C] or None; r like y or None."""
    C = y.shape[1]
    dense = y.dtype == torch.float16 and y.is_contiguous(memory_format=_mf(y))
    if not dense or (r is not None and (r.shape != y.shape or not r.is_contiguous(memory_format=_mf(r)))):
        shp = (1, -1) + (1,) * (y.dim() - 2)
        v = y.float()
        if s is not None:
            v = v * s.view(shp)
        if t is not None:
            v = v + t.view(shp)
        if r is not None:
            v = v + r.float()
        v = [lambda u: u, F.relu, lambda u: F.leaky_relu(u, 0.01), torch.tanh, torch.sigmoid][act](v)
        return y.copy_(v)
    P = y.numel() // C
    k = _kern()
    vw = 8 if C % 8 == 0 else (4 if C % 4 == 0 else (2 if C % 2 == 0 else 1))
    n = P * (C // vw)
    k.launch({8: "affine_act8", 4: "affine_act4", 2: "affine_act2", 1: "affine_act"}[vw], k.linear(n), y, C, 0 if s is None else s, 0 if t is None else t,
             0 if r is None else r, 0 if r is None else C, P, C, act)
    return y


def _canon(x):
    """Batch-1 channels-last tensors may carry a degenerate batch stride (e.g. 28 from a view): cuDNN then falls back to a
    generic kernel (8x slower 3D convs). Re-stride (no copy) to the canonical channels-last strides."""
    if x.shape[0] == 1 and x.dim() >= 4 and x.is_contiguous(memory_format=_mf(x)) and x.stride(0) != x.numel():
        st = list(x.stride()); st[0] = x.numel()
        x = x.as_strided(x.shape, st)
    return x


def conv_ep(m, x, act=0, r=None):
    """conv module (bias folded into the epilogue) -> act(conv(x) + b (+ r)) in one extra pass."""
    x = _canon(x)
    if getattr(m, "_hacc", False):
        from core.cudnn_conv import conv_half_acc
        y = conv_half_acc(x, m.weight, None, list(m.padding), list(m.stride))
    elif isinstance(m, nn.Conv3d) and m.kernel_size[:2] == (1, 1) and m.stride == (1, 1, 1) and m.groups == 1 \
            and m.kernel_size[2] > 1:
        # (1,1,k) along the innermost (D) axis as a 2D (1,k) conv on the (B, C, H*W, D) view: 9.6 vs 42 ms at Midd-H
        B, C, H, W, D = x.shape
        x = x.contiguous(memory_format=torch.channels_last_3d)
        w2 = getattr(m, "_w2d", None)
        if w2 is None:
            w2 = m._w2d = m.weight.detach().view(m.out_channels, C, 1, m.kernel_size[2]).contiguous(memory_format=torch.channels_last)
        y = F.conv2d(x.view(B, C, H * W, D), w2, None, 1, (0, m.padding[2])).view(B, m.out_channels, H, W, D)
    elif isinstance(m, (nn.ConvTranspose2d, nn.ConvTranspose3d)):
        f = F.conv_transpose2d if x.dim() == 4 else F.conv_transpose3d
        y = f(x, m.weight, None, m.stride, m.padding, m.output_padding, m.groups, m.dilation)
    else:
        f = F.conv2d if x.dim() == 4 else F.conv3d
        y = f(x, m.weight, None, m.stride, m.padding, m.dilation, m.groups)
    y = y.contiguous(memory_format=_mf(y))
    if m.bias is not None or act or r is not None:
        y = epilogue(y, m._b32 if m.bias is not None else None, act, r)
    return y


def _basic_conv_fwd(self, x):          # BasicConv with folded BN: conv -> LeakyReLU(0.01)
    if isinstance(self.bn, nn.Identity):
        return conv_ep(self.conv, x, ACT["leaky"] if self.relu else 0)
    y = conv_ep(self.conv, x)            # InstanceNorm variant
    var, mean = torch.var_mean(y.float(), dim=tuple(range(2, y.dim())), keepdim=True, unbiased=False)
    rstd = torch.rsqrt(var + self.bn.eps)
    return epilogue(y, (-mean * rstd).view(-1).contiguous(), ACT["leaky"] if self.relu else 0, s=rstd.view(-1).contiguous())


def _reduced_fwd(self, x):             # Conv3dNormActReduced with folded BN
    return conv_ep(self.conv2[0], conv_ep(self.conv1[0], x, ACT["relu"]), ACT["relu"])


def _resblock3d_fwd(self, x):          # ResnetBasicBlock(3D) with folded BN, no downsample
    out = conv_ep(self.conv1, x, ACT["relu"])
    return conv_ep(self.conv2, out, ACT["relu"], r=x)


def _resblock_fwd(self, x):           # extractor.ResidualBlock with folded BN: relu(x' + relu(bn2(conv2(relu(bn1(conv1 x))))))
    y = conv_ep(self.conv1, x, ACT["relu"])
    y = conv_ep(self.conv2, y, ACT["relu"])
    if self.downsample is not None:
        x = conv_ep(self.downsample[0], x)
    return epilogue(y, None, ACT["relu"], r=_canon(x).contiguous(memory_format=_mf(y)))


def _feature_att_fwd(self, cv, feat):  # in place on the (fresh) cost-volume activation
    return cv.mul_(torch.sigmoid(self.feat_att(feat)).unsqueeze(-1 if getattr(self, "_hwd", False) else 2))


def _to_hwd(model):
    """3D convs for (B, C, H, W, D) volumes: kernel / stride / padding axes (d, h, w) -> (h, w, d)."""
    rot = lambda t: (t[1], t[2], t[0]) if isinstance(t, tuple) and len(t) == 3 else t
    for m in model.modules():
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            m.weight.data = m.weight.data.permute(0, 1, 3, 4, 2).contiguous(memory_format=torch.channels_last_3d)
            for a in ("kernel_size", "stride", "padding", "dilation", "output_padding"):
                if hasattr(m, a):
                    setattr(m, a, rot(getattr(m, a)))


def patch_kernels(model):
    n = 0
    for m in model.cost_agg.modules():
        if isinstance(m, submodule.FeatureAtt):
            m.forward = _feature_att_fwd.__get__(m)
            m._hwd = getattr(model, "_hwd", False)
    for m in model.modules():
        if isinstance(m, _CONV) and m.bias is not None:
            m._b32 = m.bias.detach().float().contiguous()
    for m in model.modules():
        if isinstance(m, submodule.BasicConv):
            m.forward = _basic_conv_fwd.__get__(m); n += 1
        elif isinstance(m, extractor.ResidualBlock) and isinstance(m.norm1, nn.Identity) and isinstance(m.norm2, nn.Identity) \
                and (m.downsample is None or (len(m.downsample) == 2 and isinstance(m.downsample[1], nn.Identity))):
            m.forward = _resblock_fwd.__get__(m); n += 1
        elif isinstance(m, submodule.Conv3dNormActReduced) and isinstance(m.conv1[1], nn.Identity):
            c3 = getattr(model._opts, "c3d", False) and getattr(model, "_hwd", False) and m.conv1[0].in_channels % 8 == 0 \
                and m.conv1[0].out_channels % 8 == 0 and m.conv2[0].out_channels % 8 == 0 \
                and tuple(m.conv1[0].stride) == (1, 1, 1) and tuple(m.conv2[0].stride) == (1, 1, 1)
            m.forward = (_reduced_c3_fwd if c3 else _reduced_fwd).__get__(m); n += 1
        elif isinstance(m, (submodule.ResnetBasicBlock3D, submodule.ResnetBasicBlock)) and m.downsample is None \
                and isinstance(getattr(m, "bn1", None), nn.Identity) and isinstance(getattr(m, "bn2", None), nn.Identity):
            m.forward = _resblock3d_fwd.__get__(m); n += 1
    # GRUs: z and r convs merged into one conv (same input hx)
    for g in model.modules():
        if isinstance(g, update.RaftConvGRU):
            zr = nn.Conv2d(g.convz.in_channels, 2 * g.convz.out_channels, g.convz.kernel_size, padding=g.convz.padding)
            zr = zr.to(g.convz.weight.device, g.convz.weight.dtype)
            zr.weight.data = torch.cat([g.convz.weight.data, g.convr.weight.data]).contiguous(memory_format=torch.channels_last)
            zr.bias.data = torch.cat([g.convz.bias.data, g.convr.bias.data])
            zr._b32 = torch.cat([g.convz._b32, g.convr._b32]).contiguous()
            zr._hacc = getattr(g.convz, "_hacc", False)
            g.convzr = zr
    return n


# ------------------------------------------------------------------------------------------------ helpers
def _mf(x):
    return torch.channels_last if x.dim() == 4 else torch.channels_last_3d


class _Ctx:
    def __init__(self, model):
        self.o = model._opts
        self.dt = torch.float16 if self.o.half else torch.float32
        self.hwd = getattr(model, "_hwd", False)   # cost volumes as (B, C, H, W, D) instead of (B, C, D, H, W)

    def vshape(self, B, C, D, H, W):
        return (B, C, H, W, D) if self.hwd else (B, C, D, H, W)

    def dims(self, v):  # -> B, C, D, H, W
        if self.hwd:
            B, C, H, W, D = v.shape
            return B, C, D, H, W
        return v.shape

    def rows(self, v, a, b):  # H-slice of a volume
        return v[:, :, a:b] if self.hwd else v[:, :, :, a:b]

    def amp(self):
        return torch.autocast("cuda", dtype=torch.float16, enabled=not self.o.half)

    def act(self, x):  # input activation layout/dtype
        x = x.to(self.dt)
        return x.contiguous(memory_format=_mf(x)) if self.o.cl else x


def _strips(H, rows):
    h0 = 0
    while h0 < H:
        yield h0, min(H, h0 + rows)
        h0 += rows


def _rows_for(model, B, D, W, C=28):
    elem = 2 if model._opts.half else 4
    return max(8, int(model._opts.strip_mb * 2**20 // (B * C * D * W * elem)))


def normalize_image(img):
    mean = img.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = img.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return ((img / 255.0 - mean) / std).contiguous()


# ------------------------------------------------------------------------------------------------ features
def feature_fast(ft, x, ctx):
    """== Feature.forward for a batch of images (fp32 normalized input)."""
    B, C, H, W = x.shape
    divider = np.lcm(ft.patch_size, 16)
    Hr, Wr = get_resize_keep_aspect_ratio(H, W, divider=divider, max_H=1344, max_W=1344)
    xin = F.interpolate(x, size=(Hr, Wr), mode="bicubic", align_corners=False)
    if ctx.o.dino_amp:
        with torch.autocast("cuda", dtype=torch.float16):
            vit = ft.dino(xin)["out"].to(ctx.dt)
        vit = F.interpolate(vit, size=(H // 4, W // 4), mode="bilinear", align_corners=True)
    elif ctx.o.kern and B == 1:
        vit = dino_fast(ft.dino, xin.to(ctx.dt), (H // 4, W // 4))
    else:
        vit = ft.dino(xin.to(ctx.dt))["out"]
        vit = F.interpolate(vit, size=(H // 4, W // 4), mode="bilinear", align_corners=True)
    x = ctx.act(x)
    x = ft.stem(x)
    x4 = ft.stages[0](x)
    x8 = ft.stages[1](x4)
    x16 = ft.stages[2](x8)
    x32 = ft.stages[3](x16)
    x16 = ft.deconv32_16(x32, x16)
    x8 = ft.deconv16_8(x16, x8)
    x4 = ft.deconv8_4(x8, x4)
    x4 = torch.cat([x4, ctx.act(vit)], dim=1)
    x4 = ft.conv4(x4)
    return [x4, x8, x16, x32], vit


def dino_fast(da, x, out_hw):
    """== DepthAnythingFeature.forward(x)['out'] resized to out_hw (bilinear, align_corners=True), batch 1, without the
    unused depth branch (output_conv2 + disparity normalization at ViT input resolution) and without materializing the
    ViT-resolution 128-ch map: both bilinear resizes in one kernel (bilerp2_nhwc, fp16 intermediate rounding kept)."""
    pre, head = da.depth_anything.pretrained, da.depth_anything.depth_head
    h, w = x.shape[-2:]
    feats = pre.get_intermediate_layers(x, da.intermediate_layer_idx[da.encoder], return_class_token=True)
    ps = pre.patch_size
    ph_, pw_ = h // ps, w // ps
    out = []
    for i, f in enumerate(feats):
        if head.use_clstoken:
            t, cls = f[0], f[1]
            t = head.readout_projects[i](torch.cat((t, cls.unsqueeze(1).expand_as(t)), -1))
        else:
            t = f[0]
        t = t.permute(0, 2, 1).reshape((t.shape[0], t.shape[-1], ph_, pw_))
        out.append(head.resize_layers[i](head.projects[i](t)))
    sc = head.scratch
    l1, l2, l3, l4 = sc.layer1_rn(out[0]), sc.layer2_rn(out[1]), sc.layer3_rn(out[2]), sc.layer4_rn(out[3])
    del out
    p4 = sc.refinenet4(l4, size=l3.shape[2:])
    p3 = sc.refinenet3(p4, l3, size=l2.shape[2:])
    p2 = sc.refinenet2(p3, l2, size=l1.shape[2:])
    p1 = sc.refinenet1(p2, l1)
    y = sc.output_conv1(p1).contiguous(memory_format=torch.channels_last)
    B, C, Hs, Ws = y.shape
    Hd, Wd = out_hw
    o = torch.empty((B, C, Hd, Wd), device=y.device, dtype=y.dtype, memory_format=torch.channels_last)
    k = _kern()
    k.launch("bilerp2_nhwc", k.linear(Hd * Wd * C // 8), y, Hs, Ws, ph_ * ps, pw_ * ps, o, Hd, Wd, C)
    return o


def cnet_fast(cn, x, vit, ctx, num_layers=3):
    """== ContextNetDino.forward; the full-resolution stem (conv1 7x7 s1 + layer1, 64 ch) and layer2 (s2) run in
    H-strips (halo 16 input rows, even-aligned): never materializes a full-resolution 64-ch activation."""
    B, C, H, W = x.shape
    Ho = H // 2
    out = None
    orows = max(16, int(ctx.o.strip_mb * 2**20 // (W * 64 * 2 * 2 * 2)))
    halo = 16
    for o0 in range(0, Ho, orows):
        o1 = min(Ho, o0 + orows)
        a, b = max(0, 2 * o0 - halo), min(H, 2 * o1 + halo)
        a -= a % 2
        y = cn.layer2(cn.layer1(cn.relu1(cn.norm1(cn.conv1(x[:, :, a:b])))))
        y = y[:, :, o0 - a // 2:o1 - a // 2]
        if out is None:
            out = torch.empty((B, y.shape[1], Ho, y.shape[3]), device=x.device, dtype=y.dtype, memory_format=_mf(y))
        out[:, :, o0:o1] = y
        del y
    x = cn.layer3(out)
    del out
    x = cn.conv2(torch.cat([x, vit], dim=1))
    outputs04 = [f(x) for f in cn.outputs04]
    y = cn.layer4(x)
    outputs08 = [f(y) for f in cn.outputs08]
    z = cn.layer5(y)
    outputs16 = [f(z) for f in cn.outputs16]
    return (outputs04, outputs08, outputs16)


# ------------------------------------------------------------------------------------------------ cost volume
def _build_volume_strip(model, fl, fr, l12, r12, a, b, D, ctx):
    """gwc (8) + concat (12+12) volume for rows [a, b): (B, 32, D, b-a, W), == build_gwc_volume / build_concat_volume.
    gwc: per row and group all-pairs products by one fp32 GEMM against the W-flipped right features; disparity d of
    pixel w is then element (w, W-1-w+d) = a strided (diagonal) view; entries with w < d are zero (as upstream)."""
    G = model.cv_group
    B, C, _, W = fl.shape
    vol = torch.empty(ctx.vshape(B, 32, D, b - a, W), device=fl.device, dtype=ctx.dt,
                      memory_format=torch.channels_last_3d if ctx.o.cl else torch.contiguous_format)
    fast = ctx.o.kern and ctx.hwd and B == 1 and G == 8 and vol.is_contiguous(memory_format=torch.channels_last_3d)
    if fast:
        l12c = l12.contiguous(memory_format=torch.channels_last)
        r12c = r12.contiguous(memory_format=torch.channels_last)
        k = _kern()
    valid = (torch.arange(W, device=fl.device).view(1, W) >= torch.arange(D, device=fl.device).view(D, 1))  # (D,W)
    sub = 16
    for r0 in range(a, b, sub):
        r1 = min(b, r0 + sub)
        h = r1 - r0
        f1 = F.normalize(fl[:, :, r0:r1].float().reshape(B, G, C // G, h, W), dim=2)
        f2 = F.normalize(fr[:, :, r0:r1].float().reshape(B, G, C // G, h, W), dim=2)
        f1 = f1.permute(0, 1, 3, 4, 2).reshape(B * G * h, W, C // G)
        f2 = f2.flip(-1).permute(0, 1, 3, 2, 4).reshape(B * G * h, C // G, W)
        M = torch.bmm(f1, f2)                                               # (BGh, W, W) [w1, W-1-w2]
        if fast:   # one kernel writes all 32 channels of these rows (gwc gather + the two projections)
            k.launch("build_vol_hwd", k.linear(h * W * D), M, G, h, W, D, l12c[:, :, r0:r1], r12c[:, :, r0:r1],
                     vol[:, :, r0 - a:r1 - a])
            del M
            continue
        V = M.as_strided((B * G * h, D, W), (W * W, 1, W - 1), M.storage_offset() + W - 1)
        V = torch.where(valid, V, 0.0).view(B, G, h, D, W)
        if ctx.hwd:
            vol[:, :G, r0 - a:r1 - a] = V.permute(0, 1, 2, 4, 3)
        else:
            vol[:, :G, :, r0 - a:r1 - a] = V.permute(0, 1, 3, 2, 4)
        del M, V
    if fast:
        return vol
    vol[:, G:G + 12] = l12[:, :, a:b].unsqueeze(-1 if ctx.hwd else 2)
    rp = F.pad(r12[:, :, a:b], (D, 0))                                       # (B,12,h,W+D)
    win = rp.unfold(3, W, 1)                                                 # (B,12,h,D+1,W): window s = shift D-s
    win = win[:, :, :, 1:].flip(3)
    vol[:, G + 12:] = win.permute(0, 1, 2, 4, 3) if ctx.hwd else win.permute(0, 1, 3, 2, 4)
    return vol


class C3:
    """A Conv3d (hwd layout, folded BN) as a CUTLASS fp16-accumulate implicit GEMM (core/cutlass_conv.cu fused_conv3d,
    compiled once, any size) on 32-channel zero-padded NDHWC buffers: weights / bias zero-padded to 32 in / out channels,
    epilogue = bias + optional per-pixel residual + activation; padded output channels stay exactly 0."""
    CP = 32

    def __init__(self, m, pad=True):
        w = m.weight.detach()
        K, C = w.shape[:2]
        assert m.groups == 1 and tuple(m.stride) == (1, 1, 1)
        if pad:      # 32-ch zero-padded buffers, tile 128x32x32
            assert K <= self.CP and C <= self.CP
            self.K, self.cfg = self.CP, 0
            wp = torch.zeros((self.CP, self.CP) + tuple(w.shape[2:]), device=w.device, dtype=w.dtype)
            wp[:K, :C] = w
        else:        # native channel counts (multiples of 8), tile 128x64x32
            assert K % 8 == 0 and C % 8 == 0
            self.K, self.cfg, wp = K, 3, w
        self.w = wp.contiguous(memory_format=torch.channels_last_3d)
        self.b = torch.zeros(self.K, device=w.device)
        if m.bias is not None:
            self.b[:K] = m.bias.detach().float()
        self.pad = tuple(m.padding)

    def __call__(self, x, out=None, act=0, res=None):
        from core.cutlass_conv import FusedConvs, EpiArgs
        x = _canon(x)
        assert x.is_contiguous(memory_format=torch.channels_last_3d)
        if out is None:
            sp = [x.shape[2 + i] + 2 * self.pad[i] - self.w.shape[2 + i] + 1 for i in range(3)]
            out = torch.empty((x.shape[0], self.K, *sp), device=x.device, dtype=x.dtype, memory_format=torch.channels_last_3d)
        e = EpiArgs(mode=2, act=act, b=self.b.data_ptr(), o=out.data_ptr(), os=self.K,
                    c=0 if res is None else res.data_ptr(), cs=self.K)
        rc = FusedConvs.get().conv3d(self.cfg, x, self.w, e, self.pad)
        assert rc == 0, rc
        return out


def _reduced_c3_fwd(self, x):          # Conv3dNormActReduced (folded BN, stride 1, C % 8 == 0) as two C3 convs
    if not hasattr(self, "_c3"):
        self._c3 = (C3(self.conv1[0], pad=False), C3(self.conv2[0], pad=False))
    return self._c3[1](self._c3[0](x, act=ACT["relu"]), act=ACT["relu"])


def _cat_1x1(bc, a, b):
    """BasicConv(1x1x1 Conv3d, folded BN, LeakyReLU) of torch.cat((a, b), 1) as two fp16-acc GEMMs on the channels-last
    (P, C) views (no concatenated copy), bias + activation in one epilogue pass."""
    m = bc.conv
    B, Ca, H, W, D = a.shape
    if not hasattr(m, "_wa"):
        w = m.weight.detach().view(m.out_channels, -1)
        m._wa, m._wb = w[:, :Ca].contiguous(), w[:, Ca:].contiguous()
    a2 = a.contiguous(memory_format=torch.channels_last_3d).permute(0, 2, 3, 4, 1).reshape(-1, Ca)
    b2 = b.contiguous(memory_format=torch.channels_last_3d).permute(0, 2, 3, 4, 1).reshape(-1, b.shape[1])
    y = torch.addmm(F.linear(a2, m._wa), b2, m._wb.t())
    y = y.view(B, H, W, D, -1).permute(0, 4, 1, 2, 3)
    return epilogue(_canon(y), m._b32, ACT["leaky"] if bc.relu else 0)


def _deconv_up32(bc, x):
    """BasicConv(ConvTranspose3d, folded BN, LeakyReLU) with its output channels zero-padded to 32 (C3 input)."""
    m = bc.conv
    if not hasattr(m, "_w32"):
        w = m.weight.detach()                                   # (Cin, Cout, k...)
        wp = torch.zeros((w.shape[0], C3.CP) + tuple(w.shape[2:]), device=w.device, dtype=w.dtype)
        wp[:, :w.shape[1]] = w
        m._w32 = wp.contiguous(memory_format=torch.channels_last_3d)
        b = torch.zeros(C3.CP, device=w.device)
        b[:w.shape[1]] = m.bias.detach().float()
        m._b32p = b
    y = F.conv_transpose3d(_canon(x), m._w32, None, m.stride, m.padding, m.output_padding, m.groups, m.dilation)
    y = y.contiguous(memory_format=torch.channels_last_3d)
    return epilogue(y, m._b32p, ACT["leaky"] if bc.relu else 0)


def _stem_c3(model):
    cs = model.corr_stem
    if not hasattr(model, "_stem_c3"):
        model._stem_c3 = [C3(cs[0]), C3(cs[1].conv), C3(cs[2].conv1), C3(cs[2].conv2), C3(cs[3].conv1), C3(cs[3].conv2)]
    return model._stem_c3


def corr_stem_c3(model, v):
    """corr_stem on a 32-ch volume strip: 1x1 conv, BasicConv (LeakyReLU), 2 x ResnetBasicBlock3D; 3 rotating buffers."""
    c0, c1, r1a, r1b, r2a, r2b = _stem_c3(model)
    bufs = [torch.empty_like(v, memory_format=torch.channels_last_3d) for _ in range(2)]
    a = c0(v, bufs[0])
    b = c1(a, v, ACT["leaky"])          # v is free now
    t = r1a(b, bufs[0], ACT["relu"])
    a = r1b(t, bufs[1], ACT["relu"], res=b)
    t = r2a(a, b, ACT["relu"])
    return r2b(t, bufs[0], ACT["relu"], res=a)


def corr_stem_strips(model, fl, fr, D, ctx):
    """comb = corr_feature_att(corr_stem(volume), fl), strip by strip (corr_stem receptive radius 5 rows)."""
    l12 = model.proj_cmb(fl)
    r12 = model.proj_cmb(fr)
    att = torch.sigmoid(model.corr_feature_att.feat_att(fl)).unsqueeze(-1 if ctx.hwd else 2)
    B, _, H, W = fl.shape
    out = torch.empty(ctx.vshape(B, 28, D, H, W), device=fl.device, dtype=ctx.dt,   # (no full-volume .contiguous copy)
                      memory_format=torch.channels_last_3d if ctx.o.cl else torch.contiguous_format)
    halo = 5
    for h0, h1 in _strips(H, _rows_for(model, B, D, W)):
        a, b = max(0, h0 - halo), min(H, h1 + halo)
        v = _build_volume_strip(model, fl, fr, l12, r12, a, b, D, ctx)
        if ctx.o.c3d and ctx.hwd:
            v = corr_stem_c3(model, v)[:, :28]
        else:
            v = model.corr_stem(v)
        ctx.rows(out, h0, h1)[...] = ctx.rows(v, h0 - a, h1 - a) * ctx.rows(att, h0, h1)
        del v
    return out


def hourglass_fast(hg, xs, features, model, ctx):
    """== hourglass.forward; xs = [x] (consumed: freed after its last use). Full-resolution tail in strips."""
    x = xs.pop()
    # first stride-2 conv over the full-resolution volume in H-strips: cuDNN copies a 28-ch NDHWC input into a
    # channel-padded (32) buffer, i.e. a full-volume transient (+2.5 GB at Midd-F) if run in one call
    B, C, D, H, W = ctx.dims(x)
    first = hg.conv1[0]
    orows = max(4, _rows_for(model, B, D, W) // 2)
    c1 = None
    for o0 in range(0, H // 2, orows):
        o1 = min(H // 2, o0 + orows)
        a, b = max(0, 2 * o0 - 2), min(H, 2 * o1 + 2)
        y = ctx.rows(first(ctx.rows(x, a, b)), o0 - a // 2, o1 - a // 2)
        if c1 is None:
            _, Cy, Dy, _, Wy = ctx.dims(y)
            c1 = torch.empty(ctx.vshape(B, Cy, Dy, H // 2, Wy), device=x.device, dtype=y.dtype, memory_format=_mf(y))
        ctx.rows(c1, o0, o1)[...] = y
        del y
    # conv_patch is the last use of the full-resolution input volume: run it now so that the volume is freed before the
    # H/8..H/32 section (-1 full volume at that phase's peak; independent computations, same results)
    xp = _patch_conv(hg.conv_patch[0], x) if ctx.hwd else hg.conv_patch(x)
    del x
    conv1 = hg.conv1[1](c1)
    del c1
    conv1 = hg.feature_att_8(conv1, features[1])
    conv2 = hg.conv2(conv1)
    conv2 = hg.feature_att_16(conv2, features[2])
    conv3 = hg.conv3(conv2)
    conv3 = hg.feature_att_32(conv3, features[3])
    conv3_up = hg.conv3_up(conv3)
    if ctx.o.c3d and ctx.hwd:
        conv2 = hg.agg_0[2](hg.agg_0[1](_cat_1x1(hg.agg_0[0], conv3_up, conv2)))
    else:
        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = hg.agg_0(conv2)
    conv2 = hg.feature_att_up_16(conv2, features[2])
    conv2_up = hg.conv2_up(conv2)
    if ctx.o.c3d and ctx.hwd:
        conv1 = hg.agg_1[2](hg.agg_1[1](_cat_1x1(hg.agg_1[0], conv2_up, conv1)))
    else:
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = hg.agg_1(conv1)
    conv1 = hg.feature_att_up_8(conv1, features[1])
    xp = transformer_fast(hg.atts["4"], xp, ctx)
    out = torch.empty(ctx.vshape(B, C, D, H, W), device=conv1.device, dtype=conv1.dtype,
                      memory_format=torch.channels_last_3d if ctx.o.cl else torch.contiguous_format)
    halo = 2  # conv_out: 2 x (1,3,3)
    H8, H16 = ctx.dims(conv1)[3], ctx.dims(xp)[3]
    for h0, h1 in _strips(H, _rows_for(model, B, D, W)):
        r0, r1 = max(0, h0 - halo), min(H, h1 + halo)
        # transposed conv (k4 s2 p1): rows [r0, r1) need conv1 rows [r0//2 - 1, (r1+1)//2 + 1)
        a, b = max(0, r0 // 2 - 1), min(H8, (r1 + 1) // 2 + 1)
        c3 = ctx.o.c3d and ctx.hwd
        if c3:   # conv1_up with 32 (zero-padded) output channels -> conv_out as C3 convs on dense 32-ch strips
            y = ctx.rows(_deconv_up32(hg.conv1_up, ctx.rows(conv1, a, b)), r0 - 2 * a, r1 - 2 * a)
        else:
            y = ctx.rows(hg.conv1_up(ctx.rows(conv1, a, b)), r0 - 2 * a, r1 - 2 * a)
        # trilinear x4 (align_corners=False): rows [r0, r1) need xp rows [r0//4 - 1, (r1-1)//4 + 2)
        a, b = max(0, r0 // 4 - 1), min(H16, (r1 - 1) // 4 + 2)
        if c3 and B == 1 and C % 4 == 0 and xp.is_contiguous(memory_format=torch.channels_last_3d):
            y = _canon(y)   # trilinear x4 of xp added in place by one kernel (global coordinates)
            _, _, Hx, Wx, Dx = xp.shape
            k = _kern()
            k.launch("trilerp4_add", k.linear((r1 - r0) * W * D * C // 4), xp, Hx, Wx, Dx, C, y, y.shape[1], r0, r1 - r0, W, D)
        else:
            up = F.interpolate(ctx.rows(xp, a, b), scale_factor=4, mode="trilinear", align_corners=False)
            y[:, :C].add_(ctx.rows(up, r0 - 4 * a, r1 - 4 * a))   # in place: keeps y's channels-last layout
            del up
        if c3:
            y = _canon(y)
            if not hasattr(hg, "_out_c3"):
                hg._out_c3 = [C3(m) for blk in hg.conv_out for m in (blk.conv1[0], blk.conv2[0])]
            t = torch.empty_like(y, memory_format=torch.channels_last_3d)
            for i, cv in enumerate(hg._out_c3):
                src, dst = (y, t) if i % 2 == 0 else (t, y)
                cv(src, dst, ACT["relu"])
            y = y[:, :C]
            del t
        else:
            y = hg.conv_out(y)
        ctx.rows(out, h0, h1)[...] = ctx.rows(y, h0 - r0, h1 - r0)
        del y
    return out


def _patch_conv(conv, x):
    """Depthwise Conv3d with kernel == stride (non-overlapping patches) as a sum over the k^3 strided taps with fp32
    accumulation (cuDNN runs it with a generic kernel: 28 ms at Midd-H)."""
    k = conv.kernel_size
    assert conv.groups == conv.in_channels and tuple(conv.stride) == tuple(k) and conv.padding == (0, 0, 0)
    B, C, S0, S1, S2 = x.shape
    o = [S0 // k[0], S1 // k[1], S2 // k[2]]
    if B == 1 and C % 4 == 0 and x.dtype == torch.float16 and x.is_contiguous(memory_format=torch.channels_last_3d):
        if not hasattr(conv, "_wdw"):
            conv._wdw = conv.weight.detach().float().reshape(C, -1).contiguous()
            conv._bdw = conv.bias.detach().float().contiguous() if conv.bias is not None else None
        y = torch.empty((B, C, *o), device=x.device, dtype=x.dtype, memory_format=torch.channels_last_3d)
        kk = _kern()
        kk.launch("patch_dw", kk.linear(o[0] * o[1] * o[2] * C // 4), _canon(x), S1, S2, C, conv._wdw,
                  0 if conv._bdw is None else conv._bdw, y, o[0], o[1], o[2], k[0], k[1], k[2])
        return y
    xc = x[:, :, :o[0] * k[0], :o[1] * k[1], :o[2] * k[2]]
    acc = torch.zeros((B, C, *o), device=x.device, dtype=torch.float32).contiguous(memory_format=torch.channels_last_3d)
    w = conv.weight.float()
    for i in range(k[0]):
        for j in range(k[1]):
            for l in range(k[2]):
                acc.add_(xc[:, :, i::k[0], j::k[1], l::k[2]] * w[:, 0, i, j, l].view(1, C, 1, 1, 1))
    if conv.bias is not None:
        acc.add_(conv.bias.float().view(1, C, 1, 1, 1))
    return acc.to(x.dtype)


def transformer_fast(att, cv, ctx):
    """== CostVolumeDisparityAttention.forward (+ optional attention fix)."""
    B, C, D, H, W = ctx.dims(cv)
    x = (cv.permute(0, 2, 3, 4, 1) if ctx.hwd else cv.permute(0, 3, 4, 2, 1)).reshape(B * H * W, D, C)
    x = x + att.pos_embed0.pe.to(x.device, x.dtype)[:, :D]
    for layer in att.sa:
        a = layer.self_attn
        q, k, v = a.q_proj(x), a.k_proj(x), a.v_proj(x)
        q, k, v = [t.view(t.shape[0], t.shape[1], a.num_heads, a.head_dim) for t in (q, k, v)]
        # upstream since 06/2025: SDPA on (B, L, H, D) = attention over the 4 heads
        # (sequence length 4, head dim 7: the math backend is 4x faster than flash here, 1.9 vs 8.2 ms)
        with sdpa_kernel(SDPBackend.MATH):
            o = F.scaled_dot_product_attention(q, k, v)
        o = a.out_proj(o.reshape(x.shape))
        x = layer.norm1(x + o)
        x = layer.norm2(x + layer.linear2(layer.act(layer.linear1(x))))
    x = x.reshape(B, H, W, D, C)
    return x.permute(0, 4, 1, 2, 3) if ctx.hwd else x.permute(0, 4, 3, 1, 2)


def classifier_strips(model, comb, ctx):
    """init_disp = regression(softmax(classifier(comb))) strip by strip (classifier radius 6 rows)."""
    B, C, D, H, W = ctx.dims(comb)
    disp = torch.empty((B, 1, H, W), device=comb.device, dtype=torch.float32)
    dv = torch.arange(D, device=comb.device, dtype=torch.float32).view(1, D, 1, 1)
    halo = 6
    for h0, h1 in _strips(H, _rows_for(model, B, D, W)):
        a, b = max(0, h0 - halo), min(H, h1 + halo)
        y = model.classifier[1](model.classifier[0](ctx.act(ctx.rows(comb, a, b))))
        c = _conv_k7_to1(model, model.classifier[2], y)
        if ctx.hwd:     # c: (B, h, W, D)
            p = F.softmax(c[:, h0 - a:h1 - a], dim=-1)
            disp[:, 0, h0:h1] = (p * dv.view(1, 1, 1, D)).sum(-1)
        else:           # c: (B, D, h, W)
            p = F.softmax(c[:, :, h0 - a:h1 - a], dim=1)
            disp[:, :, h0:h1] = (p * dv).sum(1, keepdim=True)
    return disp


def _conv_k7_to1(model, conv, x):
    """Conv3d(C->1, k=7, pad 3) as Conv2d (7x7) over the batch of the first spatial axis (D, or H in the HWD layout)
    with output channel = tap of that axis (padded to 8), then shift-add along it in fp32. cuDNN runs the 3D
    1-output-channel conv at ~0.3 TFLOPS (294 ms at Midd-H). Returns (B, S0, S1, S2) of the input's spatial axes."""
    w2 = getattr(conv, "_w2", None)
    k = conv.kernel_size[0]
    if w2 is None:
        w = conv.weight.detach()                                             # (1, C, kd, kh, kw)
        w2 = torch.zeros((8, w.shape[1], k, k), device=w.device, dtype=w.dtype)
        w2[:k] = w[0].permute(1, 0, 2, 3)
        conv._w2 = w2 = w2.contiguous(memory_format=torch.channels_last)
    B, C, D, H, W = x.shape          # (generic spatial axes S0, S1, S2)
    out = conv.bias.float().view(1, 1, 1, 1).expand(B, D, H, W).contiguous()
    for bi in range(B):
        xb = x[bi].permute(1, 0, 2, 3)                                       # (D, C, H, W)
        if model._opts.hacc and model._opts.half:
            from core import cudnn_conv
            y = cudnn_conv.conv_half_acc(xb, w2, None, [k // 2, k // 2], [1, 1])
        else:
            y = F.conv2d(xb.contiguous(memory_format=torch.channels_last), w2.to(xb.dtype), None, 1, k // 2)
        for kd in range(k):                                                  # out[d] += y[d + kd - 3, kd]
            sft = kd - k // 2
            lo, hi = max(0, -sft), min(D, D - sft)
            out[bi, lo:hi] += y[lo + sft:hi + sft, kd].float()
    return out


# ------------------------------------------------------------------------------------------------ geometry
class GeoFast:
    """== Combined_Geo_Encoding_Volume (+ grid_sample lookups), sync-free gather lookups, fp16 storage."""

    def __init__(self, fl, fr, comb, levels, dx):
        B, C, D, H, W = comb.shape
        self.B, self.C, self.H, self.W, self.levels, self.dx = B, C, H, W, levels, dx
        g = comb.permute(0, 2, 3, 4, 1)          # (B, D, H, W, C); a view if comb is channels_last_3d
        self.geo = [g.contiguous()]
        for _ in range(levels - 1):
            g = self.geo[-1]
            Dn = g.shape[1] // 2
            self.geo.append((0.5 * (g[:, 0:2 * Dn:2] + g[:, 1:2 * Dn:2])).contiguous())
        f1 = F.normalize(fl.float(), dim=1).to(comb.dtype)
        f2 = F.normalize(fr.float(), dim=1).to(comb.dtype)
        f1 = f1.permute(0, 2, 3, 1).reshape(B * H, W, -1)
        f2 = f2.permute(0, 2, 3, 1).reshape(B * H, W, -1)
        corr = torch.bmm(f1, f2.transpose(1, 2)).view(B, H, W, W)   # (B,H,W1,W2)
        self.corr = [corr]
        for _ in range(levels - 1):
            c = self.corr[-1]
            Wn = c.shape[-1] // 2
            self.corr.append((0.5 * (c[..., 0:2 * Wn:2] + c[..., 1:2 * Wn:2])).contiguous())
        self.wpos = torch.arange(W, device=comb.device, dtype=torch.float32).view(1, 1, W, 1)
        hw = torch.arange(H * W, device=comb.device).view(1, H, W, 1)
        self.bhw = (torch.arange(B, device=comb.device).view(B, 1, 1, 1) * H * W + hw)  # (B,H,W,1)

    @staticmethod
    def _lerp_gather(flat, rows_base, stride, pos, n):
        """flat: (N, F) rows; sample index pos (float, (B,H,W,K)) along an axis of length n with row = base + i*stride."""
        x0 = torch.floor(pos)
        f = (pos - x0)
        i0 = x0.long()
        i1 = i0 + 1
        v0 = ((i0 >= 0) & (i0 <= n - 1)).to(pos.dtype)
        v1 = ((i1 >= 0) & (i1 <= n - 1)).to(pos.dtype)
        w0 = (1 - f) * v0
        w1 = f * v1
        r0 = rows_base + i0.clamp(0, n - 1) * stride
        r1 = rows_base + i1.clamp(0, n - 1) * stride
        return r0, r1, w0, w1

    def __call__(self, disp):
        """disp (B,1,H,W) fp32 -> (B, 522, H, W) channels_last, channel order [geo0 (k,c) | corr0 | geo1 | corr1]."""
        B, H, W, C = self.B, self.H, self.W, self.C
        d = disp.view(B, H, W, 1)
        outs = []
        for i in range(self.levels):
            s = 2 ** i
            g = self.geo[i]                                  # (B, Di, H, W, C)
            Di = g.shape[1]
            pos = d / s + self.dx                            # (B,H,W,K)
            # row of (b, d, h, w) in g.view(-1, C): (b*Di + d)*H*W + h*W + w
            base = (self.bhw // (H * W)) * Di * H * W + (self.bhw % (H * W))
            r0, r1, w0, w1 = self._lerp_gather(None, base, H * W, pos, Di)
            gf = g.view(-1, C)
            dt = g.dtype
            val = gf[r0] * w0.unsqueeze(-1).to(dt) + gf[r1] * w1.unsqueeze(-1).to(dt)   # (B,H,W,K,C)
            outs.append(val.view(B, H, W, -1))
            c = self.corr[i]                                 # (B,H,W,W2i)
            W2 = c.shape[-1]
            pos = self.wpos / s - d / s + self.dx
            base = self.bhw * W2
            r0, r1, w0, w1 = self._lerp_gather(None, base, 1, pos, W2)
            cf = c.view(-1)
            outs.append(cf[r0] * w0.to(dt) + cf[r1] * w1.to(dt))
        out = torch.cat(outs, dim=-1)                        # (B,H,W,522)
        return out.permute(0, 3, 1, 2)                       # channels_last view


class GeoKernel:
    """Same lookup as GeoFast via the geo_lookup_s kernel (core/fs_kernels.cu): volume = the NDHWC hourglass output
    (no copy), all-pairs correlation (P, W) fp16; pyramid level 1 averaged on the fly; upstream channel order."""

    def __init__(self, fl, fr, comb, levels, radius, hwd=False):
        assert levels == 2
        assert comb.is_contiguous(memory_format=torch.channels_last_3d)
        self.hwd = hwd
        if hwd:      # (B, C, H, W, D): memory (P, D, C)
            B, C, H, W, D = comb.shape
        else:        # (B, C, D, H, W): memory (B, D, HW, C)
            B, C, D, H, W = comb.shape
        self.g0 = comb
        self.B, self.C, self.D, self.H, self.W, self.r = B, C, D, H, W, radius
        f1 = F.normalize(fl.float(), dim=1).to(comb.dtype).permute(0, 2, 3, 1).reshape(B * H, W, -1)
        f2 = F.normalize(fr.float(), dim=1).to(comb.dtype).permute(0, 2, 3, 1).reshape(B * H, W, -1)
        self.c0 = torch.bmm(f1, f2.transpose(1, 2))         # (B*H, W, W) = (P, W)
        K = 2 * radius + 1
        self.nout = 2 * (C + 1) * K

    def __call__(self, disp, rows=None):
        """rows=(h0, h1) (batch 1, hwd): lookup for these image rows only, into a band buffer (1, h1-h0, W, npad)."""
        B, H, W = self.B, self.H, self.W
        P = B * H * W
        k = _kern()
        d = disp.contiguous()
        if rows is not None:
            assert self.hwd and B == 1
            h0, h1 = rows
            npad = (self.nout + 7) // 8 * 8
            if getattr(self, "_band", None) is None or self._band.shape[1] < h1 - h0:
                self._band = torch.zeros((1, h1 - h0, W, npad), device=disp.device, dtype=self.g0.dtype)
            out = self._band[:, :h1 - h0]
            p0, Pb, es = h0 * W, (h1 - h0) * W, self.g0.element_size()
            k.launch("geo_lookup_hwd", k.linear(Pb * 2 * (self.C + 1)), self.g0.data_ptr() + p0 * self.D * self.C * es,
                     self.C, self.D, self.c0.data_ptr() + p0 * W * es, W, W, d.data_ptr() + p0 * d.element_size(), out,
                     Pb, self.r, npad)
            return out.permute(0, 3, 1, 2)
        out = torch.empty((B, H, W, self.nout), device=disp.device, dtype=self.g0.dtype)
        if self.hwd:
            # output channels padded to a multiple of 8 (zeros; convc1 has zero weights there): no cuDNN pad copy
            npad = (self.nout + 7) // 8 * 8
            if getattr(self, "_buf", None) is None:
                self._buf = torch.zeros((B, H, W, npad), device=disp.device, dtype=self.g0.dtype)
            out = self._buf
            k.launch("geo_lookup_hwd", k.linear(P * 2 * (self.C + 1)), self.g0, self.C, self.D, self.c0, W, W, d, out, P, self.r, npad)
        else:
            k.launch("geo_lookup_s", k.linear(P * 2 * (self.C + 1)), self.g0, self.C, self.D, H * W, self.c0, W, W, d, out, P, self.r)
        return out.permute(0, 3, 1, 2)


# ------------------------------------------------------------------------------------------------ GRU loop
def _interp(x, dest):
    return F.interpolate(x, dest.shape[2:], mode="bilinear", align_corners=True)


def _sel_gru(g, att, h, *x):
    x = torch.cat(x, dim=1)
    if hasattr(g.small_gru, "convzr"):
        x = conv_ep(g.conv0[0], x, ACT["relu"])
        hx = conv_ep(g.conv1[0], torch.cat([x, h], dim=1), ACT["relu"])
    else:
        x = g.conv0(x)
        hx = g.conv1(torch.cat([x, h], dim=1))
    hs = _raft_gru(g.small_gru, h, x, hx)
    hl = _raft_gru(g.large_gru, h, x, hx)
    return torch.lerp(hl, hs, att)      # == hs * att + hl * (1 - att)


def _raft_gru(g, h, x, hx):
    if hasattr(g, "convzr"):
        zr = conv_ep(g.convzr, hx, ACT["sigmoid"])
        C = h.shape[1]
        z, r = zr[:, :C], zr[:, C:]
        q = conv_ep(g.convq, torch.cat([r * h, x], dim=1), ACT["tanh"])
    else:
        z = torch.sigmoid(g.convz(hx))
        r = torch.sigmoid(g.convr(hx))
        q = torch.tanh(g.convq(torch.cat([r * h, x], dim=1)))
    return torch.lerp(h, q, z)          # == (1-z)*h + z*q


def update_fast(ub, net, inp, corr, disp, att, dt):
    net[2] = _sel_gru(ub.gru16, att[2], net[2], inp[2], update.pool2x(net[1]))
    net[1] = _sel_gru(ub.gru08, att[1], net[1], inp[1], update.pool2x(net[0]), _interp(net[2], net[1]))
    e = ub.encoder
    d16 = disp.to(dt).contiguous(memory_format=torch.channels_last)   # (C=1: else the cat below may come out NCHW)
    if hasattr(e.convc1, "_b32"):
        R = ACT["relu"]
        cor = conv_ep(e.convc2, conv_ep(e.convc1, corr.contiguous(memory_format=torch.channels_last), R), R)
        dsp = conv_ep(e.convd2, conv_ep(e.convd1, d16.contiguous(memory_format=torch.channels_last), R), R)
        out = conv_ep(e.conv, torch.cat([cor, dsp], dim=1), R)
    else:
        cor = F.relu(e.convc2(F.relu(e.convc1(corr))))
        dsp = F.relu(e.convd2(F.relu(e.convd1(d16))))
        out = F.relu(e.conv(torch.cat([cor, dsp], dim=1)))
    mf = torch.cat([inp[0], out, d16], dim=1)
    net[0] = _sel_gru(ub.gru04, att[0], net[0], mf, _interp(net[1], net[0]))
    return net, ub.disp_head(net[0])


MLP_KERNEL = True   # DispHead EdgeNeXt MLP as one tensor-core kernel (mlp_gelu_res in core/fs_kernels.cu)


class FusedGRU:
    """The 3-level selective ConvGRU update without concatenations (Opts.fused): every GRU conv runs as a CUTLASS implicit
    GEMM with its epilogue fused (core/cutlass_conv.cu, compiled once; any size), producers write straight into the
    channel slices of the next conv's dense NHWC input buffer, and the state h lives inside the conv1 input [x | h]:
      X0  = [...]                (pooled / interpolated neighbour states, motion features written into their slices by
                                  pool2x_into / interp_into / the motion conv epilogue; conv0's constant inp channels
                                  enter as a per-pixel epilogue term conv(inp, W0[:, :128]) computed once (SPLIT))
      conv0(X0) -> relu -> x into XH = [x | h] and into RHX = [r*h | x]      (dual output)
      conv1(XH) -> relu -> HX;  zr(HX) -> z (Z), sigmoid(r) * h into RHX[:, :128]
      small q(RHX) -> hs = h + z (q - h);  large zr / q -> hl, h = lerp(hl, hs, att) in place + contiguous copy NET
    Same math as _sel_gru / _raft_gru (fp16 storage of z, q, hs, hl as there; fp16 accumulation like the hacc convs)."""
    CFG = 0   # CUTLASS tile 128x128x32 / 3 stages for every shape (measured best)
    SPLIT = True   # conv0's constant inp channels: their contribution computed once per forward (per-pixel epilogue term)

    def __init__(self, model, net, inp, att):
        from core.cutlass_conv import FusedConvs, EpiArgs
        self.fc, self.E = FusedConvs.get(), EpiArgs
        ub = model.update_block
        self.ub = ub
        self.grus = [ub.gru04, ub.gru08, ub.gru16]
        dev, dt = net[0].device, net[0].dtype
        self.hw = [(t.shape[2], t.shape[3]) for t in net]
        P = [h * w for h, w in self.hw]
        self.cx = [g.conv0[0].out_channels for g in self.grus]
        cmax = max(P[i] * (self.cx[i] + 128) for i in range(3))
        self.HX = torch.empty(cmax, device=dev, dtype=dt)
        self.RHX = torch.empty(cmax, device=dev, dtype=dt)
        self.Z = torch.empty(P[0] * 128, device=dev, dtype=dt)
        self.HSM = torch.empty(P[0] * 128, device=dev, dtype=dt)
        self.X0, self.XH, self.NET, self.att, self.C0 = [], [], [], [], []
        self.r0 = 0 if self.SPLIT else 128   # first X0 column after the inp channels
        for i in range(3):
            cx = self.cx[i]
            x0 = torch.empty(P[i], cx - 128 + self.r0, device=dev, dtype=dt)
            if self.SPLIT:
                from core.cudnn_conv import conv_half_acc
                c0 = self.grus[i].conv0[0]
                if not hasattr(c0, "_w_inp"):
                    c0._w_inp = c0.weight.detach()[:, :128].contiguous(memory_format=torch.channels_last)
                    c0._w_rest = c0.weight.detach()[:, 128:].contiguous(memory_format=torch.channels_last)
                ci = conv_half_acc(inp[i].contiguous(memory_format=torch.channels_last), c0._w_inp, None,
                                   list(c0.padding), [1, 1])
                self.C0.append(ci.contiguous(memory_format=torch.channels_last).permute(0, 2, 3, 1).reshape(P[i], cx))
            else:
                x0[:, :128] = inp[i][0].permute(1, 2, 0).reshape(P[i], 128)
                self.C0.append(None)
            xh = torch.empty(P[i], cx + 128, device=dev, dtype=dt)
            nt = net[i][0].permute(1, 2, 0).reshape(P[i], 128).contiguous()
            xh[:, cx:] = nt
            self.X0.append(x0); self.XH.append(xh); self.NET.append(nt)
            self.att.append(att[i].contiguous())
        # motion encoder: [cor (256) | dsp (64)] input of e.conv; e.conv padded to 128 outputs (last = disp channel)
        e = ub.encoder
        self.MOT = torch.empty(P[0], 320, device=dev, dtype=dt)
        if not hasattr(e.conv, "_w128"):
            w = torch.zeros((128,) + tuple(e.conv.weight.shape[1:]), device=dev, dtype=dt)
            w[:e.conv.out_channels] = e.conv.weight.data
            e.conv._w128 = w.contiguous(memory_format=torch.channels_last)
            b = torch.zeros(128, device=dev)
            b[:e.conv.out_channels] = e.conv._b32
            e.conv._b128 = b
        self._bind()

    def _conv(self, x, m, H, W, C, w=None, **kw):
        e = self.E(**{k: (v.data_ptr() if isinstance(v, torch.Tensor) else v) for k, v in kw.items()})
        w = m.weight if w is None else w
        return self.fc.bind(self.CFG, x, w, H, W, C, w.shape[0], e)

    def _bind(self):
        self.calls = []
        for i, g in enumerate(self.grus):
            H, W = self.hw[i]
            P, cx = H * W, self.cx[i]
            ct = cx + 128
            x0, xh, nt = self.X0[i], self.XH[i], self.NET[i]
            hx = self.HX[:P * ct].view(P, ct)
            rhx = self.RHX[:P * ct].view(P, ct)
            z, hsm = self.Z[:P * 128], self.HSM[:P * 128]
            h = xh[:, cx:]
            if self.SPLIT:
                c = [self._conv(x0, g.conv0[0], H, W, cx - 128, w=g.conv0[0]._w_rest, mode=2, act=1, b=g.conv0[0]._b32,
                                c=self.C0[i], cs=cx, o=xh, os=ct, o2=rhx[:, 128:], o2s=ct)]
            else:
                c = [self._conv(x0, g.conv0[0], H, W, cx, mode=2, act=1, b=g.conv0[0]._b32, o=xh, os=ct, o2=rhx[:, 128:], o2s=ct)]
            c += [
                 self._conv(xh, g.conv1[0], H, W, ct, mode=2, act=1, b=g.conv1[0]._b32, o=hx, os=ct)]
            for gru, last in ((g.small_gru, False), (g.large_gru, True)):
                c.append(self._conv(hx, gru.convzr, H, W, ct, mode=0, b=gru.convzr._b32, o=z, os=128, o2=rhx, o2s=ct, h=h, hs=ct))
                if last:
                    c.append(self._conv(rhx, gru.convq, H, W, ct, mode=3, b=gru.convq._b32, o=h, os=ct, o2=nt, o2s=128,
                                        h=h, hs=ct, z=z, zs=128, g=hsm, gs=128, att=self.att[i]))
                else:
                    c.append(self._conv(rhx, gru.convq, H, W, ct, mode=1, b=gru.convq._b32, o=hsm, os=128, h=h, hs=ct, z=z, zs=128))
            self.calls.append(c)

    def _gru(self, i):
        for c in self.calls[i]:
            c()

    def _into(self, name, src, i_src, dst, c0, i_dst):
        k = _kern()
        Hs, Ws = self.hw[i_src]
        Hd, Wd = self.hw[i_dst]
        k.launch(name, k.linear(Hd * Wd * 16), src, 128, Hs, Ws, dst[:, c0:], dst.shape[1], Hd, Wd, 128)

    BAND_MB = 192.0   # geometry lookup (528 ch) + convc1 in row bands above this lookup-buffer size (12 MP: 782 MB)

    def motion_corr(self, geo, disp):
        """convc1(relu) of the geometry lookup; in row bands (band lookup buffer, output rows copied into one (P, 256)
        map) when the full lookup would exceed BAND_MB. One band (e.g. Midd-H): exactly the unbanded calls."""
        e = self.ub.encoder
        H, W = self.hw[0]
        rows = max(1, int(self.BAND_MB * 2**20 // (W * ((geo.nout + 7) // 8 * 8) * 2)))
        if rows >= H or not geo.hwd:
            return conv_ep(e.convc1, geo(disp), ACT["relu"])
        cor = None
        for h0, h1 in _strips(H, rows):
            c = conv_ep(e.convc1, geo(disp, rows=(h0, h1)), ACT["relu"])
            if cor is None:
                cor = torch.empty((1, c.shape[1], H, W), device=c.device, dtype=c.dtype, memory_format=torch.channels_last)
            cor[:, :, h0:h1] = c
            del c
        return cor

    def step(self, corr, disp, dt):
        X0, NET = self.X0, self.NET
        r0 = self.r0
        self._into("pool2x_into", NET[1], 1, X0[2], r0, 2)
        self._gru(2)
        self._into("pool2x_into", NET[0], 0, X0[1], r0, 1)
        self._into("interp_into", NET[2], 2, X0[1], r0 + 128, 1)
        self._gru(1)
        e = self.ub.encoder
        R = ACT["relu"]
        H, W = self.hw[0]
        # corr: the lookup (B, 528, H, W), or the GeoKernel itself (lookup + convc1 in row bands, motion_corr)
        cor = self.motion_corr(corr, disp) if isinstance(corr, GeoKernel) else conv_ep(e.convc1, corr, R)  # 1x1: cuDNN
        self._run_c2(cor)
        del cor
        d16 = disp.to(dt).contiguous(memory_format=torch.channels_last)
        dsp = conv_ep(e.convd2, conv_ep(e.convd1, d16, R), R)
        self.MOT[:, 256:].view(1, H, W, 64).copy_(dsp.permute(0, 2, 3, 1))
        dsp32 = disp.contiguous()
        ea = self.E(mode=2, act=1, b=e.conv._b128.data_ptr(), o=X0[0][:, r0:].data_ptr(), os=X0[0].shape[1],
                    disp=dsp32.data_ptr())
        rc = self.fc.bind(self.CFG, self.MOT, e.conv._w128, H, W, 320, 128, ea).rc()
        assert rc == 0, rc
        self._into("interp_into", NET[1], 1, X0[0], r0 + 128, 0)
        self._gru(0)
        return self._disp_head()

    def _disp_head(self):
        """DispHead: conv3x3 + ReLU, 2 x EdgeNeXt block (dw7x7 -> Linear + GELU -> Linear * gamma + input), conv3x3 -> 1.
        Linears as fp16-acc cuDNN 1x1 convs on the NHWC map (no permutes), bias / GELU / gamma / residual in one
        epilogue pass each (CUTLASS is epilogue-bound at these small channel counts)."""
        from core.cudnn_conv import conv_half_acc
        hd = self.ub.disp_head.conv
        x = conv_ep(hd[0], self.net0(), ACT["relu"])
        for blk in (hd[2], hd[3]):
            if not hasattr(blk, "_w1"):
                C, E = blk.pwconv1.in_features, blk.pwconv1.out_features
                blk._w1 = blk.pwconv1.weight.detach().view(E, C, 1, 1).contiguous(memory_format=torch.channels_last)
                blk._w2 = blk.pwconv2.weight.detach().view(C, E, 1, 1).contiguous(memory_format=torch.channels_last)
                blk._b1 = blk.pwconv1.bias.detach().float().contiguous()
                g = blk.gamma.detach().float() if blk.gamma is not None else torch.ones(C, device=x.device)
                blk._g = g.contiguous()
                blk._t2 = (blk.pwconv2.bias.detach().float() * g).contiguous()
            B, C, H, W = x.shape
            P = H * W
            if MLP_KERNEL and B == 1 and (C, blk._w1.shape[0]) == (128, 512):
                # dw conv into a 128-row padded buffer, then the whole MLP + layer scale + residual in one kernel
                dbuf = torch.empty(((P + 127) // 128 * 128, C), device=x.device, dtype=x.dtype)
                d = dbuf[:P].view(1, H, W, C).permute(0, 3, 1, 2)
                self._dwconv(blk.dwconv, x, out=d)
                y = torch.empty_like(x)
                k = _kern()
                k.launch("mlp_gelu_res", (((P + 127) // 128, 1, 1), (128, 1, 1)), dbuf, blk.pwconv1.weight, blk._b1,
                         blk.pwconv2.weight, blk._g, blk._t2, x, y, P)
                x = y
                continue
            d = self._dwconv(blk.dwconv, x)
            y = epilogue(conv_half_acc(d, blk._w1, None, [0, 0], [1, 1]), blk._b1, ACT["gelu"])
            x = epilogue(conv_half_acc(y, blk._w2, None, [0, 0], [1, 1]), blk._t2, 0, r=x, s=blk._g)
        return hd[4](x)

    @staticmethod
    def _dwconv(dw, x, out=None):
        """depthwise 7x7 conv + bias (NHWC fp16, C % 8 == 0) with the tiled kernel of core/fs_kernels.cu."""
        B, C, H, W = x.shape
        if dw.kernel_size != (7, 7) or B != 1 or C % 8 or not x.is_contiguous(memory_format=torch.channels_last):
            y = dw(x).contiguous(memory_format=torch.channels_last)
            return y if out is None else out.copy_(y)
        if not hasattr(dw, "_wt"):
            dw._wt = dw.weight.detach().float().view(-1, 49).t().contiguous()   # [(ky, kx), C]
            dw._bf = dw.bias.detach().float().contiguous()
        d = torch.empty_like(x) if out is None else out
        k = _kern()
        k.launch("dwconv7_nhwc", (((W + 31) // 32, (H + 15) // 16, C // 8), (8, 16, 1)), x, dw._wt, dw._bf, d, H, W, C)
        return d

    def _run_c2(self, cor):
        e = self.ub.encoder
        H, W = self.hw[0]
        assert cor.is_contiguous(memory_format=torch.channels_last)
        ea = self.E(mode=2, act=1, b=e.convc2._b32.data_ptr(), o=self.MOT.data_ptr(), os=320)
        rc = self.fc.bind(self.CFG, cor, e.convc2.weight, H, W, 256, 256, ea).rc()
        assert rc == 0, rc

    def net0(self):
        H, W = self.hw[0]
        return self.NET[0].view(1, H, W, 128).permute(0, 3, 1, 2)


def upsample_fast(model, disp, net0, stem_2x):
    mask = 0.25 * model.update_block.mask(net0)
    xspx = model.spx_2_gru(mask, stem_2x)
    spx = F.softmax(model.spx_gru(xspx).float(), 1)
    return submodule.context_upsample(disp * 4.0, spx).unsqueeze(1)


# ------------------------------------------------------------------------------------------------ forward
@torch.no_grad()
def forward_fast(model, image1, image2, iters=32):
    o = model._opts
    prev = torch.backends.cuda.matmul.allow_fp16_accumulation
    torch.backends.cuda.matmul.allow_fp16_accumulation = prev or o.gacc
    try:
        return _forward(model, image1, image2, iters)
    finally:
        torch.backends.cuda.matmul.allow_fp16_accumulation = prev


def _forward(model, image1, image2, iters):
    ctx = _Ctx(model)
    o = ctx.o
    B = image1.shape[0]
    i1, i2 = normalize_image(image1), normalize_image(image2)
    with ctx.amp():
        if o.seq_feat:
            fl, vit = feature_fast(model.feature, i1, ctx)
            fr = feature_fast(model.feature, i2, ctx)[0]   # (its ViT map unused: not kept alive)
            del i2
        else:
            out, vit = feature_fast(model.feature, torch.cat([i1, i2]), ctx)
            fl, fr = [t[:B] for t in out], [t[B:] for t in out]
            vit = vit[:B]
        stem_2x = model.stem_2(ctx.act(i1))
        D = model.args.max_disp // 4
        comb = corr_stem_strips(model, fl[0], fr[0], D, ctx)
        hold = [comb]
        del comb     # hourglass_fast frees the corr_stem volume after its last use
        comb = hourglass_fast(model.cost_agg, hold, fl, model, ctx)
        disp = classifier_strips(model, comb, ctx)
        if o.kern:
            cnet = cnet_fast(model.cnet, ctx.act(i1), ctx.act(vit), ctx)
        else:
            cnet = model.cnet(ctx.act(i1), vit_feat=ctx.act(vit), num_layers=model.args.n_gru_layers)
        del i1, vit, image1, image2   # (the caller's input tensors stay alive in the caller)
        net = [torch.tanh(x[0]) for x in cnet]
        inp = [torch.relu(x[1]) for x in cnet]
        inp = [model.cam(x) * x for x in inp]
        att = [model.sam(x) for x in inp]
        if o.kern:
            geo = GeoKernel(fl[0], fr[0], comb, model.args.corr_levels, model.args.corr_radius, hwd=ctx.hwd)
        else:
            geo = GeoFast(fl[0], fr[0], comb, model.args.corr_levels, model._dx)
        del comb, fl, fr, cnet
        if o.fused and B == 1:   # (FusedGRU buffers are per image)
            fg = FusedGRU(model, net, inp, att)
            del net, inp
            for itr in range(iters):
                corr = geo if isinstance(geo, GeoKernel) else geo(disp)
                delta = fg.step(corr, disp, ctx.dt)
                disp = disp + delta.float()
            net0 = fg.net0()
            del fg
        else:
            for itr in range(iters):
                corr = geo(disp)
                net, delta = update_fast(model.update_block, net, inp, corr, disp, att, ctx.dt)
                disp = disp + delta.float()
            net0 = net[0]
        geo = att = corr = None   # geometry volume / correlation / GRU buffers before the full-resolution upsampling
        up = upsample_fast(model, disp, net0, stem_2x)
    return up.float()
