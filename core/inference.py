"""Inference path for FoundationStereo in plain PyTorch (no custom kernels; exports to ONNX / TensorRT).

    from core.inference import prepare_inference, forward_fast, Opts
    prepare_inference(model, Opts())            # once: BN folded into conv weights, fp16 weights, NHWC
    disp = forward_fast(model, L, R, iters=32)  # L, R: (1,3,H,W) 0..255, padded to /32; == model.forward(test_mode=True)

Changes vs upstream forward (math-preserving unless noted):
  - pure fp16 weights / activations (no autocast re-casts each iteration); disparity state, softmax, regression and the
    final upsampling stay fp32; BatchNorm folded into the (fp32) conv weights before the fp16 cast
  - channels_last(_3d) everywhere; layout-preserving InstanceNorm
  - cost volume built and filtered (corr_stem, hourglass full-resolution head / tail, classifier) in H-strips with halos
    in eager mode: exact (zero padding only at the true borders), peak memory ~ one full-resolution 28-ch volume. While
    exporting (torch.export / ONNX) the same computations run on the whole volume (the runtime plans memory itself)
  - group-wise correlation by GEMM (eager) / shifted views (export) instead of the per-disparity slice-assignment loop
  - classifier's Conv3d(C->1, k7) as a 2D conv over the disparity axis + shift-add (cuDNN's 3D 1-output conv is slow)
  - geometry volume = hourglass output (fp16), pyramid by pairwise disparity pooling; all-pairs correlation fp16;
    lookup = gather + linear interpolation (== grid_sample align_corners=True, zero padding) without host syncs
    (upstream: torch.unique asserts = 4 syncs per iteration)
  - upsampling mask head only after the last iteration; features of the two images one after the other
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
    strip_mb: float = 192.0    # H-strip size of the full-resolution volume work (MB per 28-ch fp16 strip), eager only
    dino_amp: bool = False     # keep the ViT in fp32 weights + autocast
    seq_feat: bool = True      # features one image at a time


def _exporting():
    return torch.compiler.is_exporting() or torch.compiler.is_compiling() or torch.onnx.is_in_onnx_export()


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
    cc1 = model.update_block.encoder.convc1
    cc1.weight.data = cc1.weight.data[:, perm].contiguous()
    if o.half:
        model.half()
        if o.dino_amp:
            model.feature.dino.float()
    if o.cl:
        for m in model.modules():
            if isinstance(m, _CONV) and m.weight.dim() in (4, 5):
                m.weight.data = m.weight.data.contiguous(
                    memory_format=torch.channels_last if m.weight.dim() == 4 else torch.channels_last_3d)
    k7 = model.classifier[2]   # Conv3d(C->1, k7) as a 2D conv with the disparity taps as output channels (padded to 8)
    w = k7.weight.detach()
    w2 = torch.zeros((8, w.shape[1], 7, 7), device=w.device, dtype=w.dtype)
    w2[:7] = w[0].permute(1, 0, 2, 3)
    k7.register_buffer("_w2", w2.contiguous(memory_format=torch.channels_last), persistent=False)
    model.register_buffer("_dx", torch.linspace(-model.args.corr_radius, model.args.corr_radius, 2 * model.args.corr_radius + 1,
                                                device=next(model.parameters()).device), persistent=False)
    return dict(folded_bn=nf)


# ------------------------------------------------------------------------------------------------ helpers
def _mf(x):
    return torch.channels_last if x.dim() == 4 else torch.channels_last_3d


class _Ctx:
    def __init__(self, model):
        self.o = model._opts
        self.dt = torch.float16 if self.o.half else torch.float32
        self.export = _exporting()

    def amp(self):
        return torch.autocast("cuda", dtype=torch.float16, enabled=not self.o.half)

    def act(self, x):  # input activation layout/dtype
        x = x.to(self.dt)
        return x.contiguous(memory_format=_mf(x)) if self.o.cl else x

    def empty(self, shape, like):
        return torch.empty(shape, device=like.device, dtype=self.dt,
                           memory_format=torch.channels_last_3d if self.o.cl else torch.contiguous_format)


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


def _resize_hw(H, W, divider, max_hw=1344):
    """== get_resize_keep_aspect_ratio(H, W, divider, 1344, 1344) for landscape inputs (W >= H), in integer ops without
    branches on the size (symbolic shapes while exporting): W' = min(ceil_d(W), max), H' = ceil_d(ceil_d(H) * W' / ceil_d(W))."""
    Hr = (H + divider - 1) // divider * divider
    Wr = (W + divider - 1) // divider * divider
    Wn = torch.sym_min(Wr, max_hw)
    Hn = (Hr * Wn + Wr * divider - 1) // (Wr * divider) * divider
    return Hn, Wn


# ------------------------------------------------------------------------------------------------ features
def feature_fast(ft, x, ctx):
    """== Feature.forward for a batch of images (fp32 normalized input)."""
    B, C, H, W = x.shape
    divider = np.lcm(ft.patch_size, 16)
    if ctx.export:
        Hr, Wr = _resize_hw(H, W, int(divider))
    else:
        Hr, Wr = get_resize_keep_aspect_ratio(H, W, divider=divider, max_H=1344, max_W=1344)
    xin = F.interpolate(x, size=(Hr, Wr), mode="bicubic", align_corners=False)
    if ctx.o.dino_amp:
        with torch.autocast("cuda", dtype=torch.float16):
            vit = ft.dino(xin)["out"].to(ctx.dt)
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


# ------------------------------------------------------------------------------------------------ cost volume
def _gwc_concat_shifted(model, fl, fr, l12, r12, D, ctx):
    """gwc (8) + concat (12+12) volume (B, 32, D, H, W) from shifted views (export path): disparity d pairs pixel w with
    right pixel w-d, zero for w < d (== build_gwc_volume(normalized) / build_concat_volume)."""
    G = model.cv_group
    B, C, H, W = fl.shape
    f1 = F.normalize(fl.float().reshape(B, G, C // G, H, W), dim=2)
    f2 = F.pad(F.normalize(fr.float().reshape(B, G, C // G, H, W), dim=2), (D - 1, 0))
    r12p = F.pad(r12, (D - 1, 0))
    gwc = torch.stack([(f1 * f2[..., D - 1 - d:D - 1 - d + W]).sum(2) for d in range(D)], 2)          # (B, G, D, H, W)
    cr = torch.stack([r12p[..., D - 1 - d:D - 1 - d + W] for d in range(D)], 2)                    # (B, 12, D, H, W)
    cl = l12.unsqueeze(2).expand(-1, -1, D, -1, -1)
    vol = torch.cat([gwc.to(ctx.dt), cl.to(ctx.dt), cr.to(ctx.dt)], 1)
    return vol.contiguous(memory_format=torch.channels_last_3d) if ctx.o.cl else vol


def _build_volume_strip(model, fl, fr, l12, r12, a, b, D, ctx):
    """gwc (8) + concat (12+12) volume for rows [a, b): (B, 32, D, b-a, W), == build_gwc_volume / build_concat_volume.
    gwc: per row and group all-pairs products by one fp32 GEMM against the W-flipped right features; disparity d of
    pixel w is then element (w, W-1-w+d) = a strided (diagonal) view; entries with w < d are zero (as upstream)."""
    G = model.cv_group
    B, C, _, W = fl.shape
    vol = ctx.empty((B, 32, D, b - a, W), fl)
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
        V = M.as_strided((B * G * h, D, W), (W * W, 1, W - 1), M.storage_offset() + W - 1)
        V = torch.where(valid, V, 0.0).view(B, G, h, D, W)
        vol[:, :G, :, r0 - a:r1 - a] = V.permute(0, 1, 3, 2, 4)
        del M, V
    vol[:, G:G + 12] = l12[:, :, a:b].unsqueeze(2)
    rp = F.pad(r12[:, :, a:b], (D, 0))                                       # (B,12,h,W+D)
    win = rp.unfold(3, W, 1)                                                 # (B,12,h,D+1,W): window s = shift D-s
    win = win[:, :, :, 1:].flip(3)
    vol[:, G + 12:] = win.permute(0, 1, 3, 2, 4)
    return vol


def corr_stem_strips(model, fl, fr, D, ctx):
    """comb = corr_feature_att(corr_stem(volume), fl), strip by strip (corr_stem receptive radius 5 rows)."""
    l12 = model.proj_cmb(fl)
    r12 = model.proj_cmb(fr)
    att = torch.sigmoid(model.corr_feature_att.feat_att(fl)).unsqueeze(2)
    if ctx.export:
        return model.corr_stem(_gwc_concat_shifted(model, fl, fr, l12, r12, D, ctx)) * att
    B, _, H, W = fl.shape
    out = ctx.empty((B, 28, D, H, W), fl)   # (no full-volume .contiguous copy)
    halo = 5
    for h0, h1 in _strips(H, _rows_for(model, B, D, W)):
        a, b = max(0, h0 - halo), min(H, h1 + halo)
        v = model.corr_stem(_build_volume_strip(model, fl, fr, l12, r12, a, b, D, ctx))
        out[:, :, :, h0:h1] = v[:, :, :, h0 - a:h1 - a] * att[:, :, :, h0:h1]
        del v
    return out


def hourglass_fast(hg, xs, features, model, ctx):
    """== hourglass.forward; xs = [x] (consumed: freed after its last use). Full-resolution head / tail in strips."""
    x = xs.pop()
    B, C, D, H, W = x.shape
    first = hg.conv1[0]
    if ctx.export:
        c1 = first(x)
    else:
        # first stride-2 conv over the full-resolution volume in H-strips: cuDNN copies a 28-ch NDHWC input into a
        # channel-padded (32) buffer, i.e. a full-volume transient (+2.5 GB at Midd-F) if run in one call
        orows = max(4, _rows_for(model, B, D, W) // 2)
        c1 = None
        for o0 in range(0, H // 2, orows):
            o1 = min(H // 2, o0 + orows)
            a, b = max(0, 2 * o0 - 2), min(H, 2 * o1 + 2)
            y = first(x[:, :, :, a:b])[:, :, :, o0 - a // 2:o1 - a // 2]
            if c1 is None:
                c1 = torch.empty((B, y.shape[1], y.shape[2], H // 2, y.shape[4]), device=x.device, dtype=y.dtype,
                                 memory_format=_mf(y))
            c1[:, :, :, o0:o1] = y
            del y
    # conv_patch is the last use of the full-resolution input volume: run it now so that the volume is freed before the
    # H/8..H/32 section (independent computations, same results)
    xp = hg.conv_patch(x)
    del x
    conv1 = hg.conv1[1](c1)
    del c1
    conv1 = hg.feature_att_8(conv1, features[1])
    conv2 = hg.conv2(conv1)
    conv2 = hg.feature_att_16(conv2, features[2])
    conv3 = hg.conv3(conv2)
    conv3 = hg.feature_att_32(conv3, features[3])
    conv3_up = hg.conv3_up(conv3)
    conv2 = hg.agg_0(torch.cat((conv3_up, conv2), dim=1))
    conv2 = hg.feature_att_up_16(conv2, features[2])
    conv2_up = hg.conv2_up(conv2)
    conv1 = hg.agg_1(torch.cat((conv2_up, conv1), dim=1))
    conv1 = hg.feature_att_up_8(conv1, features[1])
    xp = transformer_fast(hg.atts["4"], xp, ctx)
    if ctx.export:
        y = hg.conv1_up(conv1) + F.interpolate(xp, scale_factor=4, mode="trilinear", align_corners=False)
        return hg.conv_out(y)
    out = ctx.empty((B, C, D, H, W), conv1)
    halo = 2  # conv_out: 2 x (1,3,3)
    H8, H16 = conv1.shape[3], xp.shape[3]
    for h0, h1 in _strips(H, _rows_for(model, B, D, W)):
        r0, r1 = max(0, h0 - halo), min(H, h1 + halo)
        # transposed conv (k4 s2 p1): rows [r0, r1) need conv1 rows [r0//2 - 1, (r1+1)//2 + 1)
        a, b = max(0, r0 // 2 - 1), min(H8, (r1 + 1) // 2 + 1)
        y = hg.conv1_up(conv1[:, :, :, a:b])[:, :, :, r0 - 2 * a:r1 - 2 * a]
        # trilinear x4 (align_corners=False): rows [r0, r1) need xp rows [r0//4 - 1, (r1-1)//4 + 2)
        a, b = max(0, r0 // 4 - 1), min(H16, (r1 - 1) // 4 + 2)
        up = F.interpolate(xp[:, :, :, a:b], scale_factor=4, mode="trilinear", align_corners=False)
        y[:, :C].add_(up[:, :, :, r0 - 4 * a:r1 - 4 * a])   # in place: keeps y's channels-last layout
        del up
        y = hg.conv_out(y)
        out[:, :, :, h0:h1] = y[:, :, :, h0 - r0:h1 - r0]
        del y
    return out


def transformer_fast(att, cv, ctx):
    """== CostVolumeDisparityAttention.forward (upstream since 06/2025: SDPA on (B, L, H, D) = attention over the 4 heads;
    sequence length 4, head dim 7: the math backend is 4x faster than flash here)."""
    B, C, D, H, W = cv.shape
    x = cv.permute(0, 3, 4, 2, 1).reshape(B * H * W, D, C)
    x = x + att.pos_embed0.pe.to(x.device, x.dtype)[:, :D]
    for layer in att.sa:
        a = layer.self_attn
        q, k, v = a.q_proj(x), a.k_proj(x), a.v_proj(x)
        q, k, v = [t.view(t.shape[0], t.shape[1], a.num_heads, a.head_dim) for t in (q, k, v)]
        with sdpa_kernel(SDPBackend.MATH):
            o = F.scaled_dot_product_attention(q, k, v)
        o = a.out_proj(o.reshape(x.shape))
        x = layer.norm1(x + o)
        x = layer.norm2(x + layer.linear2(layer.act(layer.linear1(x))))
    return x.reshape(B, H, W, D, C).permute(0, 4, 3, 1, 2)


def _conv_k7_to1(conv, x):
    """Conv3d(C->1, k=7, pad 3) as Conv2d (7x7) over the batch of disparities with output channel = disparity tap
    (padded to 8), then shift-add along the disparity axis in fp32. cuDNN runs the 3D 1-output-channel conv at
    ~0.3 TFLOPS (294 ms at Midd-H). x (B, C, D, H, W) -> (B, D, H, W) fp32."""
    k = conv.kernel_size[0]
    B, C, D, H, W = x.shape
    outs = []
    for bi in range(B):
        xb = x[bi].permute(1, 0, 2, 3)                                       # (D, C, H, W)
        y = F.conv2d(xb.contiguous(memory_format=torch.channels_last), conv._w2.to(xb.dtype), None, 1, k // 2)
        yp = F.pad(y[:, :k].float().permute(1, 0, 2, 3), (0, 0, 0, 0, k // 2, k // 2))   # (k, D + k - 1, H, W)
        o = conv.bias.float().view(1, 1, 1)
        for kd in range(k):                                                  # out[d] += y[d + kd - 3, kd]
            o = o + yp[kd, kd:kd + D]
        outs.append(o)
    return torch.stack(outs, 0)


def classifier_strips(model, comb, ctx):
    """init_disp = regression(softmax(classifier(comb))) strip by strip (classifier radius 6 rows)."""
    B, C, D, H, W = comb.shape
    dv = torch.arange(D, device=comb.device, dtype=torch.float32).view(1, D, 1, 1)
    if ctx.export:
        c = _conv_k7_to1(model.classifier[2], model.classifier[1](model.classifier[0](comb)))
        return (F.softmax(c, dim=1) * dv).sum(1, keepdim=True)
    disp = torch.empty((B, 1, H, W), device=comb.device, dtype=torch.float32)
    halo = 6
    for h0, h1 in _strips(H, _rows_for(model, B, D, W)):
        a, b = max(0, h0 - halo), min(H, h1 + halo)
        y = model.classifier[1](model.classifier[0](ctx.act(comb[:, :, :, a:b])))
        c = _conv_k7_to1(model.classifier[2], y)
        p = F.softmax(c[:, :, h0 - a:h1 - a], dim=1)
        disp[:, :, h0:h1] = (p * dv).sum(1, keepdim=True)
    return disp


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
        self.hw = torch.arange(H * W, device=comb.device).view(1, H, W, 1)
        self.b = torch.arange(B, device=comb.device).view(B, 1, 1, 1)
        self.bhw = self.b * H * W + self.hw  # (B,H,W,1)

    @staticmethod
    def _lerp_index(rows_base, stride, pos, n):
        """sample positions pos (float, (B,H,W,K)) along an axis of length n with row = base + i*stride."""
        x0 = torch.floor(pos)
        f = (pos - x0)
        i0 = x0.long()
        i1 = i0 + 1
        v0 = ((i0 >= 0) & (i0 <= n - 1)).to(pos.dtype)
        v1 = ((i1 >= 0) & (i1 <= n - 1)).to(pos.dtype)
        r0 = rows_base + i0.clamp(0, n - 1) * stride
        r1 = rows_base + i1.clamp(0, n - 1) * stride
        return r0, r1, (1 - f) * v0, f * v1

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
            base = self.b * (Di * H * W) + self.hw
            r0, r1, w0, w1 = self._lerp_index(base, H * W, pos, Di)
            gf = g.reshape(-1, C)
            dt = g.dtype
            val = gf[r0] * w0.unsqueeze(-1).to(dt) + gf[r1] * w1.unsqueeze(-1).to(dt)   # (B,H,W,K,C)
            outs.append(val.reshape(B, H, W, -1))
            c = self.corr[i]                                 # (B,H,W,W2i)
            W2 = c.shape[-1]
            pos = self.wpos / s - d / s + self.dx
            r0, r1, w0, w1 = self._lerp_index(self.bhw * W2, 1, pos, W2)
            cf = c.reshape(-1)
            outs.append(cf[r0] * w0.to(dt) + cf[r1] * w1.to(dt))
        out = torch.cat(outs, dim=-1)                        # (B,H,W,522)
        return out.permute(0, 3, 1, 2)                       # channels_last view


# ------------------------------------------------------------------------------------------------ GRU loop
def _interp(x, dest):
    return F.interpolate(x, dest.shape[2:], mode="bilinear", align_corners=True)


def _sel_gru(g, att, h, *x):
    x = g.conv0(torch.cat(x, dim=1))
    hx = g.conv1(torch.cat([x, h], dim=1))
    hs = _raft_gru(g.small_gru, h, x, hx)
    hl = _raft_gru(g.large_gru, h, x, hx)
    return torch.lerp(hl, hs, att)      # == hs * att + hl * (1 - att)


def _raft_gru(g, h, x, hx):
    z = torch.sigmoid(g.convz(hx))
    r = torch.sigmoid(g.convr(hx))
    q = torch.tanh(g.convq(torch.cat([r * h, x], dim=1)))
    return torch.lerp(h, q, z)          # == (1-z)*h + z*q


def update_fast(ub, net, inp, corr, disp, att, dt):
    net[2] = _sel_gru(ub.gru16, att[2], net[2], inp[2], update.pool2x(net[1]))
    net[1] = _sel_gru(ub.gru08, att[1], net[1], inp[1], update.pool2x(net[0]), _interp(net[2], net[1]))
    e = ub.encoder
    d16 = disp.to(dt).contiguous(memory_format=torch.channels_last)   # (C=1: else the cat below may come out NCHW)
    cor = F.relu(e.convc2(F.relu(e.convc1(corr))))
    dsp = F.relu(e.convd2(F.relu(e.convd1(d16))))
    out = F.relu(e.conv(torch.cat([cor, dsp], dim=1)))
    mf = torch.cat([inp[0], out, d16], dim=1)
    net[0] = _sel_gru(ub.gru04, att[0], net[0], mf, _interp(net[1], net[0]))
    return net, ub.disp_head(net[0])


def upsample_fast(model, disp, net0, stem_2x):
    mask = 0.25 * model.update_block.mask(net0)
    xspx = model.spx_2_gru(mask, stem_2x)
    spx = F.softmax(model.spx_gru(xspx).float(), 1)
    return submodule.context_upsample(disp * 4.0, spx).unsqueeze(1)


# ------------------------------------------------------------------------------------------------ forward
@torch.no_grad()
def forward_fast(model, image1, image2, iters=32):
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
        cnet = model.cnet(ctx.act(i1), vit_feat=ctx.act(vit), num_layers=model.args.n_gru_layers)
        del i1, vit
        net = [torch.tanh(x[0]) for x in cnet]
        inp = [torch.relu(x[1]) for x in cnet]
        inp = [model.cam(x) * x for x in inp]
        att = [model.sam(x) for x in inp]
        geo = GeoFast(fl[0], fr[0], comb, model.args.corr_levels, model._dx)
        del comb, fl, fr, cnet
        for itr in range(iters):
            corr = geo(disp)
            net, delta = update_fast(model.update_block, net, inp, corr, disp, att, ctx.dt)
            disp = disp + delta.float()
        geo = corr = None   # geometry volume / correlation before the full-resolution upsampling
        up = upsample_fast(model, disp, net[0], stem_2x)
    return up.float()
