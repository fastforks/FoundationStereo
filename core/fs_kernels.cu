// Small size-independent CUDA kernels for FoundationStereo inference (NHWC fp16). Compiled once to a cubin / fatbin
// (core/kernels.py), launched on torch's current stream.
#include <cuda_fp16.h>
#include <stdint.h>
#include <mma.h>

#define GRID_STRIDE(i, n) for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < (n); i += (long long)gridDim.x * blockDim.x)

__device__ __forceinline__ float act_f(float v, int mode) {
    switch (mode) {
        case 1: return fmaxf(v, 0.f);                      // relu
        case 2: return v >= 0.f ? v : 0.01f * v;           // LeakyReLU(0.01)
        case 3: return tanhf(v);
        case 4: return 1.f / (1.f + __expf(-v));
        case 5: return fminf(fmaxf(v, 0.f), 6.f);          // relu6
        case 6: return 0.5f * v * (1.f + erff(v * 0.70710678118654752f));   // GELU (erf, nn.GELU default)
        default: return v;
    }
}

// y[p, c] = act(y[p, c] * s[c] + t[c] (+ r[p, c]))   (NHWC rows; s / t / r may be null). In place.
extern "C" __global__ void affine_act(__half* y, int ystride, const float* __restrict__ s, const float* __restrict__ t,
                                      const __half* __restrict__ r, int rstride, long long P, int C, int mode) {
    GRID_STRIDE(i, P * C) {
        long long p = i / C;
        int c = (int)(i - p * C);
        float v = __half2float(y[p * ystride + c]);
        if (s) v *= s[c];
        if (t) v += t[c];
        if (r) v += __half2float(r[p * rstride + c]);
        y[p * ystride + c] = __float2half(act_f(v, mode));
    }
}

// 8 channels per thread, 16-byte accesses (C, strides multiple of 8, 16-byte aligned base pointers)
extern "C" __global__ void affine_act8(__half* y, int ystride, const float* __restrict__ s, const float* __restrict__ t,
                                       const __half* __restrict__ r, int rstride, long long P, int C, int mode) {
    const int CV = C / 8;
    GRID_STRIDE(i, P * CV) {
        long long p = i / CV;
        int c0 = (int)(i - p * CV) * 8;
        uint4 raw = *reinterpret_cast<const uint4*>(y + p * ystride + c0);
        __half* hv = reinterpret_cast<__half*>(&raw);
        uint4 rraw;
        __half* rv = reinterpret_cast<__half*>(&rraw);
        if (r) rraw = *reinterpret_cast<const uint4*>(r + p * rstride + c0);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            float v = __half2float(hv[k]);
            if (s) v *= s[c0 + k];
            if (t) v += t[c0 + k];
            if (r) v += __half2float(rv[k]);
            hv[k] = __float2half(act_f(v, mode));
        }
        *reinterpret_cast<uint4*>(y + p * ystride + c0) = raw;
    }
}

// Combined geometry encoding volume lookup (IGEV), 2 pyramid levels, both volumes, = grid_sample bilinear
// (align_corners=True, zero padding) along the disparity / x axis. Only level 0 is stored (fp16); level-1 values
// are upstream's avg_pool(1x2) computed on the fly in fp32: v1[j] = (v0[2j] + v0[2j+1]) / 2, j < D0/2.
// One thread per (pixel, row): row jj = l * (C + 1) + j; j < C: geometry channel j, j == C: correlation.
// out [P, 2*(C+1)*K] fp16, channel order per level: geo (c*K + k), then corr (k)   (== upstream's cat/permute)
__device__ __forceinline__ float lvl_val(const __half* v, int i, int n, int l) {
    if (l == 0) return (i >= 0 && i < n) ? __half2float(v[i]) : 0.f;
    return (i >= 0 && i < n / 2) ? 0.5f * (__half2float(v[2 * i]) + __half2float(v[2 * i + 1])) : 0.f;
}

extern "C" __global__ void geo_lookup(const __half* __restrict__ g0, int C, int D0, const __half* __restrict__ c0,
                                      int W0, int Wimg, const float* __restrict__ disp, __half* __restrict__ out,
                                      long long P, int r) {
    const int K = 2 * r + 1;
    const int rows = 2 * (C + 1);
    GRID_STRIDE(i, P * rows) {
        long long p = i / rows;
        int jj = (int)(i - p * rows);
        int l = jj / (C + 1), j = jj - l * (C + 1);
        float sc = l ? 0.5f : 1.f;
        float d = disp[p] * sc;
        __half* o = out + p * (2 * (C + 1) * K) + l * (C + 1) * K + j * K;
        const __half* v;
        int n;
        float base;
        if (j < C) { v = g0 + (p * C + j) * D0; n = D0; base = d; }
        else { v = c0 + p * W0; n = W0; base = (float)(p % Wimg) * sc - d; }
        for (int k = 0; k < K; ++k) {
            float x = base + (float)(k - r);
            float xf = floorf(x);
            int x0 = (int)xf;
            float w = x - xf;
            o[k] = __float2half(lvl_val(v, x0, n, l) * (1.f - w) + lvl_val(v, x0 + 1, n, l) * w);
        }
    }
}

// ConvGRU gates. zr [P, 2*Hd] = conv([h|x]) without bias; czr [P, 2*Hd] = context + conv biases.
// z = sigmoid(zr[:, :Hd] + czr[:, :Hd]) stored back into zr; r = sigmoid(...); rh[p, c] = r * h  (rh: NHWC slice)
extern "C" __global__ void gru_zr(__half* zr, const __half* __restrict__ czr, const __half* __restrict__ h, int hstride,
                                  __half* rh, int rhstride, long long P, int Hd) {
    GRID_STRIDE(i, P * Hd) {
        long long p = i / Hd;
        int c = (int)(i - p * Hd);
        long long b = p * 2 * Hd;
        float z = 1.f / (1.f + __expf(-(__half2float(zr[b + c]) + __half2float(czr[b + c]))));
        float rr = 1.f / (1.f + __expf(-(__half2float(zr[b + Hd + c]) + __half2float(czr[b + Hd + c]))));
        zr[b + c] = __float2half(z);
        rh[p * rhstride + c] = __float2half(rr * __half2float(h[p * hstride + c]));
    }
}

// h = (1 - z) * h + z * tanh(q + cq), in place on h
extern "C" __global__ void gru_update(const __half* __restrict__ q, const __half* __restrict__ cq,
                                      const __half* __restrict__ zr, __half* h, int hstride, long long P, int Hd) {
    GRID_STRIDE(i, P * Hd) {
        long long p = i / Hd;
        int c = (int)(i - p * Hd);
        float z = __half2float(zr[p * 2 * Hd + c]);
        float qq = tanhf(__half2float(q[i]) + __half2float(cq[i]));
        float hh = __half2float(h[p * hstride + c]);
        h[p * hstride + c] = __float2half((1.f - z) * hh + z * qq);
    }
}

// ---- (same formulas as PyTorch), NHWC slices, C multiple of 8
// avg_pool2d(kernel 3, stride 2, padding 1, count_include_pad) of a C-channel slice into another slice.
__device__ __forceinline__ void pool2x_body(const __half* __restrict__ src, int sstride, int Hs, int Ws, __half* dst,
                                            int dstride, __half* dst2, int Hd, int Wd, int C) {
    const int CV = C / 8;  // 8 channels per thread
    GRID_STRIDE(i, (long long)Hd * Wd * CV) {
        int c = (int)(i % CV) * 8;
        long long pd = i / CV;
        int y = (int)(pd / Wd), x = (int)(pd % Wd);
        float s[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int dy = 0; dy < 3; ++dy) {
            int sy = 2 * y - 1 + dy;
            if (sy < 0 || sy >= Hs) continue;
#pragma unroll
            for (int dx = 0; dx < 3; ++dx) {
                int sx = 2 * x - 1 + dx;
                if (sx < 0 || sx >= Ws) continue;
                uint4 v = *reinterpret_cast<const uint4*>(src + ((long long)sy * Ws + sx) * sstride + c);
                const __half* h = reinterpret_cast<const __half*>(&v);
#pragma unroll
                for (int k = 0; k < 8; ++k) s[k] += __half2float(h[k]);
            }
        }
        __align__(16) __half o[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) o[k] = __float2half(s[k] / 9.f);
        *reinterpret_cast<uint4*>(dst + pd * dstride + c) = *reinterpret_cast<const uint4*>(o);
        if (dst2) *reinterpret_cast<uint4*>(dst2 + pd * dstride + c) = *reinterpret_cast<const uint4*>(o);
    }
}
extern "C" __global__ void pool2x_into(const __half* __restrict__ src, int sstride, int Hs, int Ws, __half* dst,
                                       int dstride, int Hd, int Wd, int C) {
    pool2x_body(src, sstride, Hs, Ws, dst, dstride, nullptr, Hd, Wd, C);
}

// Bilinear upsampling with align_corners=True (PyTorch formula) of a C-channel slice into another slice.
__device__ __forceinline__ void interp_body(const __half* __restrict__ src, int sstride, int Hs, int Ws, __half* dst,
                                            int dstride, __half* dst2, int Hd, int Wd, int C) {
    float rh = Hd > 1 ? (float)(Hs - 1) / (float)(Hd - 1) : 0.f;
    float rw = Wd > 1 ? (float)(Ws - 1) / (float)(Wd - 1) : 0.f;
    const int CV = C / 8;  // 8 channels per thread
    GRID_STRIDE(i, (long long)Hd * Wd * CV) {
        int c = (int)(i % CV) * 8;
        long long pd = i / CV;
        int y = (int)(pd / Wd), x = (int)(pd % Wd);
        float h1r = rh * (float)y, w1r = rw * (float)x;
        int h1 = (int)h1r, w1 = (int)w1r;
        int h1p = h1 < Hs - 1 ? 1 : 0, w1p = w1 < Ws - 1 ? 1 : 0;
        float h1l = h1r - (float)h1, w1l = w1r - (float)w1;
        float h0l = 1.f - h1l, w0l = 1.f - w1l;
        uint4 a = *reinterpret_cast<const uint4*>(src + ((long long)h1 * Ws + w1) * sstride + c);
        uint4 b = *reinterpret_cast<const uint4*>(src + ((long long)h1 * Ws + w1 + w1p) * sstride + c);
        uint4 cc = *reinterpret_cast<const uint4*>(src + ((long long)(h1 + h1p) * Ws + w1) * sstride + c);
        uint4 d = *reinterpret_cast<const uint4*>(src + ((long long)(h1 + h1p) * Ws + w1 + w1p) * sstride + c);
        const __half *v00 = reinterpret_cast<const __half*>(&a), *v01 = reinterpret_cast<const __half*>(&b);
        const __half *v10 = reinterpret_cast<const __half*>(&cc), *v11 = reinterpret_cast<const __half*>(&d);
        __align__(16) __half o[8];
#pragma unroll
        for (int k = 0; k < 8; ++k)
            o[k] = __float2half(h0l * (w0l * __half2float(v00[k]) + w1l * __half2float(v01[k])) +
                                h1l * (w0l * __half2float(v10[k]) + w1l * __half2float(v11[k])));
        *reinterpret_cast<uint4*>(dst + pd * dstride + c) = *reinterpret_cast<const uint4*>(o);
        if (dst2) *reinterpret_cast<uint4*>(dst2 + pd * dstride + c) = *reinterpret_cast<const uint4*>(o);
    }
}
extern "C" __global__ void interp_into(const __half* __restrict__ src, int sstride, int Hs, int Ws, __half* dst,
                                       int dstride, int Hd, int Wd, int C) {
    interp_body(src, sstride, Hs, Ws, dst, dstride, nullptr, Hd, Wd, C);
}

// FoundationStereo: geo_lookup on an NDHWC geometry volume (B, D0, H*W, C) (= the hourglass output in
// channels_last_3d, no transposed copy). Same math / output order as geo_lookup. sd = H*W*C (stride of d).
__device__ __forceinline__ float lvl_val_s(const __half* v, long long sd, int i, int n, int l) {
    if (l == 0) return (i >= 0 && i < n) ? __half2float(v[i * sd]) : 0.f;
    return (i >= 0 && i < n / 2) ? 0.5f * (__half2float(v[2 * i * sd]) + __half2float(v[(2 * i + 1) * sd])) : 0.f;
}

extern "C" __global__ void geo_lookup_s(const __half* __restrict__ g0, int C, int D0, int HW, const __half* __restrict__ c0,
                                        int W0, int Wimg, const float* __restrict__ disp, __half* __restrict__ out,
                                        long long P, int r) {
    const int K = 2 * r + 1;
    const int rows = 2 * (C + 1);
    const long long sd = (long long)HW * C;
    GRID_STRIDE(i, P * rows) {
        long long p = i / rows;
        int jj = (int)(i - p * rows);
        int l = jj / (C + 1), j = jj - l * (C + 1);
        float sc = l ? 0.5f : 1.f;
        float d = disp[p] * sc;
        __half* o = out + p * (2 * (C + 1) * K) + l * (C + 1) * K + j * K;
        for (int k = 0; k < K; ++k) {
            float x = (j < C ? d : (float)(p % Wimg) * sc - d) + (float)(k - r);
            float xf = floorf(x);
            int x0 = (int)xf;
            float w = x - xf;
            float a, b;
            if (j < C) {
                long long b0 = p / HW, hw = p - b0 * HW;
                const __half* v = g0 + b0 * D0 * sd + hw * C + j;
                a = lvl_val_s(v, sd, x0, D0, l); b = lvl_val_s(v, sd, x0 + 1, D0, l);
            } else {
                const __half* v = c0 + p * W0;
                a = lvl_val(v, x0, W0, l); b = lvl_val(v, x0 + 1, W0, l);
            }
            o[k] = __float2half(a * (1.f - w) + b * w);
        }
    }
}

// FoundationStereo: geo_lookup on a (B, C, H, W, D) channels_last_3d volume = memory (P, D, C). Same math / order;
// output row stride ostride >= 2*(C+1)*K (channel padding to a multiple of 8 for the following conv).
extern "C" __global__ void geo_lookup_hwd(const __half* __restrict__ g0, int C, int D0, const __half* __restrict__ c0,
                                          int W0, int Wimg, const float* __restrict__ disp, __half* __restrict__ out,
                                          long long P, int r, int ostride) {
    const int K = 2 * r + 1;
    const int rows = 2 * (C + 1);
    GRID_STRIDE(i, P * rows) {
        long long p = i / rows;
        int jj = (int)(i - p * rows);
        int l = jj / (C + 1), j = jj - l * (C + 1);
        float sc = l ? 0.5f : 1.f;
        float d = disp[p] * sc;
        __half* o = out + p * ostride + l * (C + 1) * K + j * K;
        if (r == 4) {   // the K + 1 samples are shared by neighbouring taps (same fractional weight): load each once
            float x = (j < C ? d : (float)(p % Wimg) * sc - d) - 4.f;
            float xf = floorf(x);
            int x0 = (int)xf;
            float w = x - xf;
            float v[10];
            if (j < C) {
                const __half* g = g0 + p * (long long)D0 * C + j;
#pragma unroll
                for (int k = 0; k < 10; ++k) v[k] = lvl_val_s(g, C, x0 + k, D0, l);
            } else {
                const __half* g = c0 + p * W0;
#pragma unroll
                for (int k = 0; k < 10; ++k) v[k] = lvl_val(g, x0 + k, W0, l);
            }
#pragma unroll
            for (int k = 0; k < 9; ++k) o[k] = __float2half(v[k] * (1.f - w) + v[k + 1] * w);
            continue;
        }
        for (int k = 0; k < K; ++k) {
            float x = (j < C ? d : (float)(p % Wimg) * sc - d) + (float)(k - r);
            float xf = floorf(x);
            int x0 = (int)xf;
            float w = x - xf;
            float a, b;
            if (j < C) {
                const __half* v = g0 + p * (long long)D0 * C + j;
                a = lvl_val_s(v, C, x0, D0, l); b = lvl_val_s(v, C, x0 + 1, D0, l);
            } else {
                const __half* v = c0 + p * W0;
                a = lvl_val(v, x0, W0, l); b = lvl_val(v, x0 + 1, W0, l);
            }
            o[k] = __float2half(a * (1.f - w) + b * w);
        }
    }
}

// FoundationStereo: affine_act with 4 channels per thread (8-byte accesses) for C % 4 == 0 (28-channel volumes)
extern "C" __global__ void affine_act4(__half* y, int ystride, const float* __restrict__ s, const float* __restrict__ t,
                                       const __half* __restrict__ r, int rstride, long long P, int C, int mode) {
    const int CV = C / 4;
    GRID_STRIDE(i, P * CV) {
        long long p = i / CV;
        int c0 = (int)(i - p * CV) * 4;
        uint2 raw = *reinterpret_cast<const uint2*>(y + p * ystride + c0);
        __half* hv = reinterpret_cast<__half*>(&raw);
        uint2 rraw;
        __half* rv = reinterpret_cast<__half*>(&rraw);
        if (r) rraw = *reinterpret_cast<const uint2*>(r + p * rstride + c0);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            float v = __half2float(hv[k]);
            if (s) v *= s[c0 + k];
            if (t) v += t[c0 + k];
            if (r) v += __half2float(rv[k]);
            hv[k] = __float2half(act_f(v, mode));
        }
        *reinterpret_cast<uint2*>(y + p * ystride + c0) = raw;
    }
}

// Depthwise 7x7 conv (stride 1, zero padding 3), NHWC fp16 in/out, fp32 weights w[(ky*7 + kx)*C + c] and bias b[C],
// fp32 accumulation. Block (8, 16) threads: a 16 x 32 pixel tile x 8 channels (blockIdx.z); input tile + halo and the
// weights staged in shared memory; each thread computes 4 consecutive pixels of a row (sliding window in registers).
// (EdgeNeXt blocks of FoundationStereo's DispHead; torch's NHWC depthwise kernel is ~4x slower)
extern "C" __global__ void dwconv7_nhwc(const __half* __restrict__ x, const float* __restrict__ w,
                                        const float* __restrict__ b, __half* __restrict__ y, int H, int W, int C) {
    __shared__ uint4 tile[16 + 6][32 + 6];
    __shared__ float4 wsm[49][2];
    const int c = blockIdx.z * 8;
    const int x0 = blockIdx.x * 32, y0 = blockIdx.y * 16;
    const int tid = threadIdx.y * 8 + threadIdx.x;
    for (int t = tid; t < 49 * 2; t += 128) wsm[t / 2][t % 2] = __ldg(reinterpret_cast<const float4*>(w + (t / 2) * C + c) + t % 2);
    for (int t = tid; t < 22 * 38; t += 128) {
        int ty = t / 38, tx = t % 38;
        int sy = y0 + ty - 3, sx = x0 + tx - 3;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (sy >= 0 && sy < H && sx >= 0 && sx < W) v = __ldg(reinterpret_cast<const uint4*>(x + ((long long)sy * W + sx) * C + c));
        tile[ty][tx] = v;
    }
    __syncthreads();
    const int py = y0 + threadIdx.y, pxb = threadIdx.x * 4;
    if (py >= H) return;
    float acc[4][8];
    float4 b0 = b ? __ldg(reinterpret_cast<const float4*>(b + c)) : make_float4(0, 0, 0, 0);
    float4 b1 = b ? __ldg(reinterpret_cast<const float4*>(b + c) + 1) : make_float4(0, 0, 0, 0);
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        acc[p][0] = b0.x; acc[p][1] = b0.y; acc[p][2] = b0.z; acc[p][3] = b0.w;
        acc[p][4] = b1.x; acc[p][5] = b1.y; acc[p][6] = b1.z; acc[p][7] = b1.w;
    }
#pragma unroll
    for (int dy = 0; dy < 7; ++dy) {
        float v[10][8];
#pragma unroll
        for (int i = 0; i < 10; ++i) {
            uint4 u = tile[threadIdx.y + dy][pxb + i];
            const __half2* h2 = reinterpret_cast<const __half2*>(&u);
#pragma unroll
            for (int j = 0; j < 4; ++j) { float2 f = __half22float2(h2[j]); v[i][2 * j] = f.x; v[i][2 * j + 1] = f.y; }
        }
#pragma unroll
        for (int dx = 0; dx < 7; ++dx) {
            float4 wa = wsm[dy * 7 + dx][0], wb = wsm[dy * 7 + dx][1];
            float wv[8] = {wa.x, wa.y, wa.z, wa.w, wb.x, wb.y, wb.z, wb.w};
#pragma unroll
            for (int p = 0; p < 4; ++p)
#pragma unroll
                for (int j = 0; j < 8; ++j) acc[p][j] += wv[j] * v[p + dx][j];
        }
    }
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        int px = x0 + pxb + p;
        if (px >= W) break;
        __align__(16) __half o[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) o[j] = __float2half(acc[p][j]);
        *reinterpret_cast<uint4*>(y + ((long long)py * W + px) * C + c) = *reinterpret_cast<const uint4*>(o);
    }
}

#define L1 136   // padded smem row strides (halves) of the MLP kernel
#define L2 40
// EdgeNeXt MLP (FoundationStereo DispHead), one kernel: Y = R + s2 * (GELU(X W1^T + b1) W2^T) + t2 with C = 128, E = 512
// (t2 = s2 * b2), fp16 tensor-core MMAs with fp16 accumulation (as the fp16-acc cuDNN 1x1 convs), the 128 x 512 hidden
// activation never leaves the SM. Block = 4 warps = 128 pixels (32 per warp); hidden processed in chunks of 32 with the
// weight chunks staged in shared memory. X must have >= gridDim.x * 128 rows (padded allocation); rows >= P not stored.
extern "C" __global__ void __launch_bounds__(128) mlp_gelu_res(const __half* __restrict__ X, const __half* __restrict__ W1,
                                                               const float* __restrict__ b1, const __half* __restrict__ W2,
                                                               const float* __restrict__ s2, const float* __restrict__ t2,
                                                               const __half* __restrict__ R, __half* __restrict__ Y, long long P) {
    using namespace nvcuda;
    __shared__ __align__(32) __half w1s[32 * L1];     // W1 rows c0..c0+31: [hid][in] (padded rows: no bank conflicts)
    __shared__ __align__(32) __half w2s[128 * L2];    // W2[:, c0..c0+31]: [out][hid]
    __shared__ __align__(32) __half hs[4][32 * L2];   // per warp: hidden tile [pixel][hid]
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const long long m0 = blockIdx.x * 128LL + warp * 32;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a[2][8];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) wmma::load_matrix_sync(a[i][kk], X + (m0 + 16 * i) * 128 + kk * 16, 128);
    wmma::fragment<wmma::accumulator, 16, 16, 16, __half> acc2[2][8];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int n = 0; n < 8; ++n) wmma::fill_fragment(acc2[i][n], __float2half(0.f));
    for (int c0 = 0; c0 < 512; c0 += 32) {
        __syncthreads();
#pragma unroll
        for (int q = threadIdx.x; q < 512; q += 128) {
            *reinterpret_cast<uint4*>(w1s + (q / 16) * L1 + (q % 16) * 8) = __ldg(reinterpret_cast<const uint4*>(W1 + c0 * 128) + q);
            int n = q / 4, part = q % 4;
            *reinterpret_cast<uint4*>(w2s + n * L2 + part * 8) = __ldg(reinterpret_cast<const uint4*>(W2 + n * 512 + c0 + part * 8));
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                wmma::fragment<wmma::accumulator, 16, 16, 16, __half> acc1;
                wmma::fill_fragment(acc1, __float2half(0.f));
#pragma unroll
                for (int kk = 0; kk < 8; ++kk) {
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
                    wmma::load_matrix_sync(b, w1s + 16 * j * L1 + 16 * kk, L1);
                    wmma::mma_sync(acc1, a[i][kk], b, acc1);
                }
                wmma::store_matrix_sync(hs[warp] + 16 * i * L2 + 16 * j, acc1, L2, wmma::mem_row_major);
            }
        __syncwarp();
        for (int e = lane; e < 1024; e += 32) {
            int ei = (e >> 5) * L2 + (e & 31);
            float v = __half2float(hs[warp][ei]) + __ldg(b1 + c0 + (e & 31));
            hs[warp][ei] = __float2half(0.5f * v * (1.f + erff(v * 0.70710678118654752f)));
        }
        __syncwarp();
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a2;
                wmma::load_matrix_sync(a2, hs[warp] + 16 * i * L2 + 16 * kk, L2);
#pragma unroll
                for (int n = 0; n < 8; ++n) {
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b2;
                    wmma::load_matrix_sync(b2, w2s + 16 * n * L2 + 16 * kk, L2);
                    wmma::mma_sync(acc2[i][n], a2, b2, acc2[i][n]);
                }
            }
    }
    // epilogue through shared memory (w1s | w2s >= 4 warps x 16 x 128 halves)
    __half* st = (warp < 2 ? w1s : w2s) + (warp & 1) * 16 * 128;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        __syncthreads();
#pragma unroll
        for (int n = 0; n < 8; ++n) wmma::store_matrix_sync(st + 16 * n, acc2[i][n], 128, wmma::mem_row_major);
        __syncwarp();
#pragma unroll
        for (int q = lane; q < 256; q += 32) {
            int row = q / 16, c = (q % 16) * 8;
            long long p = m0 + 16 * i + row;
            if (p >= P) continue;
            uint4 av = *reinterpret_cast<const uint4*>(st + row * 128 + c);
            uint4 rv = __ldg(reinterpret_cast<const uint4*>(R + p * 128 + c));
            const __half *ah = reinterpret_cast<const __half*>(&av), *rh = reinterpret_cast<const __half*>(&rv);
            __align__(16) __half o[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) o[k] = __float2half(__half2float(ah[k]) * __ldg(s2 + c + k) + __ldg(t2 + c + k) + __half2float(rh[k]));
            *reinterpret_cast<uint4*>(Y + p * 128 + c) = *reinterpret_cast<const uint4*>(o);
        }
    }
}

// Two chained bilinear resizes (align_corners=True, PyTorch formulas) without the intermediate tensor: src (Hs, Ws) ->
// mid (Hm, Wm) [values rounded to fp16 as the materialized intermediate] -> dst (Hd, Wd). NHWC fp16, C % 8 == 0, batch 1.
// (DPT head: output_conv1 -> ViT input resolution -> FoundationStereo's 1/4 feature resolution.)
__device__ __forceinline__ void bl_coord(float r, int i, int n, int& i0, int& ip, float& l) {
    float s = r * (float)i;
    i0 = (int)s;
    ip = i0 < n - 1 ? 1 : 0;
    l = s - (float)i0;
}
__device__ __forceinline__ void mid_val(const __half* __restrict__ src, int Ws, int C, int c, float rh, float rw, int Hs,
                                        int y, int x, float* o) {
    int h1, h1p, w1, w1p;
    float h1l, w1l;
    bl_coord(rh, y, Hs, h1, h1p, h1l);
    bl_coord(rw, x, Ws, w1, w1p, w1l);
    float h0l = 1.f - h1l, w0l = 1.f - w1l;
    uint4 a = *reinterpret_cast<const uint4*>(src + ((long long)h1 * Ws + w1) * C + c);
    uint4 b = *reinterpret_cast<const uint4*>(src + ((long long)h1 * Ws + w1 + w1p) * C + c);
    uint4 cc = *reinterpret_cast<const uint4*>(src + ((long long)(h1 + h1p) * Ws + w1) * C + c);
    uint4 d = *reinterpret_cast<const uint4*>(src + ((long long)(h1 + h1p) * Ws + w1 + w1p) * C + c);
    const __half *v00 = reinterpret_cast<const __half*>(&a), *v01 = reinterpret_cast<const __half*>(&b);
    const __half *v10 = reinterpret_cast<const __half*>(&cc), *v11 = reinterpret_cast<const __half*>(&d);
#pragma unroll
    for (int k = 0; k < 8; ++k)
        o[k] = __half2float(__float2half(h0l * (w0l * __half2float(v00[k]) + w1l * __half2float(v01[k])) +
                                         h1l * (w0l * __half2float(v10[k]) + w1l * __half2float(v11[k]))));
}
extern "C" __global__ void bilerp2_nhwc(const __half* __restrict__ src, int Hs, int Ws, int Hm, int Wm, __half* dst,
                                        int Hd, int Wd, int C) {
    float rhm = Hm > 1 ? (float)(Hs - 1) / (float)(Hm - 1) : 0.f, rwm = Wm > 1 ? (float)(Ws - 1) / (float)(Wm - 1) : 0.f;
    float rhd = Hd > 1 ? (float)(Hm - 1) / (float)(Hd - 1) : 0.f, rwd = Wd > 1 ? (float)(Wm - 1) / (float)(Wd - 1) : 0.f;
    const int CV = C / 8;
    GRID_STRIDE(i, (long long)Hd * Wd * CV) {
        int c = (int)(i % CV) * 8;
        long long pd = i / CV;
        int y = (int)(pd / Wd), x = (int)(pd % Wd);
        int h1, h1p, w1, w1p;
        float h1l, w1l;
        bl_coord(rhd, y, Hm, h1, h1p, h1l);
        bl_coord(rwd, x, Wm, w1, w1p, w1l);
        float h0l = 1.f - h1l, w0l = 1.f - w1l;
        float v00[8], v01[8], v10[8], v11[8];
        mid_val(src, Ws, C, c, rhm, rwm, Hs, h1, w1, v00);
        mid_val(src, Ws, C, c, rhm, rwm, Hs, h1, w1 + w1p, v01);
        mid_val(src, Ws, C, c, rhm, rwm, Hs, h1 + h1p, w1, v10);
        mid_val(src, Ws, C, c, rhm, rwm, Hs, h1 + h1p, w1 + w1p, v11);
        __align__(16) __half o[8];
#pragma unroll
        for (int k = 0; k < 8; ++k)
            o[k] = __float2half(h0l * (w0l * v00[k] + w1l * v01[k]) + h1l * (w0l * v10[k] + w1l * v11[k]));
        *reinterpret_cast<uint4*>(dst + pd * C + c) = *reinterpret_cast<const uint4*>(o);
    }
}

// FoundationStereo cost volume rows (hwd layout, memory [row, w, d, 32] fp16): [gwc (8 groups) | left proj (12) |
// right proj (12) shifted by d], == build_gwc_volume + build_concat_volume. M: fp32 [G, h, W, W] all-pairs group
// correlations of these rows against the W-flipped right features (gwc of pixel w at disparity d = M[g, r, w, W-1-w+d]);
// entries with w < d are zero. l12 / r12: NHWC fp16 [h, W, 12] (rows of this strip).
extern "C" __global__ void build_vol_hwd(const float* __restrict__ M, int G, int h, int W, int D,
                                         const __half* __restrict__ l12, const __half* __restrict__ r12, __half* vol) {
    GRID_STRIDE(i, (long long)h * W * D) {
        int d = (int)(i % D);
        long long rw = i / D;
        int w = (int)(rw % W), r = (int)(rw / W);
        __align__(16) __half o[32];
        bool ok = w >= d;
#pragma unroll
        for (int g = 0; g < 8; ++g)
            o[g] = (ok && g < G) ? __float2half(M[(((long long)g * h + r) * W + w) * W + (W - 1 - w + d)]) : __float2half(0.f);
        const __half* lp = l12 + rw * 12;
#pragma unroll
        for (int c = 0; c < 12; ++c) o[8 + c] = lp[c];
        const __half* rp = r12 + (rw - d) * 12;
#pragma unroll
        for (int c = 0; c < 12; ++c) o[20 + c] = ok ? rp[c] : __float2half(0.f);
        uint4* dst = reinterpret_cast<uint4*>(vol + i * 32);
        const uint4* src = reinterpret_cast<const uint4*>(o);
#pragma unroll
        for (int k = 0; k < 4; ++k) dst[k] = src[k];
    }
}

// y[r, w, d, :C] += fp16(trilinear x4 upsample (align_corners=False, PyTorch formula) of x at (h0 + r, w, d)) for the
// rows r in [0, nh) of a strip. x: [Hs, Ws, Ds, C] fp16 (full volume, NDHWC), y: [nh, W, D, ystride]. C % 4 == 0.
__device__ __forceinline__ void src4(int o, int n, int& i0, int& ip, float& l) {
    float s = fmaxf(0.25f * ((float)o + 0.5f) - 0.5f, 0.f);
    i0 = (int)s;
    ip = i0 < n - 1 ? 1 : 0;
    l = s - (float)i0;
}
extern "C" __global__ void trilerp4_add(const __half* __restrict__ x, int Hs, int Ws, int Ds, int C, __half* y, int ystride,
                                        int h0, int nh, int W, int D) {
    const int CV = C / 4;
    GRID_STRIDE(i, (long long)nh * W * D * CV) {
        int c = (int)(i % CV) * 4;
        long long v = i / CV;
        int d = (int)(v % D);
        long long t = v / D;
        int w = (int)(t % W), r = (int)(t / W);
        int a0, ap, b0, bp, e0, ep;
        float la, lb, le;
        src4(h0 + r, Hs, a0, ap, la);
        src4(w, Ws, b0, bp, lb);
        src4(d, Ds, e0, ep, le);
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            int da = k >> 2, db = (k >> 1) & 1, de = k & 1;
            float wt = (da ? la : 1.f - la) * (db ? lb : 1.f - lb) * (de ? le : 1.f - le);
            long long idx = (((long long)(a0 + da * ap) * Ws + (b0 + db * bp)) * Ds + (e0 + de * ep)) * C + c;
            uint2 q = *reinterpret_cast<const uint2*>(x + idx);
            const __half* qh = reinterpret_cast<const __half*>(&q);
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[j] += wt * __half2float(qh[j]);
        }
        __half* yp = y + v * ystride + c;
        uint2 q = *reinterpret_cast<const uint2*>(yp);
        __half* qh = reinterpret_cast<__half*>(&q);
#pragma unroll
        for (int j = 0; j < 4; ++j) qh[j] = __float2half(__half2float(qh[j]) + __half2float(__float2half(acc[j])));
        *reinterpret_cast<uint2*>(yp) = q;
    }
}

// affine_act for an even channel count (2 channels per thread as half2; dense rows: ystride == rstride == C)
extern "C" __global__ void affine_act2(__half* y, int ystride, const float* __restrict__ s, const float* __restrict__ t,
                                       const __half* __restrict__ r, int rstride, long long P, int C, int mode) {
    const int CV = C / 2;
    GRID_STRIDE(i, P * CV) {
        long long p = i / CV;
        int c0 = (int)(i - p * CV) * 2;
        __half2 hv = *reinterpret_cast<const __half2*>(y + p * ystride + c0);
        float v0 = __low2float(hv), v1 = __high2float(hv);
        if (s) { v0 *= s[c0]; v1 *= s[c0 + 1]; }
        if (t) { v0 += t[c0]; v1 += t[c0 + 1]; }
        if (r) { __half2 rv = *reinterpret_cast<const __half2*>(r + p * rstride + c0); v0 += __low2float(rv); v1 += __high2float(rv); }
        *reinterpret_cast<__half2*>(y + p * ystride + c0) = __floats2half2_rn(act_f(v0, mode), act_f(v1, mode));
    }
}

// Depthwise Conv3d with kernel == stride (k0, k1, k2), no padding (hourglass conv_patch), fp32 accumulation:
// x [S0, S1, S2, C] fp16 (NDHWC, batch 1) -> y [O0, O1, O2, C]; w fp32 [C, k0*k1*k2]; b fp32 [C] or null. C % 4 == 0.
extern "C" __global__ void patch_dw(const __half* __restrict__ x, int S1, int S2, int C, const float* __restrict__ w,
                                    const float* __restrict__ b, __half* y, int O0, int O1, int O2, int k0, int k1, int k2) {
    const int CV = C / 4;
    GRID_STRIDE(i, (long long)O0 * O1 * O2 * CV) {
        int c = (int)(i % CV) * 4;
        long long v = i / CV;
        int o2 = (int)(v % O2);
        long long t = v / O2;
        int o1 = (int)(t % O1), o0 = (int)(t / O1);
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        int K = k0 * k1 * k2, tap = 0;
        for (int a = 0; a < k0; ++a)
            for (int bb = 0; bb < k1; ++bb)
                for (int e = 0; e < k2; ++e, ++tap) {
                    long long idx = (((long long)(o0 * k0 + a) * S1 + (o1 * k1 + bb)) * S2 + (o2 * k2 + e)) * C + c;
                    uint2 q = *reinterpret_cast<const uint2*>(x + idx);
                    const __half* qh = reinterpret_cast<const __half*>(&q);
#pragma unroll
                    for (int j = 0; j < 4; ++j) acc[j] += __half2float(qh[j]) * w[(c + j) * K + tap];
                }
        uint2 q;
        __half* qh = reinterpret_cast<__half*>(&q);
#pragma unroll
        for (int j = 0; j < 4; ++j) qh[j] = __float2half(acc[j] + (b ? b[c + j] : 0.f));
        *reinterpret_cast<uint2*>(y + v * C + c) = q;
    }
}
