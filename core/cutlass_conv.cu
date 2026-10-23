// Adapted from our RAFT-Stereo / IGEV-Stereo kernels (non-square filters, explicit padding). FoundationStereo: per-channel
// fp32 bias, strided outputs everywhere (results go straight into the next conv's input buffer), and the selective-GRU
// combine (small / large RaftConvGRU blended by the per-pixel attention) as an epilogue.
// Build needs CUTLASS v4.7.1 (cb42473) headers: set CUTLASS_DIR (core/cutlass_conv.py).
//
// Size-independent implicit-GEMM convolutions with fused epilogues (CUTLASS 2.x API, sm_80-style mma.sync with fp16
// accumulation), compiled ONCE ahead of time (core/prebuilt/libcutlass_conv_*.so) -- no per-size compilation or tuning.
// Input: a dense NHWC fp16 buffer [H, W, C]; the visitor gets each thread's 8-channel accumulator vectors with their
// (pixel p, channel n) coordinates and finishes the op from registers. b: fp32 bias [N] or null. Hidden size = 128.
//   mode 0 (z|r conv, N = 256): s = acc + b[n]; n < 128: z = sigmoid(s) -> o[p*os + n];
//           n >= 128: sigmoid(s) * h[p*hs + n-128] -> o2[p*o2s + n-128]   (r*h straight into the q conv's input)
//   mode 1 (q conv, small GRU):  q = tanh(acc + b[n]); h + z (q - h) -> o[p*os + n]    (z = z[p*zs + n], h = h[p*hs + n])
//   mode 3 (q conv, large GRU + selective combine): hl = h + z (tanh(acc + b[n]) - h); hs = g[p*gs + n];
//           hn = hl + att[p] (hs - hl) -> o[p*os + n] (may alias h: in-place state update) and o2[p*o2s + n] if o2
//   mode 2 (affine): v = act(acc + b[n] + c[p*cs + n]) (c: optional per-pixel term, e.g. the precomputed contribution of
//           constant input channels; act 0 none, 1 relu, 3 tanh, 4 sigmoid) -> o[p*os + n], o2[p*o2s + n] if o2;
//           disp != null: channel N-1 is replaced by fp16(disp[p]) (motion features' last channel)
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/kernel/default_conv3d_fprop.h"
#include "cutlass/conv/kernel/implicit_gemm_convolution.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/threadblock/epilogue_with_visitor.h"

using Hf = cutlass::half_t;
using NHWC = cutlass::layout::TensorNHWC;
using NDHWC = cutlass::layout::TensorNDHWC;

extern "C" struct EpiArgs {
    int mode, act;
    const float* b;
    __half* o; int os;
    __half* o2; int o2s;
    const __half* h; int hs;
    const __half* z; int zs;
    const __half* g; int gs;
    const __half* att;
    const float* disp;
    const __half* c; int cs;
};

__device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + __expf(-x)); }

template <class OutIt>
struct GruVisitor {
    static int const kIterations = OutIt::kIterations;
    static int const kElementsPerAccess = OutIt::kElementsPerAccess;
    static_assert(kElementsPerAccess == 8, "8 channels per access");
    using ElementAccumulator = Hf;
    using AccumulatorFragment = cutlass::Array<Hf, kElementsPerAccess>;
    using ThreadMap = typename OutIt::ThreadMap;
    struct SharedStorage {};

    EpiArgs const& a;
    OutIt it;
    int M, N;
    // operands of the current step loaded in begin_step (overlaps the accumulator exchange through shared memory):
    // pz = z (modes 1, 3) / h (mode 0, n >= 128), ph = h (modes 1, 3), pg = g (mode 3)
    static int const kFrags = ThreadMap::Iterations::kCount;
    uint4 pz[kFrags], ph[kFrags], pg[kFrags];

    CUTLASS_DEVICE GruVisitor(EpiArgs const& args, typename OutIt::Params const& prm, cutlass::MatrixCoord extent,
                              int thread_idx, cutlass::MatrixCoord tb_offset)
        : a(args), it(prm, nullptr, extent, thread_idx, tb_offset), M(extent.row()), N(extent.column()) {}

    CUTLASS_DEVICE void begin_epilogue() {}
    CUTLASS_DEVICE void begin_step(int) {
        if (a.mode == 2) return;
#pragma unroll
        for (int i = 0; i < kFrags; ++i) {
            cutlass::MatrixCoord co = it.thread_start() + ThreadMap::iteration_offset(i);
            long long p = co.row();
            int n = co.column();
            if (p < M && n < N) {
                if (a.mode == 0) {
                    if (n >= 128) pz[i] = *reinterpret_cast<const uint4*>(a.h + p * a.hs + n - 128);
                } else {
                    pz[i] = *reinterpret_cast<const uint4*>(a.z + p * a.zs + n);
                    ph[i] = *reinterpret_cast<const uint4*>(a.h + p * a.hs + n);
                    if (a.mode == 3) pg[i] = *reinterpret_cast<const uint4*>(a.g + p * a.gs + n);
                }
            }
        }
    }
    CUTLASS_DEVICE void begin_row(int) {}
    CUTLASS_DEVICE void end_row(int) {}
    CUTLASS_DEVICE void end_step(int) { ++it; }
    CUTLASS_DEVICE void end_epilogue() {}

    CUTLASS_DEVICE void visit(int, int, int, int frag_idx, AccumulatorFragment const& acc) {
        cutlass::MatrixCoord co = it.thread_start() + ThreadMap::iteration_offset(frag_idx);
        long long p = co.row();
        int n = co.column();
        if (p >= M || n >= N) return;
        const __half* ah = reinterpret_cast<const __half*>(&acc);
        float s[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) s[k] = __half2float(ah[k]) + (a.b ? __ldg(a.b + n + k) : 0.f);
        __align__(16) __half o[8];
        if (a.mode == 0) {
            if (n < 128) {
#pragma unroll
                for (int k = 0; k < 8; ++k) o[k] = __float2half(sigm(s[k]));
                *reinterpret_cast<uint4*>(a.o + p * a.os + n) = *reinterpret_cast<const uint4*>(o);
            } else {
                uint4 hv = pz[frag_idx];
                const __half* hh = reinterpret_cast<const __half*>(&hv);
#pragma unroll
                for (int k = 0; k < 8; ++k) o[k] = __hmul(__float2half(sigm(s[k])), hh[k]);
                *reinterpret_cast<uint4*>(a.o2 + p * a.o2s + n - 128) = *reinterpret_cast<const uint4*>(o);
            }
        } else if (a.mode == 1 || a.mode == 3) {
            uint4 zv = pz[frag_idx], hv = ph[frag_idx], gv = pg[frag_idx];
            const __half *zh = reinterpret_cast<const __half*>(&zv), *hh = reinterpret_cast<const __half*>(&hv);
            const __half* gh = reinterpret_cast<const __half*>(&gv);
            float at = a.mode == 3 ? __half2float(a.att[p]) : 0.f;
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                float q = __half2float(__float2half(tanhf(s[k])));
                float hf = __half2float(hh[k]), z = __half2float(zh[k]);
                float hn = hf + z * (q - hf);
                if (a.mode == 3) {
                    float hl = __half2float(__float2half(hn)), hs = __half2float(gh[k]);
                    hn = hl + at * (hs - hl);
                }
                o[k] = __float2half(hn);
            }
            *reinterpret_cast<uint4*>(a.o + p * a.os + n) = *reinterpret_cast<const uint4*>(o);
            if (a.mode == 3 && a.o2) *reinterpret_cast<uint4*>(a.o2 + p * a.o2s + n) = *reinterpret_cast<const uint4*>(o);
        } else {
            uint4 cv = make_uint4(0, 0, 0, 0);
            if (a.c) cv = *reinterpret_cast<const uint4*>(a.c + p * a.cs + n);
            const __half* chh = reinterpret_cast<const __half*>(&cv);
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                float v = s[k] + __half2float(chh[k]);
                if (a.act == 1) v = fmaxf(v, 0.f);
                else if (a.act == 2) v = v >= 0.f ? v : 0.01f * v;
                else if (a.act == 3) v = tanhf(v);
                else if (a.act == 4) v = sigm(v);
                o[k] = __float2half(v);
            }
            if (a.disp && n + 8 == N) o[7] = __float2half(a.disp[p]);
            *reinterpret_cast<uint4*>(a.o + p * a.os + n) = *reinterpret_cast<const uint4*>(o);
            if (a.o2) *reinterpret_cast<uint4*>(a.o2 + p * a.o2s + n) = *reinterpret_cast<const uint4*>(o);
        }
    }
};

template <class TB, class WS, int Stages, int Swz = 1>
struct Fused {
    using EpiOp = cutlass::epilogue::thread::LinearCombination<Hf, 8, Hf, Hf>;  // only shapes the epilogue
    using Base = typename cutlass::conv::kernel::DefaultConv2dFprop<
        Hf, NHWC, Hf, NHWC, Hf, NHWC, Hf, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, TB, WS,
        cutlass::gemm::GemmShape<16, 8, 16>, EpiOp, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<Swz>, Stages,
        cutlass::arch::OpMultiplyAdd, cutlass::conv::IteratorAlgorithm::kOptimized>::Kernel;
    using Mma = typename Base::Mma;
    using OutIt = typename Base::Epilogue::OutputTileIterator;
    using Visitor = GruVisitor<OutIt>;
    using Epilogue = typename cutlass::epilogue::threadblock::EpilogueWithVisitorFromExistingEpilogue<
        Visitor, typename Base::Epilogue>::Epilogue;
    union SharedStorage {
        typename Mma::SharedStorage main_loop;
        typename Epilogue::SharedStorage epilogue;
    };
    struct Params {
        typename Base::Params base;
        EpiArgs e;
    };
};

// 3D variant (NDHWC, CUTLASS Conv3d optimized iterators; C % 8 == 0, K % 8 == 0), same visitor / epilogue modes
template <class TB, class WS, int Stages>
struct Fused3 {
    using EpiOp = cutlass::epilogue::thread::LinearCombination<Hf, 8, Hf, Hf>;
    using Base = typename cutlass::conv::kernel::DefaultConv3dFprop<
        Hf, NDHWC, Hf, NDHWC, Hf, NDHWC, Hf, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, TB, WS,
        cutlass::gemm::GemmShape<16, 8, 16>, EpiOp, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>, Stages,
        cutlass::arch::OpMultiplyAdd, cutlass::conv::IteratorAlgorithm::kOptimized>::Kernel;
    using Mma = typename Base::Mma;
    using OutIt = typename Base::Epilogue::OutputTileIterator;
    using Visitor = GruVisitor<OutIt>;
    using Epilogue = typename cutlass::epilogue::threadblock::EpilogueWithVisitorFromExistingEpilogue<
        Visitor, typename Base::Epilogue>::Epilogue;
    union SharedStorage {
        typename Mma::SharedStorage main_loop;
        typename Epilogue::SharedStorage epilogue;
    };
    struct Params {
        typename Base::Params base;
        EpiArgs e;
    };
};

CUTLASS_HOST_DEVICE int out_pixels(cutlass::conv::Conv2dProblemSize const& p) { return p.N * p.P * p.Q; }
CUTLASS_HOST_DEVICE int out_pixels(cutlass::conv::Conv3dProblemSize const& p) { return p.N * p.Z * p.P * p.Q; }

template <class F>
__global__ void __launch_bounds__(F::Base::kThreadCount) fused_conv_kernel(typename F::Params const P) {
    using Mma = typename F::Mma;
    extern __shared__ int smem_[];
    auto& ss = *reinterpret_cast<typename F::SharedStorage*>(smem_);
    auto const& prm = P.base;
    typename F::Base::ThreadblockSwizzle sw;
    cutlass::gemm::GemmCoord tile = sw.get_tile_offset(prm.swizzle_log_tile);
    if (prm.grid_tiled_shape.m() <= tile.m() || prm.grid_tiled_shape.n() <= tile.n()) return;
    int thread_idx = threadIdx.x;
    typename Mma::IteratorA iA(prm.iterator_A, prm.problem_size, prm.ptr_A, thread_idx,
                               cutlass::MatrixCoord(tile.m() * Mma::Shape::kM, tile.k() * Mma::Shape::kK));
    typename Mma::IteratorB iB(prm.iterator_B, prm.problem_size, prm.ptr_B, thread_idx,
                               cutlass::MatrixCoord(tile.k() * Mma::Shape::kK, tile.n() * Mma::Shape::kN));
    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_idx = threadIdx.x % 32;
    Mma mma(ss.main_loop, thread_idx, warp_idx, lane_idx);
    typename Mma::FragmentC acc;
    acc.clear();
    mma(prm.gemm_k_iterations, acc, iA, iB, acc, prm.gemm_k_iterations_per_channel);
    cutlass::MatrixCoord tb_off(tile.m() * Mma::Shape::kM, tile.n() * Mma::Shape::kN);
    cutlass::MatrixCoord extent(out_pixels(prm.problem_size), prm.problem_size.K);
    typename F::Visitor v(P.e, prm.iterator_D, extent, thread_idx, tb_off);
    typename F::Epilogue epi(ss.epilogue, thread_idx, warp_idx, lane_idx);
    epi(v, acc);
}

template <class F>
static int run(const void* x, const void* w, int H, int W, int C, int K, int R, int S, int ph, int pw, int stride,
               EpiArgs const& e, cudaStream_t s) {
    using Base = typename F::Base;
    int Ho = (H + 2 * ph - R) / stride + 1, Wo = (W + 2 * pw - S) / stride + 1;
    cutlass::conv::Conv2dProblemSize p({1, H, W, C}, {K, R, S, C}, {ph, ph, pw, pw}, {stride, stride}, {1, 1},
                                       {1, Ho, Wo, K}, cutlass::conv::Mode::kCrossCorrelation, 1);
    typename Base::Arguments args(p, {(Hf*)x, NHWC::packed({1, H, W, C})}, {(Hf*)w, NHWC::packed({K, R, S, C})},
                                  {nullptr, NHWC::packed({1, Ho, Wo, K})}, {nullptr, NHWC::packed({1, Ho, Wo, K})},
                                  {Hf(1), Hf(0)});
    if (cutlass::conv::device::ImplicitGemmConvolution<Base>::can_implement(args) != cutlass::Status::kSuccess) return 1;
    typename F::Params P{typename Base::Params(args, nullptr), e};
    typename Base::ThreadblockSwizzle sw;
    dim3 grid = sw.get_grid_shape(P.base.grid_tiled_shape);
    int smem = int(sizeof(typename F::SharedStorage));
    static bool attr_set = false;
    if (!attr_set) {
        if (cudaFuncSetAttribute(fused_conv_kernel<F>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess)
            return 2;
        attr_set = true;
    }
    fused_conv_kernel<F><<<grid, Base::kThreadCount, smem, s>>>(P);
    return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

template <class F>
static int launch(typename F::Params const& P, cudaStream_t s) {
    using Base = typename F::Base;
    typename Base::ThreadblockSwizzle sw;
    dim3 grid = sw.get_grid_shape(P.base.grid_tiled_shape);
    int smem = int(sizeof(typename F::SharedStorage));
    static bool attr_set = false;
    if (!attr_set) {
        if (cudaFuncSetAttribute(fused_conv_kernel<F>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess)
            return 2;
        attr_set = true;
    }
    fused_conv_kernel<F><<<grid, Base::kThreadCount, smem, s>>>(P);
    return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

// x: [N, D, H, W, C] (dense NDHWC), w: [K, T, R, S, C]; output [N, Z, P, Q, K] goes through the visitor (EpiArgs)
template <class F>
static int run3(const void* x, const void* w, const int* g, EpiArgs const& e, cudaStream_t s) {
    using Base = typename F::Base;
    int N = g[0], D = g[1], H = g[2], W = g[3], C = g[4], K = g[5], T = g[6], R = g[7], S = g[8];
    int pd = g[9], ph = g[10], pw = g[11], sd = g[12], sh = g[13], sw_ = g[14];
    int Z = (D + 2 * pd - T) / sd + 1, P = (H + 2 * ph - R) / sh + 1, Q = (W + 2 * pw - S) / sw_ + 1;
    cutlass::conv::Conv3dProblemSize p(N, D, H, W, C, K, T, R, S, Z, P, Q, pd, ph, pw, sd, sh, sw_, 1, 1, 1,
                                       cutlass::conv::Mode::kCrossCorrelation, 1, 1);
    typename Base::Arguments args(p, {(Hf*)x, NDHWC::packed({N, D, H, W, C})}, {(Hf*)w, NDHWC::packed({K, T, R, S, C})},
                                  {nullptr, NDHWC::packed({N, Z, P, Q, K})}, {nullptr, NDHWC::packed({N, Z, P, Q, K})},
                                  {Hf(1), Hf(0)});
    if (cutlass::conv::device::ImplicitGemmConvolution<Base>::can_implement(args) != cutlass::Status::kSuccess) return 1;
    typename F::Params Pm{typename Base::Params(args, nullptr), e};
    return launch<F>(Pm, s);
}

template <int M, int N, int K>
using GS = cutlass::gemm::GemmShape<M, N, K>;

// g: N, D, H, W, C, K, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w
extern "C" int fused_conv3d(int cfg, const void* x, const void* w, const int* g, const EpiArgs* e, void* stream) {
    cudaStream_t s = (cudaStream_t)stream;
    switch (cfg) {
        case 0: return run3<Fused3<GS<128, 32, 32>, GS<32, 32, 32>, 4>>(x, w, g, *e, s);
        case 1: return run3<Fused3<GS<256, 32, 32>, GS<64, 32, 32>, 4>>(x, w, g, *e, s);
        case 2: return run3<Fused3<GS<128, 32, 64>, GS<64, 32, 64>, 3>>(x, w, g, *e, s);
        case 3: return run3<Fused3<GS<128, 64, 32>, GS<64, 32, 32>, 4>>(x, w, g, *e, s);
        case 4: return run3<Fused3<GS<64, 16, 32>, GS<32, 16, 32>, 4>>(x, w, g, *e, s);
        case 5: return run3<Fused3<GS<128, 128, 32>, GS<64, 64, 32>, 3>>(x, w, g, *e, s);
    }
    return -1;
}

// tile configurations (compiled once; the caller picks one per call from the problem size by a fixed rule)
extern "C" int fused_conv_rs(int cfg, const void* x, const void* w, int H, int W, int C, int K, int R, int S, int ph,
                             int pw, int stride, const EpiArgs* e, void* stream) {
    cudaStream_t s = (cudaStream_t)stream;
    switch (cfg) {  // 128x128x32, 3 stages = the tile config cuDNN's own precompiled fprop kernel uses here
        case 0: return run<Fused<GS<128, 128, 32>, GS<64, 64, 32>, 3>>(x, w, H, W, C, K, R, S, ph, pw, stride, *e, s);
        case 1: return run<Fused<GS<64, 128, 64>, GS<32, 64, 64>, 4>>(x, w, H, W, C, K, R, S, ph, pw, stride, *e, s);  // small P
    }
    return -1;
}

// square R x R filter, padding R / 2
extern "C" int fused_conv(int cfg, const void* x, const void* w, int H, int W, int C, int K, int R, int stride,
                          const EpiArgs* e, void* stream) {
    return fused_conv_rs(cfg, x, w, H, W, C, K, R, R, R / 2, R / 2, stride, e, stream);
}
