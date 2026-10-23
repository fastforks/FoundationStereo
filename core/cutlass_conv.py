"""Fused-epilogue convolutions from core/cutlass_conv.cu (CUTLASS implicit GEMM, fp16 accumulation), compiled once
ahead of time into a shared library (core/prebuilt/libcutlass_conv_<hash>.so, sm_80/86/89/90/120 SASS + PTX); the
problem size is a runtime argument, so nothing is compiled or tuned per input size. Without the shipped library it
is built once with the active CUDA_HOME's nvcc into ~/.cache (needs CUTLASS v4.7.1 headers in CUTLASS_DIR).
"""
import ctypes
import hashlib
import os
import subprocess

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "cutlass_conv.cu")
_PREBUILT = os.path.join(_HERE, "prebuilt")
_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "fs-cutlass-conv")


def _cutlass_dir():
    if "CUTLASS_DIR" in os.environ:
        return os.environ["CUTLASS_DIR"]
    raise FileNotFoundError("CUTLASS headers not found: set CUTLASS_DIR (CUTLASS v4.7.1)")


class EpiArgs(ctypes.Structure):
    _fields_ = [("mode", ctypes.c_int), ("act", ctypes.c_int), ("b", ctypes.c_void_p),
                ("o", ctypes.c_void_p), ("os", ctypes.c_int), ("o2", ctypes.c_void_p), ("o2s", ctypes.c_int),
                ("h", ctypes.c_void_p), ("hs", ctypes.c_int), ("z", ctypes.c_void_p), ("zs", ctypes.c_int),
                ("g", ctypes.c_void_p), ("gs", ctypes.c_int), ("att", ctypes.c_void_p), ("disp", ctypes.c_void_p),
                ("c", ctypes.c_void_p), ("cs", ctypes.c_int)]


def _tag():
    return hashlib.sha1(open(_SRC, "rb").read()).hexdigest()[:16]


def build(out, archs=(80, 86, 89, 90, 120)):
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    gencode = [f"-gencode=arch=compute_{a},code=sm_{a}" for a in archs] + ["-gencode=arch=compute_80,code=compute_80"]
    subprocess.run([os.path.join(cuda_home, "bin", "nvcc"), "-O3", "-std=c++17", "--expt-relaxed-constexpr", "-shared",
                    "-Xcompiler", "-fPIC", "-cudart", "shared", f"-L{cuda_home}/lib", *gencode, f"-I{_cutlass_dir()}/include",
                    "-o", out + ".tmp", _SRC], check=True)
    os.replace(out + ".tmp", out)
    return out


def build_prebuilt():
    """Build the shipped library for the current source (run after editing cutlass_conv.cu)."""
    os.makedirs(_PREBUILT, exist_ok=True)
    for f in os.listdir(_PREBUILT):
        if f.startswith("libcutlass_conv_"):
            os.remove(os.path.join(_PREBUILT, f))
    return build(os.path.join(_PREBUILT, f"libcutlass_conv_{_tag()}.so"))


class FusedConvs:
    _inst = None

    @classmethod
    def get(cls):
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst

    def __init__(self):
        torch.cuda.init()  # torch has loaded libcudart.so.13, which the library links against
        tag = _tag()
        lib = os.path.join(_PREBUILT, f"libcutlass_conv_{tag}.so")
        if not os.path.exists(lib):
            major, minor = torch.cuda.get_device_capability()
            lib = os.path.join(_CACHE, f"libcutlass_conv_{tag}_sm{major}{minor}.so")
            if not os.path.exists(lib):
                os.makedirs(_CACHE, exist_ok=True)
                build(lib, archs=(major * 10 + minor,))
        self.lib = ctypes.CDLL(lib)
        self.lib.fused_conv_rs.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_int] * 9 + \
                                          [ctypes.POINTER(EpiArgs), ctypes.c_void_p]
        self.lib.fused_conv3d.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
                                          ctypes.POINTER(EpiArgs), ctypes.c_void_p]

    def conv3d(self, cfg, x, w, e: EpiArgs, padding, stride=(1, 1, 1)):
        """x: (N, C, A, B, Cc...) channels_last_3d fp16 [memory N, A, B, Cc, C]; w: (K, C, kA, kB, kC) channels_last_3d.
        Output geometry follows padding / stride; results only through the epilogue args. Returns the status."""
        N, C, D, H, W = x.shape
        K, _, T, R, S = w.shape
        g = (ctypes.c_int * 15)(N, D, H, W, C, K, T, R, S, *padding, *stride)
        return self.lib.fused_conv3d(cfg, x.data_ptr(), w.data_ptr(), g, ctypes.byref(e), torch.cuda.current_stream().cuda_stream)

    def bind(self, cfg, x, w, H, W, C, K, e: EpiArgs, padding=None, stride=1):
        """A no-argument launcher (on the then-current stream) for repeated identical calls. w: [K, R, S, C] NHWC
        (a channels_last [K, C, R, S] tensor); padding (ph, pw) defaults to (R // 2, S // 2)."""
        R, S = (w.shape[1], w.shape[2]) if w.dim() == 4 and w.shape[3] == C else (w.shape[2], w.shape[3])
        ph, pw = (R // 2, S // 2) if padding is None else padding
        f = self.lib.fused_conv_rs
        args = (cfg, x.data_ptr(), w.data_ptr(), H, W, C, K, R, S, ph, pw, stride, ctypes.byref(e))

        def call():
            rc = f(*args, torch.cuda.current_stream().cuda_stream)
            if rc:
                raise RuntimeError(f"fused_conv cfg {cfg} failed ({rc}) for {H}x{W}x{C} -> {K} ({R}x{S})")
        call.keep = (e, x, w)  # the struct / buffers must outlive the launcher
        call.rc = lambda: f(*args, torch.cuda.current_stream().cuda_stream)
        return call


    def run(self, cfg, x, w, H, W, C, K, e: EpiArgs, padding=None, stride=1):
        """One launch; returns the status (0 ok, 1 geometry not supported, else error)."""
        R, S = (w.shape[1], w.shape[2]) if w.dim() == 4 and w.shape[3] == C else (w.shape[2], w.shape[3])
        ph, pw = (R // 2, S // 2) if padding is None else padding
        return self.lib.fused_conv_rs(cfg, x.data_ptr(), w.data_ptr(), H, W, C, K, R, S, ph, pw, stride,
                                      ctypes.byref(e), torch.cuda.current_stream().cuda_stream)


if __name__ == "__main__":
    print(build_prebuilt())
