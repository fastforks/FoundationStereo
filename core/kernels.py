"""Custom CUDA kernels of core/fs_kernels.cu for PyTorch:
compiled once to a cubin (shipped in core/prebuilt/, else nvcc of CUDA_HOME, cached in ~/.cache/fs-stereo-torch),
loaded through the driver API, launched on torch's current stream. Size-independent: no per-shape compilation.
"""
import ctypes
import hashlib
import os
import subprocess

import torch

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fs_kernels.cu")
_PREBUILT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prebuilt")
_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "fs-stereo-torch")

# p: pointer (tensor or int address), i: int32, l: int64, f: float32
SIGNATURES = {
    "affine_act": "pipppilii",
    "affine_act8": "pipppilii",
    "affine_act4": "pipppilii",
    "affine_act2": "pipppilii",
    "patch_dw": "piiipppiiiiii",
    "geo_lookup": "piipiippli",
    "geo_lookup_s": "piiipiippli",
    "geo_lookup_hwd": "piipiippl" + "ii",
    "gru_zr": "pppipili",
    "pool2x_into": "piiipiiii",
    "interp_into": "piiipiiii",
    "gru_update": "ppppili",
    "dwconv7_nhwc": "ppppiii",
    "mlp_gelu_res": "ppppppppl",
    "bilerp2_nhwc": "piiiipiii",
    "build_vol_hwd": "piiiippp",
    "trilerp4_add": "piiiipiiiii",
}
_CTYPE = {"i": ctypes.c_int32, "l": ctypes.c_int64, "f": ctypes.c_float, "u": ctypes.c_uint32}


def _srctag():
    return hashlib.sha1(open(_SRC, "rb").read()).hexdigest()[:16]


def build_prebuilt(archs=(80, 86, 89, 90, 120)):
    """Ship the kernels for the current source: one fatbin (SASS for archs + compute_80 PTX) in core/prebuilt/."""
    os.makedirs(_PREBUILT, exist_ok=True)
    for f in os.listdir(_PREBUILT):
        if f.startswith("fs_kernels_"):
            os.remove(os.path.join(_PREBUILT, f))
    out = os.path.join(_PREBUILT, f"fs_kernels_{_srctag()}.fatbin")
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    gencode = [f"-gencode=arch=compute_{a},code=sm_{a}" for a in archs] + ["-gencode=arch=compute_80,code=compute_80"]
    subprocess.run([os.path.join(cuda_home, "bin", "nvcc"), "-fatbin", "-O3", "-std=c++17", "--expt-relaxed-constexpr",
                    *gencode, f"-I{cuda_home}/include", "-o", out + ".tmp", _SRC], check=True)
    os.replace(out + ".tmp", out)
    return out


def _check(res, what):
    if res != 0:
        raise RuntimeError(f"CUDA driver error {res} in {what}")


class Kernels:
    _inst = None

    @classmethod
    def get(cls):
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst

    def __init__(self):
        torch.cuda.init()
        major, minor = torch.cuda.get_device_capability()
        src = open(_SRC, "rb").read()
        tag = hashlib.sha1(src + f"sm_{major}{minor}".encode()).hexdigest()[:16]
        cubin = os.path.join(_PREBUILT, f"fs_kernels_{tag}.cubin")
        fat = os.path.join(_PREBUILT, f"fs_kernels_{_srctag()}.fatbin")   # shipped: sm_80/86/89/90/120 SASS + PTX
        if os.path.exists(fat):
            cubin = fat
        if not os.path.exists(cubin):
            cubin = os.path.join(_CACHE, f"fs_kernels_{tag}.cubin")
        if not os.path.exists(cubin):
            os.makedirs(_CACHE, exist_ok=True)
            cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
            subprocess.run([os.path.join(cuda_home, "bin", "nvcc"), "-cubin", "-O3", "-std=c++17", "--expt-relaxed-constexpr",
                            f"-arch=sm_{major}{minor}", f"-I{cuda_home}/include", "-o", cubin + ".tmp", _SRC], check=True)
            os.replace(cubin + ".tmp", cubin)
        self.lib = ctypes.CDLL("libcuda.so.1")
        self.lib.cuLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        torch.zeros(1, device="cuda")  # make sure the primary context is current
        self.module = ctypes.c_void_p()
        self._image = open(cubin, "rb").read()
        _check(self.lib.cuModuleLoadData(ctypes.byref(self.module), self._image), "cuModuleLoadData")
        self.funcs = {}
        for name in SIGNATURES:
            f = ctypes.c_void_p()
            _check(self.lib.cuModuleGetFunction(ctypes.byref(f), self.module, name.encode()), f"cuModuleGetFunction {name}")
            self.funcs[name] = f

    @staticmethod
    def linear(n):
        """Rust's Launch::linear: 256 threads, grid-stride loops, at most 65535*8 blocks."""
        return (min(max((n + 255) // 256, 1), 65535 * 8), 1, 1), (256, 1, 1)

    def launch(self, name, cfg, *args):
        grid, block = cfg
        sig = SIGNATURES[name]
        assert len(sig) == len(args), f"{name}: expected {len(sig)} args, got {len(args)}"
        holders = []
        for t, a in zip(sig, args):
            if t == "p":
                holders.append(ctypes.c_void_p(a.data_ptr() if isinstance(a, torch.Tensor) else (int(a) or None)))
            else:
                holders.append(_CTYPE[t](a))
        arr = (ctypes.c_void_p * len(holders))(*[ctypes.cast(ctypes.byref(h), ctypes.c_void_p) for h in holders])
        stream = torch.cuda.current_stream().cuda_stream
        _check(self.lib.cuLaunchKernel(self.funcs[name], *grid, *block, 0, stream, arr, None), f"launch {name}")


if __name__ == "__main__":
    print(build_prebuilt())
