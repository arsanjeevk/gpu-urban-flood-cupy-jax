"""Fused CUDA primitives shared by the triangular-backend fused kernels.

These replace the `scatter_add_1d` + "scale request by available capacity"
chains that are duplicated across `network_kernels.py` and
`triangular_backend.py`, each of which today costs 2+ separate CuPy kernel
launches (a `cp.zeros` + `cupyx.scatter_add`, or a scatter followed by a
gather-and-multiply). Each helper here is a single `cp.RawKernel` launch.
"""

from __future__ import annotations

from typing import Any

_SCATTER_ADD_KERNEL = None
_SCATTER_ADD_KERNEL_F64 = None
_LIMIT_APPLY_KERNEL = None
_LIMIT_APPLY_KERNEL_F64 = None

_SCATTER_ADD_KERNEL_SOURCE = r"""
extern "C" __global__
void scatter_add_1d_kernel(
    const int* __restrict__ indices,
    const float* __restrict__ values,
    float* __restrict__ out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    atomicAdd(&out[indices[i]], values[i]);
}
"""

# Double-precision twin, used for node-volume-scale accumulators where
# float32's ~7 significant digits lose real precision once a node's
# accumulated volume reaches the tens-of-thousands-of-m3 range (e.g. heavy
# surcharge). Node/link arrays are tiny (thousands of elements), so the
# extra 4 bytes/element cost nothing.
_SCATTER_ADD_KERNEL_F64_SOURCE = r"""
extern "C" __global__
void scatter_add_1d_kernel_f64(
    const int* __restrict__ indices,
    const double* __restrict__ values,
    double* __restrict__ out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    atomicAdd(&out[indices[i]], values[i]);
}
"""

_LIMIT_APPLY_KERNEL_SOURCE = r"""
extern "C" __global__
void limit_by_available_volume_apply_kernel(
    const float* __restrict__ requested_abs,
    const int* __restrict__ donor_index,
    const float* __restrict__ requested_by_donor,
    const float* __restrict__ available_volume,
    float* __restrict__ limited_abs,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int donor = donor_index[i];
    float denom = fmaxf(requested_by_donor[donor], 1.0e-12f);
    float factor = fminf(1.0f, available_volume[donor] / denom);
    limited_abs[i] = requested_abs[i] * factor;
}
"""

_LIMIT_APPLY_KERNEL_F64_SOURCE = r"""
extern "C" __global__
void limit_by_available_volume_apply_kernel_f64(
    const double* __restrict__ requested_abs,
    const int* __restrict__ donor_index,
    const double* __restrict__ requested_by_donor,
    const double* __restrict__ available_volume,
    double* __restrict__ limited_abs,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int donor = donor_index[i];
    double denom = fmax(requested_by_donor[donor], 1.0e-12);
    double factor = fmin(1.0, available_volume[donor] / denom);
    limited_abs[i] = requested_abs[i] * factor;
}
"""

_THREADS_PER_BLOCK = 256


def _blocks(n: int) -> int:
    return (n + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK


def cuda_kernels_available() -> bool:
    """Return True if CuPy and at least one CUDA device are available."""

    try:
        import cupy as cp  # type: ignore[import-not-found]

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _scatter_add_kernel():
    global _SCATTER_ADD_KERNEL
    if _SCATTER_ADD_KERNEL is not None:
        return _SCATTER_ADD_KERNEL
    import cupy as cp

    _SCATTER_ADD_KERNEL = cp.RawKernel(_SCATTER_ADD_KERNEL_SOURCE, "scatter_add_1d_kernel")
    return _SCATTER_ADD_KERNEL


def cuda_scatter_add_1d(indices: Any, values: Any, size: int) -> Any:
    """Fused `atomicAdd`-based scatter-add (CuPy float32 arrays only)."""

    import cupy as cp

    out = cp.zeros((size,), dtype=cp.float32)
    n = int(values.shape[0])
    if n == 0:
        return out
    kernel = _scatter_add_kernel()
    kernel(
        (_blocks(n),),
        (_THREADS_PER_BLOCK,),
        (
            cp.ascontiguousarray(indices, dtype=cp.int32),
            cp.ascontiguousarray(values, dtype=cp.float32),
            out,
            cp.int32(n),
        ),
    )
    return out


def _limit_apply_kernel():
    global _LIMIT_APPLY_KERNEL
    if _LIMIT_APPLY_KERNEL is not None:
        return _LIMIT_APPLY_KERNEL
    import cupy as cp

    _LIMIT_APPLY_KERNEL = cp.RawKernel(_LIMIT_APPLY_KERNEL_SOURCE, "limit_by_available_volume_apply_kernel")
    return _LIMIT_APPLY_KERNEL


def cuda_limit_by_available_volume(
    requested_abs: Any,
    donor_index: Any,
    available_volume: Any,
    donor_count: int,
) -> Any:
    """Scale each `requested_abs[i]` so the sum per `donor_index[i]` never exceeds `available_volume[donor]`.

    Fused equivalent of the `scatter_add_1d(donor, abs(x)) ->
    xp.minimum(1, avail / sum) -> gather-and-multiply` pattern duplicated in
    `network_kernels.limit_link_transfers_by_available_volume`,
    `network_kernels.limit_node_exchange_by_available_volume`, and
    `triangular_backend._limit_cell_capture_by_available_volume` /
    `_limit_cell_capture_by_indexed_available_volume`.
    """

    import cupy as cp

    n = int(requested_abs.shape[0])
    if n == 0:
        return cp.zeros((0,), dtype=cp.float32)
    requested_by_donor = cuda_scatter_add_1d(donor_index, requested_abs, donor_count)
    limited = cp.empty((n,), dtype=cp.float32)
    kernel = _limit_apply_kernel()
    kernel(
        (_blocks(n),),
        (_THREADS_PER_BLOCK,),
        (
            cp.ascontiguousarray(requested_abs, dtype=cp.float32),
            cp.ascontiguousarray(donor_index, dtype=cp.int32),
            requested_by_donor,
            cp.ascontiguousarray(available_volume, dtype=cp.float32),
            limited,
            cp.int32(n),
        ),
    )
    return limited


def _scatter_add_kernel_f64():
    global _SCATTER_ADD_KERNEL_F64
    if _SCATTER_ADD_KERNEL_F64 is not None:
        return _SCATTER_ADD_KERNEL_F64
    import cupy as cp

    _SCATTER_ADD_KERNEL_F64 = cp.RawKernel(_SCATTER_ADD_KERNEL_F64_SOURCE, "scatter_add_1d_kernel_f64")
    return _SCATTER_ADD_KERNEL_F64


def cuda_scatter_add_1d_f64(indices: Any, values: Any, size: int) -> Any:
    """Double-precision twin of `cuda_scatter_add_1d`, for node/link-volume accumulators."""

    import cupy as cp

    out = cp.zeros((size,), dtype=cp.float64)
    n = int(values.shape[0])
    if n == 0:
        return out
    kernel = _scatter_add_kernel_f64()
    kernel(
        (_blocks(n),),
        (_THREADS_PER_BLOCK,),
        (
            cp.ascontiguousarray(indices, dtype=cp.int32),
            cp.ascontiguousarray(values, dtype=cp.float64),
            out,
            cp.int32(n),
        ),
    )
    return out


def _limit_apply_kernel_f64():
    global _LIMIT_APPLY_KERNEL_F64
    if _LIMIT_APPLY_KERNEL_F64 is not None:
        return _LIMIT_APPLY_KERNEL_F64
    import cupy as cp

    _LIMIT_APPLY_KERNEL_F64 = cp.RawKernel(_LIMIT_APPLY_KERNEL_F64_SOURCE, "limit_by_available_volume_apply_kernel_f64")
    return _LIMIT_APPLY_KERNEL_F64


def cuda_limit_by_available_volume_f64(
    requested_abs: Any,
    donor_index: Any,
    available_volume: Any,
    donor_count: int,
) -> Any:
    """Double-precision twin of `cuda_limit_by_available_volume`, for node/link-volume accumulators."""

    import cupy as cp

    n = int(requested_abs.shape[0])
    if n == 0:
        return cp.zeros((0,), dtype=cp.float64)
    requested_by_donor = cuda_scatter_add_1d_f64(donor_index, requested_abs, donor_count)
    limited = cp.empty((n,), dtype=cp.float64)
    kernel = _limit_apply_kernel_f64()
    kernel(
        (_blocks(n),),
        (_THREADS_PER_BLOCK,),
        (
            cp.ascontiguousarray(requested_abs, dtype=cp.float64),
            cp.ascontiguousarray(donor_index, dtype=cp.int32),
            requested_by_donor,
            cp.ascontiguousarray(available_volume, dtype=cp.float64),
            limited,
            cp.int32(n),
        ),
    )
    return limited
