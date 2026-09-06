"""CUDA RawKernel implementations for GPU flood solver hot paths.

Replaces metal_kernels.py with NVIDIA CUDA equivalents compiled via CuPy.
The rain/infiltration loop kernel uses __shared__ memory to cache scalar
constants once per block, eliminating repeated global-memory reads.

`cuda_triangular_apply_sources` is the unstructured-mesh sibling of
`cuda_rain_infiltration_loop`: a single per-cell step (no internal
sub-step loop) that also rescales momentum, fusing the ~10 separate CuPy
launches in `triangular_backend._apply_sources` into one kernel.
"""

from __future__ import annotations

from typing import Any

_SOURCE_LOOP_KERNEL = None
_TRIANGULAR_SOURCES_KERNEL = None

# CUDA kernel source — compiled on first call, cached as module-level singleton.
_SOURCE_LOOP_KERNEL_SOURCE = r"""
extern "C" __global__
void cuda_source_loop(
    const float* __restrict__ h_in,
    const float* __restrict__ active_mask,
    const float* __restrict__ rainfall_rate,
    const float* __restrict__ infiltration_capacity,
    const float* __restrict__ dt,
    const unsigned int* __restrict__ step_count,
    float* __restrict__ h_out,
    float* __restrict__ rainfall_depth,
    float* __restrict__ infiltration_depth,
    int n_elements
) {
    __shared__ float s_rain_rate;
    __shared__ float s_infil_rate;
    __shared__ float s_dt;
    __shared__ unsigned int s_nsteps;

    if (threadIdx.x == 0) {
        s_rain_rate  = fmaxf(rainfall_rate[0], 0.0f);
        s_infil_rate = fmaxf(infiltration_capacity[0], 0.0f);
        s_dt         = dt[0];
        s_nsteps     = step_count[0];
    }
    __syncthreads();

    int elem = blockIdx.x * blockDim.x + threadIdx.x;
    if (elem >= n_elements) return;

    float depth       = h_in[elem];
    float active      = active_mask[elem];
    float rain_total  = 0.0f;
    float infil_total = 0.0f;

    for (unsigned int step = 0; step < s_nsteps; ++step) {
        float rain_depth     = s_rain_rate * s_dt * active;
        depth               += rain_depth;
        float capacity_depth = s_infil_rate * s_dt * active;
        float infil_depth    = fminf(depth, capacity_depth);
        depth               -= infil_depth;
        rain_total          += rain_depth;
        infil_total         += infil_depth;
    }

    h_out[elem]              = fmaxf(depth, 0.0f);
    rainfall_depth[elem]     = rain_total;
    infiltration_depth[elem] = infil_total;
}
"""

_THREADS_PER_BLOCK = 256


def cuda_source_loop_available() -> bool:
    """Return True if CuPy and at least one CUDA device are available."""
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def cuda_rain_infiltration_loop(
    h_m: Any,
    active_mask: Any,
    rainfall_rate_mps: float,
    infiltration_capacity_mps: float,
    timestep_s: float,
    step_count: int,
) -> tuple[Any, Any, Any]:
    """Advance rainfall/infiltration over many sub-steps in one CUDA dispatch.

    Returns ``(final_depth_m, rainfall_depth_m, infiltration_depth_m)``.
    Inputs must be CuPy float32 arrays. The inner loop runs entirely on the
    GPU so no Python-level iteration is needed.
    """

    if step_count < 0:
        raise ValueError("step_count must be non-negative.")

    import cupy as cp

    kernel = _source_loop_kernel()

    n = h_m.size
    h_out = cp.empty_like(h_m)
    rain_depth = cp.empty_like(h_m)
    infil_depth = cp.empty_like(h_m)

    rainfall = cp.array([float(rainfall_rate_mps)], dtype=cp.float32)
    infiltration = cp.array([float(infiltration_capacity_mps)], dtype=cp.float32)
    dt = cp.array([float(timestep_s)], dtype=cp.float32)
    steps = cp.array([int(step_count)], dtype=cp.uint32)

    blocks = (n + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
    kernel(
        (blocks,),
        (_THREADS_PER_BLOCK,),
        (h_m, active_mask, rainfall, infiltration, dt, steps,
         h_out, rain_depth, infil_depth, cp.int32(n)),
    )

    return h_out, rain_depth, infil_depth


def _source_loop_kernel():
    global _SOURCE_LOOP_KERNEL
    if _SOURCE_LOOP_KERNEL is not None:
        return _SOURCE_LOOP_KERNEL

    import cupy as cp

    _SOURCE_LOOP_KERNEL = cp.RawKernel(_SOURCE_LOOP_KERNEL_SOURCE, "cuda_source_loop")
    return _SOURCE_LOOP_KERNEL


_TRIANGULAR_SOURCES_KERNEL_SOURCE = r"""
extern "C" __global__
void triangular_apply_sources_kernel(
    const float* __restrict__ h_in,
    const float* __restrict__ hu_in,
    const float* __restrict__ hv_in,
    const float* __restrict__ active_mask,
    const int has_active_mask,
    const float* __restrict__ infiltration_capacity_mps,
    const int has_infiltration_array,
    const float infiltration_capacity_scalar,
    const float rainfall_rate_mps,
    const float* __restrict__ cell_area_m2,
    const float timestep_s,
    const float min_depth_m,
    float* __restrict__ h_out,
    float* __restrict__ hu_out,
    float* __restrict__ hv_out,
    float* __restrict__ rainfall_depth_m3,
    float* __restrict__ infiltration_depth_m3,
    int cell_count
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cell_count) return;

    float active = has_active_mask ? active_mask[c] : 1.0f;
    float infil_rate = has_infiltration_array ? infiltration_capacity_mps[c] : infiltration_capacity_scalar;

    float rainfall_depth = fmaxf(rainfall_rate_mps, 0.0f) * timestep_s * active;
    float h_before_infiltration = h_in[c] + rainfall_depth;
    float infiltration_depth = fminf(h_before_infiltration, fmaxf(infil_rate, 0.0f) * timestep_s * active);
    float h_new = h_before_infiltration - infiltration_depth;

    float momentum_scale = (h_before_infiltration > min_depth_m)
        ? (h_new / fmaxf(h_before_infiltration, min_depth_m))
        : 0.0f;

    bool wet = h_new > min_depth_m;
    float hu_new = wet ? (hu_in[c] * momentum_scale) : 0.0f;
    float hv_new = wet ? (hv_in[c] * momentum_scale) : 0.0f;

    h_out[c] = h_new;
    hu_out[c] = hu_new;
    hv_out[c] = hv_new;
    rainfall_depth_m3[c] = rainfall_depth * cell_area_m2[c];
    infiltration_depth_m3[c] = infiltration_depth * cell_area_m2[c];
}
"""

_TRIANGULAR_THREADS_PER_BLOCK = 256


def _triangular_blocks(n: int) -> int:
    return (n + _TRIANGULAR_THREADS_PER_BLOCK - 1) // _TRIANGULAR_THREADS_PER_BLOCK


def _triangular_sources_kernel():
    global _TRIANGULAR_SOURCES_KERNEL
    if _TRIANGULAR_SOURCES_KERNEL is not None:
        return _TRIANGULAR_SOURCES_KERNEL

    import cupy as cp

    _TRIANGULAR_SOURCES_KERNEL = cp.RawKernel(_TRIANGULAR_SOURCES_KERNEL_SOURCE, "triangular_apply_sources_kernel")
    return _TRIANGULAR_SOURCES_KERNEL


def cuda_triangular_apply_sources(
    h: Any,
    hu: Any,
    hv: Any,
    active_mask: Any | None,
    infiltration_capacity_mps: Any,
    rainfall_rate_mps: float,
    cell_area_m2: Any,
    timestep_s: float,
    min_depth_m: float,
) -> tuple[Any, Any, Any, Any, Any]:
    """Fused-CUDA equivalent of `triangular_backend._apply_sources`.

    `infiltration_capacity_mps` may be a per-cell CuPy array (the
    production case -- it varies by road/building mask) or a Python
    scalar. Returns `(h_out, hu_out, hv_out, rainfall_input_m3,
    infiltration_loss_m3)`, matching `_apply_sources`'s return contract
    (the last two already reduced via `cp.sum`).
    """

    import cupy as cp

    cell_count = int(h.shape[0])
    h_in = cp.ascontiguousarray(h, dtype=cp.float32)
    hu_in = cp.ascontiguousarray(hu, dtype=cp.float32)
    hv_in = cp.ascontiguousarray(hv, dtype=cp.float32)

    has_active_mask = active_mask is not None
    active_mask_arr = active_mask if has_active_mask else cp.zeros((1,), dtype=cp.float32)

    has_infiltration_array = hasattr(infiltration_capacity_mps, "shape")
    if has_infiltration_array:
        infiltration_arr = cp.ascontiguousarray(infiltration_capacity_mps, dtype=cp.float32)
        infiltration_scalar = 0.0
    else:
        infiltration_arr = cp.zeros((1,), dtype=cp.float32)
        infiltration_scalar = float(infiltration_capacity_mps)

    h_out = cp.empty((cell_count,), dtype=cp.float32)
    hu_out = cp.empty((cell_count,), dtype=cp.float32)
    hv_out = cp.empty((cell_count,), dtype=cp.float32)
    rainfall_depth_m3 = cp.empty((cell_count,), dtype=cp.float32)
    infiltration_depth_m3 = cp.empty((cell_count,), dtype=cp.float32)

    kernel = _triangular_sources_kernel()
    kernel(
        (_triangular_blocks(cell_count),),
        (_TRIANGULAR_THREADS_PER_BLOCK,),
        (
            h_in,
            hu_in,
            hv_in,
            active_mask_arr,
            cp.int32(1 if has_active_mask else 0),
            infiltration_arr,
            cp.int32(1 if has_infiltration_array else 0),
            cp.float32(infiltration_scalar),
            cp.float32(rainfall_rate_mps),
            cp.ascontiguousarray(cell_area_m2, dtype=cp.float32),
            cp.float32(timestep_s),
            cp.float32(min_depth_m),
            h_out,
            hu_out,
            hv_out,
            rainfall_depth_m3,
            infiltration_depth_m3,
            cp.int32(cell_count),
        ),
    )

    return h_out, hu_out, hv_out, cp.sum(rainfall_depth_m3), cp.sum(infiltration_depth_m3)


def triangular_sources_cuda_available() -> bool:
    """Return True if CuPy and at least one CUDA device are available."""

    return cuda_source_loop_available()
