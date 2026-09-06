"""Fused CUDA kernels for triangle <-> 1D-storage-node inlet exchange.

Replaces the ~25-40 separate CuPy elementwise/scatter launches in
`triangular_backend._exchange_triangular_surface_pipe_nodes` /
`exchange_triangular_surface_nodes` (operating on inlet/node-scale arrays,
typically a few thousand elements -- small enough that launch overhead,
not compute, dominates) with 4 kernel launches: a per-inlet "request"
kernel (branches on the 4 capture rules), two shared atomic-scatter-sum
kernels (capacity available per cell / per node), and a per-inlet "apply"
kernel that reads back the summed totals and atomically scatters the
final exchange into per-cell and per-node delta arrays.

Node-volume-scale quantities (`node_volume`, `node_head`, `delta_head`,
and everything derived from them) use `double`, not `float`: the
reference (vectorized CuPy) implementation never downcasts
`node_volume_m3` from its native float64, and under heavy surcharge it
can grow into the tens of thousands of m3, where float32's ~7 significant
digits measurably lose precision step over step. Node/link arrays are
tiny (thousands of elements), so the extra 4 bytes/element is free. Only
the mesh-scale surface arrays (`h_m`, `bed_elevation_m`, `cell_area_m2`,
bounded to a 0-10ish m range) stay float32, matching the SWE kernels.

Also provides a fused kernel for the simple-path
`_return_node_overflow_to_triangles_with_nodes` (the only path the
production solver exercises -- it never sets
`surcharge_cell_index`/`surcharge_node_index`/`surcharge_return_weight`).
The weighted-table path is left to the reference implementation.
"""

from __future__ import annotations

from typing import Any

from gurugram_flood.kernels.scatter_cuda_kernels import cuda_scatter_add_1d_f64

_REQUEST_KERNEL = None
_APPLY_KERNEL = None
_OVERFLOW_RETURN_KERNEL = None

_CAPTURE_RULE_CAPACITY = 0
_CAPTURE_RULE_HEAD_LIMITED_CAPACITY = 1
_CAPTURE_RULE_ORIFICE = 2
_CAPTURE_RULE_WEIR_ORIFICE = 3

_CAPTURE_RULE_CODES = {
    "capacity": _CAPTURE_RULE_CAPACITY,
    "head_limited_capacity": _CAPTURE_RULE_HEAD_LIMITED_CAPACITY,
    "orifice": _CAPTURE_RULE_ORIFICE,
    "weir_orifice": _CAPTURE_RULE_WEIR_ORIFICE,
}

_REQUEST_KERNEL_SOURCE = r"""
extern "C" __global__
void inlet_exchange_request_kernel(
    const float* __restrict__ h_m,
    const float* __restrict__ bed_elevation_m,
    const int* __restrict__ inlet_cell_index,
    const int* __restrict__ inlet_node_index,
    const double* __restrict__ node_volume,
    const float* __restrict__ node_invert_elevation_m,
    const float* __restrict__ node_storage_area_m2,
    const double* __restrict__ inlet_capture_capacity_m3ps,
    const int has_capture_capacity,
    const double* __restrict__ inlet_surcharge_capacity_m3ps,
    const int has_surcharge_capacity,
    const int capture_rule,
    const float inlet_orifice_area_m2,
    const float inlet_orifice_coefficient,
    const float inlet_weir_perimeter_m,
    const float inlet_weir_coefficient,
    const float gravity_mps2,
    const float timestep_s,
    double* __restrict__ requested_capture,
    double* __restrict__ requested_surcharge,
    int inlet_count
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= inlet_count) return;

    int cell = inlet_cell_index[i];
    int node = inlet_node_index[i];

    double surface_head = (double)(bed_elevation_m[cell] + h_m[cell]);
    double node_head = (double)node_invert_elevation_m[node]
        + fmax(node_volume[node], 0.0) / fmax((double)node_storage_area_m2[node], 1.0e-12);
    double delta_head = surface_head - node_head;

    double cap_capacity = has_capture_capacity
        ? fmax(inlet_capture_capacity_m3ps[i], 0.0) * (double)timestep_s
        : 1.0e300;
    double sur_capacity = has_surcharge_capacity
        ? fmax(inlet_surcharge_capacity_m3ps[i], 0.0) * (double)timestep_s
        : 1.0e300;

    double req_cap = 0.0;
    double req_sur = 0.0;

    if (capture_rule == 0) {
        // "capacity"
        req_cap = cap_capacity;
        req_sur = 0.0;
    } else if (capture_rule == 1) {
        // "head_limited_capacity"
        req_cap = (delta_head > 0.0) ? cap_capacity : 0.0;
        req_sur = (delta_head < 0.0) ? sur_capacity : 0.0;
    } else if (capture_rule == 2) {
        // "orifice"
        double exchange_q = (double)(inlet_orifice_coefficient * inlet_orifice_area_m2)
            * sqrt(2.0 * (double)gravity_mps2 * fabs(delta_head))
            * ((delta_head > 0.0) - (delta_head < 0.0));
        req_cap = fmin(fmax(exchange_q, 0.0) * (double)timestep_s, cap_capacity);
        req_sur = fmin(fmax(-exchange_q, 0.0) * (double)timestep_s, sur_capacity);
    } else {
        // "weir_orifice" (default)
        double orifice_q = (double)(inlet_orifice_coefficient * inlet_orifice_area_m2)
            * sqrt(2.0 * (double)gravity_mps2 * fabs(delta_head))
            * ((delta_head > 0.0) - (delta_head < 0.0));
        float h_cell = h_m[cell];
        float equivalent_circular = sqrtf(4.0f * 3.14159265358979323846f * fmaxf(inlet_orifice_area_m2, 0.0f));
        float effective_perimeter = (inlet_weir_perimeter_m > 0.0f) ? inlet_weir_perimeter_m : equivalent_circular;
        float depth = fmaxf(h_cell, 0.0f);
        double weir_q = (double)(inlet_weir_coefficient * fmaxf(effective_perimeter, 0.0f) * depth)
            * sqrt(2.0 * (double)gravity_mps2 * (double)depth);
        req_cap = fmin(fmin(fmax(orifice_q, 0.0), weir_q) * (double)timestep_s, cap_capacity);
        req_sur = fmin(fmax(-orifice_q, 0.0) * (double)timestep_s, sur_capacity);
    }

    requested_capture[i] = fmax(req_cap, 0.0);
    requested_surcharge[i] = fmax(req_sur, 0.0);
}
"""

_APPLY_KERNEL_SOURCE = r"""
extern "C" __global__
void inlet_exchange_apply_kernel(
    const float* __restrict__ h_m,
    const float* __restrict__ cell_area_m2,
    const int* __restrict__ inlet_cell_index,
    const int* __restrict__ inlet_node_index,
    const double* __restrict__ node_volume,
    const double* __restrict__ requested_capture,
    const double* __restrict__ requested_surcharge,
    const double* __restrict__ requested_capture_by_cell,
    const double* __restrict__ requested_surcharge_by_node,
    float* __restrict__ h_delta,
    double* __restrict__ node_volume_delta,
    double* __restrict__ capture_by_cell_m3,
    double* __restrict__ surcharge_by_node_m3,
    double* __restrict__ capture_m3_out,
    double* __restrict__ surcharge_m3_out,
    int inlet_count
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= inlet_count) return;

    int cell = inlet_cell_index[i];
    int node = inlet_node_index[i];

    double cell_available_m3 = (double)fmaxf(h_m[cell], 0.0f) * (double)cell_area_m2[cell];
    double cap_factor = fmin(1.0, cell_available_m3 / fmax(requested_capture_by_cell[cell], 1.0e-12));
    double capture_m3 = requested_capture[i] * cap_factor;

    double node_available_m3 = fmax(node_volume[node], 0.0);
    double sur_factor = fmin(1.0, node_available_m3 / fmax(requested_surcharge_by_node[node], 1.0e-12));
    double surcharge_m3 = requested_surcharge[i] * sur_factor;

    capture_m3_out[i] = capture_m3;
    surcharge_m3_out[i] = surcharge_m3;

    atomicAdd(&capture_by_cell_m3[cell], capture_m3);
    atomicAdd(&surcharge_by_node_m3[node], surcharge_m3);

    double cell_exchange_m3 = surcharge_m3 - capture_m3;
    atomicAdd(&h_delta[cell], (float)(cell_exchange_m3 / (double)cell_area_m2[cell]));
    atomicAdd(&node_volume_delta[node], capture_m3 - surcharge_m3);
}
"""

_OVERFLOW_RETURN_KERNEL_SOURCE = r"""
extern "C" __global__
void node_overflow_return_kernel(
    const double* __restrict__ node_volume,
    const float* __restrict__ node_storage_area_m2,
    const float* __restrict__ node_max_depth_m,
    const int* __restrict__ node_surface_cell_index,
    const float* __restrict__ cell_area_m2,
    float* __restrict__ h_delta,
    double* __restrict__ node_volume_out,
    double* __restrict__ overflow_by_node_m3,
    int node_count
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= node_count) return;

    double capacity = (double)node_storage_area_m2[n] * (double)fmaxf(node_max_depth_m[n], 0.0f);
    double overflow = fmax(node_volume[n] - capacity, 0.0);
    int cell = node_surface_cell_index[n];
    bool valid = cell >= 0;

    double returned = valid ? overflow : 0.0;
    if (valid) {
        atomicAdd(&h_delta[cell], (float)(returned / (double)cell_area_m2[cell]));
    }
    node_volume_out[n] = node_volume[n] - returned;
    overflow_by_node_m3[n] = returned;
}
"""

_THREADS_PER_BLOCK = 256


def _blocks(n: int) -> int:
    return (n + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK


def triangular_inlet_exchange_cuda_available() -> bool:
    try:
        import cupy as cp  # type: ignore[import-not-found]

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _request_kernel():
    global _REQUEST_KERNEL
    if _REQUEST_KERNEL is not None:
        return _REQUEST_KERNEL
    import cupy as cp

    _REQUEST_KERNEL = cp.RawKernel(_REQUEST_KERNEL_SOURCE, "inlet_exchange_request_kernel")
    return _REQUEST_KERNEL


def _apply_kernel():
    global _APPLY_KERNEL
    if _APPLY_KERNEL is not None:
        return _APPLY_KERNEL
    import cupy as cp

    _APPLY_KERNEL = cp.RawKernel(_APPLY_KERNEL_SOURCE, "inlet_exchange_apply_kernel")
    return _APPLY_KERNEL


def _overflow_return_kernel():
    global _OVERFLOW_RETURN_KERNEL
    if _OVERFLOW_RETURN_KERNEL is not None:
        return _OVERFLOW_RETURN_KERNEL
    import cupy as cp

    _OVERFLOW_RETURN_KERNEL = cp.RawKernel(_OVERFLOW_RETURN_KERNEL_SOURCE, "node_overflow_return_kernel")
    return _OVERFLOW_RETURN_KERNEL


def inlet_exchange_cuda(
    h_m: Any,
    surface_topology: Any,
    coupling_topology: Any,
    node_volume: Any,
    timestep_s: float,
    inlet_orifice_area_m2: float,
    inlet_orifice_coefficient: float,
    inlet_weir_perimeter_m: float,
    inlet_weir_coefficient: float,
    gravity_mps2: float,
    inlet_capture_rule: str,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Fused-CUDA equivalent of `_exchange_triangular_surface_pipe_nodes`
    (the `return_by_node=True` shape -- the variant used inside
    `step_triangular_dynamic_pipe_network`). Does NOT include the node
    overflow return step; callers that need it (e.g. the public
    `exchange_triangular_surface_nodes`) call
    `return_node_overflow_to_triangles_cuda` themselves afterward, matching
    the reference functions' own call structure.

    Returns `(h_m, node_volume, capture_total_m3, surcharge_total_m3,
    surface_momentum_scale, surcharge_by_node_m3)`. `node_volume` is
    double precision throughout (see module docstring); `h_m` stays
    float32.
    """

    import cupy as cp

    cell_count = surface_topology.cell_count
    node_count = coupling_topology.node_count
    inlet_count = coupling_topology.inlet_count

    h_m = cp.ascontiguousarray(h_m, dtype=cp.float32)
    node_volume = cp.maximum(cp.ascontiguousarray(node_volume, dtype=cp.float64), 0.0)

    if inlet_count == 0:
        unit_scale = h_m * 0.0 + 1.0
        zero = cp.float64(0.0)
        return h_m, node_volume, zero, zero, unit_scale, node_volume * 0.0

    cell_index = coupling_topology.inlet_cell_index
    node_index = coupling_topology.inlet_node_index

    has_capture_capacity = coupling_topology.inlet_capture_capacity_m3ps is not None
    capture_capacity = (
        cp.ascontiguousarray(coupling_topology.inlet_capture_capacity_m3ps, dtype=cp.float64)
        if has_capture_capacity
        else cp.zeros((1,), dtype=cp.float64)
    )
    has_surcharge_capacity = coupling_topology.inlet_surcharge_capacity_m3ps is not None
    surcharge_capacity = (
        cp.ascontiguousarray(coupling_topology.inlet_surcharge_capacity_m3ps, dtype=cp.float64)
        if has_surcharge_capacity
        else cp.zeros((1,), dtype=cp.float64)
    )

    capture_rule_code = _CAPTURE_RULE_CODES.get(inlet_capture_rule, _CAPTURE_RULE_WEIR_ORIFICE)

    requested_capture = cp.empty((inlet_count,), dtype=cp.float64)
    requested_surcharge = cp.empty((inlet_count,), dtype=cp.float64)

    request_kernel = _request_kernel()
    request_kernel(
        (_blocks(inlet_count),),
        (_THREADS_PER_BLOCK,),
        (
            h_m,
            surface_topology.bed_elevation_m,
            cell_index,
            node_index,
            node_volume,
            coupling_topology.node_invert_elevation_m,
            coupling_topology.node_storage_area_m2,
            capture_capacity,
            cp.int32(1 if has_capture_capacity else 0),
            surcharge_capacity,
            cp.int32(1 if has_surcharge_capacity else 0),
            cp.int32(capture_rule_code),
            cp.float32(inlet_orifice_area_m2),
            cp.float32(inlet_orifice_coefficient),
            cp.float32(inlet_weir_perimeter_m),
            cp.float32(inlet_weir_coefficient),
            cp.float32(gravity_mps2),
            cp.float32(timestep_s),
            requested_capture,
            requested_surcharge,
            cp.int32(inlet_count),
        ),
    )

    requested_capture_by_cell = cuda_scatter_add_1d_f64(cell_index, requested_capture, cell_count)
    requested_surcharge_by_node = cuda_scatter_add_1d_f64(node_index, requested_surcharge, node_count)

    h_delta = cp.zeros((cell_count,), dtype=cp.float32)
    node_volume_delta = cp.zeros((node_count,), dtype=cp.float64)
    capture_by_cell_m3 = cp.zeros((cell_count,), dtype=cp.float64)
    surcharge_by_node_m3 = cp.zeros((node_count,), dtype=cp.float64)
    capture_m3_out = cp.empty((inlet_count,), dtype=cp.float64)
    surcharge_m3_out = cp.empty((inlet_count,), dtype=cp.float64)

    apply_kernel = _apply_kernel()
    apply_kernel(
        (_blocks(inlet_count),),
        (_THREADS_PER_BLOCK,),
        (
            h_m,
            surface_topology.cell_area_m2,
            cell_index,
            node_index,
            node_volume,
            requested_capture,
            requested_surcharge,
            requested_capture_by_cell,
            requested_surcharge_by_node,
            h_delta,
            node_volume_delta,
            capture_by_cell_m3,
            surcharge_by_node_m3,
            capture_m3_out,
            surcharge_m3_out,
            cp.int32(inlet_count),
        ),
    )

    h_m = h_m + h_delta
    node_volume = node_volume + node_volume_delta

    capture_depth_m = capture_by_cell_m3.astype(cp.float32) / surface_topology.cell_area_m2
    surface_momentum_scale = cp.where(
        h_m > 1.0e-12,
        cp.maximum(h_m - capture_depth_m, 0.0) / cp.maximum(h_m, 1.0e-12),
        0.0,
    )

    return (
        h_m,
        node_volume,
        cp.sum(capture_m3_out),
        cp.sum(surcharge_m3_out),
        surface_momentum_scale,
        surcharge_by_node_m3,
    )


def return_node_overflow_to_triangles_cuda(
    h_m: Any,
    surface_topology: Any,
    coupling_topology: Any,
    node_volume: Any,
) -> tuple[Any, Any, Any, Any]:
    """Fused-CUDA equivalent of the simple-path
    `_return_node_overflow_to_triangles_with_nodes` (production never sets
    the weighted surcharge-return table, so only that path is fused here).
    """

    import cupy as cp

    node_count = coupling_topology.node_count
    if node_count == 0 or int(node_volume.shape[0]) == 0:
        zero = cp.sum(h_m * 0.0)
        return h_m, node_volume, zero, node_volume * 0.0

    h_m = cp.ascontiguousarray(h_m, dtype=cp.float32)
    node_volume = cp.ascontiguousarray(node_volume, dtype=cp.float64)

    h_delta = cp.zeros((surface_topology.cell_count,), dtype=cp.float32)
    node_volume_out = cp.empty((node_count,), dtype=cp.float64)
    overflow_by_node_m3 = cp.empty((node_count,), dtype=cp.float64)

    kernel = _overflow_return_kernel()
    kernel(
        (_blocks(node_count),),
        (_THREADS_PER_BLOCK,),
        (
            node_volume,
            coupling_topology.node_storage_area_m2,
            coupling_topology.node_max_depth_m,
            coupling_topology.node_surface_cell_index,
            surface_topology.cell_area_m2,
            h_delta,
            node_volume_out,
            overflow_by_node_m3,
            cp.int32(node_count),
        ),
    )

    h_m = h_m + h_delta
    return h_m, node_volume_out, cp.sum(overflow_by_node_m3), overflow_by_node_m3
