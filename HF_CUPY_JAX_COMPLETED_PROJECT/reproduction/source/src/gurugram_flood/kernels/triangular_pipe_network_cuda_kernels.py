"""Fused CUDA kernels for the triangular dynamic pipe-network link/outfall step.

Replaces the link-flow-update + node-continuity + outfall-discharge block
of `triangular_backend.step_triangular_dynamic_pipe_network` (and the
identical block in `step_triangular_pipe_dry_weather_network`) -- today a
chain of ~15-20 separate CuPy launches over link/node-scale arrays
(typically a few thousand elements, so launch overhead rather than compute
dominates) -- with 4 kernel launches: a per-link flow-update kernel
(inlines `saint_venant_link_flow_update` and the link area/hydraulic-
radius helpers from `pipe_kernels.py`, supporting both the production
precomputed-area path and the circular/Preissmann-slot + rectangular-
open-channel path), a shared atomic-scatter-sum kernel for the donor-node
volume-limiting reduction, a per-link "apply" kernel that finalizes the
signed transfer and scatters it into node volume deltas, and a per-node
outfall-discharge kernel.

`node_volume`, `node_head`, and everything derived from a head difference
(slope, flow, transfer) use `double`: the reference implementation never
downcasts `node_volume_m3` from its native float64, and under heavy
surcharge it can grow into the tens of thousands of m3, where float32's
~7 significant digits measurably lose precision -- both in the
accumulated state itself and in head-difference subtraction once both
heads share a large common offset. Geometry constants (length, area,
radius, diameter, manning's n, rectangular dims) stay float32, matching
the SWE kernels; node/link arrays are tiny, so the extra precision is
free.
"""

from __future__ import annotations

from typing import Any

from gurugram_flood.kernels.scatter_cuda_kernels import cuda_scatter_add_1d_f64

_LINK_FLOW_KERNEL = None
_LINK_APPLY_KERNEL = None
_OUTFALL_KERNEL = None

_LINK_FLOW_KERNEL_SOURCE = r"""
extern "C" __global__
void pipe_link_flow_kernel(
    const double* __restrict__ link_flow_m3ps,
    const double* __restrict__ node_head,
    const int* __restrict__ link_from_node_index,
    const int* __restrict__ link_to_node_index,
    const float* __restrict__ link_length_m,
    const float* __restrict__ link_area_m2,
    const float* __restrict__ link_hydraulic_radius_m,
    const float* __restrict__ link_manning_n,
    const float* __restrict__ link_flow_capacity_m3ps,
    const int has_flow_capacity,
    const float* __restrict__ link_minor_loss_coefficient,
    const int has_minor_loss,
    const float* __restrict__ link_invert_elevation_m,
    const int has_link_invert,
    const float* __restrict__ node_invert_elevation_m,
    const float* __restrict__ link_diameter_m,
    const int has_diameter,
    const float pressurized_wave_speed_mps,
    const int* __restrict__ link_geometry_code,
    const float* __restrict__ link_rect_width_m,
    const float* __restrict__ link_rect_height_m,
    const int has_rect_geometry,
    const float gravity_mps2,
    const float timestep_s,
    double* __restrict__ transfer_m3,
    int* __restrict__ donor_node,
    double* __restrict__ requested_abs,
    int link_count
) {
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    if (l >= link_count) return;

    int from_node = link_from_node_index[l];
    int to_node = link_to_node_index[l];
    double from_head = node_head[from_node];
    double to_head = node_head[to_node];

    double invert = has_link_invert
        ? (double)link_invert_elevation_m[l]
        : fmin((double)node_invert_elevation_m[from_node], (double)node_invert_elevation_m[to_node]);
    double depth = fmax(0.5 * (from_head + to_head) - invert, 0.0);

    float area, radius;
    if (has_diameter) {
        float diameter = fmaxf(link_diameter_m[l], 1.0e-12f);
        float radius_geom = 0.5f * diameter;
        float open_depth = fminf((float)depth, diameter);
        float ratio = fmaxf(fminf((radius_geom - open_depth) / radius_geom, 1.0f), -1.0f);
        float theta = 2.0f * acosf(ratio);
        float open_area = 0.5f * radius_geom * radius_geom * (theta - sinf(theta));
        float full_area = 0.25f * 3.14159265358979323846f * diameter * diameter;
        float circ_area = ((float)depth >= diameter) ? full_area : open_area;
        float full_perimeter = 3.14159265358979323846f * diameter;
        float circ_perimeter = ((float)depth >= diameter) ? full_perimeter : (radius_geom * theta);

        float slot_width = gravity_mps2 * full_area / fmaxf(pressurized_wave_speed_mps * pressurized_wave_speed_mps, 1.0e-12f);
        float pressure_depth = fmaxf((float)depth - diameter, 0.0f);
        float pressurized_area = circ_area + slot_width * pressure_depth;
        float pressurized_radius = circ_area / fmaxf(circ_perimeter, 1.0e-12f);

        area = pressurized_area;
        radius = pressurized_radius;
    } else {
        area = link_area_m2[l];
        radius = link_hydraulic_radius_m[l];
    }

    if (has_rect_geometry && link_geometry_code[l] == 2) {
        float open_depth = fminf(fmaxf((float)depth, 0.0f), fmaxf(link_rect_height_m[l], 1.0e-6f));
        float open_area = fmaxf(link_rect_width_m[l], 1.0e-6f) * open_depth;
        float open_perimeter = fmaxf(link_rect_width_m[l] + 2.0f * open_depth, 1.0e-12f);
        area = open_area;
        radius = open_area / open_perimeter;
    }

    double length = fmax((double)link_length_m[l], 1.0e-6);
    double safe_area = fmax((double)area, 1.0e-12);
    double safe_radius = fmax((double)radius, 1.0e-12);
    double flow = link_flow_m3ps[l];
    double slope = (from_head - to_head) / length;
    double manning_n = fmax((double)link_manning_n[l], 0.0);
    double friction = (double)gravity_mps2 * manning_n * manning_n * fabs(flow) / (pow(safe_radius, 4.0 / 3.0) * safe_area);
    double minor_loss_coef = has_minor_loss ? fmax((double)link_minor_loss_coefficient[l], 0.0) : 0.0;
    double minor_loss = minor_loss_coef * fabs(flow) / (2.0 * length * safe_area);
    double numerator = flow + (double)gravity_mps2 * safe_area * (double)timestep_s * slope;
    double updated = numerator / (1.0 + (double)timestep_s * (friction + minor_loss));
    if (has_flow_capacity) {
        double capacity = fmax((double)link_flow_capacity_m3ps[l], 0.0);
        updated = fmin(fmax(updated, -capacity), capacity);
    }

    double transfer = updated * (double)timestep_s;
    bool positive = transfer >= 0.0;
    transfer_m3[l] = transfer;
    donor_node[l] = positive ? from_node : to_node;
    requested_abs[l] = fabs(transfer);
}
"""

_LINK_APPLY_KERNEL_SOURCE = r"""
extern "C" __global__
void pipe_link_apply_kernel(
    const int* __restrict__ link_from_node_index,
    const int* __restrict__ link_to_node_index,
    const int* __restrict__ donor_node,
    const double* __restrict__ requested_abs,
    const double* __restrict__ requested_by_donor,
    const double* __restrict__ node_volume,
    const float timestep_s,
    double* __restrict__ node_volume_delta,
    double* __restrict__ link_flow_out,
    double* __restrict__ abs_transfer_out,
    int link_count
) {
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    if (l >= link_count) return;

    int donor = donor_node[l];
    int from_node = link_from_node_index[l];
    int to_node = link_to_node_index[l];

    double factor = fmin(1.0, fmax(node_volume[donor], 0.0) / fmax(requested_by_donor[donor], 1.0e-12));
    double limited_abs = requested_abs[l] * factor;
    bool positive = donor == from_node;
    double signed_transfer = positive ? limited_abs : -limited_abs;

    atomicAdd(&node_volume_delta[from_node], -signed_transfer);
    atomicAdd(&node_volume_delta[to_node], signed_transfer);

    link_flow_out[l] = signed_transfer / fmax((double)timestep_s, 1.0e-12);
    abs_transfer_out[l] = limited_abs;
}
"""

_OUTFALL_KERNEL_SOURCE = r"""
extern "C" __global__
void pipe_outfall_kernel(
    const double* __restrict__ node_volume,
    const float* __restrict__ node_invert_elevation_m,
    const float* __restrict__ node_storage_area_m2,
    const float* __restrict__ outfall_area_m2,
    const float* __restrict__ outfall_coefficient,
    const float* __restrict__ outfall_tailwater_head_m,
    const float* __restrict__ outfall_flow_capacity_m3ps,
    const int has_outfall_capacity,
    const float gravity_mps2,
    const float timestep_s,
    double* __restrict__ node_volume_out,
    double* __restrict__ outfall_m3_by_node,
    int node_count
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= node_count) return;

    double volume = fmax(node_volume[n], 0.0);
    double node_head = (double)node_invert_elevation_m[n] + volume / fmax((double)node_storage_area_m2[n], 1.0e-12);

    double delta_head = node_head - (double)outfall_tailwater_head_m[n];
    double sign = (delta_head > 0.0) - (delta_head < 0.0);
    double outfall_q = (double)(outfall_coefficient[n] * outfall_area_m2[n])
        * sqrt(2.0 * (double)gravity_mps2 * fabs(delta_head)) * sign;
    if (has_outfall_capacity) {
        outfall_q = fmin(outfall_q, fmax((double)outfall_flow_capacity_m3ps[n], 0.0));
    }
    double outfall_m3 = fmin(volume, fmax(outfall_q, 0.0) * (double)timestep_s);

    node_volume_out[n] = volume - outfall_m3;
    outfall_m3_by_node[n] = outfall_m3;
}
"""

_THREADS_PER_BLOCK = 256


def _blocks(n: int) -> int:
    return (n + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK


def triangular_pipe_network_cuda_available() -> bool:
    try:
        import cupy as cp  # type: ignore[import-not-found]

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _link_flow_kernel():
    global _LINK_FLOW_KERNEL
    if _LINK_FLOW_KERNEL is not None:
        return _LINK_FLOW_KERNEL
    import cupy as cp

    _LINK_FLOW_KERNEL = cp.RawKernel(_LINK_FLOW_KERNEL_SOURCE, "pipe_link_flow_kernel")
    return _LINK_FLOW_KERNEL


def _link_apply_kernel():
    global _LINK_APPLY_KERNEL
    if _LINK_APPLY_KERNEL is not None:
        return _LINK_APPLY_KERNEL
    import cupy as cp

    _LINK_APPLY_KERNEL = cp.RawKernel(_LINK_APPLY_KERNEL_SOURCE, "pipe_link_apply_kernel")
    return _LINK_APPLY_KERNEL


def _outfall_kernel():
    global _OUTFALL_KERNEL
    if _OUTFALL_KERNEL is not None:
        return _OUTFALL_KERNEL
    import cupy as cp

    _OUTFALL_KERNEL = cp.RawKernel(_OUTFALL_KERNEL_SOURCE, "pipe_outfall_kernel")
    return _OUTFALL_KERNEL


def _node_head(node_volume: Any, invert: Any, storage_area: Any, cp: Any) -> Any:
    return invert + cp.maximum(node_volume, 0.0) / cp.maximum(storage_area, 1.0e-12)


def step_triangular_pipe_links_and_outfalls_cuda(
    coupling_topology: Any,
    pipe_topology: Any,
    node_volume: Any,
    link_flow: Any,
    timestep_s: float,
    gravity_mps2: float,
) -> tuple[Any, Any, Any, Any, Any]:
    """Fused-CUDA equivalent of the link-update + outfall block shared by
    `step_triangular_dynamic_pipe_network` and
    `step_triangular_pipe_dry_weather_network` (everything after inlet
    exchange / dry-weather inflow, i.e. from the Saint-Venant link update
    through the outfall discharge).

    Returns `(node_volume, link_flow, link_abs_flow_m3,
    outfall_discharge_m3, outfall_m3_by_node)`. `node_volume` and
    `link_flow` are double precision throughout (see module docstring).
    Requires `outfall_area_m2`/`outfall_coefficient`/
    `outfall_tailwater_head_m` to be set (the production/test contract) --
    callers must fall back to the reference path if any of those is None.
    """

    import cupy as cp

    node_count = coupling_topology.node_count
    link_count = pipe_topology.link_count

    node_volume = cp.maximum(cp.ascontiguousarray(node_volume, dtype=cp.float64), 0.0)
    link_flow = cp.ascontiguousarray(link_flow, dtype=cp.float64)

    if link_count > 0:
        node_head = _node_head(
            node_volume, coupling_topology.node_invert_elevation_m, coupling_topology.node_storage_area_m2, cp
        )

        has_flow_capacity = pipe_topology.link_flow_capacity_m3ps is not None
        flow_capacity = pipe_topology.link_flow_capacity_m3ps if has_flow_capacity else cp.zeros((1,), dtype=cp.float32)
        has_minor_loss = pipe_topology.link_minor_loss_coefficient is not None
        minor_loss = (
            pipe_topology.link_minor_loss_coefficient if has_minor_loss else cp.zeros((1,), dtype=cp.float32)
        )
        has_link_invert = pipe_topology.link_invert_elevation_m is not None
        link_invert = (
            pipe_topology.link_invert_elevation_m if has_link_invert else cp.zeros((1,), dtype=cp.float32)
        )
        has_diameter = pipe_topology.link_diameter_m is not None
        link_diameter = pipe_topology.link_diameter_m if has_diameter else cp.zeros((1,), dtype=cp.float32)
        has_rect_geometry = (
            pipe_topology.link_geometry_code is not None
            and pipe_topology.link_rect_width_m is not None
            and pipe_topology.link_rect_height_m is not None
        )
        link_geometry_code = (
            pipe_topology.link_geometry_code if has_rect_geometry else cp.zeros((1,), dtype=cp.int32)
        )
        link_rect_width = pipe_topology.link_rect_width_m if has_rect_geometry else cp.zeros((1,), dtype=cp.float32)
        link_rect_height = pipe_topology.link_rect_height_m if has_rect_geometry else cp.zeros((1,), dtype=cp.float32)

        transfer_m3 = cp.empty((link_count,), dtype=cp.float64)
        donor_node = cp.empty((link_count,), dtype=cp.int32)
        requested_abs = cp.empty((link_count,), dtype=cp.float64)

        flow_kernel = _link_flow_kernel()
        flow_kernel(
            (_blocks(link_count),),
            (_THREADS_PER_BLOCK,),
            (
                link_flow,
                node_head,
                pipe_topology.link_from_node_index,
                pipe_topology.link_to_node_index,
                pipe_topology.link_length_m,
                pipe_topology.link_area_m2,
                pipe_topology.link_hydraulic_radius_m,
                pipe_topology.link_manning_n,
                flow_capacity,
                cp.int32(1 if has_flow_capacity else 0),
                minor_loss,
                cp.int32(1 if has_minor_loss else 0),
                link_invert,
                cp.int32(1 if has_link_invert else 0),
                coupling_topology.node_invert_elevation_m,
                link_diameter,
                cp.int32(1 if has_diameter else 0),
                cp.float32(pipe_topology.pressurized_wave_speed_mps),
                link_geometry_code,
                link_rect_width,
                link_rect_height,
                cp.int32(1 if has_rect_geometry else 0),
                cp.float32(gravity_mps2),
                cp.float32(timestep_s),
                transfer_m3,
                donor_node,
                requested_abs,
                cp.int32(link_count),
            ),
        )

        requested_by_donor = cuda_scatter_add_1d_f64(donor_node, requested_abs, node_count)

        node_volume_delta = cp.zeros((node_count,), dtype=cp.float64)
        link_flow_new = cp.empty((link_count,), dtype=cp.float64)
        abs_transfer = cp.empty((link_count,), dtype=cp.float64)

        apply_kernel = _link_apply_kernel()
        apply_kernel(
            (_blocks(link_count),),
            (_THREADS_PER_BLOCK,),
            (
                pipe_topology.link_from_node_index,
                pipe_topology.link_to_node_index,
                donor_node,
                requested_abs,
                requested_by_donor,
                node_volume,
                cp.float32(timestep_s),
                node_volume_delta,
                link_flow_new,
                abs_transfer,
                cp.int32(link_count),
            ),
        )

        node_volume = cp.maximum(node_volume + node_volume_delta, 0.0)
        link_flow = link_flow_new
        link_abs_flow_m3 = cp.sum(abs_transfer)
    else:
        link_abs_flow_m3 = cp.float64(0.0)

    outfall_m3_by_node = cp.empty((node_count,), dtype=cp.float64)
    node_volume_after_outfall = cp.empty((node_count,), dtype=cp.float64)

    has_outfall_capacity = pipe_topology.outfall_flow_capacity_m3ps is not None
    outfall_capacity = (
        pipe_topology.outfall_flow_capacity_m3ps if has_outfall_capacity else cp.zeros((1,), dtype=cp.float32)
    )

    outfall_kernel = _outfall_kernel()
    outfall_kernel(
        (_blocks(node_count),),
        (_THREADS_PER_BLOCK,),
        (
            node_volume,
            coupling_topology.node_invert_elevation_m,
            coupling_topology.node_storage_area_m2,
            pipe_topology.outfall_area_m2,
            pipe_topology.outfall_coefficient,
            pipe_topology.outfall_tailwater_head_m,
            outfall_capacity,
            cp.int32(1 if has_outfall_capacity else 0),
            cp.float32(gravity_mps2),
            cp.float32(timestep_s),
            node_volume_after_outfall,
            outfall_m3_by_node,
            cp.int32(node_count),
        ),
    )

    return node_volume_after_outfall, link_flow, link_abs_flow_m3, cp.sum(outfall_m3_by_node), outfall_m3_by_node
