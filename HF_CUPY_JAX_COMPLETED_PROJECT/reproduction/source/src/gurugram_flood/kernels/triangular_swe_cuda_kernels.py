"""Fused CUDA kernels for the triangular shallow-water surface step.

Replaces `triangular_backend._advance_triangular_swe`'s ~60-80 separate
CuPy elementwise/scatter launches (over up to ~2.5M edges and ~1.7M cells)
with two `cp.RawKernel` launches: one per-edge flux+scatter kernel (inlines
`_right_edge_state`, `hydrostatic_reconstruction_flux_pair`, and
`hll_shallow_water_normal_flux` from `equations.py`, and does the
left/right cell scatter via `atomicAdd` instead of 6 separate
`scatter_add_1d` calls) and one per-cell mass-update + Manning-friction
kernel. The math is a literal translation of
`triangular_backend._advance_triangular_swe` and `equations.py` -- keep the
two in sync if either changes.
"""

from __future__ import annotations

from typing import Any

_EDGE_FLUX_KERNEL = None
_CELL_UPDATE_KERNEL = None

_EDGE_FLUX_KERNEL_SOURCE = r"""
extern "C" __global__
void swe_edge_flux_kernel(
    const float* __restrict__ h,
    const float* __restrict__ hu,
    const float* __restrict__ hv,
    const float* __restrict__ bed,
    const float* __restrict__ active_mask,
    const int has_active_mask,
    const int* __restrict__ edge_left_cell,
    const int* __restrict__ edge_right_cell,
    const float* __restrict__ edge_normal_x,
    const float* __restrict__ edge_normal_y,
    const float* __restrict__ edge_length_m,
    const int* __restrict__ boundary_code,
    const int has_stage_boundary,
    const float boundary_stage_m,
    const float gravity_mps2,
    const float min_depth_m,
    const float timestep_s,
    float* __restrict__ h_delta,
    float* __restrict__ hu_delta,
    float* __restrict__ hv_delta,
    float* __restrict__ boundary_flux_m3_per_edge,
    int edge_count
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= edge_count) return;

    int left = edge_left_cell[e];
    int right = edge_right_cell[e];
    bool interior = right >= 0;
    int safe_right = interior ? right : left;

    bool left_active, right_active;
    if (has_active_mask) {
        left_active = active_mask[left] > 0.5f;
        right_active = active_mask[safe_right] > 0.5f;
    } else {
        left_active = true;
        right_active = true;
    }
    bool active_interior = interior && left_active && right_active;
    bool inactive_right_boundary = interior && left_active && !right_active;

    float h_left = left_active ? h[left] : 0.0f;
    float hu_left = left_active ? hu[left] : 0.0f;
    float hv_left = left_active ? hv[left] : 0.0f;
    float bed_left = bed[left];

    float h_right_base = right_active ? h[safe_right] : 0.0f;
    float hu_right_base = right_active ? hu[safe_right] : 0.0f;
    float hv_right_base = right_active ? hv[safe_right] : 0.0f;
    float bed_right_base = bed[safe_right];

    int bcode = inactive_right_boundary ? 0 : boundary_code[e];

    float nx = edge_normal_x[e];
    float ny = edge_normal_y[e];

    // _right_edge_state
    float normal_momentum = hu_left * nx + hv_left * ny;
    float hu_reflect = hu_left - 2.0f * normal_momentum * nx;
    float hv_reflect = hv_left - 2.0f * normal_momentum * ny;
    float stage = has_stage_boundary ? boundary_stage_m : 0.0f;
    float h_stage = fmaxf(stage - bed_left, 0.0f);
    bool reflective = (bcode == 0) || (bcode == 2);
    bool stage_boundary = (bcode == 3);

    float h_boundary = stage_boundary ? h_stage : h_left;
    float hu_boundary = stage_boundary ? 0.0f : (reflective ? hu_reflect : hu_left);
    float hv_boundary = stage_boundary ? 0.0f : (reflective ? hv_reflect : hv_left);

    // NOTE: the reference (_advance_triangular_swe) calls _right_edge_state
    // with interior=active_interior, not the raw edge `interior` flag, so an
    // edge bordering an inactive (e.g. building) cell is treated as a
    // reflective wall against `left`, not a zeroed/phantom neighbor.
    float h_right = active_interior ? h_right_base : h_boundary;
    float hu_right = active_interior ? hu_right_base : hu_boundary;
    float hv_right = active_interior ? hv_right_base : hv_boundary;
    float bed_right = active_interior ? bed_right_base : bed_left;

    // hydrostatic_reconstruction_flux_pair
    float h_l = fmaxf(h_left, 0.0f);
    float h_r = fmaxf(h_right, 0.0f);
    float eta_l = h_l + bed_left;
    float eta_r = h_r + bed_right;
    float bed_star = fmaxf(bed_left, bed_right);
    float h_l_star = fmaxf(eta_l - bed_star, 0.0f);
    float h_r_star = fmaxf(eta_r - bed_star, 0.0f);

    bool wet_l0 = h_l > min_depth_m;
    bool wet_r0 = h_r > min_depth_m;
    float u_l0 = wet_l0 ? hu_left / fmaxf(h_l, min_depth_m) : 0.0f;
    float v_l0 = wet_l0 ? hv_left / fmaxf(h_l, min_depth_m) : 0.0f;
    float u_r0 = wet_r0 ? hu_right / fmaxf(h_r, min_depth_m) : 0.0f;
    float v_r0 = wet_r0 ? hv_right / fmaxf(h_r, min_depth_m) : 0.0f;

    float hu_l_star = u_l0 * h_l_star;
    float hv_l_star = v_l0 * h_l_star;
    float hu_r_star = u_r0 * h_r_star;
    float hv_r_star = v_r0 * h_r_star;

    // hll_shallow_water_normal_flux (operating on the *_star reconstructed states)
    float hh_l = fmaxf(h_l_star, 0.0f);
    float hh_r = fmaxf(h_r_star, 0.0f);
    bool wet_l = hh_l > min_depth_m;
    bool wet_r = hh_r > min_depth_m;
    float ul = wet_l ? hu_l_star / fmaxf(hh_l, min_depth_m) : 0.0f;
    float vl = wet_l ? hv_l_star / fmaxf(hh_l, min_depth_m) : 0.0f;
    float ur = wet_r ? hu_r_star / fmaxf(hh_r, min_depth_m) : 0.0f;
    float vr = wet_r ? hv_r_star / fmaxf(hh_r, min_depth_m) : 0.0f;

    float un_l = ul * nx + vl * ny;
    float un_r = ur * nx + vr * ny;
    float c_l = sqrtf(gravity_mps2 * hh_l);
    float c_r = sqrtf(gravity_mps2 * hh_r);

    float pressure_l = 0.5f * gravity_mps2 * hh_l * hh_l;
    float pressure_r = 0.5f * gravity_mps2 * hh_r * hh_r;

    float flux_h_l = hh_l * un_l;
    float flux_hu_l = hu_l_star * un_l + pressure_l * nx;
    float flux_hv_l = hv_l_star * un_l + pressure_l * ny;
    float flux_h_r = hh_r * un_r;
    float flux_hu_r = hu_r_star * un_r + pressure_r * nx;
    float flux_hv_r = hv_r_star * un_r + pressure_r * ny;

    float speed_l = fminf(un_l - c_l, un_r - c_r);
    float speed_r = fmaxf(un_l + c_l, un_r + c_r);
    float denom = (fabsf(speed_r - speed_l) > 1.0e-12f) ? (speed_r - speed_l) : 1.0f;

    float hll_h = (speed_r * flux_h_l - speed_l * flux_h_r + speed_l * speed_r * (hh_r - hh_l)) / denom;
    float hll_hu = (speed_r * flux_hu_l - speed_l * flux_hu_r + speed_l * speed_r * (hu_r_star - hu_l_star)) / denom;
    float hll_hv = (speed_r * flux_hv_l - speed_l * flux_hv_r + speed_l * speed_r * (hv_r_star - hv_l_star)) / denom;

    float flux_h, flux_hu, flux_hv;
    if (speed_l >= 0.0f) {
        flux_h = flux_h_l; flux_hu = flux_hu_l; flux_hv = flux_hv_l;
    } else if (speed_r <= 0.0f) {
        flux_h = flux_h_r; flux_hu = flux_hu_r; flux_hv = flux_hv_r;
    } else {
        flux_h = hll_h; flux_hu = hll_hu; flux_hv = hll_hv;
    }

    bool dry_edge = !(wet_l || wet_r);
    if (dry_edge) {
        flux_h = 0.0f; flux_hu = 0.0f; flux_hv = 0.0f;
    }

    float pressure_correction_l = 0.5f * gravity_mps2 * (h_l * h_l - h_l_star * h_l_star);
    float pressure_correction_r = 0.5f * gravity_mps2 * (h_r * h_r - h_r_star * h_r_star);

    float left_flux_h = flux_h;
    float left_flux_hu = flux_hu + pressure_correction_l * nx;
    float left_flux_hv = flux_hv + pressure_correction_l * ny;

    float right_flux_h = flux_h;
    float right_flux_hu = flux_hu + pressure_correction_r * nx;
    float right_flux_hv = flux_hv + pressure_correction_r * ny;

    bool outflow_only_boundary = (!interior) && (bcode == 4);
    bool boundary_inflow = outflow_only_boundary && (left_flux_h < 0.0f);
    if (boundary_inflow) {
        left_flux_h = 0.0f; left_flux_hu = 0.0f; left_flux_hv = 0.0f;
    }

    float edge_length = edge_length_m[e];

    atomicAdd(&h_delta[left], -left_flux_h * edge_length);
    atomicAdd(&hu_delta[left], -left_flux_hu * edge_length);
    atomicAdd(&hv_delta[left], -left_flux_hv * edge_length);

    if (interior && active_interior) {
        atomicAdd(&h_delta[right], right_flux_h * edge_length);
        atomicAdd(&hu_delta[right], right_flux_hu * edge_length);
        atomicAdd(&hv_delta[right], right_flux_hv * edge_length);
    }

    bool boundary = !active_interior;
    boundary_flux_m3_per_edge[e] = boundary ? (left_flux_h * edge_length * timestep_s) : 0.0f;
}
"""

_CELL_UPDATE_KERNEL_SOURCE = r"""
extern "C" __global__
void swe_cell_update_kernel(
    const float* __restrict__ h,
    const float* __restrict__ hu,
    const float* __restrict__ hv,
    const float* __restrict__ h_delta,
    const float* __restrict__ hu_delta,
    const float* __restrict__ hv_delta,
    const float* __restrict__ cell_area_m2,
    const float* __restrict__ manning_n,
    const float* __restrict__ active_mask,
    const int has_active_mask,
    const float timestep_s,
    const float gravity_mps2,
    const float min_depth_m,
    float* __restrict__ h_out,
    float* __restrict__ hu_out,
    float* __restrict__ hv_out,
    int cell_count
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cell_count) return;

    bool cell_active = has_active_mask ? (active_mask[c] > 0.5f) : true;
    float area = cell_area_m2[c];

    float h_raw = h[c] + timestep_s * h_delta[c] / area;
    float hu_raw = hu[c] + timestep_s * hu_delta[c] / area;
    float hv_raw = hv[c] + timestep_s * hv_delta[c] / area;

    float h_new = cell_active ? fmaxf(h_raw, 0.0f) : 0.0f;
    float hu_new = cell_active ? hu_raw : 0.0f;
    float hv_new = cell_active ? hv_raw : 0.0f;

    bool wet = h_new > min_depth_m;
    float h_safe = fmaxf(h_new, min_depth_m);
    float u = hu_new / h_safe;
    float v = hv_new / h_safe;
    float speed = sqrtf(u * u + v * v);
    float mn = manning_n[c];
    float damping = 1.0f + timestep_s * gravity_mps2 * mn * mn * speed / powf(h_safe, 4.0f / 3.0f);

    h_out[c] = h_new;
    hu_out[c] = wet ? (hu_new / damping) : 0.0f;
    hv_out[c] = wet ? (hv_new / damping) : 0.0f;
}
"""

_THREADS_PER_BLOCK = 256


def _blocks(n: int) -> int:
    return (n + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK


def triangular_swe_cuda_available() -> bool:
    """Return True if CuPy and at least one CUDA device are available."""

    try:
        import cupy as cp  # type: ignore[import-not-found]

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _edge_flux_kernel():
    global _EDGE_FLUX_KERNEL
    if _EDGE_FLUX_KERNEL is not None:
        return _EDGE_FLUX_KERNEL
    import cupy as cp

    _EDGE_FLUX_KERNEL = cp.RawKernel(_EDGE_FLUX_KERNEL_SOURCE, "swe_edge_flux_kernel")
    return _EDGE_FLUX_KERNEL


def _cell_update_kernel():
    global _CELL_UPDATE_KERNEL
    if _CELL_UPDATE_KERNEL is not None:
        return _CELL_UPDATE_KERNEL
    import cupy as cp

    _CELL_UPDATE_KERNEL = cp.RawKernel(_CELL_UPDATE_KERNEL_SOURCE, "swe_cell_update_kernel")
    return _CELL_UPDATE_KERNEL


def advance_triangular_swe_cuda(
    state: Any,
    topology: Any,
    forcing: Any,
    timestep_s: float,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Fused-CUDA equivalent of `triangular_backend._advance_triangular_swe`.

    Same return contract: `(h_new, hu_new, hv_new, boundary_net_m3,
    boundary_in_m3, boundary_out_m3)`. All state/topology arrays must
    already be CuPy float32/int32 arrays (i.e. post
    `triangular_topology_to_backend`).
    """

    import cupy as cp

    cell_count = topology.cell_count
    edge_count = topology.edge_count

    # Defensive: the production driver constructs state arrays via
    # `xp.array(...)` without an explicit dtype, so they can come in as
    # float64 even though every topology array (and the documented solver
    # default) is float32. The vectorized CuPy reference path silently
    # upcasts and "just works"; a RawKernel with hardcoded `float*`
    # pointers does not -- it must be given true float32 buffers.
    h_m = cp.ascontiguousarray(state.h_m, dtype=cp.float32)
    hu_m2ps = cp.ascontiguousarray(state.hu_m2ps, dtype=cp.float32)
    hv_m2ps = cp.ascontiguousarray(state.hv_m2ps, dtype=cp.float32)

    h_delta = cp.zeros((cell_count,), dtype=cp.float32)
    hu_delta = cp.zeros((cell_count,), dtype=cp.float32)
    hv_delta = cp.zeros((cell_count,), dtype=cp.float32)
    boundary_flux_m3_per_edge = cp.empty((edge_count,), dtype=cp.float32)

    has_active_mask = topology.active_mask is not None
    active_mask = topology.active_mask if has_active_mask else cp.zeros((1,), dtype=cp.float32)

    has_stage_boundary = bool(topology.has_stage_boundary)
    boundary_stage_m = 0.0 if forcing.boundary_stage_m is None else float(forcing.boundary_stage_m)

    edge_kernel = _edge_flux_kernel()
    edge_kernel(
        (_blocks(edge_count),),
        (_THREADS_PER_BLOCK,),
        (
            h_m,
            hu_m2ps,
            hv_m2ps,
            topology.bed_elevation_m,
            active_mask,
            cp.int32(1 if has_active_mask else 0),
            topology.edge_left_cell,
            topology.edge_right_cell,
            topology.edge_normal_x,
            topology.edge_normal_y,
            topology.edge_length_m,
            topology.boundary_code,
            cp.int32(1 if has_stage_boundary else 0),
            cp.float32(boundary_stage_m),
            cp.float32(forcing.gravity_mps2),
            cp.float32(forcing.min_depth_m),
            cp.float32(timestep_s),
            h_delta,
            hu_delta,
            hv_delta,
            boundary_flux_m3_per_edge,
            cp.int32(edge_count),
        ),
    )

    h_out = cp.empty((cell_count,), dtype=cp.float32)
    hu_out = cp.empty((cell_count,), dtype=cp.float32)
    hv_out = cp.empty((cell_count,), dtype=cp.float32)

    cell_kernel = _cell_update_kernel()
    cell_kernel(
        (_blocks(cell_count),),
        (_THREADS_PER_BLOCK,),
        (
            h_m,
            hu_m2ps,
            hv_m2ps,
            h_delta,
            hu_delta,
            hv_delta,
            topology.cell_area_m2,
            topology.manning_n,
            active_mask,
            cp.int32(1 if has_active_mask else 0),
            cp.float32(timestep_s),
            cp.float32(forcing.gravity_mps2),
            cp.float32(forcing.min_depth_m),
            h_out,
            hu_out,
            hv_out,
            cp.int32(cell_count),
        ),
    )

    boundary_in_m3 = cp.sum(cp.maximum(-boundary_flux_m3_per_edge, 0.0))
    boundary_out_m3 = cp.sum(cp.maximum(boundary_flux_m3_per_edge, 0.0))
    boundary_net_m3 = boundary_in_m3 - boundary_out_m3

    return h_out, hu_out, hv_out, boundary_net_m3, boundary_in_m3, boundary_out_m3
