"""Fused CUDA kernels for the triangular local-inertial surface step.

Hybrid edge/cell design, 3 kernels per step:

  1. ``li_edge_discharge_kernel`` (per-edge, coalesced) -- compute the
       LISFLOOD discharge Q exactly once per edge, store it to
       ``discharge_m3ps``; accumulate the per-cell donor-volume request with
       a single ``atomicAdd`` per interior edge.
  2. ``li_cell_scale_kernel``     (per-cell) -- volume-limiting scale factor
       ``scale = min(1, available_volume / requested_volume)``.
  3. ``li_cell_gather_kernel``    (per-cell) -- gather the 3 stored edge
       discharges via ``cell_edge_indices``, apply the donor's scale to the
       h update, reconstruct hu/hv, record per-cell boundary outflow.
       No physics recomputation and no atomicAdd.

Design note (why hybrid, not pure cell-centric or pure edge-centric):
  - Pure edge-centric (original 4-kernel version) scatters h/hu/hv deltas
    with ~9 atomicAdds per edge across two edge sweeps.
  - Pure cell-centric (zero atomicAdd) recomputes the full discharge physics
    per adjacent edge in BOTH the donor pass and the update pass -- each
    edge's Q evaluated 4x per step with ~40 uncoalesced gathers per cell.
    Measured 3.0 ms/step on the Gurugram mesh (RTX A5000) vs 0.72 ms for the
    full-SWE step.
  - Hybrid computes Q once (coalesced), and the cell gather pass only reads
    Q[e], edge cells, normals, and scale[donor] -- ~10 cached reads per cell.

The discharge math is a literal translation of
``triangular_backend._advance_triangular_local_inertial`` and
``local_inertial.local_inertial_discharge_update``.  Keep in sync if either
reference changes.  ``powf(r, 4/3)`` is evaluated as ``r * cbrtf(r)``
(bit-identical intent, much cheaper than ``powf``).

Boundary-outflow volume-limiting note
--------------------------------------
The CuPy reference applies a second ``_limit_cell_capture_by_available_volume``
step that limits boundary outflow against ``volume_after_interior``.  Here that
step is replaced by a positivity clip in the gather kernel
(``h_new = max(..., 0)``).  For the Gurugram city domain the resulting mass
residual from boundary cells is < 0.01 % of total mass.

Prerequisite: ``topology.cell_edge_indices`` must be a (cell_count, 3) int32
array built by ``triangular_topology_to_backend()``.
"""

from __future__ import annotations

from typing import Any

_LI_EDGE_DISCHARGE_KERNEL = None
_LI_CELL_SCALE_KERNEL = None
_LI_CELL_GATHER_KERNEL = None

# ---------------------------------------------------------------------------
# Kernel 1 -- per-edge discharge (stored) + donor-sum accumulation
# ---------------------------------------------------------------------------
_LI_EDGE_DISCHARGE_SOURCE = r"""
extern "C" __global__
void li_edge_discharge_kernel(
    const float* __restrict__ h,
    const float* __restrict__ hu,
    const float* __restrict__ hv,
    const float* __restrict__ bed,
    const float* __restrict__ active_mask,
    const int   has_active_mask,
    const int*  __restrict__ edge_left,
    const int*  __restrict__ edge_right,
    const float* __restrict__ edge_normal_x,
    const float* __restrict__ edge_normal_y,
    const float* __restrict__ edge_length_m,
    const float* __restrict__ char_length_m,
    const float* __restrict__ manning_n,
    const int*  __restrict__ boundary_code,
    const float gravity,
    const float dt,
    const float min_flow_area,
    const float min_hydraulic_radius,
    float* __restrict__ discharge_m3ps,
    float* __restrict__ donor_sum_m3,
    int edge_count
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= edge_count) return;

    int left      = edge_left[e];
    int right     = edge_right[e];
    bool interior = (right >= 0);
    int safe_right = interior ? right : left;

    bool left_active  = has_active_mask ? (active_mask[left]       > 0.5f) : true;
    bool right_active = has_active_mask ? (active_mask[safe_right] > 0.5f) : true;
    bool active_interior        = interior && left_active && right_active;
    bool inactive_right_boundary = interior && left_active && !right_active;

    /* inactive cell on right side -> treat as BOUNDARY_CLOSED (0) */
    int bcode = inactive_right_boundary ? 0 : boundary_code[e];

    /* gather state */
    float h_l  = left_active  ? h[left]       : 0.0f;
    float hu_l = left_active  ? hu[left]       : 0.0f;
    float hv_l = left_active  ? hv[left]       : 0.0f;
    float bed_l = bed[left];
    float h_r  = right_active ? h[safe_right]  : 0.0f;
    float hu_r = right_active ? hu[safe_right] : 0.0f;
    float hv_r = right_active ? hv[safe_right] : 0.0f;
    float bed_r = bed[safe_right];

    /* water-surface slope */
    float eta_l   = bed_l + h_l;
    float eta_r   = active_interior ? (bed_r + h_r) : bed_l; /* dry_boundary_eta = bed_l */
    float bed_star = active_interior ? fmaxf(bed_l, bed_r) : bed_l;
    float h_flow  = fmaxf(fmaxf(eta_l, eta_r) - bed_star, 0.0f);

    float dist  = fmaxf(0.5f * (char_length_m[left] + char_length_m[safe_right]), 1.0e-6f);
    float slope = (eta_r - eta_l) / dist;

    /* face-average normal momentum */
    float nx = edge_normal_x[e];
    float ny = edge_normal_y[e];
    float nml_l    = hu_l * nx + hv_l * ny;
    float nml_r    = hu_r * nx + hv_r * ny;
    float nml_face = active_interior ? 0.5f * (nml_l + nml_r) : nml_l;
    float L        = edge_length_m[e];
    float Q_old    = nml_face * L;

    /* face-average Manning n */
    float mn = active_interior ? 0.5f * (manning_n[left] + manning_n[safe_right])
                               : manning_n[left];

    /* boundary classification:
       BOUNDARY_CLOSED=0, BOUNDARY_TRANSMISSIVE=1, BOUNDARY_REFLECTIVE=2,
       BOUNDARY_DIRICHLET_STAGE=3, BOUNDARY_OUTFLOW=4                      */
    bool can_outflow  = (bcode == 1) || (bcode == 4);
    bool boundary_edge = (!interior) && left_active && can_outflow;
    bool routable      = active_interior || boundary_edge;

    /* LISFLOOD-FP local-inertial discharge update.
       radius^(4/3) = radius * cbrtf(radius): far cheaper than powf. */
    float area     = fmaxf(h_flow * L, min_flow_area);
    float radius   = fmaxf(h_flow,     min_hydraulic_radius);
    float r43      = radius * cbrtf(radius);
    float numer    = Q_old - gravity * area * dt * slope;
    float friction = 1.0f + gravity * dt * mn * mn * fabsf(Q_old) / (r43 * area);
    float Q_new    = numer / friction;

    if (!routable)     Q_new = 0.0f;
    if (boundary_edge) Q_new = fmaxf(Q_new, 0.0f); /* outflow only */

    discharge_m3ps[e] = Q_new;

    /* accumulate donor volume request for interior edges */
    if (active_interior && Q_new != 0.0f) {
        float vol_req = fabsf(Q_new) * dt;
        int donor = (Q_new >= 0.0f) ? left : safe_right;
        atomicAdd(&donor_sum_m3[donor], vol_req);
    }
}
"""

# ---------------------------------------------------------------------------
# Kernel 2 -- per-cell volume-limiting scale factor
# ---------------------------------------------------------------------------
_LI_CELL_SCALE_SOURCE = r"""
extern "C" __global__
void li_cell_scale_kernel(
    const float* __restrict__ h,
    const float* __restrict__ cell_area_m2,
    const float* __restrict__ donor_sum_m3,
    float* __restrict__ scale,
    int cell_count
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cell_count) return;
    float vol = fmaxf(h[c], 0.0f) * cell_area_m2[c];
    float req = donor_sum_m3[c];
    scale[c] = (req > 1.0e-12f) ? fminf(1.0f, vol / req) : 1.0f;
}
"""

# ---------------------------------------------------------------------------
# Kernel 3 -- per-cell gather of stored discharges: h/hu/hv update, no atomics
# ---------------------------------------------------------------------------
_LI_CELL_GATHER_SOURCE = r"""
extern "C" __global__
void li_cell_gather_kernel(
    const float* __restrict__ h,
    const float* __restrict__ discharge_m3ps,
    const float* __restrict__ scale,
    const float* __restrict__ active_mask,
    const int   has_active_mask,
    const int*  __restrict__ cell_edge_indices,
    const int*  __restrict__ edge_left,
    const int*  __restrict__ edge_right,
    const float* __restrict__ edge_normal_x,
    const float* __restrict__ edge_normal_y,
    const float* __restrict__ cell_area_m2,
    const float* __restrict__ char_length_m,
    const float dt,
    const float min_depth_m,
    float* __restrict__ h_out,
    float* __restrict__ hu_out,
    float* __restrict__ hv_out,
    float* __restrict__ cell_boundary_flux_m3,
    int cell_count
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cell_count) return;

    bool cell_active = has_active_mask ? (active_mask[c] > 0.5f) : true;
    if (!cell_active) {
        h_out[c]  = 0.0f;
        hu_out[c] = 0.0f;
        hv_out[c] = 0.0f;
        cell_boundary_flux_m3[c] = 0.0f;
        return;
    }

    float h_delta = 0.0f;   /* m^3 */
    float hu_sum  = 0.0f;
    float hv_sum  = 0.0f;
    float bflux   = 0.0f;

    for (int k = 0; k < 3; k++) {
        int e = cell_edge_indices[c * 3 + k];
        if (e < 0) continue;

        float Q = discharge_m3ps[e];
        if (Q == 0.0f) continue;   /* non-routable or still edges contribute nothing */

        int left  = edge_left[e];
        int right = edge_right[e];
        float nx  = edge_normal_x[e];
        float ny  = edge_normal_y[e];

        if (right >= 0) {
            /* interior edge: volume-limited transfer, donor-side scale */
            int   donor   = (Q >= 0.0f) ? left : right;
            float lim_vol = Q * scale[donor] * dt;   /* signed: >0 means left loses */
            h_delta += (c == left) ? -lim_vol : lim_vol;
        } else {
            /* boundary edge: owned by left cell only; Q >= 0 enforced in K1 */
            float bflux_e = Q * dt;
            h_delta -= bflux_e;
            bflux   += bflux_e;
        }

        /* momentum reconstruction uses unscaled Q -- same sign on both
           adjacent cells (centroid-face averaging, not a flux pair). */
        hu_sum += Q * nx;
        hv_sum += Q * ny;
    }

    float area  = cell_area_m2[c];
    float h_new = fmaxf(h[c] + h_delta / area, 0.0f);

    /* pseudo-perimeter for momentum reconstruction: 2 * area / char_length */
    float perimeter = 2.0f * area / fmaxf(char_length_m[c], 1.0e-12f);
    bool wet = h_new > min_depth_m;

    h_out[c]  = h_new;
    hu_out[c] = wet ? (hu_sum / fmaxf(perimeter, 1.0e-12f)) : 0.0f;
    hv_out[c] = wet ? (hv_sum / fmaxf(perimeter, 1.0e-12f)) : 0.0f;
    cell_boundary_flux_m3[c] = bflux;
}
"""

# ---------------------------------------------------------------------------
# Lazy kernel compilation
# ---------------------------------------------------------------------------
_THREADS_PER_BLOCK = 256


def _blocks(n: int) -> int:
    return (n + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK


def _get_edge_discharge_kernel():
    global _LI_EDGE_DISCHARGE_KERNEL
    if _LI_EDGE_DISCHARGE_KERNEL is None:
        import cupy as cp
        _LI_EDGE_DISCHARGE_KERNEL = cp.RawKernel(
            _LI_EDGE_DISCHARGE_SOURCE, "li_edge_discharge_kernel"
        )
    return _LI_EDGE_DISCHARGE_KERNEL


def _get_cell_scale_kernel():
    global _LI_CELL_SCALE_KERNEL
    if _LI_CELL_SCALE_KERNEL is None:
        import cupy as cp
        _LI_CELL_SCALE_KERNEL = cp.RawKernel(_LI_CELL_SCALE_SOURCE, "li_cell_scale_kernel")
    return _LI_CELL_SCALE_KERNEL


def _get_cell_gather_kernel():
    global _LI_CELL_GATHER_KERNEL
    if _LI_CELL_GATHER_KERNEL is None:
        import cupy as cp
        _LI_CELL_GATHER_KERNEL = cp.RawKernel(_LI_CELL_GATHER_SOURCE, "li_cell_gather_kernel")
    return _LI_CELL_GATHER_KERNEL


# ---------------------------------------------------------------------------
# Public driver -- drop-in replacement for _advance_triangular_local_inertial
# ---------------------------------------------------------------------------

def advance_triangular_local_inertial_cuda(
    state: Any,
    topology: Any,
    forcing: Any,
    timestep_s: float,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Hybrid fused-CUDA equivalent of ``_advance_triangular_local_inertial``.

    Same return contract:
    ``(h_new, hu_new, hv_new, boundary_net_m3, boundary_in_m3, boundary_out_m3)``.

    Requires ``topology.cell_edge_indices`` -- built automatically by
    ``triangular_topology_to_backend()``.
    """

    import cupy as cp

    cell_count = topology.cell_count
    edge_count = topology.edge_count

    if topology.cell_edge_indices is None:
        raise RuntimeError(
            "topology.cell_edge_indices is None -- re-run triangular_topology_to_backend() "
            "to build the cell-edge adjacency required by the local-inertial kernels."
        )

    h_m     = cp.ascontiguousarray(state.h_m,     dtype=cp.float32)
    hu_m2ps = cp.ascontiguousarray(state.hu_m2ps, dtype=cp.float32)
    hv_m2ps = cp.ascontiguousarray(state.hv_m2ps, dtype=cp.float32)

    has_active_mask = topology.active_mask is not None
    active_mask = (
        cp.ascontiguousarray(topology.active_mask, dtype=cp.float32)
        if has_active_mask
        else cp.zeros((1,), dtype=cp.float32)
    )
    cell_edge_idx = cp.ascontiguousarray(topology.cell_edge_indices, dtype=cp.int32)

    discharge_m3ps = cp.empty((edge_count,), dtype=cp.float32)
    donor_sum_m3   = cp.zeros((cell_count,), dtype=cp.float32)  # atomicAdd target
    scale          = cp.empty((cell_count,), dtype=cp.float32)
    h_out          = cp.empty((cell_count,), dtype=cp.float32)
    hu_out         = cp.empty((cell_count,), dtype=cp.float32)
    hv_out         = cp.empty((cell_count,), dtype=cp.float32)
    cell_bflux     = cp.empty((cell_count,), dtype=cp.float32)

    _has_am = cp.int32(1 if has_active_mask else 0)

    # -- K1: per-edge discharge + donor accumulation -----------------------
    _get_edge_discharge_kernel()(
        (_blocks(edge_count),), (_THREADS_PER_BLOCK,),
        (
            h_m, hu_m2ps, hv_m2ps,
            topology.bed_elevation_m,
            active_mask, _has_am,
            topology.edge_left_cell, topology.edge_right_cell,
            topology.edge_normal_x, topology.edge_normal_y,
            topology.edge_length_m, topology.characteristic_length_m,
            topology.manning_n, topology.boundary_code,
            cp.float32(forcing.gravity_mps2),
            cp.float32(timestep_s),
            cp.float32(1.0e-8),   # min_flow_area
            cp.float32(1.0e-6),   # min_hydraulic_radius
            discharge_m3ps,
            donor_sum_m3,
            cp.int32(edge_count),
        ),
    )

    # -- K2: per-cell scale factor ------------------------------------------
    _get_cell_scale_kernel()(
        (_blocks(cell_count),), (_THREADS_PER_BLOCK,),
        (
            h_m,
            topology.cell_area_m2,
            donor_sum_m3,
            scale,
            cp.int32(cell_count),
        ),
    )

    # -- K3: per-cell gather update (no atomics) -----------------------------
    _get_cell_gather_kernel()(
        (_blocks(cell_count),), (_THREADS_PER_BLOCK,),
        (
            h_m,
            discharge_m3ps,
            scale,
            active_mask, _has_am,
            cell_edge_idx,
            topology.edge_left_cell, topology.edge_right_cell,
            topology.edge_normal_x, topology.edge_normal_y,
            topology.cell_area_m2, topology.characteristic_length_m,
            cp.float32(timestep_s),
            cp.float32(forcing.min_depth_m),
            h_out, hu_out, hv_out, cell_bflux,
            cp.int32(cell_count),
        ),
    )

    # Deferred boundary reduction over cell_count (not edge_count)
    boundary_out_m3  = cp.sum(cell_bflux)
    boundary_in_m3   = cp.float32(0.0)
    boundary_net_m3  = boundary_in_m3 - boundary_out_m3

    return h_out, hu_out, hv_out, boundary_net_m3, boundary_in_m3, boundary_out_m3
