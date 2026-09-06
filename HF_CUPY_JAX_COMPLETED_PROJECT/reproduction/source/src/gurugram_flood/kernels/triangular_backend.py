"""Backend-neutral unstructured triangular shallow-water executor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from gurugram_flood.kernels.equations import hydrostatic_reconstruction_flux_pair, stable_timestep_s, stable_local_inertial_timestep_s
from gurugram_flood.kernels.local_inertial import local_inertial_discharge_update
from gurugram_flood.kernels.network_kernels import (
    limit_link_transfers_by_available_volume,
    limit_node_exchange_by_available_volume,
    node_delta_from_cell_exchange,
    node_delta_from_link_transfers,
    scatter_add_1d,
)
from gurugram_flood.kernels.pipe_kernels import (
    circular_conduit_hydraulic_radius_m,
    preissmann_slot_area_m2,
    saint_venant_link_flow_update,
)


BOUNDARY_CLOSED = 0
BOUNDARY_TRANSMISSIVE = 1
BOUNDARY_REFLECTIVE = 2
BOUNDARY_DIRICHLET_STAGE = 3
BOUNDARY_OUTFLOW = 4

# Single revert switch for the fused-CUDA-kernel hot paths (SWE edge/cell
# update, source/infiltration, inlet exchange, pipe network). When False,
# every dispatch site below falls back to the original vectorized
# CuPy/NumPy reference implementation.
_USE_FUSED_KERNELS = True
_USE_FUSED_SWE = True
_USE_FUSED_LOCAL_INERTIAL = True
_USE_FUSED_SOURCES = True
_USE_FUSED_INLET_EXCHANGE = True
_USE_FUSED_OVERFLOW_RETURN = True
_USE_FUSED_PIPE_LINKS = True


def _is_cupy_array(array: Any) -> bool:
    return type(array).__module__.split(".", maxsplit=1)[0] == "cupy"


@dataclass(frozen=True)
class TriangularSurfaceState:
    """Backend-resident 2D state on triangular finite-volume cells."""

    h_m: Any
    hu_m2ps: Any
    hv_m2ps: Any


@dataclass(frozen=True)
class TriangularSurfaceTopology:
    """Flattened triangle and edge topology for an unstructured SWE step."""

    bed_elevation_m: Any
    manning_n: Any
    cell_area_m2: Any
    characteristic_length_m: Any
    edge_left_cell: Any
    edge_right_cell: Any
    edge_normal_x: Any
    edge_normal_y: Any
    edge_length_m: Any
    boundary_code: Any
    active_mask: Any | None = None
    has_stage_boundary: bool = False
    cell_edge_indices: Any | None = None

    @property
    def cell_count(self) -> int:
        return int(self.cell_area_m2.shape[0])

    @property
    def edge_count(self) -> int:
        return int(self.edge_length_m.shape[0])

    @property
    def boundary_edge_count(self) -> int:
        return int(np.count_nonzero(np.asarray(self.edge_right_cell) < 0))

    @property
    def interior_edge_count(self) -> int:
        return self.edge_count - self.boundary_edge_count


@dataclass(frozen=True)
class TriangularSurfaceForcing:
    """Source terms and coefficients for one triangular surface step."""

    rainfall_rate_mps: Any | float = 0.0
    infiltration_capacity_mps: Any | float = 0.0
    boundary_stage_m: Any | float | None = None
    gravity_mps2: float = 9.80665
    min_depth_m: float = 1.0e-5


@dataclass(frozen=True)
class TriangularSurfaceDiagnostics:
    """Backend-resident volume diagnostics from one triangular step."""

    rainfall_input_m3: Any
    infiltration_loss_m3: Any
    boundary_net_flux_m3: Any
    boundary_inflow_m3: Any
    boundary_outflow_m3: Any
    final_surface_volume_m3: Any


@dataclass(frozen=True)
class TriangularSurfaceMassBalance:
    """Accumulated volume balance for a triangular surface run."""

    step_count: int
    initial_surface_volume_m3: Any
    rainfall_input_m3: Any
    infiltration_loss_m3: Any
    boundary_net_flux_m3: Any
    boundary_inflow_m3: Any
    boundary_outflow_m3: Any
    final_surface_volume_m3: Any
    mass_residual_m3: Any


@dataclass(frozen=True)
class TriangularNodeCouplingTopology:
    """Triangle-to-storage-node 1D/2D exchange topology."""

    inlet_cell_index: Any
    inlet_node_index: Any
    node_surface_cell_index: Any
    node_invert_elevation_m: Any
    node_storage_area_m2: Any
    node_max_depth_m: Any
    node_initial_fill_depth_m: Any | None = None
    inlet_capture_capacity_m3ps: Any | None = None
    inlet_surcharge_capacity_m3ps: Any | None = None
    surcharge_cell_index: Any | None = None
    surcharge_node_index: Any | None = None
    surcharge_return_weight: Any | None = None
    surcharge_return_weight_is_normalized: bool = False
    node_x_m: Any | None = None
    node_y_m: Any | None = None

    @property
    def inlet_count(self) -> int:
        return int(self.inlet_cell_index.shape[0])

    @property
    def node_count(self) -> int:
        return int(self.node_storage_area_m2.shape[0])


@dataclass(frozen=True)
class TriangularNodeExchangeDiagnostics:
    """Backend-resident diagnostics for triangle-node exchange."""

    surface_capture_m3: Any
    surface_surcharge_m3: Any
    overflow_surcharge_m3: Any
    final_node_volume_m3: Any
    surface_momentum_scale: Any | None = None
    surface_surcharge_m3_by_node: Any | None = None
    overflow_surcharge_m3_by_node: Any | None = None


@dataclass(frozen=True)
class TriangularPipeNetworkTopology:
    """Backend-resident dynamic pipe graph for triangular 1D/2D coupling.

    Link flow is positive from `link_from_node_index` to
    `link_to_node_index`. Outfall arrays are per node so terminal and
    intermediate outlets can be represented without a separate sparse table.
    """

    link_from_node_index: Any
    link_to_node_index: Any
    link_length_m: Any
    link_area_m2: Any
    link_hydraulic_radius_m: Any
    link_manning_n: Any
    link_flow_capacity_m3ps: Any | None = None
    outfall_area_m2: Any | None = None
    outfall_coefficient: Any | None = None
    outfall_tailwater_head_m: Any | None = None
    outfall_flow_capacity_m3ps: Any | None = None
    link_invert_elevation_m: Any | None = None
    link_diameter_m: Any | None = None
    link_minor_loss_coefficient: Any | None = None
    link_geometry_code: Any | None = None
    link_rect_width_m: Any | None = None
    link_rect_height_m: Any | None = None
    pressurized_wave_speed_mps: float = 30.0

    @property
    def link_count(self) -> int:
        return int(self.link_length_m.shape[0])


@dataclass(frozen=True)
class TriangularPipeNetworkState:
    """Backend-resident dynamic state for triangular pipe-network coupling."""

    node_volume_m3: Any
    link_flow_m3ps: Any


@dataclass(frozen=True)
class TriangularPipeExchangeDiagnostics:
    """Backend-resident diagnostics for one triangular dynamic pipe step."""

    surface_capture_m3: Any
    surface_surcharge_m3: Any
    link_abs_flow_m3: Any
    outfall_discharge_m3: Any
    overflow_surcharge_m3: Any
    unreturned_overflow_m3: Any
    final_node_volume_m3: Any
    surface_momentum_scale: Any | None = None
    outfall_m3_by_node: Any | None = None
    surface_surcharge_m3_by_node: Any | None = None
    overflow_surcharge_m3_by_node: Any | None = None


@dataclass(frozen=True)
class TriangularPipeDryWeatherDiagnostics:
    """Backend-resident diagnostics for dry-weather sewer warm-up."""

    dry_weather_inflow_m3: Any
    link_abs_flow_m3: Any
    outfall_discharge_m3: Any
    overflow_surcharge_m3: Any
    final_node_volume_m3: Any
    outfall_m3_by_node: Any | None = None
    overflow_surcharge_m3_by_node: Any | None = None


def build_triangular_surface_topology(
    nodes_xy: np.ndarray,
    triangles: np.ndarray,
    bed_elevation_m: np.ndarray,
    manning_n: np.ndarray | float | None = None,
    active_mask: np.ndarray | None = None,
    boundary_mode: str = "closed",
) -> TriangularSurfaceTopology:
    """Build a static edge topology from ANUGA-style node and triangle arrays.

    Triangle vertices are re-oriented counter-clockwise on the host before edge
    normals are generated. Each edge normal points outward from `edge_left_cell`;
    for interior edges this is also the direction from left cell to right cell.
    """

    nodes = np.asarray(nodes_xy, dtype=np.float64)
    tri = np.asarray(triangles, dtype=np.int64)
    bed = np.asarray(bed_elevation_m, dtype=np.float32)
    if nodes.ndim != 2 or nodes.shape[1] != 2:
        raise ValueError("nodes_xy must have shape (node_count, 2).")
    if tri.ndim != 2 or tri.shape[1] != 3:
        raise ValueError("triangles must have shape (triangle_count, 3).")
    if bed.shape != (tri.shape[0],):
        raise ValueError("bed_elevation_m must have one value per triangle.")
    if np.any(tri < 0) or np.any(tri >= nodes.shape[0]):
        raise ValueError("triangles contain node indices outside nodes_xy.")

    oriented_tri, cell_area = _orient_triangles_ccw(nodes, tri)
    if np.any(cell_area <= 1.0e-12):
        raise ValueError("triangles contain degenerate or near-zero-area cells.")

    (
        edge_left_cell,
        edge_right_cell,
        edge_normal_x,
        edge_normal_y,
        edge_length_m,
        perimeter_m,
    ) = _build_edges(nodes, oriented_tri)

    boundary_code = np.full(edge_left_cell.shape, -1, dtype=np.int32)
    boundary_mask = edge_right_cell < 0
    boundary_code[boundary_mask] = boundary_mode_code(boundary_mode)

    if manning_n is None:
        manning = np.full(tri.shape[0], 0.035, dtype=np.float32)
    elif np.isscalar(manning_n):
        manning = np.full(tri.shape[0], float(manning_n), dtype=np.float32)
    else:
        manning = np.asarray(manning_n, dtype=np.float32)
        if manning.shape != (tri.shape[0],):
            raise ValueError("manning_n must be scalar or have one value per triangle.")

    active = None
    if active_mask is not None:
        active = np.asarray(active_mask, dtype=np.float32)
        if active.shape != (tri.shape[0],):
            raise ValueError("active_mask must have one value per triangle.")
        active_bool = active > 0.5
        interior = edge_right_cell >= 0
        swap = interior & (~active_bool[edge_left_cell]) & active_bool[edge_right_cell]
        if np.any(swap):
            original_left = edge_left_cell.copy()
            edge_left_cell[swap] = edge_right_cell[swap]
            edge_right_cell[swap] = original_left[swap]
            edge_normal_x[swap] *= -1.0
            edge_normal_y[swap] *= -1.0

    characteristic_length = 2.0 * cell_area / np.maximum(perimeter_m, 1.0e-12)
    return TriangularSurfaceTopology(
        bed_elevation_m=bed.astype(np.float32),
        manning_n=manning.astype(np.float32),
        cell_area_m2=cell_area.astype(np.float32),
        characteristic_length_m=characteristic_length.astype(np.float32),
        edge_left_cell=edge_left_cell.astype(np.int32),
        edge_right_cell=edge_right_cell.astype(np.int32),
        edge_normal_x=edge_normal_x.astype(np.float32),
        edge_normal_y=edge_normal_y.astype(np.float32),
        edge_length_m=edge_length_m.astype(np.float32),
        boundary_code=boundary_code,
        active_mask=active,
        has_stage_boundary=boundary_mode_code(boundary_mode) == BOUNDARY_DIRICHLET_STAGE,
    )


def triangular_topology_to_backend(topology: TriangularSurfaceTopology, backend: Any) -> TriangularSurfaceTopology:
    """Copy a host triangular topology to the selected array backend."""

    xp = backend.xp if hasattr(backend, "xp") else backend
    float_dtype = getattr(xp, "float32")
    int_dtype = getattr(xp, "int32", int)
    active = None if topology.active_mask is None else xp.array(np.asarray(topology.active_mask), dtype=float_dtype)

    # Build cell→edge adjacency (cell_count, 3) for cell-centric LI kernels.
    # Triangular mesh: every cell has exactly 3 edges. O(E) vectorised.
    if topology.cell_edge_indices is None:
        el_np = np.asarray(topology.edge_left_cell, dtype=np.int32)
        er_np = np.asarray(topology.edge_right_cell, dtype=np.int32)
        n_cells = int(np.asarray(topology.cell_area_m2).shape[0])
        n_edges = el_np.shape[0]
        edge_idx = np.arange(n_edges, dtype=np.int32)
        # Each interior edge contributes to two cells; boundary edge to one.
        all_cells = np.concatenate([el_np, np.where(er_np >= 0, er_np, -1)])
        all_edges = np.concatenate([edge_idx, edge_idx])
        valid = all_cells >= 0
        vc = all_cells[valid]
        ve = all_edges[valid]
        sort_idx = np.argsort(vc, kind="stable")
        vc_s = vc[sort_idx]
        ve_s = ve[sort_idx]
        cell_starts = np.searchsorted(vc_s, np.arange(n_cells, dtype=np.int32))
        slot_idx = np.arange(len(vc_s), dtype=np.int32) - cell_starts[vc_s]
        cei = np.full((n_cells, 3), -1, dtype=np.int32)
        ok = slot_idx < 3
        cei[vc_s[ok], slot_idx[ok]] = ve_s[ok]
        cell_edge_indices = xp.array(cei, dtype=int_dtype)
    else:
        cell_edge_indices = xp.array(np.asarray(topology.cell_edge_indices), dtype=int_dtype)

    return TriangularSurfaceTopology(
        bed_elevation_m=xp.array(np.asarray(topology.bed_elevation_m), dtype=float_dtype),
        manning_n=xp.array(np.asarray(topology.manning_n), dtype=float_dtype),
        cell_area_m2=xp.array(np.asarray(topology.cell_area_m2), dtype=float_dtype),
        characteristic_length_m=xp.array(np.asarray(topology.characteristic_length_m), dtype=float_dtype),
        edge_left_cell=xp.array(np.asarray(topology.edge_left_cell), dtype=int_dtype),
        edge_right_cell=xp.array(np.asarray(topology.edge_right_cell), dtype=int_dtype),
        edge_normal_x=xp.array(np.asarray(topology.edge_normal_x), dtype=float_dtype),
        edge_normal_y=xp.array(np.asarray(topology.edge_normal_y), dtype=float_dtype),
        edge_length_m=xp.array(np.asarray(topology.edge_length_m), dtype=float_dtype),
        boundary_code=xp.array(np.asarray(topology.boundary_code), dtype=int_dtype),
        active_mask=active,
        has_stage_boundary=topology.has_stage_boundary,
        cell_edge_indices=cell_edge_indices,
    )


def triangular_node_coupling_to_backend(
    coupling: TriangularNodeCouplingTopology,
    backend: Any,
) -> TriangularNodeCouplingTopology:
    """Copy a host triangle-node coupling topology to the selected backend."""

    xp = backend.xp if hasattr(backend, "xp") else backend
    float_dtype = getattr(xp, "float32")
    int_dtype = getattr(xp, "int32", int)
    capture_capacity = (
        None
        if coupling.inlet_capture_capacity_m3ps is None
        else xp.array(np.asarray(coupling.inlet_capture_capacity_m3ps), dtype=float_dtype)
    )
    surcharge_capacity = (
        None
        if coupling.inlet_surcharge_capacity_m3ps is None
        else xp.array(np.asarray(coupling.inlet_surcharge_capacity_m3ps), dtype=float_dtype)
    )
    surcharge_cell_index = (
        None
        if coupling.surcharge_cell_index is None
        else xp.array(np.asarray(coupling.surcharge_cell_index), dtype=int_dtype)
    )
    surcharge_node_index = (
        None
        if coupling.surcharge_node_index is None
        else xp.array(np.asarray(coupling.surcharge_node_index), dtype=int_dtype)
    )
    surcharge_return_weight = (
        None
        if coupling.surcharge_return_weight is None
        else xp.array(np.asarray(coupling.surcharge_return_weight), dtype=float_dtype)
    )
    node_x_m = None if coupling.node_x_m is None else xp.array(np.asarray(coupling.node_x_m), dtype=float_dtype)
    node_y_m = None if coupling.node_y_m is None else xp.array(np.asarray(coupling.node_y_m), dtype=float_dtype)
    node_initial_fill_depth_m = (
        None
        if coupling.node_initial_fill_depth_m is None
        else xp.array(np.asarray(coupling.node_initial_fill_depth_m), dtype=float_dtype)
    )
    return TriangularNodeCouplingTopology(
        inlet_cell_index=xp.array(np.asarray(coupling.inlet_cell_index), dtype=int_dtype),
        inlet_node_index=xp.array(np.asarray(coupling.inlet_node_index), dtype=int_dtype),
        node_surface_cell_index=xp.array(np.asarray(coupling.node_surface_cell_index), dtype=int_dtype),
        node_invert_elevation_m=xp.array(np.asarray(coupling.node_invert_elevation_m), dtype=float_dtype),
        node_storage_area_m2=xp.array(np.asarray(coupling.node_storage_area_m2), dtype=float_dtype),
        node_max_depth_m=xp.array(np.asarray(coupling.node_max_depth_m), dtype=float_dtype),
        node_initial_fill_depth_m=node_initial_fill_depth_m,
        inlet_capture_capacity_m3ps=capture_capacity,
        inlet_surcharge_capacity_m3ps=surcharge_capacity,
        surcharge_cell_index=surcharge_cell_index,
        surcharge_node_index=surcharge_node_index,
        surcharge_return_weight=surcharge_return_weight,
        surcharge_return_weight_is_normalized=coupling.surcharge_return_weight_is_normalized,
        node_x_m=node_x_m,
        node_y_m=node_y_m,
    )


def triangular_pipe_network_to_backend(
    pipe_topology: TriangularPipeNetworkTopology,
    backend: Any,
) -> TriangularPipeNetworkTopology:
    """Copy a host triangular pipe-network topology to the selected backend."""

    xp = backend.xp if hasattr(backend, "xp") else backend
    float_dtype = getattr(xp, "float32")
    int_dtype = getattr(xp, "int32", int)

    def optional_float_array(value: Any | None) -> Any | None:
        return None if value is None else xp.array(np.asarray(value), dtype=float_dtype)

    def optional_int_array(value: Any | None) -> Any | None:
        return None if value is None else xp.array(np.asarray(value), dtype=int_dtype)

    return TriangularPipeNetworkTopology(
        link_from_node_index=xp.array(np.asarray(pipe_topology.link_from_node_index), dtype=int_dtype),
        link_to_node_index=xp.array(np.asarray(pipe_topology.link_to_node_index), dtype=int_dtype),
        link_length_m=xp.array(np.asarray(pipe_topology.link_length_m), dtype=float_dtype),
        link_area_m2=xp.array(np.asarray(pipe_topology.link_area_m2), dtype=float_dtype),
        link_hydraulic_radius_m=xp.array(np.asarray(pipe_topology.link_hydraulic_radius_m), dtype=float_dtype),
        link_manning_n=xp.array(np.asarray(pipe_topology.link_manning_n), dtype=float_dtype),
        link_flow_capacity_m3ps=optional_float_array(pipe_topology.link_flow_capacity_m3ps),
        outfall_area_m2=optional_float_array(pipe_topology.outfall_area_m2),
        outfall_coefficient=optional_float_array(pipe_topology.outfall_coefficient),
        outfall_tailwater_head_m=optional_float_array(pipe_topology.outfall_tailwater_head_m),
        outfall_flow_capacity_m3ps=optional_float_array(pipe_topology.outfall_flow_capacity_m3ps),
        link_invert_elevation_m=optional_float_array(pipe_topology.link_invert_elevation_m),
        link_diameter_m=optional_float_array(pipe_topology.link_diameter_m),
        link_minor_loss_coefficient=optional_float_array(pipe_topology.link_minor_loss_coefficient),
        link_geometry_code=optional_int_array(pipe_topology.link_geometry_code),
        link_rect_width_m=optional_float_array(pipe_topology.link_rect_width_m),
        link_rect_height_m=optional_float_array(pipe_topology.link_rect_height_m),
        pressurized_wave_speed_mps=pipe_topology.pressurized_wave_speed_mps,
    )


def step_triangular_surface(
    state: TriangularSurfaceState,
    topology: TriangularSurfaceTopology,
    forcing: TriangularSurfaceForcing,
    timestep_s: float,
) -> tuple[TriangularSurfaceState, TriangularSurfaceDiagnostics]:
    """Advance one explicit finite-volume SWE step on a triangular mesh."""

    if timestep_s <= 0.0:
        raise ValueError("timestep_s must be positive.")
    if topology.has_stage_boundary and forcing.boundary_stage_m is None:
        raise ValueError("dirichlet_stage triangular boundary requires boundary_stage_m.")

    xp = _xp_from_array(state.h_m)
    if _USE_FUSED_KERNELS and _USE_FUSED_SWE and _is_cupy_array(state.h_m):
        from gurugram_flood.kernels.triangular_swe_cuda_kernels import advance_triangular_swe_cuda

        h, hu, hv, boundary_net_m3, boundary_in_m3, boundary_out_m3 = advance_triangular_swe_cuda(
            state=state,
            topology=topology,
            forcing=forcing,
            timestep_s=timestep_s,
        )
    else:
        h, hu, hv, boundary_net_m3, boundary_in_m3, boundary_out_m3 = _advance_triangular_swe(
            state=state,
            topology=topology,
            forcing=forcing,
            timestep_s=timestep_s,
            xp=xp,
        )
    h, hu, hv, rainfall_input_m3, infiltration_loss_m3 = _apply_sources(
        h=h,
        hu=hu,
        hv=hv,
        topology=topology,
        forcing=forcing,
        timestep_s=timestep_s,
        xp=xp,
    )

    new_state = TriangularSurfaceState(h_m=h, hu_m2ps=hu, hv_m2ps=hv)
    diagnostics = TriangularSurfaceDiagnostics(
        rainfall_input_m3=rainfall_input_m3,
        infiltration_loss_m3=infiltration_loss_m3,
        boundary_net_flux_m3=boundary_net_m3,
        boundary_inflow_m3=boundary_in_m3,
        boundary_outflow_m3=boundary_out_m3,
        final_surface_volume_m3=xp.sum(h * topology.cell_area_m2),
    )
    return new_state, diagnostics


def step_triangular_surface_local_inertial(
    state: TriangularSurfaceState,
    topology: TriangularSurfaceTopology,
    forcing: TriangularSurfaceForcing,
    timestep_s: float,
) -> tuple[TriangularSurfaceState, TriangularSurfaceDiagnostics]:
    """Advance one fast local-inertial routing step on a triangular mesh.

    This mode neglects convective acceleration and reconstructs edge
    discharges from local water-surface slopes. It is intended for city-scale
    inundation screening, while ``step_triangular_surface`` remains the full
    SWE reference path.
    """

    if timestep_s <= 0.0:
        raise ValueError("timestep_s must be positive.")
    if topology.has_stage_boundary and forcing.boundary_stage_m is None:
        raise ValueError("dirichlet_stage triangular boundary requires boundary_stage_m.")

    xp = _xp_from_array(state.h_m)
    if _USE_FUSED_KERNELS and _USE_FUSED_LOCAL_INERTIAL and _is_cupy_array(state.h_m):
        from gurugram_flood.kernels.triangular_local_inertial_cuda_kernels import (
            advance_triangular_local_inertial_cuda,
        )
        h, hu, hv, boundary_net_m3, boundary_in_m3, boundary_out_m3 = (
            advance_triangular_local_inertial_cuda(
                state=state,
                topology=topology,
                forcing=forcing,
                timestep_s=timestep_s,
            )
        )
    else:
        h, hu, hv, boundary_net_m3, boundary_in_m3, boundary_out_m3 = (
            _advance_triangular_local_inertial(
                state=state,
                topology=topology,
                forcing=forcing,
                timestep_s=timestep_s,
                xp=xp,
            )
        )
    h, hu, hv, rainfall_input_m3, infiltration_loss_m3 = _apply_sources(
        h=h,
        hu=hu,
        hv=hv,
        topology=topology,
        forcing=forcing,
        timestep_s=timestep_s,
        xp=xp,
    )

    new_state = TriangularSurfaceState(h_m=h, hu_m2ps=hu, hv_m2ps=hv)
    diagnostics = TriangularSurfaceDiagnostics(
        rainfall_input_m3=rainfall_input_m3,
        infiltration_loss_m3=infiltration_loss_m3,
        boundary_net_flux_m3=boundary_net_m3,
        boundary_inflow_m3=boundary_in_m3,
        boundary_outflow_m3=boundary_out_m3,
        final_surface_volume_m3=xp.sum(h * topology.cell_area_m2),
    )
    return new_state, diagnostics


def exchange_triangular_surface_nodes(
    h_m: Any,
    surface_topology: TriangularSurfaceTopology,
    coupling_topology: TriangularNodeCouplingTopology,
    node_volume_m3: Any,
    timestep_s: float,
    inlet_orifice_area_m2: float = 0.04,
    inlet_orifice_coefficient: float = 0.62,
    inlet_weir_perimeter_m: float = 0.0,
    inlet_weir_coefficient: float = 0.45,
    gravity_mps2: float = 9.80665,
    inlet_capture_rule: str = "orifice",
) -> tuple[Any, Any, TriangularNodeExchangeDiagnostics]:
    """Exchange water between triangular surface cells and 1D storage nodes.

    Delegates to `_exchange_triangular_surface_pipe_nodes` (identical
    physics, just with `return_by_node=True`) followed by
    `_return_node_overflow_to_triangles_with_nodes` -- this is the same
    call structure `step_triangular_dynamic_pipe_network` uses, and it
    means both entry points share one fused-CUDA-kernel dispatch path
    instead of duplicating it.
    """

    if timestep_s <= 0.0:
        raise ValueError("timestep_s must be positive.")
    if inlet_capture_rule not in {"orifice", "weir_orifice", "capacity", "head_limited_capacity"}:
        raise ValueError("inlet_capture_rule must be 'orifice', 'weir_orifice', 'capacity', or 'head_limited_capacity'.")
    xp = _xp_from_array(h_m)
    node_volume = xp.maximum(node_volume_m3, 0.0)

    (
        h_m,
        node_volume,
        capture_total_m3,
        surcharge_total_m3,
        surface_momentum_scale,
        surcharge_by_node_m3,
    ) = _exchange_triangular_surface_pipe_nodes(
        h_m=h_m,
        surface_topology=surface_topology,
        coupling_topology=coupling_topology,
        node_volume=node_volume,
        timestep_s=timestep_s,
        inlet_orifice_area_m2=inlet_orifice_area_m2,
        inlet_orifice_coefficient=inlet_orifice_coefficient,
        inlet_weir_perimeter_m=inlet_weir_perimeter_m,
        inlet_weir_coefficient=inlet_weir_coefficient,
        gravity_mps2=gravity_mps2,
        inlet_capture_rule=inlet_capture_rule,
        xp=xp,
        return_by_node=True,
    )
    h_m, node_volume, overflow, overflow_by_node = _return_node_overflow_to_triangles_with_nodes(
        h_m,
        surface_topology=surface_topology,
        coupling_topology=coupling_topology,
        node_volume=node_volume,
        xp=xp,
    )
    return h_m, node_volume, TriangularNodeExchangeDiagnostics(
        surface_capture_m3=capture_total_m3,
        surface_surcharge_m3=surcharge_total_m3,
        overflow_surcharge_m3=overflow,
        final_node_volume_m3=xp.sum(node_volume),
        surface_momentum_scale=surface_momentum_scale,
        surface_surcharge_m3_by_node=surcharge_by_node_m3,
        overflow_surcharge_m3_by_node=overflow_by_node,
    )


def step_triangular_dynamic_pipe_network(
    h_m: Any,
    surface_topology: TriangularSurfaceTopology,
    coupling_topology: TriangularNodeCouplingTopology,
    pipe_topology: TriangularPipeNetworkTopology,
    pipe_state: TriangularPipeNetworkState,
    timestep_s: float,
    inlet_orifice_area_m2: float = 0.04,
    inlet_orifice_coefficient: float = 0.62,
    inlet_weir_perimeter_m: float = 0.0,
    inlet_weir_coefficient: float = 0.45,
    gravity_mps2: float = 9.80665,
    inlet_capture_rule: str = "orifice",
) -> tuple[Any, TriangularPipeNetworkState, TriangularPipeExchangeDiagnostics]:
    """Advance triangular 1D/2D exchange plus dynamic pipe links and outfalls."""

    if timestep_s <= 0.0:
        raise ValueError("timestep_s must be positive.")
    if inlet_capture_rule not in {"orifice", "weir_orifice", "capacity", "head_limited_capacity"}:
        raise ValueError("inlet_capture_rule must be 'orifice', 'weir_orifice', 'capacity', or 'head_limited_capacity'.")
    if int(pipe_state.node_volume_m3.shape[0]) != coupling_topology.node_count:
        raise ValueError("pipe_state.node_volume_m3 must have one value per coupling node.")
    if int(pipe_state.link_flow_m3ps.shape[0]) != pipe_topology.link_count:
        raise ValueError("pipe_state.link_flow_m3ps must have one value per pipe link.")

    xp = _xp_from_array(h_m)
    zero = timestep_s * 0.0
    node_volume = xp.maximum(pipe_state.node_volume_m3, 0.0)
    link_flow = pipe_state.link_flow_m3ps

    (
        h_m,
        node_volume,
        surface_capture_m3,
        surface_surcharge_m3,
        surface_momentum_scale,
        surface_surcharge_by_node_m3,
    ) = _exchange_triangular_surface_pipe_nodes(
        h_m=h_m,
        surface_topology=surface_topology,
        coupling_topology=coupling_topology,
        node_volume=node_volume,
        timestep_s=timestep_s,
        inlet_orifice_area_m2=inlet_orifice_area_m2,
        inlet_orifice_coefficient=inlet_orifice_coefficient,
        inlet_weir_perimeter_m=inlet_weir_perimeter_m,
        inlet_weir_coefficient=inlet_weir_coefficient,
        gravity_mps2=gravity_mps2,
        inlet_capture_rule=inlet_capture_rule,
        xp=xp,
        return_by_node=True,
    )

    can_fuse_links = (
        _USE_FUSED_KERNELS
        and _USE_FUSED_PIPE_LINKS
        and _is_cupy_array(h_m)
        and pipe_topology.outfall_area_m2 is not None
        and pipe_topology.outfall_coefficient is not None
        and pipe_topology.outfall_tailwater_head_m is not None
    )
    if can_fuse_links:
        from gurugram_flood.kernels.triangular_pipe_network_cuda_kernels import (
            step_triangular_pipe_links_and_outfalls_cuda,
        )

        (
            node_volume,
            link_flow,
            link_abs_flow_m3,
            outfall_discharge_m3,
            outfall_m3,
        ) = step_triangular_pipe_links_and_outfalls_cuda(
            coupling_topology=coupling_topology,
            pipe_topology=pipe_topology,
            node_volume=node_volume,
            link_flow=link_flow,
            timestep_s=timestep_s,
            gravity_mps2=gravity_mps2,
        )
    elif pipe_topology.link_count > 0:
        node_head = _node_head(
            node_volume,
            coupling_topology.node_invert_elevation_m,
            coupling_topology.node_storage_area_m2,
            xp,
        )
        from_node = pipe_topology.link_from_node_index
        to_node = pipe_topology.link_to_node_index
        candidate_flow = saint_venant_link_flow_update(
            link_flow,
            from_head_m=node_head[from_node],
            to_head_m=node_head[to_node],
            link_length_m=pipe_topology.link_length_m,
            flow_area_m2=_triangular_dynamic_link_area_m2(pipe_topology, coupling_topology, node_head, xp),
            hydraulic_radius_m=_triangular_dynamic_link_hydraulic_radius_m(
                pipe_topology,
                coupling_topology,
                node_head,
                xp,
            ),
            manning_n=pipe_topology.link_manning_n,
            timestep_s=timestep_s,
            gravity_mps2=gravity_mps2,
            minor_loss_coefficient=_triangular_dynamic_link_minor_loss(pipe_topology),
            flow_capacity_m3ps=pipe_topology.link_flow_capacity_m3ps,
        )
        transfer_m3 = limit_link_transfers_by_available_volume(
            candidate_flow * timestep_s,
            from_node,
            to_node,
            node_volume,
        )
        node_volume = node_volume + node_delta_from_link_transfers(
            transfer_m3,
            from_node,
            to_node,
            coupling_topology.node_count,
        )
        node_volume = xp.maximum(node_volume, 0.0)
        link_flow = transfer_m3 / max(timestep_s, 1.0e-12)
        link_abs_flow_m3 = xp.sum(xp.abs(transfer_m3))
    else:
        link_abs_flow_m3 = zero

    if not can_fuse_links:
        node_head = _node_head(
            node_volume,
            coupling_topology.node_invert_elevation_m,
            coupling_topology.node_storage_area_m2,
            xp,
        )
        outfall_m3 = _triangular_pipe_outfall_discharge_m3(
            pipe_topology=pipe_topology,
            node_head=node_head,
            node_volume=node_volume,
            timestep_s=timestep_s,
            gravity_mps2=gravity_mps2,
            xp=xp,
        )
        node_volume = xp.maximum(node_volume - outfall_m3, 0.0)
        outfall_discharge_m3 = xp.sum(outfall_m3)

    h_m, node_volume, overflow_surcharge_m3, overflow_surcharge_by_node_m3 = _return_node_overflow_to_triangles_with_nodes(
        h_m,
        surface_topology=surface_topology,
        coupling_topology=coupling_topology,
        node_volume=node_volume,
        xp=xp,
    )
    unreturned_overflow_m3 = _triangular_unreturned_node_overflow_m3(coupling_topology, node_volume, xp)
    new_state = TriangularPipeNetworkState(node_volume_m3=node_volume, link_flow_m3ps=link_flow)
    diagnostics = TriangularPipeExchangeDiagnostics(
        surface_capture_m3=surface_capture_m3,
        surface_surcharge_m3=surface_surcharge_m3,
        link_abs_flow_m3=link_abs_flow_m3,
        outfall_discharge_m3=outfall_discharge_m3,
        overflow_surcharge_m3=overflow_surcharge_m3,
        unreturned_overflow_m3=unreturned_overflow_m3,
        final_node_volume_m3=xp.sum(node_volume),
        surface_momentum_scale=surface_momentum_scale,
        outfall_m3_by_node=outfall_m3,
        surface_surcharge_m3_by_node=surface_surcharge_by_node_m3,
        overflow_surcharge_m3_by_node=overflow_surcharge_by_node_m3,
    )
    return h_m, new_state, diagnostics


def step_triangular_pipe_dry_weather_network(
    coupling_topology: TriangularNodeCouplingTopology,
    pipe_topology: TriangularPipeNetworkTopology,
    pipe_state: TriangularPipeNetworkState,
    timestep_s: float,
    node_inflow_m3ps: Any | None = None,
    gravity_mps2: float = 9.80665,
    outfall_tailwater_head_m: Any | None = None,
    cap_node_overflow: bool = True,
) -> tuple[TriangularPipeNetworkState, TriangularPipeDryWeatherDiagnostics]:
    """Advance sewer-only dry-weather flow through the dynamic pipe graph.

    This warm-up deliberately excludes surface inlet exchange. It lets greywater
    fill and route through mapped sewer links and mapped outfalls before rainfall
    starts, while reporting any pre-rain node capacity exceedance as surcharge
    potential instead of silently placing that water on the 2D surface.
    """

    if timestep_s <= 0.0:
        raise ValueError("timestep_s must be positive.")
    if int(pipe_state.node_volume_m3.shape[0]) != coupling_topology.node_count:
        raise ValueError("pipe_state.node_volume_m3 must have one value per coupling node.")
    if int(pipe_state.link_flow_m3ps.shape[0]) != pipe_topology.link_count:
        raise ValueError("pipe_state.link_flow_m3ps must have one value per pipe link.")

    xp = _xp_from_array(pipe_state.node_volume_m3)
    zero = xp.sum(pipe_state.node_volume_m3 * 0.0)
    node_volume = xp.maximum(pipe_state.node_volume_m3, 0.0)
    link_flow = pipe_state.link_flow_m3ps

    if node_inflow_m3ps is None:
        inflow_rate = node_volume * 0.0
    else:
        if int(node_inflow_m3ps.shape[0]) != coupling_topology.node_count:
            raise ValueError("node_inflow_m3ps must have one value per coupling node.")
        inflow_rate = xp.maximum(node_inflow_m3ps, 0.0)
    inflow_m3 = inflow_rate * timestep_s
    node_volume = node_volume + inflow_m3
    dry_weather_inflow_m3 = xp.sum(inflow_m3)

    if pipe_topology.link_count > 0:
        node_head = _node_head(
            node_volume,
            coupling_topology.node_invert_elevation_m,
            coupling_topology.node_storage_area_m2,
            xp,
        )
        from_node = pipe_topology.link_from_node_index
        to_node = pipe_topology.link_to_node_index
        candidate_flow = saint_venant_link_flow_update(
            link_flow,
            from_head_m=node_head[from_node],
            to_head_m=node_head[to_node],
            link_length_m=pipe_topology.link_length_m,
            flow_area_m2=_triangular_dynamic_link_area_m2(pipe_topology, coupling_topology, node_head, xp),
            hydraulic_radius_m=_triangular_dynamic_link_hydraulic_radius_m(
                pipe_topology,
                coupling_topology,
                node_head,
                xp,
            ),
            manning_n=pipe_topology.link_manning_n,
            timestep_s=timestep_s,
            gravity_mps2=gravity_mps2,
            minor_loss_coefficient=_triangular_dynamic_link_minor_loss(pipe_topology),
            flow_capacity_m3ps=pipe_topology.link_flow_capacity_m3ps,
        )
        transfer_m3 = limit_link_transfers_by_available_volume(
            candidate_flow * timestep_s,
            from_node,
            to_node,
            node_volume,
        )
        node_volume = node_volume + node_delta_from_link_transfers(
            transfer_m3,
            from_node,
            to_node,
            coupling_topology.node_count,
        )
        node_volume = xp.maximum(node_volume, 0.0)
        link_flow = transfer_m3 / max(timestep_s, 1.0e-12)
        link_abs_flow_m3 = xp.sum(xp.abs(transfer_m3))
    else:
        link_abs_flow_m3 = zero

    node_head = _node_head(
        node_volume,
        coupling_topology.node_invert_elevation_m,
        coupling_topology.node_storage_area_m2,
        xp,
    )
    outfall_m3 = _triangular_pipe_outfall_discharge_m3(
        pipe_topology=pipe_topology,
        node_head=node_head,
        node_volume=node_volume,
        timestep_s=timestep_s,
        gravity_mps2=gravity_mps2,
        xp=xp,
        outfall_tailwater_head_m=outfall_tailwater_head_m,
    )
    node_volume = xp.maximum(node_volume - outfall_m3, 0.0)
    outfall_discharge_m3 = xp.sum(outfall_m3)

    capacity = coupling_topology.node_storage_area_m2 * xp.maximum(coupling_topology.node_max_depth_m, 0.0)
    overflow_by_node_m3 = xp.maximum(node_volume - capacity, 0.0)
    overflow_surcharge_m3 = xp.sum(overflow_by_node_m3)
    if cap_node_overflow:
        node_volume = xp.minimum(node_volume, capacity)

    new_state = TriangularPipeNetworkState(node_volume_m3=node_volume, link_flow_m3ps=link_flow)
    diagnostics = TriangularPipeDryWeatherDiagnostics(
        dry_weather_inflow_m3=dry_weather_inflow_m3,
        link_abs_flow_m3=link_abs_flow_m3,
        outfall_discharge_m3=outfall_discharge_m3,
        overflow_surcharge_m3=overflow_surcharge_m3,
        final_node_volume_m3=xp.sum(node_volume),
        outfall_m3_by_node=outfall_m3,
        overflow_surcharge_m3_by_node=overflow_by_node_m3,
    )
    return new_state, diagnostics


def stable_triangular_timestep_s(
    topology: TriangularSurfaceTopology,
    state: TriangularSurfaceState,
    cfl: float = 0.45,
    gravity_mps2: float = 9.80665,
    min_depth_m: float = 1.0e-5,
) -> float:
    """Return a global explicit timestep using triangle characteristic lengths."""

    return stable_timestep_s(
        topology.characteristic_length_m,
        state.h_m,
        state.hu_m2ps,
        state.hv_m2ps,
        cfl=cfl,
        gravity_mps2=gravity_mps2,
        min_depth_m=min_depth_m,
    )


def stable_local_inertial_triangular_timestep_s(
    topology: TriangularSurfaceTopology,
    state: TriangularSurfaceState,
    cfl: float = 0.45,
    gravity_mps2: float = 9.80665,
    min_depth_m: float = 1.0e-5,
) -> float:
    """Return a timestep using gravity-wave speed only (no velocity term).

    Valid for local-inertial routing. Gives larger dt and fewer steps than
    ``stable_triangular_timestep_s`` when flow is fast.
    """

    return stable_local_inertial_timestep_s(
        topology.characteristic_length_m,
        state.h_m,
        cfl=cfl,
        gravity_mps2=gravity_mps2,
        min_depth_m=min_depth_m,
    )


def initialize_triangular_mass_balance(
    state: TriangularSurfaceState,
    topology: TriangularSurfaceTopology,
) -> TriangularSurfaceMassBalance:
    """Create a backend-resident mass-balance accumulator."""

    xp = _xp_from_array(state.h_m)
    zero = xp.sum(state.h_m * 0.0)
    initial_surface_volume_m3 = xp.sum(state.h_m * topology.cell_area_m2)
    return TriangularSurfaceMassBalance(
        step_count=0,
        initial_surface_volume_m3=initial_surface_volume_m3,
        rainfall_input_m3=zero,
        infiltration_loss_m3=zero,
        boundary_net_flux_m3=zero,
        boundary_inflow_m3=zero,
        boundary_outflow_m3=zero,
        final_surface_volume_m3=initial_surface_volume_m3,
        mass_residual_m3=zero,
    )


def accumulate_triangular_mass_balance(
    balance: TriangularSurfaceMassBalance,
    diagnostics: TriangularSurfaceDiagnostics,
) -> TriangularSurfaceMassBalance:
    """Return an updated accumulated triangular mass balance."""

    rainfall_input_m3 = balance.rainfall_input_m3 + diagnostics.rainfall_input_m3
    infiltration_loss_m3 = balance.infiltration_loss_m3 + diagnostics.infiltration_loss_m3
    boundary_net_flux_m3 = balance.boundary_net_flux_m3 + diagnostics.boundary_net_flux_m3
    boundary_inflow_m3 = balance.boundary_inflow_m3 + diagnostics.boundary_inflow_m3
    boundary_outflow_m3 = balance.boundary_outflow_m3 + diagnostics.boundary_outflow_m3
    mass_residual_m3 = (
        balance.initial_surface_volume_m3
        + rainfall_input_m3
        + boundary_net_flux_m3
        - infiltration_loss_m3
        - diagnostics.final_surface_volume_m3
    )
    return TriangularSurfaceMassBalance(
        step_count=balance.step_count + 1,
        initial_surface_volume_m3=balance.initial_surface_volume_m3,
        rainfall_input_m3=rainfall_input_m3,
        infiltration_loss_m3=infiltration_loss_m3,
        boundary_net_flux_m3=boundary_net_flux_m3,
        boundary_inflow_m3=boundary_inflow_m3,
        boundary_outflow_m3=boundary_outflow_m3,
        final_surface_volume_m3=diagnostics.final_surface_volume_m3,
        mass_residual_m3=mass_residual_m3,
    )


def boundary_mode_code(mode: str) -> int:
    """Return the integer code used by the triangular edge topology."""

    value = mode.lower().replace("-", "_")
    if value in {"closed", "none", "no_flux", "wall"}:
        return BOUNDARY_CLOSED
    if value in {"transmissive", "open", "free"}:
        return BOUNDARY_TRANSMISSIVE
    if value in {"outflow", "open_outflow", "outflow_only", "no_inflow"}:
        return BOUNDARY_OUTFLOW
    if value in {"reflective"}:
        return BOUNDARY_REFLECTIVE
    if value in {"dirichlet_stage", "stage", "fixed_stage"}:
        return BOUNDARY_DIRICHLET_STAGE
    raise ValueError(f"Unsupported triangular boundary mode: {mode}.")


def _advance_triangular_swe(
    state: TriangularSurfaceState,
    topology: TriangularSurfaceTopology,
    forcing: TriangularSurfaceForcing,
    timestep_s: float,
    xp: Any,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    left = topology.edge_left_cell
    right = topology.edge_right_cell
    interior = right >= 0
    safe_right = xp.where(interior, right, left)
    if topology.active_mask is None:
        left_active = left == left
        right_active = right == right
    else:
        active = topology.active_mask > 0.5
        left_active = active[left]
        right_active = active[safe_right]
    active_interior = interior & left_active & right_active
    inactive_right_boundary = interior & left_active & ~right_active

    h_left = xp.where(left_active, state.h_m[left], 0.0)
    hu_left = xp.where(left_active, state.hu_m2ps[left], 0.0)
    hv_left = xp.where(left_active, state.hv_m2ps[left], 0.0)
    bed_left = topology.bed_elevation_m[left]
    h_right_base = xp.where(right_active, state.h_m[safe_right], 0.0)
    hu_right_base = xp.where(right_active, state.hu_m2ps[safe_right], 0.0)
    hv_right_base = xp.where(right_active, state.hv_m2ps[safe_right], 0.0)
    bed_right_base = topology.bed_elevation_m[safe_right]
    boundary_code = xp.where(inactive_right_boundary, BOUNDARY_CLOSED, topology.boundary_code)

    h_right, hu_right, hv_right, bed_right = _right_edge_state(
        h_left=h_left,
        hu_left=hu_left,
        hv_left=hv_left,
        bed_left=bed_left,
        h_right_base=h_right_base,
        hu_right_base=hu_right_base,
        hv_right_base=hv_right_base,
        bed_right_base=bed_right_base,
        normal_x=topology.edge_normal_x,
        normal_y=topology.edge_normal_y,
        interior=active_interior,
        boundary_code=boundary_code,
        boundary_stage_m=forcing.boundary_stage_m,
        xp=xp,
    )

    left_flux, right_flux = hydrostatic_reconstruction_flux_pair(
        h_left,
        hu_left,
        hv_left,
        bed_left,
        h_right,
        hu_right,
        hv_right,
        bed_right,
        normal_x=topology.edge_normal_x,
        normal_y=topology.edge_normal_y,
        gravity_mps2=forcing.gravity_mps2,
        min_depth_m=forcing.min_depth_m,
    )
    outflow_only_boundary = (~interior) & (boundary_code == BOUNDARY_OUTFLOW)
    boundary_inflow = outflow_only_boundary & (left_flux[0] < 0.0)
    left_flux = (
        xp.where(boundary_inflow, 0.0, left_flux[0]),
        xp.where(boundary_inflow, 0.0, left_flux[1]),
        xp.where(boundary_inflow, 0.0, left_flux[2]),
    )
    edge_length = topology.edge_length_m
    cell_count = topology.cell_count

    h_delta = scatter_add_1d(left, -left_flux[0] * edge_length, cell_count)
    hu_delta = scatter_add_1d(left, -left_flux[1] * edge_length, cell_count)
    hv_delta = scatter_add_1d(left, -left_flux[2] * edge_length, cell_count)

    safe_right_for_scatter = xp.where(interior, right, 0)
    h_delta = h_delta + scatter_add_1d(
        safe_right_for_scatter,
        xp.where(active_interior, right_flux[0] * edge_length, 0.0),
        cell_count,
    )
    hu_delta = hu_delta + scatter_add_1d(
        safe_right_for_scatter,
        xp.where(active_interior, right_flux[1] * edge_length, 0.0),
        cell_count,
    )
    hv_delta = hv_delta + scatter_add_1d(
        safe_right_for_scatter,
        xp.where(active_interior, right_flux[2] * edge_length, 0.0),
        cell_count,
    )

    cell_active = state.h_m == state.h_m if topology.active_mask is None else topology.active_mask > 0.5
    h_new = xp.where(cell_active, xp.maximum(state.h_m + timestep_s * h_delta / topology.cell_area_m2, 0.0), 0.0)
    hu_new = xp.where(cell_active, state.hu_m2ps + timestep_s * hu_delta / topology.cell_area_m2, 0.0)
    hv_new = xp.where(cell_active, state.hv_m2ps + timestep_s * hv_delta / topology.cell_area_m2, 0.0)
    wet = h_new > forcing.min_depth_m
    speed = xp.sqrt((hu_new / xp.maximum(h_new, forcing.min_depth_m)) ** 2 + (hv_new / xp.maximum(h_new, forcing.min_depth_m)) ** 2)
    damping = 1.0 + timestep_s * forcing.gravity_mps2 * topology.manning_n**2 * speed / xp.maximum(
        h_new,
        forcing.min_depth_m,
    ) ** (4.0 / 3.0)
    hu_new = xp.where(wet, hu_new / damping, 0.0)
    hv_new = xp.where(wet, hv_new / damping, 0.0)

    boundary = ~active_interior
    outward_boundary_flux_m3 = xp.where(boundary, left_flux[0] * edge_length * timestep_s, 0.0)
    boundary_in_m3 = xp.sum(xp.maximum(-outward_boundary_flux_m3, 0.0))
    boundary_out_m3 = xp.sum(xp.maximum(outward_boundary_flux_m3, 0.0))
    boundary_net_m3 = boundary_in_m3 - boundary_out_m3
    return h_new, hu_new, hv_new, boundary_net_m3, boundary_in_m3, boundary_out_m3


def _advance_triangular_local_inertial(
    state: TriangularSurfaceState,
    topology: TriangularSurfaceTopology,
    forcing: TriangularSurfaceForcing,
    timestep_s: float,
    xp: Any,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    left = topology.edge_left_cell
    right = topology.edge_right_cell
    interior = right >= 0
    safe_right = xp.where(interior, right, left)
    if topology.active_mask is None:
        left_active = left == left
        right_active = right == right
    else:
        active = topology.active_mask > 0.5
        left_active = active[left]
        right_active = active[safe_right]
    active_interior = interior & left_active & right_active
    inactive_right_boundary = interior & left_active & ~right_active
    boundary_code = xp.where(inactive_right_boundary, BOUNDARY_CLOSED, topology.boundary_code)

    h_left = xp.where(left_active, state.h_m[left], 0.0)
    hu_left = xp.where(left_active, state.hu_m2ps[left], 0.0)
    hv_left = xp.where(left_active, state.hv_m2ps[left], 0.0)
    bed_left = topology.bed_elevation_m[left]
    h_right = xp.where(right_active, state.h_m[safe_right], 0.0)
    hu_right = xp.where(right_active, state.hu_m2ps[safe_right], 0.0)
    hv_right = xp.where(right_active, state.hv_m2ps[safe_right], 0.0)
    bed_right = topology.bed_elevation_m[safe_right]

    eta_left = bed_left + h_left
    eta_right_interior = bed_right + h_right
    dry_boundary_eta = bed_left
    eta_right = xp.where(active_interior, eta_right_interior, dry_boundary_eta)
    bed_star = xp.where(active_interior, xp.maximum(bed_left, bed_right), bed_left)
    h_flow = xp.maximum(xp.maximum(eta_left, eta_right) - bed_star, 0.0)
    edge_length = topology.edge_length_m
    distance = xp.maximum(
        0.5 * (topology.characteristic_length_m[left] + topology.characteristic_length_m[safe_right]),
        1.0e-6,
    )
    water_surface_slope = (eta_right - eta_left) / distance

    normal_momentum_left = hu_left * topology.edge_normal_x + hv_left * topology.edge_normal_y
    normal_momentum_right = hu_right * topology.edge_normal_x + hv_right * topology.edge_normal_y
    normal_momentum_face = xp.where(active_interior, 0.5 * (normal_momentum_left + normal_momentum_right), normal_momentum_left)
    discharge_old_m3ps = normal_momentum_face * edge_length
    manning_face = xp.where(active_interior, 0.5 * (topology.manning_n[left] + topology.manning_n[safe_right]), topology.manning_n[left])
    boundary_can_outflow = (boundary_code == BOUNDARY_OUTFLOW) | (boundary_code == BOUNDARY_TRANSMISSIVE)
    boundary_edge = (~interior) & left_active & boundary_can_outflow
    routable_edge = active_interior | boundary_edge
    discharge_m3ps = local_inertial_discharge_update(
        discharge_old_m3ps=discharge_old_m3ps,
        flow_area_m2=h_flow * edge_length,
        hydraulic_radius_m=h_flow,
        water_surface_slope=water_surface_slope,
        manning_n=manning_face,
        timestep_s=timestep_s,
        gravity_mps2=forcing.gravity_mps2,
    )
    discharge_m3ps = xp.where(routable_edge, discharge_m3ps, 0.0)
    discharge_m3ps = xp.where(boundary_edge, xp.maximum(discharge_m3ps, 0.0), discharge_m3ps)

    cell_volume = xp.maximum(state.h_m, 0.0) * topology.cell_area_m2
    requested_interior_m3 = xp.where(active_interior, discharge_m3ps * timestep_s, 0.0)
    interior_transfer_m3 = limit_link_transfers_by_available_volume(
        requested_interior_m3,
        left,
        safe_right,
        cell_volume,
    )
    interior_volume_delta = node_delta_from_link_transfers(
        interior_transfer_m3,
        left,
        safe_right,
        topology.cell_count,
    )
    volume_after_interior = xp.maximum(cell_volume + interior_volume_delta, 0.0)
    requested_boundary_out_m3 = xp.where(boundary_edge, xp.maximum(discharge_m3ps, 0.0) * timestep_s, 0.0)
    boundary_out_m3_by_edge = _limit_cell_capture_by_available_volume(
        requested_boundary_out_m3,
        left,
        volume_after_interior,
        topology.cell_count,
    )
    volume_delta = interior_volume_delta - scatter_add_1d(left, boundary_out_m3_by_edge, topology.cell_count)
    cell_active = state.h_m == state.h_m if topology.active_mask is None else topology.active_mask > 0.5
    h_new = xp.where(cell_active, xp.maximum(state.h_m + volume_delta / topology.cell_area_m2, 0.0), 0.0)

    perimeter = 2.0 * topology.cell_area_m2 / xp.maximum(topology.characteristic_length_m, 1.0e-12)
    edge_x_m3ps = discharge_m3ps * topology.edge_normal_x
    edge_y_m3ps = discharge_m3ps * topology.edge_normal_y
    hu_sum = scatter_add_1d(left, edge_x_m3ps, topology.cell_count) + scatter_add_1d(
        safe_right,
        xp.where(active_interior, edge_x_m3ps, 0.0),
        topology.cell_count,
    )
    hv_sum = scatter_add_1d(left, edge_y_m3ps, topology.cell_count) + scatter_add_1d(
        safe_right,
        xp.where(active_interior, edge_y_m3ps, 0.0),
        topology.cell_count,
    )
    wet = h_new > forcing.min_depth_m
    hu_new = xp.where(wet, hu_sum / xp.maximum(perimeter, 1.0e-12), 0.0)
    hv_new = xp.where(wet, hv_sum / xp.maximum(perimeter, 1.0e-12), 0.0)

    boundary_out_m3 = xp.sum(boundary_out_m3_by_edge)
    boundary_in_m3 = boundary_out_m3 * 0.0
    boundary_net_m3 = boundary_in_m3 - boundary_out_m3
    return h_new, hu_new, hv_new, boundary_net_m3, boundary_in_m3, boundary_out_m3


def _right_edge_state(
    h_left: Any,
    hu_left: Any,
    hv_left: Any,
    bed_left: Any,
    h_right_base: Any,
    hu_right_base: Any,
    hv_right_base: Any,
    bed_right_base: Any,
    normal_x: Any,
    normal_y: Any,
    interior: Any,
    boundary_code: Any,
    boundary_stage_m: Any | float | None,
    xp: Any,
) -> tuple[Any, Any, Any, Any]:
    normal_momentum = hu_left * normal_x + hv_left * normal_y
    hu_reflect = hu_left - 2.0 * normal_momentum * normal_x
    hv_reflect = hv_left - 2.0 * normal_momentum * normal_y
    stage = 0.0 if boundary_stage_m is None else boundary_stage_m
    h_stage = xp.maximum(stage - bed_left, 0.0)

    reflective = (boundary_code == BOUNDARY_CLOSED) | (boundary_code == BOUNDARY_REFLECTIVE)
    stage_boundary = boundary_code == BOUNDARY_DIRICHLET_STAGE

    h_boundary = xp.where(stage_boundary, h_stage, h_left)
    hu_boundary = xp.where(stage_boundary, 0.0, xp.where(reflective, hu_reflect, hu_left))
    hv_boundary = xp.where(stage_boundary, 0.0, xp.where(reflective, hv_reflect, hv_left))

    h_right = xp.where(interior, h_right_base, h_boundary)
    hu_right = xp.where(interior, hu_right_base, hu_boundary)
    hv_right = xp.where(interior, hv_right_base, hv_boundary)
    bed_right = xp.where(interior, bed_right_base, bed_left)
    return h_right, hu_right, hv_right, bed_right


def _apply_sources(
    h: Any,
    hu: Any,
    hv: Any,
    topology: TriangularSurfaceTopology,
    forcing: TriangularSurfaceForcing,
    timestep_s: float,
    xp: Any,
) -> tuple[Any, Any, Any, Any, Any]:
    if _USE_FUSED_KERNELS and _USE_FUSED_SOURCES and _is_cupy_array(h) and isinstance(forcing.rainfall_rate_mps, (int, float)):
        from gurugram_flood.kernels.cuda_kernels import cuda_triangular_apply_sources

        return cuda_triangular_apply_sources(
            h=h,
            hu=hu,
            hv=hv,
            active_mask=topology.active_mask,
            infiltration_capacity_mps=forcing.infiltration_capacity_mps,
            rainfall_rate_mps=forcing.rainfall_rate_mps,
            cell_area_m2=topology.cell_area_m2,
            timestep_s=timestep_s,
            min_depth_m=forcing.min_depth_m,
        )

    active = h * 0.0 + 1.0 if topology.active_mask is None else topology.active_mask
    rainfall_depth = xp.maximum(forcing.rainfall_rate_mps, 0.0) * timestep_s * active
    h = h + rainfall_depth
    h_before_infiltration = h
    infiltration_depth = xp.minimum(
        h,
        xp.maximum(forcing.infiltration_capacity_mps, 0.0) * timestep_s * active,
    )
    h = h - infiltration_depth
    momentum_scale = xp.where(
        h_before_infiltration > forcing.min_depth_m,
        h / xp.maximum(h_before_infiltration, forcing.min_depth_m),
        0.0,
    )
    hu = hu * momentum_scale
    hv = hv * momentum_scale
    wet = h > forcing.min_depth_m
    hu = xp.where(wet, hu, 0.0)
    hv = xp.where(wet, hv, 0.0)
    return (
        h,
        hu,
        hv,
        xp.sum(rainfall_depth * topology.cell_area_m2),
        xp.sum(infiltration_depth * topology.cell_area_m2),
    )


def _return_node_overflow_to_triangles(
    h_m: Any,
    surface_topology: TriangularSurfaceTopology,
    coupling_topology: TriangularNodeCouplingTopology,
    node_volume: Any,
    xp: Any,
) -> tuple[Any, Any, Any]:
    h_m, node_volume, overflow_surcharge_m3, _overflow_by_node_m3 = _return_node_overflow_to_triangles_with_nodes(
        h_m=h_m,
        surface_topology=surface_topology,
        coupling_topology=coupling_topology,
        node_volume=node_volume,
        xp=xp,
    )
    return h_m, node_volume, overflow_surcharge_m3


def _return_node_overflow_to_triangles_with_nodes(
    h_m: Any,
    surface_topology: TriangularSurfaceTopology,
    coupling_topology: TriangularNodeCouplingTopology,
    node_volume: Any,
    xp: Any,
) -> tuple[Any, Any, Any, Any]:
    if coupling_topology.node_count == 0 or int(node_volume.shape[0]) == 0:
        return h_m, node_volume, xp.sum(h_m * 0.0), node_volume * 0.0
    if (
        _USE_FUSED_KERNELS
        and _USE_FUSED_OVERFLOW_RETURN
        and _is_cupy_array(h_m)
        and coupling_topology.surcharge_cell_index is None
        and coupling_topology.surcharge_node_index is None
        and coupling_topology.surcharge_return_weight is None
    ):
        from gurugram_flood.kernels.triangular_inlet_exchange_cuda_kernels import (
            return_node_overflow_to_triangles_cuda,
        )

        return return_node_overflow_to_triangles_cuda(h_m, surface_topology, coupling_topology, node_volume)

    capacity = coupling_topology.node_storage_area_m2 * xp.maximum(coupling_topology.node_max_depth_m, 0.0)
    overflow = xp.maximum(node_volume - capacity, 0.0)
    if (
        coupling_topology.surcharge_cell_index is not None
        and coupling_topology.surcharge_node_index is not None
        and coupling_topology.surcharge_return_weight is not None
    ):
        entry_node = coupling_topology.surcharge_node_index
        entry_cell = coupling_topology.surcharge_cell_index
        valid = (entry_node >= 0) & (entry_cell >= 0)
        safe_node = xp.where(valid, entry_node, 0)
        safe_cell = xp.where(valid, entry_cell, 0)
        raw_weight = xp.where(valid, xp.maximum(coupling_topology.surcharge_return_weight, 0.0), 0.0)
        if coupling_topology.surcharge_return_weight_is_normalized:
            normalized_weight = raw_weight
        else:
            weight_sum_by_node = scatter_add_1d(safe_node, raw_weight, coupling_topology.node_count)
            normalized_weight = xp.where(
                valid,
                raw_weight / xp.maximum(weight_sum_by_node[safe_node], 1.0e-12),
                0.0,
            )
        returned_by_entry = overflow[safe_node] * normalized_weight
        h_m = h_m + scatter_add_1d(
            safe_cell,
            returned_by_entry / surface_topology.cell_area_m2[safe_cell],
            surface_topology.cell_count,
        )
        returned_by_node = scatter_add_1d(safe_node, returned_by_entry, coupling_topology.node_count)
        returned_by_node = xp.minimum(returned_by_node, overflow)
        node_volume = node_volume - returned_by_node

        # Many production drainage nodes do not have explicit surcharge-entry
        # rows, but they still carry a valid nearest surface receiver cell.
        # Without this per-node fallback, their overflow remains in 1D storage
        # and is counted as unreturned mass every step.
        residual_overflow = xp.maximum(node_volume - capacity, 0.0)
        has_explicit_surcharge = scatter_add_1d(
            safe_node,
            xp.where(valid, raw_weight * 0.0 + 1.0, 0.0),
            coupling_topology.node_count,
        ) > 0.0
        fallback_valid = (~has_explicit_surcharge) & (coupling_topology.node_surface_cell_index >= 0)
        fallback_cell = xp.where(fallback_valid, coupling_topology.node_surface_cell_index, 0)
        fallback_returned = xp.where(fallback_valid, residual_overflow, 0.0)
        h_m = h_m + scatter_add_1d(
            fallback_cell,
            fallback_returned / surface_topology.cell_area_m2[fallback_cell],
            surface_topology.cell_count,
        )
        node_volume = node_volume - fallback_returned
        total_returned_by_node = returned_by_node + fallback_returned
        return h_m, node_volume, xp.sum(total_returned_by_node), total_returned_by_node

    valid = coupling_topology.node_surface_cell_index >= 0
    safe_cell = xp.where(valid, coupling_topology.node_surface_cell_index, 0)
    returned = xp.where(valid, overflow, 0.0)
    h_m = h_m + scatter_add_1d(
        safe_cell,
        returned / surface_topology.cell_area_m2[safe_cell],
        surface_topology.cell_count,
    )
    node_volume = node_volume - returned
    return h_m, node_volume, xp.sum(returned), returned


def _exchange_triangular_surface_pipe_nodes(
    h_m: Any,
    surface_topology: TriangularSurfaceTopology,
    coupling_topology: TriangularNodeCouplingTopology,
    node_volume: Any,
    timestep_s: float,
    inlet_orifice_area_m2: float,
    inlet_orifice_coefficient: float,
    inlet_weir_perimeter_m: float,
    inlet_weir_coefficient: float,
    gravity_mps2: float,
    inlet_capture_rule: str,
    xp: Any,
    return_by_node: bool = False,
) -> tuple[Any, ...]:
    if _USE_FUSED_KERNELS and _USE_FUSED_INLET_EXCHANGE and _is_cupy_array(h_m):
        from gurugram_flood.kernels.triangular_inlet_exchange_cuda_kernels import inlet_exchange_cuda

        result = inlet_exchange_cuda(
            h_m=h_m,
            surface_topology=surface_topology,
            coupling_topology=coupling_topology,
            node_volume=node_volume,
            timestep_s=timestep_s,
            inlet_orifice_area_m2=inlet_orifice_area_m2,
            inlet_orifice_coefficient=inlet_orifice_coefficient,
            inlet_weir_perimeter_m=inlet_weir_perimeter_m,
            inlet_weir_coefficient=inlet_weir_coefficient,
            gravity_mps2=gravity_mps2,
            inlet_capture_rule=inlet_capture_rule,
        )
        return result if return_by_node else result[:5]

    zero = timestep_s * 0.0
    if coupling_topology.inlet_count == 0:
        unit_scale = h_m * 0.0 + 1.0
        if return_by_node:
            return h_m, node_volume, zero, zero, unit_scale, node_volume * 0.0
        return h_m, node_volume, zero, zero, unit_scale

    cell_index = coupling_topology.inlet_cell_index
    node_index = coupling_topology.inlet_node_index
    surface_head = surface_topology.bed_elevation_m[cell_index] + h_m[cell_index]
    if inlet_capture_rule == "capacity":
        requested_capture = _inlet_capacity_for_step(
            coupling_topology.inlet_capture_capacity_m3ps,
            surface_head * 0.0,
            timestep_s,
            xp,
        )
        requested_surcharge = requested_capture * 0.0
    elif inlet_capture_rule == "head_limited_capacity":
        node_head = _node_head(
            node_volume,
            coupling_topology.node_invert_elevation_m,
            coupling_topology.node_storage_area_m2,
            xp,
        )
        delta_head = surface_head - node_head[node_index]
        requested_capture = xp.where(
            delta_head > 0.0,
            _inlet_capacity_for_step(coupling_topology.inlet_capture_capacity_m3ps, delta_head, timestep_s, xp),
            0.0,
        )
        requested_surcharge = xp.where(
            delta_head < 0.0,
            _inlet_capacity_for_step(coupling_topology.inlet_surcharge_capacity_m3ps, -delta_head, timestep_s, xp),
            0.0,
        )
    elif inlet_capture_rule == "orifice":
        node_head = _node_head(
            node_volume,
            coupling_topology.node_invert_elevation_m,
            coupling_topology.node_storage_area_m2,
            xp,
        )
        delta_head = surface_head - node_head[node_index]
        exchange_q = _orifice_q(
            delta_head,
            inlet_orifice_area_m2,
            inlet_orifice_coefficient,
            gravity_mps2,
            xp,
        )
        requested_capture = xp.maximum(exchange_q, 0.0) * timestep_s
        requested_capture = xp.minimum(
            requested_capture,
            _inlet_capacity_for_step(coupling_topology.inlet_capture_capacity_m3ps, requested_capture, timestep_s, xp),
        )
        requested_surcharge = xp.maximum(-exchange_q, 0.0) * timestep_s
        requested_surcharge = xp.minimum(
            requested_surcharge,
            _inlet_capacity_for_step(
                coupling_topology.inlet_surcharge_capacity_m3ps,
                requested_surcharge,
                timestep_s,
                xp,
            ),
        )
    else:
        node_head = _node_head(
            node_volume,
            coupling_topology.node_invert_elevation_m,
            coupling_topology.node_storage_area_m2,
            xp,
        )
        delta_head = surface_head - node_head[node_index]
        orifice_q = _orifice_q(
            delta_head,
            inlet_orifice_area_m2,
            inlet_orifice_coefficient,
            gravity_mps2,
            xp,
        )
        weir_q = _weir_capture_q(
            h_m[cell_index],
            _effective_weir_perimeter_m(inlet_orifice_area_m2, inlet_weir_perimeter_m, xp),
            inlet_weir_coefficient,
            gravity_mps2,
            xp,
        )
        requested_capture = xp.minimum(xp.maximum(orifice_q, 0.0), weir_q) * timestep_s
        requested_capture = xp.minimum(
            requested_capture,
            _inlet_capacity_for_step(coupling_topology.inlet_capture_capacity_m3ps, requested_capture, timestep_s, xp),
        )
        requested_surcharge = xp.maximum(-orifice_q, 0.0) * timestep_s
        requested_surcharge = xp.minimum(
            requested_surcharge,
            _inlet_capacity_for_step(
                coupling_topology.inlet_surcharge_capacity_m3ps,
                requested_surcharge,
                timestep_s,
                xp,
            ),
        )

    capture_m3 = _limit_cell_capture_by_indexed_available_volume(
        requested_capture,
        cell_index,
        h_m[cell_index] * surface_topology.cell_area_m2[cell_index],
        surface_topology.cell_count,
    )
    capture_by_cell_m3 = scatter_add_1d(cell_index, capture_m3, surface_topology.cell_count)
    capture_depth_m = capture_by_cell_m3 / surface_topology.cell_area_m2
    surface_momentum_scale = xp.where(
        h_m > 1.0e-12,
        xp.maximum(h_m - capture_depth_m, 0.0) / xp.maximum(h_m, 1.0e-12),
        0.0,
    )
    surcharge_m3 = limit_node_exchange_by_available_volume(requested_surcharge, node_index, node_volume)
    surcharge_by_node_m3 = scatter_add_1d(node_index, surcharge_m3, coupling_topology.node_count)
    cell_exchange = surcharge_m3 - capture_m3
    h_m = h_m + scatter_add_1d(
        cell_index,
        cell_exchange / surface_topology.cell_area_m2[cell_index],
        surface_topology.cell_count,
    )
    node_volume = node_volume + node_delta_from_cell_exchange(
        capture_m3 - surcharge_m3,
        node_index,
        coupling_topology.node_count,
    )
    if return_by_node:
        return h_m, node_volume, xp.sum(capture_m3), xp.sum(surcharge_m3), surface_momentum_scale, surcharge_by_node_m3
    return h_m, node_volume, xp.sum(capture_m3), xp.sum(surcharge_m3), surface_momentum_scale


def _triangular_dynamic_link_depth_m(
    pipe_topology: TriangularPipeNetworkTopology,
    coupling_topology: TriangularNodeCouplingTopology,
    node_head: Any,
    xp: Any,
) -> Any:
    from_node = pipe_topology.link_from_node_index
    to_node = pipe_topology.link_to_node_index
    invert = (
        pipe_topology.link_invert_elevation_m
        if pipe_topology.link_invert_elevation_m is not None
        else xp.minimum(coupling_topology.node_invert_elevation_m[from_node], coupling_topology.node_invert_elevation_m[to_node])
    )
    return xp.maximum(0.5 * (node_head[from_node] + node_head[to_node]) - invert, 0.0)


def _triangular_dynamic_link_area_m2(
    pipe_topology: TriangularPipeNetworkTopology,
    coupling_topology: TriangularNodeCouplingTopology,
    node_head: Any,
    xp: Any,
) -> Any:
    depth = _triangular_dynamic_link_depth_m(pipe_topology, coupling_topology, node_head, xp)
    if pipe_topology.link_diameter_m is None:
        pressurized_area = pipe_topology.link_area_m2
    else:
        pressurized_area = preissmann_slot_area_m2(
            depth,
            pipe_topology.link_diameter_m,
            pressurized_wave_speed_mps=pipe_topology.pressurized_wave_speed_mps,
        )
    if (
        pipe_topology.link_geometry_code is None
        or pipe_topology.link_rect_width_m is None
        or pipe_topology.link_rect_height_m is None
    ):
        return pressurized_area
    open_depth = xp.minimum(xp.maximum(depth, 0.0), xp.maximum(pipe_topology.link_rect_height_m, 1.0e-6))
    open_area = xp.maximum(pipe_topology.link_rect_width_m, 1.0e-6) * open_depth
    return xp.where(pipe_topology.link_geometry_code == 2, open_area, pressurized_area)


def _triangular_dynamic_link_hydraulic_radius_m(
    pipe_topology: TriangularPipeNetworkTopology,
    coupling_topology: TriangularNodeCouplingTopology,
    node_head: Any,
    xp: Any,
) -> Any:
    depth = _triangular_dynamic_link_depth_m(pipe_topology, coupling_topology, node_head, xp)
    if pipe_topology.link_diameter_m is None:
        pressurized_radius = pipe_topology.link_hydraulic_radius_m
    else:
        pressurized_radius = circular_conduit_hydraulic_radius_m(depth, pipe_topology.link_diameter_m)
    if (
        pipe_topology.link_geometry_code is None
        or pipe_topology.link_rect_width_m is None
        or pipe_topology.link_rect_height_m is None
    ):
        return pressurized_radius
    open_depth = xp.minimum(xp.maximum(depth, 0.0), xp.maximum(pipe_topology.link_rect_height_m, 1.0e-6))
    open_area = xp.maximum(pipe_topology.link_rect_width_m, 1.0e-6) * open_depth
    open_perimeter = xp.maximum(pipe_topology.link_rect_width_m + 2.0 * open_depth, 1.0e-12)
    open_radius = open_area / open_perimeter
    return xp.where(pipe_topology.link_geometry_code == 2, open_radius, pressurized_radius)


def _triangular_dynamic_link_minor_loss(pipe_topology: TriangularPipeNetworkTopology) -> Any:
    return 0.0 if pipe_topology.link_minor_loss_coefficient is None else pipe_topology.link_minor_loss_coefficient


def _triangular_pipe_outfall_discharge_m3(
    pipe_topology: TriangularPipeNetworkTopology,
    node_head: Any,
    node_volume: Any,
    timestep_s: float,
    gravity_mps2: float,
    xp: Any,
    outfall_tailwater_head_m: Any | None = None,
) -> Any:
    tailwater_head = (
        pipe_topology.outfall_tailwater_head_m
        if outfall_tailwater_head_m is None
        else outfall_tailwater_head_m
    )
    if (
        pipe_topology.outfall_area_m2 is None
        or pipe_topology.outfall_coefficient is None
        or tailwater_head is None
    ):
        return node_volume * 0.0
    outfall_q = _orifice_q(
        node_head - tailwater_head,
        pipe_topology.outfall_area_m2,
        pipe_topology.outfall_coefficient,
        gravity_mps2,
        xp,
    )
    if pipe_topology.outfall_flow_capacity_m3ps is not None:
        outfall_q = xp.minimum(outfall_q, xp.maximum(pipe_topology.outfall_flow_capacity_m3ps, 0.0))
    return xp.minimum(xp.maximum(node_volume, 0.0), xp.maximum(outfall_q, 0.0) * timestep_s)


def _triangular_unreturned_node_overflow_m3(
    coupling_topology: TriangularNodeCouplingTopology,
    node_volume: Any,
    xp: Any,
) -> Any:
    capacity = coupling_topology.node_storage_area_m2 * xp.maximum(coupling_topology.node_max_depth_m, 0.0)
    return xp.sum(xp.maximum(node_volume - capacity, 0.0))


def _limit_cell_capture_by_available_volume(requested_capture_m3: Any, cell_index: Any, surface_volume_m3: Any, cell_count: int) -> Any:
    xp = _xp_from_array(requested_capture_m3)
    requested = xp.maximum(requested_capture_m3, 0.0)
    requested_by_cell = scatter_add_1d(cell_index, requested, cell_count)
    factor_by_cell = xp.minimum(1.0, surface_volume_m3 / xp.maximum(requested_by_cell, 1.0e-12))
    return requested * factor_by_cell[cell_index]


def _limit_cell_capture_by_indexed_available_volume(
    requested_capture_m3: Any,
    cell_index: Any,
    available_volume_at_index_m3: Any,
    cell_count: int,
) -> Any:
    xp = _xp_from_array(requested_capture_m3)
    requested = xp.maximum(requested_capture_m3, 0.0)
    requested_by_cell = scatter_add_1d(cell_index, requested, cell_count)
    factor_at_index = xp.minimum(
        1.0,
        available_volume_at_index_m3 / xp.maximum(requested_by_cell[cell_index], 1.0e-12),
    )
    return requested * factor_at_index


def _node_head(node_volume: Any, invert: Any, storage_area: Any, xp: Any) -> Any:
    return invert + xp.maximum(node_volume, 0.0) / xp.maximum(storage_area, 1.0e-12)


def _orifice_q(delta_head: Any, area: Any, coefficient: Any, gravity_mps2: float, xp: Any) -> Any:
    return coefficient * area * xp.sqrt(2.0 * gravity_mps2 * xp.abs(delta_head)) * xp.sign(delta_head)


def _weir_capture_q(surface_depth_m: Any, perimeter_m: Any, coefficient: Any, gravity_mps2: float, xp: Any) -> Any:
    depth = xp.maximum(surface_depth_m, 0.0)
    return coefficient * xp.maximum(perimeter_m, 0.0) * depth * xp.sqrt(2.0 * gravity_mps2 * depth)


def _effective_weir_perimeter_m(orifice_area_m2: Any, configured_perimeter_m: Any, xp: Any) -> Any:
    equivalent_circular = xp.sqrt(4.0 * np.pi * xp.maximum(orifice_area_m2, 0.0))
    return xp.where(configured_perimeter_m > 0.0, configured_perimeter_m, equivalent_circular)


def _inlet_capacity_for_step(capacity_m3ps: Any | None, reference: Any, timestep_s: float, xp: Any) -> Any:
    if capacity_m3ps is None:
        return reference * 0.0 + np.inf
    return xp.maximum(capacity_m3ps, 0.0) * timestep_s


def _orient_triangles_ccw(nodes: np.ndarray, triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    oriented = triangles.copy()
    p0 = nodes[oriented[:, 0]]
    p1 = nodes[oriented[:, 1]]
    p2 = nodes[oriented[:, 2]]
    signed_twice_area = _cross2(p1 - p0, p2 - p0)
    clockwise = signed_twice_area < 0.0
    if np.any(clockwise):
        swapped = oriented[clockwise, 1].copy()
        oriented[clockwise, 1] = oriented[clockwise, 2]
        oriented[clockwise, 2] = swapped
        signed_twice_area[clockwise] = -signed_twice_area[clockwise]
    return oriented, 0.5 * signed_twice_area


def _build_edges(nodes: np.ndarray, triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    edge_index_by_key: dict[tuple[int, int], int] = {}
    left_cells: list[int] = []
    right_cells: list[int] = []
    normals_x: list[float] = []
    normals_y: list[float] = []
    lengths: list[float] = []
    perimeter = np.zeros(triangles.shape[0], dtype=np.float64)

    for cell_index, (a, b, c) in enumerate(triangles):
        for start, end in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            p0 = nodes[start]
            p1 = nodes[end]
            edge_vector = p1 - p0
            length = float(np.sqrt(np.dot(edge_vector, edge_vector)))
            if length <= 1.0e-12:
                raise ValueError("triangles contain a zero-length edge.")
            perimeter[cell_index] += length
            key = (start, end) if start < end else (end, start)
            existing_index = edge_index_by_key.get(key)
            if existing_index is None:
                edge_index_by_key[key] = len(left_cells)
                left_cells.append(cell_index)
                right_cells.append(-1)
                normals_x.append(float(edge_vector[1] / length))
                normals_y.append(float(-edge_vector[0] / length))
                lengths.append(length)
                continue
            if right_cells[existing_index] >= 0:
                raise ValueError("triangular topology contains a non-manifold edge shared by more than two cells.")
            right_cells[existing_index] = cell_index

    return (
        np.asarray(left_cells, dtype=np.int32),
        np.asarray(right_cells, dtype=np.int32),
        np.asarray(normals_x, dtype=np.float32),
        np.asarray(normals_y, dtype=np.float32),
        np.asarray(lengths, dtype=np.float32),
        perimeter,
    )


def _cross2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]


def _xp_from_array(array: Any):
    module_name = type(array).__module__
    module_root = module_name.split(".", maxsplit=1)[0]
    if module_root == "cupy":
        import cupy as cp  # type: ignore[import-not-found]

        return cp
    return np
