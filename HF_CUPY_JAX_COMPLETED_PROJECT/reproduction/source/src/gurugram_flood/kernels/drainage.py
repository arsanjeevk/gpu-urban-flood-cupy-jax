"""Drainage inlet and 1D-2D exchange kernels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gurugram_flood.kernels.pipe_kernels import (
    circular_conduit_hydraulic_radius_m,
    preissmann_slot_area_m2,
    saint_venant_link_flow_update,
)
from gurugram_flood.kernels.network_kernels import (
    limit_link_transfers_by_available_volume,
    limit_node_exchange_by_available_volume,
    node_delta_from_link_transfers,
)


@dataclass(frozen=True)
class InletGeometry:
    """Representative inlet geometry for algebraic 1D-2D exchange."""

    area_m2: float
    perimeter_m: float
    weir_coefficient: float = 0.45
    orifice_coefficient: float = 0.62


@dataclass(frozen=True)
class DrainageNetworkGeometry:
    """Flat-array drainage network geometry for conservative 1D-2D coupling.

    `inlet_node_index` maps each 2D surface cell to a node id, or `-1` when
    the cell has no inlet. Each node may route to one downstream node and/or
    discharge through an outfall. Surcharge returns overflow to
    `node_surface_row/col`.
    """

    inlet_node_index: np.ndarray
    node_surface_row: np.ndarray
    node_surface_col: np.ndarray
    downstream_node_index: np.ndarray
    link_capacity_m3ps: np.ndarray
    outfall_capacity_m3ps: np.ndarray
    node_capacity_m3: np.ndarray

    @property
    def node_count(self) -> int:
        return int(self.node_capacity_m3.size)


@dataclass(frozen=True)
class DrainageNetworkState:
    """Dynamic drainage network state and cumulative volume diagnostics."""

    node_volume_m3: np.ndarray
    cumulative_inlet_capture_m3: float = 0.0
    cumulative_link_flow_m3: float = 0.0
    cumulative_outfall_discharge_m3: float = 0.0
    cumulative_surcharge_m3: float = 0.0
    cumulative_unreturned_overflow_m3: float = 0.0


@dataclass(frozen=True)
class DrainageStepResult:
    """Result from one conservative drainage coupling step."""

    surface_depth_m: np.ndarray
    state: DrainageNetworkState
    inlet_capture_m3: float
    link_flow_m3: float
    outfall_discharge_m3: float
    surcharge_m3: float
    unreturned_overflow_m3: float


@dataclass(frozen=True)
class DynamicPipeNetworkGeometry:
    """Head-based 1D pipe network geometry for hydraulic coupling.

    Link flow is positive from `link_from_node_index` to `link_to_node_index`.
    Node head is represented as invert elevation plus stored volume divided by
    node storage area. This is a controlled first step toward SWMM-style
    dynamic-wave routing.
    """

    inlet_node_index: np.ndarray
    node_surface_row: np.ndarray
    node_surface_col: np.ndarray
    node_invert_elevation_m: np.ndarray
    node_storage_area_m2: np.ndarray
    node_max_depth_m: np.ndarray
    link_from_node_index: np.ndarray
    link_to_node_index: np.ndarray
    link_length_m: np.ndarray
    link_area_m2: np.ndarray
    link_hydraulic_radius_m: np.ndarray
    link_manning_n: np.ndarray
    link_flow_capacity_m3ps: np.ndarray
    outfall_area_m2: np.ndarray
    outfall_coefficient: np.ndarray
    outfall_tailwater_head_m: np.ndarray
    link_invert_elevation_m: np.ndarray | None = None
    link_diameter_m: np.ndarray | None = None
    link_minor_loss_coefficient: np.ndarray | None = None
    pressurized_wave_speed_mps: float = 30.0

    @property
    def node_count(self) -> int:
        return int(self.node_storage_area_m2.size)

    @property
    def link_count(self) -> int:
        return int(self.link_length_m.size)

    @property
    def node_capacity_m3(self) -> np.ndarray:
        return np.maximum(self.node_storage_area_m2, 1.0e-12) * np.maximum(self.node_max_depth_m, 0.0)


@dataclass(frozen=True)
class DynamicPipeNetworkState:
    """Dynamic head-based pipe network state."""

    node_volume_m3: np.ndarray
    link_flow_m3ps: np.ndarray
    cumulative_surface_exchange_m3: float = 0.0
    cumulative_link_abs_flow_m3: float = 0.0
    cumulative_outfall_discharge_m3: float = 0.0
    cumulative_surcharge_m3: float = 0.0
    cumulative_unreturned_overflow_m3: float = 0.0


@dataclass(frozen=True)
class DynamicPipeStepResult:
    """Result from one head-based pipe-network coupling step."""

    surface_depth_m: np.ndarray
    state: DynamicPipeNetworkState
    surface_capture_m3: float
    surface_surcharge_m3: float
    link_abs_flow_m3: float
    outfall_discharge_m3: float
    overflow_surcharge_m3: float
    unreturned_overflow_m3: float


def initialize_drainage_state(
    geometry: DrainageNetworkGeometry,
    initial_node_volume_m3: float | np.ndarray = 0.0,
) -> DrainageNetworkState:
    """Create a drainage state with validated node volumes."""

    _validate_network_geometry(geometry)
    if np.isscalar(initial_node_volume_m3):
        node_volume = np.full(geometry.node_count, float(initial_node_volume_m3), dtype=float)
    else:
        node_volume = np.asarray(initial_node_volume_m3, dtype=float).copy()
        if node_volume.shape != (geometry.node_count,):
            raise ValueError("initial_node_volume_m3 must be scalar or one value per node.")
    if np.any(node_volume < 0.0):
        raise ValueError("Initial node volumes must be non-negative.")
    return DrainageNetworkState(node_volume_m3=node_volume)


def initialize_dynamic_pipe_state(
    geometry: DynamicPipeNetworkGeometry,
    initial_node_volume_m3: float | np.ndarray = 0.0,
    initial_link_flow_m3ps: float | np.ndarray = 0.0,
) -> DynamicPipeNetworkState:
    """Create a dynamic pipe-network state with validated arrays."""

    _validate_dynamic_pipe_geometry(geometry)
    if np.isscalar(initial_node_volume_m3):
        node_volume = np.full(geometry.node_count, float(initial_node_volume_m3), dtype=float)
    else:
        node_volume = np.asarray(initial_node_volume_m3, dtype=float).copy()
        if node_volume.shape != (geometry.node_count,):
            raise ValueError("initial_node_volume_m3 must be scalar or one value per node.")
    if np.isscalar(initial_link_flow_m3ps):
        link_flow = np.full(geometry.link_count, float(initial_link_flow_m3ps), dtype=float)
    else:
        link_flow = np.asarray(initial_link_flow_m3ps, dtype=float).copy()
        if link_flow.shape != (geometry.link_count,):
            raise ValueError("initial_link_flow_m3ps must be scalar or one value per link.")
    if np.any(node_volume < 0.0):
        raise ValueError("Initial node volumes must be non-negative.")
    return DynamicPipeNetworkState(node_volume_m3=node_volume, link_flow_m3ps=link_flow)


def weir_capture_m3ps(
    surface_depth_m: np.ndarray,
    perimeter_m: float,
    coefficient: float = 0.45,
    gravity_mps2: float = 9.80665,
) -> np.ndarray:
    """Return inlet weir capture capacity from surface depth."""

    depth = np.maximum(surface_depth_m, 0.0)
    return coefficient * perimeter_m * depth * np.sqrt(2.0 * gravity_mps2 * depth)


def orifice_exchange_m3ps(
    head_difference_m: np.ndarray,
    area_m2: float,
    coefficient: float = 0.62,
    gravity_mps2: float = 9.80665,
) -> np.ndarray:
    """Return signed orifice exchange flow from head difference.

    Positive flow means surface-to-network capture. Negative flow means network
    surcharge back to the surface.
    """

    delta = np.asarray(head_difference_m, dtype=float)
    return coefficient * area_m2 * np.sqrt(2.0 * gravity_mps2 * np.abs(delta)) * np.sign(delta)


def node_head_m(
    node_volume_m3: np.ndarray,
    node_invert_elevation_m: np.ndarray,
    node_storage_area_m2: np.ndarray,
) -> np.ndarray:
    """Return node hydraulic head from volume-storage relation."""

    return node_invert_elevation_m + np.maximum(node_volume_m3, 0.0) / np.maximum(node_storage_area_m2, 1.0e-12)


def dynamic_pipe_link_flow_update(
    link_flow_m3ps: np.ndarray,
    from_head_m: np.ndarray,
    to_head_m: np.ndarray,
    link_length_m: np.ndarray,
    link_area_m2: np.ndarray,
    link_hydraulic_radius_m: np.ndarray,
    link_manning_n: np.ndarray,
    timestep_s: float,
    gravity_mps2: float = 9.80665,
    link_flow_capacity_m3ps: np.ndarray | None = None,
    minor_loss_coefficient: np.ndarray | float = 0.0,
) -> np.ndarray:
    """Update pipe link flow with a semi-implicit inertial Saint-Venant step."""

    return saint_venant_link_flow_update(
        link_flow_m3ps=link_flow_m3ps,
        from_head_m=from_head_m,
        to_head_m=to_head_m,
        link_length_m=link_length_m,
        flow_area_m2=link_area_m2,
        hydraulic_radius_m=link_hydraulic_radius_m,
        manning_n=link_manning_n,
        timestep_s=timestep_s,
        gravity_mps2=gravity_mps2,
        minor_loss_coefficient=minor_loss_coefficient,
        flow_capacity_m3ps=link_flow_capacity_m3ps,
    )


def apply_inlet_capture_depth(
    depth_m: np.ndarray,
    inlet_mask: np.ndarray,
    cell_area_m2: float,
    timestep_s: float,
    geometry: InletGeometry,
    capacity_limit_m3ps: float | None = None,
) -> tuple[np.ndarray, float]:
    """Remove inlet-captured water from surface depth with strict mass limiting."""

    if inlet_mask.shape != depth_m.shape:
        raise ValueError("inlet_mask shape must match depth_m shape.")
    updated_depth = np.maximum(depth_m.astype(float).copy(), 0.0)
    capture_capacity = weir_capture_m3ps(
        updated_depth,
        perimeter_m=geometry.perimeter_m,
        coefficient=geometry.weir_coefficient,
    )
    if capacity_limit_m3ps is not None:
        capture_capacity = np.minimum(capture_capacity, max(capacity_limit_m3ps, 0.0))
    requested_capture_m3 = capture_capacity * timestep_s * inlet_mask.astype(bool)
    available_m3 = updated_depth * cell_area_m2
    captured_m3 = np.minimum(available_m3, requested_capture_m3)
    updated_depth -= captured_m3 / cell_area_m2
    return updated_depth.astype(depth_m.dtype, copy=False), float(captured_m3.sum())


def step_drainage_network_coupling(
    surface_depth_m: np.ndarray,
    geometry: DrainageNetworkGeometry,
    state: DrainageNetworkState,
    cell_area_m2: float,
    timestep_s: float,
    inlet_geometry: InletGeometry,
    inlet_capacity_limit_m3ps: float | np.ndarray | None = None,
) -> DrainageStepResult:
    """Advance one conservative 1D-2D drainage coupling step."""

    _validate_network_geometry(geometry, expected_surface_shape=surface_depth_m.shape)
    if state.node_volume_m3.shape != (geometry.node_count,):
        raise ValueError("state.node_volume_m3 must have one value per node.")
    if cell_area_m2 <= 0.0:
        raise ValueError("cell_area_m2 must be positive.")
    if timestep_s < 0.0:
        raise ValueError("timestep_s must be non-negative.")

    updated_depth = np.maximum(surface_depth_m.astype(float).copy(), 0.0)
    inlet_mask = geometry.inlet_node_index >= 0
    capture_capacity_m3ps = weir_capture_m3ps(
        updated_depth,
        perimeter_m=inlet_geometry.perimeter_m,
        coefficient=inlet_geometry.weir_coefficient,
    )
    if inlet_capacity_limit_m3ps is not None:
        capture_capacity_m3ps = np.minimum(
            capture_capacity_m3ps,
            _capacity_limit_array(inlet_capacity_limit_m3ps, updated_depth.shape),
        )
    requested_capture_m3 = capture_capacity_m3ps * timestep_s * inlet_mask
    available_surface_m3 = updated_depth * cell_area_m2
    captured_by_cell_m3 = np.minimum(available_surface_m3, requested_capture_m3)
    updated_depth -= captured_by_cell_m3 / cell_area_m2

    node_volume_m3 = np.maximum(state.node_volume_m3.astype(float).copy(), 0.0)
    if np.any(inlet_mask):
        np.add.at(node_volume_m3, geometry.inlet_node_index[inlet_mask], captured_by_cell_m3[inlet_mask])
    inlet_capture_m3 = float(captured_by_cell_m3.sum())

    downstream = geometry.downstream_node_index.astype(int)
    link_mask = (downstream >= 0) & (geometry.link_capacity_m3ps > 0.0)
    link_flow_m3_by_node = np.zeros(geometry.node_count, dtype=float)
    link_flow_m3_by_node[link_mask] = np.minimum(
        node_volume_m3[link_mask],
        np.maximum(geometry.link_capacity_m3ps[link_mask], 0.0) * timestep_s,
    )
    node_volume_m3 -= link_flow_m3_by_node
    if np.any(link_mask):
        np.add.at(node_volume_m3, downstream[link_mask], link_flow_m3_by_node[link_mask])
    link_flow_m3 = float(link_flow_m3_by_node.sum())

    outfall_discharge_m3_by_node = np.minimum(
        node_volume_m3,
        np.maximum(geometry.outfall_capacity_m3ps, 0.0) * timestep_s,
    )
    node_volume_m3 -= outfall_discharge_m3_by_node
    outfall_discharge_m3 = float(outfall_discharge_m3_by_node.sum())

    overflow_m3 = np.maximum(node_volume_m3 - np.maximum(geometry.node_capacity_m3, 0.0), 0.0)
    valid_surcharge = (
        (overflow_m3 > 0.0)
        & (geometry.node_surface_row >= 0)
        & (geometry.node_surface_col >= 0)
        & (geometry.node_surface_row < updated_depth.shape[0])
        & (geometry.node_surface_col < updated_depth.shape[1])
    )
    surcharge_m3 = float(overflow_m3[valid_surcharge].sum())
    if np.any(valid_surcharge):
        np.add.at(
            updated_depth,
            (geometry.node_surface_row[valid_surcharge], geometry.node_surface_col[valid_surcharge]),
            overflow_m3[valid_surcharge] / cell_area_m2,
        )
        node_volume_m3[valid_surcharge] -= overflow_m3[valid_surcharge]
    unreturned_overflow_m3 = float(overflow_m3[~valid_surcharge].sum())

    new_state = DrainageNetworkState(
        node_volume_m3=node_volume_m3,
        cumulative_inlet_capture_m3=state.cumulative_inlet_capture_m3 + inlet_capture_m3,
        cumulative_link_flow_m3=state.cumulative_link_flow_m3 + link_flow_m3,
        cumulative_outfall_discharge_m3=state.cumulative_outfall_discharge_m3 + outfall_discharge_m3,
        cumulative_surcharge_m3=state.cumulative_surcharge_m3 + surcharge_m3,
        cumulative_unreturned_overflow_m3=state.cumulative_unreturned_overflow_m3 + unreturned_overflow_m3,
    )
    return DrainageStepResult(
        surface_depth_m=updated_depth.astype(surface_depth_m.dtype, copy=False),
        state=new_state,
        inlet_capture_m3=inlet_capture_m3,
        link_flow_m3=link_flow_m3,
        outfall_discharge_m3=outfall_discharge_m3,
        surcharge_m3=surcharge_m3,
        unreturned_overflow_m3=unreturned_overflow_m3,
    )


def step_dynamic_pipe_network_coupling(
    surface_depth_m: np.ndarray,
    surface_bed_elevation_m: np.ndarray,
    geometry: DynamicPipeNetworkGeometry,
    state: DynamicPipeNetworkState,
    cell_area_m2: float,
    timestep_s: float,
    inlet_geometry: InletGeometry,
) -> DynamicPipeStepResult:
    """Advance one head-based 1D pipe and 2D surface coupling step."""

    _validate_dynamic_pipe_geometry(geometry, expected_surface_shape=surface_depth_m.shape)
    if surface_bed_elevation_m.shape != surface_depth_m.shape:
        raise ValueError("surface_bed_elevation_m shape must match surface_depth_m shape.")
    if state.node_volume_m3.shape != (geometry.node_count,):
        raise ValueError("state.node_volume_m3 must have one value per node.")
    if state.link_flow_m3ps.shape != (geometry.link_count,):
        raise ValueError("state.link_flow_m3ps must have one value per link.")
    if cell_area_m2 <= 0.0:
        raise ValueError("cell_area_m2 must be positive.")
    if timestep_s < 0.0:
        raise ValueError("timestep_s must be non-negative.")

    updated_depth = np.maximum(surface_depth_m.astype(float).copy(), 0.0)
    node_volume_m3 = np.maximum(state.node_volume_m3.astype(float).copy(), 0.0)
    link_flow_m3ps = state.link_flow_m3ps.astype(float).copy()

    capture_m3, head_surcharge_m3 = _apply_head_based_surface_exchange(
        updated_depth=updated_depth,
        surface_bed_elevation_m=surface_bed_elevation_m,
        geometry=geometry,
        node_volume_m3=node_volume_m3,
        cell_area_m2=cell_area_m2,
        timestep_s=timestep_s,
        inlet_geometry=inlet_geometry,
    )

    heads = node_head_m(node_volume_m3, geometry.node_invert_elevation_m, geometry.node_storage_area_m2)
    from_node = geometry.link_from_node_index.astype(int)
    to_node = geometry.link_to_node_index.astype(int)
    candidate_flow_m3ps = dynamic_pipe_link_flow_update(
        link_flow_m3ps,
        from_head_m=heads[from_node],
        to_head_m=heads[to_node],
        link_length_m=geometry.link_length_m,
        link_area_m2=_dynamic_link_area_m2(geometry, heads[from_node], heads[to_node]),
        link_hydraulic_radius_m=_dynamic_link_hydraulic_radius_m(geometry, heads[from_node], heads[to_node]),
        link_manning_n=geometry.link_manning_n,
        timestep_s=timestep_s,
        link_flow_capacity_m3ps=geometry.link_flow_capacity_m3ps,
        minor_loss_coefficient=_dynamic_link_minor_loss_coefficient(geometry),
    )
    transfer_m3 = limit_link_transfers_by_available_volume(
        candidate_flow_m3ps * timestep_s,
        from_node,
        to_node,
        node_volume_m3,
    )
    if geometry.link_count > 0:
        node_volume_m3 += node_delta_from_link_transfers(transfer_m3, from_node, to_node, geometry.node_count)
    link_flow_m3ps = transfer_m3 / max(timestep_s, 1.0e-12)
    link_abs_flow_m3 = float(np.abs(transfer_m3).sum())

    heads = node_head_m(node_volume_m3, geometry.node_invert_elevation_m, geometry.node_storage_area_m2)
    outfall_q_m3ps = np.maximum(
        orifice_exchange_m3ps(
            heads - geometry.outfall_tailwater_head_m,
            area_m2=1.0,
            coefficient=1.0,
        ),
        0.0,
    )
    outfall_q_m3ps *= np.maximum(geometry.outfall_area_m2, 0.0) * np.maximum(geometry.outfall_coefficient, 0.0)
    outfall_discharge_m3_by_node = np.minimum(node_volume_m3, outfall_q_m3ps * timestep_s)
    node_volume_m3 -= outfall_discharge_m3_by_node
    outfall_discharge_m3 = float(outfall_discharge_m3_by_node.sum())

    overflow_surcharge_m3, unreturned_overflow_m3 = _return_node_overflow_to_surface(
        updated_depth,
        geometry.node_surface_row,
        geometry.node_surface_col,
        node_volume_m3,
        geometry.node_capacity_m3,
        cell_area_m2,
    )

    new_state = DynamicPipeNetworkState(
        node_volume_m3=node_volume_m3,
        link_flow_m3ps=link_flow_m3ps,
        cumulative_surface_exchange_m3=state.cumulative_surface_exchange_m3 + capture_m3 - head_surcharge_m3,
        cumulative_link_abs_flow_m3=state.cumulative_link_abs_flow_m3 + link_abs_flow_m3,
        cumulative_outfall_discharge_m3=state.cumulative_outfall_discharge_m3 + outfall_discharge_m3,
        cumulative_surcharge_m3=state.cumulative_surcharge_m3 + head_surcharge_m3 + overflow_surcharge_m3,
        cumulative_unreturned_overflow_m3=state.cumulative_unreturned_overflow_m3 + unreturned_overflow_m3,
    )
    return DynamicPipeStepResult(
        surface_depth_m=updated_depth.astype(surface_depth_m.dtype, copy=False),
        state=new_state,
        surface_capture_m3=capture_m3,
        surface_surcharge_m3=head_surcharge_m3,
        link_abs_flow_m3=link_abs_flow_m3,
        outfall_discharge_m3=outfall_discharge_m3,
        overflow_surcharge_m3=overflow_surcharge_m3,
        unreturned_overflow_m3=unreturned_overflow_m3,
    )


def _capacity_limit_array(limit: float | np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if np.isscalar(limit):
        return np.full(shape, max(float(limit), 0.0), dtype=float)
    array = np.asarray(limit, dtype=float)
    if array.shape != shape:
        raise ValueError(f"inlet_capacity_limit_m3ps shape {array.shape} does not match surface shape {shape}.")
    return np.maximum(array, 0.0)


def _apply_head_based_surface_exchange(
    updated_depth: np.ndarray,
    surface_bed_elevation_m: np.ndarray,
    geometry: DynamicPipeNetworkGeometry,
    node_volume_m3: np.ndarray,
    cell_area_m2: float,
    timestep_s: float,
    inlet_geometry: InletGeometry,
) -> tuple[float, float]:
    inlet_mask = geometry.inlet_node_index >= 0
    if not np.any(inlet_mask):
        return 0.0, 0.0
    node_ids = geometry.inlet_node_index[inlet_mask].astype(int)
    surface_head_m = surface_bed_elevation_m[inlet_mask] + updated_depth[inlet_mask]
    heads = node_head_m(node_volume_m3, geometry.node_invert_elevation_m, geometry.node_storage_area_m2)
    exchange_q_m3ps = orifice_exchange_m3ps(
        surface_head_m - heads[node_ids],
        area_m2=inlet_geometry.area_m2,
        coefficient=inlet_geometry.orifice_coefficient,
    )

    requested_capture_m3 = np.maximum(exchange_q_m3ps, 0.0) * timestep_s
    available_surface_m3 = updated_depth[inlet_mask] * cell_area_m2
    captured_m3 = np.minimum(available_surface_m3, requested_capture_m3)
    updated_depth[inlet_mask] -= captured_m3 / cell_area_m2
    np.add.at(node_volume_m3, node_ids, captured_m3)

    requested_surcharge_m3 = np.maximum(-exchange_q_m3ps, 0.0) * timestep_s
    surcharge_m3 = _limit_node_surcharge_requests(requested_surcharge_m3, node_ids, node_volume_m3)
    updated_depth[inlet_mask] += surcharge_m3 / cell_area_m2
    np.add.at(node_volume_m3, node_ids, -surcharge_m3)
    return float(captured_m3.sum()), float(surcharge_m3.sum())


def _limit_node_surcharge_requests(
    requested_surcharge_m3: np.ndarray,
    node_ids: np.ndarray,
    node_volume_m3: np.ndarray,
) -> np.ndarray:
    return limit_node_exchange_by_available_volume(requested_surcharge_m3, node_ids, node_volume_m3)


def _limit_link_transfers(
    requested_transfer_m3: np.ndarray,
    node_volume_m3: np.ndarray,
    from_node: np.ndarray,
    to_node: np.ndarray,
) -> np.ndarray:
    return limit_link_transfers_by_available_volume(requested_transfer_m3, from_node, to_node, node_volume_m3)


def _return_node_overflow_to_surface(
    updated_depth: np.ndarray,
    node_surface_row: np.ndarray,
    node_surface_col: np.ndarray,
    node_volume_m3: np.ndarray,
    node_capacity_m3: np.ndarray,
    cell_area_m2: float,
) -> tuple[float, float]:
    overflow_m3 = np.maximum(node_volume_m3 - np.maximum(node_capacity_m3, 0.0), 0.0)
    valid_surcharge = (
        (overflow_m3 > 0.0)
        & (node_surface_row >= 0)
        & (node_surface_col >= 0)
        & (node_surface_row < updated_depth.shape[0])
        & (node_surface_col < updated_depth.shape[1])
    )
    surcharge_m3 = float(overflow_m3[valid_surcharge].sum())
    if np.any(valid_surcharge):
        np.add.at(
            updated_depth,
            (node_surface_row[valid_surcharge], node_surface_col[valid_surcharge]),
            overflow_m3[valid_surcharge] / cell_area_m2,
        )
        node_volume_m3[valid_surcharge] -= overflow_m3[valid_surcharge]
    return surcharge_m3, float(overflow_m3[~valid_surcharge].sum())


def _dynamic_link_depth_m(
    geometry: DynamicPipeNetworkGeometry,
    from_head_m: np.ndarray,
    to_head_m: np.ndarray,
) -> np.ndarray:
    invert = (
        np.asarray(geometry.link_invert_elevation_m, dtype=float)
        if geometry.link_invert_elevation_m is not None
        else np.minimum(from_head_m, to_head_m) * 0.0 + np.minimum(
            geometry.node_invert_elevation_m[geometry.link_from_node_index],
            geometry.node_invert_elevation_m[geometry.link_to_node_index],
        )
    )
    return np.maximum(0.5 * (from_head_m + to_head_m) - invert, 0.0)


def _dynamic_link_area_m2(
    geometry: DynamicPipeNetworkGeometry,
    from_head_m: np.ndarray,
    to_head_m: np.ndarray,
) -> np.ndarray:
    if geometry.link_diameter_m is None:
        return geometry.link_area_m2
    return preissmann_slot_area_m2(
        _dynamic_link_depth_m(geometry, from_head_m, to_head_m),
        geometry.link_diameter_m,
        pressurized_wave_speed_mps=geometry.pressurized_wave_speed_mps,
    )


def _dynamic_link_hydraulic_radius_m(
    geometry: DynamicPipeNetworkGeometry,
    from_head_m: np.ndarray,
    to_head_m: np.ndarray,
) -> np.ndarray:
    if geometry.link_diameter_m is None:
        return geometry.link_hydraulic_radius_m
    return circular_conduit_hydraulic_radius_m(
        _dynamic_link_depth_m(geometry, from_head_m, to_head_m),
        geometry.link_diameter_m,
    )


def _dynamic_link_minor_loss_coefficient(geometry: DynamicPipeNetworkGeometry) -> np.ndarray | float:
    return 0.0 if geometry.link_minor_loss_coefficient is None else geometry.link_minor_loss_coefficient


def _validate_network_geometry(
    geometry: DrainageNetworkGeometry,
    expected_surface_shape: tuple[int, int] | None = None,
) -> None:
    if geometry.inlet_node_index.ndim != 2:
        raise ValueError("inlet_node_index must be a 2D array.")
    if expected_surface_shape is not None and geometry.inlet_node_index.shape != expected_surface_shape:
        raise ValueError("inlet_node_index shape must match surface_depth_m shape.")
    node_count = geometry.node_count
    for name in (
        "node_surface_row",
        "node_surface_col",
        "downstream_node_index",
        "link_capacity_m3ps",
        "outfall_capacity_m3ps",
        "node_capacity_m3",
    ):
        value = np.asarray(getattr(geometry, name))
        if value.shape != (node_count,):
            raise ValueError(f"{name} must have one value per node.")
    if np.any(geometry.inlet_node_index < -1) or np.any(geometry.inlet_node_index >= node_count):
        raise ValueError("inlet_node_index contains an invalid node id.")
    downstream = geometry.downstream_node_index
    if np.any(downstream < -1) or np.any(downstream >= node_count):
        raise ValueError("downstream_node_index contains an invalid node id.")
    if np.any(geometry.link_capacity_m3ps < 0.0):
        raise ValueError("link capacities must be non-negative.")
    if np.any(geometry.outfall_capacity_m3ps < 0.0):
        raise ValueError("outfall capacities must be non-negative.")
    if np.any(geometry.node_capacity_m3 < 0.0):
        raise ValueError("node capacities must be non-negative.")


def _validate_dynamic_pipe_geometry(
    geometry: DynamicPipeNetworkGeometry,
    expected_surface_shape: tuple[int, int] | None = None,
) -> None:
    if geometry.inlet_node_index.ndim != 2:
        raise ValueError("inlet_node_index must be a 2D array.")
    if expected_surface_shape is not None and geometry.inlet_node_index.shape != expected_surface_shape:
        raise ValueError("inlet_node_index shape must match surface_depth_m shape.")
    node_count = geometry.node_count
    link_count = geometry.link_count
    for name in (
        "node_surface_row",
        "node_surface_col",
        "node_invert_elevation_m",
        "node_storage_area_m2",
        "node_max_depth_m",
        "outfall_area_m2",
        "outfall_coefficient",
        "outfall_tailwater_head_m",
    ):
        value = np.asarray(getattr(geometry, name))
        if value.shape != (node_count,):
            raise ValueError(f"{name} must have one value per node.")
    for name in (
        "link_from_node_index",
        "link_to_node_index",
        "link_length_m",
        "link_area_m2",
        "link_hydraulic_radius_m",
        "link_manning_n",
        "link_flow_capacity_m3ps",
    ):
        value = np.asarray(getattr(geometry, name))
        if value.shape != (link_count,):
            raise ValueError(f"{name} must have one value per link.")
    if np.any(geometry.inlet_node_index < -1) or np.any(geometry.inlet_node_index >= node_count):
        raise ValueError("inlet_node_index contains an invalid node id.")
    for name in ("link_from_node_index", "link_to_node_index"):
        value = np.asarray(getattr(geometry, name))
        if np.any(value < 0) or np.any(value >= node_count):
            raise ValueError(f"{name} contains an invalid node id.")
    if np.any(geometry.node_storage_area_m2 <= 0.0):
        raise ValueError("node storage areas must be positive.")
    if np.any(geometry.node_max_depth_m < 0.0):
        raise ValueError("node max depths must be non-negative.")
    if np.any(geometry.link_length_m <= 0.0):
        raise ValueError("link lengths must be positive.")
    if np.any(geometry.link_area_m2 <= 0.0):
        raise ValueError("link areas must be positive.")
    if np.any(geometry.link_hydraulic_radius_m <= 0.0):
        raise ValueError("link hydraulic radii must be positive.")
    if np.any(geometry.link_manning_n < 0.0):
        raise ValueError("link Manning values must be non-negative.")
    if np.any(geometry.link_flow_capacity_m3ps < 0.0):
        raise ValueError("link flow capacities must be non-negative.")
    if np.any(geometry.outfall_area_m2 < 0.0):
        raise ValueError("outfall areas must be non-negative.")
    if np.any(geometry.outfall_coefficient < 0.0):
        raise ValueError("outfall coefficients must be non-negative.")
    for name in ("link_invert_elevation_m", "link_diameter_m", "link_minor_loss_coefficient"):
        value = getattr(geometry, name)
        if value is not None and np.asarray(value).shape != (link_count,):
            raise ValueError(f"{name} must have one value per link when provided.")
    if geometry.link_diameter_m is not None and np.any(geometry.link_diameter_m <= 0.0):
        raise ValueError("link diameters must be positive.")
    if geometry.link_minor_loss_coefficient is not None and np.any(geometry.link_minor_loss_coefficient < 0.0):
        raise ValueError("minor loss coefficients must be non-negative.")
    if geometry.pressurized_wave_speed_mps <= 0.0:
        raise ValueError("pressurized_wave_speed_mps must be positive.")
