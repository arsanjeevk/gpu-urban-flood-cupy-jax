"""Backend timestep executor for coupled 2D surface and 1D pipe routing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from gurugram_flood.kernels.equations import hll_shallow_water_normal_flux, hydrostatic_reconstruction_flux_pair
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


@dataclass(frozen=True)
class CoupledBackendState:
    """Backend-resident dynamic state for one coupled timestep."""

    h_m: Any
    hu_m2ps: Any
    hv_m2ps: Any
    node_volume_m3: Any
    link_flow_m3ps: Any


@dataclass(frozen=True)
class CoupledBackendTopology:
    """Backend-resident structured-grid and pipe-network topology."""

    bed_elevation_m: Any
    manning_n: Any
    dx_m: float
    dy_m: float
    inlet_cell_index: Any
    inlet_node_index: Any
    node_surface_cell_index: Any
    node_invert_elevation_m: Any
    node_storage_area_m2: Any
    node_max_depth_m: Any
    link_from_node_index: Any
    link_to_node_index: Any
    link_length_m: Any
    link_invert_elevation_m: Any
    link_diameter_m: Any
    link_manning_n: Any
    link_flow_capacity_m3ps: Any
    outfall_area_m2: Any
    outfall_coefficient: Any
    outfall_tailwater_head_m: Any
    active_mask: Any | None = None
    link_minor_loss_coefficient: Any | float = 0.0
    pressurized_wave_speed_mps: float = 30.0
    inlet_capture_capacity_m3ps: Any | None = None
    inlet_surcharge_capacity_m3ps: Any | None = None

    @property
    def cell_area_m2(self) -> float:
        return self.dx_m * self.dy_m

    @property
    def node_count(self) -> int:
        return int(self.node_storage_area_m2.shape[0])

    @property
    def link_count(self) -> int:
        return int(self.link_length_m.shape[0])


@dataclass(frozen=True)
class PackedCoupledBackendTopology:
    """Packed backend topology with fewer live buffers for compiled loops."""

    bed_elevation_m: Any
    manning_n: Any
    dx_m: float
    dy_m: float
    inlet_pair_index: Any
    node_index_props: Any
    node_props: Any
    link_node_index: Any
    link_props: Any
    outfall_props: Any
    active_mask: Any | None = None
    inlet_props: Any | None = None
    pressurized_wave_speed_mps: float = 30.0

    @property
    def cell_area_m2(self) -> float:
        return self.dx_m * self.dy_m

    @property
    def node_count(self) -> int:
        return int(self.node_props.shape[0])

    @property
    def link_count(self) -> int:
        return int(self.link_props.shape[0])


def pack_coupled_backend_topology(topology: CoupledBackendTopology) -> PackedCoupledBackendTopology:
    """Pack coupled topology arrays to reduce compiled-function buffer count."""

    xp = _xp_from_array(topology.bed_elevation_m)
    inlet_pair_index = _stack_columns(xp, topology.inlet_cell_index, topology.inlet_node_index)
    inlet_props = _stack_columns(
        xp,
        _optional_capacity_or_unbounded(topology.inlet_capture_capacity_m3ps, topology.inlet_cell_index),
        _optional_capacity_or_unbounded(topology.inlet_surcharge_capacity_m3ps, topology.inlet_cell_index),
    )
    node_index_props = topology.node_surface_cell_index.reshape((-1, 1))
    node_props = _stack_columns(
        xp,
        topology.node_invert_elevation_m,
        topology.node_storage_area_m2,
        topology.node_max_depth_m,
    )
    link_node_index = _stack_columns(xp, topology.link_from_node_index, topology.link_to_node_index)
    link_props = _stack_columns(
        xp,
        topology.link_length_m,
        topology.link_invert_elevation_m,
        topology.link_diameter_m,
        topology.link_manning_n,
        topology.link_flow_capacity_m3ps,
        _as_column_array(topology.link_minor_loss_coefficient, topology.link_length_m),
    )
    outfall_props = _stack_columns(
        xp,
        topology.outfall_area_m2,
        topology.outfall_coefficient,
        topology.outfall_tailwater_head_m,
    )
    return PackedCoupledBackendTopology(
        bed_elevation_m=topology.bed_elevation_m,
        manning_n=topology.manning_n,
        dx_m=topology.dx_m,
        dy_m=topology.dy_m,
        inlet_pair_index=inlet_pair_index,
        node_index_props=node_index_props,
        node_props=node_props,
        link_node_index=link_node_index,
        link_props=link_props,
        outfall_props=outfall_props,
        active_mask=topology.active_mask,
        inlet_props=inlet_props,
        pressurized_wave_speed_mps=topology.pressurized_wave_speed_mps,
    )


def unpack_coupled_backend_topology(packed: PackedCoupledBackendTopology) -> CoupledBackendTopology:
    """Expand packed topology buffers into the standard coupled topology view."""

    return CoupledBackendTopology(
        bed_elevation_m=packed.bed_elevation_m,
        manning_n=packed.manning_n,
        dx_m=packed.dx_m,
        dy_m=packed.dy_m,
        inlet_cell_index=packed.inlet_pair_index[:, 0],
        inlet_node_index=packed.inlet_pair_index[:, 1],
        node_surface_cell_index=packed.node_index_props[:, 0],
        node_invert_elevation_m=packed.node_props[:, 0],
        node_storage_area_m2=packed.node_props[:, 1],
        node_max_depth_m=packed.node_props[:, 2],
        link_from_node_index=packed.link_node_index[:, 0],
        link_to_node_index=packed.link_node_index[:, 1],
        link_length_m=packed.link_props[:, 0],
        link_invert_elevation_m=packed.link_props[:, 1],
        link_diameter_m=packed.link_props[:, 2],
        link_manning_n=packed.link_props[:, 3],
        link_flow_capacity_m3ps=packed.link_props[:, 4],
        outfall_area_m2=packed.outfall_props[:, 0],
        outfall_coefficient=packed.outfall_props[:, 1],
        outfall_tailwater_head_m=packed.outfall_props[:, 2],
        active_mask=packed.active_mask,
        link_minor_loss_coefficient=packed.link_props[:, 5],
        pressurized_wave_speed_mps=packed.pressurized_wave_speed_mps,
        inlet_capture_capacity_m3ps=None if packed.inlet_props is None else packed.inlet_props[:, 0],
        inlet_surcharge_capacity_m3ps=None if packed.inlet_props is None else packed.inlet_props[:, 1],
    )


@dataclass(frozen=True)
class CoupledBackendForcing:
    """Backend-compatible source terms and coupling coefficients."""

    rainfall_rate_mps: Any | float = 0.0
    infiltration_capacity_mps: Any | float = 0.0
    boundary: "CoupledBackendBoundaryConditions" = field(default_factory=lambda: CoupledBackendBoundaryConditions())
    inlet_orifice_area_m2: float = 0.04
    inlet_orifice_coefficient: float = 0.62
    gravity_mps2: float = 9.80665
    min_depth_m: float = 1.0e-5


@dataclass(frozen=True)
class CoupledBackendBoundaryConditions:
    """Per-side boundary modes for the backend structured-grid surface step."""

    west_mode: str = "closed"
    east_mode: str = "closed"
    south_mode: str = "closed"
    north_mode: str = "closed"
    west_stage_m: Any | float | None = None
    east_stage_m: Any | float | None = None
    south_stage_m: Any | float | None = None
    north_stage_m: Any | float | None = None


@dataclass(frozen=True)
class CoupledBackendDiagnostics:
    """Backend-resident volume diagnostics from one timestep."""

    rainfall_input_m3: Any
    infiltration_loss_m3: Any
    boundary_west_net_flux_m3: Any
    boundary_east_net_flux_m3: Any
    boundary_south_net_flux_m3: Any
    boundary_north_net_flux_m3: Any
    boundary_net_flux_m3: Any
    boundary_inflow_m3: Any
    boundary_outflow_m3: Any
    surface_capture_m3: Any
    surface_surcharge_m3: Any
    link_abs_flow_m3: Any
    outfall_discharge_m3: Any
    overflow_surcharge_m3: Any
    final_surface_volume_m3: Any
    final_node_volume_m3: Any


@dataclass(frozen=True)
class CoupledBackendMassBalance:
    """Backend-resident accumulated volume balance over many coupled steps."""

    step_count: int
    initial_surface_volume_m3: Any
    initial_node_volume_m3: Any
    rainfall_input_m3: Any
    infiltration_loss_m3: Any
    boundary_west_net_flux_m3: Any
    boundary_east_net_flux_m3: Any
    boundary_south_net_flux_m3: Any
    boundary_north_net_flux_m3: Any
    boundary_net_flux_m3: Any
    boundary_inflow_m3: Any
    boundary_outflow_m3: Any
    surface_capture_m3: Any
    surface_surcharge_m3: Any
    link_abs_flow_m3: Any
    outfall_discharge_m3: Any
    overflow_surcharge_m3: Any
    final_surface_volume_m3: Any
    final_node_volume_m3: Any
    mass_residual_m3: Any


@dataclass(frozen=True)
class _BackendBoundaryVolumeStep:
    """Backend-resident boundary water volume, positive into the model domain."""

    west_m3: Any
    east_m3: Any
    south_m3: Any
    north_m3: Any


def initialize_coupled_backend_mass_balance(
    state: CoupledBackendState,
    topology: CoupledBackendTopology,
) -> CoupledBackendMassBalance:
    """Create a backend-resident mass-balance accumulator for a run."""

    xp = _xp_from_array(state.h_m)
    zero = _zero_scalar_like(state.h_m, xp)
    initial_surface_volume_m3 = xp.sum(state.h_m) * topology.cell_area_m2
    initial_node_volume_m3 = xp.sum(state.node_volume_m3)
    return CoupledBackendMassBalance(
        step_count=0,
        initial_surface_volume_m3=initial_surface_volume_m3,
        initial_node_volume_m3=initial_node_volume_m3,
        rainfall_input_m3=zero,
        infiltration_loss_m3=zero,
        boundary_west_net_flux_m3=zero,
        boundary_east_net_flux_m3=zero,
        boundary_south_net_flux_m3=zero,
        boundary_north_net_flux_m3=zero,
        boundary_net_flux_m3=zero,
        boundary_inflow_m3=zero,
        boundary_outflow_m3=zero,
        surface_capture_m3=zero,
        surface_surcharge_m3=zero,
        link_abs_flow_m3=zero,
        outfall_discharge_m3=zero,
        overflow_surcharge_m3=zero,
        final_surface_volume_m3=initial_surface_volume_m3,
        final_node_volume_m3=initial_node_volume_m3,
        mass_residual_m3=zero,
    )


def accumulate_coupled_backend_mass_balance(
    balance: CoupledBackendMassBalance,
    diagnostics: CoupledBackendDiagnostics,
) -> CoupledBackendMassBalance:
    """Return an updated backend-resident run balance from one step diagnostic."""

    rainfall_input_m3 = balance.rainfall_input_m3 + diagnostics.rainfall_input_m3
    infiltration_loss_m3 = balance.infiltration_loss_m3 + diagnostics.infiltration_loss_m3
    boundary_west_net_flux_m3 = balance.boundary_west_net_flux_m3 + diagnostics.boundary_west_net_flux_m3
    boundary_east_net_flux_m3 = balance.boundary_east_net_flux_m3 + diagnostics.boundary_east_net_flux_m3
    boundary_south_net_flux_m3 = balance.boundary_south_net_flux_m3 + diagnostics.boundary_south_net_flux_m3
    boundary_north_net_flux_m3 = balance.boundary_north_net_flux_m3 + diagnostics.boundary_north_net_flux_m3
    boundary_net_flux_m3 = balance.boundary_net_flux_m3 + diagnostics.boundary_net_flux_m3
    boundary_inflow_m3 = balance.boundary_inflow_m3 + diagnostics.boundary_inflow_m3
    boundary_outflow_m3 = balance.boundary_outflow_m3 + diagnostics.boundary_outflow_m3
    surface_capture_m3 = balance.surface_capture_m3 + diagnostics.surface_capture_m3
    surface_surcharge_m3 = balance.surface_surcharge_m3 + diagnostics.surface_surcharge_m3
    link_abs_flow_m3 = balance.link_abs_flow_m3 + diagnostics.link_abs_flow_m3
    outfall_discharge_m3 = balance.outfall_discharge_m3 + diagnostics.outfall_discharge_m3
    overflow_surcharge_m3 = balance.overflow_surcharge_m3 + diagnostics.overflow_surcharge_m3
    mass_residual_m3 = (
        balance.initial_surface_volume_m3
        + balance.initial_node_volume_m3
        + rainfall_input_m3
        + boundary_net_flux_m3
        - infiltration_loss_m3
        - outfall_discharge_m3
        - diagnostics.final_surface_volume_m3
        - diagnostics.final_node_volume_m3
    )
    return CoupledBackendMassBalance(
        step_count=balance.step_count + 1,
        initial_surface_volume_m3=balance.initial_surface_volume_m3,
        initial_node_volume_m3=balance.initial_node_volume_m3,
        rainfall_input_m3=rainfall_input_m3,
        infiltration_loss_m3=infiltration_loss_m3,
        boundary_west_net_flux_m3=boundary_west_net_flux_m3,
        boundary_east_net_flux_m3=boundary_east_net_flux_m3,
        boundary_south_net_flux_m3=boundary_south_net_flux_m3,
        boundary_north_net_flux_m3=boundary_north_net_flux_m3,
        boundary_net_flux_m3=boundary_net_flux_m3,
        boundary_inflow_m3=boundary_inflow_m3,
        boundary_outflow_m3=boundary_outflow_m3,
        surface_capture_m3=surface_capture_m3,
        surface_surcharge_m3=surface_surcharge_m3,
        link_abs_flow_m3=link_abs_flow_m3,
        outfall_discharge_m3=outfall_discharge_m3,
        overflow_surcharge_m3=overflow_surcharge_m3,
        final_surface_volume_m3=diagnostics.final_surface_volume_m3,
        final_node_volume_m3=diagnostics.final_node_volume_m3,
        mass_residual_m3=mass_residual_m3,
    )


def step_coupled_backend(
    state: CoupledBackendState,
    topology: CoupledBackendTopology,
    forcing: CoupledBackendForcing,
    timestep_s: float,
) -> tuple[CoupledBackendState, CoupledBackendDiagnostics]:
    """Advance one staged backend timestep.

    This is the first integrated executor for Mac GPU scale: 2D structured SWE,
    rainfall/infiltration, inlet exchange, pipe Saint-Venant update, node
    continuity, tailwater outfalls, and surcharge are all expressed as backend
    array operations.
    """

    xp = _xp_from_array(state.h_m)
    h, hu, hv, boundary_step = _advance_surface_swe(
        h=state.h_m,
        hu=state.hu_m2ps,
        hv=state.hv_m2ps,
        bed=topology.bed_elevation_m,
        manning_n=topology.manning_n,
        dx_m=topology.dx_m,
        dy_m=topology.dy_m,
        timestep_s=timestep_s,
        gravity_mps2=forcing.gravity_mps2,
        min_depth_m=forcing.min_depth_m,
        boundary=forcing.boundary,
    )
    h, hu, hv, rainfall_input_m3, infiltration_loss_m3 = _apply_surface_sources(
        h=h,
        hu=hu,
        hv=hv,
        forcing=forcing,
        topology=topology,
        timestep_s=timestep_s,
    )

    node_volume = xp.maximum(state.node_volume_m3, 0.0)
    link_flow = state.link_flow_m3ps
    h_flat = h.reshape((-1,))
    bed_flat = topology.bed_elevation_m.reshape((-1,))

    h_flat, node_volume, surface_capture_m3, surface_surcharge_m3 = _exchange_surface_nodes(
        h_flat=h_flat,
        bed_flat=bed_flat,
        topology=topology,
        node_volume=node_volume,
        forcing=forcing,
        timestep_s=timestep_s,
    )

    node_head = _node_head(node_volume, topology.node_invert_elevation_m, topology.node_storage_area_m2, xp)
    from_node = topology.link_from_node_index
    to_node = topology.link_to_node_index
    link_depth = xp.maximum(
        0.5 * (node_head[from_node] + node_head[to_node]) - topology.link_invert_elevation_m,
        0.0,
    )
    link_area = preissmann_slot_area_m2(
        link_depth,
        topology.link_diameter_m,
        pressurized_wave_speed_mps=topology.pressurized_wave_speed_mps,
        gravity_mps2=forcing.gravity_mps2,
    )
    link_radius = circular_conduit_hydraulic_radius_m(link_depth, topology.link_diameter_m)
    candidate_flow = saint_venant_link_flow_update(
        link_flow,
        from_head_m=node_head[from_node],
        to_head_m=node_head[to_node],
        link_length_m=topology.link_length_m,
        flow_area_m2=link_area,
        hydraulic_radius_m=link_radius,
        manning_n=topology.link_manning_n,
        timestep_s=timestep_s,
        gravity_mps2=forcing.gravity_mps2,
        minor_loss_coefficient=topology.link_minor_loss_coefficient,
        flow_capacity_m3ps=topology.link_flow_capacity_m3ps,
    )
    transfer_m3 = limit_link_transfers_by_available_volume(
        candidate_flow * timestep_s,
        from_node,
        to_node,
        node_volume,
    )
    node_volume = node_volume + node_delta_from_link_transfers(transfer_m3, from_node, to_node, topology.node_count)
    link_flow = transfer_m3 / max(timestep_s, 1.0e-12)
    link_abs_flow_m3 = xp.sum(xp.abs(transfer_m3))

    node_head = _node_head(node_volume, topology.node_invert_elevation_m, topology.node_storage_area_m2, xp)
    outfall_q = _orifice_q(
        node_head - topology.outfall_tailwater_head_m,
        topology.outfall_area_m2,
        topology.outfall_coefficient,
        forcing.gravity_mps2,
        xp,
    )
    outfall_m3 = xp.minimum(node_volume, xp.maximum(outfall_q, 0.0) * timestep_s)
    node_volume = node_volume - outfall_m3
    outfall_discharge_m3 = xp.sum(outfall_m3)

    h_flat, node_volume, overflow_surcharge_m3 = _return_overflow_to_surface(
        h_flat,
        topology=topology,
        node_volume=node_volume,
        xp=xp,
    )
    h = h_flat.reshape(state.h_m.shape)
    wet = h > forcing.min_depth_m
    hu = xp.where(wet, hu, 0.0)
    hv = xp.where(wet, hv, 0.0)

    new_state = CoupledBackendState(
        h_m=h,
        hu_m2ps=hu,
        hv_m2ps=hv,
        node_volume_m3=node_volume,
        link_flow_m3ps=link_flow,
    )
    boundary_net_flux_m3 = (
        boundary_step.west_m3
        + boundary_step.east_m3
        + boundary_step.south_m3
        + boundary_step.north_m3
    )
    diagnostics = CoupledBackendDiagnostics(
        rainfall_input_m3=rainfall_input_m3,
        infiltration_loss_m3=infiltration_loss_m3,
        boundary_west_net_flux_m3=boundary_step.west_m3,
        boundary_east_net_flux_m3=boundary_step.east_m3,
        boundary_south_net_flux_m3=boundary_step.south_m3,
        boundary_north_net_flux_m3=boundary_step.north_m3,
        boundary_net_flux_m3=boundary_net_flux_m3,
        boundary_inflow_m3=(
            xp.maximum(boundary_step.west_m3, 0.0)
            + xp.maximum(boundary_step.east_m3, 0.0)
            + xp.maximum(boundary_step.south_m3, 0.0)
            + xp.maximum(boundary_step.north_m3, 0.0)
        ),
        boundary_outflow_m3=(
            xp.maximum(-boundary_step.west_m3, 0.0)
            + xp.maximum(-boundary_step.east_m3, 0.0)
            + xp.maximum(-boundary_step.south_m3, 0.0)
            + xp.maximum(-boundary_step.north_m3, 0.0)
        ),
        surface_capture_m3=surface_capture_m3,
        surface_surcharge_m3=surface_surcharge_m3,
        link_abs_flow_m3=link_abs_flow_m3,
        outfall_discharge_m3=outfall_discharge_m3,
        overflow_surcharge_m3=overflow_surcharge_m3,
        final_surface_volume_m3=xp.sum(h) * topology.cell_area_m2,
        final_node_volume_m3=xp.sum(node_volume),
    )
    return new_state, diagnostics


def _advance_surface_swe(
    h: Any,
    hu: Any,
    hv: Any,
    bed: Any,
    manning_n: Any,
    dx_m: float,
    dy_m: float,
    timestep_s: float,
    gravity_mps2: float,
    min_depth_m: float,
    boundary: CoupledBackendBoundaryConditions,
) -> tuple[Any, Any, Any, _BackendBoundaryVolumeStep]:
    xp = _xp_from_array(h)
    h_new = h
    hu_new = hu
    hv_new = hv
    zero = _zero_scalar_like(h, xp)

    left_flux, right_flux = hydrostatic_reconstruction_flux_pair(
        h[:, :-1],
        hu[:, :-1],
        hv[:, :-1],
        bed[:, :-1],
        h[:, 1:],
        hu[:, 1:],
        hv[:, 1:],
        bed[:, 1:],
        normal_x=1.0,
        normal_y=0.0,
        gravity_mps2=gravity_mps2,
        min_depth_m=min_depth_m,
    )
    h_new = h_new - timestep_s / dx_m * _pad_right(left_flux[0], xp) + timestep_s / dx_m * _pad_left(right_flux[0], xp)
    hu_new = hu_new - timestep_s / dx_m * _pad_right(left_flux[1], xp) + timestep_s / dx_m * _pad_left(right_flux[1], xp)
    hv_new = hv_new - timestep_s / dx_m * _pad_right(left_flux[2], xp) + timestep_s / dx_m * _pad_left(right_flux[2], xp)
    h_new, hu_new, hv_new, west_m3, east_m3 = _apply_x_boundary_fluxes(
        h_new=h_new,
        hu_new=hu_new,
        hv_new=hv_new,
        h=h,
        hu=hu,
        hv=hv,
        bed=bed,
        dx_m=dx_m,
        dy_m=dy_m,
        timestep_s=timestep_s,
        gravity_mps2=gravity_mps2,
        min_depth_m=min_depth_m,
        boundary=boundary,
        xp=xp,
        zero=zero,
    )

    top_flux, bottom_flux = hydrostatic_reconstruction_flux_pair(
        h[:-1, :],
        hu[:-1, :],
        hv[:-1, :],
        bed[:-1, :],
        h[1:, :],
        hu[1:, :],
        hv[1:, :],
        bed[1:, :],
        normal_x=0.0,
        normal_y=1.0,
        gravity_mps2=gravity_mps2,
        min_depth_m=min_depth_m,
    )
    h_new = h_new - timestep_s / dy_m * _pad_bottom(top_flux[0], xp) + timestep_s / dy_m * _pad_top(bottom_flux[0], xp)
    hu_new = hu_new - timestep_s / dy_m * _pad_bottom(top_flux[1], xp) + timestep_s / dy_m * _pad_top(bottom_flux[1], xp)
    hv_new = hv_new - timestep_s / dy_m * _pad_bottom(top_flux[2], xp) + timestep_s / dy_m * _pad_top(bottom_flux[2], xp)
    h_new, hu_new, hv_new, south_m3, north_m3 = _apply_y_boundary_fluxes(
        h_new=h_new,
        hu_new=hu_new,
        hv_new=hv_new,
        h=h,
        hu=hu,
        hv=hv,
        bed=bed,
        dx_m=dx_m,
        dy_m=dy_m,
        timestep_s=timestep_s,
        gravity_mps2=gravity_mps2,
        min_depth_m=min_depth_m,
        boundary=boundary,
        xp=xp,
        zero=zero,
    )

    h_new = xp.maximum(h_new, 0.0)
    wet = h_new > min_depth_m
    speed = xp.sqrt((hu_new / xp.maximum(h_new, min_depth_m)) ** 2 + (hv_new / xp.maximum(h_new, min_depth_m)) ** 2)
    damping = 1.0 + timestep_s * gravity_mps2 * manning_n**2 * speed / xp.maximum(h_new, min_depth_m) ** (4.0 / 3.0)
    hu_new = xp.where(wet, hu_new / damping, 0.0)
    hv_new = xp.where(wet, hv_new / damping, 0.0)
    return h_new, hu_new, hv_new, _BackendBoundaryVolumeStep(
        west_m3=west_m3,
        east_m3=east_m3,
        south_m3=south_m3,
        north_m3=north_m3,
    )


def _apply_x_boundary_fluxes(
    h_new: Any,
    hu_new: Any,
    hv_new: Any,
    h: Any,
    hu: Any,
    hv: Any,
    bed: Any,
    dx_m: float,
    dy_m: float,
    timestep_s: float,
    gravity_mps2: float,
    min_depth_m: float,
    boundary: CoupledBackendBoundaryConditions,
    xp: Any,
    zero: Any,
) -> tuple[Any, Any, Any, Any, Any]:
    west_m3 = zero
    east_m3 = zero
    width = int(h.shape[1])

    west_mode = _normalise_boundary_mode(boundary.west_mode)
    if west_mode != "closed":
        h_ghost, hu_ghost, hv_ghost = _x_ghost_state(
            side="west",
            mode=west_mode,
            stage_m=boundary.west_stage_m,
            h=h,
            hu=hu,
            hv=hv,
            bed=bed,
            xp=xp,
        )
        flux_h, flux_hu, flux_hv = hll_shallow_water_normal_flux(
            h_ghost,
            hu_ghost,
            hv_ghost,
            h[:, 0],
            hu[:, 0],
            hv[:, 0],
            normal_x=1.0,
            normal_y=0.0,
            gravity_mps2=gravity_mps2,
            min_depth_m=min_depth_m,
        )
        h_new = h_new + timestep_s / dx_m * _pad_west_edge(flux_h, width, xp)
        hu_new = hu_new + timestep_s / dx_m * _pad_west_edge(flux_hu, width, xp)
        hv_new = hv_new + timestep_s / dx_m * _pad_west_edge(flux_hv, width, xp)
        west_m3 = xp.sum(flux_h) * dy_m * timestep_s

    east_mode = _normalise_boundary_mode(boundary.east_mode)
    if east_mode != "closed":
        h_ghost, hu_ghost, hv_ghost = _x_ghost_state(
            side="east",
            mode=east_mode,
            stage_m=boundary.east_stage_m,
            h=h,
            hu=hu,
            hv=hv,
            bed=bed,
            xp=xp,
        )
        flux_h, flux_hu, flux_hv = hll_shallow_water_normal_flux(
            h[:, -1],
            hu[:, -1],
            hv[:, -1],
            h_ghost,
            hu_ghost,
            hv_ghost,
            normal_x=1.0,
            normal_y=0.0,
            gravity_mps2=gravity_mps2,
            min_depth_m=min_depth_m,
        )
        h_new = h_new - timestep_s / dx_m * _pad_east_edge(flux_h, width, xp)
        hu_new = hu_new - timestep_s / dx_m * _pad_east_edge(flux_hu, width, xp)
        hv_new = hv_new - timestep_s / dx_m * _pad_east_edge(flux_hv, width, xp)
        east_m3 = -xp.sum(flux_h) * dy_m * timestep_s

    return h_new, hu_new, hv_new, west_m3, east_m3


def _apply_y_boundary_fluxes(
    h_new: Any,
    hu_new: Any,
    hv_new: Any,
    h: Any,
    hu: Any,
    hv: Any,
    bed: Any,
    dx_m: float,
    dy_m: float,
    timestep_s: float,
    gravity_mps2: float,
    min_depth_m: float,
    boundary: CoupledBackendBoundaryConditions,
    xp: Any,
    zero: Any,
) -> tuple[Any, Any, Any, Any, Any]:
    south_m3 = zero
    north_m3 = zero
    height = int(h.shape[0])

    south_mode = _normalise_boundary_mode(boundary.south_mode)
    if south_mode != "closed":
        h_ghost, hu_ghost, hv_ghost = _y_ghost_state(
            side="south",
            mode=south_mode,
            stage_m=boundary.south_stage_m,
            h=h,
            hu=hu,
            hv=hv,
            bed=bed,
            xp=xp,
        )
        flux_h, flux_hu, flux_hv = hll_shallow_water_normal_flux(
            h_ghost,
            hu_ghost,
            hv_ghost,
            h[0, :],
            hu[0, :],
            hv[0, :],
            normal_x=0.0,
            normal_y=1.0,
            gravity_mps2=gravity_mps2,
            min_depth_m=min_depth_m,
        )
        h_new = h_new + timestep_s / dy_m * _pad_south_edge(flux_h, height, xp)
        hu_new = hu_new + timestep_s / dy_m * _pad_south_edge(flux_hu, height, xp)
        hv_new = hv_new + timestep_s / dy_m * _pad_south_edge(flux_hv, height, xp)
        south_m3 = xp.sum(flux_h) * dx_m * timestep_s

    north_mode = _normalise_boundary_mode(boundary.north_mode)
    if north_mode != "closed":
        h_ghost, hu_ghost, hv_ghost = _y_ghost_state(
            side="north",
            mode=north_mode,
            stage_m=boundary.north_stage_m,
            h=h,
            hu=hu,
            hv=hv,
            bed=bed,
            xp=xp,
        )
        flux_h, flux_hu, flux_hv = hll_shallow_water_normal_flux(
            h[-1, :],
            hu[-1, :],
            hv[-1, :],
            h_ghost,
            hu_ghost,
            hv_ghost,
            normal_x=0.0,
            normal_y=1.0,
            gravity_mps2=gravity_mps2,
            min_depth_m=min_depth_m,
        )
        h_new = h_new - timestep_s / dy_m * _pad_north_edge(flux_h, height, xp)
        hu_new = hu_new - timestep_s / dy_m * _pad_north_edge(flux_hu, height, xp)
        hv_new = hv_new - timestep_s / dy_m * _pad_north_edge(flux_hv, height, xp)
        north_m3 = -xp.sum(flux_h) * dx_m * timestep_s

    return h_new, hu_new, hv_new, south_m3, north_m3


def _x_ghost_state(
    side: str,
    mode: str,
    stage_m: Any | float | None,
    h: Any,
    hu: Any,
    hv: Any,
    bed: Any,
    xp: Any,
) -> tuple[Any, Any, Any]:
    column = 0 if side == "west" else -1
    h_edge = h[:, column]
    hu_edge = hu[:, column]
    hv_edge = hv[:, column]
    if mode == "reflective":
        return h_edge, -hu_edge, hv_edge
    if mode == "transmissive":
        return h_edge, hu_edge, hv_edge
    if mode == "dirichlet_stage":
        h_ghost = _stage_depth(stage_m, bed[:, column], xp)
        return h_ghost, h_ghost * 0.0, h_ghost * 0.0
    raise ValueError(f"Unsupported x-boundary mode: {mode}.")


def _y_ghost_state(
    side: str,
    mode: str,
    stage_m: Any | float | None,
    h: Any,
    hu: Any,
    hv: Any,
    bed: Any,
    xp: Any,
) -> tuple[Any, Any, Any]:
    row = 0 if side == "south" else -1
    h_edge = h[row, :]
    hu_edge = hu[row, :]
    hv_edge = hv[row, :]
    if mode == "reflective":
        return h_edge, hu_edge, -hv_edge
    if mode == "transmissive":
        return h_edge, hu_edge, hv_edge
    if mode == "dirichlet_stage":
        h_ghost = _stage_depth(stage_m, bed[row, :], xp)
        return h_ghost, h_ghost * 0.0, h_ghost * 0.0
    raise ValueError(f"Unsupported y-boundary mode: {mode}.")


def _stage_depth(stage_m: Any | float | None, bed_edge: Any, xp: Any) -> Any:
    if stage_m is None:
        raise ValueError("dirichlet_stage boundary mode requires a stage value.")
    return xp.maximum(stage_m - bed_edge, 0.0)


def _normalise_boundary_mode(mode: str) -> str:
    value = mode.lower().replace("-", "_")
    if value in {"closed", "none", "no_flux"}:
        return "closed"
    if value in {"reflective", "wall"}:
        return "reflective"
    if value in {"transmissive", "open", "free"}:
        return "transmissive"
    if value in {"dirichlet_stage", "stage", "fixed_stage"}:
        return "dirichlet_stage"
    raise ValueError(f"Unsupported boundary mode: {mode}.")


def _apply_surface_sources(
    h: Any,
    hu: Any,
    hv: Any,
    forcing: CoupledBackendForcing,
    topology: CoupledBackendTopology,
    timestep_s: float,
) -> tuple[Any, Any, Any, Any, Any]:
    xp = _xp_from_array(h)
    active = h * 0.0 + 1.0 if topology.active_mask is None else topology.active_mask
    rainfall_depth = xp.maximum(forcing.rainfall_rate_mps, 0.0) * timestep_s * active
    h = h + rainfall_depth
    infiltration_depth = xp.minimum(
        h,
        xp.maximum(forcing.infiltration_capacity_mps, 0.0) * timestep_s * active,
    )
    h = h - infiltration_depth
    wet = h > forcing.min_depth_m
    hu = xp.where(wet, hu, 0.0)
    hv = xp.where(wet, hv, 0.0)
    return h, hu, hv, xp.sum(rainfall_depth) * topology.cell_area_m2, xp.sum(infiltration_depth) * topology.cell_area_m2


def _exchange_surface_nodes(
    h_flat: Any,
    bed_flat: Any,
    topology: CoupledBackendTopology,
    node_volume: Any,
    forcing: CoupledBackendForcing,
    timestep_s: float,
) -> tuple[Any, Any, Any, Any]:
    xp = _xp_from_array(h_flat)
    if int(topology.inlet_cell_index.shape[0]) == 0:
        return h_flat, node_volume, xp.asarray(0.0), xp.asarray(0.0)
    surface_head = bed_flat[topology.inlet_cell_index] + h_flat[topology.inlet_cell_index]
    heads = _node_head(node_volume, topology.node_invert_elevation_m, topology.node_storage_area_m2, xp)
    delta_head = surface_head - heads[topology.inlet_node_index]
    exchange_q = _orifice_q(
        delta_head,
        forcing.inlet_orifice_area_m2,
        forcing.inlet_orifice_coefficient,
        forcing.gravity_mps2,
        xp,
    )
    requested_capture = xp.maximum(exchange_q, 0.0) * timestep_s
    requested_capture = xp.minimum(
        requested_capture,
        _inlet_capacity_for_step(topology.inlet_capture_capacity_m3ps, requested_capture, timestep_s, xp),
    )
    available_surface = h_flat[topology.inlet_cell_index] * topology.cell_area_m2
    capture_m3 = xp.minimum(available_surface, requested_capture)
    requested_surcharge = xp.maximum(-exchange_q, 0.0) * timestep_s
    requested_surcharge = xp.minimum(
        requested_surcharge,
        _inlet_capacity_for_step(topology.inlet_surcharge_capacity_m3ps, requested_surcharge, timestep_s, xp),
    )
    surcharge_m3 = limit_node_exchange_by_available_volume(
        requested_surcharge,
        topology.inlet_node_index,
        node_volume,
    )
    cell_exchange = surcharge_m3 - capture_m3
    h_flat = h_flat + scatter_add_1d(topology.inlet_cell_index, cell_exchange / topology.cell_area_m2, int(h_flat.shape[0]))
    node_volume = node_volume + node_delta_from_cell_exchange(capture_m3 - surcharge_m3, topology.inlet_node_index, topology.node_count)
    return h_flat, node_volume, xp.sum(capture_m3), xp.sum(surcharge_m3)


def _return_overflow_to_surface(
    h_flat: Any,
    topology: CoupledBackendTopology,
    node_volume: Any,
    xp: Any,
) -> tuple[Any, Any, Any]:
    capacity = topology.node_storage_area_m2 * xp.maximum(topology.node_max_depth_m, 0.0)
    overflow = xp.maximum(node_volume - capacity, 0.0)
    valid = topology.node_surface_cell_index >= 0
    returned = xp.where(valid, overflow, 0.0)
    h_flat = h_flat + scatter_add_1d(
        xp.where(valid, topology.node_surface_cell_index, 0),
        returned / topology.cell_area_m2,
        int(h_flat.shape[0]),
    )
    node_volume = node_volume - returned
    return h_flat, node_volume, xp.sum(returned)


def _node_head(node_volume: Any, invert: Any, storage_area: Any, xp: Any) -> Any:
    return invert + xp.maximum(node_volume, 0.0) / xp.maximum(storage_area, 1.0e-12)


def _orifice_q(delta_head: Any, area: Any, coefficient: Any, gravity_mps2: float, xp: Any) -> Any:
    return coefficient * area * xp.sqrt(2.0 * gravity_mps2 * xp.abs(delta_head)) * xp.sign(delta_head)


def _stack_columns(xp: Any, *arrays: Any) -> Any:
    return xp.stack(arrays, axis=1)


def _as_column_array(value: Any, reference: Any) -> Any:
    if hasattr(value, "shape"):
        return value
    return reference * 0.0 + value


def _optional_capacity_or_unbounded(value: Any | None, reference: Any) -> Any:
    if value is None:
        return reference * 0.0 + np.inf
    return value


def _inlet_capacity_for_step(capacity_m3ps: Any | None, reference: Any, timestep_s: float, xp: Any) -> Any:
    if capacity_m3ps is None:
        return reference * 0.0 + np.inf
    return xp.maximum(capacity_m3ps, 0.0) * timestep_s


def _zero_scalar_like(array: Any, xp: Any) -> Any:
    return xp.sum(array * 0.0)


def _pad_right(array: Any, xp: Any) -> Any:
    return xp.pad(array, ((0, 0), (0, 1)))


def _pad_left(array: Any, xp: Any) -> Any:
    return xp.pad(array, ((0, 0), (1, 0)))


def _pad_bottom(array: Any, xp: Any) -> Any:
    return xp.pad(array, ((0, 1), (0, 0)))


def _pad_top(array: Any, xp: Any) -> Any:
    return xp.pad(array, ((1, 0), (0, 0)))


def _pad_west_edge(array: Any, width: int, xp: Any) -> Any:
    return xp.pad(array.reshape((-1, 1)), ((0, 0), (0, width - 1)))


def _pad_east_edge(array: Any, width: int, xp: Any) -> Any:
    return xp.pad(array.reshape((-1, 1)), ((0, 0), (width - 1, 0)))


def _pad_south_edge(array: Any, height: int, xp: Any) -> Any:
    return xp.pad(array.reshape((1, -1)), ((0, height - 1), (0, 0)))


def _pad_north_edge(array: Any, height: int, xp: Any) -> Any:
    return xp.pad(array.reshape((1, -1)), ((height - 1, 0), (0, 0)))


def _xp_from_array(array: Any):
    module_name = type(array).__module__
    module_root = module_name.split(".", maxsplit=1)[0]
    if module_root == "cupy":
        import cupy as cp  # type: ignore[import-not-found]

        return cp
    return np
