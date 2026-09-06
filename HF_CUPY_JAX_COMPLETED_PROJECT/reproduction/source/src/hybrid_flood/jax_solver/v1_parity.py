"""Pure-JAX primitives matching the V1 exact-production numerical sequence.

This module intentionally follows V1's array semantics rather than the
structured-grid solver.  The unchanged V1 CuPy run remains the release gate.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


GRAVITY = 9.80665
MIN_DEPTH = 1.0e-5
BOUNDARY_CLOSED = 0
BOUNDARY_TRANSMISSIVE = 1
BOUNDARY_REFLECTIVE = 2
BOUNDARY_STAGE = 3
BOUNDARY_OUTFLOW = 4


class SurfaceTopology(NamedTuple):
    bed: jnp.ndarray
    manning_n: jnp.ndarray
    cell_area: jnp.ndarray
    characteristic_length: jnp.ndarray
    edge_left: jnp.ndarray
    edge_right: jnp.ndarray
    edge_nx: jnp.ndarray
    edge_ny: jnp.ndarray
    edge_length: jnp.ndarray
    boundary_code: jnp.ndarray
    active: jnp.ndarray


class CouplingTopology(NamedTuple):
    inlet_cell: jnp.ndarray
    inlet_node: jnp.ndarray
    node_surface_cell: jnp.ndarray
    node_invert: jnp.ndarray
    node_storage_area: jnp.ndarray
    node_max_depth: jnp.ndarray
    inlet_capture_capacity: jnp.ndarray
    inlet_surcharge_capacity: jnp.ndarray
    surcharge_cell: jnp.ndarray
    surcharge_node: jnp.ndarray
    surcharge_weight: jnp.ndarray
    surcharge_weight_normalized: jnp.ndarray


class PipeTopology(NamedTuple):
    link_from: jnp.ndarray
    link_to: jnp.ndarray
    link_length: jnp.ndarray
    link_area: jnp.ndarray
    link_radius: jnp.ndarray
    link_manning: jnp.ndarray
    link_capacity: jnp.ndarray
    link_invert: jnp.ndarray
    link_diameter: jnp.ndarray
    link_minor_loss: jnp.ndarray
    link_geometry_code: jnp.ndarray
    link_rect_width: jnp.ndarray
    link_rect_height: jnp.ndarray
    outfall_area: jnp.ndarray
    outfall_coefficient: jnp.ndarray
    outfall_tailwater: jnp.ndarray
    outfall_capacity: jnp.ndarray


class RechargeTopology(NamedTuple):
    source_cell: jnp.ndarray
    storage_capacity: jnp.ndarray
    capture_capacity: jnp.ndarray
    recharge_capacity: jnp.ndarray
    stop_depth: jnp.ndarray


class ParityState(NamedTuple):
    h: jnp.ndarray
    hu: jnp.ndarray
    hv: jnp.ndarray
    wet_time_s: jnp.ndarray
    node_volume: jnp.ndarray
    link_flow: jnp.ndarray
    recharge_storage: jnp.ndarray


class StepDiagnostics(NamedTuple):
    rainfall_input: jnp.ndarray
    infiltration_loss: jnp.ndarray
    boundary_net_flux: jnp.ndarray
    boundary_inflow: jnp.ndarray
    boundary_outflow: jnp.ndarray
    external_boundary_inflow: jnp.ndarray
    surface_capture: jnp.ndarray
    surface_surcharge: jnp.ndarray
    outfall_discharge: jnp.ndarray
    overflow_surcharge: jnp.ndarray
    unreturned_overflow: jnp.ndarray
    recharge_capture: jnp.ndarray
    recharge_loss: jnp.ndarray


def scatter_add(indices: jnp.ndarray, values: jnp.ndarray, size: int) -> jnp.ndarray:
    return jnp.zeros((size,), dtype=values.dtype).at[indices].add(values)


def _hll(
    h_l, hu_l, hv_l, h_r, hu_r, hv_r, nx, ny,
    gravity: float = GRAVITY, dry: float = MIN_DEPTH,
):
    h_l = jnp.maximum(h_l, 0.0)
    h_r = jnp.maximum(h_r, 0.0)
    wet_l, wet_r = h_l > dry, h_r > dry
    u_l = jnp.where(wet_l, hu_l / jnp.maximum(h_l, dry), 0.0)
    v_l = jnp.where(wet_l, hv_l / jnp.maximum(h_l, dry), 0.0)
    u_r = jnp.where(wet_r, hu_r / jnp.maximum(h_r, dry), 0.0)
    v_r = jnp.where(wet_r, hv_r / jnp.maximum(h_r, dry), 0.0)
    un_l, un_r = u_l * nx + v_l * ny, u_r * nx + v_r * ny
    c_l, c_r = jnp.sqrt(gravity * h_l), jnp.sqrt(gravity * h_r)
    pressure_l, pressure_r = 0.5 * gravity * h_l**2, 0.5 * gravity * h_r**2
    f_l = jnp.stack((h_l * un_l, hu_l * un_l + pressure_l * nx, hv_l * un_l + pressure_l * ny))
    f_r = jnp.stack((h_r * un_r, hu_r * un_r + pressure_r * nx, hv_r * un_r + pressure_r * ny))
    q_l, q_r = jnp.stack((h_l, hu_l, hv_l)), jnp.stack((h_r, hu_r, hv_r))
    s_l = jnp.minimum(un_l - c_l, un_r - c_r)
    s_r = jnp.maximum(un_l + c_l, un_r + c_r)
    denom = jnp.where(jnp.abs(s_r - s_l) > 1.0e-12, s_r - s_l, 1.0)
    middle = (s_r * f_l - s_l * f_r + s_l * s_r * (q_r - q_l)) / denom
    flux = jnp.where(s_l[None] >= 0.0, f_l, jnp.where(s_r[None] <= 0.0, f_r, middle))
    return jnp.where((wet_l | wet_r)[None], flux, 0.0)


def _hydrostatic_flux_pair(h_l, hu_l, hv_l, z_l, h_r, hu_r, hv_r, z_r, nx, ny):
    h_l, h_r = jnp.maximum(h_l, 0.0), jnp.maximum(h_r, 0.0)
    z_star = jnp.maximum(z_l, z_r)
    hs_l = jnp.maximum(h_l + z_l - z_star, 0.0)
    hs_r = jnp.maximum(h_r + z_r - z_star, 0.0)
    wet_l, wet_r = h_l > MIN_DEPTH, h_r > MIN_DEPTH
    u_l = jnp.where(wet_l, hu_l / jnp.maximum(h_l, MIN_DEPTH), 0.0)
    v_l = jnp.where(wet_l, hv_l / jnp.maximum(h_l, MIN_DEPTH), 0.0)
    u_r = jnp.where(wet_r, hu_r / jnp.maximum(h_r, MIN_DEPTH), 0.0)
    v_r = jnp.where(wet_r, hv_r / jnp.maximum(h_r, MIN_DEPTH), 0.0)
    flux = _hll(hs_l, u_l * hs_l, v_l * hs_l, hs_r, u_r * hs_r, v_r * hs_r, nx, ny)
    pc_l = 0.5 * GRAVITY * (h_l**2 - hs_l**2)
    pc_r = 0.5 * GRAVITY * (h_r**2 - hs_r**2)
    correction_l = jnp.stack((jnp.zeros_like(pc_l), pc_l * nx, pc_l * ny))
    correction_r = jnp.stack((jnp.zeros_like(pc_r), pc_r * nx, pc_r * ny))
    return flux + correction_l, flux + correction_r


def stable_timestep(state: ParityState, topology: SurfaceTopology, cfl: float = 0.85):
    h_safe = jnp.maximum(state.h, MIN_DEPTH)
    speed = jnp.sqrt((state.hu / h_safe) ** 2 + (state.hv / h_safe) ** 2)
    speed = speed + jnp.sqrt(GRAVITY * h_safe)
    local = cfl * topology.characteristic_length / jnp.maximum(speed, 1.0e-12)
    return jnp.min(jnp.where(jnp.isfinite(local), local, jnp.inf))


def surface_flux_step(state: ParityState, topology: SurfaceTopology, dt):
    left, right = topology.edge_left, topology.edge_right
    active = topology.active > 0.5
    interior = right >= 0
    safe_right = jnp.where(interior, right, left)
    left_active = active[left]
    right_active = active[safe_right]
    active_interior = interior & left_active & right_active
    inactive_right = interior & left_active & ~right_active
    code = jnp.where(inactive_right, BOUNDARY_CLOSED, topology.boundary_code)
    h_l = jnp.where(left_active, state.h[left], 0.0)
    hu_l = jnp.where(left_active, state.hu[left], 0.0)
    hv_l = jnp.where(left_active, state.hv[left], 0.0)
    z_l = topology.bed[left]
    h_rb = jnp.where(right_active, state.h[safe_right], 0.0)
    hu_rb = jnp.where(right_active, state.hu[safe_right], 0.0)
    hv_rb = jnp.where(right_active, state.hv[safe_right], 0.0)
    z_rb = topology.bed[safe_right]
    normal_momentum = hu_l * topology.edge_nx + hv_l * topology.edge_ny
    hu_reflect = hu_l - 2.0 * normal_momentum * topology.edge_nx
    hv_reflect = hv_l - 2.0 * normal_momentum * topology.edge_ny
    reflective = (code == BOUNDARY_CLOSED) | (code == BOUNDARY_REFLECTIVE)
    h_r = jnp.where(active_interior, h_rb, h_l)
    hu_r = jnp.where(active_interior, hu_rb, jnp.where(reflective, hu_reflect, hu_l))
    hv_r = jnp.where(active_interior, hv_rb, jnp.where(reflective, hv_reflect, hv_l))
    z_r = jnp.where(active_interior, z_rb, z_l)
    flux_l, flux_r = _hydrostatic_flux_pair(
        h_l, hu_l, hv_l, z_l, h_r, hu_r, hv_r, z_r,
        topology.edge_nx, topology.edge_ny,
    )
    outflow_edge = (~interior) & (code == BOUNDARY_OUTFLOW)
    blocked_inflow = outflow_edge & (flux_l[0] < 0.0)
    flux_l = jnp.where(blocked_inflow[None], 0.0, flux_l)
    edge_length = topology.edge_length
    delta_l = -flux_l * edge_length
    delta_r = jnp.where(active_interior[None], flux_r * edge_length, 0.0)
    count = topology.cell_area.shape[0]
    safe_scatter = jnp.where(interior, right, 0)
    delta = jnp.stack([
        scatter_add(left, delta_l[i], count) + scatter_add(safe_scatter, delta_r[i], count)
        for i in range(3)
    ])
    h = jnp.where(active, jnp.maximum(state.h + dt * delta[0] / topology.cell_area, 0.0), 0.0)
    hu = jnp.where(active, state.hu + dt * delta[1] / topology.cell_area, 0.0)
    hv = jnp.where(active, state.hv + dt * delta[2] / topology.cell_area, 0.0)
    wet = h > MIN_DEPTH
    speed = jnp.sqrt((hu / jnp.maximum(h, MIN_DEPTH)) ** 2 + (hv / jnp.maximum(h, MIN_DEPTH)) ** 2)
    damping = 1.0 + dt * GRAVITY * topology.manning_n**2 * speed / jnp.maximum(h, MIN_DEPTH) ** (4.0 / 3.0)
    hu, hv = jnp.where(wet, hu / damping, 0.0), jnp.where(wet, hv / damping, 0.0)
    boundary = ~active_interior
    outward = jnp.where(boundary, flux_l[0] * edge_length * dt, 0.0)
    boundary_in = jnp.sum(jnp.maximum(-outward, 0.0))
    boundary_out = jnp.sum(jnp.maximum(outward, 0.0))
    return state._replace(h=h, hu=hu, hv=hv), boundary_in - boundary_out, boundary_in, boundary_out


def apply_rainfall_horton(
    state: ParityState,
    topology: SurfaceTopology,
    rainfall_rate_mps,
    rainfall_multiplier,
    horton_f0_mps,
    horton_fc_mps,
    horton_k_per_s,
    dt,
):
    rain_rate = jnp.maximum(rainfall_rate_mps, 0.0) * rainfall_multiplier
    rain_depth = rain_rate * dt * topology.active
    h_before = state.h + rain_depth
    capacity = horton_fc_mps + (horton_f0_mps - horton_fc_mps) * jnp.exp(
        -horton_k_per_s * state.wet_time_s
    )
    infiltration = jnp.minimum(h_before, jnp.maximum(capacity, 0.0) * dt * topology.active)
    h = h_before - infiltration
    scale = jnp.where(h_before > MIN_DEPTH, h / jnp.maximum(h_before, MIN_DEPTH), 0.0)
    hu = jnp.where(h > MIN_DEPTH, state.hu * scale, 0.0)
    hv = jnp.where(h > MIN_DEPTH, state.hv * scale, 0.0)
    return (
        state._replace(h=h, hu=hu, hv=hv),
        jnp.sum(rain_depth * topology.cell_area),
        jnp.sum(infiltration * topology.cell_area),
    )


def clamp_velocity(state: ParityState, maximum_speed: float = 5.0):
    wet = state.h > MIN_DEPTH
    speed = jnp.where(
        wet,
        jnp.sqrt(state.hu**2 + state.hv**2) / jnp.maximum(state.h, MIN_DEPTH),
        0.0,
    )
    scale = jnp.minimum(1.0, maximum_speed / jnp.maximum(speed, 1.0e-12))
    return state._replace(
        hu=jnp.where(wet, state.hu * scale, 0.0),
        hv=jnp.where(wet, state.hv * scale, 0.0),
    )


def _limit_by_group(requested, group, available, group_count):
    requested = jnp.maximum(requested, 0.0)
    total = scatter_add(group, requested, group_count)
    factor = jnp.minimum(1.0, available / jnp.maximum(total, 1.0e-12))
    return requested * factor[group]


def _node_head(volume, coupling: CouplingTopology):
    return coupling.node_invert + jnp.maximum(volume, 0.0) / jnp.maximum(coupling.node_storage_area, 1.0e-12)


def inlet_exchange(state: ParityState, surface: SurfaceTopology, coupling: CouplingTopology, dt):
    cell, node = coupling.inlet_cell, coupling.inlet_node
    node_head = _node_head(state.node_volume, coupling)
    surface_head = surface.bed[cell] + state.h[cell]
    delta_head = surface_head - node_head[node]
    exchange_q = 0.62 * 0.04 * jnp.sqrt(2.0 * GRAVITY * jnp.abs(delta_head)) * jnp.sign(delta_head)
    requested_capture = jnp.minimum(
        jnp.maximum(exchange_q, 0.0) * dt,
        jnp.maximum(coupling.inlet_capture_capacity, 0.0) * dt,
    )
    requested_surcharge = jnp.minimum(
        jnp.maximum(-exchange_q, 0.0) * dt,
        jnp.maximum(coupling.inlet_surcharge_capacity, 0.0) * dt,
    )
    available_by_cell = state.h * surface.cell_area
    capture = _limit_by_group(requested_capture, cell, available_by_cell, surface.cell_area.shape[0])
    surcharge = _limit_by_group(
        requested_surcharge, node, state.node_volume, coupling.node_storage_area.shape[0]
    )
    cell_exchange = scatter_add(
        cell, (surcharge - capture) / surface.cell_area[cell], surface.cell_area.shape[0]
    )
    node_exchange = scatter_add(
        node, capture - surcharge, coupling.node_storage_area.shape[0]
    )
    return (
        state._replace(
            h=jnp.maximum(state.h + cell_exchange, 0.0),
            node_volume=jnp.maximum(state.node_volume + node_exchange, 0.0),
        ),
        jnp.sum(capture),
        jnp.sum(surcharge),
    )


def _circular_area(depth, diameter):
    diameter = jnp.maximum(diameter, 1.0e-12)
    radius = 0.5 * diameter
    open_depth = jnp.minimum(jnp.maximum(depth, 0.0), diameter)
    theta = 2.0 * jnp.arccos(jnp.clip((radius - open_depth) / radius, -1.0, 1.0))
    open_area = 0.5 * radius**2 * (theta - jnp.sin(theta))
    full = 0.25 * jnp.pi * diameter**2
    return jnp.where(depth >= diameter, full, open_area)


def _circular_radius(depth, diameter):
    diameter = jnp.maximum(diameter, 1.0e-12)
    radius = 0.5 * diameter
    open_depth = jnp.minimum(jnp.maximum(depth, 0.0), diameter)
    theta = 2.0 * jnp.arccos(jnp.clip((radius - open_depth) / radius, -1.0, 1.0))
    perimeter = jnp.where(depth >= diameter, jnp.pi * diameter, radius * theta)
    return _circular_area(depth, diameter) / jnp.maximum(perimeter, 1.0e-12)


def pipe_links_and_outfalls(state: ParityState, coupling: CouplingTopology, pipe: PipeTopology, dt):
    head = _node_head(state.node_volume, coupling)
    source, target = pipe.link_from, pipe.link_to
    depth = jnp.maximum(0.5 * (head[source] + head[target]) - pipe.link_invert, 0.0)
    slot_width = GRAVITY * (0.25 * jnp.pi * jnp.maximum(pipe.link_diameter, 1.0e-12) ** 2) / 30.0**2
    circular_area = _circular_area(depth, pipe.link_diameter)
    flow_area = circular_area + slot_width * jnp.maximum(depth - pipe.link_diameter, 0.0)
    circular_radius = _circular_radius(depth, pipe.link_diameter)
    open_depth = jnp.minimum(jnp.maximum(depth, 0.0), jnp.maximum(pipe.link_rect_height, 1.0e-6))
    open_area = jnp.maximum(pipe.link_rect_width, 1.0e-6) * open_depth
    open_radius = open_area / jnp.maximum(pipe.link_rect_width + 2.0 * open_depth, 1.0e-12)
    flow_area = jnp.where(pipe.link_geometry_code == 2, open_area, flow_area)
    radius = jnp.where(pipe.link_geometry_code == 2, open_radius, circular_radius)
    length = jnp.maximum(pipe.link_length, 1.0e-6)
    area = jnp.maximum(flow_area, 1.0e-12)
    radius = jnp.maximum(radius, 1.0e-12)
    slope = (head[source] - head[target]) / length
    friction = GRAVITY * jnp.maximum(pipe.link_manning, 0.0) ** 2 * jnp.abs(state.link_flow) / (
        radius ** (4.0 / 3.0) * area
    )
    minor = jnp.maximum(pipe.link_minor_loss, 0.0) * jnp.abs(state.link_flow) / (2.0 * length * area)
    candidate = (state.link_flow + GRAVITY * area * dt * slope) / (1.0 + dt * (friction + minor))
    candidate = jnp.clip(candidate, -jnp.maximum(pipe.link_capacity, 0.0), jnp.maximum(pipe.link_capacity, 0.0))
    requested = candidate * dt
    positive = requested >= 0.0
    donor = jnp.where(positive, source, target)
    requested_abs = jnp.abs(requested)
    by_donor = scatter_add(donor, requested_abs, state.node_volume.shape[0])
    factor = jnp.minimum(1.0, state.node_volume / jnp.maximum(by_donor, 1.0e-12))
    transfer = jnp.where(positive, 1.0, -1.0) * requested_abs * factor[donor]
    node_delta = scatter_add(
        jnp.concatenate((source, target)),
        jnp.concatenate((-transfer, transfer)),
        state.node_volume.shape[0],
    )
    node_volume = jnp.maximum(state.node_volume + node_delta, 0.0)
    link_flow = transfer / jnp.maximum(dt, 1.0e-12)
    head = _node_head(node_volume, coupling)
    outfall_q = pipe.outfall_coefficient * pipe.outfall_area * jnp.sqrt(
        2.0 * GRAVITY * jnp.abs(head - pipe.outfall_tailwater)
    ) * jnp.sign(head - pipe.outfall_tailwater)
    outfall_q = jnp.minimum(outfall_q, jnp.maximum(pipe.outfall_capacity, 0.0))
    outfall = jnp.minimum(node_volume, jnp.maximum(outfall_q, 0.0) * dt)
    node_volume = jnp.maximum(node_volume - outfall, 0.0)
    return state._replace(node_volume=node_volume, link_flow=link_flow), jnp.sum(outfall)


def return_node_overflow(state: ParityState, surface: SurfaceTopology, coupling: CouplingTopology):
    capacity = coupling.node_storage_area * jnp.maximum(coupling.node_max_depth, 0.0)
    overflow = jnp.maximum(state.node_volume - capacity, 0.0)
    valid = (coupling.surcharge_node >= 0) & (coupling.surcharge_cell >= 0)
    safe_node = jnp.where(valid, coupling.surcharge_node, 0)
    safe_cell = jnp.where(valid, coupling.surcharge_cell, 0)
    raw_weight = jnp.where(valid, jnp.maximum(coupling.surcharge_weight, 0.0), 0.0)
    weight_sum = scatter_add(safe_node, raw_weight, state.node_volume.shape[0])
    normalized = jnp.where(
        coupling.surcharge_weight_normalized,
        raw_weight,
        jnp.where(valid, raw_weight / jnp.maximum(weight_sum[safe_node], 1.0e-12), 0.0),
    )
    returned_entry = overflow[safe_node] * normalized
    h = state.h + scatter_add(
        safe_cell, returned_entry / surface.cell_area[safe_cell], surface.cell_area.shape[0]
    )
    returned_node = scatter_add(safe_node, returned_entry, state.node_volume.shape[0])
    returned_node = jnp.minimum(returned_node, overflow)
    node_volume = state.node_volume - returned_node
    residual = jnp.maximum(node_volume - capacity, 0.0)
    explicit = scatter_add(
        safe_node, jnp.where(valid, jnp.ones_like(raw_weight), 0.0), state.node_volume.shape[0]
    ) > 0
    fallback_valid = (~explicit) & (coupling.node_surface_cell >= 0)
    fallback_cell = jnp.where(fallback_valid, coupling.node_surface_cell, 0)
    fallback = jnp.where(fallback_valid, residual, 0.0)
    h = h + scatter_add(
        fallback_cell, fallback / surface.cell_area[fallback_cell], surface.cell_area.shape[0]
    )
    node_volume = node_volume - fallback
    total = returned_node + fallback
    unreturned = jnp.sum(jnp.maximum(node_volume - capacity, 0.0))
    return state._replace(h=h, node_volume=node_volume), jnp.sum(total), unreturned


def apply_external_inflow(state: ParityState, receiver_cell, q_m3ps, surface: SurfaceTopology, dt):
    added = jnp.maximum(q_m3ps, 0.0) * dt
    depth = scatter_add(
        receiver_cell, added / surface.cell_area[receiver_cell], surface.cell_area.shape[0]
    )
    return state._replace(h=state.h + depth), jnp.sum(added)


def apply_recharge(state: ParityState, surface: SurfaceTopology, recharge: RechargeTopology, dt):
    cell = recharge.source_cell
    area = jnp.maximum(surface.cell_area[cell], 1.0e-6)
    available = jnp.maximum(state.h[cell] - recharge.stop_depth, 0.0) * area
    capture = jnp.minimum(
        jnp.minimum(available, recharge.capture_capacity * dt),
        jnp.maximum(recharge.storage_capacity - state.recharge_storage, 0.0),
    )
    removed_by_cell = scatter_add(cell, capture, surface.cell_area.shape[0])
    h = jnp.maximum(state.h - removed_by_cell / surface.cell_area, 0.0)
    storage = state.recharge_storage + capture
    loss = jnp.minimum(storage, recharge.recharge_capacity * dt)
    return state._replace(h=h, recharge_storage=storage - loss), jnp.sum(capture), jnp.sum(loss)


def advance_parity_step(
    state: ParityState,
    surface: SurfaceTopology,
    coupling: CouplingTopology,
    pipe: PipeTopology,
    recharge: RechargeTopology,
    *,
    rainfall_rate_mps,
    rainfall_multiplier,
    horton_f0_mps,
    horton_fc_mps,
    horton_k_per_s,
    external_receiver_cell,
    external_q_m3ps,
    dt,
):
    """Follow the V1 engine's exact per-step process ordering."""

    state, boundary_net, boundary_in, boundary_out = surface_flux_step(state, surface, dt)
    state, rain, infiltration = apply_rainfall_horton(
        state, surface, rainfall_rate_mps, rainfall_multiplier,
        horton_f0_mps, horton_fc_mps, horton_k_per_s, dt,
    )
    state = clamp_velocity(state)
    state, external = apply_external_inflow(
        state, external_receiver_cell, external_q_m3ps, surface, dt
    )
    state, capture, direct_surcharge = inlet_exchange(state, surface, coupling, dt)
    state, outfall = pipe_links_and_outfalls(state, coupling, pipe, dt)
    state, overflow, unreturned = return_node_overflow(state, surface, coupling)
    state, recharge_capture, recharge_loss = apply_recharge(state, surface, recharge, dt)
    state = state._replace(
        wet_time_s=state.wet_time_s + dt * (state.h > MIN_DEPTH)
    )
    diagnostics = StepDiagnostics(
        rain, infiltration, boundary_net, boundary_in, boundary_out, external,
        capture, direct_surcharge, outfall, overflow, unreturned,
        recharge_capture, recharge_loss,
    )
    return state, diagnostics
