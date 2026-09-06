"""The citywide flood simulation engine.

This is the notebook time loop (``gurugram_nowcast_run.ipynb`` /
``gurugram_cuda_impl_style_full_city_deployment_new.ipynb``) turned into a
function. The numerical sequence per step is preserved exactly:

1. global CFL timestep, cached for ``cfl_update_interval_steps`` steps and
   shrunk by ``cfl_reuse_safety_factor`` when reused;
2. rainfall + Horton infiltration forcing for the step;
3. fused CUDA surface update (full SWE or local-inertial);
4. ANUGA-style velocity clamp;
5. (optional) external boundary inflow added as depth;
6. fused CUDA dynamic pipe-network step (inlet exchange, links, outfalls);
7. (optional) surface recharge capture;
8. wet-time accumulation for Horton decay.

Any change to this ordering or to the operations themselves changes the
validated numerical behaviour — treat this file as part of the solver, not
as orchestration glue.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..config import RunConfig
from ..domain.june5_production_drainage import apply_june5_surface_recharge, recharge_to_backend
from ..domain.production_event_support import (
    apply_external_boundary_inflow_depth,
    scale_node_volume_to_target,
)
from ..domain.production_infiltration_support import (
    build_infiltration_capacity_for_step,
    rainfall_rate_for_step,
    wet_time_increment_s,
)
from ..exceptions import BackendUnavailableError, SimulationError
from ..kernels.backends import get_array_backend
from ..kernels.config import BackendConfig
from ..kernels.triangular_backend import (
    TriangularPipeNetworkState,
    TriangularSurfaceForcing,
    TriangularSurfaceState,
    stable_local_inertial_triangular_timestep_s,
    stable_triangular_timestep_s,
    step_triangular_dynamic_pipe_network,
    step_triangular_surface,
    step_triangular_surface_local_inertial,
    triangular_node_coupling_to_backend,
    triangular_pipe_network_to_backend,
    triangular_topology_to_backend,
)
from ..logging_utils import get_logger
from ..preprocessing.city_data import CityDomain
from ..preprocessing.rainfall import RainfallEvent
from .stabilizers import clamp_velocity, to_host

log = get_logger("solver")

# Progress callback: (fraction_complete, snapshot_info_dict) -> None
ProgressCallback = Callable[[float, dict[str, Any]], None]


@dataclass
class SimulationResult:
    """In-memory result of one simulation run (host arrays only)."""

    # Snapshots: index 0 is the first snapshot at/after t=0.
    snapshot_times_min: np.ndarray            # (n_snapshots,)
    depth_snapshots_m: np.ndarray             # (n_snapshots, n_triangles) float32
    speed_snapshots_mps: np.ndarray           # (n_snapshots, n_triangles) float32
    node_volume_snapshots_m3: np.ndarray      # (n_snapshots, n_nodes) float32

    # Final state
    final_depth_m: np.ndarray
    final_hu_m2ps: np.ndarray
    final_hv_m2ps: np.ndarray
    final_node_volume_m3: np.ndarray
    final_link_flow_m3ps: np.ndarray

    # Envelopes
    max_depth_m: np.ndarray
    max_speed_mps: np.ndarray

    # Mass balance and cumulative diagnostics (m3)
    mass_balance: dict[str, float]

    # Run metadata
    step_count: int
    runtime_s: float
    cfl_eval_count: int
    cfl_reused_count: int
    backend_name: str
    is_gpu: bool
    duration_min: float

    # References to the static inputs (for exporters)
    city: CityDomain = field(repr=False, default=None)
    rainfall: RainfallEvent = field(repr=False, default=None)
    config: RunConfig = field(repr=False, default=None)


def _get_backend(name: str):
    try:
        backend = get_array_backend(BackendConfig(name=name))
    except Exception as exc:
        raise BackendUnavailableError(
            f"Could not initialise the '{name}' array backend: {exc}. "
            "Install a CuPy wheel matching the CUDA runtime and ensure an NVIDIA "
            "device is visible. CPU fallback is disabled for production runs."
        ) from exc
    return backend


def run_simulation(
    city: CityDomain,
    rainfall: RainfallEvent,
    cfg: RunConfig,
    progress_callback: ProgressCallback | None = None,
) -> SimulationResult:
    """Run the full citywide event and return host-side results.

    ``city`` comes from :func:`gurugram_flood.preprocessing.load_city_domain`,
    ``rainfall`` from :func:`gurugram_flood.preprocessing.load_rainfall_event`.
    """
    solver_cfg = cfg.solver
    physics_cfg = cfg.physics
    topo = city.topologies
    mesh = city.mesh

    duration_min = solver_cfg.duration_min if solver_cfg.duration_min is not None else rainfall.duration_min
    if duration_min <= 0:
        raise SimulationError(f"Simulation duration must be positive, got {duration_min} min")

    backend = _get_backend(solver_cfg.backend)
    xp = backend.xp
    log.info("Backend: %s (GPU=%s)", backend.name, backend.is_gpu)
    log.info(
        "Run: %.0f min, surface=%s, infiltration=%s, CFL=%.2f, max_dt=%.2fs, snapshots every %.0f min",
        duration_min, solver_cfg.surface_step_mode, physics_cfg.infiltration_mode,
        solver_cfg.cfl, solver_cfg.max_dt_s, solver_cfg.snapshot_interval_min,
    )

    # ---- move static topology + initial state to the backend ---------------
    surface_topology = triangular_topology_to_backend(topo.surface_topology, backend)
    coupling_topology = triangular_node_coupling_to_backend(topo.coupling_topology, backend)
    pipe_topology = triangular_pipe_network_to_backend(topo.pipe_topology, backend)

    state = TriangularSurfaceState(
        h_m=xp.array(topo.surface_state.h_m),
        hu_m2ps=xp.array(topo.surface_state.hu_m2ps),
        hv_m2ps=xp.array(topo.surface_state.hv_m2ps),
    )
    pipe_state = TriangularPipeNetworkState(
        node_volume_m3=xp.array(topo.pipe_state.node_volume_m3),
        link_flow_m3ps=xp.array(topo.pipe_state.link_flow_m3ps),
    )
    wet_time_s = xp.zeros((mesh.triangle_count,), dtype=xp.float32)

    simple_infiltration = None
    if city.simple_infiltration_mps is not None:
        simple_infiltration = xp.array(city.simple_infiltration_mps)

    recharge_backend = None
    recharge_storage = None
    if city.recharge_topology is not None:
        recharge_backend = recharge_to_backend(city.recharge_topology, backend)
        recharge_storage = xp.zeros((city.recharge_topology.row_count,), dtype=xp.float32)

    if physics_cfg.match_reference_initial_node_volume and city.topology_mode == "exact_production":
        scaled_volume, init_scale = scale_node_volume_to_target(
            pipe_state.node_volume_m3, physics_cfg.reference_initial_node_volume_m3, xp
        )
        pipe_state = TriangularPipeNetworkState(
            node_volume_m3=scaled_volume, link_flow_m3ps=pipe_state.link_flow_m3ps
        )
        log.info(
            "Scaled initial node volume to reference %.1f m3 (scale=%.6f)",
            physics_cfg.reference_initial_node_volume_m3, float(init_scale),
        )

    initial_surface_vol_m3 = float(np.sum(to_host(state.h_m) * topo.surface_topology.cell_area_m2))
    initial_node_vol_m3 = float(to_host(pipe_state.node_volume_m3).sum())

    # ---- time loop ----------------------------------------------------------
    final_time_s = duration_min * 60.0
    snapshot_interval_s = solver_cfg.snapshot_interval_min * 60.0
    next_snapshot_s = 0.0

    snapshot_times_min: list[float] = []
    depth_snapshots: list[np.ndarray] = []
    speed_snapshots: list[np.ndarray] = []
    node_volume_snapshots: list[np.ndarray] = []

    t_sim = 0.0
    step_count = 0
    cached_dt = None
    cfl_eval_count = 0
    cfl_reused_count = 0
    cum = {
        "rainfall_input_m3": 0.0,
        "infiltration_loss_m3": 0.0,
        "boundary_net_flux_m3": 0.0,
        "outfall_discharge_m3": 0.0,
        "overflow_surcharge_m3": 0.0,
        "unreturned_overflow_m3": 0.0,
        "external_boundary_inflow_m3": 0.0,
        "surface_recharge_capture_m3": 0.0,
        "surface_recharge_loss_m3": 0.0,
    }

    valid_mask = mesh.active_mask & ~mesh.building_mask
    t_run0 = time.perf_counter()

    while t_sim < final_time_s:
        if cached_dt is None or step_count % solver_cfg.cfl_update_interval_steps == 0:
            if solver_cfg.surface_step_mode == "local_inertial":
                # LI stability needs only gravity-wave speed sqrt(g*h): larger dt, fewer steps
                dt = float(stable_local_inertial_triangular_timestep_s(surface_topology, state, cfl=solver_cfg.cfl))
            else:
                dt = float(stable_triangular_timestep_s(surface_topology, state, cfl=solver_cfg.cfl))
            cached_dt = dt
            cfl_eval_count += 1
        else:
            dt = cached_dt * solver_cfg.cfl_reuse_safety_factor
            cfl_reused_count += 1
        dt = min(dt, solver_cfg.max_dt_s, final_time_s - t_sim)
        dt = max(dt, 1.0e-3)

        base_rainfall_rate_mps = rainfall.rate_fn(t_sim)
        rainfall_rate_step = rainfall_rate_for_step(
            xp=xp,
            base_rainfall_rate_mps=base_rainfall_rate_mps,
            mode=physics_cfg.infiltration_mode,
            fields=city.infiltration_fields,
        )
        infiltration_capacity_step = build_infiltration_capacity_for_step(
            xp=xp,
            mode=physics_cfg.infiltration_mode,
            wet_time_s=wet_time_s,
            fields=city.infiltration_fields,
            fallback_capacity_mps=simple_infiltration,
        )
        forcing = TriangularSurfaceForcing(
            rainfall_rate_mps=rainfall_rate_step,
            infiltration_capacity_mps=infiltration_capacity_step,
        )

        if solver_cfg.surface_step_mode == "local_inertial":
            state, surf_diag = step_triangular_surface_local_inertial(
                state, surface_topology, forcing, timestep_s=dt
            )
        else:
            state, surf_diag = step_triangular_surface(
                state, surface_topology, forcing, timestep_s=dt
            )
        state = clamp_velocity(state, solver_cfg.max_speed_mps, xp)

        if city.external_boundary is not None:
            h_with_ext, ext_added_m3 = apply_external_boundary_inflow_depth(
                state.h_m,
                surface_topology.cell_area_m2,
                city.external_boundary,
                timestep_s=dt,
                time_s=t_sim + 0.5 * dt,
                xp=xp,
            )
            state = TriangularSurfaceState(h_m=h_with_ext, hu_m2ps=state.hu_m2ps, hv_m2ps=state.hv_m2ps)
            cum["external_boundary_inflow_m3"] += float(ext_added_m3)

        h, pipe_state, pipe_diag = step_triangular_dynamic_pipe_network(
            state.h_m,
            surface_topology=surface_topology,
            coupling_topology=coupling_topology,
            pipe_topology=pipe_topology,
            pipe_state=pipe_state,
            timestep_s=dt,
        )

        if recharge_backend is not None:
            h, recharge_storage, r_capture, r_loss = apply_june5_surface_recharge(
                h, recharge_backend, recharge_storage, surface_topology.cell_area_m2, dt, xp
            )
            cum["surface_recharge_capture_m3"] += float(r_capture)
            cum["surface_recharge_loss_m3"] += float(r_loss)

        state = TriangularSurfaceState(h_m=h, hu_m2ps=state.hu_m2ps, hv_m2ps=state.hv_m2ps)
        wet_time_s = wet_time_s + wet_time_increment_s(
            xp=xp, depth_m=state.h_m, dt_s=dt, min_depth_m=forcing.min_depth_m
        )

        cum["rainfall_input_m3"] += float(surf_diag.rainfall_input_m3)
        cum["infiltration_loss_m3"] += float(surf_diag.infiltration_loss_m3)
        cum["boundary_net_flux_m3"] += float(surf_diag.boundary_net_flux_m3)
        cum["outfall_discharge_m3"] += float(pipe_diag.outfall_discharge_m3)
        cum["overflow_surcharge_m3"] += float(pipe_diag.overflow_surcharge_m3)
        cum["unreturned_overflow_m3"] += float(pipe_diag.unreturned_overflow_m3)

        t_sim += dt
        step_count += 1

        if step_count % solver_cfg.finite_check_interval_steps == 0:
            if not bool(xp.isfinite(state.h_m).all()):
                raise SimulationError(
                    f"Non-finite depth detected at step {step_count} (t={t_sim/60:.1f} min). "
                    "Lower solver.cfl_update_interval_steps or solver.cfl_reuse_safety_factor."
                )

        if t_sim >= next_snapshot_s or t_sim >= final_time_s:
            h_host = to_host(state.h_m)
            hu_host = to_host(state.hu_m2ps)
            hv_host = to_host(state.hv_m2ps)
            speed = np.zeros_like(h_host)
            wet = h_host > 1.0e-4
            speed[wet] = np.sqrt(hu_host[wet] ** 2 + hv_host[wet] ** 2) / h_host[wet]

            snapshot_times_min.append(t_sim / 60.0)
            depth_snapshots.append(h_host.astype(np.float32))
            speed_snapshots.append(speed.astype(np.float32))
            node_volume_snapshots.append(to_host(pipe_state.node_volume_m3).astype(np.float32))

            wet_count = int((h_host[valid_mask] >= cfg.outputs.wet_threshold_m).sum())
            elapsed = time.perf_counter() - t_run0
            info = {
                "t_min": t_sim / 60.0,
                "dt_s": dt,
                "step": step_count,
                "wet_triangles": wet_count,
                "max_depth_m": float(h_host.max()),
                "steps_per_s": step_count / max(elapsed, 1.0e-9),
                "elapsed_s": elapsed,
            }
            log.info(
                "t=%7.1f min | dt=%5.3fs | step=%9s | wet=%9s | max_h=%6.3f m | %.1f steps/s",
                info["t_min"], dt, f"{step_count:,}", f"{wet_count:,}",
                info["max_depth_m"], info["steps_per_s"],
            )
            if progress_callback is not None:
                progress_callback(min(t_sim / final_time_s, 1.0), info)
            next_snapshot_s += snapshot_interval_s

    runtime_s = time.perf_counter() - t_run0
    log.info(
        "Done: %s steps in %.2f min wall-clock (%.1f steps/s)",
        f"{step_count:,}", runtime_s / 60.0, step_count / max(runtime_s, 1e-9),
    )

    # ---- finals + mass balance ----------------------------------------------
    final_h = to_host(state.h_m).astype(np.float32)
    final_hu = to_host(state.hu_m2ps).astype(np.float32)
    final_hv = to_host(state.hv_m2ps).astype(np.float32)
    final_node_volume = to_host(pipe_state.node_volume_m3).astype(np.float32)
    final_link_flow = to_host(pipe_state.link_flow_m3ps).astype(np.float32)

    depth_stack = np.stack(depth_snapshots)
    speed_stack = np.stack(speed_snapshots)
    max_depth = np.max(depth_stack, axis=0).astype(np.float32)
    max_speed = np.max(speed_stack, axis=0).astype(np.float32)

    final_surface_vol_m3 = float(np.sum(final_h * topo.surface_topology.cell_area_m2))
    final_node_vol_m3 = float(final_node_volume.sum())
    lhs = initial_surface_vol_m3 + initial_node_vol_m3 + cum["rainfall_input_m3"] + cum["external_boundary_inflow_m3"]
    rhs = (
        final_surface_vol_m3 + final_node_vol_m3
        + cum["infiltration_loss_m3"] + cum["outfall_discharge_m3"]
        + cum["unreturned_overflow_m3"] + cum["surface_recharge_loss_m3"]
        - cum["boundary_net_flux_m3"]
    )
    mass_residual_m3 = lhs - rhs
    mass_balance = {
        **cum,
        "initial_surface_volume_m3": initial_surface_vol_m3,
        "initial_node_volume_m3": initial_node_vol_m3,
        "final_surface_volume_m3": final_surface_vol_m3,
        "final_node_volume_m3": final_node_vol_m3,
        "mass_residual_m3": mass_residual_m3,
        "mass_residual_ratio": mass_residual_m3 / max(abs(lhs), 1.0),
    }
    log.info(
        "Mass balance: residual %.1f m3 (ratio %.2e)",
        mass_residual_m3, mass_balance["mass_residual_ratio"],
    )

    return SimulationResult(
        snapshot_times_min=np.asarray(snapshot_times_min, dtype=np.float32),
        depth_snapshots_m=depth_stack,
        speed_snapshots_mps=speed_stack,
        node_volume_snapshots_m3=np.stack(node_volume_snapshots),
        final_depth_m=final_h,
        final_hu_m2ps=final_hu,
        final_hv_m2ps=final_hv,
        final_node_volume_m3=final_node_volume,
        final_link_flow_m3ps=final_link_flow,
        max_depth_m=max_depth,
        max_speed_mps=max_speed,
        mass_balance=mass_balance,
        step_count=step_count,
        runtime_s=runtime_s,
        cfl_eval_count=cfl_eval_count,
        cfl_reused_count=cfl_reused_count,
        backend_name=backend.name,
        is_gpu=bool(backend.is_gpu),
        duration_min=float(duration_min),
        city=city,
        rainfall=rainfall,
        config=cfg,
    )
