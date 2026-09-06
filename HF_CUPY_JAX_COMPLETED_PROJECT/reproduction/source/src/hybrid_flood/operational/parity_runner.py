"""Full-event exact-topology JAX runner for V1 numerical parity."""

from __future__ import annotations

from functools import partial
import json
from pathlib import Path
import shutil
from time import perf_counter
from typing import NamedTuple
import uuid

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

from hybrid_flood.jax_solver.v1_parity import (
    ParityState,
    StepDiagnostics,
    advance_parity_step,
    stable_timestep,
)

from .errors import BackendUnavailableError, ConfigurationError, SimulationError
from .parity_inputs import LoadedParityInputs, ParityForcing, load_parity_inputs
from .rainfall import load_rainfall_csv


class DeviceParityInputs(NamedTuple):
    surface: object
    coupling: object
    pipe: object
    recharge: object
    forcing: ParityForcing


class MassLedger(NamedTuple):
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


class AdvanceCarry(NamedTuple):
    state: ParityState
    time_s: jnp.ndarray
    time_compensation_s: jnp.ndarray
    cached_dt_s: jnp.ndarray
    global_step: jnp.ndarray
    ledger: MassLedger
    ledger_compensation: MassLedger


class BoundedAdvanceCarry(NamedTuple):
    state: ParityState
    time_s: jnp.ndarray
    time_compensation_s: jnp.ndarray
    cached_dt_s: jnp.ndarray
    global_step: jnp.ndarray
    ledger: MassLedger
    ledger_compensation: MassLedger
    interval_step: jnp.ndarray


def _zero_ledger() -> MassLedger:
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    return MassLedger(*(zero for _ in range(len(MassLedger._fields))))


def _accumulate_compensated(
    ledger: MassLedger, compensation: MassLedger, diagnostics: StepDiagnostics,
) -> tuple[MassLedger, MassLedger]:
    """Kahan accumulation reproduces V1's long-running float64 host ledger."""

    updated: list[jnp.ndarray] = []
    corrections: list[jnp.ndarray] = []
    for total, correction, increment in zip(ledger, compensation, diagnostics, strict=True):
        corrected_increment = increment - correction
        next_total = total + corrected_increment
        next_correction = (next_total - total) - corrected_increment
        updated.append(next_total)
        corrections.append(next_correction)
    return MassLedger(*updated), MassLedger(*corrections)


def _rain_rate(forcing: ParityForcing, time_s):
    index = jnp.searchsorted(forcing.rainfall_end_s, time_s, side="right")
    safe = jnp.clip(index, 0, forcing.rainfall_rate_mps.shape[0] - 1)
    valid = (index < forcing.rainfall_rate_mps.shape[0]) & (
        time_s >= forcing.rainfall_start_s[safe]
    )
    return jnp.where(valid, forcing.rainfall_rate_mps[safe], 0.0)


def _external_q(forcing: ParityForcing, time_s):
    times = forcing.external_times_s
    upper = jnp.searchsorted(times, time_s, side="right")
    lower = jnp.clip(upper - 1, 0, times.shape[0] - 1)
    upper_safe = jnp.clip(upper, 0, times.shape[0] - 1)
    denominator = jnp.maximum(times[upper_safe] - times[lower], 1.0e-12)
    fraction = jnp.clip((time_s - times[lower]) / denominator, 0.0, 1.0)
    return (
        forcing.external_q_m3ps[lower]
        + fraction * (forcing.external_q_m3ps[upper_safe] - forcing.external_q_m3ps[lower])
    )


def _perform_numerical_step(
    carry: AdvanceCarry, final_time_s, inputs: DeviceParityInputs,
) -> AdvanceCarry:
    """One exact V1-order numerical step, shared by scan and while runners."""

    state_i, time_i, time_compensation_i, cached_i, step_i, ledger_i, ledger_compensation_i = carry
    evaluate_cfl = (step_i % 32) == 0

    def calculate(_):
        return stable_timestep(state_i, inputs.surface, cfl=0.85)

    evaluated_dt = lax.cond(evaluate_cfl, calculate, lambda _: cached_i, operand=None)
    candidate = jnp.where(evaluate_cfl, evaluated_dt, evaluated_dt * 0.8)
    # V1 records immediately after crossing a requested reporting hour and
    # truncates only the final event step.
    dt = jnp.maximum(
        jnp.minimum(jnp.minimum(candidate, 1.0), final_time_s - time_i), 1.0e-3
    )
    rain = _rain_rate(inputs.forcing, time_i)
    external_q = _external_q(inputs.forcing, time_i + 0.5 * dt)
    new_state, diagnostics = advance_parity_step(
        state_i, inputs.surface, inputs.coupling, inputs.pipe, inputs.recharge,
        rainfall_rate_mps=rain,
        rainfall_multiplier=inputs.forcing.rainfall_multiplier,
        horton_f0_mps=inputs.forcing.horton_f0_mps,
        horton_fc_mps=inputs.forcing.horton_fc_mps,
        horton_k_per_s=inputs.forcing.horton_k_per_s,
        external_receiver_cell=inputs.forcing.external_receiver_cell,
        external_q_m3ps=external_q,
        dt=dt,
    )
    corrected_dt = dt - time_compensation_i
    next_time = time_i + corrected_dt
    next_time_compensation = (next_time - time_i) - corrected_dt
    next_ledger, next_ledger_compensation = _accumulate_compensated(
        ledger_i, ledger_compensation_i, diagnostics
    )
    return AdvanceCarry(
        new_state, next_time, next_time_compensation, evaluated_dt,
        step_i + 1, next_ledger, next_ledger_compensation,
    )


@partial(jax.jit, static_argnames=("max_steps",))
def _advance_to_time_scan(
    state: ParityState,
    time_s,
    time_compensation_s,
    cached_dt_s,
    global_step,
    ledger: MassLedger,
    ledger_compensation: MassLedger,
    target_time_s,
    final_time_s,
    inputs: DeviceParityInputs,
    *,
    max_steps: int,
):
    """Legacy fixed scan retained for reproducibility and A/B benchmarking."""

    def scan_step(carry, _):
        return lax.cond(
            carry.time_s < target_time_s,
            lambda active: _perform_numerical_step(active, final_time_s, inputs),
            lambda unchanged: unchanged,
            carry,
        ), None

    final, _ = lax.scan(
        scan_step,
        AdvanceCarry(
            state, time_s, time_compensation_s, cached_dt_s, global_step,
            ledger, ledger_compensation,
        ),
        xs=None,
        length=max_steps,
    )
    return final


@partial(
    jax.jit,
    static_argnames=("max_steps",),
    donate_argnums=(0,),
)
def _advance_to_time_while(
    state: ParityState,
    time_s,
    time_compensation_s,
    cached_dt_s,
    global_step,
    ledger: MassLedger,
    ledger_compensation: MassLedger,
    target_time_s,
    final_time_s,
    inputs: DeviceParityInputs,
    *,
    max_steps: int,
):
    """Execute only real timesteps and terminate on the requested output time."""

    initial = BoundedAdvanceCarry(
        state, time_s, time_compensation_s, cached_dt_s, global_step,
        ledger, ledger_compensation, jnp.asarray(0, dtype=jnp.int32),
    )

    def condition(carry: BoundedAdvanceCarry):
        return (carry.time_s < target_time_s) & (carry.interval_step < max_steps)

    def body(carry: BoundedAdvanceCarry):
        advanced = _perform_numerical_step(
            AdvanceCarry(*carry[:7]), final_time_s, inputs
        )
        return BoundedAdvanceCarry(*advanced, carry.interval_step + 1)

    final = lax.while_loop(condition, body, initial)
    return AdvanceCarry(*final[:7])


def _speed(state: ParityState) -> np.ndarray:
    h, hu, hv = (np.asarray(state.h), np.asarray(state.hu), np.asarray(state.hv))
    speed = np.zeros_like(h, dtype=np.float32)
    wet = h > 1.0e-4
    speed[wet] = np.sqrt(hu[wet] ** 2 + hv[wet] ** 2) / h[wet]
    return speed


def run_jax_v1_parity(
    *,
    rainfall_csv: str | Path,
    output_dir: str | Path,
    v1_root: str | Path,
    duration_min: float = 840.0,
    require_gpu: bool = True,
    max_steps_per_hour: int = 20_000,
    loop_mode: str = "while",
) -> dict[str, object]:
    """Execute full-SWE surface and complete drainage on exact V1 topology."""

    if duration_min <= 0:
        raise ConfigurationError("Parity duration must be positive.")
    if loop_mode not in {"while", "scan"}:
        raise ConfigurationError("loop_mode must be 'while' or 'scan'.")
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise ConfigurationError(f"Parity output directory already exists: {destination}")
    devices = jax.devices()
    gpu = any(device.platform == "gpu" for device in devices)
    if require_gpu and not gpu:
        raise BackendUnavailableError("jax_v1_parity requires a visible JAX GPU.")

    load_started = perf_counter()
    inputs = load_parity_inputs(rainfall_csv=rainfall_csv, v1_root=v1_root)
    device_inputs = DeviceParityInputs(
        inputs.surface, inputs.coupling, inputs.pipe, inputs.recharge, inputs.forcing
    )
    load_runtime = perf_counter() - load_started
    initial_surface = float(np.sum(np.asarray(inputs.initial_state.h) * inputs.host["surface_cell_area_m2"]))
    initial_node = float(np.sum(np.asarray(inputs.initial_state.node_volume)))
    duration_s = float(duration_min * 60.0)
    requested_times = np.arange(0.0, duration_s + 1.0, 3600.0, dtype=np.float32)
    if requested_times[-1] < duration_s:
        requested_times = np.append(requested_times, np.float32(duration_s))

    state = inputs.initial_state
    current_time = jnp.asarray(0.0, dtype=jnp.float32)
    time_compensation = jnp.asarray(0.0, dtype=jnp.float32)
    cached_dt = jnp.asarray(1.0, dtype=jnp.float32)
    global_step = jnp.asarray(0, dtype=jnp.int32)
    ledger = _zero_ledger()
    ledger_compensation = _zero_ledger()
    depth_snapshots = []
    speed_snapshots = []
    node_snapshots = []
    actual_output_times: list[float] = []
    steps_at_snapshot: list[int] = []
    chunk_runtimes_s: list[float] = []
    advance_to_time = _advance_to_time_while if loop_mode == "while" else _advance_to_time_scan
    run_started = perf_counter()
    for output_index, requested in enumerate(requested_times):
        # V1's t=0 snapshot is taken after its first numerical step.
        target = np.float32(1.0e-6) if output_index == 0 else requested
        chunk_started = perf_counter()
        (
            state, current_time, time_compensation, cached_dt, global_step,
            ledger, ledger_compensation,
        ) = advance_to_time(
            state, current_time, time_compensation, cached_dt, global_step,
            ledger, ledger_compensation,
            jnp.asarray(target, dtype=jnp.float32), jnp.asarray(duration_s, dtype=jnp.float32),
            device_inputs,
            max_steps=max_steps_per_hour,
        )
        jax.block_until_ready(state.h)
        chunk_runtimes_s.append(perf_counter() - chunk_started)
        reached = float(np.asarray(current_time))
        if reached + 1.0e-3 < float(target):
            raise SimulationError(
                f"max_steps_per_hour={max_steps_per_hour} reached only {reached:.3f}s "
                f"of target {float(target):.3f}s."
            )
        if not bool(np.isfinite(np.asarray(state.h)).all()):
            raise SimulationError(f"Non-finite surface depth at t={reached:.3f}s.")
        if not bool(np.isfinite(np.asarray(state.node_volume)).all()):
            raise SimulationError(f"Non-finite drainage storage at t={reached:.3f}s.")
        depth_snapshots.append(np.asarray(state.h, dtype=np.float32))
        speed_snapshots.append(_speed(state))
        node_snapshots.append(np.asarray(state.node_volume, dtype=np.float32))
        actual_output_times.append(reached)
        steps_at_snapshot.append(int(np.asarray(global_step)))
    solver_runtime = perf_counter() - run_started

    depth = np.stack(depth_snapshots)
    speed = np.stack(speed_snapshots)
    node_volume = np.stack(node_snapshots)
    ledger_host = {name: float(np.asarray(value)) for name, value in zip(MassLedger._fields, ledger, strict=True)}
    final_surface = float(np.sum(depth[-1].astype(np.float64) * inputs.host["surface_cell_area_m2"]))
    final_node = float(np.sum(node_volume[-1].astype(np.float64)))
    final_recharge_storage = float(np.sum(np.asarray(state.recharge_storage, dtype=np.float64)))
    lhs = initial_surface + initial_node + ledger_host["rainfall_input"] + ledger_host["external_boundary_inflow"]
    rhs_v1 = (
        final_surface + final_node + ledger_host["infiltration_loss"]
        + ledger_host["outfall_discharge"] + ledger_host["unreturned_overflow"]
        + ledger_host["recharge_loss"] - ledger_host["boundary_net_flux"]
    )
    rhs_complete = rhs_v1 + final_recharge_storage
    v1_residual = lhs - rhs_v1
    complete_residual = lhs - rhs_complete
    report = {
        "implementation": "hybrid_flood_v2_jax_v1_parity",
        "model_version": "0.3.0-parity",
        "model_variant": "jax_v1_parity",
        "topology_mode": "exact_production",
        "surface_step_mode": "full_swe",
        "drainage_mode": "v1_full_dynamic_network",
        "boundary_condition": "v1_code4_outflow_only_with_historical_external_inflow",
        "backend": "jax_gpu" if gpu else "jax_cpu",
        "is_gpu": gpu,
        "devices": [str(device) for device in devices],
        "duration_min": duration_min,
        "runtime_s": solver_runtime,
        "loop_mode": loop_mode,
        "maximum_steps_per_output_interval": max_steps_per_hour,
        "compiled_loop_iteration_budget": int(len(requested_times) * max_steps_per_hour),
        "executed_loop_iterations": int(np.asarray(global_step)),
        "inactive_tail_iterations_avoided": (
            int(len(requested_times) * max_steps_per_hour - int(np.asarray(global_step)))
            if loop_mode == "while" else 0
        ),
        "chunk_runtimes_s": chunk_runtimes_s,
        "first_chunk_compile_and_execution_s": chunk_runtimes_s[0],
        "remaining_chunks_execution_s": float(sum(chunk_runtimes_s[1:])),
        "input_load_s": load_runtime,
        "step_count": int(np.asarray(global_step)),
        "steps_at_snapshot": steps_at_snapshot,
        "steps_per_snapshot_interval": np.diff(
            np.asarray([0, *steps_at_snapshot], dtype=np.int64)
        ).tolist(),
        "snapshot_count": len(actual_output_times),
        "max_depth_m": float(np.max(depth)),
        "wet_triangle_count_005m": int(np.sum(
            (depth[-1] >= 0.05)
            & inputs.host["active_surface_mask"].astype(bool)
            & ~inputs.host["building_mask"].astype(bool)
        )),
        "drainage_node_count": int(node_volume.shape[1]),
        "drainage_link_count": int(inputs.host["link_from_node_index"].size),
        "mapped_inlet_count": int(inputs.host["inlet_cell_index"].size),
        "initial_surface_volume_m3": initial_surface,
        "initial_node_volume_m3": initial_node,
        "final_surface_volume_m3": final_surface,
        "final_node_volume_m3": final_node,
        "final_recharge_storage_m3": final_recharge_storage,
        **{f"{name}_m3": value for name, value in ledger_host.items()},
        "v1_compatible_mass_residual_m3": v1_residual,
        "v1_compatible_mass_residual_ratio": v1_residual / max(abs(lhs), 1.0),
        "complete_mass_residual_m3": complete_residual,
        "complete_mass_residual_ratio": complete_residual / max(abs(lhs), 1.0),
        "validation_status": "parity_candidate_unvalidated",
        "reference_model": "v1_cuda_reference",
        "observationally_validated": False,
        "input_report": inputs.report,
    }
    partial = destination.parent / f".{destination.name}.partial-{uuid.uuid4().hex}"
    partial.mkdir(parents=True, exist_ok=False)
    try:
        np.savez_compressed(
            partial / "flood_simulation_outputs.npz",
            snapshot_times_min=np.asarray(actual_output_times, dtype=np.float32) / 60.0,
            snapshot_times_s=np.asarray(actual_output_times, dtype=np.float32),
            steps_at_snapshot=np.asarray(steps_at_snapshot, dtype=np.int64),
            depth_snapshots_m=depth,
            speed_snapshots_mps=speed,
            drain_node_volume_snapshots_m3=node_volume,
            final_depth_m=depth[-1],
            max_depth_m=np.max(depth, axis=0),
            final_hu_m2ps=np.asarray(state.hu, dtype=np.float32),
            final_hv_m2ps=np.asarray(state.hv, dtype=np.float32),
            max_speed_mps=np.max(speed, axis=0),
            final_node_volume_m3=node_volume[-1],
            final_link_flow_m3ps=np.asarray(state.link_flow, dtype=np.float32),
            nodes_world=inputs.host["nodes_world"].astype(np.float32),
            triangles=inputs.host["triangles"].astype(np.int32),
            centroids_world=inputs.host["centroids_world"].astype(np.float32),
            terrain_elevation_m=inputs.host["terrain_elevation_m"].astype(np.float32),
            active_surface_mask=inputs.host["active_surface_mask"].astype(bool),
            building_mask=inputs.host["building_mask"].astype(bool),
            road_mask=inputs.host["road_mask"].astype(bool),
            drain_node_x_m=inputs.host["node_x_m"].astype(np.float32),
            drain_node_y_m=inputs.host["node_y_m"].astype(np.float32),
            drain_link_from=inputs.host["link_from_node_index"].astype(np.int32),
            drain_link_to=inputs.host["link_to_node_index"].astype(np.int32),
        )
        (partial / "run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        partial.rename(destination)
    except Exception:
        if partial.exists():
            shutil.rmtree(partial)
        raise
    return report
