"""End-to-end operational JAX run with V1-compatible artifacts."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import jax.numpy as jnp
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree

from hybrid_flood.jax_solver.boundary_conditions import all_reflective
from hybrid_flood.jax_solver.shallow_water_2d import ShallowWaterSolver, load_structured_grid, result_to_xarray
from hybrid_flood.jax_solver.numerics import ScanResult, SWEState

from .artifacts import write_v1_artifacts
from .assets import rasterize_operational_fields
from .errors import BackendUnavailableError, ConfigurationError, DataFileMissingError, SimulationError
from .rainfall import load_rainfall_csv

MM_HR_TO_M_S = 1e-3 / 3600


def _run_hourly_chunks(solver: ShallowWaterSolver, output_times: np.ndarray) -> ScanResult:
    """Carry state through bounded scans to avoid full-event XLA compile pressure."""
    state = solver.initial_state()
    depth, momentum_x, momentum_y = [np.asarray(state.h)], [np.asarray(state.hu)], [np.asarray(state.hv)]
    total_steps = 0
    final_time = float(output_times[0])
    for start, stop in zip(output_times[:-1], output_times[1:], strict=True):
        interval = float(stop - start)
        # Peak rainfall can reduce the CFL step below two seconds. Keep one
        # fixed, reusable scan shape with ample per-hour headroom.
        max_steps = max(8_000, int(np.ceil(interval / 10.0)) * 8)
        chunk = solver.run(state, np.asarray([start, stop], dtype=np.float32), max_steps=max_steps)
        state = SWEState(chunk.states.h[-1], chunk.states.hu[-1], chunk.states.hv[-1])
        depth.append(np.asarray(state.h)); momentum_x.append(np.asarray(state.hu)); momentum_y.append(np.asarray(state.hv))
        total_steps += int(np.asarray(chunk.steps_used)); final_time = float(np.asarray(chunk.final_time_s))
    states = SWEState(jnp.asarray(np.stack(depth)), jnp.asarray(np.stack(momentum_x)), jnp.asarray(np.stack(momentum_y)))
    return ScanResult(
        states=states, output_times_s=jnp.asarray(output_times), final_time_s=jnp.asarray(final_time),
        outputs_written=jnp.asarray(len(output_times), dtype=jnp.int32),
        steps_used=jnp.asarray(total_steps, dtype=jnp.int32),
    )


def _boundary_forcing(path: Path, grid):
    if not path.is_file():
        return np.array([0, 1], np.float32), np.zeros(2, np.float32), np.zeros_like(grid.bed)
    table = pd.read_csv(path)
    hydrograph = table.groupby("time_s", sort=True).q_m3ps.sum()
    peaks = table.groupby("inflow_id").q_m3ps.max()
    locations = table.groupby("inflow_id").first()
    multiplier = np.zeros_like(grid.bed, dtype=np.float32)
    weights = peaks / max(peaks.sum(), np.finfo(float).eps)

    # Place source hydrographs on actual valid hydraulic boundary cells.  The
    # former searchsorted mapping could place an external inflow in the domain
    # interior or in a V1-invalid cell.
    boundary_mask = grid.domain_mask & ~binary_erosion(
        grid.domain_mask, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
        border_value=0,
    )
    boundary_rows, boundary_columns = np.nonzero(boundary_mask)
    if not len(boundary_rows):
        raise ConfigurationError("The V1-masked grid has no cells available for boundary inflow.")
    boundary_tree = cKDTree(np.column_stack((
        grid.x[boundary_columns], grid.y[boundary_rows],
    )))
    for key, row in locations.iterrows():
        _, nearest = boundary_tree.query([row.x_utm43n_m, row.y_utm43n_m])
        grid_row = int(boundary_rows[nearest])
        column = int(boundary_columns[nearest])
        multiplier[grid_row, column] += float(weights[key]) / (grid.resolution_m**2)
    return hydrograph.index.to_numpy(np.float32), hydrograph.to_numpy(np.float32), multiplier


def run_operational(
    *, rainfall_csv: str | Path, output_dir: str | Path, prepared_dir: str | Path,
    v1_root: str | Path, domain_path: str | Path, resolution_m: float = 20.0,
    recession_hours: float = 2.0, scenario: str = "upload", variant: str = "both",
    require_gpu: bool = True, checkpoint: str | Path | None = None,
) -> dict:
    if resolution_m not in {10.0, 20.0, 30.0, 50.0}:
        raise ConfigurationError("Resolution must be one of 10, 20, 30, or 50 metres.")
    if variant not in {"raw_jax", "hybrid", "both"}:
        raise ConfigurationError("Variant must be raw_jax, hybrid, or both.")
    try:
        import jax
        gpu = any(device.platform == "gpu" for device in jax.devices())
    except Exception as exc:
        raise BackendUnavailableError(str(exc)) from exc
    if require_gpu and not gpu:
        raise BackendUnavailableError("GPU execution is required but JAX found no GPU.")
    prepared, root = Path(prepared_dir), Path(v1_root)
    manifest_path = prepared / "asset_manifest.json"
    if not manifest_path.is_file():
        raise DataFileMissingError("Run prepare-assets before starting a real scenario.")
    event = load_rainfall_csv(rainfall_csv)
    polygon_grid = load_structured_grid(
        root / "data/terrain/hydraulic_dtm_fabdem_5m.tif",
        prepared / "v1_roughness.tif", domain_path, resolution_m=resolution_m,
    )
    fields = rasterize_operational_fields(polygon_grid, prepared)
    v1_valid_mask = polygon_grid.domain_mask & np.asarray(fields["valid_domain_mask"], dtype=bool)
    hydraulic_mask = v1_valid_mask & np.asarray(fields["active_mask"], dtype=bool)
    if not hydraulic_mask.any():
        raise ConfigurationError("V1 valid/active masks leave no hydraulic cells in the V2 domain.")
    grid = replace(
        polygon_grid,
        bed=np.where(hydraulic_mask, polygon_grid.bed, 0.0).astype(np.float32),
        manning_n=np.where(hydraulic_mask, polygon_grid.manning_n, 0.0).astype(np.float32),
        domain_mask=hydraulic_mask,
    )
    rain_times = np.append(
        event.frame.start_min.to_numpy(np.float32) * 60, np.float32(event.duration_s)
    )
    rain_rates = np.append(
        event.frame.intensity_mmhr.to_numpy(np.float32) * MM_HR_TO_M_S, np.float32(0.0)
    )
    if scenario == "historical":
        inflow = _boundary_forcing(
            root / "data/external_boundary_inflow/gurugram_external_boundary_inflow_july2025_133mm_12h.csv", grid
        )
    else:
        inflow = (np.array([0, 1], np.float32), np.zeros(2, np.float32), np.zeros_like(grid.bed))
    decay_per_hour = np.maximum(fields["horton_decay_per_hr"], 1e-6)
    solver = ShallowWaterSolver(
        grid, rainfall_times_s=rain_times, rainfall_rates_m_s=rain_rates,
        boundaries=all_reflective(),
        infiltration_initial_m_s=fields["horton_f0_mmhr"] * MM_HR_TO_M_S,
        infiltration_final_m_s=fields["horton_fc_mmhr"] * MM_HR_TO_M_S,
        infiltration_decay_s=3600.0 / decay_per_hour,
        drainage_capacity_m_s=fields["drainage_capacity_m_s"],
        drainage_activation_depth_m=0.05,
        boundary_inflow_times_s=inflow[0], boundary_inflow_rates_m_s=inflow[1],
        boundary_inflow_multiplier=inflow[2], max_dt_s=10.0, cfl=0.45,
    )
    duration = event.duration_s + recession_hours * 3600
    output_times = np.arange(0, duration + 1, 3600, dtype=np.float32)
    started = perf_counter()
    result = _run_hourly_chunks(solver, output_times)
    runtime = perf_counter() - started
    dataset = result_to_xarray(result, grid, dry_tolerance_m=1e-4)
    valid = dataset.depth.values[:, grid.domain_mask]
    if not np.isfinite(valid).all() or np.min(valid) < 0:
        raise SimulationError("Simulation produced negative or non-finite depths.")
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=False)
    warnings = [
        "Drainage uses capacity sinks; pipe routing, surcharge and backwater are not simulated.",
        "Hybrid is validated against V1, not observed flooding.",
        "Hydraulic cells are limited by the rasterized V1 valid and active surface masks.",
    ]
    cell_area_m2 = resolution_m**2
    polygon_area_km2 = float(polygon_grid.domain_mask.sum() * cell_area_m2 / 1e6)
    v1_valid_area_km2 = float(v1_valid_mask.sum() * cell_area_m2 / 1e6)
    hydraulic_area_km2 = float(hydraulic_mask.sum() * cell_area_m2 / 1e6)
    base_report = {
        "implementation": "hybrid_flood_v2_jax", "model_version": "0.3.0",
        "scenario": scenario, "runtime_s": runtime, "gpu_accelerated": gpu,
        "grid_resolution_m": resolution_m, "drainage_mode": "simplified_capacity_sink",
        "hydraulic_mask_source": "v1_valid_domain_mask_and_active_surface_mask",
        "polygon_domain_area_km2": polygon_area_km2,
        "v1_valid_area_km2": v1_valid_area_km2,
        "hydraulic_domain_area_km2": hydraulic_area_km2,
        "excluded_polygon_area_km2": polygon_area_km2 - hydraulic_area_km2,
        "hydraulic_cell_count": int(hydraulic_mask.sum()),
        "rainfall_total_mm": event.total_depth_mm, "rainfall_peak_mm_hr": event.peak_intensity_mm_hr,
        "duration_s": duration, "max_depth_m": float(np.nanmax(dataset.depth.values)),
        "mass_residual_ratio": None,
        "mass_balance_status": "source_sink_ledger_pending_gpu_validation",
        "validation_status": "v1_reference_only", "reference_model": "v1", "warnings": warnings,
        "asset_manifest": json.loads(manifest_path.read_text(encoding="utf-8")),
    }
    artifacts = {}
    raw_report = {**base_report, "model_variant": "raw_jax"}
    artifacts["raw_jax"] = write_v1_artifacts(dataset, output / "raw_jax", event, model_variant="raw_jax", report=raw_report)
    checkpoint_present = checkpoint is not None and Path(checkpoint).is_file()
    hybrid_available = False
    if variant in {"hybrid", "both"}:
        # Never relabel raw output as a hybrid prediction. Existing research
        # checkpoints are ANUGA/synthetic-grid checkpoints and are not valid
        # for this real-data feature signature.
        warnings.append(
            "Hybrid unavailable: a release-gated V1-teacher checkpoint for this grid is required."
        )
    comparison = {
        "schema_version": 1, "reference_model": "v1", "observationally_validated": False,
        "hybrid_available": hybrid_available, "checkpoint_present": checkpoint_present,
        "variants": artifacts, "metrics": {}, "warnings": warnings,
    }
    (output / "comparison_manifest.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    return comparison
