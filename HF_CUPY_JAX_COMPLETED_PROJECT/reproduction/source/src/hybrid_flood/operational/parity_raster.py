"""Volume-conservative conversion of V1 triangle results to public map grids."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr

from hybrid_flood.jax_solver.shallow_water_2d import load_structured_grid

from .artifacts import write_v1_artifacts
from .errors import ConfigurationError, SimulationError
from .parity_assets import TOPOLOGY_RELATIVE
from .rainfall import load_rainfall_csv


def rasterize_reference_to_grid(
    *,
    reference_dir: str | Path,
    output_dir: str | Path,
    rainfall_csv: str | Path,
    v1_root: str | Path,
    prepared_dir: str | Path,
    domain_path: str | Path,
    resolution_m: float = 30.0,
    model_variant: str | None = None,
) -> dict[str, object]:
    """Lump triangle volume by centroid onto a regular grid without losing water volume."""

    reference = Path(reference_dir).resolve()
    native_path = reference / "flood_simulation_outputs.npz"
    reference_report_path = reference / "run_report.json"
    if not native_path.is_file() or not reference_report_path.is_file():
        raise ConfigurationError(f"Reference directory lacks V1 NPZ/report artifacts: {reference}")
    output = Path(output_dir).resolve()
    if output.exists():
        raise ConfigurationError(f"Raster output directory already exists: {output}")
    if resolution_m <= 0:
        raise ConfigurationError("Raster output resolution must be positive.")

    root = Path(v1_root).resolve()
    reference_report = json.loads(reference_report_path.read_text(encoding="utf-8"))
    variant = model_variant or str(reference_report.get("model_variant", "v1_cuda_reference"))
    grid = load_structured_grid(
        root / "data/terrain/hydraulic_dtm_fabdem_5m.tif",
        Path(prepared_dir).resolve() / "v1_roughness.tif",
        domain_path,
        resolution_m=resolution_m,
    )
    with np.load(root / TOPOLOGY_RELATIVE, allow_pickle=False) as topology:
        centroids = topology["centroids_world"].astype(np.float64)
        triangle_area = topology["surface_cell_area_m2"].astype(np.float64)
        active = topology["active_surface_mask"].astype(bool)

    x0 = grid.x[0] - resolution_m / 2.0
    y0 = grid.y[0] - resolution_m / 2.0
    columns = np.floor((centroids[:, 0] - x0) / resolution_m).astype(np.int64)
    rows = np.floor((centroids[:, 1] - y0) / resolution_m).astype(np.int64)
    inside = (
        active & (rows >= 0) & (rows < len(grid.y))
        & (columns >= 0) & (columns < len(grid.x))
    )
    if int(inside.sum()) != int(active.sum()):
        raise SimulationError(
            f"30 m grid does not cover every active V1 triangle: mapped {inside.sum()}/{active.sum()}."
        )
    linear = rows[inside] * len(grid.x) + columns[inside]
    cell_count = len(grid.y) * len(grid.x)
    source_area = triangle_area[inside]
    covered_area = np.bincount(linear, weights=source_area, minlength=cell_count)
    mapped_mask = covered_area.reshape(grid.bed.shape) > 0

    with np.load(native_path, allow_pickle=False) as native:
        required = {"snapshot_times_s", "depth_snapshots_m", "speed_snapshots_mps"}
        absent = sorted(required.difference(native.files))
        if absent:
            raise ConfigurationError(f"V1 reference NPZ lacks arrays: {absent}")
        times = native["snapshot_times_s"].astype(np.float64)
        triangle_depth = native["depth_snapshots_m"].astype(np.float32)
        triangle_speed = native["speed_snapshots_mps"].astype(np.float32)
    if triangle_depth.shape[1] != len(active) or triangle_speed.shape != triangle_depth.shape:
        raise ConfigurationError("V1 reference snapshot arrays do not match the production topology.")

    depth_grid = np.full((len(times), *grid.bed.shape), np.nan, dtype=np.float32)
    speed_grid = np.full_like(depth_grid, np.nan)
    subcell_max = np.full_like(depth_grid, np.nan)
    relative_volume_error: list[float] = []
    cell_area = float(resolution_m**2)
    for time_index in range(len(times)):
        depth = np.maximum(triangle_depth[time_index, inside].astype(np.float64), 0.0)
        speed = np.maximum(triangle_speed[time_index, inside].astype(np.float64), 0.0)
        volume_by_cell = np.bincount(
            linear, weights=depth * source_area, minlength=cell_count
        )
        momentum_weighted = np.bincount(
            linear, weights=depth * source_area * speed, minlength=cell_count
        )
        grid_depth_flat = volume_by_cell / cell_area
        grid_speed_flat = np.divide(
            momentum_weighted, volume_by_cell,
            out=np.zeros_like(momentum_weighted), where=volume_by_cell > 0,
        )
        maximum_flat = np.full(cell_count, -np.inf, dtype=np.float64)
        np.maximum.at(maximum_flat, linear, depth)
        maximum_flat[maximum_flat == -np.inf] = np.nan
        depth_grid[time_index, mapped_mask] = grid_depth_flat.reshape(grid.bed.shape)[mapped_mask]
        speed_grid[time_index, mapped_mask] = grid_speed_flat.reshape(grid.bed.shape)[mapped_mask]
        subcell_max[time_index] = maximum_flat.reshape(grid.bed.shape).astype(np.float32)
        native_volume = float(np.sum(depth * source_area))
        raster_volume = float(np.sum(grid_depth_flat) * cell_area)
        relative_volume_error.append(
            abs(raster_volume - native_volume) / max(abs(native_volume), 1.0)
        )

    elevation = grid.bed.astype(np.float32, copy=True)
    elevation[~mapped_mask] = np.nan
    dataset = xr.Dataset(
        data_vars={
            "elevation": (("y", "x"), elevation),
            "depth": (("time", "y", "x"), depth_grid),
            "velocity": (("time", "y", "x"), speed_grid),
            "max_subcell_depth": (("time", "y", "x"), subcell_max),
        },
        coords={"time": times, "x": grid.x, "y": grid.y},
        attrs={
            "title": f"{variant} conservatively rasterized for the V2 public map",
            "crs": grid.crs.to_string(),
            "grid_resolution_m": resolution_m,
            "rasterization": "triangle_centroid_lumped_volume_conservative",
        },
    )
    report = {
        **reference_report,
        "implementation": f"v2_{variant}_public_grid",
        "model_variant": variant,
        "model_version": reference_report.get(
            "model_version", "v1-frozen-reference" if variant == "v1_cuda_reference" else "unknown"
        ),
        "drainage_mode": reference_report.get("drainage_mode", "v1_full_dynamic_network"),
        "boundary_condition": reference_report.get(
            "boundary_condition", "v1_code4_outflow_only_with_historical_external_inflow"
        ),
        "solver_native_geometry": "v1_exact_production_triangles",
        "public_grid_resolution_m": resolution_m,
        "rasterization": "triangle_centroid_lumped_volume_conservative",
        "native_active_triangle_count": int(active.sum()),
        "native_active_area_km2": float(source_area.sum() / 1e6),
        "public_grid_cell_count": int(mapped_mask.sum()),
        "maximum_raster_volume_error_ratio": float(max(relative_volume_error, default=0.0)),
        "validation_status": reference_report.get(
            "validation_status", "reference_model_only" if variant == "v1_cuda_reference" else "unvalidated"
        ),
        "reference_model": reference_report.get(
            "reference_model", "v1" if variant == "v1_cuda_reference" else "none"
        ),
        "observationally_validated": False,
    }
    event = load_rainfall_csv(rainfall_csv)
    artifacts = write_v1_artifacts(
        dataset, output, event, model_variant=variant, report=report,
    )
    subcell_path = output / "subcell_max_depth.npz"
    np.savez_compressed(
        subcell_path, snapshot_times_s=times, max_subcell_depth_m=subcell_max,
        x=grid.x, y=grid.y, mapped_surface_mask=mapped_mask,
    )
    manifest = {
        "schema_version": 1,
        "model_variant": variant,
        "triangle_native_npz": str(native_path),
        "grid_artifacts": artifacts,
        "subcell_max_npz": str(subcell_path),
        "maximum_raster_volume_error_ratio": report["maximum_raster_volume_error_ratio"],
    }
    (output / "raster_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    comparison_path = reference / "comparison_manifest.json"
    if comparison_path.is_file() and output.parent == reference:
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        comparison.setdefault("candidate_artifacts", {})[f"public_grid_{resolution_m:g}m"] = {
            "output_dir": str(output),
            "raster_manifest": str(output / "raster_manifest.json"),
            "run_npz": artifacts["run_npz"],
            "run_report": artifacts["report_json"],
            "hourly_summary_csv": artifacts["hourly_summary_csv"],
            "subcell_max_npz": str(subcell_path),
        }
        temporary = comparison_path.with_suffix(".json.partial")
        temporary.write_text(json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(comparison_path)
    return manifest
