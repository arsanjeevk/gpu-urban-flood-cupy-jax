"""Write the established V1 artifact contract from structured-grid results."""

from __future__ import annotations

import csv
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import xarray as xr
from pyproj import Transformer


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_v1_artifacts(
    dataset: xr.Dataset,
    output_dir: str | Path,
    rainfall_event,
    *,
    model_variant: str,
    report: dict,
    wet_threshold_m: float = 0.05,
) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    geojson_dir = output / "hourly_geojsons"
    geojson_dir.mkdir(exist_ok=True)
    times = np.asarray(dataset.time.values, dtype=float)
    depth = np.asarray(dataset.depth.values, dtype=np.float32)
    speed = np.asarray(dataset.velocity.values, dtype=np.float32)
    x, y = np.asarray(dataset.x.values), np.asarray(dataset.y.values)
    resolution = float(dataset.attrs["grid_resolution_m"])
    transformer = Transformer.from_crs(dataset.attrs["crs"], "EPSG:4326", always_xy=True)
    start = rainfall_event.frame["timestamp"].iloc[0]
    summary: list[dict] = []
    for index, actual_time in enumerate(times):
        # Native adaptive solvers record immediately after crossing a requested
        # reporting hour, so actual_time is intentionally not always divisible
        # by 3600.  Each supplied public snapshot still represents one frame.
        hour = int(round(actual_time / 3600.0))
        wet = np.argwhere(np.isfinite(depth[index]) & (depth[index] >= wet_threshold_m))
        features = []
        for row, column in wet:
            x0, x1 = x[column] - resolution / 2, x[column] + resolution / 2
            y0, y1 = y[row] - resolution / 2, y[row] + resolution / 2
            ring_xy = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
            ring = [list(transformer.transform(px, py)) for px, py in ring_xy]
            rain_rows = rainfall_event.frame[
                (rainfall_event.frame.start_min * 60 <= actual_time)
                & (rainfall_event.frame.end_min * 60 > actual_time)
            ]
            intensity = float(rain_rows.intensity_mmhr.iloc[0]) if len(rain_rows) else 0.0
            features.append({
                "type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": {
                    "cell_id": int(row * len(x) + column), "hour": hour,
                    "timestamp_ist": str(start + timedelta(seconds=float(actual_time))),
                    "actual_time_s": float(actual_time), "intensity_mmhr": intensity,
                    "depth_m": round(float(depth[index, row, column]), 4),
                    "speed_mps": round(float(speed[index, row, column]), 4),
                    "model_variant": model_variant,
                    "solver_version": report.get("model_version", "unknown"),
                    "grid_resolution_m": resolution,
                    "drainage_mode": report.get("drainage_mode", "unknown"),
                    "validation_status": report.get("validation_status", "unvalidated"),
                    "reference_model": report.get("reference_model", "none"),
                },
            })
        filename = f"hour_{hour:02d}_flood_depth.geojson"
        target = geojson_dir / filename
        target.write_text(json.dumps({"type": "FeatureCollection", "features": features}, separators=(",", ":")), encoding="utf-8")
        summary.append({
            "hour": hour, "timestamp_ist": str(start + timedelta(seconds=float(actual_time))),
            "intensity_mmhr": features[0]["properties"]["intensity_mmhr"] if features else 0.0,
            "actual_time_s": float(actual_time), "wet_cells_ge_threshold": len(features),
            "max_depth_m": float(np.nanmax(depth[index])), "geojson_file": filename,
            "geojson_size_kb": round(target.stat().st_size / 1024, 1), "model_variant": model_variant,
            "solver_version": report.get("model_version", "unknown"),
            "grid_resolution_m": resolution,
            "drainage_mode": report.get("drainage_mode", "unknown"),
            "validation_status": report.get("validation_status", "unvalidated"),
            "reference_model": report.get("reference_model", "none"),
        })
    summary_path = geojson_dir / "hourly_geojson_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader(); writer.writerows(summary)
    npz_path = output / "flood_simulation_outputs.npz"
    np.savez_compressed(
        npz_path, snapshot_times_s=times, snapshot_times_min=times / 60,
        depth_snapshots_m=depth, speed_snapshots_mps=speed,
        x=x, y=y, terrain_elevation_m=np.asarray(dataset.elevation.values),
        active_surface_mask=np.isfinite(dataset.elevation.values), model_variant=model_variant,
    )
    report_path = output / "run_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
    # Lightweight self-contained QA map and plot; operational UI consumes GeoJSON/WebP.
    (output / "hourly_depth_basemap.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>FloodAstra V2</title>"
        f"<h1>{model_variant} flood output</h1><p>See hourly_geojsons and run_report.json.</p>",
        encoding="utf-8",
    )
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(times / 3600, np.nanmax(depth, axis=(1, 2)))
    axis.set(xlabel="Simulation hour", ylabel="Maximum depth (m)", title=f"{model_variant} flood depth")
    figure.tight_layout(); figure.savefig(output / "flood_timeseries.png", dpi=140); plt.close(figure)
    return {
        "run_npz": str(npz_path), "hourly_geojson_dir": str(geojson_dir),
        "hourly_summary_csv": str(summary_path), "report_json": str(report_path),
        "basemap_html": str(output / "hourly_depth_basemap.html"),
        "timeseries_png": str(output / "flood_timeseries.png"),
    }
