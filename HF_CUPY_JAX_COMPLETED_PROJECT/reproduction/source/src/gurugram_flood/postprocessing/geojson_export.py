"""Hourly flood-depth GeoJSON export (the primary frontend artifact).

Reproduces the nowcast notebook's exporter: one WGS84 GeoJSON per target
hour, each feature a wet mesh triangle with depth/speed/rainfall metadata,
plus an ``hourly_geojson_summary.csv`` index the frontend can read first.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

from ..logging_utils import get_logger
from ..solver.engine import SimulationResult

log = get_logger("geojson")

GEOJSON_DIR_NAME = "hourly_geojsons"
SUMMARY_CSV_NAME = "hourly_geojson_summary.csv"


@dataclass
class HourlySelection:
    """Snapshot indices chosen for each whole hour of the run."""

    target_hours: np.ndarray       # (n_hours,) int, 0..last_hour
    snapshot_indices: np.ndarray   # (n_hours,) index into snapshot arrays
    actual_times_s: np.ndarray     # (n_hours,) times of the chosen snapshots


def select_hourly_snapshots(snapshot_times_min: np.ndarray) -> HourlySelection:
    """Pick, for each whole hour, the snapshot nearest in time.

    With the default 60-min snapshot interval the selection is exact; with
    finer intervals it picks the closest available snapshot.
    """
    times_s = np.asarray(snapshot_times_min, dtype=float) * 60.0
    last_hour = int(np.floor(times_s.max() / 3600.0 + 1e-6))
    target_hours = np.arange(0, last_hour + 1, dtype=int)
    target_s = target_hours.astype(float) * 3600.0
    indices = np.array([int(np.argmin(np.abs(times_s - ts))) for ts in target_s], dtype=int)
    return HourlySelection(
        target_hours=target_hours,
        snapshot_indices=indices,
        actual_times_s=times_s[indices],
    )


def export_hourly_geojsons(result: SimulationResult, out_dir: Path) -> tuple[Path, Path]:
    """Write one GeoJSON per hour + a summary CSV. Returns (geojson_dir, summary_csv)."""
    cfg = result.config
    mesh = result.city.mesh
    threshold_m = cfg.outputs.geojson_threshold_m

    geojson_dir = out_dir / GEOJSON_DIR_NAME
    geojson_dir.mkdir(parents=True, exist_ok=True)

    valid = mesh.active_mask & ~mesh.building_mask
    selection = select_hourly_snapshots(result.snapshot_times_min)

    # Node coordinates reprojected once to WGS84.
    nodes_world = mesh.nodes_xy.astype(np.float64)
    triangles = mesh.triangles.astype(np.int32)
    ll_tx = Transformer.from_crs(cfg.outputs.mesh_crs, "EPSG:4326", always_xy=True)
    lon, lat = ll_tx.transform(nodes_world[:, 0], nodes_world[:, 1])
    nodes_ll = np.column_stack([lon, lat])

    timestamps = result.rainfall.timestamps_by_hour
    intensities = result.rainfall.intensity_by_hour

    summary_rows = []
    log.info("Exporting %d hourly GeoJSONs to %s ...", len(selection.target_hours), geojson_dir)
    for row_i, hour in enumerate(selection.target_hours):
        si = int(selection.snapshot_indices[row_i])
        actual_t_s = float(selection.actual_times_s[row_i])
        depth = result.depth_snapshots_m[si]
        speed = result.speed_snapshots_mps[si]
        timestamp = timestamps.get(int(hour), "")
        intensity = intensities.get(int(hour), 0.0)

        keep = np.where(valid & np.isfinite(depth) & (depth >= threshold_m))[0]
        features = []
        for cell in keep:
            ring = nodes_ll[triangles[cell]].tolist()
            ring.append(ring[0])  # close polygon
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [ring]},
                    "properties": {
                        "cell_id": int(cell),
                        "hour": int(hour),
                        "timestamp_ist": timestamp,
                        "actual_time_s": round(actual_t_s, 1),
                        "intensity_mmhr": intensity,
                        "depth_m": round(float(depth[cell]), 4),
                        "speed_mps": round(float(speed[cell]), 4),
                    },
                }
            )

        geojson_path = geojson_dir / f"hour_{int(hour):02d}_flood_depth.geojson"
        geojson_path.write_text(
            json.dumps({"type": "FeatureCollection", "features": features}, separators=(",", ":")),
            encoding="utf-8",
        )

        max_d = float(depth[valid].max()) if np.any(valid & np.isfinite(depth)) else 0.0
        summary_rows.append(
            {
                "hour": int(hour),
                "timestamp_ist": timestamp,
                "intensity_mmhr": intensity,
                "actual_time_s": round(actual_t_s, 1),
                "wet_cells_ge_threshold": int(len(keep)),
                "max_depth_m": round(max_d, 4),
                "geojson_file": geojson_path.name,
                "geojson_size_kb": round(geojson_path.stat().st_size / 1024, 1),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = geojson_dir / SUMMARY_CSV_NAME
    summary_df.to_csv(summary_csv, index=False)
    log.info("Saved %d GeoJSONs + summary: %s", len(summary_rows), summary_csv)
    return geojson_dir, summary_csv
