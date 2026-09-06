"""Prepare immutable V1 production assets for the structured V2 solver."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy.spatial import cKDTree

from .errors import DataFileMissingError


REQUIRED = {
    "topology": "data/production_topology/d_june5_exact_production_topology_and_mesh.npz",
    "horton": "data/production_topology/horton_bcr_fields.npz",
    "dem": "data/terrain/hydraulic_dtm_fabdem_5m.tif",
    "recharge": "data/drainage_june5/recharge/surface_recharge_exchange_table.csv",
    "historical_rainfall": "examples/july2025_133mm_12h.csv",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_v1_assets(v1_root: str | Path) -> dict[str, object]:
    root = Path(v1_root).resolve()
    assets = {name: root / relative for name, relative in REQUIRED.items()}
    missing = [str(path) for path in assets.values() if not path.is_file()]
    return {
        "root": str(root), "ready": not missing, "missing": missing,
        "assets": {name: str(path) for name, path in assets.items()},
    }


def prepare_v1_assets(v1_root: str | Path, output_dir: str | Path) -> Path:
    status = check_v1_assets(v1_root)
    if not status["ready"]:
        raise DataFileMissingError("Missing V1 assets: " + ", ".join(status["missing"]))
    root, destination = Path(v1_root).resolve(), Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    paths = {name: root / relative for name, relative in REQUIRED.items()}
    with np.load(paths["topology"]) as topology, np.load(paths["horton"]) as horton:
        centroids = topology["centroids_world"].astype(np.float64)
        fields = {
            "manning_n": topology["manning_n"],
            "valid_domain_mask": topology["valid_domain_mask"].astype(bool),
            "active_mask": topology["active_surface_mask"].astype(bool),
            "building_mask": topology["building_mask"].astype(bool),
            "road_mask": topology["road_mask"].astype(bool),
            "horton_f0_mmhr": horton["horton_f0_mmhr"],
            "horton_fc_mmhr": horton["horton_fc_mmhr"],
            "horton_decay_per_hr": horton["horton_decay_per_hr"],
            "bcr_open_fraction": horton["bcr_open_fraction"],
        }
        np.savez_compressed(destination / "v1_triangle_fields.npz", centroids=centroids, **fields)
        inlet_cells = topology["inlet_cell_index"]
        np.savez_compressed(
            destination / "v1_capacity_assets.npz",
            inlet_x_m=centroids[inlet_cells, 0], inlet_y_m=centroids[inlet_cells, 1],
            inlet_capacity_m3ps=topology["inlet_capture_capacity_m3ps"],
            outfall_x_m=topology["node_x_m"], outfall_y_m=topology["node_y_m"],
            outfall_capacity_m3ps=topology["outfall_flow_capacity_m3ps"],
        )
    # Produce a V1 roughness raster aligned to the authoritative hydraulic DTM.
    with rasterio.open(paths["dem"]) as dem:
        x = dem.transform.c + (np.arange(dem.width) + 0.5) * dem.transform.a
        y = dem.transform.f + (np.arange(dem.height) + 0.5) * dem.transform.e
        xx, yy = np.meshgrid(x, y)
        _, nearest = cKDTree(centroids).query(np.column_stack((xx.ravel(), yy.ravel())))
        roughness = fields["manning_n"][nearest].reshape(dem.height, dem.width).astype(np.float32)
        profile = dem.profile.copy()
        profile.update(dtype="float32", count=1, nodata=np.nan, compress="deflate")
        with rasterio.open(destination / "v1_roughness.tif", "w", **profile) as target:
            target.write(roughness, 1)
    manifest = {
        "schema_version": 1, "source_root": str(root), "crs": "EPSG:32643",
        "assets": {name: {"path": str(path), "sha256": _sha256(path)} for name, path in paths.items()},
        "generated": ["v1_triangle_fields.npz", "v1_capacity_assets.npz", "v1_roughness.tif"],
    }
    manifest_path = destination / "asset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def rasterize_operational_fields(grid, prepared_dir: str | Path) -> dict[str, np.ndarray]:
    prepared = Path(prepared_dir)
    with np.load(prepared / "v1_triangle_fields.npz") as source:
        xx, yy = np.meshgrid(grid.x, grid.y)
        _, nearest = cKDTree(source["centroids"]).query(np.column_stack((xx.ravel(), yy.ravel())))
        result = {name: source[name][nearest].reshape(grid.bed.shape) for name in source.files if name != "centroids"}
    valid_surface = (
        grid.domain_mask
        & np.asarray(result["valid_domain_mask"], dtype=bool)
        & np.asarray(result["active_mask"], dtype=bool)
    )
    if not valid_surface.any():
        raise ValueError("V1 valid/active masks do not overlap the V2 grid.")

    # Capacity assets must act on hydraulic cells.  Direct searchsorted mapping
    # can put an inlet in an inactive hole (or just outside a jagged boundary),
    # so snap every enabled asset to the nearest valid cell deterministically.
    valid_rows, valid_columns = np.nonzero(valid_surface)
    valid_xy = np.column_stack((grid.x[valid_columns], grid.y[valid_rows]))
    valid_tree = cKDTree(valid_xy)
    drainage = np.zeros(grid.bed.shape, dtype=np.float32)
    with np.load(prepared / "v1_capacity_assets.npz") as assets:
        for prefix in ("inlet", "outfall"):
            capacity = assets[f"{prefix}_capacity_m3ps"]
            valid = np.isfinite(capacity) & (capacity > 0)
            coordinates = np.column_stack((
                assets[f"{prefix}_x_m"][valid],
                assets[f"{prefix}_y_m"][valid],
            ))
            _, nearest = valid_tree.query(coordinates)
            rows = valid_rows[nearest]
            columns = valid_columns[nearest]
            np.add.at(drainage, (rows, columns), capacity[valid] / (grid.resolution_m**2))
    result["drainage_capacity_m_s"] = drainage
    return result
