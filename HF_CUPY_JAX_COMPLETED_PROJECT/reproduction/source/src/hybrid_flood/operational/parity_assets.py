"""Validation contract for the immutable V1 production topology."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .assets import _sha256
from .errors import ConfigurationError, DataFileMissingError


TOPOLOGY_RELATIVE = Path(
    "data/production_gurugram/production_topology/"
    "d_june5_exact_production_topology_and_mesh.npz"
)
HORTON_RELATIVE = Path("data/production_gurugram/production_topology/horton_bcr_fields.npz")
RECHARGE_RELATIVE = Path(
    "data/production_gurugram/drainage_june5/recharge/"
    "surface_recharge_exchange_table.csv"
)
BOUNDARY_INFLOW_RELATIVE = Path(
    "data/production_gurugram/external_boundary_inflow/"
    "gurugram_external_boundary_inflow_july2025_133mm_12h.csv"
)

EXPECTED_SHAPES = {
    "nodes_world": (499_901, 2),
    "triangles": (981_880, 3),
    "centroids_world": (981_880, 2),
    "bed_elevation_m": (981_880,),
    "active_surface_mask": (981_880,),
    "surface_cell_area_m2": (981_880,),
    "surface_edge_left_cell": (1_490_015,),
    "surface_edge_right_cell": (1_490_015,),
    "surface_boundary_code": (1_490_015,),
    "inlet_cell_index": (61_317,),
    "inlet_node_index": (61_317,),
    "node_storage_area_m2": (118_768,),
    "node_initial_fill_depth_m": (118_768,),
    "link_from_node_index": (139_798,),
    "link_to_node_index": (139_798,),
    "link_flow_capacity_m3ps": (139_798,),
    "outfall_flow_capacity_m3ps": (118_768,),
}


def validate_v1_parity_assets(v1_root: str | Path) -> dict[str, object]:
    """Fail fast unless the exact topology required for numerical parity is present."""

    root = Path(v1_root).resolve()
    paths = {
        "topology": root / TOPOLOGY_RELATIVE,
        "horton": root / HORTON_RELATIVE,
        "recharge": root / RECHARGE_RELATIVE,
        "historical_boundary_inflow": root / BOUNDARY_INFLOW_RELATIVE,
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise DataFileMissingError("Missing V1 parity assets: " + ", ".join(missing))

    with np.load(paths["topology"], allow_pickle=False) as topology:
        absent = sorted(set(EXPECTED_SHAPES).difference(topology.files))
        if absent:
            raise ConfigurationError(f"V1 topology is missing parity arrays: {absent}")
        wrong = {
            name: {"expected": shape, "actual": topology[name].shape}
            for name, shape in EXPECTED_SHAPES.items()
            if topology[name].shape != shape
        }
        if wrong:
            raise ConfigurationError(f"V1 topology shape mismatch: {wrong}")

        right = topology["surface_edge_right_cell"]
        boundary = right < 0
        boundary_codes, boundary_counts = np.unique(
            topology["surface_boundary_code"][boundary], return_counts=True
        )
        boundary_distribution = {
            str(int(code)): int(count)
            for code, count in zip(boundary_codes, boundary_counts, strict=True)
        }
        if boundary_distribution != {"4": 34_390}:
            raise ConfigurationError(
                "V1 exterior boundary contract changed; expected 34,390 outflow-only edges "
                f"(code 4), found {boundary_distribution}."
            )
        active = topology["active_surface_mask"].astype(bool)
        cell_area = topology["surface_cell_area_m2"].astype(np.float64)
        if np.any(cell_area <= 0) or not np.isfinite(cell_area).all():
            raise ConfigurationError("V1 surface cell areas must be positive and finite.")
        for name in ("inlet_capture_capacity_m3ps", "link_flow_capacity_m3ps", "outfall_flow_capacity_m3ps"):
            values = topology[name]
            if np.any(~np.isfinite(values)) or np.any(values < 0):
                raise ConfigurationError(f"V1 {name} contains negative or non-finite values.")
        topology_summary = {
            "triangle_count": int(topology["triangles"].shape[0]),
            "active_triangle_count": int(active.sum()),
            "active_area_km2": float(cell_area[active].sum() / 1e6),
            "edge_count": int(right.size),
            "exterior_boundary_distribution": boundary_distribution,
            "inlet_count": int(topology["inlet_cell_index"].size),
            "node_count": int(topology["node_storage_area_m2"].size),
            "link_count": int(topology["link_from_node_index"].size),
        }

    with np.load(paths["horton"], allow_pickle=False) as horton:
        required = {
            "horton_f0_mmhr", "horton_fc_mmhr", "horton_decay_per_hr",
            "building_fraction", "impervious_fraction", "bcr_open_fraction",
        }
        absent = sorted(required.difference(horton.files))
        if absent:
            raise ConfigurationError(f"V1 Horton/BCR file is missing arrays: {absent}")
        if any(horton[name].shape != (981_880,) for name in required):
            raise ConfigurationError("Every V1 Horton/BCR array must have one value per triangle.")

    with paths["recharge"].open(newline="", encoding="utf-8", errors="replace") as handle:
        recharge_rows = list(csv.DictReader(handle))
    enabled_recharge = sum(
        str(row.get("hydraulic_enabled", "")).strip().lower() in {"1", "true", "yes", "y"}
        for row in recharge_rows
    )

    return {
        "schema_version": 1,
        "ready": True,
        "source_root": str(root),
        "crs": "EPSG:32643",
        "reference_profile": "v1_exact_production_full_swe",
        "topology": topology_summary,
        "recharge": {"input_rows": len(recharge_rows), "enabled_rows": enabled_recharge},
        "assets": {
            name: {"path": str(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size}
            for name, path in paths.items()
        },
    }
