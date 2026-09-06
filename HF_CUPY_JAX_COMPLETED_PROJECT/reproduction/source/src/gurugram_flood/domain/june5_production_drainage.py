"""June 5 production drainage adapters for the CUDA Impl Style runner.

This module consumes the same prepared drainage inputs used by the latest
completed Gurugram_full_data operational run (`D_june_5`):

- d_june_5_actual_storm_plus_sewer_segments.csv
- d_june_5_manhole_segment_edge_inlets.csv
- d_june_5_sewer_stp_solver_sinks.csv
- surface_recharge_exchange_table.csv

The goal is not to reinterpret raw government GIS. These are already parsed
and QA-screened production tables, so this adapter preserves their hydraulic
columns wherever possible.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from gurugram_flood.domain.real_gurugram_domain import (
    DrainageGraph,
    FullCityTopologies,
    NODE_MAX_DEPTH_M_DEFAULT,
    NODE_STORAGE_AREA_M2_DEFAULT,
    RealMesh,
    sample_dem_at_points,
)


def _as_float(value: Any, default: float = np.nan) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _as_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    text = str(value).strip().lower()
    if text == "":
        return default
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return default


def _node_key(x: float, y: float, snap_decimals: int) -> tuple[float, float]:
    return (round(float(x), snap_decimals), round(float(y), snap_decimals))


def _section_from_row(row: dict[str, str]) -> tuple[float, float, float]:
    """Return area, hydraulic radius, diameter from prepared hydraulic columns."""

    diameter = _as_float(row.get("nominal_diameter_m"))
    width = _as_float(row.get("nominal_width_m"))
    depth = _as_float(row.get("nominal_depth_m"))
    if np.isfinite(diameter) and diameter > 0:
        area = np.pi * diameter * diameter / 4.0
        radius = diameter / 4.0
        return float(area), float(radius), float(diameter)
    if np.isfinite(width) and width > 0 and np.isfinite(depth) and depth > 0:
        area = width * depth
        radius = area / max(width + 2.0 * depth, 1.0e-6)
        return float(area), float(radius), np.nan
    capacity = _as_float(row.get("effective_capacity_90pct_m3s"), _as_float(row.get("capacity_full_m3s"), 0.05))
    area = max(capacity / 1.5, np.pi * 0.30 * 0.30 / 4.0)
    radius = max(np.sqrt(area / np.pi) / 2.0, 0.075)
    return float(area), float(radius), np.nan


@dataclass(frozen=True)
class June5GraphBuildSummary:
    input_rows: int
    included_rows: int
    skipped_rows: int
    storm_links: int
    sewer_links: int
    total_length_m: float
    storm_length_m: float
    sewer_length_m: float
    node_count: int


@dataclass(frozen=True)
class June5RechargeTopology:
    source_cell_index: np.ndarray
    overflow_cell_index: np.ndarray
    storage_capacity_m3: np.ndarray
    capture_capacity_m3ps: np.ndarray
    recharge_capacity_m3ps: np.ndarray
    activation_depth_m: np.ndarray
    stop_depth_m: np.ndarray

    @property
    def row_count(self) -> int:
        return int(self.source_cell_index.size)


def load_june5_drainage_graph(
    segment_csv: str | Path,
    dem_path: str | Path | None = None,
    snap_decimals: int = 3,
    endpoint_snap_m: float = 5.0,
    include_network_types: tuple[str, ...] = ("storm", "sewer"),
) -> tuple[DrainageGraph, June5GraphBuildSummary]:
    """Build a dynamic pipe graph from the June 5 production segment table.

    Rows are included only when they are inside the model frame, have finite
    endpoints, positive length, and positive prepared capacity. This mirrors the
    production solver's use of the prepared table without re-parsing raw GIS.
    """

    segment_csv = Path(segment_csv)
    raw_records: list[dict[str, Any]] = []
    endpoints: list[tuple[float, float]] = []

    input_rows = skipped = 0
    storm_len = sewer_len = 0.0

    with segment_csv.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            input_rows += 1
            net = (row.get("network_type") or "").strip().lower()
            if net not in include_network_types:
                skipped += 1
                continue
            x1 = _as_float(row.get("upstream_x")); y1 = _as_float(row.get("upstream_y"))
            x2 = _as_float(row.get("downstream_x")); y2 = _as_float(row.get("downstream_y"))
            lm = _as_float(row.get("length_m"))
            cap = _as_float(row.get("effective_capacity_90pct_m3s"), _as_float(row.get("capacity_full_m3s"), 0.0))
            if not all(np.isfinite(v) for v in (x1, y1, x2, y2, lm, cap)) or lm <= 0 or cap <= 0:
                skipped += 1
                continue
            rec = {
                "row": row,
                "net": net,
                "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                "length_m": float(lm), "capacity_m3ps": float(cap),
                "up_inv": _as_float(row.get("upstream_invert_m")),
                "dn_inv": _as_float(row.get("downstream_invert_m")),
            }
            raw_records.append(rec)
            endpoints.append((float(x1), float(y1)))
            endpoints.append((float(x2), float(y2)))

    if not raw_records:
        raise ValueError(f"No valid June 5 drainage rows found in {segment_csv}")

    endpoint_xy = np.asarray(endpoints, dtype=np.float64)
    parent = np.arange(endpoint_xy.shape[0], dtype=np.int64)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return int(i)

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    if endpoint_snap_m > 0:
        tree = cKDTree(endpoint_xy)
        for i, j in tree.query_pairs(r=float(endpoint_snap_m)):
            union(int(i), int(j))

    roots = np.array([find(i) for i in range(endpoint_xy.shape[0])], dtype=np.int64)
    unique_roots, endpoint_node_index = np.unique(roots, return_inverse=True)
    node_xy = np.array([endpoint_xy[roots == r].mean(axis=0) for r in unique_roots], dtype=np.float64)
    node_x = node_xy[:, 0].tolist()
    node_y = node_xy[:, 1].tolist()
    node_invert_values: list[list[float]] = [[] for _ in range(len(node_x))]

    link_from: list[int] = []
    link_to: list[int] = []
    length_m: list[float] = []
    area_m2: list[float] = []
    radius_m: list[float] = []
    manning_n: list[float] = []
    capacity_m3ps: list[float] = []
    diameter_m: list[float] = []
    network_type: list[str] = []
    segment_id: list[str] = []

    for rec_i, rec in enumerate(raw_records):
        n1 = int(endpoint_node_index[2 * rec_i])
        n2 = int(endpoint_node_index[2 * rec_i + 1])
        if np.isfinite(rec["up_inv"]):
            node_invert_values[n1].append(float(rec["up_inv"]))
        if np.isfinite(rec["dn_inv"]):
            node_invert_values[n2].append(float(rec["dn_inv"]))
        if n1 == n2:
            skipped += 1
            continue
        row = rec["row"]
        a, r, d = _section_from_row(row)
        n = _as_float(row.get("roughness_n"), 0.015)
        link_from.append(n1)
        link_to.append(n2)
        length_m.append(float(rec["length_m"]))
        area_m2.append(float(max(a, 1.0e-6)))
        radius_m.append(float(max(r, 1.0e-6)))
        manning_n.append(float(n if np.isfinite(n) and n > 0 else 0.015))
        capacity_m3ps.append(float(rec["capacity_m3ps"]))
        diameter_m.append(float(d))
        network_type.append(str(rec["net"]))
        segment_id.append(str(row.get("segment_id") or ""))
        if rec["net"] == "storm":
            storm_len += float(rec["length_m"])
        else:
            sewer_len += float(rec["length_m"])

    node_x_arr = np.asarray(node_x, dtype=np.float64)
    node_y_arr = np.asarray(node_y, dtype=np.float64)
    invert = np.full(len(node_x_arr), np.nan, dtype=np.float64)
    for i, values in enumerate(node_invert_values):
        if values:
            invert[i] = float(np.nanmean(values))
    if np.any(~np.isfinite(invert)):
        if dem_path is None:
            fill = np.nanmedian(invert[np.isfinite(invert)]) if np.any(np.isfinite(invert)) else 225.0
            invert[~np.isfinite(invert)] = fill
        else:
            dem = sample_dem_at_points(dem_path, node_x_arr[~np.isfinite(invert)], node_y_arr[~np.isfinite(invert)])
            invert[~np.isfinite(invert)] = dem - 1.5

    graph = DrainageGraph(
        node_x_m=node_x_arr,
        node_y_m=node_y_arr,
        node_is_outfall=np.zeros(len(node_x_arr), dtype=bool),
        node_outfall_trunk_id=[""] * len(node_x_arr),
        link_from=np.asarray(link_from, dtype=np.int64),
        link_to=np.asarray(link_to, dtype=np.int64),
        link_length_m=np.asarray(length_m, dtype=np.float64),
        link_area_m2=np.asarray(area_m2, dtype=np.float64),
        link_hydraulic_radius_m=np.asarray(radius_m, dtype=np.float64),
        link_manning_n=np.asarray(manning_n, dtype=np.float64),
        link_capacity_m3ps=np.asarray(capacity_m3ps, dtype=np.float64),
        link_diameter_m=np.asarray(diameter_m, dtype=np.float64),
    )
    graph.node_invert_elevation_m = invert  # type: ignore[attr-defined]
    graph.link_network_type = np.asarray(network_type, dtype=object)  # type: ignore[attr-defined]
    graph.link_segment_id = np.asarray(segment_id, dtype=object)  # type: ignore[attr-defined]

    summary = June5GraphBuildSummary(
        input_rows=input_rows,
        included_rows=len(link_from),
        skipped_rows=skipped,
        storm_links=int(np.sum(graph.link_network_type == "storm")),  # type: ignore[attr-defined]
        sewer_links=int(np.sum(graph.link_network_type == "sewer")),  # type: ignore[attr-defined]
        total_length_m=float(np.sum(graph.link_length_m)),
        storm_length_m=float(storm_len),
        sewer_length_m=float(sewer_len),
        node_count=graph.node_count,
    )
    return graph, summary


def map_points_to_active_cells(mesh: RealMesh, x_m: np.ndarray, y_m: np.ndarray) -> np.ndarray:
    active_idx = np.where(mesh.active_mask)[0]
    tree = cKDTree(mesh.centroids_xy[active_idx])
    _, nearest = tree.query(np.column_stack([x_m, y_m]))
    return active_idx[nearest].astype(np.int32)


def map_nodes_to_mesh_triangles(graph: DrainageGraph, mesh: RealMesh) -> np.ndarray:
    return map_points_to_active_cells(mesh, graph.node_x_m, graph.node_y_m)


def build_june5_inlets(
    graph: DrainageGraph,
    mesh: RealMesh,
    node_cell_index: np.ndarray,
    manhole_csv: str | Path,
    storm_surface_inlet_fraction: float = 0.05,
    default_manhole_capacity_m3ps: float = 0.02,
    max_manhole_snap_m: float = 14.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Create inlet rows: storm network nodes plus mapped sewer manhole points."""

    link_net = np.asarray(graph.link_network_type)  # type: ignore[attr-defined]
    storm_links = np.where(link_net == "storm")[0]
    sewer_links = np.where(link_net == "sewer")[0]

    storm_node_cap = np.zeros(graph.node_count, dtype=np.float64)
    storm_cap = graph.link_capacity_m3ps[storm_links] * storm_surface_inlet_fraction
    np.add.at(storm_node_cap, graph.link_from[storm_links], storm_cap)
    np.add.at(storm_node_cap, graph.link_to[storm_links], storm_cap)
    storm_nodes = np.where(storm_node_cap > 0)[0]

    inlet_nodes: list[int] = storm_nodes.astype(int).tolist()
    inlet_cells: list[int] = node_cell_index[storm_nodes].astype(int).tolist()
    inlet_caps: list[float] = np.clip(storm_node_cap[storm_nodes], 0.0, 5.0).astype(float).tolist()

    sewer_nodes = np.unique(np.concatenate([graph.link_from[sewer_links], graph.link_to[sewer_links]]))
    sewer_xy = np.column_stack([graph.node_x_m[sewer_nodes], graph.node_y_m[sewer_nodes]])
    sewer_tree = cKDTree(sewer_xy) if len(sewer_nodes) else None

    mh_input = mh_mapped = mh_too_far = 0
    with Path(manhole_csv).open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if rows:
        mx = np.array([_as_float(r.get("projected_x_utm43n_m"), _as_float(r.get("x_utm43n_m"))) for r in rows], dtype=float)
        my = np.array([_as_float(r.get("projected_y_utm43n_m"), _as_float(r.get("y_utm43n_m"))) for r in rows], dtype=float)
        valid = np.isfinite(mx) & np.isfinite(my)
        cell_for_point = np.full(len(rows), -1, dtype=np.int32)
        if np.any(valid):
            cell_for_point[valid] = map_points_to_active_cells(mesh, mx[valid], my[valid])
        for i, row in enumerate(rows):
            mh_input += 1
            if not valid[i] or sewer_tree is None:
                mh_too_far += 1
                continue
            dist, j = sewer_tree.query([mx[i], my[i]])
            if dist > max_manhole_snap_m:
                mh_too_far += 1
                continue
            cap = _as_float(row.get("inlet_capacity_m3ps"), default_manhole_capacity_m3ps)
            inlet_nodes.append(int(sewer_nodes[j]))
            inlet_cells.append(int(cell_for_point[i]))
            inlet_caps.append(float(cap if np.isfinite(cap) and cap > 0 else default_manhole_capacity_m3ps))
            mh_mapped += 1

    info = {
        "storm_inlet_count": float(len(storm_nodes)),
        "storm_inlet_capacity_m3ps_sum": float(np.sum(storm_node_cap[storm_nodes])),
        "manhole_input_count": float(mh_input),
        "manhole_mapped_count": float(mh_mapped),
        "manhole_skipped_too_far_or_invalid": float(mh_too_far),
        "manhole_capacity_m3ps_sum": float(np.sum(inlet_caps[len(storm_nodes):])),
    }
    return (
        np.asarray(inlet_cells, dtype=np.int32),
        np.asarray(inlet_nodes, dtype=np.int32),
        np.asarray(inlet_caps, dtype=np.float64),
        info,
    )


def attach_june5_stp_sinks(
    graph: DrainageGraph,
    stp_csv: str | Path,
    max_snap_m: float = 5000.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Map active June 5 STP sink rows to nearest sewer-family graph nodes."""

    outfall_area = np.zeros(graph.node_count, dtype=np.float64)
    outfall_coef = np.zeros(graph.node_count, dtype=np.float64)
    outfall_tailwater = np.zeros(graph.node_count, dtype=np.float64)
    outfall_capacity = np.zeros(graph.node_count, dtype=np.float64)

    link_net = np.asarray(graph.link_network_type)  # type: ignore[attr-defined]
    sewer_links = np.where(link_net == "sewer")[0]
    sewer_nodes = np.unique(np.concatenate([graph.link_from[sewer_links], graph.link_to[sewer_links]]))
    tree = cKDTree(np.column_stack([graph.node_x_m[sewer_nodes], graph.node_y_m[sewer_nodes]])) if len(sewer_nodes) else None

    input_count = active_count = mapped_count = skipped_count = 0
    capacity_sum = 0.0
    storage_sum = 0.0
    with Path(stp_csv).open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            input_count += 1
            if not _as_bool(row.get("hydraulically_active"), False):
                continue
            active_count += 1
            x = _as_float(row.get("x_utm43n_m")); y = _as_float(row.get("y_utm43n_m"))
            cap = _as_float(row.get("capacity_m3ps"), 0.0)
            storage = _as_float(row.get("storage_m3"), 0.0)
            if tree is None or not np.isfinite(x) or not np.isfinite(y) or cap <= 0:
                skipped_count += 1
                continue
            dist, j = tree.query([x, y])
            if dist > max_snap_m:
                skipped_count += 1
                continue
            node = int(sewer_nodes[j])
            outfall_area[node] = max(outfall_area[node], 100.0)
            outfall_coef[node] = 1.0
            outfall_tailwater[node] = float(graph.node_invert_elevation_m[node] - 10.0)  # type: ignore[attr-defined]
            outfall_capacity[node] += float(cap)
            graph.node_is_outfall[node] = True
            graph.node_outfall_trunk_id[node] = "JUNE5_STP"
            capacity_sum += float(cap)
            storage_sum += max(float(storage), 0.0)
            mapped_count += 1

    info = {
        "stp_input_count": float(input_count),
        "stp_active_count": float(active_count),
        "stp_mapped_count": float(mapped_count),
        "stp_skipped_count": float(skipped_count),
        "stp_capacity_m3ps_sum": float(capacity_sum),
        "stp_storage_m3_sum": float(storage_sum),
    }
    return outfall_area, outfall_coef, outfall_tailwater, outfall_capacity, info


def load_june5_recharge_topology(recharge_csv: str | Path, mesh: RealMesh) -> tuple[June5RechargeTopology, dict[str, float]]:
    rows: list[dict[str, str]] = []
    with Path(recharge_csv).open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            if not _as_bool(row.get("hydraulic_enabled"), False):
                continue
            rows.append(row)
    if not rows:
        empty = np.array([], dtype=np.int32)
        z = np.array([], dtype=np.float64)
        return June5RechargeTopology(empty, empty, z, z, z, z, z), {"row_count": 0.0}
    sx = np.array([_as_float(r.get("x_m")) for r in rows], dtype=float)
    sy = np.array([_as_float(r.get("y_m")) for r in rows], dtype=float)
    source_cell = np.array([int(_as_float(r.get("source_cell_index"), -1)) for r in rows], dtype=np.int64)
    invalid = (source_cell < 0) | (source_cell >= mesh.triangle_count) | (~mesh.active_mask[np.clip(source_cell, 0, mesh.triangle_count - 1)])
    if np.any(invalid):
        source_cell[invalid] = map_points_to_active_cells(mesh, sx[invalid], sy[invalid])
    overflow_cell = np.array([int(_as_float(r.get("overflow_cell_index"), -1)) for r in rows], dtype=np.int64)
    bad_overflow = (overflow_cell < 0) | (overflow_cell >= mesh.triangle_count)
    overflow_cell[bad_overflow] = source_cell[bad_overflow]
    topo = June5RechargeTopology(
        source_cell_index=source_cell.astype(np.int32),
        overflow_cell_index=overflow_cell.astype(np.int32),
        storage_capacity_m3=np.array([_as_float(r.get("storage_capacity_m3"), 0.0) for r in rows], dtype=np.float64),
        capture_capacity_m3ps=np.array([_as_float(r.get("capture_capacity_m3ps"), 0.0) for r in rows], dtype=np.float64),
        recharge_capacity_m3ps=np.array([_as_float(r.get("recharge_capacity_m3ps"), 0.0) for r in rows], dtype=np.float64),
        activation_depth_m=np.array([_as_float(r.get("activation_depth_m"), 0.05) for r in rows], dtype=np.float64),
        stop_depth_m=np.array([_as_float(r.get("stop_depth_m"), 0.02) for r in rows], dtype=np.float64),
    )
    info = {
        "row_count": float(topo.row_count),
        "storage_capacity_m3_sum": float(np.sum(topo.storage_capacity_m3)),
        "capture_capacity_m3ps_sum": float(np.sum(topo.capture_capacity_m3ps)),
        "recharge_capacity_m3ps_sum": float(np.sum(topo.recharge_capacity_m3ps)),
    }
    return topo, info


def build_june5_full_topologies(
    mesh: RealMesh,
    manning_n: np.ndarray,
    graph: DrainageGraph,
    node_cell_index: np.ndarray,
    inlet_cell_index: np.ndarray,
    inlet_node_index: np.ndarray,
    inlet_capacity_m3ps: np.ndarray,
    outfall_area_m2: np.ndarray,
    outfall_coefficient: np.ndarray,
    outfall_tailwater_head_m: np.ndarray,
    outfall_flow_capacity_m3ps: np.ndarray | None = None,
    boundary_mode: str = "outflow",
    drainage_capacity_fraction: float = 1.0,
    sewer_dry_weather_flow_fraction: float = 0.4,
    initial_drainage_storage_fraction: float = 0.0,
) -> FullCityTopologies:
    """Assemble CUDA topologies using June 5 production drainage semantics."""

    from gurugram_flood.kernels.triangular_backend import (
        TriangularNodeCouplingTopology,
        TriangularPipeNetworkState,
        TriangularPipeNetworkTopology,
        TriangularSurfaceState,
        build_triangular_surface_topology,
    )

    surface_topology = build_triangular_surface_topology(
        nodes_xy=mesh.nodes_xy,
        triangles=mesh.triangles,
        bed_elevation_m=mesh.terrain_elevation_m,
        manning_n=manning_n,
        active_mask=mesh.active_mask,
        boundary_mode=boundary_mode,
    )

    link_net = np.asarray(graph.link_network_type)  # type: ignore[attr-defined]
    derated_capacity = graph.link_capacity_m3ps.astype(np.float64) * drainage_capacity_fraction
    derated_capacity[link_net == "sewer"] *= max(0.0, 1.0 - sewer_dry_weather_flow_fraction)

    coupling_topology = TriangularNodeCouplingTopology(
        inlet_cell_index=inlet_cell_index.astype(np.int32),
        inlet_node_index=inlet_node_index.astype(np.int32),
        node_surface_cell_index=node_cell_index.astype(np.int32),
        node_invert_elevation_m=graph.node_invert_elevation_m.astype(np.float64),  # type: ignore[attr-defined]
        node_storage_area_m2=np.full(graph.node_count, NODE_STORAGE_AREA_M2_DEFAULT, dtype=np.float64),
        node_max_depth_m=np.full(graph.node_count, NODE_MAX_DEPTH_M_DEFAULT, dtype=np.float64),
        inlet_capture_capacity_m3ps=inlet_capacity_m3ps.astype(np.float64),
        inlet_surcharge_capacity_m3ps=inlet_capacity_m3ps.astype(np.float64),
        node_x_m=graph.node_x_m.astype(np.float64),
        node_y_m=graph.node_y_m.astype(np.float64),
    )
    pipe_topology = TriangularPipeNetworkTopology(
        link_from_node_index=graph.link_from.astype(np.int32),
        link_to_node_index=graph.link_to.astype(np.int32),
        link_length_m=graph.link_length_m.astype(np.float64),
        link_area_m2=graph.link_area_m2.astype(np.float64),
        link_hydraulic_radius_m=graph.link_hydraulic_radius_m.astype(np.float64),
        link_manning_n=graph.link_manning_n.astype(np.float64),
        link_flow_capacity_m3ps=derated_capacity.astype(np.float64),
        outfall_area_m2=outfall_area_m2.astype(np.float64),
        outfall_coefficient=outfall_coefficient.astype(np.float64),
        outfall_tailwater_head_m=outfall_tailwater_head_m.astype(np.float64),
        outfall_flow_capacity_m3ps=None if outfall_flow_capacity_m3ps is None else outfall_flow_capacity_m3ps.astype(np.float64),
    )
    surface_state = TriangularSurfaceState(
        h_m=np.zeros(mesh.triangle_count, dtype=np.float64),
        hu_m2ps=np.zeros(mesh.triangle_count, dtype=np.float64),
        hv_m2ps=np.zeros(mesh.triangle_count, dtype=np.float64),
    )
    initial_node_volume_m3 = NODE_STORAGE_AREA_M2_DEFAULT * NODE_MAX_DEPTH_M_DEFAULT * initial_drainage_storage_fraction
    pipe_state = TriangularPipeNetworkState(
        node_volume_m3=np.full(graph.node_count, initial_node_volume_m3, dtype=np.float64),
        link_flow_m3ps=np.zeros(graph.link_count, dtype=np.float64),
    )
    return FullCityTopologies(mesh, graph, surface_topology, coupling_topology, pipe_topology, surface_state, pipe_state)


def recharge_to_backend(topo: June5RechargeTopology, backend: Any) -> June5RechargeTopology:
    xp = backend.xp if hasattr(backend, "xp") else backend
    return June5RechargeTopology(
        source_cell_index=xp.asarray(topo.source_cell_index, dtype=getattr(xp, "int32", int)),
        overflow_cell_index=xp.asarray(topo.overflow_cell_index, dtype=getattr(xp, "int32", int)),
        storage_capacity_m3=xp.asarray(topo.storage_capacity_m3, dtype=getattr(xp, "float32")),
        capture_capacity_m3ps=xp.asarray(topo.capture_capacity_m3ps, dtype=getattr(xp, "float32")),
        recharge_capacity_m3ps=xp.asarray(topo.recharge_capacity_m3ps, dtype=getattr(xp, "float32")),
        activation_depth_m=xp.asarray(topo.activation_depth_m, dtype=getattr(xp, "float32")),
        stop_depth_m=xp.asarray(topo.stop_depth_m, dtype=getattr(xp, "float32")),
    )


def apply_june5_surface_recharge(
    h_m: Any,
    recharge_topology: June5RechargeTopology,
    recharge_storage_m3: Any,
    cell_area_m2: Any,
    timestep_s: float,
    xp: Any,
) -> tuple[Any, Any, Any, Any]:
    """Apply direct_capture_storage_recharge for 447 prepared pond/RWH rows."""

    if recharge_topology.row_count == 0:
        zero = xp.sum(h_m * 0.0)
        return h_m, recharge_storage_m3, zero, zero
    cell = recharge_topology.source_cell_index
    area = xp.maximum(cell_area_m2[cell], 1.0e-6)
    depth = h_m[cell]
    removable_depth = xp.maximum(depth - recharge_topology.stop_depth_m, 0.0)
    available_surface_m3 = removable_depth * area
    capture_request = recharge_topology.capture_capacity_m3ps * timestep_s
    remaining_storage = xp.maximum(recharge_topology.storage_capacity_m3 - recharge_storage_m3, 0.0)
    capture_m3 = xp.minimum(xp.minimum(available_surface_m3, capture_request), remaining_storage)
    # Repeated source cells are possible, but the table is small and duplicate capture is intentionally
    # conservative; production table is already pre-snapped and deduped by structure_group_index.
    new_h = h_m.copy()
    new_h[cell] = xp.maximum(new_h[cell] - capture_m3 / area, 0.0)
    storage = recharge_storage_m3 + capture_m3
    recharge_loss = xp.minimum(storage, recharge_topology.recharge_capacity_m3ps * timestep_s)
    storage = storage - recharge_loss
    return new_h, storage, xp.sum(capture_m3), xp.sum(recharge_loss)
