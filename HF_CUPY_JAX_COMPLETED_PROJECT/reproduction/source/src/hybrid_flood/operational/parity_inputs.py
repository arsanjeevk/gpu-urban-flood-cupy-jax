"""Host-side loading of exact V1 arrays for the JAX parity engine."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from hybrid_flood.jax_solver.v1_parity import (
    CouplingTopology,
    ParityState,
    PipeTopology,
    RechargeTopology,
    SurfaceTopology,
)

from .parity_assets import (
    BOUNDARY_INFLOW_RELATIVE,
    HORTON_RELATIVE,
    RECHARGE_RELATIVE,
    TOPOLOGY_RELATIVE,
    validate_v1_parity_assets,
)
from .rainfall import load_rainfall_csv


MM_HR_TO_M_S = 1.0e-3 / 3600.0


class ParityForcing(NamedTuple):
    rainfall_start_s: jnp.ndarray
    rainfall_end_s: jnp.ndarray
    rainfall_rate_mps: jnp.ndarray
    rainfall_multiplier: jnp.ndarray
    horton_f0_mps: jnp.ndarray
    horton_fc_mps: jnp.ndarray
    horton_k_per_s: jnp.ndarray
    external_times_s: jnp.ndarray
    external_receiver_cell: jnp.ndarray
    external_q_m3ps: jnp.ndarray


@dataclass(frozen=True)
class LoadedParityInputs:
    surface: SurfaceTopology
    coupling: CouplingTopology
    pipe: PipeTopology
    recharge: RechargeTopology
    initial_state: ParityState
    forcing: ParityForcing
    host: dict[str, np.ndarray]
    report: dict[str, object]


def _enabled(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _load_recharge(path: Path, active: np.ndarray, centroids: np.ndarray) -> tuple[RechargeTopology, dict]:
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        rows = [row for row in csv.DictReader(handle) if _enabled(row.get("hydraulic_enabled"))]
    active_index = np.flatnonzero(active)
    tree = cKDTree(centroids[active_index])
    source: list[int] = []
    storage: list[float] = []
    capture: list[float] = []
    loss: list[float] = []
    stop: list[float] = []
    remapped = 0
    for row in rows:
        cell = int(float(row.get("source_cell_index") or -1))
        if cell < 0 or cell >= len(active) or not active[cell]:
            _, nearest = tree.query([float(row["x_m"]), float(row["y_m"])])
            cell = int(active_index[nearest])
            remapped += 1
        source.append(cell)
        storage.append(float(row.get("storage_capacity_m3") or 0.0))
        capture.append(float(row.get("capture_capacity_m3ps") or 0.0))
        loss.append(float(row.get("recharge_capacity_m3ps") or 0.0))
        stop.append(float(row.get("stop_depth_m") or 0.02))
    return RechargeTopology(
        jnp.asarray(source, dtype=jnp.int32), jnp.asarray(storage, dtype=jnp.float32),
        jnp.asarray(capture, dtype=jnp.float32), jnp.asarray(loss, dtype=jnp.float32),
        jnp.asarray(stop, dtype=jnp.float32),
    ), {"row_count": len(rows), "remapped_source_count": remapped}


def _external_forcing(path: Path, active: np.ndarray, centroids: np.ndarray):
    frame = pd.read_csv(path)
    active_index = np.flatnonzero(active)
    tree = cKDTree(centroids[active_index])
    inflow_ids = frame["inflow_id"].astype(str).to_numpy()
    x = pd.to_numeric(frame["x_utm43n_m"], errors="raise").to_numpy(float)
    y = pd.to_numeric(frame["y_utm43n_m"], errors="raise").to_numpy(float)
    distance, nearest = tree.query(np.column_stack((x, y)))
    receiver_by_row = active_index[nearest]
    time = pd.to_numeric(frame["time_s"], errors="raise").to_numpy(float)
    q = pd.to_numeric(frame["q_m3ps"], errors="raise").to_numpy(float)
    times = np.asarray(sorted(set(time)), dtype=np.float64)
    time_index = {float(value): index for index, value in enumerate(times)}
    receiver_key: dict[tuple[str, int], int] = {}
    receivers: list[int] = []
    columns: list[np.ndarray] = []
    for inflow_id, cell, t, flow in zip(inflow_ids, receiver_by_row, time, q, strict=True):
        key = (str(inflow_id), int(cell))
        if key not in receiver_key:
            receiver_key[key] = len(receivers)
            receivers.append(int(cell))
            columns.append(np.zeros(len(times), dtype=np.float64))
        columns[receiver_key[key]][time_index[float(t)]] += float(flow)
    q_matrix = np.stack(columns, axis=1).astype(np.float32)
    return (
        times.astype(np.float32), np.asarray(receivers, dtype=np.int32), q_matrix,
        {
            "receiver_count": len(receivers),
            "maximum_snap_m": float(np.max(distance)),
            "snap_warning_count": int(np.sum(distance > 500.0)),
            "peak_total_m3ps": float(np.max(q_matrix.sum(axis=1))),
            "hydrograph_volume_m3": float(np.trapezoid(q_matrix.sum(axis=1), times)),
        },
    )


def load_parity_inputs(
    *, rainfall_csv: str | Path, v1_root: str | Path,
) -> LoadedParityInputs:
    root = Path(v1_root).resolve()
    asset_report = validate_v1_parity_assets(root)
    with np.load(root / TOPOLOGY_RELATIVE, allow_pickle=False) as d:
        host = {name: d[name] for name in d.files}
    active = host["active_surface_mask"].astype(bool)
    surface = SurfaceTopology(
        jnp.asarray(host["bed_elevation_m"], dtype=jnp.float32),
        jnp.asarray(host["manning_n"], dtype=jnp.float32),
        jnp.asarray(host["surface_cell_area_m2"], dtype=jnp.float32),
        jnp.asarray(host["surface_characteristic_length_m"], dtype=jnp.float32),
        jnp.asarray(host["surface_edge_left_cell"], dtype=jnp.int32),
        jnp.asarray(host["surface_edge_right_cell"], dtype=jnp.int32),
        jnp.asarray(host["surface_edge_normal_x"], dtype=jnp.float32),
        jnp.asarray(host["surface_edge_normal_y"], dtype=jnp.float32),
        jnp.asarray(host["surface_edge_length_m"], dtype=jnp.float32),
        jnp.asarray(host["surface_boundary_code"], dtype=jnp.int32),
        jnp.asarray(active),
    )
    coupling = CouplingTopology(
        jnp.asarray(host["inlet_cell_index"], dtype=jnp.int32),
        jnp.asarray(host["inlet_node_index"], dtype=jnp.int32),
        jnp.asarray(host["node_surface_cell_index"], dtype=jnp.int32),
        jnp.asarray(host["node_invert_elevation_m"], dtype=jnp.float32),
        jnp.asarray(host["node_storage_area_m2"], dtype=jnp.float32),
        jnp.asarray(host["node_max_depth_m"], dtype=jnp.float32),
        jnp.asarray(host["inlet_capture_capacity_m3ps"], dtype=jnp.float32),
        jnp.asarray(host["inlet_surcharge_capacity_m3ps"], dtype=jnp.float32),
        jnp.asarray(host["surcharge_cell_index"], dtype=jnp.int32),
        jnp.asarray(host["surcharge_node_index"], dtype=jnp.int32),
        jnp.asarray(host["surcharge_return_weight"], dtype=jnp.float32),
        jnp.asarray(bool(host["surcharge_return_weight_is_normalized"][0])),
    )
    pipe = PipeTopology(
        jnp.asarray(host["link_from_node_index"], dtype=jnp.int32),
        jnp.asarray(host["link_to_node_index"], dtype=jnp.int32),
        jnp.asarray(host["link_length_m"], dtype=jnp.float32),
        jnp.asarray(host["link_area_m2"], dtype=jnp.float32),
        jnp.asarray(host["link_hydraulic_radius_m"], dtype=jnp.float32),
        jnp.asarray(host["link_manning_n"], dtype=jnp.float32),
        jnp.asarray(host["link_flow_capacity_m3ps"], dtype=jnp.float32),
        jnp.asarray(host["link_invert_elevation_m"], dtype=jnp.float32),
        jnp.asarray(host["link_diameter_m"], dtype=jnp.float32),
        jnp.asarray(host["link_minor_loss_coefficient"], dtype=jnp.float32),
        jnp.asarray(host["link_geometry_code"], dtype=jnp.int32),
        jnp.asarray(host["link_rect_width_m"], dtype=jnp.float32),
        jnp.asarray(host["link_rect_height_m"], dtype=jnp.float32),
        jnp.asarray(host["outfall_area_m2"], dtype=jnp.float32),
        jnp.asarray(host["outfall_coefficient"], dtype=jnp.float32),
        jnp.asarray(host["outfall_tailwater_head_m"], dtype=jnp.float32),
        jnp.asarray(host["outfall_flow_capacity_m3ps"], dtype=jnp.float32),
    )
    recharge, recharge_report = _load_recharge(
        root / RECHARGE_RELATIVE, active, host["centroids_world"]
    )
    external_times, receivers, external_q, external_report = _external_forcing(
        root / BOUNDARY_INFLOW_RELATIVE, active, host["centroids_world"]
    )
    event = load_rainfall_csv(rainfall_csv)
    with np.load(root / HORTON_RELATIVE, allow_pickle=False) as horton:
        f0 = horton["horton_f0_mmhr"].astype(np.float32) * MM_HR_TO_M_S
        fc = horton["horton_fc_mmhr"].astype(np.float32) * MM_HR_TO_M_S
        decay = horton["horton_decay_per_hr"].astype(np.float32) / 3600.0
    rainfall_multiplier = host.get("rainfall_multiplier", np.ones(len(active), np.float32))
    forcing = ParityForcing(
        jnp.asarray(event.frame.start_min.to_numpy(np.float32) * 60.0),
        jnp.asarray(event.frame.end_min.to_numpy(np.float32) * 60.0),
        jnp.asarray(event.frame.intensity_mmhr.to_numpy(np.float32) * MM_HR_TO_M_S),
        jnp.asarray(rainfall_multiplier, dtype=jnp.float32),
        jnp.asarray(f0), jnp.asarray(fc), jnp.asarray(decay),
        jnp.asarray(external_times), jnp.asarray(receivers), jnp.asarray(external_q),
    )
    initial_node_volume = (
        host["node_storage_area_m2"].astype(np.float32)
        * host["node_initial_fill_depth_m"].astype(np.float32)
    )
    initial_state = ParityState(
        jnp.zeros(len(active), dtype=jnp.float32),
        jnp.zeros(len(active), dtype=jnp.float32),
        jnp.zeros(len(active), dtype=jnp.float32),
        jnp.zeros(len(active), dtype=jnp.float32),
        jnp.asarray(initial_node_volume),
        jnp.zeros(len(host["link_from_node_index"]), dtype=jnp.float32),
        jnp.zeros(len(recharge.source_cell), dtype=jnp.float32),
    )
    return LoadedParityInputs(
        surface, coupling, pipe, recharge, initial_state, forcing, host,
        {"assets": asset_report, "recharge": recharge_report, "external_inflow": external_report},
    )
