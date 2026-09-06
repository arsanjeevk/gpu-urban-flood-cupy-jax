"""Helpers for notebook-level production event features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from gurugram_flood.kernels.network_kernels import scatter_add_1d


def _trapezoid_compat(y: np.ndarray, x: np.ndarray) -> float:
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is not None:
        return float(trapezoid(y, x))
    return float(np.trapz(y, x))


@dataclass(frozen=True)
class ExternalBoundaryInflowSurface:
    receiver_cell_index: np.ndarray
    times_s: np.ndarray
    q_m3ps_by_receiver: np.ndarray
    peak_total_m3ps: float
    report: dict[str, Any]


def _first_present_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_to_actual = {str(col).lower(): str(col) for col in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_actual:
            return lower_to_actual[candidate.lower()]
    return None


def _time_s(frame: pd.DataFrame) -> np.ndarray:
    time_col = _first_present_column(frame, ["time_s", "seconds", "t_s"])
    if time_col is not None:
        return pd.to_numeric(frame[time_col], errors="coerce").to_numpy(dtype=np.float64)
    minute_col = _first_present_column(frame, ["time_min", "minute", "minutes", "t_min"])
    if minute_col is not None:
        return pd.to_numeric(frame[minute_col], errors="coerce").to_numpy(dtype=np.float64) * 60.0
    hour_col = _first_present_column(frame, ["time_hr", "hour", "hours", "t_hr"])
    if hour_col is not None:
        return pd.to_numeric(frame[hour_col], errors="coerce").to_numpy(dtype=np.float64) * 3600.0
    raise ValueError("External boundary inflow CSV requires time_s, time_min, or time_hr.")


def _flow_m3ps(frame: pd.DataFrame) -> np.ndarray:
    flow_col = _first_present_column(frame, ["q_m3ps", "flow_m3ps", "discharge_m3ps"])
    if flow_col is None:
        raise ValueError("External boundary inflow CSV requires q_m3ps or equivalent flow column.")
    q = pd.to_numeric(frame[flow_col], errors="coerce").to_numpy(dtype=np.float64)
    if np.any(~np.isfinite(q)):
        raise ValueError("External boundary inflow flow column contains non-finite values.")
    return q


def _cell_index(frame: pd.DataFrame) -> np.ndarray | None:
    cell_col = _first_present_column(frame, ["cell_index", "receiver_cell_index", "triangle_index"])
    if cell_col is None:
        return None
    cell_index = pd.to_numeric(frame[cell_col], errors="coerce").to_numpy(dtype=np.float64)
    if np.any(~np.isfinite(cell_index)):
        raise ValueError("External boundary inflow cell_index contains non-finite values.")
    return cell_index.astype(np.int64)


def _xy_utm(frame: pd.DataFrame) -> tuple[np.ndarray | None, np.ndarray | None]:
    x_col = _first_present_column(frame, ["x_utm43n_m", "utm43n_easting_m", "easting_m", "x"])
    y_col = _first_present_column(frame, ["y_utm43n_m", "utm43n_northing_m", "northing_m", "y"])
    if x_col is None or y_col is None:
        return None, None
    x = pd.to_numeric(frame[x_col], errors="coerce").to_numpy(dtype=np.float64)
    y = pd.to_numeric(frame[y_col], errors="coerce").to_numpy(dtype=np.float64)
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise ValueError("External boundary inflow UTM coordinates contain non-finite values.")
    return x, y


def load_external_boundary_inflow_surface(
    csv_path: str | Path,
    centroids_xy: np.ndarray,
    active_mask: np.ndarray,
    max_snap_m: float = 500.0,
) -> ExternalBoundaryInflowSurface:
    frame = pd.read_csv(Path(csv_path))
    if frame.empty:
        raise ValueError(f"External boundary inflow CSV is empty: {csv_path}")
    time_s = _time_s(frame)
    q_m3ps = _flow_m3ps(frame)
    inflow_col = _first_present_column(frame, ["inflow_id", "source_id", "id"])
    inflow_ids = (
        frame[inflow_col].astype(str).to_numpy()
        if inflow_col is not None
        else np.asarray([f"INFLOW_{i:04d}" for i in range(len(frame))], dtype=object)
    )
    raw_cell_index = _cell_index(frame)
    x_utm, y_utm = _xy_utm(frame)

    active_cells = np.flatnonzero(active_mask.astype(bool))
    if active_cells.size == 0:
        raise ValueError("No active cells available for external inflow mapping.")
    if raw_cell_index is not None:
        if np.any(raw_cell_index < 0) or np.any(raw_cell_index >= centroids_xy.shape[0]):
            raise ValueError("External boundary inflow cell_index is outside mesh range.")
        if np.any(~active_mask[raw_cell_index]):
            raise ValueError("External boundary inflow cell_index contains inactive cells.")
        receiver_cell_index = raw_cell_index.astype(np.int32)
        snap_distance_m = np.zeros(receiver_cell_index.shape, dtype=np.float64)
    else:
        if x_utm is None or y_utm is None:
            raise ValueError("External boundary inflow CSV requires cell_index or UTM coordinates.")
        active_xy = centroids_xy[active_cells]
        dist, nearest = cKDTree(active_xy).query(np.column_stack([x_utm, y_utm]), k=1)
        receiver_cell_index = active_cells[np.asarray(nearest, dtype=np.int64)].astype(np.int32)
        snap_distance_m = np.asarray(dist, dtype=np.float64)

    receiver_key_to_index: dict[tuple[str, int], int] = {}
    receiver_records: list[dict[str, Any]] = []
    times = np.asarray(sorted(set(float(v) for v in time_s)), dtype=np.float64)
    time_to_index = {float(value): index for index, value in enumerate(times)}
    q_columns: list[np.ndarray] = []

    for row_index, (inflow_id, cell, snap_m) in enumerate(zip(inflow_ids, receiver_cell_index, snap_distance_m)):
        key = (str(inflow_id), int(cell))
        if key not in receiver_key_to_index:
            receiver_key_to_index[key] = len(receiver_records)
            receiver_records.append(
                {
                    "receiver_index": int(len(receiver_records)),
                    "inflow_id": str(inflow_id),
                    "cell_index": int(cell),
                    "x_utm43n_m": float(centroids_xy[int(cell), 0]),
                    "y_utm43n_m": float(centroids_xy[int(cell), 1]),
                    "snap_distance_m_max": float(snap_m),
                    "snap_warning": bool(float(snap_m) > float(max_snap_m)),
                }
            )
            q_columns.append(np.zeros((times.size,), dtype=np.float64))
        receiver_index = receiver_key_to_index[key]
        q_columns[receiver_index][time_to_index[float(time_s[row_index])]] += float(q_m3ps[row_index])

    q_by_receiver = np.vstack(q_columns).T.astype(np.float32) if q_columns else np.zeros((times.size, 0), dtype=np.float32)
    totals = np.sum(q_by_receiver, axis=1) if q_by_receiver.size else np.zeros((times.size,), dtype=np.float32)
    report = {
        "enabled": True,
        "csv": str(csv_path),
        "receiver_count": int(len(receiver_records)),
        "peak_total_m3ps": float(np.max(totals)) if totals.size else 0.0,
        "hydrograph_volume_m3": _trapezoid_compat(totals.astype(np.float64), times) if times.size >= 2 else 0.0,
        "max_snap_m": float(np.max(snap_distance_m)) if snap_distance_m.size else 0.0,
        "snap_warning_count": int(sum(1 for record in receiver_records if record["snap_warning"])),
        "receiver_records": receiver_records[:20],
    }
    return ExternalBoundaryInflowSurface(
        receiver_cell_index=np.asarray([record["cell_index"] for record in receiver_records], dtype=np.int32),
        times_s=times.astype(np.float64),
        q_m3ps_by_receiver=q_by_receiver,
        peak_total_m3ps=float(report["peak_total_m3ps"]),
        report=report,
    )


def external_boundary_inflow_q_m3ps(external: ExternalBoundaryInflowSurface | None, time_s: float) -> np.ndarray:
    if external is None or external.q_m3ps_by_receiver.size == 0:
        return np.zeros((0,), dtype=np.float32)
    times = np.asarray(external.times_s, dtype=np.float64)
    q = np.asarray(external.q_m3ps_by_receiver, dtype=np.float64)
    if time_s <= float(times[0]):
        return q[0].astype(np.float32)
    if time_s >= float(times[-1]):
        return q[-1].astype(np.float32)
    return np.asarray([np.interp(time_s, times, q[:, j]) for j in range(q.shape[1])], dtype=np.float32)


def apply_external_boundary_inflow_depth(
    h_m: Any,
    cell_area_m2: Any,
    external: ExternalBoundaryInflowSurface | None,
    timestep_s: float,
    time_s: float,
    xp: Any,
) -> tuple[Any, Any]:
    if external is None or external.receiver_cell_index.size == 0:
        zero = xp.sum(h_m * 0.0)
        return h_m, zero
    q_m3ps = xp.asarray(external_boundary_inflow_q_m3ps(external, time_s), dtype=getattr(xp, "float32", float))
    receiver = xp.asarray(external.receiver_cell_index, dtype=getattr(xp, "int32", int))
    added_m3 = q_m3ps * timestep_s
    new_h = h_m + scatter_add_1d(receiver, added_m3 / cell_area_m2[receiver], int(h_m.shape[0]))
    return new_h, xp.sum(added_m3)


def scale_node_volume_to_target(node_volume: Any, target_total_m3: float, xp: Any) -> tuple[Any, float]:
    current_total = float(xp.sum(node_volume))
    if current_total <= 0.0 or target_total_m3 < 0.0:
        return node_volume, 1.0
    scale = float(target_total_m3) / current_total
    return node_volume * scale, scale
