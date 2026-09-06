"""Rainfall CSV loading, validation and normalisation.

The rainfall CSV is the *only* runtime input a caller must provide. Two
layouts are accepted:

1. **Hyetograph layout** (produced by the nowcast fetcher and used for
   historical events)::

       step,start_min,end_min,duration_min,intensity_mmhr,depth_mm[,timestamp_ist,...]

   Missing derivable columns are filled in (``duration_min`` from
   start/end, ``intensity_mmhr`` from ``depth_mm`` or vice versa).

2. **Timestamp layout** — one row per interval with a timestamp and a
   rainfall depth::

       timestamp,rainfall_mm            (aliases: timestamp_ist/time/datetime,
                                         depth_mm/precip_mm/rain_mm)

   Interval length is inferred from consecutive timestamps (falls back to
   60 min for a single row).

Both are normalised to the canonical hyetograph DataFrame consumed by the
solver. The time-to-rate lookup is delegated to
:func:`gurugram_flood.domain.real_gurugram_domain.hyetograph_rate_fn` so the
numerical behaviour is identical to the validated notebook runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from ..domain.real_gurugram_domain import hyetograph_rate_fn
from ..exceptions import RainfallInputError
from ..logging_utils import get_logger

log = get_logger("rainfall")

CANONICAL_COLUMNS = ["step", "start_min", "end_min", "duration_min", "intensity_mmhr", "depth_mm"]

_TIMESTAMP_ALIASES = ["timestamp_ist", "timestamp", "time", "datetime"]
_DEPTH_ALIASES = ["depth_mm", "rainfall_mm", "precip_mm", "rain_mm", "precipitation_mm"]

# Sanity bounds — violations raise, they are almost certainly unit mistakes.
MAX_INTENSITY_MMHR = 500.0   # world-record hourly rain is ~305 mm/hr
MAX_TOTAL_DEPTH_MM = 2000.0


@dataclass
class RainfallEvent:
    """Validated rainfall forcing ready for the solver."""

    hyetograph: pd.DataFrame                 # canonical columns (+ timestamp_ist when known)
    rate_fn: Callable[[float], float]        # simulation time (s) -> rainfall rate (m/s)
    total_depth_mm: float
    duration_min: float
    source_csv: str
    # hour index (0-based from simulation start) -> IST timestamp string / intensity
    timestamps_by_hour: dict[int, str] = field(default_factory=dict)
    intensity_by_hour: dict[int, float] = field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        return len(self.hyetograph)


def _find_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
    lower = {c.lower().strip(): c for c in df.columns}
    for alias in aliases:
        if alias in lower:
            return lower[alias]
    return None


def _from_timestamp_layout(df: pd.DataFrame, ts_col: str, depth_col: str) -> pd.DataFrame:
    """Convert a timestamp+depth table into the canonical hyetograph layout."""
    ts = pd.to_datetime(df[ts_col], errors="coerce")
    if ts.isna().any():
        bad = df.loc[ts.isna(), ts_col].iloc[0]
        raise RainfallInputError(f"Unparseable timestamp in column '{ts_col}': {bad!r}")
    if not ts.is_monotonic_increasing:
        raise RainfallInputError("Timestamps must be sorted in increasing order")

    depth_mm = pd.to_numeric(df[depth_col], errors="coerce")
    minutes = (ts - ts.iloc[0]).dt.total_seconds().to_numpy() / 60.0
    if len(minutes) > 1:
        diffs = np.diff(minutes)
        if np.any(diffs <= 0):
            raise RainfallInputError("Duplicate or non-increasing timestamps found")
        durations = np.append(diffs, diffs[-1])
    else:
        durations = np.array([60.0])

    out = pd.DataFrame(
        {
            "step": np.arange(1, len(df) + 1),
            "start_min": minutes,
            "end_min": minutes + durations,
            "duration_min": durations,
            "depth_mm": depth_mm.to_numpy(dtype=float),
        }
    )
    out["intensity_mmhr"] = out["depth_mm"] / out["duration_min"] * 60.0
    out["timestamp_ist"] = df[ts_col].astype(str).to_numpy()
    return out


def _normalise_hyetograph_layout(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.lower().strip() for c in out.columns]

    if "start_min" not in out.columns or "end_min" not in out.columns:
        raise RainfallInputError(
            "Hyetograph CSV must contain 'start_min' and 'end_min' columns "
            f"(found: {list(df.columns)})"
        )
    for col in ("start_min", "end_min", "duration_min", "intensity_mmhr", "depth_mm"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "duration_min" not in out.columns or out["duration_min"].isna().any():
        out["duration_min"] = out["end_min"] - out["start_min"]

    has_intensity = "intensity_mmhr" in out.columns and not out["intensity_mmhr"].isna().any()
    has_depth = "depth_mm" in out.columns and not out["depth_mm"].isna().any()
    if not has_intensity and not has_depth:
        raise RainfallInputError("Hyetograph CSV needs an 'intensity_mmhr' or 'depth_mm' column")
    if not has_intensity:
        out["intensity_mmhr"] = out["depth_mm"] / out["duration_min"] * 60.0
    if not has_depth:
        out["depth_mm"] = out["intensity_mmhr"] * out["duration_min"] / 60.0

    if "step" not in out.columns:
        out["step"] = np.arange(1, len(out) + 1)
    return out


def _validate(hyeto: pd.DataFrame, source: str) -> None:
    if len(hyeto) == 0:
        raise RainfallInputError(f"Rainfall CSV has no data rows: {source}")

    for col in ("start_min", "end_min", "duration_min", "intensity_mmhr"):
        if hyeto[col].isna().any():
            raise RainfallInputError(f"Column '{col}' contains missing/non-numeric values in {source}")

    if (hyeto["intensity_mmhr"] < 0).any():
        raise RainfallInputError(f"Negative rainfall intensity found in {source}")
    if (hyeto["duration_min"] <= 0).any():
        raise RainfallInputError(f"Non-positive interval duration found in {source}")

    starts = hyeto["start_min"].to_numpy(dtype=float)
    ends = hyeto["end_min"].to_numpy(dtype=float)
    if not np.all(np.diff(starts) > 0):
        raise RainfallInputError(f"Rainfall intervals must be sorted by start_min in {source}")
    if np.any(ends[:-1] > starts[1:] + 1e-9):
        raise RainfallInputError(f"Overlapping rainfall intervals found in {source}")

    peak = float(hyeto["intensity_mmhr"].max())
    if peak > MAX_INTENSITY_MMHR:
        raise RainfallInputError(
            f"Peak intensity {peak:.1f} mm/hr exceeds the plausibility limit "
            f"({MAX_INTENSITY_MMHR} mm/hr) — check the units of {source}"
        )
    total = float((hyeto["intensity_mmhr"] * hyeto["duration_min"] / 60.0).sum())
    if total > MAX_TOTAL_DEPTH_MM:
        raise RainfallInputError(
            f"Total event depth {total:.0f} mm exceeds the plausibility limit "
            f"({MAX_TOTAL_DEPTH_MM} mm) — check the units of {source}"
        )


def load_rainfall_event(csv_path: str | Path) -> RainfallEvent:
    """Load, validate and normalise a user rainfall CSV.

    Raises :class:`RainfallInputError` with an actionable message on any
    formatting or plausibility problem.
    """
    path = Path(csv_path)
    if not path.exists():
        raise RainfallInputError(f"Rainfall CSV not found: {path}")
    try:
        raw = pd.read_csv(path)
    except Exception as exc:  # pandas raises many parser types
        raise RainfallInputError(f"Could not parse rainfall CSV {path}: {exc}") from exc

    lower_cols = {c.lower().strip() for c in raw.columns}
    if "start_min" in lower_cols and "end_min" in lower_cols:
        hyeto = _normalise_hyetograph_layout(raw)
    else:
        ts_col = _find_column(raw, _TIMESTAMP_ALIASES)
        depth_col = _find_column(raw, _DEPTH_ALIASES)
        if ts_col is None or depth_col is None:
            raise RainfallInputError(
                f"Unrecognised rainfall CSV layout in {path}. Provide either the "
                "hyetograph layout (start_min, end_min, intensity_mmhr or depth_mm) "
                "or a timestamp layout (timestamp + rainfall_mm)."
            )
        hyeto = _from_timestamp_layout(raw, ts_col, depth_col)

    _validate(hyeto, str(path))

    total_mm = float((hyeto["intensity_mmhr"] * hyeto["duration_min"] / 60.0).sum())
    duration_min = float(hyeto["end_min"].max())

    # Hour-indexed lookups used by the GeoJSON/basemap exporters.
    timestamps_by_hour: dict[int, str] = {}
    intensity_by_hour: dict[int, float] = {}
    for _, row in hyeto.iterrows():
        hour = int(row["start_min"]) // 60
        intensity_by_hour.setdefault(hour, float(row["intensity_mmhr"]))
        if "timestamp_ist" in hyeto.columns and pd.notna(row.get("timestamp_ist")):
            timestamps_by_hour.setdefault(hour, str(row["timestamp_ist"]))

    log.info(
        "Rainfall event: %d intervals, %.1f mm total over %.0f min (peak %.1f mm/hr) from %s",
        len(hyeto), total_mm, duration_min, float(hyeto["intensity_mmhr"].max()), path.name,
    )

    return RainfallEvent(
        hyetograph=hyeto,
        rate_fn=hyetograph_rate_fn(hyeto),
        total_depth_mm=total_mm,
        duration_min=duration_min,
        source_csv=str(path),
        timestamps_by_hour=timestamps_by_hour,
        intensity_by_hour=intensity_by_hour,
    )
