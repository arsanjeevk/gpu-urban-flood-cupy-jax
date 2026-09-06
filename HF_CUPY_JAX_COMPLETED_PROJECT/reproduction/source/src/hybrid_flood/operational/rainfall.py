"""V1-compatible rainfall validation and canonicalization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .errors import RainfallInputError


@dataclass(frozen=True)
class RainfallEvent:
    frame: pd.DataFrame
    duration_s: float
    total_depth_mm: float
    peak_intensity_mm_hr: float
    source: str

    def write_parquet(self, path: str | Path, *, scenario: str = "operational") -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        output = pd.DataFrame(
            {
                "timestamp": self.frame["timestamp"],
                "scenario": scenario,
                "rainfall_mm_hr": self.frame["intensity_mmhr"],
                "is_missing_timestamp": False,
                "units": "mm/hr",
                "source": self.frame["source"],
            }
        )
        output.to_parquet(target, index=False)
        return target


def _timestamp_frame(source: pd.DataFrame) -> pd.DataFrame:
    if not {"timestamp", "rainfall_mm"}.issubset(source.columns):
        raise RainfallInputError(
            "Rainfall CSV requires start_min/end_min with intensity_mmhr or depth_mm, "
            "or timestamp/rainfall_mm."
        )
    timestamps = pd.to_datetime(source["timestamp"], errors="coerce", utc=True)
    if timestamps.isna().any() or timestamps.duplicated().any():
        raise RainfallInputError("Rainfall timestamps must be valid and unique.")
    order = np.argsort(timestamps.to_numpy())
    timestamps = timestamps.iloc[order].reset_index(drop=True)
    rain = pd.to_numeric(source.iloc[order]["rainfall_mm"], errors="coerce").reset_index(drop=True)
    if len(timestamps) < 2:
        raise RainfallInputError("Timestamp rainfall requires at least two rows.")
    durations = timestamps.diff().dt.total_seconds().shift(-1)
    durations.iloc[-1] = durations.iloc[-2]
    return pd.DataFrame(
        {
            "start_min": (timestamps - timestamps.iloc[0]).dt.total_seconds() / 60.0,
            "end_min": (timestamps - timestamps.iloc[0]).dt.total_seconds() / 60.0
            + durations / 60.0,
            "duration_min": durations / 60.0,
            "intensity_mmhr": rain * 3600.0 / durations,
            "depth_mm": rain,
            "timestamp": timestamps,
            "source": source.iloc[order].get("source", pd.Series("upload", index=order)).values,
        }
    )


def load_rainfall_csv(path: str | Path) -> RainfallEvent:
    source_path = Path(path)
    try:
        source = pd.read_csv(source_path)
    except Exception as exc:
        raise RainfallInputError(f"Could not read rainfall CSV: {exc}") from exc
    if source.empty:
        raise RainfallInputError("Rainfall CSV has no data rows.")
    if {"start_min", "end_min"}.issubset(source.columns):
        frame = source.copy()
        for name in ("start_min", "end_min"):
            frame[name] = pd.to_numeric(frame[name], errors="coerce")
        frame["duration_min"] = frame["end_min"] - frame["start_min"]
        if "intensity_mmhr" in frame:
            frame["intensity_mmhr"] = pd.to_numeric(frame["intensity_mmhr"], errors="coerce")
            frame["depth_mm"] = frame["intensity_mmhr"] * frame["duration_min"] / 60.0
        elif "depth_mm" in frame:
            frame["depth_mm"] = pd.to_numeric(frame["depth_mm"], errors="coerce")
            frame["intensity_mmhr"] = frame["depth_mm"] * 60.0 / frame["duration_min"]
        else:
            raise RainfallInputError("Rainfall CSV requires intensity_mmhr or depth_mm.")
        if "timestamp_ist" in frame:
            frame["timestamp"] = pd.to_datetime(frame["timestamp_ist"], errors="coerce", utc=True)
        else:
            epoch = pd.Timestamp("2000-01-01", tz="UTC")
            frame["timestamp"] = epoch + pd.to_timedelta(frame["start_min"], unit="min")
        frame["source"] = frame.get("source", "upload")
    else:
        frame = _timestamp_frame(source)
    frame = frame.sort_values("start_min").reset_index(drop=True)
    numeric = frame[["start_min", "end_min", "duration_min", "intensity_mmhr", "depth_mm"]]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise RainfallInputError("Rainfall contains missing or non-finite values.")
    if (numeric < 0).any().any() or (frame["duration_min"] <= 0).any():
        raise RainfallInputError("Rainfall values and times must be non-negative with positive durations.")
    if (frame["start_min"].iloc[1:].to_numpy() < frame["end_min"].iloc[:-1].to_numpy()).any():
        raise RainfallInputError("Rainfall intervals overlap.")
    if frame["intensity_mmhr"].max() > 500 or frame["depth_mm"].sum() > 2_000:
        raise RainfallInputError("Rainfall magnitude is implausible; check units.")
    return RainfallEvent(
        frame=frame,
        duration_s=float(frame["end_min"].max() * 60.0),
        total_depth_mm=float(frame["depth_mm"].sum()),
        peak_intensity_mm_hr=float(frame["intensity_mmhr"].max()),
        source=str(frame["source"].iloc[0]),
    )
