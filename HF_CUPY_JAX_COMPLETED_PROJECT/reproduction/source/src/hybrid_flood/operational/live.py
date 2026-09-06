"""Live hourly rainfall acquisition using the public Open-Meteo forecast API."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

from .errors import BackendUnavailableError


def fetch_live_rainfall(output_path: str | Path, *, latitude: float = 28.4595, longitude: float = 77.0266, hours: int = 24) -> Path:
    query = urlencode({
        "latitude": latitude, "longitude": longitude, "hourly": "precipitation,precipitation_probability",
        "forecast_hours": hours, "timezone": "Asia/Kolkata",
    })
    try:
        with urlopen(f"https://api.open-meteo.com/v1/forecast?{query}", timeout=30) as response:  # noqa: S310
            payload = json.load(response)
        hourly = payload["hourly"]
        timestamp = pd.to_datetime(hourly["time"])
        rain = pd.Series(hourly["precipitation"], dtype=float)
    except Exception as exc:
        raise BackendUnavailableError(f"Live rainfall fetch failed: {exc}") from exc
    frame = pd.DataFrame({
        "step": range(1, len(rain) + 1), "start_min": 60 * pd.RangeIndex(len(rain)),
        "end_min": 60 * (pd.RangeIndex(len(rain)) + 1), "duration_min": 60,
        "intensity_mmhr": rain, "depth_mm": rain, "source": "open_meteo_forecast",
        "timestamp_ist": timestamp.astype(str),
        "precipitation_probability_pct": hourly.get("precipitation_probability", [None] * len(rain)),
    })
    target = Path(output_path); target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    return target
