"""High-level Python API — the single entry point for backend integration.

Typical use::

    from gurugram_flood import run_flood_simulation

    run = run_flood_simulation(
        rainfall_csv="nowcast.csv",
        output_dir="outputs/run_2026_07_10",
    )
    print(run.report["max_depth_m"])
    print(run.artifacts["hourly_geojson_dir"])

Everything else (mesh, drainage topology, infiltration fields, recharge)
is loaded automatically from the packaged ``data/`` directory.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .config import RunConfig, load_config
from .logging_utils import get_logger
from .postprocessing.basemap import export_basemap_html
from .postprocessing.geojson_export import export_hourly_geojsons
from .postprocessing.outputs import build_report, save_report, save_run_npz, save_timeseries_png
from .preprocessing.city_data import CityDomain, load_city_domain
from .preprocessing.rainfall import RainfallEvent, load_rainfall_event
from .solver.engine import ProgressCallback, SimulationResult, run_simulation

log = get_logger("api")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = _PROJECT_ROOT / "outputs"


@dataclass
class FloodRunResult:
    """What a caller gets back from :func:`run_flood_simulation`."""

    output_dir: Path
    report: dict[str, Any]
    # Artifact name -> path (only artifacts that were actually written):
    #   run_npz, report_json, hourly_geojson_dir, hourly_summary_csv,
    #   basemap_html, timeseries_png
    artifacts: dict[str, Path] = field(default_factory=dict)
    # In-memory result for further processing (arrays are host NumPy).
    simulation: SimulationResult | None = field(default=None, repr=False)


def validate_rainfall_csv(rainfall_csv: str | Path) -> RainfallEvent:
    """Validate a rainfall CSV without running anything heavy.

    Raises :class:`gurugram_flood.exceptions.RainfallInputError` on problems.
    """
    return load_rainfall_event(rainfall_csv)


def load_city(config: RunConfig | str | Path | None = None,
              overrides: Mapping[str, Any] | None = None) -> tuple[CityDomain, RunConfig]:
    """Load the static city datasets once (reusable across multiple runs)."""
    cfg = config if isinstance(config, RunConfig) else load_config(config, overrides)
    return load_city_domain(cfg), cfg


def run_flood_simulation(
    rainfall_csv: str | Path,
    output_dir: str | Path | None = None,
    config: RunConfig | str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
    city: CityDomain | None = None,
    progress_callback: ProgressCallback | None = None,
    keep_arrays: bool = False,
) -> FloodRunResult:
    """Run the full pipeline: rainfall CSV -> CUDA simulation -> artifacts.

    Parameters
    ----------
    rainfall_csv:
        The only required runtime input. See
        :mod:`gurugram_flood.preprocessing.rainfall` for accepted layouts.
    output_dir:
        Where artifacts are written. Defaults to
        ``outputs/run_<UTC timestamp>`` under the project root.
    config:
        A :class:`RunConfig`, a YAML config path, or None for defaults.
    overrides:
        Dotted-key config overrides, e.g. ``{"solver.duration_min": 720}``.
    city:
        Pre-loaded :class:`CityDomain` to skip the ~topology load when
        running many simulations in one process (see :func:`load_city`).
    progress_callback:
        ``(fraction_complete, snapshot_info) -> None``, called at every
        snapshot — for job-queue progress reporting.
    keep_arrays:
        If True, the returned object keeps the in-memory
        :class:`SimulationResult` (large; ~GBs for long runs).
    """
    t_total0 = time.perf_counter()
    cfg = config if isinstance(config, RunConfig) else load_config(config, overrides)

    rainfall = load_rainfall_event(rainfall_csv)

    if city is None:
        city = load_city_domain(cfg)

    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = DEFAULT_OUTPUT_ROOT / f"run_{stamp}"
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    result = run_simulation(city, rainfall, cfg, progress_callback=progress_callback)

    artifacts: dict[str, Path] = {}
    if cfg.outputs.write_npz:
        artifacts["run_npz"] = save_run_npz(result, out_path)
    if cfg.outputs.write_hourly_geojson:
        geojson_dir, summary_csv = export_hourly_geojsons(result, out_path)
        artifacts["hourly_geojson_dir"] = geojson_dir
        artifacts["hourly_summary_csv"] = summary_csv
    if cfg.outputs.write_basemap_html:
        basemap = export_basemap_html(result, out_path)
        if basemap is not None:
            artifacts["basemap_html"] = basemap
    if cfg.outputs.write_timeseries_png:
        png = save_timeseries_png(result, out_path)
        if png is not None:
            artifacts["timeseries_png"] = png

    report = build_report(result, {k: str(v) for k, v in artifacts.items()})
    report["total_pipeline_s"] = time.perf_counter() - t_total0
    artifacts["report_json"] = save_report(report, out_path)

    log.info("Pipeline complete in %.1f s -> %s", report["total_pipeline_s"], out_path)
    return FloodRunResult(
        output_dir=out_path,
        report=report,
        artifacts=artifacts,
        simulation=result if keep_arrays else None,
    )
