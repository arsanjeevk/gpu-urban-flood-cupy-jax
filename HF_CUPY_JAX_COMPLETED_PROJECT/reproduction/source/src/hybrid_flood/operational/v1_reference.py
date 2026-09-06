"""Execute the unchanged V1 CuPy model as V2's numerical parity reference."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import uuid

from .errors import BackendUnavailableError, ConfigurationError, SimulationError
from .parity_assets import validate_v1_parity_assets


REFERENCE_SCRIPT = r"""
import json
import sys
from gurugram_flood.api import run_flood_simulation

rainfall, output_dir, duration_min, write_geojson = sys.argv[1:5]
run = run_flood_simulation(
    rainfall_csv=rainfall,
    output_dir=output_dir,
    overrides={
        "solver.backend": "cupy",
        "solver.surface_step_mode": "full_swe",
        "solver.duration_min": float(duration_min),
        "solver.snapshot_interval_min": 60.0,
        "solver.max_dt_s": 1.0,
        "solver.cfl": 0.85,
        "solver.cfl_update_interval_steps": 32,
        "solver.cfl_reuse_safety_factor": 0.8,
        "solver.max_speed_mps": 5.0,
        "physics.infiltration_mode": "production_horton",
        "physics.bcr_open_fraction": True,
        "physics.bcr_min_open_fraction": 0.05,
        "physics.use_surface_recharge": True,
        "physics.use_external_boundary_inflow": True,
        "physics.external_boundary_max_snap_m": 500.0,
        "outputs.write_npz": True,
        "outputs.write_hourly_geojson": write_geojson == "1",
        "outputs.write_basemap_html": False,
        "outputs.write_timeseries_png": True,
    },
)
print(json.dumps({"status": "ok", "report": run.report, "artifacts": {k: str(v) for k, v in run.artifacts.items()}}))
"""


def default_v1_python(v1_root: str | Path) -> Path:
    root = Path(v1_root).resolve()
    candidate = root.parent / "Test_Software/backend/.venv-linux/bin/python"
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise BackendUnavailableError(
            f"V1 CUDA Python is unavailable or not executable: {candidate}"
        )
    return candidate


def _published_paths(value, partial: Path, destination: Path):
    """Replace temporary publish paths recursively in V1 metadata."""

    if isinstance(value, dict):
        return {key: _published_paths(item, partial, destination) for key, item in value.items()}
    if isinstance(value, list):
        return [_published_paths(item, partial, destination) for item in value]
    if isinstance(value, str):
        return value.replace(str(partial), str(destination))
    return value


def run_v1_cuda_reference(
    *,
    rainfall_csv: str | Path,
    output_dir: str | Path,
    v1_root: str | Path,
    duration_min: float,
    v1_python: str | Path | None = None,
    write_geojson: bool = False,
) -> dict[str, object]:
    """Run V1 atomically and publish its triangle-native artifacts under V2."""

    if duration_min <= 0:
        raise ConfigurationError("Reference duration must be positive.")
    rainfall = Path(rainfall_csv).resolve()
    if not rainfall.is_file():
        raise ConfigurationError(f"Rainfall CSV does not exist: {rainfall}")
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise ConfigurationError(f"Reference output directory already exists: {destination}")

    parity = validate_v1_parity_assets(v1_root)
    interpreter = Path(v1_python).resolve() if v1_python else default_v1_python(v1_root)
    if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        raise BackendUnavailableError(f"V1 CUDA Python is unavailable: {interpreter}")

    partial = destination.parent / f".{destination.name}.partial-{uuid.uuid4().hex}"
    partial.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment["GURUGRAM_FLOOD_DATA_DIR"] = str(
        Path(v1_root).resolve() / "data" / "production_gurugram"
    )
    command = [
        str(interpreter), "-c", REFERENCE_SCRIPT, str(rainfall), str(partial),
        str(float(duration_min)), "1" if write_geojson else "0",
    ]
    try:
        completed = subprocess.run(
            command, env=environment, text=True, capture_output=True, check=False,
        )
        if completed.returncode != 0:
            raise SimulationError(
                "V1 CUDA reference failed with exit code "
                f"{completed.returncode}: {completed.stderr[-4000:]}"
            )
        report_path = partial / "run_report.json"
        native_npz = partial / "flood_simulation_outputs.npz"
        if not report_path.is_file() or not native_npz.is_file():
            raise SimulationError("V1 reference completed without required report/NPZ artifacts.")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("implementation") != "gurugram_flood_backend_fused_cuda":
            raise SimulationError("Unexpected implementation returned by the V1 reference process.")
        if report.get("surface_step_mode") != "full_swe" or report.get("backend") != "cupy":
            raise SimulationError("V1 reference did not use the required full-SWE CuPy profile.")
        report = _published_paths(report, partial, destination)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        reference_manifest = {
            "schema_version": 1,
            "model_variant": "v1_cuda_reference",
            "validation_status": "reference_model_only",
            "observationally_validated": False,
            "parity_assets": parity,
            "report": report,
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
        }
        (partial / "reference_manifest.json").write_text(
            json.dumps(reference_manifest, indent=2), encoding="utf-8"
        )
        partial.rename(destination)
        return reference_manifest
    except Exception:
        if partial.exists():
            shutil.rmtree(partial)
        raise
