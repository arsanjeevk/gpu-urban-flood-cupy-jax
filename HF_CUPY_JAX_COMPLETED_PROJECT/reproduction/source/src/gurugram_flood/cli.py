"""Command line interface: ``gurugram-flood``.

Subcommands
-----------
run                 Run a flood simulation from a rainfall CSV.
validate-rainfall   Validate a rainfall CSV without running the solver.
check               Check environment, backend and data files.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import load_config
from .exceptions import GurugramFloodError
from .logging_utils import setup_logging


def _add_run_parser(subparsers) -> None:
    p = subparsers.add_parser("run", help="Run a flood simulation from a rainfall CSV")
    p.add_argument("--rainfall", required=True, help="Path to the rainfall CSV (the only runtime input)")
    p.add_argument("--output-dir", default=None, help="Artifact directory (default: outputs/run_<timestamp>)")
    p.add_argument("--config", default=None, help="YAML config file (default: built-in nowcast defaults)")
    p.add_argument("--duration-min", type=float, default=None,
                   help="Simulated duration in minutes (default: the rainfall CSV's coverage)")
    p.add_argument("--backend", choices=["cupy"], default=None,
                   help="GPU array backend (CuPy only)")
    p.add_argument("--surface-mode", choices=["full_swe", "local_inertial"], default=None,
                   help="Surface stepping scheme (default from config: full_swe)")
    p.add_argument("--snapshot-interval-min", type=float, default=None,
                   help="Snapshot interval in minutes (default from config: 60)")
    p.add_argument("--no-basemap", action="store_true", help="Skip the Bokeh basemap HTML")
    p.add_argument("--no-npz", action="store_true", help="Skip the full NPZ artifact")
    p.add_argument("--log-file", default=None, help="Also write logs to this file")
    p.add_argument("--quiet", action="store_true", help="Only warnings and errors on the console")


def _cmd_run(args) -> int:
    from .api import run_flood_simulation

    overrides = {
        "solver.duration_min": args.duration_min,
        "solver.backend": args.backend,
        "solver.surface_step_mode": args.surface_mode,
        "solver.snapshot_interval_min": args.snapshot_interval_min,
    }
    if args.no_basemap:
        overrides["outputs.write_basemap_html"] = False
    if args.no_npz:
        overrides["outputs.write_npz"] = False

    run = run_flood_simulation(
        rainfall_csv=args.rainfall,
        output_dir=args.output_dir,
        config=args.config,
        overrides=overrides,
    )
    print(json.dumps(
        {
            "status": "ok",
            "output_dir": str(run.output_dir),
            "max_depth_m": run.report["max_depth_m"],
            "runtime_s": run.report["runtime_s"],
            "mass_residual_ratio": run.report["mass_residual_ratio"],
            "artifacts": {k: str(v) for k, v in run.artifacts.items()},
        },
        indent=2,
    ))
    return 0


def _cmd_validate_rainfall(args) -> int:
    from .api import validate_rainfall_csv

    event = validate_rainfall_csv(args.rainfall)
    print(json.dumps(
        {
            "status": "ok",
            "intervals": event.n_steps,
            "total_depth_mm": round(event.total_depth_mm, 2),
            "duration_min": event.duration_min,
            "peak_intensity_mmhr": float(event.hyetograph["intensity_mmhr"].max()),
        },
        indent=2,
    ))
    return 0


def _cmd_check(args) -> int:
    cfg = load_config(args.config)
    checks: dict[str, str] = {}

    required = ["production_topology_npz", "horton_bcr_npz", "recharge_csv"]
    optional = ["external_boundary_inflow_csv", "mesh_npz", "dem_tif",
                "drain_capacities_geojson", "outfall_inventory_geojson", "tailwater_priors_csv"]
    ok = True
    for name in required + optional:
        path = cfg.paths.resolve(name)
        if path.exists():
            checks[name] = f"ok ({path.stat().st_size / 1e6:.1f} MB)"
        else:
            checks[name] = "MISSING" + ("" if name in optional else " (required)")
            if name in required:
                ok = False

    try:
        import cupy

        n_gpu = cupy.cuda.runtime.getDeviceCount()
        device = cupy.cuda.runtime.getDeviceProperties(0)["name"].decode() if n_gpu else None
        checks["cupy"] = f"ok ({n_gpu} GPU(s): {device})" if n_gpu else "installed, but no GPU visible"
    except Exception as exc:  # ImportError or CUDA runtime failure
        checks["cupy"] = f"UNAVAILABLE ({type(exc).__name__}); GPU execution is blocked"
        ok = False

    print(json.dumps({"status": "ok" if ok else "not-ready", "checks": checks}, indent=2))
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gurugram-flood",
        description="Gurugram citywide GPU flood simulation backend",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_run_parser(subparsers)

    p_val = subparsers.add_parser("validate-rainfall", help="Validate a rainfall CSV")
    p_val.add_argument("rainfall", help="Path to the rainfall CSV")

    p_check = subparsers.add_parser("check", help="Check environment and data files")
    p_check.add_argument("--config", default=None, help="YAML config file")

    args = parser.parse_args(argv)
    level = logging.WARNING if getattr(args, "quiet", False) else logging.INFO
    setup_logging(level=level, log_file=getattr(args, "log_file", None))

    try:
        if args.command == "run":
            return _cmd_run(args)
        if args.command == "validate-rainfall":
            return _cmd_validate_rainfall(args)
        if args.command == "check":
            return _cmd_check(args)
        parser.error(f"unknown command {args.command}")
    except GurugramFloodError as exc:
        print(json.dumps({"status": "error", "error_type": type(exc).__name__, "message": str(exc)}), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
