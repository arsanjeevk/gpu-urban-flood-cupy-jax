"""Operational command-line contract."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .assets import check_v1_assets, prepare_v1_assets
from .comparison_products import publish_jax_speed_comparison, publish_same_condition_comparison
from .errors import OperationalError
from .live import fetch_live_rainfall
from .parity_assets import validate_v1_parity_assets
from .parity_raster import rasterize_reference_to_grid
from .parity_runner import run_jax_v1_parity
from .parity_validation import validate_v1_parity_run
from .rainfall import load_rainfall_csv
from .runner import run_operational
from .tiles import generate_xyz_tiles
from .v1_reference import run_v1_cuda_reference


def _defaults() -> tuple[Path, Path]:
    project = Path(__file__).resolve().parents[3]
    v1 = Path(os.environ.get("FLOODASTRA_V1_ROOT", project))
    return project, v1


def parser() -> argparse.ArgumentParser:
    project, v1 = _defaults()
    root = argparse.ArgumentParser(prog="v2-flood")
    commands = root.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check")
    check.add_argument("--v1-root", type=Path, default=v1)
    prepare = commands.add_parser("prepare-assets")
    prepare.add_argument("--v1-root", type=Path, default=v1)
    prepare.add_argument("--output-dir", type=Path, default=project / "data/operational")
    parity_check = commands.add_parser("check-parity")
    parity_check.add_argument("--v1-root", type=Path, default=v1)
    reference = commands.add_parser("run-reference")
    reference.add_argument("--rainfall", type=Path)
    reference.add_argument("--output-dir", type=Path, required=True)
    reference.add_argument("--duration-min", type=float, default=840.0)
    reference.add_argument("--v1-root", type=Path, default=v1)
    reference.add_argument("--v1-python", type=Path)
    reference.add_argument("--write-geojson", action="store_true")
    jax_parity = commands.add_parser("run-jax-parity")
    jax_parity.add_argument("--rainfall", type=Path)
    jax_parity.add_argument("--output-dir", type=Path, required=True)
    jax_parity.add_argument("--duration-min", type=float, default=840.0)
    jax_parity.add_argument("--v1-root", type=Path, default=v1)
    jax_parity.add_argument("--allow-cpu", action="store_true")
    jax_parity.add_argument("--max-steps-per-hour", type=int, default=20_000)
    jax_parity.add_argument("--loop-mode", choices=("while", "scan"), default="while")
    raster = commands.add_parser("rasterize-reference")
    raster.add_argument("--reference-dir", type=Path, required=True)
    raster.add_argument("--output-dir", type=Path, required=True)
    raster.add_argument("--rainfall", type=Path)
    raster.add_argument("--resolution", type=float, default=30.0)
    raster.add_argument("--model-variant")
    raster.add_argument("--prepared-dir", type=Path, default=project / "data/operational")
    raster.add_argument("--v1-root", type=Path, default=v1)
    raster.add_argument("--domain", type=Path, default=project / "data/interim/domain_epsg32643.gpkg")
    parity_validate = commands.add_parser("validate-parity")
    parity_validate.add_argument("--reference-dir", type=Path, required=True)
    parity_validate.add_argument("--candidate-dir", type=Path, required=True)
    parity_validate.add_argument("--threshold-m", type=float, default=0.05)
    comparison = commands.add_parser("publish-comparison")
    comparison.add_argument("--reference-dir", type=Path, required=True)
    comparison.add_argument("--candidate-dir", type=Path, required=True)
    comparison.add_argument("--reference-public-dir", type=Path, required=True)
    comparison.add_argument("--candidate-public-dir", type=Path, required=True)
    comparison.add_argument("--rainfall", type=Path)
    comparison.add_argument("--output-dir", type=Path, required=True)
    speed_comparison = commands.add_parser("publish-speed-comparison")
    speed_comparison.add_argument("--reference-dir", type=Path, required=True)
    speed_comparison.add_argument("--previous-jax-dir", type=Path, required=True)
    speed_comparison.add_argument("--optimized-jax-dir", type=Path, required=True)
    speed_comparison.add_argument("--output-dir", type=Path, required=True)
    validate = commands.add_parser("validate-rainfall")
    validate.add_argument("rainfall", type=Path)
    run = commands.add_parser("run")
    run.add_argument("--rainfall", type=Path)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--scenario", choices=("historical", "upload", "live"), default="upload")
    run.add_argument("--variant", choices=("raw_jax", "hybrid", "both"), default="both")
    run.add_argument("--resolution", type=float, choices=(10, 20, 30, 50), default=20)
    run.add_argument("--recession-hours", type=float, default=2)
    run.add_argument("--prepared-dir", type=Path, default=project / "data/operational")
    run.add_argument("--v1-root", type=Path, default=v1)
    run.add_argument("--domain", type=Path, default=project / "data/interim/domain_epsg32643.gpkg")
    run.add_argument("--checkpoint", type=Path)
    run.add_argument("--allow-cpu", action="store_true")
    tiles = commands.add_parser("tiles")
    tiles.add_argument("run_variant_dir", type=Path)
    tiles.add_argument("--min-zoom", type=int, default=9)
    tiles.add_argument("--max-zoom", type=int, default=15)
    tiles.add_argument("--threshold-m", type=float, default=0.05)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "check":
            payload = check_v1_assets(args.v1_root)
            try:
                import jax
                payload["devices"] = [str(device) for device in jax.devices()]
                payload["gpu_available"] = any(device.platform == "gpu" for device in jax.devices())
            except Exception as exc:
                payload.update(gpu_available=False, backend_error=str(exc))
            print(json.dumps(payload, indent=2)); return 0 if payload["ready"] else 2
        if args.command == "prepare-assets":
            print(json.dumps({"status": "ok", "manifest": str(prepare_v1_assets(args.v1_root, args.output_dir))})); return 0
        if args.command == "check-parity":
            print(json.dumps(validate_v1_parity_assets(args.v1_root), indent=2)); return 0
        if args.command == "run-reference":
            rainfall = args.rainfall or args.v1_root / "examples/july2025_133mm_12h.csv"
            result = run_v1_cuda_reference(
                rainfall_csv=rainfall, output_dir=args.output_dir, v1_root=args.v1_root,
                duration_min=args.duration_min, v1_python=args.v1_python,
                write_geojson=args.write_geojson,
            )
            print(json.dumps({"status": "ok", "output_dir": str(args.output_dir), "manifest": result})); return 0
        if args.command == "run-jax-parity":
            rainfall = args.rainfall or args.v1_root / "examples/july2025_133mm_12h.csv"
            result = run_jax_v1_parity(
                rainfall_csv=rainfall, output_dir=args.output_dir, v1_root=args.v1_root,
                duration_min=args.duration_min, require_gpu=not args.allow_cpu,
                max_steps_per_hour=args.max_steps_per_hour,
                loop_mode=args.loop_mode,
            )
            print(json.dumps({"status": "ok", "output_dir": str(args.output_dir), "report": result})); return 0
        if args.command == "rasterize-reference":
            rainfall = args.rainfall or args.v1_root / "examples/july2025_133mm_12h.csv"
            result = rasterize_reference_to_grid(
                reference_dir=args.reference_dir, output_dir=args.output_dir,
                rainfall_csv=rainfall, v1_root=args.v1_root,
                prepared_dir=args.prepared_dir, domain_path=args.domain,
                resolution_m=args.resolution, model_variant=args.model_variant,
            )
            print(json.dumps({"status": "ok", "manifest": result})); return 0
        if args.command == "validate-parity":
            result = validate_v1_parity_run(
                reference_dir=args.reference_dir, candidate_dir=args.candidate_dir,
                threshold_m=args.threshold_m,
            )
            print(json.dumps({"status": "ok" if result["passed"] else "failed", "validation": result}, indent=2))
            return 0 if result["passed"] else 2
        if args.command == "publish-comparison":
            rainfall = args.rainfall or args.reference_dir.parents[2] / "examples/july2025_133mm_12h.csv"
            result = publish_same_condition_comparison(
                reference_dir=args.reference_dir, candidate_dir=args.candidate_dir,
                reference_public_dir=args.reference_public_dir,
                candidate_public_dir=args.candidate_public_dir,
                rainfall_csv=rainfall, output_dir=args.output_dir,
            )
            print(json.dumps({"status": "ok", "comparison": result}, indent=2))
            return 0
        if args.command == "publish-speed-comparison":
            result = publish_jax_speed_comparison(
                reference_dir=args.reference_dir,
                previous_jax_dir=args.previous_jax_dir,
                optimized_jax_dir=args.optimized_jax_dir,
                output_dir=args.output_dir,
            )
            print(json.dumps({"status": "ok", "comparison": result}, indent=2))
            return 0
        if args.command == "validate-rainfall":
            event = load_rainfall_csv(args.rainfall)
            print(json.dumps({"status": "ok", "duration_s": event.duration_s, "total_depth_mm": event.total_depth_mm, "peak_intensity_mm_hr": event.peak_intensity_mm_hr})); return 0
        if args.command == "tiles":
            manifest = generate_xyz_tiles(
                args.run_variant_dir, min_zoom=args.min_zoom,
                max_zoom=args.max_zoom, threshold_m=args.threshold_m,
            )
            print(json.dumps({"status": "ok", "manifest": str(manifest)})); return 0
        rainfall = args.rainfall
        if args.scenario == "historical" and rainfall is None:
            rainfall = args.v1_root / "examples/july2025_133mm_12h.csv"
        if args.scenario == "live" and rainfall is None:
            rainfall = fetch_live_rainfall(args.output_dir.parent / f"{args.output_dir.name}_live_rainfall.csv")
        if rainfall is None:
            raise OperationalError("--rainfall is required for upload/live scenarios.")
        result = run_operational(
            rainfall_csv=rainfall, output_dir=args.output_dir, prepared_dir=args.prepared_dir,
            v1_root=args.v1_root, domain_path=args.domain, resolution_m=args.resolution,
            recession_hours=args.recession_hours, scenario=args.scenario, variant=args.variant,
            require_gpu=not args.allow_cpu, checkpoint=args.checkpoint,
        )
        print(json.dumps({"status": "ok", "output_dir": str(args.output_dir), "manifest": result})); return 0
    except OperationalError as exc:
        print(json.dumps({"status": "error", "error_type": type(exc).__name__, "message": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
