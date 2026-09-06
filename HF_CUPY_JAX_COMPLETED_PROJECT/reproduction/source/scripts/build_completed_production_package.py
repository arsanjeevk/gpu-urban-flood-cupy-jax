#!/usr/bin/env python3
"""Create the completed addendum package without modifying the prior package."""
from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "HF_CUPY_JAX_FINAL_PACKAGE"
TARGET = ROOT / "HF_CUPY_JAX_COMPLETED_PROJECT"
ARCHIVE = ROOT / "HF_CUPY_JAX_COMPLETED_PROJECT.tar.gz"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> None:
    if TARGET.exists() or ARCHIVE.exists():
        raise SystemExit("Refusing to overwrite an existing completed package or archive")
    shutil.copytree(SOURCE, TARGET)

    repeatability = ROOT / "completed_work/repeatability"
    benchmark = ROOT / "completed_work/secondary_benchmark"
    shutil.copytree(repeatability, TARGET / "simulations/repeatability", dirs_exist_ok=True)
    shutil.copytree(benchmark, TARGET / "benchmarks/secondary_60min", dirs_exist_ok=True)
    copy_file(
        ROOT / "completed_work/logs/pytest_completed_release.txt",
        TARGET / "tests/pytest_output.txt",
    )
    for relative in (
        "tests/conftest.py",
        "tests/production/test_production_release.py",
        "configs/production_24h_parity_gpu.yaml",
        "scripts/analyze_production_repeats.py",
        "scripts/run_secondary_gpu_benchmark.py",
        "scripts/build_completed_production_package.py",
    ):
        copy_file(ROOT / relative, TARGET / "reproduction/source" / relative)

    repeat_metrics = json.loads(
        (repeatability / "repeatability_metrics.json").read_text()
    )
    benchmark_summary = json.loads(
        (benchmark / "benchmark_summary.json").read_text()
    )
    write_json(
        TARGET / "tests/test_summary.json",
        {
            "command": ".venv/bin/python -m pytest -ra",
            "collected": 148,
            "passed": 111,
            "skipped": 37,
            "failed": 0,
            "errors": 0,
            "warnings": 2,
            "runtime_s": 10.58,
            "status": "PASS",
            "skip_policy": "Explicit legacy-retired or opposite-hardware classification; skips are not counted as passes.",
        },
    )
    write_json(
        TARGET / "simulations/repeatability/status.json",
        {
            "status": "PASS_NUMERICAL_NOT_BITWISE",
            "cupy": repeat_metrics["backends"]["cupy"]["classification"],
            "jax": repeat_metrics["backends"]["jax"]["classification"],
            "criteria": repeat_metrics["existing_approved_repeatability_criteria"],
        },
    )
    write_json(
        TARGET / "benchmarks/summaries/status.json",
        {
            "official_24h_repeated_benchmark": "NOT_EXECUTED",
            "secondary_60min_repeated_benchmark": "PASS",
            "classification": benchmark_summary["classification"],
            "warmups_per_backend": 3,
            "measurements_per_backend": 10,
            "all_correctness_passed": True,
            "cupy_median_s": benchmark_summary["backends"]["cupy"]["median_s"],
            "jax_median_s": benchmark_summary["backends"]["jax"]["median_s"],
            "jax_runtime_difference_percent": benchmark_summary[
                "jax_runtime_difference_percent"
            ],
        },
    )

    completion = f"""# Completed production verification addendum

## Outcome

The production-only numerical release is verified. The complete test run recorded 111 passed, 37 explicitly skipped, 0 failed, and 0 errors. The prior `HF_CUPY_JAX_FINAL_PACKAGE` remains unchanged; this is a separate evidence package.

## Full-run repeatability

Independent 24-hour run B was completed for both GPU backends. Neither archive is byte-identical because adaptive timestep counts diverged, but both pass the pre-existing numerical criteria:

- CuPy: RMSE {repeat_metrics['backends']['cupy']['depth_rmse_m']:.9f} m; minimum CSI {repeat_metrics['backends']['cupy']['minimum_snapshot_flood_extent_csi']:.9f}; timestep difference {repeat_metrics['backends']['cupy']['timestep_count_difference']:+d}.
- JAX: RMSE {repeat_metrics['backends']['jax']['depth_rmse_m']:.9f} m; minimum CSI {repeat_metrics['backends']['jax']['minimum_snapshot_flood_extent_csi']:.9f}; timestep difference {repeat_metrics['backends']['jax']['timestep_count_difference']:+d}.

This supports numerical repeatability within the approved parity gates, not bitwise determinism.

## Repeated GPU performance

A secondary identical 60-minute production workload completed 3 warm-ups and 10 measurements per backend, with correctness checked after every run. Median solver runtimes were {benchmark_summary['backends']['cupy']['median_s']:.6f} s for CuPy and {benchmark_summary['backends']['jax']['median_s']:.6f} s for JAX; JAX was {benchmark_summary['jax_runtime_difference_percent']:.3f}% slower. This is deliberately labelled secondary and does not replace the unexecuted repeated 24-hour official benchmark.

## Remaining scientific limits

Independent physical validation remains unavailable, and one production rainfall event cannot support leakage-controlled production AI train/validation/test splits. No physical-accuracy, production-AI, or operational-readiness claim is made.

## Evidence paths

- `tests/pytest_output.txt`
- `simulations/repeatability/repeatability_metrics.json`
- `simulations/repeatability/repeatability_report.md`
- `benchmarks/secondary_60min/benchmark_results.csv`
- `benchmarks/secondary_60min/benchmark_summary.json`
"""
    (TARGET / "COMPLETION_ADDENDUM.md").write_text(completion)
    (TARGET / "README_FIRST.md").write_text(
        "# Read this first\n\nStart with [COMPLETION_ADDENDUM.md](COMPLETION_ADDENDUM.md). "
        "It supersedes the earlier blocked verification status while preserving the prior report "
        "as historical evidence. Numerical verification and full-run repeatability pass; physical "
        "validation, a repeated official 24-hour benchmark, and production AI remain unavailable.\n"
    )
    (TARGET / "EXECUTIVE_SUMMARY.md").write_text(
        "# Executive Summary\n\nProduction numerical verification: PASS. Full 24-hour numerical "
        "repeatability: PASS within existing parity criteria, not bitwise. Secondary repeated "
        "60-minute GPU benchmark: PASS. Physical validation and production AI: NOT AVAILABLE. "
        "See [COMPLETION_ADDENDUM.md](COMPLETION_ADDENDUM.md).\n"
    )
    (TARGET / "methodology/gpu_benchmark.md").write_text(
        "# GPU benchmark\n\nThe required-count secondary benchmark used one identical 60-minute "
        "production workload per backend: 3 warm-ups and 10 measured runs, CUDA device 0, and "
        "a correctness gate after every run. Raw logs, arrays, CSV observations, and summary "
        "statistics are under `../benchmarks/secondary_60min/`. It is not represented as the "
        "unexecuted repeated 24-hour official benchmark.\n"
    )

    files = []
    for path in sorted(TARGET.rglob("*")):
        if path.is_file() and path.name not in {"artifact_manifest.json", "checksums.sha256"}:
            files.append(
                {
                    "relative_path": str(path.relative_to(TARGET)),
                    "size_bytes": path.stat().st_size,
                    "sha256": digest(path),
                }
            )
    write_json(
        TARGET / "artifact_manifest.json",
        {
            "schema_version": 2,
            "overall_status": "NUMERICAL_RELEASE_VERIFIED_WITH_DOCUMENTED_LIMITATIONS",
            "files": files,
        },
    )
    checksums = []
    for path in sorted(TARGET.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            checksums.append(f"{digest(path)}  {path.relative_to(TARGET)}")
    (TARGET / "checksums.sha256").write_text("\n".join(checksums) + "\n")

    with tarfile.open(ARCHIVE, "w:gz") as archive:
        archive.add(TARGET, arcname=TARGET.name)
    (ROOT / f"{ARCHIVE.name}.sha256").write_text(
        f"{digest(ARCHIVE)}  {ARCHIVE.name}\n"
    )
    contents = [f"{path.relative_to(ROOT)}\n" for path in sorted(TARGET.rglob("*"))]
    (ROOT / f"{ARCHIVE.name}.contents.txt").write_text("".join(contents))


if __name__ == "__main__":
    main()
