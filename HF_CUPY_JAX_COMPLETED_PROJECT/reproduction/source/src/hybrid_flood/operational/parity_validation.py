"""Reproducible validation of the JAX V1-parity solver against frozen V1 output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .errors import ConfigurationError


DEPTH_THRESHOLD_M = 0.05
LEDGER_TERMS = {
    "rainfall_input_m3": "rainfall_input_m3",
    "infiltration_loss_m3": "infiltration_loss_m3",
    "boundary_net_flux_m3": "boundary_net_flux_m3",
    "external_boundary_inflow_m3": "external_boundary_inflow_m3",
    "outfall_discharge_m3": "outfall_discharge_m3",
    "recharge_capture_m3": "surface_recharge_capture_m3",
    "recharge_loss_m3": "surface_recharge_loss_m3",
}


def _read_report(directory: Path) -> dict[str, Any]:
    path = directory / "run_report.json"
    if not path.is_file():
        raise ConfigurationError(f"Missing run report: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(candidate: float, reference: float) -> float:
    return (candidate - reference) / max(abs(reference), 1.0)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def validate_v1_parity_run(
    *,
    reference_dir: str | Path,
    candidate_dir: str | Path,
    threshold_m: float = DEPTH_THRESHOLD_M,
) -> dict[str, Any]:
    """Compare aligned exact-topology snapshots and apply V1-parity gates."""

    reference = Path(reference_dir).resolve()
    candidate = Path(candidate_dir).resolve()
    reference_npz = reference / "flood_simulation_outputs.npz"
    candidate_npz = candidate / "flood_simulation_outputs.npz"
    if not reference_npz.is_file() or not candidate_npz.is_file():
        raise ConfigurationError("Reference and candidate must both contain flood_simulation_outputs.npz.")
    reference_report = _read_report(reference)
    candidate_report = _read_report(candidate)

    with np.load(reference_npz, allow_pickle=False) as ref, np.load(
        candidate_npz, allow_pickle=False
    ) as cand:
        required = {"snapshot_times_s", "depth_snapshots_m"}
        if not required.issubset(ref.files) or not required.issubset(cand.files):
            raise ConfigurationError("Parity archives require snapshot_times_s and depth_snapshots_m.")
        ref_times = ref["snapshot_times_s"].astype(np.float64)
        cand_times = cand["snapshot_times_s"].astype(np.float64)
        ref_depth = ref["depth_snapshots_m"].astype(np.float32)
        cand_depth = cand["depth_snapshots_m"].astype(np.float32)
        if ref_depth.shape != cand_depth.shape or ref_times.shape != cand_times.shape:
            raise ConfigurationError(
                f"Parity shape mismatch: reference {ref_depth.shape}/{ref_times.shape}, "
                f"candidate {cand_depth.shape}/{cand_times.shape}."
            )
        if "active_surface_mask" not in cand.files:
            raise ConfigurationError("Candidate archive lacks active_surface_mask.")
        valid = cand["active_surface_mask"].astype(bool)
        if "building_mask" in cand.files:
            valid &= ~cand["building_mask"].astype(bool)
        if valid.shape != ref_depth.shape[1:]:
            raise ConfigurationError("Candidate domain mask does not match triangle snapshots.")
        if "centroids_world" not in cand.files:
            raise ConfigurationError("Candidate archive lacks exact-topology centroids_world.")

        # The native topology stores cell area in the immutable V1 asset.  For the
        # validation volume gate, recover it from the candidate report source path.
        topology_path = Path(
            candidate_report["input_report"]["assets"]["assets"]["topology"]["path"]
        )
        with np.load(topology_path, allow_pickle=False) as topology:
            area = topology["surface_cell_area_m2"].astype(np.float64)
        if area.shape != valid.shape:
            raise ConfigurationError("Production triangle areas do not match parity output.")

        finite_non_negative = bool(
            np.isfinite(cand_depth).all() and np.min(cand_depth) >= -1.0e-7
        )
        hourly: list[dict[str, float | int]] = []
        for index, (ref_time, cand_time) in enumerate(zip(ref_times, cand_times, strict=True)):
            reference_values = ref_depth[index, valid].astype(np.float64)
            candidate_values = cand_depth[index, valid].astype(np.float64)
            difference = candidate_values - reference_values
            ref_wet = reference_values >= threshold_m
            cand_wet = candidate_values >= threshold_m
            union = ref_wet | cand_wet
            intersection = ref_wet & cand_wet
            reference_volume = float(np.sum(ref_depth[index].astype(np.float64) * area))
            candidate_volume = float(np.sum(cand_depth[index].astype(np.float64) * area))
            hourly.append({
                "snapshot_index": index,
                "reference_time_s": float(ref_time),
                "candidate_time_s": float(cand_time),
                "time_delta_s": float(cand_time - ref_time),
                "valid_depth_rmse_m": float(np.sqrt(np.mean(difference**2))),
                "reference_wet_depth_rmse_m": float(
                    np.sqrt(np.mean(difference[ref_wet] ** 2)) if ref_wet.any() else 0.0
                ),
                "depth_absolute_error_p99_m": float(np.percentile(np.abs(difference), 99)),
                "flood_extent_csi": float(intersection.sum() / max(int(union.sum()), 1)),
                "reference_wet_triangle_count": int(ref_wet.sum()),
                "candidate_wet_triangle_count": int(cand_wet.sum()),
                "reference_surface_volume_m3": reference_volume,
                "candidate_surface_volume_m3": candidate_volume,
                "surface_volume_error_fraction": _relative(candidate_volume, reference_volume),
                "reference_max_depth_m": float(np.max(reference_values)),
                "candidate_max_depth_m": float(np.max(candidate_values)),
            })

        node_metrics: dict[str, float] = {}
        if "final_node_volume_m3" in ref.files and "final_node_volume_m3" in cand.files:
            node_difference = (
                cand["final_node_volume_m3"].astype(np.float64)
                - ref["final_node_volume_m3"].astype(np.float64)
            )
            node_metrics = {
                "final_node_volume_rmse_m3": float(np.sqrt(np.mean(node_difference**2))),
                "final_node_volume_mae_m3": float(np.mean(np.abs(node_difference))),
                "final_node_volume_max_absolute_error_m3": float(np.max(np.abs(node_difference))),
            }
        link_metrics: dict[str, float] = {}
        if "final_link_flow_m3ps" in ref.files and "final_link_flow_m3ps" in cand.files:
            link_difference = (
                cand["final_link_flow_m3ps"].astype(np.float64)
                - ref["final_link_flow_m3ps"].astype(np.float64)
            )
            link_metrics = {
                "final_link_flow_rmse_m3ps": float(np.sqrt(np.mean(link_difference**2))),
                "final_link_flow_mae_m3ps": float(np.mean(np.abs(link_difference))),
                "final_link_flow_max_absolute_error_m3ps": float(np.max(np.abs(link_difference))),
            }

    ledger_comparison = {}
    for candidate_term, reference_term in LEDGER_TERMS.items():
        if reference_term in reference_report and candidate_term in candidate_report:
            ref_value = float(reference_report[reference_term])
            cand_value = float(candidate_report[candidate_term])
            ledger_comparison[candidate_term] = {
                "reference_field": reference_term,
                "reference": ref_value,
                "candidate": cand_value,
                "relative_error_fraction": _relative(cand_value, ref_value),
            }

    maximum_time_delta = max(abs(float(row["time_delta_s"])) for row in hourly)
    maximum_rmse = max(float(row["valid_depth_rmse_m"]) for row in hourly)
    minimum_csi = min(float(row["flood_extent_csi"]) for row in hourly)
    maximum_volume_error = max(abs(float(row["surface_volume_error_fraction"])) for row in hourly)
    final = hourly[-1]
    overall_reference_max = float(reference_report["max_depth_m"])
    overall_candidate_max = float(candidate_report["max_depth_m"])
    maximum_ledger_error = max(
        (abs(float(value["relative_error_fraction"])) for value in ledger_comparison.values()),
        default=float("inf"),
    )
    reference_mass_ratio = abs(float(reference_report.get("mass_residual_ratio", float("inf"))))
    candidate_mass_ratio = abs(
        float(candidate_report.get("v1_compatible_mass_residual_ratio", float("inf")))
    )
    gates = {
        "finite_non_negative_depth": finite_non_negative,
        "snapshot_alignment_le_1s": maximum_time_delta <= 1.0,
        "all_hour_depth_rmse_le_0_05m": maximum_rmse <= 0.05,
        "all_hour_csi_ge_0_95": minimum_csi >= 0.95,
        "all_hour_volume_error_le_2pct": maximum_volume_error <= 0.02,
        "overall_max_depth_error_le_5pct": abs(
            _relative(overall_candidate_max, overall_reference_max)
        ) <= 0.05,
        "source_sink_ledger_error_le_2pct": maximum_ledger_error <= 0.02,
        "mass_residual_regression_le_0_1pct": (
            candidate_mass_ratio - reference_mass_ratio
        ) <= 0.001,
    }
    passed = all(gates.values())
    payload: dict[str, Any] = {
        "schema_version": 1,
        "validation_status": "v1_parity" if passed else "v1_parity_failed",
        "passed": passed,
        "reference_model": "v1_cuda_reference",
        "observationally_validated": False,
        "warning": "Numerical parity with V1 is not validation against observed flooding.",
        "reference_dir": str(reference),
        "candidate_dir": str(candidate),
        "threshold_m": threshold_m,
        "gates": gates,
        "summary": {
            "maximum_snapshot_time_delta_s": maximum_time_delta,
            "maximum_hourly_valid_depth_rmse_m": maximum_rmse,
            "minimum_hourly_flood_extent_csi": minimum_csi,
            "maximum_hourly_surface_volume_error_fraction": maximum_volume_error,
            "final_valid_depth_rmse_m": final["valid_depth_rmse_m"],
            "final_flood_extent_csi": final["flood_extent_csi"],
            "final_surface_volume_error_fraction": final["surface_volume_error_fraction"],
            "overall_reference_max_depth_m": overall_reference_max,
            "overall_candidate_max_depth_m": overall_candidate_max,
            "maximum_source_sink_ledger_error_fraction": maximum_ledger_error,
            "reference_mass_residual_ratio": reference_mass_ratio,
            "candidate_v1_compatible_mass_residual_ratio": candidate_mass_ratio,
        },
        "hourly": hourly,
        "ledger_comparison": ledger_comparison,
        "drainage_state": {**node_metrics, **link_metrics},
    }
    validation_path = candidate / "parity_validation.json"
    _atomic_json(validation_path, payload)
    candidate_report.update({
        "validation_status": payload["validation_status"],
        "reference_model": "v1_cuda_reference",
        "observationally_validated": False,
        "drainage_mode": "v1_full_dynamic_network",
        "boundary_condition": "v1_code4_outflow_only_with_historical_external_inflow",
        "parity_validation": str(validation_path),
        "parity_validation_summary": payload["summary"],
    })
    _atomic_json(candidate / "run_report.json", candidate_report)
    manifest = {
        "schema_version": 1,
        "reference_variant": "v1_cuda_reference",
        "candidate_variant": "jax_v1_parity",
        "reference_artifacts": {
            "run_report": str(reference / "run_report.json"),
            "native_npz": str(reference_npz),
        },
        "candidate_artifacts": {
            "run_report": str(candidate / "run_report.json"),
            "native_npz": str(candidate_npz),
            "validation": str(validation_path),
        },
        "validation_status": payload["validation_status"],
        "observationally_validated": False,
        "summary": payload["summary"],
    }
    _atomic_json(candidate / "comparison_manifest.json", manifest)
    return payload
