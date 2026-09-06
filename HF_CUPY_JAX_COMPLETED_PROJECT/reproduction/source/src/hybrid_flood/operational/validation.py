"""Release gates for V1-teacher hybrid checkpoints."""

from __future__ import annotations

import json
from pathlib import Path


def evaluate_release_gates(metrics: dict[str, float]) -> dict[str, object]:
    gates = {
        "wet_depth_rmse_skill": metrics.get("hybrid_reference_wet_depth_rmse_skill_percent", float("-inf")) >= 10.0,
        "csi_not_degraded": metrics.get("critical_success_index_change", float("-inf")) >= 0.0,
        "event_volume": abs(metrics.get("hybrid_event_volume_error_fraction", float("inf"))) <= 0.05,
        "finite_non_negative": bool(metrics.get("finite_non_negative", False)),
        "mass_balance": not bool(metrics.get("mass_balance_regressed", True)),
    }
    return {"passed": all(gates.values()), "gates": gates, "reference_model": "v1", "observationally_validated": False}


def write_release_decision(metrics_path: str | Path, output_path: str | Path) -> Path:
    metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    decision = evaluate_release_gates(metrics)
    target = Path(output_path); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
    return target
