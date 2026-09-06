"""Backend-neutral infiltration kernels for urban flood modeling."""

from __future__ import annotations

from typing import Any

import numpy as np


def _xp_from_array(array: Any):
    module_name = type(array).__module__
    module_root = module_name.split(".", maxsplit=1)[0]
    if module_root == "cupy":
        import cupy as cp  # type: ignore[import-not-found]

        return cp
    return np


def horton_capacity_mps(
    wet_time_s: Any,
    initial_capacity_mmhr: float,
    final_capacity_mmhr: float,
    decay_per_hr: float,
) -> Any:
    """Return Horton infiltration capacity in metres per second."""

    xp = _xp_from_array(wet_time_s)
    wet_time_hr = xp.maximum(wet_time_s, 0.0) / 3600.0
    capacity_mmhr = final_capacity_mmhr + (initial_capacity_mmhr - final_capacity_mmhr) * xp.exp(
        -decay_per_hr * wet_time_hr
    )
    return xp.maximum(capacity_mmhr, 0.0) / 1000.0 / 3600.0


def green_ampt_capacity_mps(
    cumulative_infiltration_m: Any,
    saturated_hydraulic_conductivity_mmhr: float,
    suction_head_m: float,
    moisture_deficit: float,
    min_cumulative_infiltration_m: float = 1.0e-6,
) -> Any:
    """Return Green-Ampt infiltration capacity in metres per second."""

    xp = _xp_from_array(cumulative_infiltration_m)
    ks_mps = saturated_hydraulic_conductivity_mmhr / 1000.0 / 3600.0
    cumulative = xp.maximum(cumulative_infiltration_m, min_cumulative_infiltration_m)
    suction_storage_m = max(suction_head_m, 0.0) * max(moisture_deficit, 0.0)
    return ks_mps * (1.0 + suction_storage_m / cumulative)


def infiltration_depth_for_step(
    available_depth_m: Any,
    capacity_mps: Any,
    timestep_s: float,
    active_fraction: Any = 1.0,
    max_depth_m: float | None = None,
) -> Any:
    """Apply capacity-limited infiltration for one step."""

    xp = _xp_from_array(available_depth_m)
    capacity_depth_m = xp.maximum(capacity_mps, 0.0) * timestep_s * active_fraction
    if max_depth_m is not None:
        capacity_depth_m = xp.minimum(capacity_depth_m, max_depth_m)
    return xp.minimum(xp.maximum(available_depth_m, 0.0), capacity_depth_m)
