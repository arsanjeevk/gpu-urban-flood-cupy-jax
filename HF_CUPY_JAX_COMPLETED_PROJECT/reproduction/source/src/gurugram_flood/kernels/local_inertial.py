"""Local-inertial flood routing kernels for fast inundation modes."""

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


def local_inertial_discharge_update(
    discharge_old_m3ps: Any,
    flow_area_m2: Any,
    hydraulic_radius_m: Any,
    water_surface_slope: Any,
    manning_n: Any,
    timestep_s: float,
    gravity_mps2: float = 9.80665,
    min_flow_area_m2: float = 1.0e-8,
    min_hydraulic_radius_m: float = 1.0e-6,
) -> Any:
    """Update interface discharge using a LISFLOOD-FP-style local-inertial step.

    This neglects convective acceleration. It is intended for fast inundation
    modes, not shock-dominated full-SWE validation cases.
    """

    xp = _xp_from_array(discharge_old_m3ps)
    area = xp.maximum(flow_area_m2, min_flow_area_m2)
    radius = xp.maximum(hydraulic_radius_m, min_hydraulic_radius_m)
    numerator = discharge_old_m3ps - gravity_mps2 * area * timestep_s * water_surface_slope
    denominator = 1.0 + (
        gravity_mps2
        * timestep_s
        * manning_n**2
        * xp.abs(discharge_old_m3ps)
        / (radius ** (4.0 / 3.0) * area)
    )
    return numerator / denominator

