"""Backend-neutral pipe hydraulics kernels for GPU-scale drainage routing."""

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


def circular_conduit_area_m2(depth_m: Any, diameter_m: Any) -> Any:
    """Return wetted area for a circular conduit with optional pressurized depth."""

    xp = _xp_from_array(depth_m)
    diameter = xp.maximum(diameter_m, 1.0e-12)
    radius = 0.5 * diameter
    depth = xp.maximum(depth_m, 0.0)
    open_depth = xp.minimum(depth, diameter)
    theta = 2.0 * xp.arccos(xp.clip((radius - open_depth) / radius, -1.0, 1.0))
    open_area = 0.5 * radius * radius * (theta - xp.sin(theta))
    full_area = 0.25 * np.pi * diameter * diameter
    return xp.where(depth >= diameter, full_area, open_area)


def circular_conduit_wetted_perimeter_m(depth_m: Any, diameter_m: Any) -> Any:
    """Return wetted perimeter for a circular conduit."""

    xp = _xp_from_array(depth_m)
    diameter = xp.maximum(diameter_m, 1.0e-12)
    radius = 0.5 * diameter
    depth = xp.maximum(depth_m, 0.0)
    open_depth = xp.minimum(depth, diameter)
    theta = 2.0 * xp.arccos(xp.clip((radius - open_depth) / radius, -1.0, 1.0))
    full_perimeter = np.pi * diameter
    return xp.where(depth >= diameter, full_perimeter, radius * theta)


def circular_conduit_top_width_m(
    depth_m: Any,
    diameter_m: Any,
    pressurized_wave_speed_mps: float = 30.0,
    gravity_mps2: float = 9.80665,
) -> Any:
    """Return hydraulic top width with a Preissmann slot above crown depth."""

    xp = _xp_from_array(depth_m)
    diameter = xp.maximum(diameter_m, 1.0e-12)
    depth = xp.maximum(depth_m, 0.0)
    open_depth = xp.minimum(depth, diameter)
    open_top_width = 2.0 * xp.sqrt(xp.maximum(open_depth * (diameter - open_depth), 0.0))
    slot_width = preissmann_slot_width_m(diameter, pressurized_wave_speed_mps, gravity_mps2)
    return xp.where(depth >= diameter, slot_width, open_top_width)


def circular_conduit_hydraulic_radius_m(depth_m: Any, diameter_m: Any) -> Any:
    """Return hydraulic radius for a circular conduit."""

    area = circular_conduit_area_m2(depth_m, diameter_m)
    perimeter = circular_conduit_wetted_perimeter_m(depth_m, diameter_m)
    xp = _xp_from_array(area)
    return area / xp.maximum(perimeter, 1.0e-12)


def preissmann_slot_width_m(
    diameter_m: Any,
    pressurized_wave_speed_mps: float = 30.0,
    gravity_mps2: float = 9.80665,
) -> Any:
    """Return Preissmann slot width that approximates a pressure wave speed."""

    xp = _xp_from_array(diameter_m)
    diameter = xp.maximum(diameter_m, 1.0e-12)
    full_area = 0.25 * np.pi * diameter * diameter
    wave_speed = max(float(pressurized_wave_speed_mps), 1.0e-6)
    return gravity_mps2 * full_area / (wave_speed * wave_speed)


def preissmann_slot_area_m2(
    depth_m: Any,
    diameter_m: Any,
    pressurized_wave_speed_mps: float = 30.0,
    gravity_mps2: float = 9.80665,
) -> Any:
    """Return circular conduit area with Preissmann slot storage above crown."""

    xp = _xp_from_array(depth_m)
    base_area = circular_conduit_area_m2(depth_m, diameter_m)
    slot_width = preissmann_slot_width_m(diameter_m, pressurized_wave_speed_mps, gravity_mps2)
    pressure_depth = xp.maximum(depth_m - diameter_m, 0.0)
    return base_area + slot_width * pressure_depth


def conduit_wave_speed_mps(
    area_m2: Any,
    top_width_m: Any,
    gravity_mps2: float = 9.80665,
) -> Any:
    """Return shallow-water wave celerity for conduit area/top-width state."""

    xp = _xp_from_array(area_m2)
    return xp.sqrt(gravity_mps2 * xp.maximum(area_m2, 0.0) / xp.maximum(top_width_m, 1.0e-12))


def saint_venant_link_flow_update(
    link_flow_m3ps: Any,
    from_head_m: Any,
    to_head_m: Any,
    link_length_m: Any,
    flow_area_m2: Any,
    hydraulic_radius_m: Any,
    manning_n: Any,
    timestep_s: float,
    gravity_mps2: float = 9.80665,
    minor_loss_coefficient: Any = 0.0,
    flow_capacity_m3ps: Any | None = None,
) -> Any:
    """Semi-implicit Saint-Venant momentum update for independent pipe links.

    This is backend-neutral and GPU-scale: all inputs may be large NumPy or
    CuPy arrays. Continuity/node scatter remains handled outside this
    pure per-link kernel.
    """

    xp = _xp_from_array(link_flow_m3ps)
    length = xp.maximum(link_length_m, 1.0e-6)
    area = xp.maximum(flow_area_m2, 1.0e-12)
    radius = xp.maximum(hydraulic_radius_m, 1.0e-12)
    flow = link_flow_m3ps
    slope = (from_head_m - to_head_m) / length
    friction = gravity_mps2 * xp.maximum(manning_n, 0.0) ** 2 * xp.abs(flow) / (radius ** (4.0 / 3.0) * area)
    minor_loss = xp.maximum(minor_loss_coefficient, 0.0) * xp.abs(flow) / (2.0 * length * area)
    numerator = flow + gravity_mps2 * area * timestep_s * slope
    updated = numerator / (1.0 + timestep_s * (friction + minor_loss))
    if flow_capacity_m3ps is not None:
        capacity = xp.maximum(flow_capacity_m3ps, 0.0)
        updated = xp.clip(updated, -capacity, capacity)
    return updated

