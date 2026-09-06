"""Backend-neutral shallow-water equation kernels."""

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


def hll_shallow_water_normal_flux(
    h_left: Any,
    hu_left: Any,
    hv_left: Any,
    h_right: Any,
    hu_right: Any,
    hv_right: Any,
    normal_x: Any,
    normal_y: Any,
    gravity_mps2: float = 9.80665,
    min_depth_m: float = 1.0e-5,
) -> tuple[Any, Any, Any]:
    """Compute HLL flux for the 2D shallow-water equations across an edge.

    The returned tuple is the normal flux for `[h, hu, hv]`. Inputs may be
    scalars, NumPy arrays, or CuPy arrays.
    """

    xp = _xp_from_array(h_left)
    h_l = xp.maximum(h_left, 0.0)
    h_r = xp.maximum(h_right, 0.0)
    wet_l = h_l > min_depth_m
    wet_r = h_r > min_depth_m

    u_l = xp.where(wet_l, hu_left / xp.maximum(h_l, min_depth_m), 0.0)
    v_l = xp.where(wet_l, hv_left / xp.maximum(h_l, min_depth_m), 0.0)
    u_r = xp.where(wet_r, hu_right / xp.maximum(h_r, min_depth_m), 0.0)
    v_r = xp.where(wet_r, hv_right / xp.maximum(h_r, min_depth_m), 0.0)

    un_l = u_l * normal_x + v_l * normal_y
    un_r = u_r * normal_x + v_r * normal_y
    c_l = xp.sqrt(gravity_mps2 * h_l)
    c_r = xp.sqrt(gravity_mps2 * h_r)

    pressure_l = 0.5 * gravity_mps2 * h_l * h_l
    pressure_r = 0.5 * gravity_mps2 * h_r * h_r

    flux_h_l = h_l * un_l
    flux_hu_l = hu_left * un_l + pressure_l * normal_x
    flux_hv_l = hv_left * un_l + pressure_l * normal_y
    flux_h_r = h_r * un_r
    flux_hu_r = hu_right * un_r + pressure_r * normal_x
    flux_hv_r = hv_right * un_r + pressure_r * normal_y

    speed_l = xp.minimum(un_l - c_l, un_r - c_r)
    speed_r = xp.maximum(un_l + c_l, un_r + c_r)
    denom = xp.where(xp.abs(speed_r - speed_l) > 1.0e-12, speed_r - speed_l, 1.0)

    hll_h = (speed_r * flux_h_l - speed_l * flux_h_r + speed_l * speed_r * (h_r - h_l)) / denom
    hll_hu = (speed_r * flux_hu_l - speed_l * flux_hu_r + speed_l * speed_r * (hu_right - hu_left)) / denom
    hll_hv = (speed_r * flux_hv_l - speed_l * flux_hv_r + speed_l * speed_r * (hv_right - hv_left)) / denom

    flux_h = xp.where(speed_l >= 0.0, flux_h_l, xp.where(speed_r <= 0.0, flux_h_r, hll_h))
    flux_hu = xp.where(speed_l >= 0.0, flux_hu_l, xp.where(speed_r <= 0.0, flux_hu_r, hll_hu))
    flux_hv = xp.where(speed_l >= 0.0, flux_hv_l, xp.where(speed_r <= 0.0, flux_hv_r, hll_hv))

    dry_edge = ~(wet_l | wet_r)
    return (
        xp.where(dry_edge, 0.0, flux_h),
        xp.where(dry_edge, 0.0, flux_hu),
        xp.where(dry_edge, 0.0, flux_hv),
    )


def hydrostatic_reconstruction_flux_pair(
    h_left: Any,
    hu_left: Any,
    hv_left: Any,
    bed_left: Any,
    h_right: Any,
    hu_right: Any,
    hv_right: Any,
    bed_right: Any,
    normal_x: Any,
    normal_y: Any,
    gravity_mps2: float = 9.80665,
    min_depth_m: float = 1.0e-5,
) -> tuple[tuple[Any, Any, Any], tuple[Any, Any, Any]]:
    """Return well-balanced hydrostatic fluxes for left and right cells.

    The first tuple is the flux contribution used by the left cell on an edge
    whose normal points from left to right. The second tuple is the matching
    contribution used by the right cell. This follows the hydrostatic
    reconstruction idea of preserving a lake at rest over variable bed.
    """

    xp = _xp_from_array(h_left)
    h_l = xp.maximum(h_left, 0.0)
    h_r = xp.maximum(h_right, 0.0)
    eta_l = h_l + bed_left
    eta_r = h_r + bed_right
    bed_star = xp.maximum(bed_left, bed_right)
    h_l_star = xp.maximum(eta_l - bed_star, 0.0)
    h_r_star = xp.maximum(eta_r - bed_star, 0.0)

    wet_l = h_l > min_depth_m
    wet_r = h_r > min_depth_m
    u_l = xp.where(wet_l, hu_left / xp.maximum(h_l, min_depth_m), 0.0)
    v_l = xp.where(wet_l, hv_left / xp.maximum(h_l, min_depth_m), 0.0)
    u_r = xp.where(wet_r, hu_right / xp.maximum(h_r, min_depth_m), 0.0)
    v_r = xp.where(wet_r, hv_right / xp.maximum(h_r, min_depth_m), 0.0)

    hu_l_star = u_l * h_l_star
    hv_l_star = v_l * h_l_star
    hu_r_star = u_r * h_r_star
    hv_r_star = v_r * h_r_star

    flux_h, flux_hu, flux_hv = hll_shallow_water_normal_flux(
        h_l_star,
        hu_l_star,
        hv_l_star,
        h_r_star,
        hu_r_star,
        hv_r_star,
        normal_x=normal_x,
        normal_y=normal_y,
        gravity_mps2=gravity_mps2,
        min_depth_m=min_depth_m,
    )

    pressure_correction_l = 0.5 * gravity_mps2 * (h_l * h_l - h_l_star * h_l_star)
    pressure_correction_r = 0.5 * gravity_mps2 * (h_r * h_r - h_r_star * h_r_star)
    left_flux = (
        flux_h,
        flux_hu + pressure_correction_l * normal_x,
        flux_hv + pressure_correction_l * normal_y,
    )
    right_flux = (
        flux_h,
        flux_hu + pressure_correction_r * normal_x,
        flux_hv + pressure_correction_r * normal_y,
    )
    return left_flux, right_flux


def stable_timestep_s(
    cell_length_m: Any,
    h: Any,
    hu: Any,
    hv: Any,
    cfl: float = 0.45,
    gravity_mps2: float = 9.80665,
    min_depth_m: float = 1.0e-5,
) -> float:
    """Return a global explicit timestep from the shallow-water CFL condition."""

    xp = _xp_from_array(h)
    h_safe = xp.maximum(h, min_depth_m)
    speed = xp.sqrt((hu / h_safe) ** 2 + (hv / h_safe) ** 2) + xp.sqrt(gravity_mps2 * h_safe)
    local_dt = cfl * cell_length_m / xp.maximum(speed, 1.0e-12)
    finite_dt = xp.where(xp.isfinite(local_dt), local_dt, xp.inf)
    dt = xp.min(finite_dt)
    if hasattr(dt, "get"):
        dt = dt.get()
    if hasattr(dt, "item"):
        dt = dt.item()
    if not np.isfinite(dt):
        raise ValueError("No finite cells are available for timestep calculation.")
    return float(dt)


def stable_local_inertial_timestep_s(
    cell_length_m: Any,
    h: Any,
    cfl: float = 0.45,
    gravity_mps2: float = 9.80665,
    min_depth_m: float = 1.0e-5,
) -> float:
    """Return a global explicit timestep using gravity-wave speed only.

    For the local-inertial approximation, stability requires only the
    gravity-wave celerity sqrt(g·h) — not the full SWE speed |u| + sqrt(g·h).
    This yields larger timesteps than the full-SWE CFL, reducing step count.
    """

    xp = _xp_from_array(h)
    h_safe = xp.maximum(h, min_depth_m)
    c = xp.sqrt(gravity_mps2 * h_safe)
    local_dt = cfl * cell_length_m / xp.maximum(c, 1.0e-12)
    finite_dt = xp.where(xp.isfinite(local_dt), local_dt, xp.inf)
    dt = xp.min(finite_dt)
    if hasattr(dt, "get"):
        dt = dt.get()
    if hasattr(dt, "item"):
        dt = dt.item()
    if not np.isfinite(dt):
        raise ValueError("No finite cells are available for timestep calculation.")
    return float(dt)
