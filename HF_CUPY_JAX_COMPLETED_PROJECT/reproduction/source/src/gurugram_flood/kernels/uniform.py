"""Uniform-grid flood solver MVP for benchmark development."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, pi, sin
import time
from typing import Callable

import numpy as np

from gurugram_flood.kernels.drainage import (
    DrainageNetworkGeometry,
    DrainageNetworkState,
    InletGeometry,
    step_drainage_network_coupling,
)
from gurugram_flood.kernels.equations import (
    hll_shallow_water_normal_flux,
    hydrostatic_reconstruction_flux_pair,
    stable_timestep_s,
)


StageBoundary = Callable[[float], float]
SourceRate = float | np.ndarray | Callable[[float], float | np.ndarray]


@dataclass(frozen=True)
class UniformGridCase:
    """Uniform-grid benchmark case definition."""

    name: str
    nx: int = 60
    ny: int = 30
    length_m: float = 10.0
    width_m: float = 5.0
    final_time_s: float = 10.0
    output_interval_s: float = 0.5
    cfl: float = 0.45
    manning_n: float = 0.1
    gravity_mps2: float = 9.80665
    min_depth_m: float = 1.0e-5
    max_timestep_s: float = 0.02


@dataclass(frozen=True)
class UniformSourceTerms:
    """Rainfall and infiltration source terms for a uniform-grid run."""

    rainfall_rate_mps: SourceRate = 0.0
    infiltration_capacity_mps: SourceRate = 0.0
    active_mask: np.ndarray | None = None


@dataclass(frozen=True)
class UniformDrainageCoupling:
    """Drainage network coupling state for the uniform-grid MVP."""

    geometry: DrainageNetworkGeometry
    state: DrainageNetworkState
    inlet_geometry: InletGeometry
    inlet_capacity_limit_m3ps: float | np.ndarray | None = None


@dataclass
class UniformGridResult:
    """Result arrays and diagnostics from the uniform-grid MVP."""

    case: UniformGridCase
    times_s: np.ndarray
    final_depth_m: np.ndarray
    max_depth_m: np.ndarray
    final_hu_m2ps: np.ndarray
    final_hv_m2ps: np.ndarray
    bed_elevation_m: np.ndarray
    runtime_s: float
    step_count: int
    diagnostics: dict[str, float | int | str]


@dataclass(frozen=True)
class BoundaryVolumeStep:
    """Per-step boundary water volume, positive into the model domain."""

    west_m3: float = 0.0
    east_m3: float = 0.0
    south_m3: float = 0.0
    north_m3: float = 0.0

    @property
    def net_m3(self) -> float:
        return self.west_m3 + self.east_m3 + self.south_m3 + self.north_m3

    @property
    def inflow_m3(self) -> float:
        return sum(max(value, 0.0) for value in self._values())

    @property
    def outflow_m3(self) -> float:
        return sum(max(-value, 0.0) for value in self._values())

    def _values(self) -> tuple[float, float, float, float]:
        return self.west_m3, self.east_m3, self.south_m3, self.north_m3


def build_anuga_runup_case(
    nx: int = 60,
    ny: int = 30,
    final_time_s: float = 10.0,
    output_interval_s: float = 0.5,
) -> tuple[UniformGridCase, np.ndarray, np.ndarray, np.ndarray, StageBoundary]:
    """Build a rasterized version of the official ANUGA simple runup example."""

    case = UniformGridCase(
        name="anuga_simple_runup",
        nx=nx,
        ny=ny,
        final_time_s=final_time_s,
        output_interval_s=output_interval_s,
    )
    x = (np.arange(nx, dtype=float) + 0.5) * case.length_m / nx
    y = (np.arange(ny, dtype=float) + 0.5) * case.width_m / ny
    xx, _ = np.meshgrid(x, y)
    bed_elevation_m = -0.5 * xx
    initial_stage_m = -0.4
    initial_depth_m = np.maximum(initial_stage_m - bed_elevation_m, 0.0)
    manning_n = np.full((ny, nx), case.manning_n, dtype=float)

    def east_stage(t: float) -> float:
        return (0.1 * sin(t * 2.0 * pi) - 0.3) * exp(-t)

    return case, bed_elevation_m, initial_depth_m, manning_n, east_stage


def build_lake_at_rest_case(
    nx: int = 60,
    ny: int = 30,
    final_time_s: float = 2.0,
    output_interval_s: float = 1.0,
) -> tuple[UniformGridCase, np.ndarray, np.ndarray, np.ndarray]:
    """Build a variable-bed still-water benchmark case."""

    case = UniformGridCase(
        name="lake_at_rest",
        nx=nx,
        ny=ny,
        final_time_s=final_time_s,
        output_interval_s=output_interval_s,
        manning_n=0.03,
    )
    x = (np.arange(nx, dtype=float) + 0.5) * case.length_m / nx
    y = (np.arange(ny, dtype=float) + 0.5) * case.width_m / ny
    xx, yy = np.meshgrid(x, y)
    bed_elevation_m = 0.15 * np.sin(xx) * np.cos(yy)
    initial_stage_m = 0.4
    initial_depth_m = np.maximum(initial_stage_m - bed_elevation_m, 0.0)
    manning_n = np.full((ny, nx), case.manning_n, dtype=float)
    return case, bed_elevation_m, initial_depth_m, manning_n


def build_flat_dam_break_case(
    nx: int = 80,
    ny: int = 20,
    final_time_s: float = 1.0,
    output_interval_s: float = 0.5,
) -> tuple[UniformGridCase, np.ndarray, np.ndarray, np.ndarray]:
    """Build a closed flat-bed dam-break benchmark case."""

    case = UniformGridCase(
        name="flat_dam_break",
        nx=nx,
        ny=ny,
        final_time_s=final_time_s,
        output_interval_s=output_interval_s,
        manning_n=0.0,
        max_timestep_s=0.01,
    )
    bed_elevation_m = np.zeros((ny, nx), dtype=float)
    initial_depth_m = np.zeros((ny, nx), dtype=float)
    initial_depth_m[:, : nx // 2] = 1.0
    initial_depth_m[:, nx // 2 :] = 0.1
    manning_n = np.zeros_like(initial_depth_m)
    return case, bed_elevation_m, initial_depth_m, manning_n


def run_uniform_grid_solver(
    case: UniformGridCase,
    bed_elevation_m: np.ndarray,
    initial_depth_m: np.ndarray,
    manning_n: np.ndarray,
    east_stage_boundary: StageBoundary | None = None,
    source_terms: UniformSourceTerms | None = None,
    drainage_coupling: UniformDrainageCoupling | None = None,
) -> UniformGridResult:
    """Run the first explicit finite-volume uniform-grid solver."""

    started = time.perf_counter()
    dx = case.length_m / case.nx
    dy = case.width_m / case.ny
    h = np.maximum(initial_depth_m.astype(float).copy(), 0.0)
    hu = np.zeros_like(h)
    hv = np.zeros_like(h)
    max_depth = h.copy()
    times = [0.0]
    next_output = case.output_interval_s
    t = 0.0
    step_count = 0
    cell_area = dx * dy
    initial_volume_m3 = float(h.sum() * cell_area)
    rainfall_input_m3 = 0.0
    infiltration_loss_m3 = 0.0
    inlet_capture_m3 = 0.0
    drainage_outfall_m3 = 0.0
    drainage_surcharge_m3 = 0.0
    unreturned_overflow_m3 = 0.0
    drainage_state = drainage_coupling.state if drainage_coupling is not None else None
    boundary_west_m3 = 0.0
    boundary_east_m3 = 0.0
    boundary_south_m3 = 0.0
    boundary_north_m3 = 0.0
    boundary_inflow_m3 = 0.0
    boundary_outflow_m3 = 0.0

    while t < case.final_time_s - 1.0e-12:
        dt = stable_timestep_s(
            min(dx, dy),
            h,
            hu,
            hv,
            cfl=case.cfl,
            gravity_mps2=case.gravity_mps2,
            min_depth_m=case.min_depth_m,
        )
        dt = min(dt, case.max_timestep_s, case.final_time_s - t, max(next_output - t, 1.0e-12))

        h, hu, hv, boundary_step = _advance_one_step(
            h=h,
            hu=hu,
            hv=hv,
            bed_elevation_m=bed_elevation_m,
            manning_n=manning_n,
            dx=dx,
            dy=dy,
            dt=dt,
            t=t,
            case=case,
            east_stage_boundary=east_stage_boundary,
        )
        boundary_west_m3 += boundary_step.west_m3
        boundary_east_m3 += boundary_step.east_m3
        boundary_south_m3 += boundary_step.south_m3
        boundary_north_m3 += boundary_step.north_m3
        boundary_inflow_m3 += boundary_step.inflow_m3
        boundary_outflow_m3 += boundary_step.outflow_m3
        rainfall_added, infiltration_removed = _apply_depth_sources(
            h=h,
            source_terms=source_terms,
            t=t,
            dt=dt,
            shape=h.shape,
        )
        rainfall_input_m3 += rainfall_added * cell_area
        infiltration_loss_m3 += infiltration_removed * cell_area
        if drainage_coupling is not None and drainage_state is not None:
            drainage_result = step_drainage_network_coupling(
                h,
                geometry=drainage_coupling.geometry,
                state=drainage_state,
                cell_area_m2=cell_area,
                timestep_s=dt,
                inlet_geometry=drainage_coupling.inlet_geometry,
                inlet_capacity_limit_m3ps=drainage_coupling.inlet_capacity_limit_m3ps,
            )
            h = drainage_result.surface_depth_m.astype(float, copy=False)
            drainage_state = drainage_result.state
            inlet_capture_m3 += drainage_result.inlet_capture_m3
            drainage_outfall_m3 += drainage_result.outfall_discharge_m3
            drainage_surcharge_m3 += drainage_result.surcharge_m3
            unreturned_overflow_m3 += drainage_result.unreturned_overflow_m3
        t += dt
        step_count += 1
        max_depth = np.maximum(max_depth, h)
        if t >= next_output - 1.0e-10 or t >= case.final_time_s - 1.0e-10:
            times.append(float(t))
            next_output += case.output_interval_s

    runtime_s = time.perf_counter() - started
    final_volume_m3 = float(h.sum() * cell_area)
    final_drainage_storage_m3 = float(drainage_state.node_volume_m3.sum()) if drainage_state is not None else 0.0
    initial_drainage_storage_m3 = (
        float(drainage_coupling.state.node_volume_m3.sum()) if drainage_coupling is not None else 0.0
    )
    boundary_net_flux_m3 = boundary_west_m3 + boundary_east_m3 + boundary_south_m3 + boundary_north_m3
    mass_residual_m3 = (
        initial_volume_m3
        + initial_drainage_storage_m3
        + rainfall_input_m3
        + boundary_net_flux_m3
        - infiltration_loss_m3
        - drainage_outfall_m3
        - final_volume_m3
        - final_drainage_storage_m3
    )
    implied_boundary_net_flux_m3 = (
        final_volume_m3
        + final_drainage_storage_m3
        + infiltration_loss_m3
        + drainage_outfall_m3
        - initial_volume_m3
        - initial_drainage_storage_m3
        - rainfall_input_m3
    )
    mass_balance_status = "closed" if east_stage_boundary is None else "open_boundary_tracked"
    diagnostics = {
        "solver": "gpu_flood_uniform_numpy",
        "mass_balance_status": mass_balance_status,
        "initial_volume_m3": initial_volume_m3,
        "volume_m3": final_volume_m3,
        "volume_change_m3": final_volume_m3 - initial_volume_m3,
        "rainfall_input_m3": rainfall_input_m3,
        "infiltration_loss_m3": infiltration_loss_m3,
        "inlet_capture_m3": inlet_capture_m3,
        "drainage_outfall_m3": drainage_outfall_m3,
        "drainage_surcharge_m3": drainage_surcharge_m3,
        "initial_drainage_storage_m3": initial_drainage_storage_m3,
        "final_drainage_storage_m3": final_drainage_storage_m3,
        "unreturned_overflow_m3": unreturned_overflow_m3,
        "boundary_west_net_flux_m3": boundary_west_m3,
        "boundary_east_net_flux_m3": boundary_east_m3,
        "boundary_south_net_flux_m3": boundary_south_m3,
        "boundary_north_net_flux_m3": boundary_north_m3,
        "boundary_net_flux_m3": boundary_net_flux_m3,
        "boundary_inflow_m3": boundary_inflow_m3,
        "boundary_outflow_m3": boundary_outflow_m3,
        "implied_boundary_net_flux_m3": implied_boundary_net_flux_m3,
        "mass_residual_m3": mass_residual_m3,
        "untracked_boundary_volume_m3": 0.0,
        "max_depth_m": float(max_depth.max()),
        "wet_area_m2": float((h > case.min_depth_m).sum() * cell_area),
        "step_count": int(step_count),
        "runtime_s": float(runtime_s),
    }
    return UniformGridResult(
        case=case,
        times_s=np.asarray(times, dtype=np.float32),
        final_depth_m=h.astype(np.float32),
        max_depth_m=max_depth.astype(np.float32),
        final_hu_m2ps=hu.astype(np.float32),
        final_hv_m2ps=hv.astype(np.float32),
        bed_elevation_m=bed_elevation_m.astype(np.float32),
        runtime_s=runtime_s,
        step_count=step_count,
        diagnostics=diagnostics,
    )


def centerline_samples(depth_m: np.ndarray, count: int = 25) -> list[float]:
    """Return evenly spaced depth samples along the horizontal centerline."""

    row = depth_m.shape[0] // 2
    columns = np.linspace(0, depth_m.shape[1] - 1, count).round().astype(int)
    return [float(value) for value in depth_m[row, columns]]


def _apply_depth_sources(
    h: np.ndarray,
    source_terms: UniformSourceTerms | None,
    t: float,
    dt: float,
    shape: tuple[int, int],
) -> tuple[float, float]:
    if source_terms is None:
        return 0.0, 0.0

    active_mask = np.ones(shape, dtype=float) if source_terms.active_mask is None else source_terms.active_mask.astype(float)
    rainfall_depth = np.maximum(_rate_to_array(source_terms.rainfall_rate_mps, t, shape), 0.0) * dt * active_mask
    h += rainfall_depth
    infiltration_capacity_depth = (
        np.maximum(_rate_to_array(source_terms.infiltration_capacity_mps, t, shape), 0.0) * dt * active_mask
    )
    infiltration_depth = np.minimum(h, infiltration_capacity_depth)
    h -= infiltration_depth
    return float(rainfall_depth.sum()), float(infiltration_depth.sum())


def _rate_to_array(rate: SourceRate, t: float, shape: tuple[int, int]) -> np.ndarray:
    value = rate(t) if callable(rate) else rate
    if np.isscalar(value):
        return np.full(shape, float(value), dtype=float)
    array = np.asarray(value, dtype=float)
    if array.shape != shape:
        raise ValueError(f"Source term array shape {array.shape} does not match domain shape {shape}.")
    return array


def _advance_one_step(
    h: np.ndarray,
    hu: np.ndarray,
    hv: np.ndarray,
    bed_elevation_m: np.ndarray,
    manning_n: np.ndarray,
    dx: float,
    dy: float,
    dt: float,
    t: float,
    case: UniformGridCase,
    east_stage_boundary: StageBoundary | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, BoundaryVolumeStep]:
    h_new = h.copy()
    hu_new = hu.copy()
    hv_new = hv.copy()

    # Vertical edges, positive normal in +x.
    left_flux, right_flux = hydrostatic_reconstruction_flux_pair(
        h[:, :-1],
        hu[:, :-1],
        hv[:, :-1],
        bed_elevation_m[:, :-1],
        h[:, 1:],
        hu[:, 1:],
        hv[:, 1:],
        bed_elevation_m[:, 1:],
        normal_x=1.0,
        normal_y=0.0,
        gravity_mps2=case.gravity_mps2,
        min_depth_m=case.min_depth_m,
    )
    h_new[:, :-1] -= dt / dx * left_flux[0]
    hu_new[:, :-1] -= dt / dx * left_flux[1]
    hv_new[:, :-1] -= dt / dx * left_flux[2]
    h_new[:, 1:] += dt / dx * right_flux[0]
    hu_new[:, 1:] += dt / dx * right_flux[1]
    hv_new[:, 1:] += dt / dx * right_flux[2]

    x_boundary = _apply_x_boundaries(
        h_new,
        hu_new,
        hv_new,
        h,
        hu,
        hv,
        bed_elevation_m,
        dt,
        dx,
        dy,
        t,
        case,
        east_stage_boundary,
    )

    # Horizontal edges, positive normal in +y.
    left_flux, right_flux = hydrostatic_reconstruction_flux_pair(
        h[:-1, :],
        hu[:-1, :],
        hv[:-1, :],
        bed_elevation_m[:-1, :],
        h[1:, :],
        hu[1:, :],
        hv[1:, :],
        bed_elevation_m[1:, :],
        normal_x=0.0,
        normal_y=1.0,
        gravity_mps2=case.gravity_mps2,
        min_depth_m=case.min_depth_m,
    )
    h_new[:-1, :] -= dt / dy * left_flux[0]
    hu_new[:-1, :] -= dt / dy * left_flux[1]
    hv_new[:-1, :] -= dt / dy * left_flux[2]
    h_new[1:, :] += dt / dy * right_flux[0]
    hu_new[1:, :] += dt / dy * right_flux[1]
    hv_new[1:, :] += dt / dy * right_flux[2]

    y_boundary = _apply_y_reflective_boundaries(h_new, hu_new, hv_new, h, hu, hv, dt, dx, dy, case)

    h_new = np.maximum(h_new, 0.0)
    wet = h_new > case.min_depth_m
    hu_new = np.where(wet, hu_new, 0.0)
    hv_new = np.where(wet, hv_new, 0.0)

    # Semi-implicit Manning damping after well-balanced hydrostatic fluxes.
    speed = np.sqrt((hu_new / np.maximum(h_new, case.min_depth_m)) ** 2 + (hv_new / np.maximum(h_new, case.min_depth_m)) ** 2)
    damping = 1.0 + dt * case.gravity_mps2 * manning_n**2 * speed / np.maximum(h_new, case.min_depth_m) ** (4.0 / 3.0)
    hu_new = np.where(wet, hu_new / damping, 0.0)
    hv_new = np.where(wet, hv_new / damping, 0.0)

    boundary_volume = BoundaryVolumeStep(
        west_m3=x_boundary.west_m3,
        east_m3=x_boundary.east_m3,
        south_m3=y_boundary.south_m3,
        north_m3=y_boundary.north_m3,
    )
    return h_new, hu_new, hv_new, boundary_volume


def _apply_x_boundaries(
    h_new: np.ndarray,
    hu_new: np.ndarray,
    hv_new: np.ndarray,
    h: np.ndarray,
    hu: np.ndarray,
    hv: np.ndarray,
    bed_elevation_m: np.ndarray,
    dt: float,
    dx: float,
    dy: float,
    t: float,
    case: UniformGridCase,
    east_stage_boundary: StageBoundary | None,
) -> BoundaryVolumeStep:
    flux_h, flux_hu, flux_hv = hll_shallow_water_normal_flux(
        h[:, 0],
        -hu[:, 0],
        hv[:, 0],
        h[:, 0],
        hu[:, 0],
        hv[:, 0],
        normal_x=1.0,
        normal_y=0.0,
        gravity_mps2=case.gravity_mps2,
        min_depth_m=case.min_depth_m,
    )
    h_new[:, 0] += dt / dx * flux_h
    hu_new[:, 0] += dt / dx * flux_hu
    hv_new[:, 0] += dt / dx * flux_hv
    west_m3 = float(np.sum(flux_h) * dy * dt)

    if east_stage_boundary is None:
        h_ghost = h[:, -1]
        hu_ghost = -hu[:, -1]
    else:
        h_ghost = np.maximum(east_stage_boundary(t + dt) - bed_elevation_m[:, -1], 0.0)
        hu_ghost = np.zeros_like(h_ghost)
    flux_h, flux_hu, flux_hv = hll_shallow_water_normal_flux(
        h[:, -1],
        hu[:, -1],
        hv[:, -1],
        h_ghost,
        hu_ghost,
        hv[:, -1],
        normal_x=1.0,
        normal_y=0.0,
        gravity_mps2=case.gravity_mps2,
        min_depth_m=case.min_depth_m,
    )
    h_new[:, -1] -= dt / dx * flux_h
    hu_new[:, -1] -= dt / dx * flux_hu
    hv_new[:, -1] -= dt / dx * flux_hv
    east_m3 = float(-np.sum(flux_h) * dy * dt)
    return BoundaryVolumeStep(west_m3=west_m3, east_m3=east_m3)


def _apply_y_reflective_boundaries(
    h_new: np.ndarray,
    hu_new: np.ndarray,
    hv_new: np.ndarray,
    h: np.ndarray,
    hu: np.ndarray,
    hv: np.ndarray,
    dt: float,
    dx: float,
    dy: float,
    case: UniformGridCase,
) -> BoundaryVolumeStep:
    flux_h, flux_hu, flux_hv = hll_shallow_water_normal_flux(
        h[0, :],
        hu[0, :],
        -hv[0, :],
        h[0, :],
        hu[0, :],
        hv[0, :],
        normal_x=0.0,
        normal_y=1.0,
        gravity_mps2=case.gravity_mps2,
        min_depth_m=case.min_depth_m,
    )
    h_new[0, :] += dt / dy * flux_h
    hu_new[0, :] += dt / dy * flux_hu
    hv_new[0, :] += dt / dy * flux_hv
    south_m3 = float(np.sum(flux_h) * dx * dt)

    flux_h, flux_hu, flux_hv = hll_shallow_water_normal_flux(
        h[-1, :],
        hu[-1, :],
        hv[-1, :],
        h[-1, :],
        hu[-1, :],
        -hv[-1, :],
        normal_x=0.0,
        normal_y=1.0,
        gravity_mps2=case.gravity_mps2,
        min_depth_m=case.min_depth_m,
    )
    h_new[-1, :] -= dt / dy * flux_h
    hu_new[-1, :] -= dt / dy * flux_hu
    hv_new[-1, :] -= dt / dy * flux_hv
    north_m3 = float(-np.sum(flux_h) * dx * dt)
    return BoundaryVolumeStep(south_m3=south_m3, north_m3=north_m3)
