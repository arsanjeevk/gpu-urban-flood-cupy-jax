"""Configuration objects for the GPU-first flood solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


BackendName = Literal["auto", "cupy"]
PrecisionName = Literal["float32", "float64"]
InfiltrationModel = Literal["off", "horton", "green_ampt"]
AdaptiveGridMode = Literal["uniform", "static_patches", "static_quadtree"]
BuildingMode = Literal["solid", "high_obstruction", "porosity"]
DrainageMode = Literal["off", "capacity_sink", "graph_network", "dynamic_wave"]
SurfaceSolverMode = Literal["full_swe", "local_inertial", "diffusive", "screening"]


@dataclass(frozen=True)
class BackendConfig:
    """Numerical backend and precision selection."""

    name: BackendName = "auto"
    precision: PrecisionName = "float32"

    def __post_init__(self) -> None:
        if self.name not in {"auto", "cupy"}:
            raise ValueError(f"Unsupported backend: {self.name}")
        if self.precision not in {"float32", "float64"}:
            raise ValueError(f"Unsupported precision: {self.precision}")


@dataclass(frozen=True)
class SurfacePhysicsConfig:
    """Core 2D shallow-water numerical parameters."""

    solver_mode: SurfaceSolverMode = "full_swe"
    gravity_mps2: float = 9.80665
    cfl: float = 0.45
    min_depth_m: float = 1.0e-5
    dry_tolerance_m: float = 1.0e-6
    manning_default_n: float = 0.035
    manning_road_n: float = 0.018
    manning_building_n: float = 0.20

    def __post_init__(self) -> None:
        if self.solver_mode not in {"full_swe", "local_inertial", "diffusive", "screening"}:
            raise ValueError(f"Unsupported surface solver mode: {self.solver_mode}")
        if self.gravity_mps2 <= 0.0:
            raise ValueError("gravity_mps2 must be positive.")
        if not 0.0 < self.cfl <= 1.0:
            raise ValueError("cfl must be in the interval (0, 1].")
        if self.min_depth_m <= 0.0:
            raise ValueError("min_depth_m must be positive.")


@dataclass(frozen=True)
class InfiltrationConfig:
    """Infiltration model parameters for pervious urban cells."""

    model: InfiltrationModel = "green_ampt"
    horton_initial_mmhr: float = 50.0
    horton_final_mmhr: float = 6.0
    horton_decay_per_hr: float = 4.0
    green_ampt_ks_mmhr: float = 6.5
    green_ampt_suction_head_m: float = 0.16
    green_ampt_moisture_deficit: float = 0.16
    max_infiltration_depth_m: float | None = None

    def __post_init__(self) -> None:
        if self.model not in {"off", "horton", "green_ampt"}:
            raise ValueError(f"Unsupported infiltration model: {self.model}")
        if self.horton_initial_mmhr < self.horton_final_mmhr:
            raise ValueError("horton_initial_mmhr must be greater than or equal to horton_final_mmhr.")
        if self.green_ampt_ks_mmhr < 0.0:
            raise ValueError("green_ampt_ks_mmhr must be non-negative.")
        if not 0.0 <= self.green_ampt_moisture_deficit <= 1.0:
            raise ValueError("green_ampt_moisture_deficit must be in [0, 1].")


@dataclass(frozen=True)
class UrbanLayerConfig:
    """Rules for converting urban evidence layers into hydraulic fields."""

    building_mode: BuildingMode = "solid"
    road_refinement: bool = True
    inlet_refinement: bool = True
    building_refinement: bool = True
    use_lulc_manning: bool = True
    use_lulc_infiltration: bool = True
    impervious_infiltration_scale: float = 0.05

    def __post_init__(self) -> None:
        if self.building_mode not in {"solid", "high_obstruction", "porosity"}:
            raise ValueError(f"Unsupported building mode: {self.building_mode}")
        if not 0.0 <= self.impervious_infiltration_scale <= 1.0:
            raise ValueError("impervious_infiltration_scale must be in [0, 1].")


@dataclass(frozen=True)
class AdaptiveGridConfig:
    """Static adaptive grid strategy for GPU-friendly topology."""

    mode: AdaptiveGridMode = "static_patches"
    base_cell_size_m: float = 5.0
    max_refinement_level: int = 2
    road_level: int = 1
    inlet_level: int = 2
    building_edge_level: int = 1
    low_point_level: int = 2

    def __post_init__(self) -> None:
        if self.mode not in {"uniform", "static_patches", "static_quadtree"}:
            raise ValueError(f"Unsupported adaptive grid mode: {self.mode}")
        if self.base_cell_size_m <= 0.0:
            raise ValueError("base_cell_size_m must be positive.")
        if self.max_refinement_level < 0:
            raise ValueError("max_refinement_level must be non-negative.")
        for field_name in ("road_level", "inlet_level", "building_edge_level", "low_point_level"):
            level = getattr(self, field_name)
            if level < 0 or level > self.max_refinement_level:
                raise ValueError(f"{field_name} must be between 0 and max_refinement_level.")


@dataclass(frozen=True)
class DrainageCouplingConfig:
    """Drainage and 1D-2D coupling strategy."""

    mode: DrainageMode = "graph_network"
    inlet_weir_coefficient: float = 0.45
    inlet_orifice_coefficient: float = 0.62
    default_inlet_area_m2: float = 0.04
    default_inlet_perimeter_m: float = 0.8
    node_storage_surcharge_enabled: bool = True

    def __post_init__(self) -> None:
        if self.mode not in {"off", "capacity_sink", "graph_network", "dynamic_wave"}:
            raise ValueError(f"Unsupported drainage mode: {self.mode}")
        if self.inlet_weir_coefficient < 0.0:
            raise ValueError("inlet_weir_coefficient must be non-negative.")
        if self.inlet_orifice_coefficient < 0.0:
            raise ValueError("inlet_orifice_coefficient must be non-negative.")
        if self.default_inlet_area_m2 < 0.0:
            raise ValueError("default_inlet_area_m2 must be non-negative.")


@dataclass(frozen=True)
class GpuFloodConfig:
    """Top-level GPU flood solver configuration."""

    backend: BackendConfig = BackendConfig()
    surface: SurfacePhysicsConfig = SurfacePhysicsConfig()
    infiltration: InfiltrationConfig = InfiltrationConfig()
    urban_layers: UrbanLayerConfig = UrbanLayerConfig()
    adaptive_grid: AdaptiveGridConfig = AdaptiveGridConfig()
    drainage: DrainageCouplingConfig = DrainageCouplingConfig()
    output_interval_s: float = 300.0
    max_runtime_s: float = 6.0 * 3600.0

    def __post_init__(self) -> None:
        if self.output_interval_s <= 0.0:
            raise ValueError("output_interval_s must be positive.")
        if self.max_runtime_s <= 0.0:
            raise ValueError("max_runtime_s must be positive.")
