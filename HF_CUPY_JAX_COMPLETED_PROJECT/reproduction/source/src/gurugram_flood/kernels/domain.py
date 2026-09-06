"""Urban raster domain compiler for the GPU flood engine."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gurugram_flood.kernels.config import InfiltrationConfig, SurfacePhysicsConfig, UrbanLayerConfig


@dataclass(frozen=True)
class UrbanRasterDomain:
    """Raster evidence layers for an urban flood domain."""

    bed_elevation_m: np.ndarray
    building_mask: np.ndarray
    road_mask: np.ndarray
    valid_mask: np.ndarray | None = None
    pervious_fraction: np.ndarray | None = None
    impervious_fraction: np.ndarray | None = None
    lulc_id: np.ndarray | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.bed_elevation_m.shape


@dataclass(frozen=True)
class HydraulicFields:
    """Compiled hydraulic arrays consumed by solvers."""

    active_mask: np.ndarray
    solid_mask: np.ndarray
    manning_n: np.ndarray
    infiltration_capacity_mps: np.ndarray
    pervious_fraction: np.ndarray
    impervious_fraction: np.ndarray


def compile_hydraulic_fields(
    domain: UrbanRasterDomain,
    surface: SurfacePhysicsConfig | None = None,
    urban: UrbanLayerConfig | None = None,
    infiltration: InfiltrationConfig | None = None,
) -> HydraulicFields:
    """Compile urban evidence layers into solver-ready hydraulic fields."""

    surface_config = surface or SurfacePhysicsConfig()
    urban_config = urban or UrbanLayerConfig()
    infiltration_config = infiltration or InfiltrationConfig()
    _validate_domain_shapes(domain)

    valid_mask = np.ones(domain.shape, dtype=bool) if domain.valid_mask is None else domain.valid_mask.astype(bool)
    building_mask = domain.building_mask.astype(bool) & valid_mask
    road_mask = domain.road_mask.astype(bool) & valid_mask & ~building_mask
    if urban_config.building_mode == "solid":
        solid_mask = building_mask | ~valid_mask
    else:
        solid_mask = ~valid_mask
    active_mask = ~solid_mask

    manning_n = np.full(domain.shape, surface_config.manning_default_n, dtype=np.float32)
    manning_n[road_mask] = surface_config.manning_road_n
    if urban_config.building_mode != "solid":
        manning_n[building_mask] = surface_config.manning_building_n

    if domain.pervious_fraction is None:
        pervious_fraction = np.where(road_mask | building_mask, 0.0, 1.0).astype(np.float32)
    else:
        pervious_fraction = np.clip(domain.pervious_fraction.astype(np.float32), 0.0, 1.0)
    if domain.impervious_fraction is None:
        impervious_fraction = np.clip(1.0 - pervious_fraction, 0.0, 1.0).astype(np.float32)
    else:
        impervious_fraction = np.clip(domain.impervious_fraction.astype(np.float32), 0.0, 1.0)

    infiltration_capacity_mps = _infiltration_capacity_mps(
        infiltration_config,
        urban_config,
        pervious_fraction=pervious_fraction,
        impervious_fraction=impervious_fraction,
    ).astype(np.float32)
    infiltration_capacity_mps[solid_mask] = 0.0

    return HydraulicFields(
        active_mask=active_mask,
        solid_mask=solid_mask,
        manning_n=manning_n,
        infiltration_capacity_mps=infiltration_capacity_mps,
        pervious_fraction=pervious_fraction,
        impervious_fraction=impervious_fraction,
    )


def _infiltration_capacity_mps(
    infiltration: InfiltrationConfig,
    urban: UrbanLayerConfig,
    pervious_fraction: np.ndarray,
    impervious_fraction: np.ndarray,
) -> np.ndarray:
    if infiltration.model == "off":
        base_mmhr = np.zeros_like(pervious_fraction, dtype=np.float32)
    elif infiltration.model == "horton":
        base_mmhr = np.full_like(pervious_fraction, infiltration.horton_initial_mmhr, dtype=np.float32)
    else:
        base_mmhr = np.full_like(pervious_fraction, infiltration.green_ampt_ks_mmhr, dtype=np.float32)
    effective_mmhr = base_mmhr * (
        pervious_fraction + urban.impervious_infiltration_scale * impervious_fraction
    )
    return effective_mmhr / 1000.0 / 3600.0


def _validate_domain_shapes(domain: UrbanRasterDomain) -> None:
    shape = domain.shape
    for name in (
        "building_mask",
        "road_mask",
        "valid_mask",
        "pervious_fraction",
        "impervious_fraction",
        "lulc_id",
    ):
        value = getattr(domain, name)
        if value is not None and value.shape != shape:
            raise ValueError(f"{name} shape {value.shape} does not match bed_elevation_m shape {shape}.")

