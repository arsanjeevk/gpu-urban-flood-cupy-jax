"""Static refinement planning for GPU-friendly urban meshes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gurugram_flood.kernels.config import AdaptiveGridConfig


@dataclass(frozen=True)
class StaticRefinementPlan:
    """Per-base-cell static refinement levels."""

    levels: np.ndarray
    max_level: int
    base_cell_size_m: float

    @property
    def refined_cell_size_m(self) -> np.ndarray:
        return self.base_cell_size_m / (2.0 ** self.levels)


def build_static_refinement_plan(
    shape: tuple[int, int],
    config: AdaptiveGridConfig | None = None,
    road_mask: np.ndarray | None = None,
    inlet_mask: np.ndarray | None = None,
    building_edge_mask: np.ndarray | None = None,
    low_point_mask: np.ndarray | None = None,
) -> StaticRefinementPlan:
    """Build a static refinement-level raster from urban priority masks."""

    resolved_config = config or AdaptiveGridConfig()
    levels = np.zeros(shape, dtype=np.uint8)
    if resolved_config.mode == "uniform":
        return StaticRefinementPlan(
            levels=levels,
            max_level=0,
            base_cell_size_m=resolved_config.base_cell_size_m,
        )

    _raise_to_level(levels, road_mask, resolved_config.road_level)
    _raise_to_level(levels, building_edge_mask, resolved_config.building_edge_level)
    _raise_to_level(levels, low_point_mask, resolved_config.low_point_level)
    _raise_to_level(levels, inlet_mask, resolved_config.inlet_level)
    levels = np.minimum(levels, resolved_config.max_refinement_level).astype(np.uint8)
    return StaticRefinementPlan(
        levels=levels,
        max_level=int(levels.max()) if levels.size else 0,
        base_cell_size_m=resolved_config.base_cell_size_m,
    )


def _raise_to_level(levels: np.ndarray, mask: np.ndarray | None, level: int) -> None:
    if mask is None or level <= 0:
        return
    if mask.shape != levels.shape:
        raise ValueError(f"Refinement mask shape {mask.shape} does not match levels shape {levels.shape}.")
    levels[mask.astype(bool)] = np.maximum(levels[mask.astype(bool)], level)

