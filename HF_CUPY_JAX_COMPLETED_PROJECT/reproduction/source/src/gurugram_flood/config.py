"""Run configuration: dataclasses + YAML loading.

All tunables that used to live as module-level constants in the deployment
notebooks are collected here. Defaults reproduce the production *nowcast*
run (`gurugram_nowcast_run.ipynb`): exact production topology, full-SWE
surface stepping, Horton infiltration, surface recharge enabled.

Configuration precedence (lowest to highest):

1. dataclass defaults (this file)
2. YAML config file passed to :func:`load_config`
3. explicit ``overrides`` mapping (dotted keys, e.g. ``"solver.duration_min"``)
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from .exceptions import ConfigError

# Repository layout anchor: <project root>/src/gurugram_flood/config.py
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
# For non-editable installs (site-packages), point GURUGRAM_FLOOD_DATA_DIR at
# the data directory; otherwise the repo-local data/ is used.
DEFAULT_DATA_DIR = Path(
    os.environ.get(
        "GURUGRAM_FLOOD_DATA_DIR",
        _PROJECT_ROOT / "data" / "production_gurugram",
    )
)


@dataclass
class PathsConfig:
    """Locations of the static city datasets.

    Only ``data_dir`` normally needs changing; the individual entries are
    paths relative to ``data_dir`` (absolute paths are also accepted).
    """

    data_dir: str = str(DEFAULT_DATA_DIR)

    # Exact production topology export (mesh + drainage, the primary input).
    production_topology_npz: str = "production_topology/d_june5_exact_production_topology_and_mesh.npz"
    # Horton/BCR forcing fields sidecar (the production NPZ lacks per-cell f0).
    horton_bcr_npz: str = "production_topology/horton_bcr_fields.npz"
    # Surface recharge exchange table (June-5 production drainage dataset).
    recharge_csv: str = "drainage_june5/recharge/surface_recharge_exchange_table.csv"
    # Historical July-2025 external boundary inflow hydrograph (event preset only).
    external_boundary_inflow_csv: str = "external_boundary_inflow/gurugram_external_boundary_inflow_july2025_133mm_12h.csv"

    # Legacy best_guess topology inputs (only used when topology.mode == "legacy_best_guess").
    mesh_npz: str = "mesh/citywide_anuga_mesh_arrays.npz"
    dem_tif: str = "terrain/hydraulic_dtm_fabdem_5m.tif"
    drain_capacities_geojson: str = "drainage/best_guess_drain_capacities.geojson"
    outfall_inventory_geojson: str = "drainage/outfall_inventory.geojson"
    tailwater_priors_csv: str = "pragmatic_priors/outfall_tailwater_priors.csv"

    def resolve(self, name: str) -> Path:
        """Resolve a configured path entry against ``data_dir``."""
        raw = Path(getattr(self, name))
        return raw if raw.is_absolute() else Path(self.data_dir) / raw


@dataclass
class TopologyConfig:
    """Which city topology to simulate on."""

    # "exact_production": exact D_june_5 production mesh + drainage export (recommended).
    # "legacy_best_guess": rebuild drainage from best_guess GeoJSON + DEM (older, slower).
    mode: str = "exact_production"
    # legacy_best_guess scenario derating (matches the citywide_partial70 scenario).
    drainage_capacity_fraction: float = 0.7
    initial_drainage_storage_fraction: float = 0.5
    tailwater_case: str = "moderate"


@dataclass
class SolverConfig:
    """Time stepping and backend selection (notebook-verified values)."""

    backend: str = "cupy"                 # production execution is GPU-only
    surface_step_mode: str = "full_swe"   # "full_swe" or "local_inertial"
    # None -> simulate exactly the duration covered by the rainfall CSV.
    duration_min: float | None = None
    snapshot_interval_min: float = 60.0
    max_dt_s: float = 1.0
    cfl: float = 0.85
    cfl_update_interval_steps: int = 32
    cfl_reuse_safety_factor: float = 0.8
    # ANUGA-DE0-style velocity clamp; stabiliser, preserves momentum direction.
    max_speed_mps: float = 5.0
    # How often (in steps) to check depths for NaN/Inf blow-up.
    finite_check_interval_steps: int = 1000


@dataclass
class PhysicsConfig:
    """Forcing/loss processes."""

    infiltration_mode: str = "production_horton"  # "production_horton" or "simple_constant"
    bcr_open_fraction: bool = True
    bcr_min_open_fraction: float = 0.05
    use_surface_recharge: bool = True
    # simple_constant mode: open-ground infiltration capacity (mm/hr, Gurugram silty clay loam).
    simple_infiltration_open_mmhr: float = 6.5
    simple_infiltration_road_factor: float = 0.05

    # Historical-event features (off for nowcasts).
    use_external_boundary_inflow: bool = False
    external_boundary_max_snap_m: float = 500.0
    match_reference_initial_node_volume: bool = False
    reference_initial_node_volume_m3: float = 541810.5625


@dataclass
class OutputsConfig:
    """Which artifacts to write and their thresholds."""

    wet_threshold_m: float = 0.05
    geojson_threshold_m: float = 0.05
    write_npz: bool = True
    write_hourly_geojson: bool = True
    write_basemap_html: bool = True
    write_timeseries_png: bool = True
    # Basemap slider rendering limits (matches the notebooks).
    basemap_max_features: int = 120_000
    basemap_color_max_m: float = 3.0
    # Mesh CRS (UTM 43N) — used when reprojecting outputs to WGS84 / WebMercator.
    mesh_crs: str = "EPSG:32643"


@dataclass
class RunConfig:
    """Top-level configuration for one simulation run."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    outputs: OutputsConfig = field(default_factory=OutputsConfig)

    def validate(self) -> None:
        if self.topology.mode not in ("exact_production", "legacy_best_guess"):
            raise ConfigError(f"topology.mode must be 'exact_production' or 'legacy_best_guess', got {self.topology.mode!r}")
        if self.solver.backend != "cupy":
            raise ConfigError(
                f"production solver is GPU-only; solver.backend must be 'cupy', "
                f"got {self.solver.backend!r}"
            )
        if self.solver.surface_step_mode not in ("full_swe", "local_inertial"):
            raise ConfigError(f"solver.surface_step_mode must be 'full_swe' or 'local_inertial', got {self.solver.surface_step_mode!r}")
        if self.physics.infiltration_mode not in ("production_horton", "simple_constant"):
            raise ConfigError(f"physics.infiltration_mode must be 'production_horton' or 'simple_constant', got {self.physics.infiltration_mode!r}")
        if self.physics.infiltration_mode == "production_horton" and self.topology.mode != "exact_production":
            raise ConfigError("physics.infiltration_mode='production_horton' requires topology.mode='exact_production'")
        if self.solver.duration_min is not None and self.solver.duration_min <= 0:
            raise ConfigError("solver.duration_min must be positive")
        if not 0 < self.solver.cfl <= 1:
            raise ConfigError("solver.cfl must be in (0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _apply_section(section_obj: Any, values: Mapping[str, Any], section_name: str) -> None:
    valid = {f.name for f in dataclasses.fields(section_obj)}
    for key, value in values.items():
        if key not in valid:
            raise ConfigError(f"Unknown config key '{section_name}.{key}'")
        setattr(section_obj, key, value)


def load_config(
    config_file: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> RunConfig:
    """Build a validated :class:`RunConfig`.

    ``config_file`` is an optional YAML file with any subset of the sections
    (paths / topology / solver / physics / outputs). ``overrides`` uses
    dotted keys, e.g. ``{"solver.duration_min": 720}``.
    """
    cfg = RunConfig()

    if config_file is not None:
        path = Path(config_file)
        if not path.exists():
            raise ConfigError(f"Config file not found: {path}")
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ConfigError(f"Config file must be a YAML mapping: {path}")
        sections = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}
        for section_name, values in loaded.items():
            if section_name not in sections:
                raise ConfigError(f"Unknown config section '{section_name}' in {path}")
            if values is None:
                continue
            if not isinstance(values, dict):
                raise ConfigError(f"Config section '{section_name}' must be a mapping in {path}")
            _apply_section(sections[section_name], values, section_name)

    for dotted, value in (overrides or {}).items():
        if value is None:
            continue
        try:
            section_name, key = dotted.split(".", 1)
        except ValueError:
            raise ConfigError(f"Override keys must be 'section.key', got {dotted!r}") from None
        if not hasattr(cfg, section_name):
            raise ConfigError(f"Unknown config section in override '{dotted}'")
        _apply_section(getattr(cfg, section_name), {key: value}, section_name)

    cfg.validate()
    return cfg
