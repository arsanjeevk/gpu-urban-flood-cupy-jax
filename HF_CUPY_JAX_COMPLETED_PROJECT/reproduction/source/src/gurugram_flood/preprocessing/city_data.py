"""Loading of the static Gurugram city datasets.

Everything in this module is *static* input: mesh, drainage topology,
infiltration/BCR fields, recharge table, historical boundary inflow. The
only runtime input (the rainfall CSV) is handled by
:mod:`gurugram_flood.preprocessing.rainfall`.

The heavy lifting is delegated to the validated domain adapters
(:mod:`gurugram_flood.domain`); this module adds path resolution, existence
checks, logging and a single bundled :class:`CityDomain` result the solver
consumes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config import RunConfig
from ..domain import june5_production_drainage as june5
from ..domain import real_gurugram_domain as rgd
from ..domain.production_event_support import (
    ExternalBoundaryInflowSurface,
    load_external_boundary_inflow_surface,
)
from ..domain.production_infiltration_support import (
    ProductionInfiltrationFields,
    prepare_production_infiltration_fields,
)
from ..domain.production_topology_loader import load_exact_production_topologies
from ..exceptions import DataFileMissingError
from ..logging_utils import get_logger

log = get_logger("city_data")


@dataclass
class CityDomain:
    """All static inputs bundled for one simulation run.

    ``topologies`` is either a ``ProductionTopologies`` (exact-production
    mode) or a ``FullCityTopologies`` (legacy mode); both expose ``mesh``,
    ``surface_topology``, ``coupling_topology``, ``pipe_topology``,
    ``surface_state`` and ``pipe_state``.
    """

    topologies: Any
    topology_mode: str
    # production_horton infiltration fields (None in simple_constant mode).
    infiltration_fields: ProductionInfiltrationFields | None
    # simple_constant per-cell infiltration capacity (m/s); None in horton mode.
    simple_infiltration_mps: np.ndarray | None
    # surface recharge (None when disabled).
    recharge_topology: Any | None
    recharge_info: dict | None
    # historical external boundary inflow (None when disabled).
    external_boundary: ExternalBoundaryInflowSurface | None

    @property
    def mesh(self) -> rgd.RealMesh:
        return self.topologies.mesh


def _require(cfg: RunConfig, name: str):
    path = cfg.paths.resolve(name)
    if not path.exists():
        raise DataFileMissingError(path, name)
    return path


def _load_exact_production(cfg: RunConfig):
    npz_path = _require(cfg, "production_topology_npz")
    t0 = time.perf_counter()
    topo = load_exact_production_topologies(npz_path)
    mesh = topo.mesh
    log.info(
        "Exact production topology loaded in %.1fs: %s triangles, %s drainage nodes, "
        "%s links, %s mapped inlets",
        time.perf_counter() - t0,
        f"{mesh.triangle_count:,}",
        f"{topo.coupling_topology.node_count:,}",
        f"{topo.pipe_topology.link_count:,}",
        f"{topo.coupling_topology.inlet_count:,}",
    )
    return topo


def _load_legacy_best_guess(cfg: RunConfig):
    mesh = rgd.load_real_mesh(_require(cfg, "mesh_npz"))
    manning = rgd.build_manning_n(mesh)
    log.info("Legacy mesh loaded: %s triangles", f"{mesh.triangle_count:,}")

    assets = rgd.load_drainage_assets(_require(cfg, "drain_capacities_geojson"))
    graph = rgd.build_drainage_graph(assets, snap_tolerance_m=3.0)
    graph = rgd.assign_node_inverts_and_direction(graph, _require(cfg, "dem_tif"))
    graph, of_area, of_coef, of_tw = rgd.attach_real_outfalls(
        graph,
        _require(cfg, "outfall_inventory_geojson"),
        _require(cfg, "tailwater_priors_csv"),
        tailwater_case=cfg.topology.tailwater_case,
    )
    node_cell_index = rgd.map_nodes_to_mesh_triangles(graph, mesh)
    log.info(
        "Legacy drainage graph: %s nodes, %s links, %d real outfalls",
        f"{graph.node_count:,}", f"{graph.link_count:,}", int(graph.node_is_outfall.sum()),
    )

    return rgd.build_full_topologies(
        mesh, manning, graph, node_cell_index, of_area, of_coef, of_tw,
        drainage_capacity_fraction=cfg.topology.drainage_capacity_fraction,
        initial_drainage_storage_fraction=cfg.topology.initial_drainage_storage_fraction,
    )


def _build_horton_fields(cfg: RunConfig, topo) -> ProductionInfiltrationFields:
    bcr_path = _require(cfg, "horton_bcr_npz")
    bcr = np.load(bcr_path)
    mesh = topo.mesh
    fields = prepare_production_infiltration_fields(
        active_mask=mesh.active_mask,
        rainfall_multiplier=topo.rainfall_multiplier,
        horton_f0_mmhr=bcr["horton_f0_mmhr"],
        horton_fc_mmhr=bcr["horton_fc_mmhr"],
        horton_decay_per_hr=bcr["horton_decay_per_hr"],
        building_fraction=bcr["building_fraction"],
        impervious_fraction=topo.impervious_fraction,
        use_bcr_open_fraction=cfg.physics.bcr_open_fraction,
        min_open_fraction=cfg.physics.bcr_min_open_fraction,
    )
    active = mesh.active_mask
    log.info(
        "Horton/BCR fields: f0_eff=%.3f mm/hr, fc_eff=%.3f mm/hr (active-cell means)",
        float(bcr["horton_f0_mmhr"][active].mean()),
        float(bcr["horton_fc_mmhr"][active].mean()),
    )
    return fields


def _build_simple_infiltration(cfg: RunConfig, mesh: rgd.RealMesh) -> np.ndarray:
    open_mps = cfg.physics.simple_infiltration_open_mmhr / 1000.0 / 3600.0
    infiltration = np.full(mesh.triangle_count, open_mps, dtype=np.float32)
    infiltration[mesh.road_mask] = open_mps * cfg.physics.simple_infiltration_road_factor
    infiltration[mesh.building_mask] = 0.0
    return infiltration


def load_city_domain(cfg: RunConfig) -> CityDomain:
    """Load every static dataset required by ``cfg`` and bundle it."""

    if cfg.topology.mode == "exact_production":
        topo = _load_exact_production(cfg)
    else:
        topo = _load_legacy_best_guess(cfg)
    mesh = topo.mesh

    infiltration_fields = None
    simple_infiltration = None
    if cfg.physics.infiltration_mode == "production_horton":
        # Horton/BCR fields come from the required sidecar NPZ; missing
        # per-cell arrays in the topology export (e.g. rainfall_multiplier)
        # fall back to neutral defaults inside the domain adapter, exactly
        # as in the validated notebook runs.
        infiltration_fields = _build_horton_fields(cfg, topo)
    else:
        simple_infiltration = _build_simple_infiltration(cfg, mesh)

    recharge_topology = None
    recharge_info = None
    if cfg.physics.use_surface_recharge:
        recharge_topology, recharge_info = june5.load_june5_recharge_topology(
            _require(cfg, "recharge_csv"), mesh
        )
        log.info("Surface recharge enabled: %s", recharge_info)

    external_boundary = None
    if cfg.physics.use_external_boundary_inflow:
        external_boundary = load_external_boundary_inflow_surface(
            _require(cfg, "external_boundary_inflow_csv"),
            centroids_xy=mesh.centroids_xy,
            active_mask=mesh.active_mask,
            max_snap_m=cfg.physics.external_boundary_max_snap_m,
        )
        log.info(
            "External boundary inflow: %s receivers, %.1f m3 hydrograph volume",
            f"{external_boundary.report['receiver_count']:,}",
            external_boundary.report["hydrograph_volume_m3"],
        )

    return CityDomain(
        topologies=topo,
        topology_mode=cfg.topology.mode,
        infiltration_fields=infiltration_fields,
        simple_infiltration_mps=simple_infiltration,
        recharge_topology=recharge_topology,
        recharge_info=recharge_info,
        external_boundary=external_boundary,
    )
