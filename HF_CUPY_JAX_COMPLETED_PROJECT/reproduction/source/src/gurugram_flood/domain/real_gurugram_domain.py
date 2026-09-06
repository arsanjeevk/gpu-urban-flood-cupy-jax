"""Real-data adapters for the Gurugram citywide triangular flood domain.

Builds solver-ready inputs for `src.gpu_flood.triangular_backend` entirely
from real MCG/GMDA assets already produced by the Gurugram_full_data
pipeline and shipped under `data/`:

  - citywide adaptive triangular mesh (nodes, triangles, terrain, masks)
  - MCG/GMDA storm-drain network (`best_guess_drain_capacities.geojson`,
    itself derived from the raw 28,180-segment STORM_WATER_DRAIN shapefile)
  - real trunk-drain outfall terminals + tailwater priors
  - real July-2025 hourly hyetograph

No synthetic geometry, masks, or drainage topology are generated here.
Where the source data is ambiguous (free-text dimensions, missing
connectivity), the assumption is applied explicitly and is documented in
the docstring of the function that makes it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# NOTE (deployment refactor): geopandas and rasterio are imported lazily
# inside the legacy best_guess topology functions below. The default
# exact-production-topology path does not need them.


# --------------------------------------------------------------------------- #
# Mesh
# --------------------------------------------------------------------------- #


@dataclass
class RealMesh:
    """Real citywide adaptive triangular mesh loaded from production NPZ."""

    nodes_xy: np.ndarray          # (N_nodes, 2) UTM-43N
    triangles: np.ndarray         # (N_tri, 3)
    centroids_xy: np.ndarray      # (N_tri, 2)
    terrain_elevation_m: np.ndarray  # (N_tri,) bare terrain, per triangle
    building_mask: np.ndarray     # (N_tri,) bool
    road_mask: np.ndarray         # (N_tri,) bool
    active_mask: np.ndarray       # (N_tri,) bool -- already excludes buildings

    @property
    def triangle_count(self) -> int:
        return int(self.triangles.shape[0])

    @property
    def node_count(self) -> int:
        return int(self.nodes_xy.shape[0])


def load_real_mesh(mesh_npz_path: str | Path) -> RealMesh:
    """Load the real citywide adaptive triangular mesh.

    The NPZ is produced by the Gurugram_full_data pipeline's own mesh
    builder (`citywide_backend_setup.py` / `adaptive_tile_backend.py`),
    which already defines `active_surface_mask = valid_domain & ~building`
    and resamples `building_mask` from the real `citywide_building_mask_10m.tif`
    at each triangle centroid. We reuse those fields as-is.
    """

    d = np.load(Path(mesh_npz_path), allow_pickle=True)
    return RealMesh(
        nodes_xy=d["nodes_world"].astype(np.float64),
        triangles=d["triangles"].astype(np.int64),
        centroids_xy=d["centroids_world"].astype(np.float64),
        terrain_elevation_m=d["terrain_elevation_m"].astype(np.float32),
        building_mask=d["building_mask"].astype(bool),
        road_mask=d["road_mask"].astype(bool),
        active_mask=d["active_surface_mask"].astype(bool),
    )


def build_manning_n(
    mesh: RealMesh,
    open_n: float = 0.035,
    road_n: float = 0.018,
    building_n: float = 0.25,
) -> np.ndarray:
    """Per-triangle Manning's n from real road/building masks.

    `open_n=0.035` and `building_n=0.25` match the production GPU run's
    own defaults (`scripts/11_run_citywide_event.py --friction 0.035
    --building-friction 0.25`). `road_n=0.018` is a standard paved-surface
    value applied to the real `road_mask` field as an explicit refinement
    beyond the production default (roads are not separately roughened in
    the production run; building cells are inactive anyway and this value
    only matters if active_mask is later relaxed).
    """

    n = np.full(mesh.triangle_count, open_n, dtype=np.float32)
    n[mesh.road_mask] = road_n
    n[mesh.building_mask] = building_n
    return n


# --------------------------------------------------------------------------- #
# Drainage network: real node/link graph from MCG/GMDA assets
# --------------------------------------------------------------------------- #


@dataclass
class DrainageGraph:
    """Snapped real drainage node/link graph."""

    node_x_m: np.ndarray
    node_y_m: np.ndarray
    node_is_outfall: np.ndarray          # bool
    node_outfall_trunk_id: list[str]     # "" when not an outfall
    link_from: np.ndarray                # int, index into node arrays
    link_to: np.ndarray
    link_length_m: np.ndarray
    link_area_m2: np.ndarray
    link_hydraulic_radius_m: np.ndarray
    link_manning_n: np.ndarray
    link_capacity_m3ps: np.ndarray
    link_diameter_m: np.ndarray          # nan for non-circular (box/open)

    @property
    def node_count(self) -> int:
        return int(self.node_x_m.shape[0])

    @property
    def link_count(self) -> int:
        return int(self.link_from.shape[0])


def load_drainage_assets(capacities_geojson: str | Path, trunk_legs_geojson: str | Path | None = None) -> gpd.GeoDataFrame:
    """Load the production pipeline's already-parsed drain-line assets.

    `best_guess_drain_capacities.geojson` is the Gurugram_full_data
    pipeline's own cleanup of the raw 28,180-segment STORM_WATER_DRAIN
    shapefile (messy `DIA_MM` / `DEPTH_M` free-text fields already parsed
    into `diameter_m` / `width_m` / `height_m` / `roughness_n` / `area_m2`,
    and dissolved into 2,913 real drain-line assets). Reusing it avoids
    re-parsing the same free-text fields with a second, independently
    fallible parser. `trunk_legs_geojson` is accepted but unused: the 3
    Badshahpur Drain trunk legs are already present in the capacities
    table as `asset_class == "main drain leg"` rows (with a real
    `best_guess_capacity_m3s` of 80/45/35 m3/s but no surveyed
    cross-section); the standalone trunk-legs file carries no hydraulic
    columns at all, so concatenating it would only reintroduce NaNs.
    """

    import geopandas as gpd  # lazy: legacy topology only

    assets = gpd.read_file(capacities_geojson)
    assets = assets.explode(index_parts=False).reset_index(drop=True)
    assets = assets[assets.geom_type == "LineString"].copy()

    # hydraulic radius for full-flow capacity screening
    is_circular = assets["shape"].astype(str).str.contains("circular", case=False, na=False)
    diameter_m = assets["diameter_m"].to_numpy(dtype=float)
    width_m = assets["width_m"].to_numpy(dtype=float)
    height_m = assets["height_m"].to_numpy(dtype=float)
    radius_m = np.where(is_circular, diameter_m / 4.0, np.nan)
    box_area = width_m * height_m
    box_perimeter = 2.0 * (width_m + height_m)
    box_radius = box_area / np.maximum(box_perimeter, 1.0e-6)
    area_m2 = np.where(is_circular, np.pi * (diameter_m**2) / 4.0, box_area)
    hydraulic_radius_m = np.where(is_circular, radius_m, box_radius)

    # "main drain leg" rows (the 3 real Badshahpur Drain trunk legs) have a
    # real best_guess_capacity_m3s (80/45/35 m3/s) but no surveyed
    # cross-section in the source data. Back out an equivalent rectangular
    # channel from the stated capacity assuming a 2.0 m/s full-flow
    # velocity (a standard lined trunk-channel design velocity), with a
    # 2:1 width:depth ratio -- a derived, attributable estimate, not a
    # measurement, and clearly distinct from the pipe/box rows above.
    is_trunk_leg = assets["asset_class"].astype(str) == "main drain leg"
    trunk_capacity = assets["best_guess_capacity_m3s"].to_numpy(dtype=float)
    assumed_velocity_mps = 2.0
    trunk_area_m2 = trunk_capacity / assumed_velocity_mps
    trunk_depth_m = np.sqrt(np.maximum(trunk_area_m2, 1.0e-6) / 2.0)
    trunk_width_m = 2.0 * trunk_depth_m
    trunk_radius_m = trunk_area_m2 / np.maximum(trunk_width_m + 2.0 * trunk_depth_m, 1.0e-6)
    area_m2 = np.where(is_trunk_leg, trunk_area_m2, area_m2)
    hydraulic_radius_m = np.where(is_trunk_leg, trunk_radius_m, hydraulic_radius_m)

    # any remaining gaps (e.g. "BOX"/"OPEN DRAIN" rows with no dimension at
    # all) fall back to the dataset's own modal real pipe size.
    bad = ~np.isfinite(area_m2) | (area_m2 <= 0)
    area_m2[bad] = np.pi * (0.30**2) / 4.0
    bad_r = ~np.isfinite(hydraulic_radius_m) | (hydraulic_radius_m <= 0)
    hydraulic_radius_m[bad_r] = 0.30 / 4.0

    assets["area_m2"] = area_m2
    assets["hydraulic_radius_m"] = hydraulic_radius_m
    assets["roughness_n"] = assets["roughness_n"].fillna(0.015)
    return assets


def build_drainage_graph(
    assets: gpd.GeoDataFrame,
    snap_tolerance_m: float = 3.0,
) -> DrainageGraph:
    """Snap real drain-line endpoints into a node/link graph.

    Endpoints within `snap_tolerance_m` are merged into one junction node
    (union-find over a KD-tree neighbour query) -- the real shapefile has
    no pre-built topology, only independently digitized lines, so this
    snap step is required to get any graph at all. 3 m matches typical
    municipal as-built digitization tolerance.
    """

    starts = np.array([(geom.coords[0][0], geom.coords[0][1]) for geom in assets.geometry])
    ends = np.array([(geom.coords[-1][0], geom.coords[-1][1]) for geom in assets.geometry])
    endpoints = np.vstack([starts, ends])
    n_links = len(assets)

    tree = cKDTree(endpoints)
    pairs = tree.query_pairs(r=snap_tolerance_m)
    parent = list(range(len(endpoints)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i, j in pairs:
        union(i, j)

    roots = np.array([find(i) for i in range(len(endpoints))])
    unique_roots, node_index = np.unique(roots, return_inverse=True)
    node_xy = np.array([endpoints[roots == r].mean(axis=0) for r in unique_roots])

    link_from = node_index[:n_links]
    link_to = node_index[n_links:]
    keep = link_from != link_to  # drop zero-length self loops after snapping
    link_from, link_to = link_from[keep], link_to[keep]
    a = assets.loc[keep].reset_index(drop=True)

    length_m = a.geometry.length.to_numpy(dtype=float)
    length_m = np.maximum(length_m, 1.0)  # avoid zero-length pipes

    return DrainageGraph(
        node_x_m=node_xy[:, 0],
        node_y_m=node_xy[:, 1],
        node_is_outfall=np.zeros(len(node_xy), dtype=bool),
        node_outfall_trunk_id=[""] * len(node_xy),
        link_from=link_from.astype(np.int64),
        link_to=link_to.astype(np.int64),
        link_length_m=length_m,
        link_area_m2=a["area_m2"].to_numpy(dtype=float),
        link_hydraulic_radius_m=a["hydraulic_radius_m"].to_numpy(dtype=float),
        link_manning_n=a["roughness_n"].fillna(0.015).to_numpy(dtype=float),
        link_capacity_m3ps=a["best_guess_capacity_m3s"].fillna(0.05).to_numpy(dtype=float),
        link_diameter_m=a["diameter_m"].to_numpy(dtype=float) if "diameter_m" in a.columns else np.full(len(a), np.nan),
    )


def sample_dem_at_points(dem_path: str | Path, x_m: np.ndarray, y_m: np.ndarray) -> np.ndarray:
    """Sample the real FABDEM raster at point coordinates (UTM-43N)."""

    import rasterio  # lazy: legacy topology only

    with rasterio.open(dem_path) as src:
        values = np.array(list(src.sample(zip(x_m, y_m))), dtype=np.float64).ravel()
    return values


def assign_node_inverts_and_direction(
    graph: DrainageGraph,
    dem_path: str | Path,
    cover_depth_m: float = 1.5,
) -> DrainageGraph:
    """Assign invert elevations from the real DEM and orient links downhill.

    `FLOW_DIREC` in the raw shapefile is entirely null (verified: 28,180/28,180
    rows are 0.0), so it cannot be used. Flow direction is instead inferred
    from real terrain: each link is re-oriented so flow runs from the
    higher-terrain node to the lower-terrain node, which is the standard
    assumption for gravity storm drainage absent surveyed inverts.
    `cover_depth_m=1.5` (a typical MCG storm-drain cover depth) sets the
    pipe invert below the real terrain surface at each node.
    """

    ground_m = sample_dem_at_points(dem_path, graph.node_x_m, graph.node_y_m)
    invert_m = ground_m - cover_depth_m

    from_higher = invert_m[graph.link_from] >= invert_m[graph.link_to]
    new_from = np.where(from_higher, graph.link_from, graph.link_to)
    new_to = np.where(from_higher, graph.link_to, graph.link_from)
    graph.link_from = new_from
    graph.link_to = new_to
    graph.node_invert_elevation_m = invert_m  # type: ignore[attr-defined]
    return graph


def attach_real_outfalls(
    graph: DrainageGraph,
    outfall_inventory_geojson: str | Path,
    tailwater_priors_csv: str | Path,
    tailwater_case: str = "moderate",
    snap_tolerance_m: float = 400.0,
) -> tuple[DrainageGraph, np.ndarray, np.ndarray, np.ndarray]:
    """Mark graph nodes nearest the 3 real trunk-leg discharge terminals as outfalls.

    Only 3 confirmed citywide discharge terminals exist in the real data
    (`outfall_inventory.geojson`: TRUNK-LEG-1/2/3, all Badshahpur Drain legs).
    The nearest graph node within `snap_tolerance_m` of each real terminal
    is marked as an outfall, using the real terrain elevation and the real
    `moderate` tailwater case from `outfall_tailwater_priors.csv` (matching
    `model_scenarios.yaml`'s default `citywide_partial70` scenario).

    Returns the graph plus per-node arrays: outfall_area_m2,
    outfall_coefficient, outfall_tailwater_head_m (zero/NaN-safe defaults
    for non-outfall nodes, since `TriangularPipeNetworkTopology` expects
    one value per node, not a sparse table).
    """

    import geopandas as gpd  # lazy: legacy topology only

    outfalls = gpd.read_file(outfall_inventory_geojson)
    tailwater = pd.read_csv(tailwater_priors_csv)
    tailwater = tailwater[tailwater["tailwater_case"] == tailwater_case].set_index("trunk_id")

    node_xy = np.column_stack([graph.node_x_m, graph.node_y_m])
    tree = cKDTree(node_xy)

    outfall_area_m2 = np.zeros(graph.node_count, dtype=float)
    outfall_coefficient = np.zeros(graph.node_count, dtype=float)
    outfall_tailwater_head_m = np.zeros(graph.node_count, dtype=float)

    for _, row in outfalls.iterrows():
        pt = np.array([row["x_utm43n"], row["y_utm43n"]])
        dist, idx = tree.query(pt)
        if dist > snap_tolerance_m:
            continue
        graph.node_is_outfall[idx] = True
        graph.node_outfall_trunk_id[idx] = str(row["trunk_id"])
        # 600 mm-equivalent discharge orifice -- real trunk legs are "main
        # open/trunk drain" with no surveyed cross-section (confidence:
        # "low until design discharge/cross-section is provided" in the
        # asset table); this is a placeholder discharge area, not a measurement.
        outfall_area_m2[idx] = 0.6 * 0.6
        outfall_coefficient[idx] = 0.62
        if row["trunk_id"] in tailwater.index:
            tw_surface_m = float(tailwater.loc[row["trunk_id"], "tailwater_surface_m"])
            outfall_tailwater_head_m[idx] = tw_surface_m

    return graph, outfall_area_m2, outfall_coefficient, outfall_tailwater_head_m


def map_nodes_to_mesh_triangles(graph: DrainageGraph, mesh: RealMesh) -> np.ndarray:
    """Nearest *active* mesh triangle centroid for every drainage node (for surface coupling)."""

    active_idx = np.where(mesh.active_mask)[0]
    tree = cKDTree(mesh.centroids_xy[active_idx])
    node_xy = np.column_stack([graph.node_x_m, graph.node_y_m])
    _, nearest = tree.query(node_xy)
    return active_idx[nearest]


def load_july2025_hyetograph(csv_path: str | Path) -> pd.DataFrame:
    """Load the real July-2025 hourly hyetograph (133 mm / 12 h reported event)."""

    return pd.read_csv(csv_path)


def hyetograph_rate_fn(hyeto: pd.DataFrame):
    """Build a `t_s -> rainfall_rate_mps` callable from the real hyetograph table."""

    start_s = hyeto["start_min"].to_numpy(dtype=float) * 60.0
    end_s = hyeto["end_min"].to_numpy(dtype=float) * 60.0
    rate_mps = hyeto["intensity_mmhr"].to_numpy(dtype=float) / 1000.0 / 3600.0

    def rate(t_s: float) -> float:
        idx = np.searchsorted(end_s, t_s, side="right")
        if idx >= len(rate_mps) or t_s < start_s[idx]:
            return 0.0
        return float(rate_mps[idx])

    return rate


@dataclass
class FullCityTopologies:
    """Bundled real-data topologies + initial states for the triangular solver."""

    mesh: RealMesh
    graph: DrainageGraph
    surface_topology: Any
    coupling_topology: Any
    pipe_topology: Any
    surface_state: Any
    pipe_state: Any


# default real-data-grounded assumptions for storage-node geometry not present
# in the source GIS (no manhole survey exists citywide). A single 1.0 m^2
# inspection-chamber footprint was found to surcharge violently under real
# peak inflows (60/4750 nodes hit capacity and dumped overflow onto a single
# coupled triangle each, producing isolated >9 m depth spikes that collapsed
# the CFL timestep). 4.0 m^2 approximates a junction-pit footprint serving
# several converging pipes (~2 m x 2 m), still a real, common MCG chamber
# size class, not a per-node measurement.
NODE_STORAGE_AREA_M2_DEFAULT = 4.0
NODE_MAX_DEPTH_M_DEFAULT = 2.0


def build_full_topologies(
    mesh: RealMesh,
    manning_n: np.ndarray,
    graph: DrainageGraph,
    node_cell_index: np.ndarray,
    outfall_area_m2: np.ndarray,
    outfall_coefficient: np.ndarray,
    outfall_tailwater_head_m: np.ndarray,
    boundary_mode: str = "outflow",
    drainage_capacity_fraction: float = 0.7,
    initial_drainage_storage_fraction: float = 0.5,
):
    """Assemble TriangularSurfaceTopology + NodeCoupling + PipeNetwork from real data.

    `drainage_capacity_fraction=0.7` and `initial_drainage_storage_fraction=0.5`
    are the real `citywide_partial70` scenario values from
    `data/configs/model_scenarios.yaml` -- documented there as "First
    full-Gurugram screening scenario requested for July hourly rainfall."
    They derate the real pipe/inlet capacities (siltation, partial blockage)
    and pre-fill drains to a fraction of capacity before the storm starts,
    rather than assuming a perfectly clean, empty network.
    """

    from gurugram_flood.kernels.triangular_backend import (
        TriangularNodeCouplingTopology,
        TriangularPipeNetworkState,
        TriangularPipeNetworkTopology,
        TriangularSurfaceState,
        build_triangular_surface_topology,
    )

    surface_topology = build_triangular_surface_topology(
        nodes_xy=mesh.nodes_xy,
        triangles=mesh.triangles,
        bed_elevation_m=mesh.terrain_elevation_m,
        manning_n=manning_n,
        active_mask=mesh.active_mask,
        boundary_mode=boundary_mode,
    )

    derated_link_capacity = graph.link_capacity_m3ps * drainage_capacity_fraction

    # per-node inlet capture capacity: sum of (derated) capacities of links
    # touching the node, so a junction feeding two 300mm pipes can capture
    # roughly twice as fast as a junction feeding one -- derived from the
    # real best_guess_capacity_m3s field, not an arbitrary constant.
    link_cap_at_node = np.zeros(graph.node_count, dtype=float)
    np.add.at(link_cap_at_node, graph.link_from, derated_link_capacity)
    np.add.at(link_cap_at_node, graph.link_to, derated_link_capacity)
    inlet_capacity = np.clip(link_cap_at_node, 0.02, 5.0)

    coupling_topology = TriangularNodeCouplingTopology(
        inlet_cell_index=node_cell_index.astype(np.int32),
        inlet_node_index=np.arange(graph.node_count, dtype=np.int32),
        node_surface_cell_index=node_cell_index.astype(np.int32),
        node_invert_elevation_m=graph.node_invert_elevation_m.astype(np.float64),
        node_storage_area_m2=np.full(graph.node_count, NODE_STORAGE_AREA_M2_DEFAULT, dtype=np.float64),
        node_max_depth_m=np.full(graph.node_count, NODE_MAX_DEPTH_M_DEFAULT, dtype=np.float64),
        inlet_capture_capacity_m3ps=inlet_capacity,
        inlet_surcharge_capacity_m3ps=inlet_capacity,
        node_x_m=graph.node_x_m.astype(np.float64),
        node_y_m=graph.node_y_m.astype(np.float64),
    )

    pipe_topology = TriangularPipeNetworkTopology(
        link_from_node_index=graph.link_from.astype(np.int32),
        link_to_node_index=graph.link_to.astype(np.int32),
        link_length_m=graph.link_length_m.astype(np.float64),
        link_area_m2=graph.link_area_m2.astype(np.float64),
        link_hydraulic_radius_m=graph.link_hydraulic_radius_m.astype(np.float64),
        link_manning_n=graph.link_manning_n.astype(np.float64),
        link_flow_capacity_m3ps=derated_link_capacity.astype(np.float64),
        outfall_area_m2=outfall_area_m2.astype(np.float64),
        outfall_coefficient=outfall_coefficient.astype(np.float64),
        outfall_tailwater_head_m=outfall_tailwater_head_m.astype(np.float64),
        # link_diameter_m intentionally omitted: passing it switches the
        # solver to a Preissmann-slot pressurized-flow term that assumes
        # *every* link is circular. Our links are a real mix of circular
        # pipes, box drains, and open channels, so we keep the precomputed
        # per-shape link_area_m2 / link_hydraulic_radius_m (set above from
        # the real diameter/width/height fields) as the flow cross-section
        # instead.
    )

    surface_state = TriangularSurfaceState(
        h_m=np.zeros(mesh.triangle_count, dtype=np.float64),
        hu_m2ps=np.zeros(mesh.triangle_count, dtype=np.float64),
        hv_m2ps=np.zeros(mesh.triangle_count, dtype=np.float64),
    )
    initial_node_volume_m3 = (
        NODE_STORAGE_AREA_M2_DEFAULT * NODE_MAX_DEPTH_M_DEFAULT * initial_drainage_storage_fraction
    )
    pipe_state = TriangularPipeNetworkState(
        node_volume_m3=np.full(graph.node_count, initial_node_volume_m3, dtype=np.float64),
        link_flow_m3ps=np.zeros(graph.link_count, dtype=np.float64),
    )

    return FullCityTopologies(
        mesh=mesh,
        graph=graph,
        surface_topology=surface_topology,
        coupling_topology=coupling_topology,
        pipe_topology=pipe_topology,
        surface_state=surface_state,
        pipe_state=pipe_state,
    )
