"""Load exact production topology exported from Gurugram_full_data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gurugram_flood.kernels.triangular_backend import (
    TriangularNodeCouplingTopology,
    TriangularPipeNetworkState,
    TriangularPipeNetworkTopology,
    TriangularSurfaceState,
    TriangularSurfaceTopology,
)
from gurugram_flood.domain.real_gurugram_domain import RealMesh


@dataclass(frozen=True)
class ProductionTopologies:
    mesh: RealMesh
    surface_topology: TriangularSurfaceTopology
    coupling_topology: TriangularNodeCouplingTopology
    pipe_topology: TriangularPipeNetworkTopology
    surface_state: TriangularSurfaceState
    pipe_state: TriangularPipeNetworkState
    building_fraction: np.ndarray | None = None
    impervious_fraction: np.ndarray | None = None
    rainfall_multiplier: np.ndarray | None = None
    horton_f0_mmhr: np.ndarray | None = None
    horton_fc_mmhr: np.ndarray | None = None
    horton_decay_per_hr: np.ndarray | None = None


def _optional_array(data: np.lib.npyio.NpzFile, name: str):
    if name not in data.files:
        return None
    value = data[name]
    if value.size == 0:
        return None
    return value


def load_exact_production_topologies(npz_path: str | Path) -> ProductionTopologies:
    """Load the exported exact mesh, surface topology, coupling, and pipe graph."""

    d = np.load(Path(npz_path), allow_pickle=False)
    mesh = RealMesh(
        nodes_xy=d["nodes_world"].astype(np.float64),
        triangles=d["triangles"].astype(np.int64),
        centroids_xy=d["centroids_world"].astype(np.float64),
        terrain_elevation_m=d["terrain_elevation_m"].astype(np.float32),
        building_mask=d["building_mask"].astype(bool),
        road_mask=d["road_mask"].astype(bool),
        active_mask=d["active_surface_mask"].astype(bool),
    )
    surface_topology = TriangularSurfaceTopology(
        bed_elevation_m=d["bed_elevation_m"].astype(np.float64),
        manning_n=d["manning_n"].astype(np.float64),
        cell_area_m2=d["surface_cell_area_m2"].astype(np.float64),
        characteristic_length_m=d["surface_characteristic_length_m"].astype(np.float64),
        edge_left_cell=d["surface_edge_left_cell"].astype(np.int32),
        edge_right_cell=d["surface_edge_right_cell"].astype(np.int32),
        edge_normal_x=d["surface_edge_normal_x"].astype(np.float64),
        edge_normal_y=d["surface_edge_normal_y"].astype(np.float64),
        edge_length_m=d["surface_edge_length_m"].astype(np.float64),
        boundary_code=d["surface_boundary_code"].astype(np.int32),
        active_mask=d["active_surface_mask"].astype(np.float64),
    )
    coupling_topology = TriangularNodeCouplingTopology(
        inlet_cell_index=d["inlet_cell_index"].astype(np.int32),
        inlet_node_index=d["inlet_node_index"].astype(np.int32),
        node_surface_cell_index=d["node_surface_cell_index"].astype(np.int32),
        node_invert_elevation_m=d["node_invert_elevation_m"].astype(np.float64),
        node_storage_area_m2=d["node_storage_area_m2"].astype(np.float64),
        node_max_depth_m=d["node_max_depth_m"].astype(np.float64),
        node_initial_fill_depth_m=_optional_array(d, "node_initial_fill_depth_m"),
        inlet_capture_capacity_m3ps=_optional_array(d, "inlet_capture_capacity_m3ps"),
        inlet_surcharge_capacity_m3ps=_optional_array(d, "inlet_surcharge_capacity_m3ps"),
        surcharge_cell_index=_optional_array(d, "surcharge_cell_index"),
        surcharge_node_index=_optional_array(d, "surcharge_node_index"),
        surcharge_return_weight=_optional_array(d, "surcharge_return_weight"),
        surcharge_return_weight_is_normalized=bool(d["surcharge_return_weight_is_normalized"][0]),
        node_x_m=_optional_array(d, "node_x_m"),
        node_y_m=_optional_array(d, "node_y_m"),
    )
    pipe_topology = TriangularPipeNetworkTopology(
        link_from_node_index=d["link_from_node_index"].astype(np.int32),
        link_to_node_index=d["link_to_node_index"].astype(np.int32),
        link_length_m=d["link_length_m"].astype(np.float64),
        link_area_m2=d["link_area_m2"].astype(np.float64),
        link_hydraulic_radius_m=d["link_hydraulic_radius_m"].astype(np.float64),
        link_manning_n=d["link_manning_n"].astype(np.float64),
        link_flow_capacity_m3ps=_optional_array(d, "link_flow_capacity_m3ps"),
        outfall_area_m2=_optional_array(d, "outfall_area_m2"),
        outfall_coefficient=_optional_array(d, "outfall_coefficient"),
        outfall_tailwater_head_m=_optional_array(d, "outfall_tailwater_head_m"),
        outfall_flow_capacity_m3ps=_optional_array(d, "outfall_flow_capacity_m3ps"),
        link_invert_elevation_m=_optional_array(d, "link_invert_elevation_m"),
        link_diameter_m=_optional_array(d, "link_diameter_m"),
        link_minor_loss_coefficient=_optional_array(d, "link_minor_loss_coefficient"),
        link_geometry_code=_optional_array(d, "link_geometry_code"),
        link_rect_width_m=_optional_array(d, "link_rect_width_m"),
        link_rect_height_m=_optional_array(d, "link_rect_height_m"),
    )
    surface_state = TriangularSurfaceState(
        h_m=np.zeros(mesh.triangle_count, dtype=np.float64),
        hu_m2ps=np.zeros(mesh.triangle_count, dtype=np.float64),
        hv_m2ps=np.zeros(mesh.triangle_count, dtype=np.float64),
    )
    initial_fill = coupling_topology.node_initial_fill_depth_m
    if initial_fill is None:
        initial_node_volume = np.zeros(coupling_topology.node_count, dtype=np.float64)
    else:
        initial_node_volume = (
            np.asarray(coupling_topology.node_storage_area_m2, dtype=np.float64)
            * np.asarray(initial_fill, dtype=np.float64)
        )
    pipe_state = TriangularPipeNetworkState(
        node_volume_m3=initial_node_volume.astype(np.float64),
        link_flow_m3ps=np.zeros(pipe_topology.link_count, dtype=np.float64),
    )
    return ProductionTopologies(
        mesh,
        surface_topology,
        coupling_topology,
        pipe_topology,
        surface_state,
        pipe_state,
        building_fraction=_optional_array(d, "building_fraction"),
        impervious_fraction=_optional_array(d, "impervious_fraction"),
        rainfall_multiplier=_optional_array(d, "rainfall_multiplier"),
        horton_f0_mmhr=_optional_array(d, "horton_f0_mmhr"),
        horton_fc_mmhr=_optional_array(d, "horton_fc_mmhr"),
        horton_decay_per_hr=_optional_array(d, "horton_decay_per_hr"),
    )
