"""Core output artifacts: compressed NPZ, run report JSON, timeseries PNG.

Array names in the NPZ match the notebook outputs
(``gurugram_nowcast_outputs.npz``) so existing downstream consumers keep
working unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..logging_utils import get_logger
from ..solver.engine import SimulationResult

log = get_logger("outputs")

RUN_NPZ_NAME = "flood_simulation_outputs.npz"
REPORT_NAME = "run_report.json"
TIMESERIES_PNG_NAME = "flood_timeseries.png"


def save_run_npz(result: SimulationResult, out_dir: Path) -> Path:
    """Write the full simulation output NPZ (notebook-compatible array names)."""
    mesh = result.city.mesh
    topo = result.city.topologies
    path = out_dir / RUN_NPZ_NAME

    arrays = dict(
        snapshot_times_min=result.snapshot_times_min,
        snapshot_times_s=result.snapshot_times_min * 60.0,
        depth_snapshots_m=result.depth_snapshots_m,
        speed_snapshots_mps=result.speed_snapshots_mps,
        drain_node_volume_snapshots_m3=result.node_volume_snapshots_m3,
        final_depth_m=result.final_depth_m,
        max_depth_m=result.max_depth_m,
        final_hu_m2ps=result.final_hu_m2ps,
        final_hv_m2ps=result.final_hv_m2ps,
        max_speed_mps=result.max_speed_mps,
        final_node_volume_m3=result.final_node_volume_m3,
        final_link_flow_m3ps=result.final_link_flow_m3ps,
        nodes_world=mesh.nodes_xy.astype(np.float32),
        triangles=mesh.triangles.astype(np.int32),
        centroids_world=mesh.centroids_xy.astype(np.float32),
        terrain_elevation_m=mesh.terrain_elevation_m.astype(np.float32),
        active_surface_mask=mesh.active_mask.astype(bool),
        building_mask=mesh.building_mask.astype(bool),
        road_mask=mesh.road_mask.astype(bool),
        drain_node_x_m=np.asarray(topo.coupling_topology.node_x_m, dtype=np.float32),
        drain_node_y_m=np.asarray(topo.coupling_topology.node_y_m, dtype=np.float32),
        drain_link_from=np.asarray(topo.pipe_topology.link_from_node_index, dtype=np.int32),
        drain_link_to=np.asarray(topo.pipe_topology.link_to_node_index, dtype=np.int32),
    )
    np.savez_compressed(path, **arrays)
    log.info("Saved NPZ: %s (%.1f MB)", path, path.stat().st_size / 1e6)
    return path


def build_report(result: SimulationResult, artifacts: dict[str, str]) -> dict:
    """Assemble the JSON-serialisable run report."""
    cfg = result.config
    mesh = result.city.mesh
    topo = result.city.topologies
    wet_threshold = cfg.outputs.wet_threshold_m
    valid = mesh.active_mask & ~mesh.building_mask

    report = {
        "implementation": "gurugram_flood_backend_fused_cuda",
        "topology_mode": result.city.topology_mode,
        "surface_step_mode": cfg.solver.surface_step_mode,
        "infiltration_mode": cfg.physics.infiltration_mode,
        "backend": result.backend_name,
        "is_gpu": result.is_gpu,
        "rainfall_csv": result.rainfall.source_csv,
        "rainfall_total_mm": result.rainfall.total_depth_mm,
        "duration_min": result.duration_min,
        "runtime_s": result.runtime_s,
        "step_count": result.step_count,
        "steps_per_s": result.step_count / max(result.runtime_s, 1e-9),
        "cfl": cfg.solver.cfl,
        "cfl_update_interval_steps": cfg.solver.cfl_update_interval_steps,
        "cfl_reuse_safety_factor": cfg.solver.cfl_reuse_safety_factor,
        "max_dt_s": cfg.solver.max_dt_s,
        "maximum_allowed_speed_mps": cfg.solver.max_speed_mps,
        "cfl_eval_count": result.cfl_eval_count,
        "cfl_reused_count": result.cfl_reused_count,
        "snapshot_count": int(len(result.snapshot_times_min)),
        "max_depth_m": float(np.nanmax(result.max_depth_m)),
        "wet_triangle_count_005m": int((result.final_depth_m[valid] >= wet_threshold).sum()),
        "drainage_node_count": int(topo.coupling_topology.node_count),
        "drainage_link_count": int(topo.pipe_topology.link_count),
        "mapped_inlet_count": int(topo.coupling_topology.inlet_count),
        **{k: float(v) for k, v in result.mass_balance.items()},
        "artifacts": artifacts,
        "config": cfg.to_dict(),
    }
    return report


def save_report(report: dict, out_dir: Path) -> Path:
    path = out_dir / REPORT_NAME
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("Saved report: %s", path)
    return path


def save_timeseries_png(result: SimulationResult, out_dir: Path) -> Path | None:
    """Wet-area / peak-depth / mean-depth timeseries plot (needs matplotlib)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed; skipping timeseries PNG")
        return None

    mesh = result.city.mesh
    wet_threshold = result.config.outputs.wet_threshold_m
    valid = mesh.active_mask & ~mesh.building_mask
    times_hr = result.snapshot_times_min / 60.0
    depths = result.depth_snapshots_m

    wet_counts = [(d[valid] >= wet_threshold).sum() for d in depths]
    max_depths = [float(d[valid].max()) for d in depths]
    mean_depths = [float(d[valid].mean()) for d in depths]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(times_hr, wet_counts, "o-", color="#2563eb")
    axes[0].set_xlabel("Time (h)")
    axes[0].set_ylabel(f"Wet triangles >= {wet_threshold*100:.0f} cm")
    axes[0].set_title("Flooded area")
    axes[1].plot(times_hr, max_depths, "o-", color="#1e3a8a")
    axes[1].set_xlabel("Time (h)")
    axes[1].set_ylabel("Max depth (m)")
    axes[1].set_title("Peak depth")
    axes[2].plot(times_hr, mean_depths, "o-", color="#047857")
    axes[2].set_xlabel("Time (h)")
    axes[2].set_ylabel("Mean depth (m)")
    axes[2].set_title("Mean depth")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()

    path = out_dir / TIMESERIES_PNG_NAME
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved timeseries plot: %s", path)
    return path
