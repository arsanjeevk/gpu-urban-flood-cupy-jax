from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "comparisons/user_24h_160_09mm"
TOPOLOGY = ROOT / "data/production_gurugram/production_topology/d_june5_exact_production_topology_and_mesh.npz"


def test_exact_production_topology_contract() -> None:
    with np.load(TOPOLOGY, allow_pickle=False) as data:
        assert data["triangles"].shape == (981_880, 3)
        assert int(data["active_surface_mask"].sum()) == 817_573
        assert data["node_x_m"].shape == (118_768,)
        assert data["link_from_node_index"].shape == (139_798,)
        assert data["inlet_cell_index"].shape == (61_317,)
        assert int((data["outfall_area_m2"] > 0).sum()) == 46
        exterior = data["surface_edge_right_cell"] < 0
        assert int((exterior & (data["surface_boundary_code"] == 4)).sum()) == 34_390
        assert np.isfinite(data["terrain_elevation_m"]).all()
        assert np.isfinite(data["manning_n"]).all()


def test_preserved_full_run_artifacts_and_parity() -> None:
    cupy = json.loads((RUN / "cupy/run_report.json").read_text())
    jax = json.loads((RUN / "jax/run_report.json").read_text())
    parity = json.loads((RUN / "jax/parity_validation.json").read_text())
    assert cupy["is_gpu"] is True and cupy["backend"] == "cupy"
    assert jax["is_gpu"] is True and jax["backend"] == "jax_gpu"
    assert cupy["duration_min"] == jax["duration_min"] == 1440
    assert cupy["snapshot_count"] == jax["snapshot_count"] == 25
    assert parity["passed"] is True
    assert all(parity["gates"].values())


def test_raw_full_run_arrays_are_compatible() -> None:
    with np.load(RUN / "cupy/flood_simulation_outputs.npz", allow_pickle=False) as cupy, np.load(
        RUN / "jax/flood_simulation_outputs.npz", allow_pickle=False
    ) as jax:
        assert cupy["depth_snapshots_m"].shape == jax["depth_snapshots_m"].shape == (25, 981_880)
        assert cupy["depth_snapshots_m"].dtype == jax["depth_snapshots_m"].dtype == np.float32
        assert np.isfinite(cupy["depth_snapshots_m"]).all()
        assert np.isfinite(jax["depth_snapshots_m"]).all()
        assert np.min(cupy["depth_snapshots_m"]) >= -1e-7
        assert np.min(jax["depth_snapshots_m"]) >= -1e-7


def test_cupy_selects_cuda_device_zero() -> None:
    cupy = pytest.importorskip("cupy")
    if cupy.cuda.runtime.getDeviceCount() < 1:
        pytest.fail("Production release requires CUDA; CPU fallback is forbidden")
    assert cupy.cuda.runtime.getDevice() == 0
    assert cupy.asarray([1.0]).device.id == 0


def test_jax_selects_gpu() -> None:
    jax = pytest.importorskip("jax")
    devices = jax.devices()
    assert devices, "Production release requires a JAX device"
    assert all(device.platform == "gpu" for device in devices)
