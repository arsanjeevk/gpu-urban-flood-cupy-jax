"""Solver stabilisation helpers (moved verbatim from the run scripts)."""

from __future__ import annotations

from ..kernels.triangular_backend import TriangularSurfaceState


def clamp_velocity(state: "TriangularSurfaceState", max_speed_mps: float, xp) -> TriangularSurfaceState:
    """Cap velocity magnitude after a step, matching ANUGA DE0's `maximum_allowed_speed`.

    Direction is preserved; only magnitude is rescaled down where it exceeds
    the cap. This is a real production-solver stabilization technique (used
    by both the ANUGA DE0 run and the MLX `local_inertial` run in
    Gurugram_full_data), not introduced here -- it stops an isolated
    numerical shock (e.g. a drainage-node overflow dumping into one
    triangle) from setting an unrealistically tiny *global* CFL timestep
    for the whole mesh. The physics (full shallow-water equations) are
    unchanged; this only removes a non-physical artifact spike.
    """

    speed = xp.sqrt(state.hu_m2ps ** 2 + state.hv_m2ps ** 2) / xp.maximum(state.h_m, 1.0e-8)
    scale = xp.minimum(1.0, max_speed_mps / xp.maximum(speed, 1.0e-12))
    return TriangularSurfaceState(
        h_m=state.h_m,
        hu_m2ps=state.hu_m2ps * scale,
        hv_m2ps=state.hv_m2ps * scale,
    )


def to_host(arr):
    """Copy a backend array (CuPy or NumPy) to a host NumPy array."""
    import numpy as np

    return np.asarray(arr.get()) if hasattr(arr, "get") else np.asarray(arr)
