"""Simulation engine wrapping the fused CUDA triangular SWE + pipe-network kernels."""

from .engine import SimulationResult, run_simulation
from .stabilizers import clamp_velocity, to_host

__all__ = ["SimulationResult", "run_simulation", "clamp_velocity", "to_host"]
