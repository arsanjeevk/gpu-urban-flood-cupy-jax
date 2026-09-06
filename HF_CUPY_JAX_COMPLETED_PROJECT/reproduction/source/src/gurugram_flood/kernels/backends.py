"""Array backend selection for GPU flood kernels — CuPy/CUDA only."""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType

from gurugram_flood.kernels.config import BackendConfig


@dataclass(frozen=True)
class ArrayBackend:
    """Resolved array backend used by numerical kernels."""

    name: str
    xp: ModuleType
    is_gpu: bool


def get_array_backend(config: BackendConfig | None = None) -> ArrayBackend:
    """Resolve a NumPy-compatible array backend.

    GPU backend is CuPy (NVIDIA CUDA). Production execution never falls back
    to CPU because doing so would make benchmark and operational reports
    incomparable.
    """

    resolved_config = config or BackendConfig()
    if resolved_config.name in {"auto", "cupy"}:
        try:
            import cupy as cp  # type: ignore[import-not-found]

            if cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("No CUDA devices found.")
        except Exception as exc:
            raise RuntimeError("CuPy/CUDA production backend is unavailable.") from exc
        return ArrayBackend(name="cupy", xp=cp, is_gpu=True)

    raise ValueError(
        f"Unsupported production backend: {resolved_config.name!r}; only CuPy/CUDA is allowed."
    )
