"""Gurugram citywide GPU (CUDA/CuPy) flood simulation backend.

Public API
----------
- :func:`run_flood_simulation` — rainfall CSV in, artifacts out.
- :func:`validate_rainfall_csv` — cheap upfront validation of user input.
- :func:`load_city` — pre-load static city data for repeated runs.
- :func:`load_config` / :class:`RunConfig` — configuration handling.
- :mod:`gurugram_flood.exceptions` — typed error hierarchy.
"""

from .api import FloodRunResult, load_city, run_flood_simulation, validate_rainfall_csv
from .config import RunConfig, load_config
from .exceptions import (
    BackendUnavailableError,
    ConfigError,
    DataFileMissingError,
    GurugramFloodError,
    RainfallInputError,
    SimulationError,
)
from .logging_utils import setup_logging

__version__ = "1.0.0"

__all__ = [
    "run_flood_simulation",
    "validate_rainfall_csv",
    "load_city",
    "FloodRunResult",
    "RunConfig",
    "load_config",
    "setup_logging",
    "GurugramFloodError",
    "ConfigError",
    "DataFileMissingError",
    "RainfallInputError",
    "BackendUnavailableError",
    "SimulationError",
    "__version__",
]
