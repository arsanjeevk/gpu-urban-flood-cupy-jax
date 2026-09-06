"""Exception hierarchy for the Gurugram flood backend.

Every error raised deliberately by this package derives from
:class:`GurugramFloodError`, so callers (CLI, web service) can catch one
type and translate it into a clean user-facing message / HTTP status.
"""

from __future__ import annotations


class GurugramFloodError(Exception):
    """Base class for all errors raised by the gurugram_flood package."""


class ConfigError(GurugramFloodError):
    """Invalid or inconsistent run configuration."""


class DataFileMissingError(GurugramFloodError):
    """A required static city dataset was not found on disk."""

    def __init__(self, path, role: str):
        self.path = str(path)
        self.role = role
        super().__init__(
            f"Required data file for '{role}' not found: {self.path}. "
            "Check that the data/ directory is complete or set paths.data_dir "
            "in the run configuration."
        )


class RainfallInputError(GurugramFloodError):
    """The user-supplied rainfall CSV is malformed or physically implausible."""


class BackendUnavailableError(GurugramFloodError):
    """The requested array backend (e.g. CuPy/CUDA) could not be initialised."""


class SimulationError(GurugramFloodError):
    """The solver failed mid-run (e.g. non-finite depths detected)."""
