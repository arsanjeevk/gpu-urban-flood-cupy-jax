"""Stable error types shared by the CLI and HTTP service."""


class OperationalError(RuntimeError):
    """Base class for user-facing operational failures."""


class RainfallInputError(OperationalError):
    """Rainfall input is invalid or incomplete."""


class DataFileMissingError(OperationalError):
    """A required immutable model asset is unavailable."""


class ConfigurationError(OperationalError):
    """The requested model configuration is unsafe or unsupported."""


class BackendUnavailableError(OperationalError):
    """A required compute backend or checkpoint is unavailable."""


class SimulationError(OperationalError):
    """The numerical run failed validation."""
