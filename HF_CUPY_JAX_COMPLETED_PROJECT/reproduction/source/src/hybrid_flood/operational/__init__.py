"""Operational real-scenario interfaces for the Gurugram V2 product."""

from .rainfall import RainfallEvent, load_rainfall_csv

__all__ = ["RainfallEvent", "load_rainfall_csv"]
