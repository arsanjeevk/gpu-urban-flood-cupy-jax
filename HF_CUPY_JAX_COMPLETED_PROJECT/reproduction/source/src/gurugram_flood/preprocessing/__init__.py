"""Input preprocessing: rainfall CSV (runtime) and static city datasets."""

from .city_data import CityDomain, load_city_domain
from .rainfall import RainfallEvent, load_rainfall_event

__all__ = ["CityDomain", "load_city_domain", "RainfallEvent", "load_rainfall_event"]
