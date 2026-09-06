"""Validated Gurugram domain adapters (ported unchanged from the deployment package).

These modules contain the numerically validated data adapters:

- ``real_gurugram_domain``: mesh loading, legacy drainage graph building,
  hyetograph helpers.
- ``production_topology_loader``: exact D_june_5 production topology NPZ loader.
- ``production_event_support``: external boundary inflow, node-volume scaling.
- ``production_infiltration_support``: production Horton infiltration +
  rainfall multiplier fields.
- ``june5_production_drainage``: June-5 drainage/recharge adapters.

Only import paths were rewritten during the deployment refactor; the
numerical logic is byte-identical to the notebook-validated versions.
Submodules are intentionally not imported here to keep optional heavy
dependencies (geopandas, rasterio) lazy.
"""
