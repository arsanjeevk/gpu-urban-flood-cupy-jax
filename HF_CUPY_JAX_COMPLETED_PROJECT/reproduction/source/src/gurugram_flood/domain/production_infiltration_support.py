"""Production-style Horton infiltration and rainfall multiplier support."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ProductionInfiltrationFields:
    """Per-cell arrays needed for production Horton infiltration and rainfall scaling."""

    rainfall_multiplier: np.ndarray   # shape: n_cells, float32
    horton_f0_mps: np.ndarray         # initial infiltration rate (m/s), float32
    horton_fc_mps: np.ndarray         # steady-state rate (m/s), float32
    horton_k_per_s: np.ndarray        # Horton decay constant (1/s), float32
    open_fraction: np.ndarray         # 1 - building_fraction, float32
    active_mask: np.ndarray           # bool, shape: n_cells


# Module-level cache: id(fields) → dict of GPU arrays, keyed by xp module id.
# Avoids repeated CPU→GPU transfers inside the hot simulation loop.
_GPU_CACHE: dict[int, dict[int, dict[str, Any]]] = {}


def _gpu_fields(fields: ProductionInfiltrationFields, xp: Any) -> dict[str, Any]:
    """Return GPU copies of all field arrays, cached per (fields, xp) pair."""
    fid = id(fields)
    xid = id(xp)
    if fid not in _GPU_CACHE or xid not in _GPU_CACHE[fid]:
        _GPU_CACHE.setdefault(fid, {})[xid] = {
            "rm": xp.asarray(fields.rainfall_multiplier, dtype=np.float32),
            "of": xp.asarray(fields.open_fraction, dtype=np.float32),
            "f0": xp.asarray(fields.horton_f0_mps, dtype=np.float32),
            "fc": xp.asarray(fields.horton_fc_mps, dtype=np.float32),
            "k":  xp.asarray(fields.horton_k_per_s, dtype=np.float32),
        }
    return _GPU_CACHE[fid][xid]


def prepare_production_infiltration_fields(
    *,
    active_mask: np.ndarray,
    rainfall_multiplier: np.ndarray | None,
    horton_f0_mmhr: np.ndarray | None,
    horton_fc_mmhr: np.ndarray | None,
    horton_decay_per_hr: np.ndarray | None,
    building_fraction: np.ndarray | None,
    impervious_fraction: np.ndarray | None,
    use_bcr_open_fraction: bool = True,
    min_open_fraction: float = 0.05,
) -> ProductionInfiltrationFields:
    """Build per-cell infiltration / rainfall fields from production export arrays."""

    n = len(active_mask)
    mmhr_to_mps = 1.0 / (1000.0 * 3600.0)

    if rainfall_multiplier is not None and rainfall_multiplier.size == n:
        rm = np.asarray(rainfall_multiplier, dtype=np.float32)
    else:
        rm = np.ones(n, dtype=np.float32)

    if use_bcr_open_fraction and building_fraction is not None and building_fraction.size == n:
        bf = np.asarray(building_fraction, dtype=np.float32)
        open_frac = np.clip(1.0 - bf, min_open_fraction, 1.0).astype(np.float32)
    else:
        open_frac = np.ones(n, dtype=np.float32)

    if horton_f0_mmhr is not None and horton_f0_mmhr.size == n:
        f0 = (np.asarray(horton_f0_mmhr, dtype=np.float64) * mmhr_to_mps).astype(np.float32)
    else:
        f0 = np.full(n, 6.5 * mmhr_to_mps, dtype=np.float32)

    if horton_fc_mmhr is not None and horton_fc_mmhr.size == n:
        fc = (np.asarray(horton_fc_mmhr, dtype=np.float64) * mmhr_to_mps).astype(np.float32)
    else:
        fc = np.full(n, 1.0 * mmhr_to_mps, dtype=np.float32)

    if horton_decay_per_hr is not None and horton_decay_per_hr.size == n:
        k = (np.asarray(horton_decay_per_hr, dtype=np.float64) / 3600.0).astype(np.float32)
    else:
        k = np.full(n, 2.0 / 3600.0, dtype=np.float32)

    return ProductionInfiltrationFields(
        rainfall_multiplier=rm,
        horton_f0_mps=f0,
        horton_fc_mps=fc,
        horton_k_per_s=k,
        open_fraction=open_frac,
        active_mask=active_mask.astype(bool),
    )


def rainfall_rate_for_step(
    *,
    xp: Any,
    base_rainfall_rate_mps: float,
    mode: str,
    fields: ProductionInfiltrationFields | None,
) -> Any:
    """Return per-cell effective rainfall rate (m/s) for this timestep."""
    if mode == "production_horton" and fields is not None:
        g = _gpu_fields(fields, xp)
        return np.float32(base_rainfall_rate_mps) * g["rm"]
    return base_rainfall_rate_mps


def build_infiltration_capacity_for_step(
    *,
    xp: Any,
    mode: str,
    wet_time_s: Any,
    fields: ProductionInfiltrationFields | None,
    fallback_capacity_mps: Any,
) -> Any:
    """Return per-cell infiltration capacity (m/s) via Horton decay.

    f(t) = fc + (f0 - fc) * exp(-k * wet_time_s)
    """
    if mode == "production_horton" and fields is not None:
        g = _gpu_fields(fields, xp)
        f0, fc, k = g["f0"], g["fc"], g["k"]
        wt = xp.asarray(wet_time_s, dtype=np.float32)
        return fc + (f0 - fc) * xp.exp(-k * wt)
    return fallback_capacity_mps


def wet_time_increment_s(
    *,
    xp: Any,
    depth_m: Any,
    dt_s: float,
    min_depth_m: float = 1.0e-5,
) -> Any:
    """Return per-cell wet-time increment (s): dt_s where wet, 0 where dry."""
    return np.float32(dt_s) * (depth_m > min_depth_m)
