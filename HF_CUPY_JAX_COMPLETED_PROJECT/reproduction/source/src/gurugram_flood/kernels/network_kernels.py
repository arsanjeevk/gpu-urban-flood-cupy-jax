"""Backend-neutral network continuity kernels for drainage coupling."""

from __future__ import annotations

from typing import Any

import numpy as np


def _xp_from_array(array: Any):
    module_name = type(array).__module__
    module_root = module_name.split(".", maxsplit=1)[0]
    if module_root == "cupy":
        import cupy as cp  # type: ignore[import-not-found]

        return cp
    return np


def scatter_add_1d(indices: Any, values: Any, size: int) -> Any:
    """Return a 1D scatter-add accumulation vector.

    This is the backend-neutral primitive needed for node continuity on GPU.
    NumPy uses ``np.add.at``; CuPy uses ``cupyx.scatter_add`` (CUDA atomicAdd).
    ``cupy.scatter_add`` was removed from the top-level ``cupy`` namespace in
    newer CuPy releases (kept only under ``cupyx``).
    """

    xp = _xp_from_array(values)
    if size < 0:
        raise ValueError("size must be non-negative.")
    if xp is np:
        out = np.zeros((size,), dtype=np.asarray(values).dtype)
        np.add.at(out, np.asarray(indices, dtype=int), np.asarray(values))
        return out
    # CuPy path — cupyx.scatter_add() uses CUDA atomicAdd, equivalent to np.add.at().
    import cupy as cp
    import cupyx

    out = cp.zeros((size,), dtype=values.dtype)
    cupyx.scatter_add(out, indices, values)
    return out


def limit_link_transfers_by_available_volume(
    requested_transfer_m3: Any,
    from_node_index: Any,
    to_node_index: Any,
    node_volume_m3: Any,
) -> Any:
    """Scale simultaneous link transfers so no donor node is overdrawn.

    Positive transfer moves water from `from_node` to `to_node`; negative
    transfer reverses direction. Requests from the same donor node are scaled
    proportionally when their sum exceeds available volume.
    """

    xp = _xp_from_array(requested_transfer_m3)
    transfer = requested_transfer_m3
    positive = transfer >= 0.0
    donor_node = xp.where(positive, from_node_index, to_node_index)
    requested_abs = xp.abs(transfer)
    requested_by_donor = scatter_add_1d(donor_node, requested_abs, int(node_volume_m3.shape[0]))
    factor_by_node = xp.minimum(1.0, node_volume_m3 / xp.maximum(requested_by_donor, 1.0e-12))
    limited_abs = requested_abs * factor_by_node[donor_node]
    sign = xp.where(positive, 1.0, -1.0)
    return sign * limited_abs


def node_delta_from_link_transfers(
    transfer_m3: Any,
    from_node_index: Any,
    to_node_index: Any,
    node_count: int,
) -> Any:
    """Return node volume deltas from signed link transfers."""

    xp = _xp_from_array(transfer_m3)
    indices = xp.concatenate([from_node_index, to_node_index])
    values = xp.concatenate([-transfer_m3, transfer_m3])
    return scatter_add_1d(indices, values, node_count)


def limit_node_exchange_by_available_volume(
    requested_exchange_m3: Any,
    node_index: Any,
    node_volume_m3: Any,
) -> Any:
    """Limit same-node volume requests so total withdrawal cannot exceed storage."""

    xp = _xp_from_array(requested_exchange_m3)
    requested = xp.maximum(requested_exchange_m3, 0.0)
    requested_by_node = scatter_add_1d(node_index, requested, int(node_volume_m3.shape[0]))
    factor_by_node = xp.minimum(1.0, node_volume_m3 / xp.maximum(requested_by_node, 1.0e-12))
    return requested * factor_by_node[node_index]


def node_delta_from_cell_exchange(
    exchange_m3: Any,
    node_index: Any,
    node_count: int,
) -> Any:
    """Return node deltas from cell-node exchange.

    `node_index` may contain `-1` for cells without inlets. Positive exchange
    adds water to the node; negative exchange removes water from it.
    """

    xp = _xp_from_array(exchange_m3)
    valid = node_index >= 0
    safe_index = xp.where(valid, node_index, 0)
    safe_exchange = xp.where(valid, exchange_m3, 0.0)
    return scatter_add_1d(safe_index, safe_exchange, node_count)

