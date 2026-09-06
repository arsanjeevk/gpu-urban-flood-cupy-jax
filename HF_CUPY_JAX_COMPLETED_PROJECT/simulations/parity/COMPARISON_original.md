# CuPy versus JAX/XLA: 24-hour production comparison

## Decision summary

Both GPU solvers ran the same 981,880-cell production topology and the supplied 24-hour, 160.09 mm rainfall. JAX passed every production-parity gate. CuPy was faster in this single run: 430.467 s versus 451.865 s, so JAX took 4.97% longer. Choose CuPy when this exact fused-kernel implementation's lowest measured runtime is the primary requirement. Choose JAX when solver development, composability, accelerator portability, automatic differentiation, or coupling with AI/optimization workflows is more important than this measured ~5% runtime difference.

This comparison is numerical equivalence, not observational calibration against measured flood depths.

## Inputs and physics held equal

- Rainfall: `/home/Ravi/FloodAstra/V1/HF-V1V2/examples/user_24h_rainfall.csv`; 24 hourly intervals; 160.09 mm total; 24 mm/h peak.
- Mesh: 981,880 total triangles; 817,573 active triangles.
- Terrain/bed elevation and spatial Manning roughness.
- Horton/BCR infiltration and surface recharge exchange.
- 118,768 drainage nodes, 139,798 pipe links, and 61,317 mapped inlets.
- Pipe drainage, inlet capture, surcharge return, and 46 enabled outfalls.
- Inactive cells as walls and 34,390 exterior code-4 outflow-only edges.
- Historical external boundary hydrograph enabled by the production parity profile.
- Full shallow-water equations, CFL 0.85, and maximum time step 1 s.
- Dry surface initial state and the same production drainage initial state.

## Measured results

| Metric | CuPy | JAX/XLA | Difference |
|---|---:|---:|---:|
| Solver runtime | 430.467 s | 451.865 s | JAX +21.398 s (+4.97%) |
| Accepted steps | 321,645 | 321,640 | -5 |
| Maximum depth | 9.414836884 m | 9.414829254 m | -0.000007629 m |
| Final wet triangles | 129,187 | 129,180 | -7 |
| Compatible mass residual | 0.269684% | 0.291569% | +0.021886 pp |

JAX's complete mass ledger, which also accounts for its explicit recharge-storage state, has a residual ratio of 0.166317%.

## Numerical parity

- Final depth RMSE: 0.979 mm.
- Maximum hourly depth RMSE: 0.979 mm (limit 50 mm).
- Final flood-extent CSI: 0.999590; minimum hourly CSI: 0.998031 (limit 0.95).
- Final surface-volume error: -0.00747%; maximum hourly absolute error: 0.03329% (limit 2%).
- Maximum snapshot alignment difference: 0.318359 s (limit 1 s).
- Maximum source/sink ledger difference: 1.5437% (limit 2%).

### Validation gates

- PASS — `all_hour_csi_ge_0_95`
- PASS — `all_hour_depth_rmse_le_0_05m`
- PASS — `all_hour_volume_error_le_2pct`
- PASS — `finite_non_negative_depth`
- PASS — `mass_residual_regression_le_0_1pct`
- PASS — `overall_max_depth_error_le_5pct`
- PASS — `snapshot_alignment_le_1s`
- PASS — `source_sink_ledger_error_le_2pct`

## Why use JAX

- The production time loop is expressed as a compiled XLA `while` computation, allowing whole-program optimization without maintaining handwritten CUDA C kernels for every new operation.
- Pure array/state transformations are easier to compose with calibration, optimization, surrogate models, and other JAX-based ML workflows.
- Automatic differentiation is available for differentiable parts of future parameter-estimation or sensitivity-analysis work. The present flood run itself is a forward parity run and does not claim that every discrete operation is differentiable.
- The same programming model can target supported GPU/accelerator backends, reducing solver-level CUDA-specific code.
- JAX's functional state model and compilation boundaries make experimental physics changes easier to test and transform.

## Why retain CuPy

- It was the fastest implementation in this run, by 4.97%.
- Its fused custom CUDA kernels are mature and purpose-built for this exact production workload.
- CuPy is the production numerical reference used by the parity validator, so it remains essential for regression checking.

## Interpretation limits

- Runtime figures are one run per backend, not a repeated benchmark with uncertainty bounds.
- Parity means the JAX implementation reproduces CuPy closely; it does not prove agreement with field observations.
- Maximum depth is a domain-wide numerical extreme and should not alone be interpreted as a city-wide typical flood depth.
- Historical boundary inflow is enabled in addition to the user rainfall because this run uses the exact production parity boundary profile.

## Folder contents

- `comparison_summary.json`: machine-readable headline, physics, runtime, ledger, and drainage results.
- `hourly_comparison.csv`: all 25 aligned snapshots and parity metrics.
- `figures/`: maximum depth, wet extent, RMSE, CSI, and surface-volume plots.
- `cupy/`: CuPy report and full simulation output archive.
- `jax/`: JAX report, full simulation output archive, manifest, and parity validation.
- `artifact_manifest.json`: artifact sizes and SHA-256 checksums.

## Reproduction commands

Run from `/home/Ravi/FloodAstra/V1/HF-V1V2` with the project virtual environment:

```bash
.venv/bin/python -m gurugram_flood.cli run --rainfall examples/user_24h_rainfall.csv --config configs/production_parity_gpu.yaml --output-dir comparisons/user_24h_160_09mm/cupy
.venv/bin/python -m hybrid_flood.production_gpu --backend jax --rainfall examples/user_24h_rainfall.csv --config configs/production_parity_gpu.yaml --output-dir comparisons/user_24h_160_09mm/jax --reference-dir comparisons/user_24h_160_09mm/cupy
.venv/bin/python scripts/build_production_comparison.py comparisons/user_24h_160_09mm --rainfall examples/user_24h_rainfall.csv
```
