# GPU-Accelerated Physics-Based Urban Flood Modelling

This repository presents a production-scale urban flood model implemented with two GPU
computing systems: **CuPy/CUDA** and **JAX/XLA**. Both implementations execute the same
physical model, numerical scheme, input data, precision, boundary treatment, and output
schedule. The study evaluates their numerical agreement, repeatability, and measured GPU
execution performance.

## Model and production experiment

The model solves the two-dimensional shallow-water equations on an unstructured triangular
mesh using a finite-volume method, hydrostatic reconstruction, and the HLL approximate
Riemann flux. The production calculation includes spatial bed elevation, Manning resistance,
wetting and drying, Horton infiltration, rainfall, recharge, dynamic drainage, inlet capture,
pipe conveyance, surcharge, outfalls, external inflow, and outflow-only exterior boundaries.

| Item | Production configuration |
|---|---:|
| Surface mesh | 981,880 triangles and 1,490,015 edges |
| Active domain | 817,573 triangles; 297.965823 km² |
| Drainage network | 118,768 nodes and 139,798 links |
| Surface–network connections | 61,317 mapped inlets |
| Enabled outfalls | 46 |
| Rainfall experiment | 160.09 mm over 24 hours |
| Numerical precision | float32 surface fields |
| CFL coefficient / maximum timestep | 0.85 / 1 s |
| Stored states | 25 hourly outputs |
| GPU | NVIDIA RTX PRO 4000 Blackwell |
| Execution policy | GPU only; no CPU fallback |

## Verified results

CuPy completed the production simulation in 321,645 adaptive timesteps and JAX in 321,640.
Their domain-wide maximum depths were 9.414836884 m and 9.414829254 m, respectively. Final
wet-cell counts at the 0.05 m threshold were 129,187 for CuPy and 129,180 for JAX.

All eight registered cross-backend numerical acceptance gates passed:

| Metric | Approved criterion | Observed result |
|---|---:|---:|
| Maximum hourly depth RMSE | ≤ 0.05 m | 0.000979348 m |
| Minimum hourly flood-extent CSI | ≥ 0.95 | 0.998031 |
| Maximum hourly surface-volume error | ≤ 2% | 0.033294% |
| Snapshot-time mismatch | ≤ 1 s | 0.318359 s |
| Source–sink ledger difference | ≤ 2% | 1.543653% |
| Maximum-depth relative error | ≤ 5% | 0.0000810% |
| Residual-ratio regression increase | ≤ 0.001 | 0.000218855 |
| Finite/non-negative depth | minimum ≥ −10⁻⁷ m | Passed |

Independent 24-hour repeat runs were numerically repeatable within the approved criteria but
were not byte-for-byte identical. CuPy repeatability produced depth RMSE 0.000416766 m and
minimum CSI 0.998740; JAX produced depth RMSE 0.000406100 m and minimum CSI 0.998031.

## GPU performance comparison

The completed secondary benchmark used the same 60-minute production workload for each
backend, with three warm-up runs followed by ten synchronized measurements. Every measured
run passed its correctness check.

| Backend | Median solver runtime | Median steps per second |
|---|---:|---:|
| CuPy/CUDA | 17.267041 s | 758.443 |
| JAX/XLA | 18.212524 s | 718.515 |

JAX was 5.475648% slower than CuPy for this specific workload, GPU, and software environment.
This result is hardware- and implementation-specific and is not a universal ranking of the
two GPU programming systems.

## Software verification

The final production software check collected 148 tests: **111 passed, 37 were explicitly
skipped, 0 failed, and 0 produced errors**. The retained evidence also includes machine-readable
parity tables, mass ledgers, repeatability metrics, benchmark measurements, run reports,
environment records, manifests, and checksums.

## Report and presentation

- [Final project report](reports/compact_final/final_report.pdf)
- [LaTeX report source](reports/compact_final/final_report.tex)
- [Fifteen-slide presentation](reports/compact_final/GPU_Accelerated_Urban_Flood_Modelling_15_Slides.pdf)
- [Presentation source](reports/compact_final/presentation_15_slides.tex)

The report can be rebuilt from `reports/compact_final/` with:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error final_report.tex
```

The secondary-benchmark figure can be regenerated from the retained CSV measurements with:

```bash
python build_runtime_figure.py
```

## Evidence and implementation source

The curated machine-readable evidence is stored in
[`HF_CUPY_JAX_COMPLETED_PROJECT`](HF_CUPY_JAX_COMPLETED_PROJECT/). The preserved production
implementation is under its `reproduction/source/` directory. Exact hardware and software
versions are retained in `environment/`, while numerical results are available in `tables/`,
`simulations/`, and `benchmarks/`.

Large GIS inputs, raw benchmark-run directories, and full NPZ state archives are maintained
separately because they exceed ordinary GitHub file-size limits and may require data-sharing
approval. Their configurations, manifests, hashes, summaries, and analysis code are retained
in this repository.

## Scientific scope

The completed work establishes numerical equivalence, repeatability, conservation diagnostics,
and a synchronized GPU performance comparison. It does not claim agreement with observed flood
measurements because compatible observations were unavailable. It also does not claim operational
forecasting performance or production AI results. ANUGA is used only as a limited external
numerical reference for compatible shallow-water cases and not as observational validation.
