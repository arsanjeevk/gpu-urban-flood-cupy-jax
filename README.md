# GPU-Accelerated Physics-Based Urban Flood Modelling

This repository contains the final report, presentation, implementation source, and curated
evidence for a production comparison of CuPy/CUDA and JAX/XLA implementations of the same
physics-based urban flood model.

## Main results

- Production domain: 981,880 triangles, 817,573 active triangles, and 297.965823 km² active area.
- Drainage system: 118,768 nodes, 139,798 links, 61,317 mapped inlets, and 46 enabled outfalls.
- Experiment: 160.09 mm rainfall over 24 hours, float32 surface fields, NVIDIA RTX PRO 4000
  Blackwell GPU, and no CPU fallback.
- Numerical parity: all eight registered CuPy–JAX acceptance gates passed.
- Repeatability: both backends were numerically repeatable within the approved criteria, but
  neither was bitwise deterministic.
- Secondary 60-minute benchmark: CuPy median 17.267041 s; JAX median 18.212524 s; JAX was
  5.475648% slower for this specific workload and environment.
- Software verification: 111 passed, 37 explicitly skipped, 0 failed, and 0 errors.

These results establish GPU implementation consistency and numerical verification. They do
not establish agreement with observed flooding, operational forecasting skill, or production
AI performance.

## Report and presentation

- [Final report](reports/compact_final/final_report.pdf)
- [LaTeX report source](reports/compact_final/final_report.tex)
- [Presentation](reports/compact_final/GPU_Accelerated_Urban_Flood_Modelling_15_Slides.pdf)
- [Presentation source](reports/compact_final/presentation_15_slides.tex)

To rebuild the report, run from `reports/compact_final/`:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error final_report.tex
```

The secondary-benchmark plot can be regenerated from its verified CSV record with:

```bash
python build_runtime_figure.py
```

## Repository layout

- `reports/compact_final/`: final report, slides, figures, logos, and plot-generation script.
- `HF_CUPY_JAX_COMPLETED_PROJECT/`: curated production evidence and preserved implementation
  source used by the report.
- `Ravi-20260903T125449Z-1-001/`: limited ANUGA reference evidence and the interrupted official
  24-hour benchmark progress record.

Start the evidence review with
[`COMPLETION_ADDENDUM.md`](HF_CUPY_JAX_COMPLETED_PROJECT/COMPLETION_ADDENDUM.md).

## Data availability and reproducibility boundary

Large production inputs, raw measured-run directories, and full NPZ state archives are not
stored in ordinary Git because individual files exceed GitHub's standard file-size limit and
the underlying GIS data may require sharing approval. Their manifests, hashes, configurations,
summary tables, run reports, and analysis source are retained here. The complete artifacts are
preserved separately in the project backup and should be deposited in an approved research-data
repository or Git LFS only after data-sharing permission is confirmed.

The preserved execution source is under
`HF_CUPY_JAX_COMPLETED_PROJECT/reproduction/source/`. Re-executing the complete production
simulation requires the separately retained production inputs and a compatible NVIDIA CUDA
environment. Exact recorded software and hardware versions are under
`HF_CUPY_JAX_COMPLETED_PROJECT/environment/`.

## Scientific limitations

No compatible observed flood depths, extents, discharges, or arrival times were available.
ANUGA contributes only a limited independent numerical reference on compatible small cases.
The repeated official 24-hour timing experiment remains incomplete; the completed repeated
performance result is the explicitly labelled secondary 60-minute benchmark.

