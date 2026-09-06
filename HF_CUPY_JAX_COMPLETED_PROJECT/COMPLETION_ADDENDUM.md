# Completed production verification addendum

## Outcome

The production-only numerical release is verified. The complete test run recorded 111 passed, 37 explicitly skipped, 0 failed, and 0 errors. The prior `HF_CUPY_JAX_FINAL_PACKAGE` remains unchanged; this is a separate evidence package.

## Full-run repeatability

Independent 24-hour run B was completed for both GPU backends. Neither archive is byte-identical because adaptive timestep counts diverged, but both pass the pre-existing numerical criteria:

- CuPy: RMSE 0.000416766 m; minimum CSI 0.998739661; timestep difference +28.
- JAX: RMSE 0.000406100 m; minimum CSI 0.998031186; timestep difference +24.

This supports numerical repeatability within the approved parity gates, not bitwise determinism.

## Repeated GPU performance

A secondary identical 60-minute production workload completed 3 warm-ups and 10 measurements per backend, with correctness checked after every run. Median solver runtimes were 17.267041 s for CuPy and 18.212524 s for JAX; JAX was 5.476% slower. This is deliberately labelled secondary and does not replace the unexecuted repeated 24-hour official benchmark.

## Remaining scientific limits

Independent physical validation remains unavailable, and one production rainfall event cannot support leakage-controlled production AI train/validation/test splits. No physical-accuracy, production-AI, or operational-readiness claim is made.

## Evidence paths

- `tests/pytest_output.txt`
- `simulations/repeatability/repeatability_metrics.json`
- `simulations/repeatability/repeatability_report.md`
- `benchmarks/secondary_60min/benchmark_results.csv`
- `benchmarks/secondary_60min/benchmark_summary.json`
