# GPU-Accelerated Flood Project — Stopped Checkpoint Report

**Checkpoint date:** 2026-09-03 (Asia/Kolkata)  
**State:** Work stopped on the user's instruction. No benchmark, simulation, or training process remains active.

## Executive status

The scientific-final release is **partially completed and intentionally not labelled final**. All completed evidence has been preserved. The pre-existing final and completed packages and archives remain unchanged and their archive hashes still match the values recorded below.

Completed at this checkpoint:

- initial release, archive, environment, and test audit;
- ANUGA 3.3.10 installation and independent-reference validation on compatible controlled cases;
- correction of the ANUGA smoke-test skip/configuration behavior;
- advance registration of nine controlled synthetic rainfall scenarios and complete-scenario train/validation/test splits;
- JAX full-state history serialization support for depth and both momentum components;
- explicit benchmark phase timing for CuPy and JAX;
- full official 24-hour CuPy benchmark: 3 warm-ups plus 10 accepted measured repetitions;
- partial official 24-hour JAX benchmark: 3 warm-ups plus 3 accepted measured repetitions;
- regression tests and scripts for scenario histories, AI study, environment capture, and final release construction.

Not completed at this checkpoint:

- JAX measured repetitions 4–10 of the official benchmark;
- the official paired benchmark summary and inferential statistics (these require all repetitions);
- nine full scenario-history simulations and representative CuPy/JAX scenario parity checks;
- AI training, validation, test evaluation, checkpoint, and inference benchmark;
- final full test-suite rerun after all modifications;
- authoritative PDF/report, slide deck, manifest, checksums, archive, and extraction verification for `HF_CUPY_JAX_SCIENTIFIC_FINAL`.

## Software verification

The preserved baseline verification run collected 148 tests:

- 111 passed
- 37 skipped under the documented legacy/opposite-hardware policy
- 0 failed
- 0 errors
- runtime: 10.58 s

After the audit, the blanket skip affecting five ANUGA smoke tests was removed. The five ANUGA smoke tests subsequently passed. Two new scientific-workflow regression tests also passed. Because the user stopped the work before the final whole-suite rerun, these focused results must not be substituted for a new consolidated suite count.

## Independent ANUGA numerical-reference work

ANUGA 3.3.10 was used only where the model physics could be made compatible. Results:

| Case | Depth RMSE (m) | Volume difference (m³) | Classification |
|---|---:|---:|---|
| Still water | 2.980232e-9 | 2.980232e-9 | Registered criterion passed |
| Rainfall only | 5.518733e-12 | 5.518733e-12 | Registered criterion passed |
| Moving-water dam break | 6.398733e-3 | -3.725290e-9 | Descriptive cross-method comparison |
| Terrain lake at rest | maximum stage drift 0 m | — | ANUGA-only diagnostic |

The full production-domain ANUGA run was not executed because the adapter cannot identically represent the production drainage, recharge, surcharge, outfall, external-hydrograph, and outflow-only boundary system. Running a partial substitute would answer a different scientific question. These results support implementation verification, not validation against observed flooding.

## Controlled scenario protocol

Nine controlled synthetic rainfall-forcing scenarios were registered before simulation:

- five training scenarios;
- two validation scenarios;
- two test scenarios;
- complete scenarios are the split unit, preventing time-window leakage between splits.

The scenarios include intensity scaling, early/late peaks, bimodal structure, and concentrated rainfall. Their manifest and split definition are included in `evidence/`.

## Official 24-hour GPU benchmark checkpoint

Protocol: 3 warm-ups and 10 measured repetitions per backend, with correctness checks on every accepted run. The job was stopped during JAX measured repetition 4; that interrupted repetition was not accepted as evidence.

### CuPy — complete

- Warm-ups completed: 3/3
- Accepted measurements: 10/10
- Solver runtimes (s): 424.547, 434.508, 425.194, 424.650, 434.960, 435.921, 424.907, 423.845, 427.126, 426.984
- Median solver runtime: **426.089 s**
- All ten correctness checks passed
- RMSE range: 0.000355731–0.000489993 m
- minimum CSI range: 0.997487634–0.998743521

### JAX — incomplete

- Warm-ups completed: 3/3
- Accepted measurements: 3/10
- Solver runtimes (s): 448.182, 448.328, 448.340
- Partial median: **448.328 s**
- All three completed correctness checks passed
- RMSE range: 0.000356962–0.000493843 m
- minimum CSI range: 0.997487437–0.998268126

The partial JAX median must not be presented as the official final backend comparison. No speedup claim, confidence interval, or significance test is made from this incomplete series.

## Previously completed 24-hour repeatability evidence

The earlier completed release preserved these full-run repeatability results:

- CuPy: RMSE 0.000416765878 m, CSI 0.998739661, timestep-count difference 28.
- JAX: RMSE 0.000406099634 m, CSI 0.998031186, timestep-count difference 24.

The earlier secondary 60-minute benchmark used 3 warm-ups plus 10 measured runs per backend:

- CuPy median: 17.267041 s
- JAX median: 18.212524 s
- JAX was 5.475648% slower in that secondary benchmark.

## Preserved package integrity

- `HF_CUPY_JAX_FINAL_PACKAGE.tar.gz`: `59ca07cdd99a944929dd4ac1a5e201ea421189969658fdc0b4dd8f375baa8ba6`
- `HF_CUPY_JAX_COMPLETED_PROJECT.tar.gz`: `d9b4656055b728de582be8d5fb53892cff659d7ef5a5f29312365bb15bb7476c`

## Evidence locations

Key human-readable evidence is copied into this folder under `evidence/`. Full raw results remain at:

- `../scientific_final_work/benchmarks/official_24h/raw/`
- `../scientific_final_work/validation/anuga/`
- `../scientific_final_work/scenarios/`
- `../scientific_final_work/tests/`

The complete partial scientific workspace occupies approximately 3.8 GB. Raw solver arrays were left in their original evidence tree to avoid duplicating several gigabytes in this checkpoint folder.

## Safe continuation point

The official benchmark runner is resumable. A future continuation should start with JAX measured repetition 4 and preserve the existing 3 JAX measurements and all 10 CuPy measurements. After repetitions 4–10, the remaining scenario-history, AI, full-test, and release-packaging stages can proceed. All final claims must continue to distinguish numerical verification, software testing, and observational validation.
