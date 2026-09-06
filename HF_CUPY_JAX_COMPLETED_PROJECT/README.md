# Curated completed-project evidence

This directory contains the compact evidence subset used by the final report. The authoritative
final status is `COMPLETION_ADDENDUM.md`; machine-readable results are retained in `tables/`,
`benchmarks/`, `simulations/`, `tests/`, and `environment/`.

The original full package also contained large approved production inputs, raw benchmark-run
directories, and NPZ state archives. Those files are preserved in the external project backup but
are excluded from ordinary Git. `artifact_manifest.json`, `checksums.sha256`, the archive contents
listing, and the archive SHA-256 file describe that full preserved package; they are not a claim
that every large artifact is present in this compact repository.

The preserved implementation source is under `reproduction/source/`. Full execution requires the
separately retained approved input data and the recorded CUDA environment.

