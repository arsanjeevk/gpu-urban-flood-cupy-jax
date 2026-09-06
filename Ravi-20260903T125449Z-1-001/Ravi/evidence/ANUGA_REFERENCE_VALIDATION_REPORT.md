# ANUGA numerical-reference validation

ANUGA 3.3.10 was executed independently on identical two-triangle meshes for still-water, rainfall-only, and moving-water cases. The first two have directly comparable conservation/equilibrium interpretations; moving-water metrics are descriptive because numerical fluxes and reconstruction differ. A terrain lake-at-rest case was executed in ANUGA, but no cross-solver score is claimed because the controlled JAX solver does not implement the same bed-slope treatment. A full production comparison was assessed and refused: the production drainage, recharge, surcharge, outfalls, external hydrograph, and outflow-only boundary system cannot be represented identically by the current ANUGA adapter.

**ANUGA was used as an independent numerical reference for compatible shallow-water cases. This comparison supports numerical implementation verification but does not constitute validation against observed flooding.**

Machine-readable metrics are in `anuga_comparison.csv` and `anuga_summary.json`; raw ANUGA SWW and JSON outputs are under `raw/`.
