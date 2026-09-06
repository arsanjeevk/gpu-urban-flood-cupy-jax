# Full-production repeatability report

Two independent 24-hour executions were compared for each backend. No new tolerance was invented; classification uses the existing parity criteria (RMSE ≤0.05 m, CSI ≥0.95, snapshot offset ≤1 s). Byte-level identity is reported separately.

## CUPY

Classification: **numerically_deterministic_within_existing_parity_criteria**. Byte-identical archive: False. RMSE 0.000416765878 m; max absolute depth difference 0.283989169 m; minimum CSI 0.998739661; timestep difference +28; maximum snapshot offset 0.354492 s.

## JAX

Classification: **numerically_deterministic_within_existing_parity_criteria**. Byte-identical archive: False. RMSE 0.000406099634 m; max absolute depth difference 0.603011927 m; minimum CSI 0.998031186; timestep difference +24; maximum snapshot offset 0.337891 s.

Differences correlate with timestep-count/snapshot-time divergence. The available evidence does not isolate the lower-level cause among GPU reduction ordering and wet/dry transition sensitivity, so no stronger causal claim is made.
