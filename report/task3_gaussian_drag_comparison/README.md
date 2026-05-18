# Task 3: Gaussian and DRAG Pulse Comparison

This report compares single-qubit Gaussian and DRAG `X` pulses in a three-level transmon model.

The notebook generates:

- `figures/waveform_population.png`: Gaussian/DRAG waveforms and three-level population evolution.
- `figures/beta_gate_time_scan.png`: DRAG beta scan and Gaussian/DRAG gate-time ratio scan.

Here the gate error is the single-input state-preparation error after an `X` gate,
`1 - P1(final)`, and leakage is `P2(final)`.
