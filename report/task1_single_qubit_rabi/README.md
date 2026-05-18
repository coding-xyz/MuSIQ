# Task 1: single-qubit state preparation and Rabi oscillation

This task follows the same workflow style as `examples/noise_simulation_tests`:
one task folder contains the notebook plus `circuit.yaml`, `device.yaml`,
`pulse.yaml`, `solver.yaml`, and `analyser.yaml`.

The model studies one qubit driven by one parametric `RX(theta)` gate. The
notebook sweeps the equivalent gate time / rotation angle, plots the model
population evolution, samples the lowered pulse, and shows state-preparation
fidelity for target states reachable from `|0>` with an `RX` pulse.

Notebook:

- `task1_single_qubit_rabi.ipynb`
