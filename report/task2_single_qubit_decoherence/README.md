# Task 2: Single-Qubit Decoherence

This task mirrors the report task1 workflow, but keeps a single base
`device.yaml`. The notebook changes `q0.noise` in memory for each run to
demonstrate several Lindblad decoherence channels:

- T1 relaxation from the excited state.
- Tphi-only dephasing from an RX(pi/2) superposition.
- Tphi + T1 combined decay from an RX(pi/2) superposition.
- Ramsey and spin-echo readout from a detuned superposition.

Run `task2_single_qubit_decoherence.ipynb` from this directory or the repository
root to generate the comparison plots.
