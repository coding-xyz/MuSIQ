# Task 10 Two-Qubit Gates

This task uses an explicit three-transmon model with two logical qubits (`q0`, `q1`)
and one physical coupler (`c0`).

`run.py` now does two report-style studies:

1. Single-qubit calibration on `q0`, following the spirit of task 9.
   It scans `sx` and `x` pulse amplitudes and picks the best value from the final
   populations.
2. Two-qubit gate-time sweeps, following the spirit of task 1.
   It sweeps the `cz` and `iswap` gate durations and records the final populations.

Run:

```powershell
python report/task10_two_qubit_gates/run.py
```

Outputs:

- `report/task10_two_qubit_gates/figures/sx_calibration_q0.png`
- `report/task10_two_qubit_gates/figures/x_calibration_q0.png`
- `report/task10_two_qubit_gates/figures/cz_final_population_vs_gate_time.png`
- `report/task10_two_qubit_gates/figures/iswap_final_population_vs_gate_time.png`
- `report/task10_two_qubit_gates/figures/task10_summary.json`
