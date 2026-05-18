from pathlib import Path

import numpy as np
from workflow import create_model


def _terminal_metric(analysis, name: str):
    metric = (analysis.output.metrics or {}).get(name) if analysis is not None else None
    values = getattr(metric, "values", None)
    return values[-1] if isinstance(values, list) and values else None


BASE = Path(__file__).resolve().parent
model = create_model(
    circuit_config=BASE / "circuit.yaml",
    solver_config=BASE / "solver.yaml",
    device_config=BASE / "device.yaml",
    pulse_config=BASE / "pulse.yaml",
    analyser_config=BASE / "analyser.yaml",
)
model.config.output.out_dir = str((BASE.parent / "runs" / "task6_two_qubit_cz_bell").resolve())
model.run_all()
analysis = model.get_analysis()
trajectory = model.get_trajectory()
rho_end = np.array(trajectory.density_matrix["snapshots"][-1], dtype=complex)
comp_idx = [0, 1, 3, 4]
rho_comp = rho_end[np.ix_(comp_idx, comp_idx)]
phase_optimized_phi_fidelity = float(
    0.5 * (np.real(rho_comp[0, 0]) + np.real(rho_comp[3, 3]) + 2.0 * abs(rho_comp[0, 3]))
)

print(model.out_dir)
print(
    {
        "engine": trajectory.engine if trajectory else "",
        "mean_excited": _terminal_metric(analysis, "mean_excited"),
        "variance": _terminal_metric(analysis, "variance"),
        "p00": float(np.real(rho_comp[0, 0])),
        "p11": float(np.real(rho_comp[3, 3])),
        "phase_optimized_phi_fidelity": phase_optimized_phi_fidelity,
    }
)
