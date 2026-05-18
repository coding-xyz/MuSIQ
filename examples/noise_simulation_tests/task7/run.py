from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from workflow import create_model


BASE = Path(__file__).resolve().parent
OUT = BASE.parent / "runs" / "task7_heom_1overf_spin_echo"
OUT.mkdir(parents=True, exist_ok=True)

DELAY_NS = [90.0 * i for i in range(101)]


def _metrics(model, *, sequence: str, delay_ns: float, idle_duration_ns: float) -> dict[str, float | str]:
    trajectory = model.get_trajectory()
    rho = np.array(trajectory.density_matrix["snapshots"][-1], dtype=complex)
    p1 = float(np.real(rho[1, 1]))
    coherence = float(abs(rho[0, 1]))
    return {
        "sequence": sequence,
        "delay_ns": float(delay_ns),
        "idle_duration_ns": float(idle_duration_ns),
        "p0": float(np.real(rho[0, 0])),
        "p1": p1,
        "coherence_abs": coherence,
        "x_expect": 2.0 * float(np.real(rho[0, 1])),
        "out_dir": str(model.out_dir),
    }


def run_case(*, circuit_name: str, sequence: str, delay_ns: float):
    model = create_model(
        circuit_config=BASE / circuit_name,
        solver_config=BASE / "solver.yaml",
        device_config=BASE / "device.yaml",
        pulse_config=BASE / "pulse.yaml",
        analyser_config=BASE / "analyser.yaml",
    )
    idle_duration_ns = 0.5 * delay_ns if sequence == "echo" else delay_ns
    model.pulse.extras = dict(model.pulse.extras or {})
    model.pulse.extras["idle_duration_ns"] = float(idle_duration_ns)
    model.config.output.out_dir = str(OUT)
    model.run_all()
    return _metrics(model, sequence=sequence, delay_ns=delay_ns, idle_duration_ns=idle_duration_ns)


rows = []
for delay_ns in DELAY_NS:
    rows.append(
        run_case(
            circuit_name="circuit_ramsey.yaml",
            sequence="ramsey",
            delay_ns=delay_ns,
        )
    )
    rows.append(
        run_case(
            circuit_name="circuit_echo.yaml",
            sequence="echo",
            delay_ns=delay_ns,
        )
    )

csv_path = OUT / "delay_sweep.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

summary = {
    "engine": "qutip",
    "solver": "heom",
    "noise": "one_over_f",
    "delay_ns": DELAY_NS,
    "csv": str(csv_path),
    "rows": rows,
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(OUT)
print(json.dumps({"csv": str(csv_path), "num_points": len(rows)}, indent=2))
