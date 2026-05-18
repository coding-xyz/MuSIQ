from pathlib import Path

from workflow import create_model


if __name__ == "__main__":
    base = Path("examples/noise_simulation_tests/task1")
    model = create_model(
        circuit_config=base / "circuit.yaml",
        solver_config=base / "qutip.yaml",
        device_config=base / "device.yaml",
        pulse_config=base / "pulse.yaml",
    )
    model.run()
    trajectory = model.get_trajectory()
    print(trajectory.engine if trajectory else "")
