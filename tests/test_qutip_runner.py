import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from musiq.backend.config import DeviceConfig, NoiseConfig
from musiq.backend.model.noise import lower_noise
from musiq.engines.qutip.engine import QuTiPEngine
from musiq.engines.qutip.model.collapse import build_collapse_and_noise
from musiq.engines.qutip.modes.mcwf import run_mcwf_trajectory
from musiq.engines.qutip.runtime import QutipTrajectoryRequest


def _setup(*, dt: float, control_terms: list | None = None, readout_controls: list | None = None, qutip_options=None):
    return SimpleNamespace(
        dt=dt,
        qt=None,
        solver="se",
        run_config=SimpleNamespace(qutip_options=qutip_options),
        readout_controls=list(readout_controls or []),
        model_spec=SimpleNamespace(
            hamiltonian=SimpleNamespace(control_terms=list(control_terms or [])),
            readout=SimpleNamespace(controls=list(readout_controls or [])),
            analysis_request=SimpleNamespace(config={"trajectory": {}}),
        ),
    )


def test_resolve_trajectory_config_caps_max_step_to_dt_for_time_dependent_controls():
    engine = QuTiPEngine()
    setup = _setup(dt=1.0e-10, control_terms=[object()], qutip_options={})

    cfg = engine._resolve_trajectory_config(setup)

    assert cfg.options["store_states"] is True
    assert cfg.options["max_step"] == 1.0e-10


def test_resolve_trajectory_config_preserves_stricter_user_max_step():
    engine = QuTiPEngine()
    setup = _setup(dt=1.0e-10, control_terms=[object()], qutip_options={"max_step": 5.0e-11})

    cfg = engine._resolve_trajectory_config(setup)

    assert cfg.options["max_step"] == 5.0e-11


def test_resolve_trajectory_config_leaves_max_step_unset_without_controls():
    engine = QuTiPEngine()
    setup = _setup(dt=1.0e-10, control_terms=[], qutip_options={})

    cfg = engine._resolve_trajectory_config(setup)

    assert "max_step" not in cfg.options


class _FakeQobj:
    def __init__(self, data):
        self._data = np.asarray(data, dtype=complex)

    def full(self):
        return self._data

    def __add__(self, other):
        return _FakeQobj(self._data + other.full())

    def __mul__(self, scalar):
        return _FakeQobj(self._data * scalar)

    __rmul__ = __mul__


class _FakeQt:
    def __init__(self):
        self.calls = []

    @staticmethod
    def ket2dm(state):
        vec = np.asarray(state.full(), dtype=complex).reshape(-1, 1)
        return _FakeQobj(vec @ np.conjugate(vec.T))

    def mcsolve(self, H, psi0, tlist, *, c_ops, e_ops, ntraj, options, **kwargs):
        self.calls.append({"H": H, "ntraj": ntraj, "options": options, "kwargs": kwargs})
        shot_index = len(self.calls) - 1
        amp = min(0.9, 0.2 + 0.2 * shot_index)
        ket0 = _FakeQobj([[np.sqrt(max(0.0, 1.0 - amp))], [np.sqrt(amp)]])
        ket1 = _FakeQobj([[np.sqrt(max(0.0, 1.0 - amp / 2.0))], [np.sqrt(amp / 2.0)]])
        expect = [np.asarray([shot_index, shot_index + 0.5], dtype=float) for _ in e_ops]
        return SimpleNamespace(states=[ket0, ket1], expect=expect)


def test_extract_quantum_state_trajectory_keeps_all_mcwf_runs():
    engine = QuTiPEngine()
    result = SimpleNamespace(
        states=[],
        runs_states=[
            [_FakeQobj([[1.0], [0.0]]), _FakeQobj([[0.0], [1.0]])],
            [_FakeQobj([[0.0], [1.0]]), _FakeQobj([[1.0], [0.0]])],
        ],
    )

    payload = engine._extract_quantum_state_trajectory(result, "mcwf", "wave_function")

    assert payload is not None
    assert payload["actual_kind"] == "wave_function"
    assert payload["num_runs"] == 2
    assert len(payload["runs"]) == 2
    assert payload["snapshots"] == payload["runs"][0]


def test_build_collapse_and_noise_uses_nlevel_dephasing_operator_and_preserves_stochastic_channels():
    engine = QuTiPEngine()
    setup = SimpleNamespace(
        solver="mcwf",
        model_type="transmon_nlevel",
        n_qubits=1,
        qt=None,
        run_config=SimpleNamespace(seed=17),
        model_spec=SimpleNamespace(
            noise=SimpleNamespace(
                selected_model="ou",
                collapse_channels=[
                    SimpleNamespace(target=0, kind="relaxation", rate_rad_s=2.0),
                    SimpleNamespace(target=0, kind="dephasing", rate_rad_s=3.0),
                ],
                stochastic_channels=[
                    SimpleNamespace(
                        q=0,
                        kind="ou",
                        operator="sigma_z_over_2",
                        ou_sigma_rad_s=1.0,
                        ou_tau=2.0,
                    )
                ],
            )
        ),
        readout_mode="none",
    )
    system = SimpleNamespace(
        lower_ops=[_FakeQobj([[0.0, 1.0], [0.0, 0.0]])],
        z_ops=[_FakeQobj([[0.0, 0.0], [0.0, 1.0]])],
        raise_ops=[_FakeQobj([[0.0, 0.0], [1.0, 0.0]])],
        cavity_a=None,
        H=[0],
    )

    solver_inputs = build_collapse_and_noise(engine, setup, system)

    assert len(solver_inputs.c_ops) == 2
    assert solver_inputs.selected_noise == "ou"
    assert len(solver_inputs.stochastic_channels) == 1
    assert solver_inputs.runtime_metadata["stochastic_realizations_per_shot"] is True
    dephasing_op = solver_inputs.c_ops[1] * (1.0 / engine._dephasing_collapse_prefactor(3.0, "transmon_nlevel"))
    np.testing.assert_allclose(
        dephasing_op.full(),
        np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
    )
    assert system.H == [0]


def test_run_mcwf_trajectory_resamples_stochastic_noise_per_shot_and_returns_density_matrix():
    engine = QuTiPEngine()
    fake_qt = _FakeQt()
    setup = SimpleNamespace(
        qt=fake_qt,
        tlist=np.asarray([0.0, 1.0], dtype=float),
        run_config=SimpleNamespace(ntraj=3),
        solver="mcwf",
        n_qubits=1,
        model_type="transmon_nlevel",
        frame_mode="lab",
        rwa=False,
        model_spec=SimpleNamespace(
            hamiltonian=SimpleNamespace(control_terms=[]),
            readout=None,
            noise=SimpleNamespace(selected_model="ou"),
        ),
    )
    system = SimpleNamespace(
        H=[0],
        psi0=_FakeQobj([[1.0], [0.0]]),
        e_ops=[_FakeQobj([[0.0, 0.0], [0.0, 1.0]])],
        lower_ops=[],
        raise_ops=[],
        z_ops=[_FakeQobj([[0.0, 0.0], [0.0, 1.0]])],
        cavity_a=None,
        cavity_n=None,
    )
    solver_inputs = SimpleNamespace(
        c_ops=[],
        selected_noise="ou",
        seed=11,
        stochastic_channels=[
            SimpleNamespace(
                q=0,
                kind="ou",
                operator="sigma_z_over_2",
                ou_sigma_rad_s=5.0,
                ou_tau=10.0,
            )
        ],
        runtime_metadata={"stochastic_realizations_per_shot": True},
    )
    trajectory_cfg = QutipTrajectoryRequest(
        requested_state_kind="wave_function",
        save_times="all",
        save_final_state=True,
        options={"store_states": True},
    )

    trajectory = run_mcwf_trajectory(
        engine=engine,
        setup=setup,
        system=system,
        solver_inputs=solver_inputs,
        trajectory_cfg=trajectory_cfg,
    )

    assert len(fake_qt.calls) == 3
    assert all(call["ntraj"] == 1 for call in fake_qt.calls)
    assert trajectory.wave_function is not None
    assert trajectory.wave_function["actual_kind"] == "wave_function"
    assert trajectory.wave_function["num_runs"] == 3
    assert trajectory.density_matrix is not None
    assert trajectory.density_matrix["actual_kind"] == "density_matrix"
    assert trajectory.density_matrix["num_runs"] == 3
    assert trajectory.metadata["stochastic_realizations_per_shot"] is True
    assert trajectory.metadata["mcwf_ntraj"] == 3
    assert trajectory.metadata["stochastic_shot_seeds"] == [11, 12, 13]


def test_lower_noise_no_longer_generates_legacy_channels_without_sources():
    spec = lower_noise(
        NoiseConfig(model="markovian_lindblad", sources=[]),
        DeviceConfig(),
        raw_qubits=[{"freq_Hz": 5.0e9, "anharmonicity_Hz": -2.0e8, "T1_s": 1.0e-5}],
        num_qubits=1,
        dt_s=1.0e-9,
    )

    assert spec.sources == []
    assert spec.collapse_channels == []
    assert len(spec.per_qubit_rates) == 1
    assert spec.per_qubit_rates[0].gamma1_Hz == 0.0
