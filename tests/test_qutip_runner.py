import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from musiq.engines.qutip.engine import QuTiPEngine


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
