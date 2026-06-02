import math

from musiq.schemas.components import SystemComponentSpec
from musiq.workflow.contracts import AnalyserConfig, SolverBackendConfig, SolverConfig, WorkflowRunOptions


def test_transmon_component_derives_rad_s_fields_from_hz_parameters():
    comp = SystemComponentSpec.from_dict(
        {
            "id": "q0",
            "type": "transmon",
            "basis": {"kind": "nlevel", "levels": 3},
            "parameters": {
                "freq_Hz": 5.0e9,
                "anharmonicity_Hz": -2.0e8,
            },
        }
    )

    assert comp.omega_rad_s == 2.0 * math.pi * 5.0e9
    assert comp.anharmonicity_rad_s == 2.0 * math.pi * -2.0e8


def test_solver_config_to_backend_config_normalizes_noise_and_runtime_fields():
    solver_cfg = SolverConfig(
        backend=SolverBackendConfig(level="nlevel", analysis_pipeline="custom", truncation={"q0": 3}),
        run=WorkflowRunOptions(solver_mode="me", seed=7, sweep=[{"name": "amp"}]),
    )

    backend = solver_cfg.to_backend_config(noise={"model": "markovian_lindblad"}, runtime_level="cqed")

    assert backend.level == "cqed"
    assert backend.noise == "lindblad"
    assert backend.solver == "me"
    assert backend.analysis_pipeline == "custom"
    assert backend.truncation == {"q0": 3}
    assert backend.sweep == [{"name": "amp"}]
    assert backend.seed == 7


def test_analyser_config_to_payload_merges_typed_sections_and_extras():
    analyser = AnalyserConfig(
        solver_id="solver_0",
        analysis=[{"name": "single_qubit_analysis", "level": "CASE", "metrics": ["population"]}],
        extras={"custom_flag": True},
    )
    analyser.trajectory.extras["window"] = "tail"
    analyser.report.extras["format"] = "html"

    payload = analyser.to_payload()

    assert payload["solver_id"] == "solver_0"
    assert payload["analysis"] == [{"name": "single_qubit_analysis", "level": "CASE", "metrics": ["population"]}]
    assert payload["trajectory"]["window"] == "tail"
    assert payload["report"]["format"] == "html"
    assert payload["custom_flag"] is True
