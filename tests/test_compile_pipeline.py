import json

from musiq.backend.compile_pipeline import CompilePipeline
from musiq.common.schemas import BackendConfig, CircuitGate, CircuitIR


class AppendBarrierPass:
    def run(self, circuit: CircuitIR, ctx: dict) -> CircuitIR:
        assert ctx["config"].solver == "me"
        assert ctx["hardware"] == {"schedule_policy": "serial"}
        return CircuitIR(
            schema_version=circuit.schema_version,
            format=circuit.format,
            num_qubits=circuit.num_qubits,
            num_clbits=circuit.num_clbits,
            gates=[*circuit.gates, CircuitGate(name="barrier", qubits=[])],
            source_qasm=circuit.source_qasm,
        )


def test_compile_pipeline_default_run_normalizes_and_reports():
    circuit = CircuitIR(gates=[CircuitGate(name="H", qubits=[0]), CircuitGate(name="RZ", qubits=[0], params=[0.25])])
    config = BackendConfig(solver="se")

    normalized, report = CompilePipeline().run(circuit, config)

    assert [gate.name for gate in normalized.gates] == ["h", "rz"]
    assert report["initial_gate_count"] == 2
    assert report["final_gate_count"] == 2
    assert report["passes"] == [{"name": "NormalizePass", "before": 2, "after": 2}]
    assert report["hardware_used"] is False


def test_compile_pipeline_supports_custom_passes():
    circuit = CircuitIR(gates=[CircuitGate(name="x", qubits=[0])])
    config = BackendConfig(solver="me")
    pipeline = CompilePipeline(passes=[AppendBarrierPass()])

    lowered, report = pipeline.run(circuit, config, hardware={"schedule_policy": "serial"})

    assert [gate.name for gate in lowered.gates] == ["x", "barrier"]
    assert report["passes"] == [{"name": "AppendBarrierPass", "before": 1, "after": 2}]
    assert report["final_gate_count"] == 2
    assert report["hardware_used"] is True


def test_dump_compile_report_writes_json(tmp_path):
    report = {"schema_version": "1.0", "final_gate_count": 3}

    out_path = CompilePipeline.dump_compile_report(report, tmp_path / "nested" / "compile_report.json")

    assert out_path.exists()
    assert json.loads(out_path.read_text(encoding="utf-8")) == report
