# Workflow Guide

MuSIQ accepts resource-style inputs for circuits, solvers, devices, pulses, and
analysers. The compiler refactor changes two important expectations:

- circuit inputs are schedule-first
- pulse inputs are typed gate recipes

## Circuit Inputs

You can provide either:

- `qasm_text` or `qasm_path`
- direct schedule YAML with `format: circuit_layer_yaml`

Schedule payloads must not be mixed with QASM fields in the same circuit file.

## Pulse Inputs

Pulse files are gate-first:

```yaml
schema_version: "1.0"
pulse:
  defaults: {}
  gates: {}
  channel_overrides: {}
```

Legacy pulse files based on `channels`/`carriers`/`waveforms`/`operations` are
rejected.

## Runtime Flow

1. Load circuit, solver, device, pulse, and analyser resources.
2. Build a runtime `Task`.
3. Normalize the circuit into `CircuitIR`.
4. Schedule logical gates from `CircuitIR.schedule`.
5. Resolve typed pulse recipes and lower to `PulseIR`.
6. Compile pulses and build `ModelSpec`.
7. Run the selected engine.
8. Run configured analyses.

## CLI Notes

For file-driven execution, prefer resource configs over ad hoc inline fields.
That keeps typed schemas explicit and makes runs easier to reproduce.
