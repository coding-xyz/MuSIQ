# MuSIQ

MuSIQ is a workflow-first quantum simulation project with a schedule-based
`CircuitIR`, typed pulse recipes, and reproducible model execution artifacts.

## What Changed In The Compiler Refactor

- `CircuitIR` is now schedule-first. Parallel structure lives in
  `schedule: dict[int, list[list[CircuitGate]]]`.
- Flat `CircuitIR.gates` is no longer part of the public contract.
- Pulse lowering now expects typed gate recipes under `pulse.gates`.
- `SX` and `CZ` use explicit `duration_ns` and `amplitude_Hz`.
- `VirtualZ` is represented as an angle-only logical recipe with no pulse
  duration or amplitude fields.
- Per-channel calibration patches live under `pulse.channel_overrides`.
- YAML circuits can be loaded directly with `format: circuit_layer_yaml`.

## Start Here

- [Workflow Guide](workflow.md)
- [Typed Pulse Schema](wiki/pulse_schema.md)
- [Schedule Circuit IR](wiki/circuit_ir.md)
- [System Overview](architecture.md)

## Typical File Shapes

Typed pulse config:

```yaml
schema_version: "1.0"
pulse:
  defaults:
    xy_carrier_freq_Hz: 5.0e9
    ro_carrier_freq_Hz: 6.5e9

  gates:
    sx:
      recipe_type: sx
      duration_ns: 28.0
      amplitude_Hz: 10.5e6

    virtual_z:
      recipe_type: virtual_z
```

Schedule-first circuit YAML:

```yaml
schema_version: "1.0"
format: circuit_layer_yaml
num_qubits: 2
num_clbits: 0
schedule:
  0:
    - - ['sx', [0]]
    - - ['sx', [1]]
  1:
    - - ['cz', [0, 1]]
    - - ['cz', [0, 1]]
```

## Verification Expectations

When compiler-facing behavior changes:

- update code and tests together
- update docstrings in touched modules
- update `docs/src/`
- run `mkdocs build --clean`
