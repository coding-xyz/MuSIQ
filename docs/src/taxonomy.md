# Data Taxonomy

This page summarizes the major typed objects used by MuSIQ after the compiler
refactor.

## Circuit Data

- `CircuitGate`
  - one logical operation
  - fields: `name`, `qubits`, `params`, `clbits`
- `CircuitIR`
  - normalized circuit container
  - primary field: `schedule`
  - optional metadata: `format`, `source_qasm`
- `CircuitSpec`
  - persistent snapshot of a circuit for model execution

## Pulse Data

- typed pulse config
  - external config layout under `pulse.defaults`, `pulse.gates`,
    `pulse.channel_overrides`
- `PulseIR`
  - channel-grouped pulse timeline after lowering
- `PulseSpec`
  - one physical pulse with bounds, amplitude, shape, and carrier

## Model Data

- `ExecutableModel`
  - lowered, scheduled pulse-bearing representation
- `ModelSpec`
  - engine-neutral simulation model
  - includes system, Hamiltonian, noise, readout, and runtime settings

## Workflow Data

- `CircuitConfig`
  - either `qasm_text` or direct `CircuitIR`
- `SolverConfig`
  - engine and runtime settings
- `DeviceConfig`
  - hardware and noise configuration
- `PulseConfig`
  - persisted workflow pulse resource
- `Task`
  - composed runtime execution contract

## Results

- `RunResult`
  - trajectory output for one parameter sample
- `ModelAnalysis`
  - case, parametric, or comprehensive analysis artifact
