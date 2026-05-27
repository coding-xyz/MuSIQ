# System Overview

MuSIQ is organized around a compile-to-execution pipeline:

`circuit config -> CircuitIR -> gate schedule -> PulseIR -> ExecutableModel -> ModelSpec -> engine result -> analysis`

## Core Layers

- `musiq.workflow`
  - loads user-facing resource files
  - composes runtime tasks
  - coordinates compilation, execution, and analysis
- `musiq.circuit`
  - parses OpenQASM and structured circuit YAML
  - normalizes logical gates into `CircuitIR`
- `musiq.backend`
  - runs circuit passes
  - schedules logical gates
  - builds engine-neutral model artifacts
- `musiq.pulse`
  - resolves typed pulse recipes
  - lowers logical gates into channel pulse sequences
- `musiq.schemas`
  - owns the typed IRs and result contracts

## Compiler Decisions

## Schedule-First Circuit IR

Parallel circuit structure is represented directly in `CircuitIR.schedule`.
Downstream scheduling and lowering consume that structure instead of rebuilding
parallelism from a flat gate list.

## Gate-First Pulse Schema

Pulse configuration is centered on typed logical recipes:

- `pulse.defaults`
- `pulse.gates`
- `pulse.channel_overrides`

This keeps physical defaults separate from gate calibrations and allows
channel-specific specialization without creating a second recipe namespace.

## No Legacy Pulse Compatibility Layer

Legacy pulse schemas based on:

- `channels`
- `carriers`
- `waveforms`
- `operations`
- top-level flat gate knobs such as `single_qubit_gate_amp_scale`

are intentionally rejected at load time. The typed schema is the only supported
user-facing pulse contract.
