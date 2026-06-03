# [COMPILER-P1] Typed Pulse Recipe Lowering and Parallel CircuitIR Refactor

## 0. Status
- Status: Done
- Owner: Unassigned
- Updated: 2026-06-02

## 1. Background and Goals
- Background:
  - The current pulse lowering path still treats most single-qubit XY gates as one shared recipe family driven by flat `single_qubit_*` knobs.
  - This makes calibration semantics ambiguous: for example, `x` and `sx` currently share DRAG-style knobs even though they often require distinct calibrated parameters.
  - The current pulse configuration is largely untyped at the recipe level, and several parameters are expressed as relative `amp_scale` knobs instead of explicit physical quantities.
  - The current `CircuitIR` is an ordered `list[CircuitGate]` with no first-class notion of parallel layers or per-qubit intra-layer timing structure.
  - The current workflow assumes `qasm -> CircuitIR` as the main input path, but there is no direct structured circuit file format that can bypass QASM parsing while preserving explicit parallel-layer semantics.
- Goals:
  - Introduce a typed pulse-recipe abstraction for lowering with explicit per-gate recipe contracts.
  - Reserve first-class recipe interfaces for `SX`, `CZ`, and `VirtualZ`.
  - Make `SX` and `CZ` use explicit `duration_ns` and `amplitude_Hz` fields instead of ambiguous `amp_scale`.
  - Make `VirtualZ` a typed angle-only logical operation with no duration or amplitude fields.
  - Support pulse parameter configuration per channel, so recipe instances can be specialized on a channel-by-channel basis.
  - Refactor `CircuitIR` into a parallel-aware structure keyed by integer schedule layer.
  - Add a structured YAML circuit input format that can be loaded directly without passing through QASM.
- Why now:
  - Pulse recipe semantics and circuit structure are both reaching the point where ad hoc normalization is blocking clean evolution.
  - If typed pulse recipes and typed parallel circuit IR are not introduced now, future work on calibrated lowering, scheduler semantics, and hardware-native operations will accumulate on unstable interfaces.

## 2. Scope
- In Scope:
  - Design and implement typed pulse recipe data structures for lowering.
  - Refactor pulse configuration schema to support gate-specific recipe configuration and per-channel overrides.
  - Explicitly model `SX`, `CZ`, and `VirtualZ` recipe contracts.
  - Replace ambiguous amplitude scaling for `SX` and `CZ` with explicit `amplitude_Hz`.
  - Ensure `VirtualZ` only carries an angle field and is never represented with pulse duration or amplitude.
  - Refactor `CircuitIR` from flat ordered gate lists into schedule-indexed parallel structure.
  - Define serialization and file input format for the new circuit YAML representation.
  - Add direct circuit-file loading for YAML circuit inputs that bypass QASM parsing.
  - Remove legacy pulse-config and flat-circuit compatibility assumptions so the new typed structures become the only authoritative path.
  - Retain QASM as a supported first-class user-facing input format that parses directly into the new `CircuitIR`.
- Out of Scope:
  - Full removal of QASM support in this phase.
  - Full pulse calibration optimization tooling.
  - Final hardware-native scheduling model beyond what is needed to support schedule-based parallel IR and typed lowering inputs.

## 3. Inputs and Outputs (I/O)
- Inputs:
  - Existing pulse config files under `report/` and `small_circuits/`.
  - Current pulse lowering and recipe logic in `src/musiq/pulse/catalog.py` and `src/musiq/pulse/lowering.py`.
  - Current scheduling logic in `src/musiq/backend/scheduling.py`.
  - Current circuit schema and import/export code in `src/musiq/schemas/circuit.py`, `src/musiq/circuit/import_qasm.py`, and related workflow entry points.
- Outputs:
  - A typed pulse recipe schema and implementation used by lowering.
  - A revised pulse config file structure supporting gate-level recipes and channel-level overrides.
  - A revised `CircuitIR` with explicit `schedule -> per-qubit gate-sequence` structure.
  - YAML circuit file parsing support that builds the new `CircuitIR` directly.
  - A QASM import path that maps parsed circuits directly into the new `CircuitIR` without relying on legacy circuit compatibility layers.
  - Updated pulse config files for `report/task1_single_qubit_rabi`, `report/task2_single_qubit_decoherence`, `report/task3_gaussian_drag_comparison`, and `report/task6_single_qubit_readout` in the new schema.
- Relevant schema / version:
  - `schema_version: 1.0`

## 4. Technical Proposal
- High-level design:
  - Split pulse lowering into two responsibilities:
    - resolve typed pulse recipe config for one logical operation
    - instantiate pulses/events from the typed recipe
  - Make recipe contracts explicit by gate kind instead of relying on one shared bucket of flat hardware keys.
  - Refactor `CircuitIR` to represent parallel structure directly instead of inferring it later from a flat gate list.

- Pulse recipe target architecture:
  - Introduce typed recipe config objects, for example:
    - `PulseLibraryConfig`
    - `ChannelPulseConfig`
    - `SingleQubitGateRecipeMap`
    - `TwoQubitGateRecipeMap`
    - `SxPulseRecipeConfig`
    - `CzPulseRecipeConfig`
    - `VirtualZRecipeConfig`
  - Recipe types should be explicit and discriminated, for example:
    - `recipe_type: "sx"`
    - `recipe_type: "cz"`
    - `recipe_type: "virtual_z"`
  - Lowering should resolve recipes in this order:
    - exact gate recipe
    - exact gate recipe with exact channel override
    - gate-family fallback
    - global defaults only where physically meaningful

- Required gate recipe interfaces:
  - `SX`:
    - must expose `duration_ns`
    - must expose `amplitude_Hz`
    - may expose shape-specific parameters such as `sigma_fraction`, `drag_beta`, `carrier_freq_Hz`, `phase_rad`
    - must not use `amp_scale`
  - `CZ`:
    - must expose `duration_ns`
    - must expose `amplitude_Hz`
    - may expose coupler/edge parameters such as `edge_ns`, `target_conditional_phase_rad`
    - must not use `amp_scale`
  - `VirtualZ`:
    - must expose angle only
    - must not expose `duration_ns`
    - must not expose `amplitude_Hz`
    - must lower to frame/phase update semantics only

- Typed pulse configuration structure:
  - The final implementation should use a typed structure that separates three concerns explicitly:
    - `defaults`: global fallback values that are not gate-specific calibrations
    - `gates`: typed logical gate recipe definitions
    - `channel_overrides`: per-channel calibration overrides for a specific gate recipe
  - `gates` is the primary configuration surface.
  - `channel_overrides` is a patch layer over `gates`, not an alternative recipe namespace.
  - The final implementation should converge to a typed structure equivalent to:

```yaml
schema_version: "1.0"
pulse:
  defaults:
    schedule_policy: parallel
    xy_carrier_freq_Hz: 5.0e9
    ro_carrier_freq_Hz: 6.5e9

  gates:
    sx:
      recipe_type: sx
      shape: drag
      duration_ns: 28.0
      amplitude_Hz: 10.5e6
      carrier_freq_Hz: 5.0e9
      phase_rad: 0.0
      sigma_fraction: 0.10
      drag_beta: 0.11

    virtual_z:
      recipe_type: virtual_z

    cz:
      recipe_type: cz
      shape: rect
      duration_ns: 52.0
      amplitude_Hz: 20.0e6
      carrier_freq_Hz: 0.0
      edge_ns: 2.0
      target_conditional_phase_rad: 3.141592653589793

  channel_overrides:
    XY_0:
      sx:
        amplitude_Hz: 10.8e6
        drag_beta: 0.09
        carrier_freq_Hz: 5.01e9
    XY_1:
      sx:
        carrier_freq_Hz: 4.99e9
    TC_0:
      cz:
        amplitude_Hz: 19.5e6
        duration_ns: 53.0
    RO_0:
      measure:
        carrier_freq_Hz: 6.50e9
```

  - Small naming adjustments are acceptable, but the final schema must preserve the following invariants:
    - `gates.<gate_name>` is the canonical typed recipe definition for one logical gate
    - `channel_overrides.<channel_name>.<gate_name>` is the canonical per-channel override path
    - `SX` and `CZ` are typed recipe entries
    - `duration_ns` and `amplitude_Hz` are explicit physical fields for `SX` and `CZ`
    - `amp_scale` is not used for `SX` and `CZ`
    - `VirtualZ` is angle-only and never carries duration/amplitude pulse fields
    - per-channel pulse parameter overrides are supported in config
  - The config semantics should be documented in this order:
    - define the logical gate recipe under `gates`
    - determine the physical channel during lowering
    - apply any per-channel override from `channel_overrides`
  - If later grouping is needed for implementation clarity, it may be introduced internally, but the external user-facing config must remain typed around `gates` first rather than split primarily by qubit arity.

- Lowering implementation requirements:
  - Introduce typed recipe resolution before pulse instantiation.
  - Separate logical gate handling from physical pulse generation.
  - `VirtualZ` should remain a logical frame update primitive rather than a zero-duration pulse-like pseudo-recipe.
  - `SX` and `CZ` recipe resolution must be channel-aware.
  - Existing supported logical gates such as `x`, `rx`, `ry`, `h`, `measure`, and `reset` may continue to exist, but they must resolve through the new typed lowering architecture rather than through legacy flat recipe compatibility.

- CircuitIR target structure:
  - Replace the current flat gate list model with a parallel-aware typed structure equivalent to:

```python
schedule: dict[int, list[list[CircuitGate]]]
```

  - Semantics:
    - outer `dict` key = integer schedule layer index
    - each schedule-layer value = one list indexed by qubit
    - each qubit entry = ordered intra-layer sequence of `CircuitGate`
  - This means:
    - parallelism is represented explicitly by schedule layer
    - per-qubit within-layer micro-order is represented explicitly by `list[CircuitGate]`
    - a multi-qubit gate appears in each participating qubit slot for that schedule layer and refers to the same logical operation semantics

- Required YAML circuit input format:
  - The implementation should support a file structure equivalent to:

```yaml
schema_version: "1.0"
format: circuit_layer_yaml
num_qubits: 3
num_clbits: 0
schedule:
  0:
    - - ['rz', [0], 0.7853981633974474]
      - ['sx', [0]]
      - ['rz', [0], 1.5707963267948968]
    - []
    - []
  1:
    - []
    - - ['cz', [2, 1]]
    - - ['cz', [2, 1]]
  2:
    - []
    - - ['sx', [1]]
      - ['rz', [1], 2.751108456601225]
    - - ['rz', [2], 3.141592653589793]
      - ['sx', [2]]
      - ['rz', [2], 0.39048419698856796]
  3:
    - []
    - - ['sx', [1]]
    - - ['sx', [2]]
      - ['rz', [2], 3.141592653589793]
  4:
    - []
    - - ['cz', [2, 1]]
    - - ['cz', [2, 1]]
```

  - Supported tuple-like gate forms in YAML must be documented explicitly:
    - `['gate_name', [qubits...]]`
    - `['gate_name', [qubits...], angle_or_param]`
    - if needed for future extensibility, a dict form may also be supported, but the list form above is the minimum required contract

- CircuitIR schema changes:
  - The final `CircuitIR` dataclass should be updated to represent:
    - `num_qubits`
    - `num_clbits`
    - `schedule`
    - optional `source_qasm`
    - optional circuit format/source metadata
  - The current `gates: list[CircuitGate]` field should no longer be the authoritative execution structure.
  - This issue should not preserve the old flat `gates` structure as a supported compatibility API.
  - If any temporary flat projection is needed during implementation, it must remain internal and transitional rather than user-facing or contractually supported.

- QASM isolation and migration:
  - QASM import may continue to work, but it must parse directly into the new parallel-aware `CircuitIR`.
  - The new YAML circuit format must be a first-class alternative input path that bypasses QASM entirely.
  - Later lowering, scheduling, and model-building stages must depend on the new `CircuitIR` structure rather than on QASM text.
  - QASM support, if retained, is a first-class parser path into the new IR rather than a legacy compatibility layer.
  - If QASM-driven behavior and schedule/YAML-driven behavior conflict with legacy assumptions, the new typed pulse schema and new `CircuitIR.schedule` semantics take precedence.

- Key design decisions:
  - Pulse recipe semantics should be typed and physically explicit where possible.
  - Relative scaling fields such as `amp_scale` are too ambiguous for `SX` and `CZ`, so explicit `amplitude_Hz` is required.
  - `VirtualZ` is logically distinct from pulse-bearing operations and must keep a separate angle-only interface.
  - Parallel structure belongs in circuit IR, not only in downstream scheduling heuristics.
  - A structured circuit YAML format is required so the system can accept non-QASM-generated circuits without lossy translation.

- Extension points:
  - Additional typed gate recipes such as `X`, `RX`, `RY`, echoed `CZ`, CR-based `CX`, and leakage-aware calibrations should be addable without changing the core resolver contract.
  - Additional circuit file formats may be supported later, but they must map into the same schedule-based `CircuitIR`.

## 5. Required Workflow
1. Complete code changes and required tests together.
2. Update related `docstring` content before each completion step.
3. Update related `docs/` content before marking the issue done.
4. Treat `docs/site/` as generated artifacts; edit source docs first.
5. Run `mkdocs build --clean` after doc changes to keep doc outputs synchronized.
6. Mark the issue complete only after code, tests, docstrings, and docs are all synchronized.

## 6. Task Breakdown
1. Define typed pulse recipe dataclasses and discriminated recipe interfaces for `SX`, `CZ`, and `VirtualZ`.
2. Refactor pulse config loading and validation to support gate-level typed recipes and per-channel overrides.
3. Refactor pulse recipe resolution and instantiation so `SX`, `CZ`, and `VirtualZ` use dedicated typed lowering paths.
4. Remove `amp_scale` semantics from `SX` and `CZ` recipe contracts and replace them with explicit `amplitude_Hz`.
5. Refactor `CircuitIR` schema to use `schedule -> per-qubit gate-sequence` as the primary structure.
6. Update scheduling and lowering stages to consume the new `CircuitIR` shape.
7. Add direct YAML circuit-file parsing that builds the new `CircuitIR` without QASM.
8. Keep QASM import working, if retained, by mapping parsed QASM circuits directly into the new schedule-based IR.
9. Update `report/task1_single_qubit_rabi`, `report/task2_single_qubit_decoherence`, `report/task3_gaussian_drag_comparison`, and `report/task6_single_qubit_readout` pulse config files to the new typed schema.
10. Update report workflows and small-circuit examples to the new typed pulse config and new `CircuitIR` semantics rather than relying on compatibility adapters.
11. Document the new pulse config schema and new circuit YAML file format.

## 7. Definition of Done
- [ ] Typed pulse recipe interfaces exist for `SX`, `CZ`, and `VirtualZ`.
- [ ] `SX` and `CZ` recipe configs use explicit `duration_ns` and `amplitude_Hz`.
- [ ] `SX` and `CZ` recipe configs do not depend on `amp_scale`.
- [ ] `VirtualZ` is represented as an angle-only logical recipe with no duration/amplitude fields.
- [ ] Pulse configuration supports per-channel parameter overrides.
- [ ] `CircuitIR` uses explicit schedule-based parallel structure as its primary representation.
- [ ] The system can load the new YAML circuit format directly as file input.
- [ ] No user-facing compatibility layer is retained for legacy flat pulse config or flat `CircuitIR.gates` structure.
- [ ] If QASM input remains supported, it enters through a direct parser-to-new-IR path rather than through compatibility adapters.
- [ ] Scheduling/lowering no longer depend on flat `list[CircuitGate]` as the primary execution structure.
- [ ] `report/task1_single_qubit_rabi`, `report/task2_single_qubit_decoherence`, `report/task3_gaussian_drag_comparison`, and `report/task6_single_qubit_readout` pulse config files are migrated to the new schema.
- [ ] Relevant `docstring` content is updated.
- [ ] Relevant `docs/` content is updated.
- [ ] `docs/src` and `docs/site` are synchronized through a successful build.

## 8. Test Plan
- Unit tests:
  - Validate typed pulse recipe parsing for `SX`, `CZ`, and `VirtualZ`.
  - Validate that `SX` and `CZ` reject ambiguous `amp_scale`-only configurations.
  - Validate that `VirtualZ` rejects duration/amplitude pulse fields.
  - Validate per-channel override resolution.
  - Validate YAML circuit parsing into schedule-based `CircuitIR`.
- Integration tests:
  - Run pulse lowering on circuits containing `SX`, `CZ`, and `VirtualZ` using the new typed recipe config.
  - Run the same logical circuit through both QASM import and YAML circuit import and confirm equivalent lowered semantics.
  - Validate representative report tasks after they are updated to the new typed config and new `CircuitIR` semantics.
- Regression tests:
  - Legacy flat pulse config and flat `CircuitIR.gates` inputs should fail fast with clear validation errors once the new schema is required.
  - Existing QASM-driven workflows, if still supported, should produce valid new-IR circuits and lowered pulse IR without going through compatibility adapters.
  - Validate that the pulse config files in `report/task1_single_qubit_rabi`, `report/task2_single_qubit_decoherence`, `report/task3_gaussian_drag_comparison`, and `report/task6_single_qubit_readout` load successfully under the new schema and drive successful representative runs.
- Example command:
  - `pytest -q`

## 9. Risks and Rollback
- Major risks:
  - Breaking the current scheduling/lowering path by changing the core circuit IR shape.
  - Ambiguity in how multi-qubit gates should appear across multiple qubit slots inside one schedule layer.
  - Divergence between user-facing QASM input behavior and direct schedule/YAML input behavior.
- Mitigations:
  - Keep one canonical schedule-based representation and derive any temporary flat views internally only when strictly necessary for stepwise implementation.
  - Write explicit invariants and tests for multi-qubit gate mirroring across participating qubits.
  - Treat the new typed pulse schema and new `CircuitIR.schedule` semantics as the single source of truth whenever legacy assumptions conflict.
- Rollback strategy:
  - Roll back as a whole change set if the new typed recipe path or new circuit IR path is unstable; do not accumulate partial compatibility layers as a long-term fallback.
  - Avoid exposing mixed old/new public schemas at the same time.

## 10. Dependencies and Blockers
- Prerequisites:
  - Agreement on typed recipe naming and discriminated schema shape.
  - Agreement that `schedule` is the primary unit of parallel circuit structure.
- External dependencies:
  - None required beyond the current Python/test/doc toolchain.
- Potential blockers:
  - Deciding the exact canonical representation of shared multi-qubit gates across per-qubit schedule entries.
  - Aligning QASM parsing output with schedule/YAML semantics without reintroducing legacy compatibility rules.

## 11. Estimate and Priority
- Priority: P1
- Estimated effort: 4-8 days
- Owner: Unassigned

## 12. References
- Related files:
  - `src/musiq/pulse/catalog.py`
  - `src/musiq/pulse/lowering.py`
  - `src/musiq/backend/scheduling.py`
  - `src/musiq/schemas/circuit.py`
  - `src/musiq/circuit/import_qasm.py`
  - `src/musiq/workflow/stages.py`
  - `report/task1_single_qubit_rabi/pulse.yaml`
  - `report/task2_single_qubit_decoherence/pulse.yaml`
  - `report/task3_gaussian_drag_comparison/pulse.yaml`
  - `report/task6_single_qubit_readout/pulse.yaml`
- Related issues / PRs:
  - `issues/done/ANALYSIS-P1_hierarchical-analysis-architecture-refactor.md`
- Related docs:
  - `issues/ISSUE_TEMPLATE.md`
