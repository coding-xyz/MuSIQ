# [PREP-P1] Define Ideal Prep as Initial-State Representation, Not Physical Evolution

## 0. Status
- Status: Open
- Owner: Unassigned
- Updated: 2026-06-02

## 1. Background and Goals
- Background:
  - The current workflow supports `prep_state.sequence`, but the present implementation applies it by rewriting the circuit before compilation.
  - That implementation currently rebuilds a serial circuit from prep gates plus measurement gates, which is workable for very small readout tasks but is not a sound long-term fit for schedule-first `CircuitIR`.
  - The main ambiguity is semantic: sometimes prep is intended as a physical gate sequence to simulate, but in the current report workflows it is often intended only as a concise way to say "start this run from the state that would ideally result from these gates applied to `|0...0>`".
  - Treating ideal prep as circuit rewriting unnecessarily entangles initial-state declaration with scheduling, lowering, and pulse-level timing semantics.
- Goals:
  - Define `prep` as an ideal initial-state representation by default.
  - Treat a prep sequence as a declarative description of how the initial state is derived from `|0...0>`, not as a physical circuit segment that must be evolved in the same run.
  - Ensure ideal prep does not pass through pulse lowering, schedule rewriting, or physical-noise simulation.
  - Preserve schedule-first `CircuitIR` semantics for the actual experiment circuit that enters compilation and lowering.
  - Clearly separate ideal state preparation from any future notion of physical prep circuitry.
- Why now:
  - The compiler refactor established schedule-first circuit semantics, so prep should no longer rely on flatten-and-rebuild behavior.
  - The current behavior introduces avoidable design risk by degrading scheduled circuits in order to express initial conditions.
  - This semantic split should be explicit before more workflows adopt `prep_state`.

## 2. Scope
- In Scope:
  - Define canonical semantics for ideal prep.
  - Specify how `prep_state.label` and `prep_state.sequence` should be interpreted.
  - Define where ideal prep is applied in the workflow and model-building pipeline.
  - Define how ideal prep becomes an initial state for simulation engines.
  - Audit current report workflows that use `prep_state`.
  - Add tests covering ideal prep behavior and the absence of circuit rewriting.
- Out of Scope:
  - Simulating physical prep gates, pulse distortions, or prep-gate noise in this issue.
  - Designing the full API for a future physical-prep subcircuit feature beyond reserving the semantic boundary.
  - Reworking unrelated circuit parsing or pulse schema semantics.

## 3. Inputs and Outputs (I/O)
- Inputs:
  - Current workflow prep handling in `src/musiq/workflow/stages.py`.
  - Current study/prep schema handling in `src/musiq/workflow/contracts.py` and `src/musiq/workflow/task_io.py`.
  - Current circuit compilation and lowering path built around schedule-first `CircuitIR`.
  - Existing report workflows, especially `report/task6_single_qubit_readout/solver_cqed.yaml`.
- Outputs:
  - A documented prep semantic model centered on ideal initial-state construction.
  - Updated workflow behavior so ideal prep does not rewrite the experiment circuit.
  - Engine-facing initial-state payloads derived from ideal prep.
  - Tests proving that scheduled circuits remain intact when prep is present.
- Relevant schema / version:
  - `schema_version: 1.0`

## 4. Technical Proposal
- High-level design:
  - Interpret prep as a two-step conceptual model:
    - start from `|0...0>`
    - apply an ideal logical prep map to derive the simulation initial state
  - The experiment circuit remains unchanged and enters compilation exactly as authored.
  - Ideal prep is resolved before solver execution as state construction, not as circuit evolution.

- Canonical semantics:
  - `prep_state.label` is descriptive metadata and may be used for reporting or conventional aliases.
  - `prep_state.sequence` is an ideal logical sequence that defines the desired initial state relative to `|0...0>`.
  - The sequence is not part of the physical runtime circuit for the current run.
  - The sequence does not consume simulated time.
  - The sequence does not contribute pulses, scheduler layers, or control noise.
  - The sequence does not modify the schedule of the experiment circuit.

- Required invariants:
  - Presence of ideal prep must not rewrite `CircuitIR.schedule`.
  - Presence of ideal prep must not require `flatten_schedule()` to rebuild the experiment circuit.
  - The compiled/lowered circuit should represent only the actual experiment body.
  - Solver initial state should reflect the ideal prep result.
  - If ideal prep is omitted, the default initial state remains `|0...0>`.

- Future semantic boundary:
  - If a future workflow needs to simulate physical prep operations, that must be represented explicitly as a different concept from ideal prep.
  - A future physical-prep feature may enter the actual circuit/schedule and therefore participate in timing, lowering, and noise.
  - That future feature must not overload the meaning of ideal `prep_state.sequence`.

- Current problem to remove:
  - The current prep implementation in `src/musiq/workflow/stages.py` rebuilds circuits from prep gates plus measurement gates.
  - That behavior is not the intended long-term contract and should be removed once ideal prep is implemented properly.

## 5. Required Workflow
1. Define prep semantics in code comments/docstrings before changing behavior.
2. Update workflow/model contracts so ideal prep is represented explicitly.
3. Remove circuit-rewrite behavior for ideal prep.
4. Update tests and representative report workflows.
5. Update docs after implementation behavior is stable.

## 6. Task Breakdown
1. Define the canonical semantic contract for ideal prep in workflow/config code.
2. Trace where solver initial states are currently derived and add an explicit ideal-prep injection point.
3. Remove prep-driven circuit rewriting from `src/musiq/workflow/stages.py`.
4. Ensure schedule-first `CircuitIR` remains unchanged when prep is present.
5. Map common prep sequences and labels into explicit initial states for supported solver modes.
6. Add tests proving ideal prep does not affect compiled circuit schedule or pulse lowering inputs.
7. Validate `report/task6_single_qubit_readout/solver_cqed.yaml` under the new semantics.
8. Document the distinction between ideal prep and any future physical prep circuit feature.

## 7. Definition of Done
- [ ] Ideal prep is defined as initial-state construction, not physical circuit evolution.
- [ ] `prep_state.sequence` no longer rewrites the experiment circuit.
- [ ] `CircuitIR.schedule` remains unchanged when ideal prep is present.
- [ ] No prep path relies on flattening and rebuilding the experiment circuit.
- [ ] Solver initial state reflects the ideal prep result.
- [ ] Representative workflows using prep continue to run with the intended semantics.
- [ ] Tests cover the non-rewrite invariant and expected initial-state behavior.
- [ ] Relevant docs/docstrings are updated.

## 8. Test Plan
- Unit tests:
  - Validate that ideal prep does not mutate `CircuitIR.schedule`.
  - Validate that ideal prep sequences map to expected initial states for representative one-qubit cases.
  - Validate that prep labels without explicit sequences still preserve clear default semantics.
- Integration tests:
  - Run representative readout workflows with `prep_0` and `prep_1` and confirm the experiment circuit body remains unchanged.
  - Confirm that pulse lowering sees only the actual experiment circuit, not prep pseudo-gates.
- Regression tests:
  - Confirm that current prep-enabled workflows no longer depend on circuit-rewrite behavior.
  - Confirm that schedule-first circuit semantics remain intact in the presence of prep metadata.
- Example command:
  - `pytest -q`

## 9. Risks and Rollback
- Major risks:
  - Ambiguity around whether a given workflow expects ideal prep or physical prep behavior.
  - Engine-specific differences in how initial states are represented internally.
  - Hidden coupling to the current prep-rewrite behavior in existing tests or notebooks.
- Mitigations:
  - Make the ideal-prep contract explicit in docs and tests.
  - Keep the physical-prep concept explicitly out of scope for this issue.
  - Validate current prep-enabled report tasks directly after the behavior change.
- Rollback strategy:
  - If engine integration is incomplete, keep prep metadata explicit and fail clearly rather than silently rewriting circuits again.

## 10. Dependencies and Blockers
- Prerequisites:
  - Agreement that ideal prep is the default semantic model.
  - Agreement that physical prep, if ever needed, must be represented separately.
- External dependencies:
  - None beyond the existing workflow/model/solver stack.
- Potential blockers:
  - Determining the cleanest engine-neutral representation for prepared initial states.
  - Handling prep semantics consistently across different solver modes and subsystem models.

## 11. Estimate and Priority
- Priority: P1
- Estimated effort: 1-3 days
- Owner: Unassigned

## 12. References
- Related files:
  - `src/musiq/workflow/stages.py`
  - `src/musiq/workflow/contracts.py`
  - `src/musiq/workflow/task_io.py`
  - `src/musiq/backend/model/build.py`
  - `src/musiq/workflow/model_execution.py`
  - `report/task6_single_qubit_readout/solver_cqed.yaml`
- Related issues / PRs:
  - `issues/done/COMPILER-P1_typed-pulse-recipe-and-parallel-circuit-ir-refactor.md`
  - `issues/done/ANALYSIS-P1_hierarchical-analysis-architecture-refactor.md`
