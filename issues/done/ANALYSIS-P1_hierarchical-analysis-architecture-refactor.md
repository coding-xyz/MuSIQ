# [ANALYSIS-P1] Refactor Analysis Architecture into Level/Kind Hierarchy

## 0. Status
- Status: Done
- Owner: Unassigned
- Updated: 2026-05-25

## 1. Background and Goals
- Background:
  - The current `src/musiq/analysis/` package mixes three concerns together: domain analysis logic, shared helpers, and workflow/registry orchestration.
  - The current structure implicitly contains multiple analysis types such as single-qubit state analysis, readout-chain analysis, and IQ aggregation, but those types are mostly represented as free functions plus string dispatch.
  - The recent migration to hierarchical analyser config introduced `analysis: [{name, level, metrics}]`, but the codebase still lacks a first-class architecture that matches the actual business hierarchy.
  - `metrics` currently has too much architectural weight. In the intended model, metrics should be subordinate capabilities under a concrete analysis type, not the top-level abstraction.
- Goals:
  - Establish `level -> kind -> module` as the primary analysis architecture.
  - Make `CASE`, `PARAMETRIC`, and `COMPREHENSIVE` the top-level organizing principle.
  - Introduce explicit analysis kinds under each level, such as `SingleQubit`, `MultiQubit`, `Readout`, and `IQ`.
  - Demote `metrics` to a module owned by each concrete analysis kind.
  - Provide a migration path that preserves compatibility with current configs and notebooks.
- Why now:
  - The current architecture is still small enough to reshape before more analysis types and reports are added.
  - Recent work on `analysis[].metrics` exposed that configuration semantics and runtime structure are drifting apart.
  - Without a clean hierarchy, future additions such as multi-qubit analysis, tomography, RB, and calibration analysis will become increasingly inconsistent.

## 2. Scope
- In Scope:
  - Define a canonical hierarchy for analysis execution: `level -> kind -> module`.
  - Refactor the `src/musiq/analysis/` package layout so directory structure matches the new hierarchy instead of mixing helpers, domain logic, and orchestration in flat files.
  - Refactor model-side analysis hierarchy so runtime/model objects clearly represent `CASE`, `PARAMETRIC`, and `COMPREHENSIVE` outputs and their concrete kinds.
  - Introduce typed analysis step classes or equivalent structured handlers centered on `level` and `kind`.
  - Define how `metrics` are declared and resolved within each analysis kind.
  - Migrate existing `state_analysis` / `single_qubit_analysis`, `readout_analysis`, and `iq_analysis` into the new hierarchy.
  - Define config semantics for `analysis` steps so `level` and `kind` are explicit and `name` is no longer overloaded as a type identifier.
  - Preserve backward compatibility for legacy `metrics`, `case_metrics`, and `sweep_metrics` fields during migration.
  - Keep report validation in scope for `report/task1_single_qubit_rabi`, `report/task2_single_qubit_decoherence`, `report/task3_gaussian_drag_comparison`, and `report/task6_single_qubit_readout`, allowing only small plotting-script adjustments when needed.
- Out of Scope:
  - Implement every future analysis kind such as tomography or randomized benchmarking in this issue.
  - Redesign the full result schema beyond what is needed to support the new hierarchy.
  - Remove all legacy compatibility in the first iteration.

## 3. Inputs and Outputs (I/O)
- Inputs:
  - Existing analyser configs in `report/` and any templates or tests that depend on them.
  - Current analysis modules such as `metrics.py`, `readout_chain.py`, `sensitivity.py`, `registry.py`, and `passes.py`.
  - Current workflow dispatch in `src/musiq/workflow/stages.py` and `src/musiq/workflow/model_execution.py`.
  - Current model/result schemas in `src/musiq/schemas/results.py`, `src/musiq/workflow/model.py`, and persistence/serialization code that stores analysis outputs.
- Outputs:
  - A documented architectural model for hierarchical analysis.
  - A target `src/musiq/analysis/` directory layout aligned with `level -> kind -> module`.
  - Updated model-side analysis object hierarchy and persistence boundaries that reflect the same levels and kinds.
  - New or refactored runtime abstractions for `CASE`, `PARAMETRIC`, and `COMPREHENSIVE` analysis kinds.
  - Backward-compatible config parsing with a forward path toward explicit `level` + `kind`.
  - Tests covering dispatch, compatibility, and representative analysis kinds.
- Relevant schema / version:
  - `schema_version: 1.0`

## 4. Technical Proposal
- High-level design:
  - Model analysis around top-level levels:
    - `CASE`
    - `PARAMETRIC`
    - `COMPREHENSIVE`
  - Under each level, introduce explicit analysis kinds, for example:
    - `CASE.SingleQubit`
    - `CASE.MultiQubit`
    - `CASE.Readout`
    - `PARAMETRIC.SingleQubit`
    - `COMPREHENSIVE.IQ`
  - Treat `metrics`, `signals`, `fit`, `summary`, and similar concepts as modules owned by a concrete analysis kind.
  - Keep workflow dispatch centered on `(level, kind)` instead of raw metric names or overloaded `name` strings.
- Explicit target directory structure for `src/musiq/analysis/`:
  - The implementation should converge to a structure equivalent to:

```text
src/musiq/analysis/
鈹溾攢鈹€ __init__.py
鈹溾攢鈹€ base.py                    # base abstractions for analysis levels/kinds
鈹溾攢鈹€ registry.py                # level+kind registry / dispatch
鈹溾攢鈹€ common/
鈹?  鈹溾攢鈹€ __init__.py
鈹?  鈹溾攢鈹€ state_utils.py
鈹?  鈹溾攢鈹€ trajectory_semantics.py
鈹?  鈹斺攢鈹€ observables.py
鈹溾攢鈹€ case/
鈹?  鈹溾攢鈹€ __init__.py
鈹?  鈹溾攢鈹€ single_qubit/
鈹?  鈹?  鈹溾攢鈹€ __init__.py
鈹?  鈹?  鈹溾攢鈹€ analysis.py        # CASE.SingleQubit
鈹?  鈹?  鈹斺攢鈹€ metrics.py         # population / coherence / leakage / ...
鈹?  鈹溾攢鈹€ multi_qubit/
鈹?  鈹?  鈹溾攢鈹€ __init__.py
鈹?  鈹?  鈹溾攢鈹€ analysis.py        # CASE.MultiQubit
鈹?  鈹?  鈹斺攢鈹€ metrics.py
鈹?  鈹斺攢鈹€ readout/
鈹?      鈹溾攢鈹€ __init__.py
鈹?      鈹溾攢鈹€ analysis.py        # CASE.Readout
鈹?      鈹溾攢鈹€ metrics.py         # integrated_iq / rf_signal / adc_signal / ...
鈹?      鈹斺攢鈹€ signals.py         # physical reconstruction helpers
鈹溾攢鈹€ parametric/
鈹?  鈹溾攢鈹€ __init__.py
鈹?  鈹溾攢鈹€ single_qubit/
鈹?  鈹?  鈹溾攢鈹€ __init__.py
鈹?  鈹?  鈹溾攢鈹€ analysis.py        # PARAMETRIC.SingleQubit
鈹?  鈹?  鈹斺攢鈹€ metrics.py         # final_P0 / final_P1 / final_fidelity / ...
鈹?  鈹溾攢鈹€ multi_qubit/
鈹?  鈹?  鈹溾攢鈹€ __init__.py
鈹?  鈹?  鈹溾攢鈹€ analysis.py
鈹?  鈹?  鈹斺攢鈹€ metrics.py
鈹?  鈹斺攢鈹€ readout/
鈹?      鈹溾攢鈹€ __init__.py
鈹?      鈹溾攢鈹€ analysis.py
鈹?      鈹斺攢鈹€ metrics.py
鈹溾攢鈹€ comprehensive/
鈹?  鈹溾攢鈹€ __init__.py
鈹?  鈹溾攢鈹€ iq/
鈹?  鈹?  鈹溾攢鈹€ __init__.py
鈹?  鈹?  鈹溾攢鈹€ analysis.py        # COMPREHENSIVE.IQ
鈹?  鈹?  鈹斺攢鈹€ metrics.py         # centroids / confusion_matrix / snr / ...
鈹?  鈹斺攢鈹€ cross/
鈹?      鈹溾攢鈹€ __init__.py
鈹?      鈹斺攢鈹€ analysis.py
鈹斺攢鈹€ legacy/
    鈹溾攢鈹€ __init__.py
    鈹斺攢鈹€ compatibility.py       # old names/fields -> canonical level+kind
```

  - Small filename differences are acceptable, but the final result must preserve the same architectural separation:
    - level-oriented subpackages
    - kind-oriented subpackages under each level
    - `metrics` as a subordinate module under each concrete kind
    - shared helpers isolated under `common/`
    - migration helpers isolated under `legacy/`
- Explicit target structure for `model.analysis` and persisted analysis objects:
  - The refactor should converge to a hierarchy equivalent to:

```text
model.analyses
鈹溾攢鈹€ case
鈹?  鈹溾攢鈹€ <analysis_id>
鈹?  鈹?  鈹溾攢鈹€ level = CASE
鈹?  鈹?  鈹溾攢鈹€ kind = SingleQubit | MultiQubit | Readout | ...
鈹?  鈹?  鈹溾攢鈹€ input_results = [ResultRef, ...]
鈹?  鈹?  鈹斺攢鈹€ output = CaseAnalysisOutput
鈹?  鈹?      鈹溾攢鈹€ metrics = {...}
鈹?  鈹?      鈹溾攢鈹€ signals = {...}          # when applicable
鈹?  鈹?      鈹溾攢鈹€ fit = {...}              # when applicable
鈹?  鈹?      鈹斺攢鈹€ payload = typed domain payload
鈹溾攢鈹€ parametric
鈹?  鈹溾攢鈹€ <analysis_id>
鈹?  鈹?  鈹溾攢鈹€ level = PARAMETRIC
鈹?  鈹?  鈹溾攢鈹€ kind = SingleQubit | MultiQubit | Readout | ...
鈹?  鈹?  鈹溾攢鈹€ input_results = [ResultRef, ...]
鈹?  鈹?  鈹斺攢鈹€ output = ParametricAnalysisOutput
鈹?  鈹?      鈹溾攢鈹€ axes = {...}
鈹?  鈹?      鈹溾攢鈹€ metrics = {...}
鈹?  鈹?      鈹溾攢鈹€ curves = {...}
鈹?  鈹?      鈹斺攢鈹€ payload = typed domain payload
鈹斺攢鈹€ comprehensive
    鈹溾攢鈹€ <analysis_id>
    鈹?  鈹溾攢鈹€ level = COMPREHENSIVE
    鈹?  鈹溾攢鈹€ kind = IQ | CrossStudy | ...
    鈹?  鈹溾攢鈹€ input_results = [ResultRef, ...]
    鈹?  鈹斺攢鈹€ output = ComprehensiveAnalysisOutput
    鈹?      鈹溾攢鈹€ metrics = {...}
    鈹?      鈹溾攢鈹€ summary = {...}
    鈹?      鈹斺攢鈹€ payload = typed domain payload
```

  - In schema terms, the target runtime/persistence model should make these fields explicit on every analysis object:
    - `analysis_id`
    - `level`
    - `kind`
    - optional `name` as instance label only
    - `input_results`
    - `output`
  - `ModelAnalysis`, `CaseAnalysis`, `ParametricAnalysis`, and `ComprehensiveAnalysis` should be adjusted or replaced as needed so this hierarchy is represented directly rather than inferred indirectly from string names.
  - Persistence/serialization code must preserve this explicit structure on disk.
- Key design decisions:
  - `level` is the primary architectural boundary because it determines data aggregation scope and output shape.
  - `kind` is the primary domain boundary because it determines what can be computed and how outputs are structured.
  - `metrics` is a subordinate module because the same concept name can mean different things across kinds and levels.
  - `name` should become an optional instance identifier, not the authoritative type selector.
- Suggested target config shape:
  - Example:
    - `level: CASE`
    - `kind: SingleQubit`
    - `metrics: [population, coherence_01, leakage]`
  - Example:
    - `level: CASE`
    - `kind: Readout`
    - `metrics: [integrated_iq, rf_signal, adc_signal]`
  - Example:
    - `level: COMPREHENSIVE`
    - `kind: IQ`
    - `metrics: [centroids, confusion_matrix, snr]`
- Extension points:
  - New analysis kinds should be addable without changing core workflow logic.
  - Each kind should own its supported modules and output schema.
  - Legacy aliases such as `state_analysis` should be normalized through one compatibility layer.

## 5. Required Workflow
1. Complete code changes and required tests together.
2. Update related `docstring` content before each completion step.
3. Update related `docs/` content before marking the issue done.
4. Treat `docs/site/` as generated artifacts; edit source docs first.
5. Run `mkdocs build --clean` after doc changes to keep doc outputs synchronized.
6. Mark the issue complete only after code, tests, docstrings, and docs are all synchronized.

## 6. Task Breakdown
1. Define the canonical hierarchy and migration rules for `level`, `kind`, and subordinate modules.
2. Define the target `src/musiq/analysis/` directory layout and move responsibilities so package structure reflects the new hierarchy.
3. Introduce structured analysis-step abstractions and dispatch based on `(level, kind)`.
4. Refactor model-side analysis hierarchy, including runtime objects and persistence boundaries, to match the new semantics.
5. Migrate single-qubit state analysis into a `CASE.SingleQubit` path.
6. Migrate readout-chain analysis into a `CASE.Readout` path.
7. Migrate IQ aggregation into a `COMPREHENSIVE.IQ` path.
8. Refactor config parsing so explicit `level` + `kind` is supported while keeping legacy aliases working.
9. Refactor metric resolution so metrics are resolved through the owning analysis kind rather than acting as a top-level architecture driver.
10. Add compatibility and regression tests for old and new analyser configs.
11. Validate report tasks `task1`, `task2`, `task3`, and `task6`, allowing only minor plot-script adjustments where necessary.
12. Update developer docs to explain the new architecture and extension pattern.

## 7. Definition of Done
- [ ] `CASE`, `PARAMETRIC`, and `COMPREHENSIVE` are explicit architectural concepts in analysis dispatch.
- [ ] At least `SingleQubit`, `Readout`, and `IQ` are represented as explicit analysis kinds.
- [ ] `metrics` is structurally subordinate to a concrete analysis kind.
- [ ] The final `src/musiq/analysis/` directory layout reflects the new hierarchy and separates orchestration, shared helpers, and domain analyses.
- [ ] Model/runtime analysis hierarchy is updated consistently with the package hierarchy.
- [ ] Existing report examples continue to run through a compatibility layer.
- [ ] Workflow dispatch no longer depends on overloaded `name` strings alone.
- [ ] `report/task1_single_qubit_rabi` passes.
- [ ] `report/task2_single_qubit_decoherence` passes.
- [ ] `report/task3_gaussian_drag_comparison` passes.
- [ ] `report/task6_single_qubit_readout` passes.
- [ ] Any report-side plotting adjustments are limited to minor compatibility fixes, not analysis-semantic rewrites.
- [ ] Relevant `docstring` content is updated.
- [ ] Relevant `docs/` content is updated.
- [ ] `docs/src` and `docs/site` are synchronized through a successful build.

## 8. Test Plan
- Unit tests:
  - Normalize legacy step names to canonical `level` + `kind`.
  - Resolve supported metrics from the owning analysis kind.
  - Dispatch the correct analysis handler from config.
- Integration tests:
  - Run representative single-qubit case analysis from current report configs.
  - Run representative readout case analysis and IQ comprehensive analysis.
  - Verify parametric aggregation still works after hierarchy refactor.
- Regression tests:
  - Legacy `metrics`, `case_metrics`, and `sweep_metrics` configs still behave as expected during migration.
  - Existing notebooks and report scripts that rely on `analysis.output.metrics` do not silently lose expected outputs.
  - `report/task1_single_qubit_rabi` runs successfully.
  - `report/task2_single_qubit_decoherence` runs successfully.
  - `report/task3_gaussian_drag_comparison` runs successfully.
  - `report/task6_single_qubit_readout` runs successfully.
  - Plotting code in those reports may be adjusted only minimally to accommodate the new hierarchy.
- Example command:
  - `pytest -q`

## 9. Risks and Rollback
- Major risks:
  - Breaking existing analyser configs and notebooks through partial migration.
  - Creating duplicate semantics between legacy names and canonical kinds.
  - Letting type hierarchy become too deep or too abstract relative to current code size.
- Mitigations:
  - Keep one normalization layer for legacy config aliases.
  - Migrate incrementally by introducing explicit kinds before deleting legacy paths.
  - Add focused compatibility tests for existing reports and known analysis flows.
- Rollback strategy:
  - Retain legacy config parsing and dispatch until the new hierarchy is proven stable.
  - Gate full cutover behind completion of compatibility and regression tests.

## 10. Dependencies and Blockers
- Prerequisites:
  - Agreement on canonical `level`/`kind` vocabulary.
  - Agreement on which current analyses are first-class kinds in phase 1.
- External dependencies:
  - None required beyond current Python/test/doc toolchain.
- Potential blockers:
  - Ambiguity about whether some current functions belong to `CASE` or `PARAMETRIC`.
  - Unclear ownership of output schema between workflow layer and analysis layer.

## 11. Estimate and Priority
- Priority: P1
- Estimated effort: 4-7 days
- Owner: Unassigned

## 12. References
- Related files:
  - `src/musiq/analysis/metrics.py`
  - `src/musiq/analysis/readout_chain.py`
  - `src/musiq/analysis/definitions.py`
  - `src/musiq/analysis/registry.py`
  - `src/musiq/workflow/model.py`
  - `src/musiq/workflow/stages.py`
  - `src/musiq/workflow/model_execution.py`
  - `src/musiq/schemas/results.py`
- Related reports:
  - `report/task1_single_qubit_rabi/analyser.yaml`
  - `report/task2_single_qubit_decoherence/analyser.yaml`
  - `report/task3_gaussian_drag_comparison/analyser.yaml`
  - `report/task6_single_qubit_readout/analyser.yaml`
- Related docs:
  - `issues/ISSUE_TEMPLATE.md`
