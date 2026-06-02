"""Monte-Carlo wave-function solver mode for QuTiP."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np

from musiq.common.schemas import Trajectory
from musiq.engines.qutip.modes.common import base_metadata, build_base_e_ops, standard_trajectory_from_result
from musiq.engines.qutip.model.collapse import build_sampled_stochastic_terms
from musiq.engines.qutip.runtime import QutipPlan, QutipSolverInputs, QutipSystem, QutipTrajectoryRequest


def run_mcwf(
    *,
    setup: QutipPlan,
    system: QutipSystem,
    solver_inputs: QutipSolverInputs,
    trajectory_cfg: QutipTrajectoryRequest,
    e_ops,
):
    """Run ``qutip.mcsolve`` for quantum-jump trajectories."""
    return setup.qt.mcsolve(
        system.H,
        system.psi0,
        setup.tlist,
        c_ops=solver_inputs.c_ops,
        e_ops=e_ops,
        ntraj=setup.run_config.ntraj,
        options=trajectory_cfg.options,
    )


def _to_density_matrix(qt, state):
    data = np.asarray(state.full(), dtype=complex)
    if data.ndim == 2 and data.shape[0] == data.shape[1]:
        return state
    return qt.ket2dm(state)


def _extract_single_shot_states(result) -> list[object]:
    runs_states = list(getattr(result, "runs_states", []) or [])
    if runs_states and isinstance(runs_states[0], list):
        return list(runs_states[0])
    states = list(getattr(result, "states", []) or [])
    if states and isinstance(states[0], list):
        return list(states[0])
    return states


def _single_shot_options(base_options):
    if base_options is None:
        return {"progress_bar": False}
    if isinstance(base_options, dict):
        out = dict(base_options)
        out["progress_bar"] = False
        return out
    try:
        cloned = dict(getattr(base_options, "__dict__", {}) or {})
    except Exception:
        cloned = {}
    cloned["progress_bar"] = False
    return cloned


def _emit_shot_progress(current: int, total: int) -> None:
    total = max(1, int(total))
    current = min(max(0, int(current)), total)
    width = 24
    filled = int(round(width * current / total))
    bar = "#" * filled + "-" * (width - filled)
    sys.stderr.write(f"\rMCWF shots [{bar}] {current}/{total}")
    if current >= total:
        sys.stderr.write("\n")
    sys.stderr.flush()


def _build_stochastic_mcwf_payloads(engine, *, shot_states, density_runs, requested_kind: str):
    wave_function = None
    density_matrix = None

    serialized_wave_runs = [
        [engine._serialize_qobj_state(state) for state in run]
        for run in shot_states
    ]
    if serialized_wave_runs and str(serialized_wave_runs[0][0].get("kind", "")).strip().lower() == "wave_function":
        note = ""
        if str(requested_kind).strip().lower() == "density_matrix":
            note = "requested density_matrix but mcwf stochastic shots also preserve per-shot wave_function runs"
        wave_function = {
            "requested_kind": "wave_function",
            "actual_kind": "wave_function",
            "encoding": "complex",
            "snapshots": [item.get("data", []) for item in serialized_wave_runs[0]],
            "runs": [[item.get("data", []) for item in run] for run in serialized_wave_runs],
            "num_runs": len(serialized_wave_runs),
            "note": note,
        }

    averaged_density = engine._average_qobj_sequences(density_runs)
    serialized_density = [engine._serialize_qobj_state(state) for state in averaged_density]
    serialized_density_runs = [
        [engine._serialize_qobj_state(state) for state in run]
        for run in density_runs
    ]
    density_matrix = {
        "requested_kind": "density_matrix",
        "actual_kind": "density_matrix",
        "encoding": "complex",
        "snapshots": [item.get("data", []) for item in serialized_density],
        "runs": [[item.get("data", []) for item in run] for run in serialized_density_runs],
        "num_runs": len(serialized_density_runs),
        "note": "ensemble-averaged density_matrix with per-shot runs preserved separately",
    }
    return wave_function, density_matrix


def _run_mcwf_with_per_shot_stochastic(
    *,
    engine,
    setup: QutipPlan,
    system: QutipSystem,
    solver_inputs: QutipSolverInputs,
    trajectory_cfg: QutipTrajectoryRequest,
    e_ops,
):
    qt = setup.qt
    nshots = int(max(1, setup.run_config.ntraj))
    shot_states: list[list[object]] = []
    shot_expectations: list[list[np.ndarray]] = [[] for _ in range(len(e_ops))]
    shot_options = _single_shot_options(trajectory_cfg.options)

    for shot_idx in range(nshots):
        _emit_shot_progress(shot_idx, nshots)
        shot_seed = int(solver_inputs.seed) + int(shot_idx)
        shot_H = list(system.H) + build_sampled_stochastic_terms(
            engine,
            setup=setup,
            system=system,
            stochastic_channels=list(solver_inputs.stochastic_channels or []),
            seed=shot_seed,
        )
        shot_result = setup.qt.mcsolve(
            shot_H,
            system.psi0,
            setup.tlist,
            c_ops=solver_inputs.c_ops,
            e_ops=e_ops,
            ntraj=1,
            options=shot_options,
        )
        states = _extract_single_shot_states(shot_result)
        if not states:
            continue
        shot_states.append(states)
        expect = list(getattr(shot_result, "expect", []) or [])
        for op_idx in range(len(e_ops)):
            series = np.asarray(expect[op_idx] if op_idx < len(expect) else np.zeros(setup.tlist.size), dtype=complex).reshape(-1)
            shot_expectations[op_idx].append(series)

    if not shot_states:
        raise RuntimeError("QuTiP execution failed: stochastic MCWF returned no shot states")
    _emit_shot_progress(nshots, nshots)

    density_runs = [[_to_density_matrix(qt, state) for state in run] for run in shot_states]
    expect_payload = [
        np.stack([np.asarray(series, dtype=complex).reshape(-1) for series in runs], axis=0)
        if runs
        else np.zeros((0, setup.tlist.size), dtype=complex)
        for runs in shot_expectations
    ]
    return SimpleNamespace(
        states=density_runs[0] if density_runs else [],
        shot_states=shot_states,
        runs_states=density_runs,
        expect=expect_payload,
    )


def run_mcwf_trajectory(
    *,
    engine,
    setup: QutipPlan,
    system: QutipSystem,
    solver_inputs: QutipSolverInputs,
    trajectory_cfg: QutipTrajectoryRequest,
) -> Trajectory:
    """Run MCWF mode and return a normalized trajectory."""
    base_e_ops, readout_expect_ix = build_base_e_ops(engine, setup, system)
    try:
        if list(getattr(solver_inputs, "stochastic_channels", []) or []):
            result = _run_mcwf_with_per_shot_stochastic(
                engine=engine,
                setup=setup,
                system=system,
                solver_inputs=solver_inputs,
                trajectory_cfg=trajectory_cfg,
                e_ops=base_e_ops,
            )
        else:
            result = run_mcwf(
                setup=setup,
                system=system,
                solver_inputs=solver_inputs,
                trajectory_cfg=trajectory_cfg,
                e_ops=base_e_ops,
            )
    except Exception as exc:
        raise RuntimeError(f"QuTiP execution failed: {exc}") from exc
    if list(getattr(solver_inputs, "stochastic_channels", []) or []):
        wave_function, density_matrix = _build_stochastic_mcwf_payloads(
            engine,
            shot_states=list(getattr(result, "shot_states", []) or []),
            density_runs=list(getattr(result, "runs_states", []) or []),
            requested_kind=trajectory_cfg.requested_state_kind,
        )
        metadata = base_metadata(setup, solver_inputs)
        metadata["mcwf_ntraj"] = int(max(1, setup.run_config.ntraj))
        metadata["stochastic_shot_seeds"] = [
            int(solver_inputs.seed) + int(idx)
            for idx in range(int(max(1, setup.run_config.ntraj)))
        ]
        from musiq.engines.qutip.dynamics.postprocess import attach_postprocessed_readout

        attach_postprocessed_readout(
            engine,
            metadata=metadata,
            setup=setup,
            system=system,
            solver_inputs=solver_inputs,
            result=result,
            readout_expect_ix=readout_expect_ix,
        )
        return Trajectory(
            engine="qutip",
            times=setup.tlist.astype(float).tolist(),
            wave_function=wave_function,
            density_matrix=density_matrix,
            metadata=metadata,
        )
    return standard_trajectory_from_result(
        engine,
        setup=setup,
        system=system,
        solver_inputs=solver_inputs,
        trajectory_cfg=trajectory_cfg,
        result=result,
        readout_expect_ix=readout_expect_ix,
    )


def run_hybrid_readout(
    *,
    engine,
    setup: QutipPlan,
    system: QutipSystem,
    solver_inputs: QutipSolverInputs,
    trajectory_cfg: QutipTrajectoryRequest,
) -> Trajectory:
    """Run MCWF dynamics with hybrid classical readout feedback."""
    try:
        hybrid = engine._run_hybrid_cqed_mcwf(
            setup=setup,
            system=system,
            solver_inputs=solver_inputs,
            trajectory_cfg=trajectory_cfg,
        )
    except Exception as exc:
        raise RuntimeError(f"QuTiP execution failed: {exc}") from exc

    metadata = base_metadata(setup, solver_inputs)
    metadata["hybrid_update_mode"] = setup.hybrid_update_mode
    metadata.update(dict(hybrid.get("metadata", {}) or {}))
    qstate = dict(hybrid.get("quantum_state_trajectory", {}) or {})
    if not qstate:
        qstate = dict(metadata.pop("quantum_state_trajectory", {}) or {})
    wave_function, density_matrix = engine._quantum_payloads(qstate)
    return Trajectory(
        engine="qutip",
        times=list(hybrid.get("times", setup.tlist.astype(float).tolist()) or []),
        wave_function=wave_function,
        density_matrix=density_matrix,
        metadata=metadata,
    )
