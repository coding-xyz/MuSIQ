"""Collapse-operator and stochastic-noise lowering for QuTiP."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from musiq.engines.qutip.runtime import QutipPlan, QutipRunConfig, QutipSolverInputs, QutipSystem


def build_collapse_and_noise(
    engine,
    setup: QutipPlan,
    system: QutipSystem,
) -> QutipSolverInputs:
    """Build QuTiP collapse operators and append stochastic noise terms."""
    model_spec = setup.model_spec
    model_type = setup.model_type
    n_qubits = setup.n_qubits
    c_ops = []
    runtime_metadata: dict[str, Any] = {}
    cavity_a = system.cavity_a
    if engine._is_cqed_model(model_type) and cavity_a is not None:
        readout_chain = setup.readout_chain
        cavity_kappa_int = max(0.0, 2.0 * math.pi * float(readout_chain.get("kappa_int_Hz", 0.0)))
        cavity_kappa_ext = max(0.0, 2.0 * math.pi * float(readout_chain.get("kappa_ext_Hz", 0.0)))
        if cavity_kappa_int > 0.0:
            c_ops.append(math.sqrt(cavity_kappa_int) * cavity_a)
        if cavity_kappa_ext > 0.0 and setup.readout_mode != "monitored_sme":
            c_ops.append(math.sqrt(cavity_kappa_ext) * cavity_a)

    for item in model_spec.noise.collapse_channels:
        target = int(item.target)
        if target < 0 or target >= n_qubits:
            continue
        kind = str(item.kind or "relaxation").lower()
        rate = max(0.0, float(item.rate_rad_s))
        if rate <= 0:
            continue
        if kind == "relaxation":
            c_ops.append(math.sqrt(rate) * system.lower_ops[target])
        elif kind == "dephasing":
            c_ops.append(
                engine._dephasing_collapse_prefactor(rate, model_type)
                * _dephasing_collapse_operator(setup=setup, system=system, target=target, model_type=model_type)
            )
        elif kind == "excitation":
            c_ops.append(math.sqrt(rate) * system.raise_ops[target])

    if setup.solver == "mcwf":
        selected_noise, seed, stochastic_channels = _collect_stochastic_noise_channels(setup, setup.run_config)
        if stochastic_channels:
            runtime_metadata["stochastic_realizations_per_shot"] = True
        return QutipSolverInputs(
            c_ops=c_ops,
            selected_noise=selected_noise,
            seed=seed,
            stochastic_channels=stochastic_channels,
            runtime_metadata=runtime_metadata,
        )

    selected_noise, seed = _append_stochastic_noise(engine, setup, system, setup.run_config)
    return QutipSolverInputs(
        c_ops=c_ops,
        selected_noise=selected_noise,
        seed=seed,
        runtime_metadata=runtime_metadata,
    )


def _dephasing_collapse_operator(*, setup: QutipPlan, system: QutipSystem, target: int, model_type: str):
    op = system.z_ops[target]
    if str(model_type).strip().lower() == "qubit_network":
        return op
    if getattr(setup, "qt", None) is not None:
        ident = setup.qt.qeye(op.dims[0])
    else:
        data = np.asarray(op.full(), dtype=complex)
        ident = type(op)(np.eye(data.shape[0], dtype=complex))
    return ident + (-2.0 * op)


def _collect_stochastic_noise_channels(
    setup: QutipPlan,
    run_config: QutipRunConfig,
) -> tuple[str, int, list[Any]]:
    model_spec = setup.model_spec
    selected_noise = str(model_spec.noise.selected_model or "markovian_lindblad").lower()
    seed = int(run_config.seed)
    if setup.solver == "heom":
        return selected_noise, seed, []
    stochastic = list(model_spec.noise.stochastic_channels)
    if not stochastic:
        return selected_noise, seed, []
    return selected_noise, seed, stochastic


def _append_stochastic_noise(
    engine,
    setup: QutipPlan,
    system: QutipSystem,
    run_config: QutipRunConfig,
) -> tuple[str, int]:
    model_spec = setup.model_spec
    selected_noise = str(model_spec.noise.selected_model or "markovian_lindblad").lower()
    stochastic = list(model_spec.noise.stochastic_channels)
    seed = int(run_config.seed)
    if setup.solver == "heom":
        return selected_noise, seed
    rng = np.random.default_rng(seed)
    if not stochastic:
        return selected_noise, seed
    for item in stochastic:
        channel_noise = _channel_noise_model(item, selected_noise)
        if channel_noise not in {"one_over_f", "1/f", "pink", "ou"}:
            continue
        targets = list(getattr(item, "targets", []) or [int(item.q)])
        targets = [int(target) for target in targets if 0 <= int(target) < setup.n_qubits]
        if not targets:
            continue
        if channel_noise in {"one_over_f", "1/f", "pink"}:
            series = engine._one_over_f_trace(
                tlist=setup.tlist,
                amp=float(item.one_over_f_amp_rad_s),
                fmin=float(item.one_over_f_fmin),
                fmax=float(item.one_over_f_fmax or 0.5 / max(setup.dt, 1e-12)),
                exponent=float(item.one_over_f_exponent),
                ncomp=int(run_config.one_over_f_components),
                rng=rng,
            )
        else:
            series = engine._ou_trace(
                tlist=setup.tlist,
                sigma=float(item.ou_sigma_rad_s),
                tau=float(item.ou_tau),
                rng=rng,
            )
        for target in targets:
            system.H.append(
                [
                    _stochastic_coupling_operator(setup=setup, system=system, item=item, target=target),
                    lambda t, _a=None, s=series, x=setup.tlist: float(np.interp(float(t), x, s)),
                ]
            )
    return selected_noise, seed


def build_sampled_stochastic_terms(
    engine,
    *,
    setup: QutipPlan,
    system: QutipSystem,
    stochastic_channels: list[Any],
    seed: int,
) -> list[Any]:
    if not stochastic_channels or setup.solver == "heom":
        return []
    rng = np.random.default_rng(int(seed))
    sampled_terms: list[Any] = []
    for item in stochastic_channels:
        channel_noise = _channel_noise_model(item, str(setup.model_spec.noise.selected_model or "").lower())
        if channel_noise not in {"one_over_f", "1/f", "pink", "ou"}:
            continue
        targets = list(getattr(item, "targets", []) or [int(item.q)])
        targets = [int(target) for target in targets if 0 <= int(target) < setup.n_qubits]
        if not targets:
            continue
        if channel_noise in {"one_over_f", "1/f", "pink"}:
            series = engine._one_over_f_trace(
                tlist=setup.tlist,
                amp=float(item.one_over_f_amp_rad_s),
                fmin=float(item.one_over_f_fmin),
                fmax=float(item.one_over_f_fmax or 0.5 / max(setup.dt, 1e-12)),
                exponent=float(item.one_over_f_exponent),
                ncomp=int(setup.run_config.one_over_f_components),
                rng=rng,
            )
        else:
            series = engine._ou_trace(
                tlist=setup.tlist,
                sigma=float(item.ou_sigma_rad_s),
                tau=float(item.ou_tau),
                rng=rng,
            )
        for target in targets:
            sampled_terms.append(
                [
                    _stochastic_coupling_operator(setup=setup, system=system, item=item, target=target),
                    lambda t, _a=None, s=series, x=setup.tlist: float(np.interp(float(t), x, s)),
                ]
            )
    return sampled_terms


def _channel_noise_model(item: Any, selected_noise: str) -> str:
    for name in ("kind", "model", "noise_model"):
        value = getattr(item, name, None)
        if value:
            return str(value).strip().lower()
    if isinstance(item, dict):
        for name in ("kind", "model", "noise_model"):
            value = item.get(name)
            if value:
                return str(value).strip().lower()
    return selected_noise


def _stochastic_coupling_operator(*, setup: QutipPlan, system: QutipSystem, item: Any, target: int):
    operator = str(getattr(item, "operator", "") or "").strip().lower()
    if operator in {"sigma_z_over_2", "0.5*sigma_z", "half_sigma_z"}:
        return 0.5 * system.z_ops[target]
    return system.z_ops[target]
