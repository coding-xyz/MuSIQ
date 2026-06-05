from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, fields, replace

from .common import (
    CalibrationConfig,
    CalibrationResult,
    GateCalibrationResult,
    SUPPORTED_SINGLE_QUBIT_RECIPES,
    SUPPORTED_TWO_QUBIT_RECIPES,
    apply_result_to_model,
    flatten_target_results,
    pair_targets,
    resolve_resource_ids,
    single_targets,
    validate_gate_selection,
)
from .single_qubit import calibrate_single_gate
from .two_qubit import calibrate_two_qubit_gate, resolve_iswap_target_channel

_CALIBRATION_CONFIG_FIELDS = {item.name for item in fields(CalibrationConfig)}


def _normalize_config(config: CalibrationConfig | None) -> CalibrationConfig:
    if config is None:
        return CalibrationConfig()
    return CalibrationConfig(
        gates=tuple(str(item).strip() for item in config.gates) if config.gates is not None else None,
        pulse_id=str(config.pulse_id).strip() if config.pulse_id is not None else None,
        solver_id=str(config.solver_id).strip() if config.solver_id is not None else None,
        device_id=str(config.device_id).strip() if config.device_id is not None else None,
        component_id=str(config.component_id).strip() if config.component_id is not None else None,
        component_ids=tuple(str(item).strip() for item in config.component_ids) if config.component_ids is not None else None,
        pair_component_ids=tuple((str(a).strip(), str(b).strip()) for a, b in config.pair_component_ids) if config.pair_component_ids is not None else None,
        disable_noise=bool(config.disable_noise),
        calibration_solver_mode=str(config.calibration_solver_mode).strip() if config.calibration_solver_mode is not None else None,
        update_model=bool(config.update_model),
        points=int(config.points),
        rounds=int(config.rounds),
        relative_span=float(config.relative_span),
        maxiter=int(config.maxiter),
        proximity_weight=float(config.proximity_weight),
        print_results=bool(config.print_results),
    )


def resolve_calibration_config(
    config: CalibrationConfig | None = None,
    **overrides,
) -> CalibrationConfig:
    base_config = config or CalibrationConfig()
    if not isinstance(base_config, CalibrationConfig):
        raise TypeError("Calibration config must be a CalibrationConfig instance or None.")
    unknown = sorted(set(overrides.keys()) - _CALIBRATION_CONFIG_FIELDS)
    if unknown:
        joined = ", ".join(unknown)
        raise TypeError(f"Unknown calibration option(s): {joined}")
    params = asdict(base_config)
    params.update(overrides)
    return _normalize_config(CalibrationConfig(**params))


def resolve_runtime_config(
    model,
    config: CalibrationConfig | None = None,
    **overrides,
) -> CalibrationConfig:
    settings = resolve_calibration_config(config, **overrides)
    resolved_pulse_id, resolved_solver_id, resolved_device_id = resolve_resource_ids(
        model,
        pulse_id=settings.pulse_id,
        solver_id=settings.solver_id,
        device_id=settings.device_id,
    )
    return replace(
        settings,
        pulse_id=resolved_pulse_id,
        solver_id=resolved_solver_id,
        device_id=resolved_device_id,
    )


def calibrate_model(
    model,
    config: CalibrationConfig | None = None,
    **overrides,
) -> CalibrationResult:
    settings = resolve_runtime_config(model, config, **overrides)
    ordered_gates = validate_gate_selection(model, pulse_id=settings.pulse_id, gates=list(settings.gates) if settings.gates is not None else None)
    requested_components = (
        [str(settings.component_id).strip()]
        if settings.component_id is not None
        else [str(item).strip() for item in list(settings.component_ids or []) if str(item).strip()]
    ) or None

    single_gate_targets = single_targets(
        model,
        solver_id=str(settings.solver_id),
        device_id=str(settings.device_id),
        component_ids=requested_components,
    )
    two_qubit_gate_targets = pair_targets(
        model,
        solver_id=str(settings.solver_id),
        device_id=str(settings.device_id),
        pair_component_ids=list(settings.pair_component_ids) if settings.pair_component_ids is not None else None,
        gate_name="cz" if "cz" in ordered_gates else "iswap" if "iswap" in ordered_gates else "cz",
    )

    working_model = model.copy(include_results=False)
    target_results: dict[str, dict[str, GateCalibrationResult]] = {}
    for gate_name in ordered_gates:
        if gate_name in SUPPORTED_SINGLE_QUBIT_RECIPES:
            if not single_gate_targets:
                raise ValueError(f"No single-qubit calibration targets available for gate `{gate_name}`.")
            for target in single_gate_targets:
                result = calibrate_single_gate(
                    working_model,
                    target,
                    gate_name=gate_name,
                    config=settings,
                )
                target_results.setdefault(target.key, {})[gate_name] = result
                apply_result_to_model(working_model, pulse_id=str(settings.pulse_id), result=result)
        elif gate_name in SUPPORTED_TWO_QUBIT_RECIPES:
            if not two_qubit_gate_targets:
                raise ValueError(f"No two-qubit calibration targets available for gate `{gate_name}`.")
            for target in two_qubit_gate_targets:
                active_target = resolve_iswap_target_channel(model, device_id=str(settings.device_id), target=target) if gate_name == "iswap" else target
                result = calibrate_two_qubit_gate(
                    working_model,
                    active_target,
                    gate_name=gate_name,
                    config=settings,
                )
                target_results.setdefault(active_target.key, {})[gate_name] = result
                apply_result_to_model(working_model, pulse_id=str(settings.pulse_id), result=result)
        else:
            raise ValueError(f"Unsupported calibration gate `{gate_name}`.")

    if settings.update_model:
        for gate_map in target_results.values():
            for result in gate_map.values():
                apply_result_to_model(model, pulse_id=str(settings.pulse_id), result=result)

    flattened_results, resolved_component_id = flatten_target_results(target_results)
    calibration_result = CalibrationResult(
        pulse_id=str(settings.pulse_id),
        solver_id=str(settings.solver_id),
        device_id=str(settings.device_id),
        component_id=resolved_component_id,
        disable_noise=bool(settings.disable_noise),
        calibration_solver_mode=str(settings.calibration_solver_mode) if settings.calibration_solver_mode is not None else None,
        model_updated=bool(settings.update_model),
        results=flattened_results,
        target_results=deepcopy(target_results),
    )
    if settings.print_results:
        print(calibration_result.format_summary())
    return calibration_result
