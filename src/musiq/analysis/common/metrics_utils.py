"""Shared utilities for metric computation."""

from typing import Any

def _complex_scalar(value) -> complex:
    if isinstance(value, complex):
        return value
    if isinstance(value, dict) and "__musiq_complex__" in value:
        pair = list(value.get("__musiq_complex__", []) or [])
        if len(pair) >= 2:
            return complex(float(pair[0]), float(pair[1]))
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return complex(float(value[0]), float(value[1]))
    return complex(float(value), 0.0)

def _basis_labels(dimension: int, num_qubits: int, levels: int) -> list[str]:
    if dimension <= 0:
        return []
    if num_qubits > 0 and levels > 1:
        expected = levels**num_qubits
        if expected == dimension:
            labels: list[str] = []
            for idx in range(dimension):
                digits: list[str] = []
                rem = idx
                for _ in range(num_qubits):
                    digits.append(str(rem % levels))
                    rem //= levels
                labels.append("".join(reversed(digits)))
            return labels
    return [str(i) for i in range(dimension)]

def _label_excitation_value(label: str, *, num_qubits: int) -> float:
    digits = [int(ch) for ch in str(label) if ch.isdigit()]
    if not digits:
        return 0.0
    if num_qubits > 0 and len(digits) >= num_qubits:
        return float(sum(digits[:num_qubits])) / float(num_qubits)
    return float(sum(digits)) / float(len(digits))

def _metric_terminal_value(value: Any):
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        if isinstance(value.get("values"), list) and value.get("values"):
            tail = value["values"][-1]
            if isinstance(tail, (int, float)):
                return float(tail)
        if isinstance(value.get("values"), dict):
            return None
    return None