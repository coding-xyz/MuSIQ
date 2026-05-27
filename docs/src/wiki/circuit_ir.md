# Schedule Circuit IR

`CircuitIR` is now schedule-first.

## Primary Shape

```python
schedule: dict[int, list[list[CircuitGate]]]
```

- outer key
  - integer schedule layer
- outer value
  - one list indexed by qubit lane
- lane entry
  - ordered intra-layer logical gate sequence for that qubit

## Example

```yaml
schema_version: "1.0"
format: circuit_layer_yaml
num_qubits: 3
num_clbits: 0
schedule:
  0:
    - - ['rz', [0], 0.7853981633974474]
      - ['sx', [0]]
    - []
    - []
  1:
    - []
    - - ['cz', [2, 1]]
    - - ['cz', [2, 1]]
```

## Invariants

- `schedule` is the authoritative execution structure
- multi-qubit gates are mirrored across participating lanes inside the same
  schedule layer
- flattening is an explicit helper operation, not the public circuit contract
- QASM import maps directly into this structure

## Accepted Gate Entry Forms

Minimum supported YAML forms:

- `['gate_name', [qubits...]]`
- `['gate_name', [qubits...], angle_or_param]`

Dictionary entries may also be supported internally, but list form is the
portable user-facing format.
