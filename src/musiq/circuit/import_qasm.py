"""OpenQASM 3 parsing and serialization helpers for CircuitIR."""

from __future__ import annotations

import ast
from dataclasses import asdict
import math
import re

from musiq.common.schemas import CircuitGate, CircuitIR
from musiq.schemas.circuit import build_serial_schedule, flatten_schedule


class CircuitAdapter:
    """Adapter between OpenQASM/Qiskit and ``CircuitIR``."""
    # Version Dispatchers
    _HEADER_V2_RE = re.compile(r"^OPENQASM\s+2(?:\.0)?\s*$", re.IGNORECASE)
    _HEADER_V3_RE = re.compile(r"^OPENQASM\s+3(?:\.0)?\s*$", re.IGNORECASE)

    # QASM 2.0 Specifics
    _V2_DECL_QUBIT_RE = re.compile(r"^qreg\s+([A-Za-z_]\w*)\[(\d+)\]\s*$", re.IGNORECASE)
    _V2_DECL_BIT_RE = re.compile(r"^creg\s+([A-Za-z_]\w*)\[(\d+)\]\s*$", re.IGNORECASE)
    _V2_MEASURE_RE = re.compile(
        r"^measure\s+([A-Za-z_]\w*)\[(\d+)\]\s*=\s*([A-Za-z_]\w*)\[(\d+)\]\s*$",
        re.IGNORECASE,
    )

    # QASM 3.0 Specifics
    _V3_DECL_QUBIT_RE = re.compile(r"^qubit\[(\d+)\]\s+([A-Za-z_]\w*)\s*$", re.IGNORECASE)
    _V3_DECL_BIT_RE = re.compile(r"^bit\[(\d+)\]\s+([A-Za-z_]\w*)\s*$", re.IGNORECASE)
    _V3_MEASURE_RE = re.compile(
        r"^measure\s+([A-Za-z_]\w*)\[(\d+)\]\s*->\s*([A-Za-z_]\w*)\[(\d+)\]\s*$",
        re.IGNORECASE,
    )

    # Shared
    _GATE_RE = re.compile(r"^([A-Za-z_]\w*)(?:\(([^)]*)\))?\s+(.+)\s*$", re.IGNORECASE)

    @staticmethod
    def _split_statements(qasm_text: str) -> list[str]:
        cleaned = []
        for idx, line in enumerate(qasm_text.splitlines()):
            if idx == 0:
                line = line.lstrip("\ufeff")
            line = line.split("//", 1)[0].strip()
            if line:
                cleaned.append(line)
        merged = " ".join(cleaned)
        return [s.strip() for s in merged.split(";") if s.strip()]

    @staticmethod
    def _parse_indexed_ref(token: str) -> tuple[str, int]:
        m = re.match(r"^([A-Za-z_]\w*)\[(\d+)\]$", token.strip())
        if not m:
            raise ValueError(f"Invalid indexed argument: {token}")
        return m.group(1), int(m.group(2))

    @staticmethod
    def _eval_param_expr(expr: str, bindings: dict[str, float] | None = None) -> float:
        """Evaluate a restricted numeric expression used in QASM gate parameters."""
        bindings = bindings or {}
        allowed_names = {"pi": math.pi, "tau": math.tau, "e": math.e}
        allowed_names.update({str(k): float(v) for k, v in bindings.items()})

        tree = ast.parse(expr, mode="eval")
        for node in ast.walk(tree):
            if isinstance(
                node,
                (
                    ast.Expression,
                    ast.BinOp,
                    ast.UnaryOp,
                    ast.Add,
                    ast.Sub,
                    ast.Mult,
                    ast.Div,
                    ast.Pow,
                    ast.USub,
                    ast.UAdd,
                    ast.Constant,
                    ast.Load,
                    ast.Name,
                ),
            ):
                continue
            raise ValueError(f"Unsupported parameter expression: {expr}")
        try:
            value = eval(compile(tree, "<qasm-param>", "eval"), {"__builtins__": {}}, allowed_names)
        except NameError as exc:
            raise ValueError(f"Unbound parameter in expression '{expr}': {exc}") from exc
        except Exception as exc:
            raise ValueError(f"Invalid parameter expression '{expr}': {exc}") from exc
        return float(value)

    @staticmethod
    def from_qasm(qasm_text: str, param_bindings: dict[str, float] | None = None) -> CircuitIR:
        """Parse an OpenQASM program into ``CircuitIR`` by dispatching to version-specific parsers."""
        statements = CircuitAdapter._split_statements(qasm_text)
        if not statements:
            raise ValueError("Empty QASM input")
        
        header = statements[0]
        if CircuitAdapter._HEADER_V2_RE.match(header):
            return CircuitAdapter.from_qasm2(qasm_text, statements, param_bindings)
        if CircuitAdapter._HEADER_V3_RE.match(header):
            return CircuitAdapter.from_qasm3(qasm_text, statements, param_bindings)
        
        raise ValueError("Unsupported or missing OpenQASM header (only 2.0 and 3.0 are supported)")

    @staticmethod
    def from_qasm2(qasm_text: str, statements: list[str], param_bindings: dict[str, float] | None = None) -> CircuitIR:
        """Parse OpenQASM 2.0 program."""
        qregs: dict[str, tuple[int, int]] = {}
        cregs: dict[str, tuple[int, int]] = {}
        next_q, next_c = 0, 0
        gates: list[CircuitGate] = []

        for st in statements[1:]:
            if st.lower().startswith("include "): continue

            # Qubit reg: qreg name[size]
            qd = CircuitAdapter._V2_DECL_QUBIT_RE.match(st)
            if qd:
                name, size = qd.group(1), int(qd.group(2))
                if name in qregs: raise ValueError(f"Duplicate qubit register: {name}")
                qregs[name] = (next_q, size)
                next_q += size
                continue

            # Bit reg: creg name[size]
            cd = CircuitAdapter._V2_DECL_BIT_RE.match(st)
            if cd:
                name, size = cd.group(1), int(cd.group(2))
                if name in cregs: raise ValueError(f"Duplicate bit register: {name}")
                cregs[name] = (next_c, size)
                next_c += size
                continue

            # Measure: measure q[i] = c[j]
            mm = CircuitAdapter._V2_MEASURE_RE.match(st)
            if mm:
                qreg, qidx, creg, cidx = mm.group(1), int(mm.group(2)), mm.group(3), int(mm.group(4))
                if qreg not in qregs or creg not in cregs: raise ValueError("Unknown register in measure")
                qoff, qsize = qregs[qreg]
                coff, csize = cregs[creg]
                if qidx >= qsize or cidx >= csize: raise ValueError("Index out of range in measure")
                gates.append(CircuitGate(name="measure", qubits=[qoff + qidx], clbits=[coff + cidx]))
                continue

            # Gates
            gm = CircuitAdapter._GATE_RE.match(st)
            if not gm: raise ValueError(f"Unsupported QASM 2.0 statement: '{st};'")
            name, params_raw, args_raw = gm.group(1).lower(), (gm.group(2) or "").strip(), gm.group(3).strip()
            if name == "barrier": continue
            
            arg_tokens = [a.strip() for a in args_raw.split(",") if a.strip()]
            if not arg_tokens: raise ValueError(f"Gate has no args: '{st};'")
            
            qubits = []
            for tok in arg_tokens:
                reg, idx = CircuitAdapter._parse_indexed_ref(tok)
                if reg not in qregs: raise ValueError(f"Unknown qubit register: {reg}")
                off, size = qregs[reg]
                if idx >= size: raise ValueError(f"Index out of range: {reg}[{idx}]")
                qubits.append(off + idx)

            params = []
            if params_raw:
                for val in [x.strip() for x in params_raw.split(",") if x.strip()]:
                    params.append(CircuitAdapter._eval_param_expr(val, bindings=param_bindings))
            gates.append(CircuitGate(name=name, qubits=qubits, params=params))

        return CircuitIR(
            num_qubits=next_q,
            num_clbits=next_c,
            schedule=build_serial_schedule(gates, num_qubits=next_q),
            source_qasm=qasm_text,
        )

    @staticmethod
    def from_qasm3(qasm_text: str, statements: list[str], param_bindings: dict[str, float] | None = None) -> CircuitIR:
        """Parse OpenQASM 3 program."""
        qregs: dict[str, tuple[int, int]] = {}
        cregs: dict[str, tuple[int, int]] = {}
        next_q, next_c = 0, 0
        gates: list[CircuitGate] = []

        for st in statements[1:]:
            if st.lower().startswith("include "): continue

            # Qubit reg: qubit[size] name
            qd = CircuitAdapter._V3_DECL_QUBIT_RE.match(st)
            if qd:
                size, name = int(qd.group(1)), qd.group(2)
                if name in qregs: raise ValueError(f"Duplicate qubit register: {name}")
                qregs[name] = (next_q, size)
                next_q += size
                continue

            # Bit reg: bit[size] name
            cd = CircuitAdapter._V3_DECL_BIT_RE.match(st)
            if cd:
                size, name = int(cd.group(1)), cd.group(2)
                if name in cregs: raise ValueError(f"Duplicate bit register: {name}")
                cregs[name] = (next_c, size)
                next_c += size
                continue

            # Measure: measure q[i] -> c[j]
            mm = CircuitAdapter._V3_MEASURE_RE.match(st)
            if mm:
                qreg, qidx, creg, cidx = mm.group(1), int(mm.group(2)), mm.group(3), int(mm.group(4))
                if qreg not in qregs or creg not in cregs: raise ValueError("Unknown register in measure")
                qoff, qsize = qregs[qreg]
                coff, csize = cregs[creg]
                if qidx >= qsize or cidx >= csize: raise ValueError("Index out of range in measure")
                gates.append(CircuitGate(name="measure", qubits=[qoff + qidx], clbits=[coff + cidx]))
                continue

            # Gates
            gm = CircuitAdapter._GATE_RE.match(st)
            if not gm: raise ValueError(f"Unsupported QASM 3 statement: '{st};'")
            name, params_raw, args_raw = gm.group(1).lower(), (gm.group(2) or "").strip(), gm.group(3).strip()
            if name == "barrier": continue
            
            arg_tokens = [a.strip() for a in args_raw.split(",") if a.strip()]
            if not arg_tokens: raise ValueError(f"Gate has no args: '{st};'")
            
            qubits = []
            for tok in arg_tokens:
                reg, idx = CircuitAdapter._parse_indexed_ref(tok)
                if reg not in qregs: raise ValueError(f"Unknown qubit register: {reg}")
                off, size = qregs[reg]
                if idx >= size: raise ValueError(f"Index out of range: {reg}[{idx}]")
                qubits.append(off + idx)

            params = []
            if params_raw:
                for val in [x.strip() for x in params_raw.split(",") if x.strip()]:
                    params.append(CircuitAdapter._eval_param_expr(val, bindings=param_bindings))
            gates.append(CircuitGate(name=name, qubits=qubits, params=params))

        return CircuitIR(
            num_qubits=next_q,
            num_clbits=next_c,
            schedule=build_serial_schedule(gates, num_qubits=next_q),
            source_qasm=qasm_text,
        )

    @staticmethod
    def from_qiskit(qc: object) -> CircuitIR:
        """Convert a Qiskit ``QuantumCircuit`` into ``CircuitIR``."""
        gates: list[CircuitGate] = []
        num_qubits = int(getattr(qc, "num_qubits"))
        num_clbits = int(getattr(qc, "num_clbits", 0))
        for inst in getattr(qc, "data", []):
            op = inst.operation
            if getattr(op, "name", "").lower() == "barrier":
                continue
            qargs = [qb._index for qb in inst.qubits]
            cargs = [cb._index for cb in inst.clbits]
            params = [float(p) for p in getattr(op, "params", [])]
            gates.append(CircuitGate(name=op.name, qubits=qargs, clbits=cargs, params=params))

        source_qasm = ""
        try:
            from qiskit import qasm3

            source_qasm = qasm3.dumps(qc)
        except Exception:
            source_qasm = ""

        return CircuitIR(
            num_qubits=num_qubits,
            num_clbits=num_clbits,
            schedule=build_serial_schedule(gates, num_qubits=num_qubits),
            source_qasm=source_qasm,
        )

    @staticmethod
    def to_qasm(circuit: CircuitIR, qasm_version: str = "3.0") -> str:
        """Serialize ``CircuitIR`` into OpenQASM text. Defaults to version 3.0."""
        if qasm_version == "2.0":
            return CircuitAdapter.to_qasm2(circuit)
        return CircuitAdapter.to_qasm3(circuit)

    @staticmethod
    def to_qasm2(circuit: CircuitIR) -> str:
        """Serialize ``CircuitIR`` into OpenQASM 2.0 text."""
        lines = [
            "OPENQASM 2.0;",
            f"qreg q[{circuit.num_qubits}];",
        ]
        if circuit.num_clbits:
            lines.append(f"creg c[{circuit.num_clbits}];")

        for g in flatten_schedule(circuit.schedule):
            if g.name == "barrier": continue
            qargs = ", ".join([f"q[{idx}]" for idx in g.qubits])
            if g.name == "measure" and g.clbits:
                lines.append(f"measure {qargs} = c[{g.clbits[0]}];")
            else:
                if g.params:
                    p = ", ".join([str(x) for x in g.params])
                    lines.append(f"{g.name}({p}) {qargs};")
                else:
                    lines.append(f"{g.name} {qargs};")
        return "\n".join(lines) + "\n"

    @staticmethod
    def to_qasm3(circuit: CircuitIR) -> str:
        """Serialize ``CircuitIR`` into OpenQASM 3 text."""
        lines = [
            "OPENQASM 3;",
            f"qubit[{circuit.num_qubits}] q;",
        ]
        if circuit.num_clbits:
            lines.append(f"bit[{circuit.num_clbits}] c;")

        for g in flatten_schedule(circuit.schedule):
            if g.name == "barrier": continue
            qargs = ", ".join([f"q[{idx}]" for idx in g.qubits])
            if g.name == "measure" and g.clbits:
                lines.append(f"measure {qargs} -> c[{g.clbits[0]}];")
            else:
                if g.params:
                    p = ", ".join([str(x) for x in g.params])
                    lines.append(f"{g.name}({p}) {qargs};")
                else:
                    lines.append(f"{g.name} {qargs};")
        return "\n".join(lines) + "\n"

    @staticmethod
    def to_qiskit(circuit: CircuitIR) -> object:
        """Convert ``CircuitIR`` to Qiskit ``QuantumCircuit``.

        Raises:
            RuntimeError: If Qiskit is unavailable.
        """
        try:
            from qiskit.circuit import Gate, QuantumCircuit
        except Exception as exc:
            raise RuntimeError("qiskit is required for CircuitAdapter.to_qiskit") from exc

        qc = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
        standard_gate_map = {
            "x": qc.x,
            "sx": qc.sx,
            "h": qc.h,
            "rx": qc.rx,
            "ry": qc.ry,
            "z": qc.z,
            "rz": qc.rz,
            "cx": qc.cx,
            "cz": qc.cz,
            "id": qc.id,
        }
        for g in flatten_schedule(circuit.schedule):
            if g.name == "barrier":
                continue
            if g.name == "measure":
                if len(g.qubits) != len(g.clbits):
                    raise ValueError("Measure gate must map qubits to clbits one-by-one")
                for q, c in zip(g.qubits, g.clbits):
                    qc.measure(q, c)
                continue

            if g.name in standard_gate_map:
                fn = standard_gate_map[g.name]
                if g.name in {"rx", "ry", "rz"}:
                    if len(g.params) != 1 or len(g.qubits) != 1:
                        raise ValueError(f"{g.name} requires exactly one parameter and one qubit")
                    fn(g.params[0], g.qubits[0])
                elif g.name in {"cx", "cz"}:
                    if len(g.qubits) != 2:
                        raise ValueError(f"{g.name} requires exactly two qubits")
                    fn(g.qubits[0], g.qubits[1])
                elif len(g.qubits) == 1:
                    fn(g.qubits[0])
                else:
                    raise ValueError(f"Unsupported arity for gate {g.name}")
                continue

            gate = Gate(name=g.name, num_qubits=len(g.qubits), params=list(g.params))
            qc.append(gate, qargs=g.qubits, cargs=[])

        return qc

    @staticmethod
    def to_json(circuit: CircuitIR) -> dict:
        """Convert ``CircuitIR`` dataclass to plain JSON-compatible dict."""
        return asdict(circuit)
