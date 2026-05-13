"""Generate grouped API reference pages from ``src/musiq`` docstrings."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import mkdocs_gen_files


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src" / "musiq"

GROUP_DESCRIPTIONS = {
    "analysis": "Analysis passes, observables, and post-processing helpers.",
    "backend": "Compilation, lowering, model building, and backend config helpers.",
    "circuit": "OpenQASM import, export, and circuit normalization.",
    "common": "Common schemas and shared data structures.",
    "engines": "Simulation engines and engine adapters.",
    "pulse": "Pulse compilation, PulseIR helpers, and visualization tools.",
    "qec": "QEC priors, decoders, decoder eval, and summary utilities.",
    "schemas": "Structured configuration, IR, ModelSpec, and result dataclasses.",
    "session": "Sessions, manifests, and artifact-oriented persistence helpers.",
    "ui": "CLI, notebook helpers, and lightweight result summaries.",
    "workflow": "Config loading, execution planning, model API, and workflow stages.",
}

GROUP_ENTRYPOINTS = {
    "analysis": ["analysis.AnalysisRegistry", "analysis.AnalysisRunner"],
    "backend": ["backend.CompilePipeline", "backend.load_backend_config"],
    "circuit": ["circuit.CircuitAdapter"],
    "engines": [
        "engines.QuTiPEngine",
        "engines.QOpticsEngine",
        "engines.QToolboxEngine",
        "engines.StimQECAnalysisEngine",
        "engines.CirqQECAnalysisEngine",
    ],
    "pulse": ["pulse.DefaultPulseLowering", "pulse.PulseCompiler", "pulse.build_gate_mapping_catalog"],
    "qec": ["qec.get_decoder", "qec.build_prior_and_report", "qec.summarize_logical_error"],
    "schemas": [
        "schemas.ModelSpec", 
        "schemas.SystemSpec", 
        "schemas.HamiltonianSpec", 
        "schemas.NoiseSpec", 
        "schemas.ReadoutSpec", 
        "schemas.Trajectory", 
        "schemas.ModelAnalysis"
    ],
    "session": ["session.Session"],
    "ui": ["musiq.ui.plot_default"],
    "workflow": [
        "workflow.create_model",
        "workflow.load_model",
        "workflow.load_task_config_file",
        "workflow.load_solver_config_file",
        "workflow.load_device_config_file",
        "workflow.load_pulse_config_file",
        "workflow.load_analyser_config_file",
    ],
}


def _module_name(path: Path) -> str:
    """Return the import path for a source file under ``src/``."""
    rel = path.relative_to(ROOT / "src")
    if path.name == "__init__.py":
        rel = rel.parent
    else:
        rel = rel.with_suffix("")
    return ".".join(rel.parts)


def _is_public_module(path: Path) -> bool:
    """Skip private modules when generating API pages."""
    rel_parts = path.relative_to(ROOT / "src").parts
    if path.name == "__init__.py":
        return not any(part.startswith("_") for part in rel_parts[:-1])
    return not any(part.startswith("_") for part in rel_parts)


modules = sorted(path for path in SRC.rglob("*.py") if _is_public_module(path))
grouped: dict[str, list[str]] = defaultdict(list)
for py_path in modules:
    module = _module_name(py_path)
    if module == "musiq":
        continue
    parts = module.split(".")
    group = parts[1] if len(parts) > 1 else "core"
    grouped[group].append(module)

index_lines = [
    "# API Reference",
    "",
    "This section is generated from public docstrings under `src/musiq`.",
    "",
    "## How To Read This Section",
    "",
    "- Start with the entrypoints if you only need the common public APIs.",
    "- Continue into the generated module sections when you need class, function, or schema details.",
    "- Use `docs/src/wiki` for task-oriented guides and this section for API-oriented reference.",
    "",
    "## Groups",
    "",
]
for group in sorted(grouped):
    desc = GROUP_DESCRIPTIONS.get(group, "Public APIs for this module group.")
    index_lines.append(f"- [{group}](./{group}.md): {desc}")

with mkdocs_gen_files.open("api/index.md", "w") as fd:
    fd.write("\n".join(index_lines) + "\n")

for group in sorted(grouped):
    package_module = f"musiq.{group}"
    lines = [
        f"# {group}",
        "",
        GROUP_DESCRIPTIONS.get(group, f"Public APIs under `src/musiq/{group}`."),
        "",
        "## Entrypoints",
        "",
    ]
    entries = GROUP_ENTRYPOINTS.get(group, [])
    if entries:
        for entry in entries:
            lines.append(f"- `{entry}`")
    else:
        lines.append("- No curated entrypoints for this group yet. See the module list below.")

    lines.extend(
        [
            "",
            "## Package Export",
            "",
            f"::: {package_module}",
            "",
            "## Modules",
            "",
        ]
    )

    for module in sorted(grouped[group]):
        if module == package_module:
            continue
        lines.append(f"## `{module}`")
        lines.append("")
        lines.append(f"::: {module}")
        lines.append("")

    with mkdocs_gen_files.open(f"api/{group}.md", "w") as fd:
        fd.write("\n".join(lines) + "\n")
