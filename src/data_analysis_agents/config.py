"""Read solver configuration without importing model or tool dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from .types import RunConfig


@dataclass(frozen=True)
class ResolvedPaths:
    """Resolved paths for the host runtime, never passed directly to the model."""

    project_root: Path
    manifest: Path
    raw_data: Path
    run_dir: Path


def _resolve_under(root: Path, relative: str) -> Path:
    """Resolve a path and reject symlink escapes.
    Args: root: Allowed directory; relative: Validated relative path.
    """
    candidate = (root / relative).resolve()
    if candidate == root or root not in candidate.parents:
        raise ValueError(f"path escapes its allowed directory: {relative!r}")
    return candidate


def resolve_paths(config: RunConfig, project_root: Union[str, Path]) -> ResolvedPaths:
    """Resolve configured paths without reading inputs or creating outputs.
    Args: config: Solver configuration; project_root: Repository root.
    """
    config = RunConfig.model_validate(config)
    root = Path(project_root).resolve()
    manifest = _resolve_under(root, config.paths.manifest)
    input_root = root / "data/preprocessed/task_inputs"
    if input_root not in manifest.parents:
        raise ValueError("manifest resolves outside task_inputs/")
    raw_data = _resolve_under(root, config.paths.raw_data)
    if raw_data != root / "data/raw_data":
        raise ValueError("raw_data must resolve to the project's input directory")
    run_dir = _resolve_under(
        root, f"{config.paths.runs_dir}/{config.experiment_id}/{config.run_id}"
    )
    if root / "runs" not in run_dir.parents:
        raise ValueError("run directory resolves outside runs/")
    return ResolvedPaths(root, manifest, raw_data, run_dir)


def resolve_mcp_paths(config: RunConfig, project_root: Union[str, Path]) -> tuple[Path, Path]:
    """Resolve project-local service executables, allowing a venv interpreter symlink."""
    root = Path(project_root).resolve()
    service = config.mcp.excel
    executables = []
    for relative in (service.python, service.command):
        path = root / relative
        _resolve_under(root, str(Path(relative).parent))
        executables.append(path)
    if executables[1].resolve() != executables[1]:
        raise ValueError("MCP server command must not be redirected outside its configured path")
    return executables[0], executables[1]


def load_config(
    path: Union[str, Path], project_root: Optional[Union[str, Path]] = None
) -> RunConfig:
    """Load YAML and validate schema and configured path boundaries.
    Args: path: YAML file; project_root: Root, defaulting to the YAML directory.
    """
    import yaml

    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    config = RunConfig.model_validate(data)
    root = config_path.parent if project_root is None else project_root
    resolve_paths(config, root)
    if config.mcp.excel.enabled:
        resolve_mcp_paths(config, root)
    return config
