"""Manifest-backed local Python tool, with bounded feedback and complete logs."""

from __future__ import annotations

import json
from contextlib import nullcontext
import hashlib
import math
import re
import tempfile
from pathlib import Path
from time import monotonic
from typing import Any, Optional

from pydantic import Field, ValidationError, field_validator, model_validator

from .prompts import python_prompt
from .answer_validation import ComputedResult
from .python_runner import run_python, PythonSession
from .runtime import RuntimeStore
from .schema import Record
from .types import ArtifactRef, Identifier, Observation, UsageCounters, Variant, WorkspacePolicy


def artifact_id(reference: ArtifactRef) -> str:
    """Expose a stable opaque ID instead of accepting model-supplied filesystem paths."""
    return "artifact-" + hashlib.sha256(reference.path.encode("utf-8")).hexdigest()[:24]


class PythonArguments(Record):
    """The model selects authorized IDs and supplies a Python script."""

    file_ids: list[Identifier] = Field(description="Selected attachment IDs; [] for pure computation.")
    code: str = Field(min_length=1, description="Python code using files, read_table, print, output_dir.")
    artifact_ids: list[Identifier] = Field(default_factory=list, description="Authorized saved inputs; available as files[artifact_id].")

    @field_validator("file_ids", "artifact_ids")
    @classmethod
    def unique_ids(cls, value: list[str]) -> list[str]:
        """Reject duplicate attachment IDs.
        Args: value: Validated ID list.
        """
        if len(value) != len(set(value)):
            raise ValueError("file_ids must be unique")
        return value


class PythonTool:
    """Bind execute_python to a host-selected task and A/B phase."""

    name = "execute_python"

    def __init__(self, store: RuntimeStore, task_id: str, variant: Variant = "A", tokenizer: Any = None,
                 selected_artifacts: Optional[list[ArtifactRef]] = None,
                 workspace_policy: Optional[WorkspacePolicy] = None) -> None:
        """Bind host context; the model cannot choose task identity or phase.
        Args: store: Open run; task_id: Current task; variant: A/B; tokenizer: Optional HF tokenizer.
        """
        if variant not in ("A", "B"):
            raise ValueError("variant must be A or B")
        self.store, self.task_id, self.variant, self.tokenizer = store, task_id, variant, tokenizer
        self.challenge_id = store.dataset.get_task(task_id).challenge_id
        self.selected_artifacts = selected_artifacts or []
        self.workspace_policy = workspace_policy or WorkspacePolicy()
        self.session = PythonSession()

    def close(self):
        """Release the task's persistent Python process and workspace."""
        self.session.close()

    def session_context(self, request: PythonArguments) -> dict:
        """Expose actual authorized paths and a compact inventory without serializing objects."""
        paths = self.session.prepare(self.input_files(request))
        return {"input_paths": paths, "session_generation": self.session.generation,
                "executed_cells": self.session.cells, "variables": self.session.variables,
                "state_alive": self.session.process is not None and self.session.process.poll() is None}

    def available_artifacts(self) -> dict[str, ArtifactRef]:
        """Allow current-phase outputs and only explicitly selected frozen A inputs."""
        state = self.store.load_state(self.task_id)
        records = [ref for ref in state.artifacts if ref.variant == self.variant]
        for reference in self.selected_artifacts:
            frozen = self.store.load_phase(reference.task_id, "A")
            if reference.variant != "A" or reference not in frozen.artifacts:
                raise PermissionError("selected input is outside frozen A artifacts")
            if reference.task_id not in state.dependency_ids and not (self.variant == "B" and reference.task_id == self.task_id):
                raise PermissionError("selected input is outside authorized sources")
            records.append(reference)
        return {artifact_id(ref): ref for ref in records}

    def input_files(self, request: PythonArguments) -> dict[str, Path]:
        """Resolve selected inputs and bind an unambiguous requested workspace version."""
        files = {key: self.store.dataset.resolve_asset(self.task_id, key) for key in request.file_ids}
        available = self.available_artifacts()
        for key in request.artifact_ids:
            if key not in available:
                raise PermissionError(f"artifact_id was not authorized: {key}")
            files[key] = self.store.artifact_path(available[key])
        source = self.workspace_policy.source_task_id
        if source:
            versions = [ref for ref in self.selected_artifacts if ref.task_id == source and
                        Path(ref.path).suffix.lower() in (".xlsx", ".xlsm", ".xlsb", ".csv")]
            workbook_ids = [asset.file_id for asset in self.store.dataset.list_assets(self.task_id)
                            if asset.reference.role in ("workbook", "table")]
            requested = set(workbook_ids).intersection(request.file_ids)
            if requested:
                if len(workbook_ids) != 1 or len(versions) != 1:
                    raise PermissionError("workspace version is ambiguous; use its explicit artifact_id")
                files[workbook_ids[0]] = self.store.artifact_path(versions[0])
        return files

    def definition(self) -> dict:
        """Return the function definition for the future model adapter.
        Args: None.
        """
        return {"type": "function", "function": {
            "name": self.name, "description": "Compute using selected task attachments and Python.",
            "parameters": PythonArguments.model_json_schema(),
        }}

    def system_prompt(self) -> str:
        """Build task-specific tool instructions without scoring labels.
        Args: None.
        """
        return python_prompt(self.store.dataset.list_assets(self.task_id))

    def _clip(self, text: str) -> tuple[str, bool]:
        """Limit feedback by tokenizer, or by UTF-8 bytes until model integration.
        Args: text: Complete observation text.
        """
        limit = self.store.config.tool_return_tokens
        if self.tokenizer is not None:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            return self.tokenizer.decode(ids[:limit]), len(ids) > limit
        encoded = text.encode("utf-8")
        return encoded[:limit].decode("utf-8", errors="ignore"), len(encoded) > limit

    def run(self, arguments: dict, remaining_seconds: Optional[float] = None, *, tool_name: str | None = None) -> Observation:
        """Validate, execute, and persist one call; model errors become observations.
        Args: arguments: file_ids/code; remaining_seconds: Host-measured phase time left, if known.
        """
        call = ToolExecution(self, tool_name or self.name, arguments)
        denied = call.begin(remaining_seconds)
        if denied:
            return denied
        started, artifacts = call.started, call.artifacts
        returncode, outputs = None, []
        save = call.save

        try:
            request = PythonArguments.model_validate(arguments)
            files = self.input_files(request)
        except (ValidationError, PermissionError, FileNotFoundError) as error:
            status, exception_type = "denied", type(error).__name__
            summary = str(error)
            save("request.json", json.dumps(arguments, ensure_ascii=False))
            save("error.txt", summary)
        else:
            save("code.py", request.code)
            save("inputs.json", json.dumps({key: str(path) for key, path in files.items()}, ensure_ascii=False))
            with (nullcontext(str(self.session.workspace)) if tool_name is None else
                  tempfile.TemporaryDirectory(prefix="agent-python-")) as directory:
                workspace = Path(directory)
                try:
                    timeout = call.remaining(self.store.config.tool_timeout_seconds)
                    previous_outputs = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in (workspace / "outputs").glob("*") if p.is_file()}
                    result = (self.session.run(files, request.code, max(0.0, timeout), seed=self.store.config.seed)
                              if tool_name is None else
                              run_python(workspace, files, request.code, max(0.0, timeout), seed=self.store.config.seed))
                    status, exception_type = result.status, result.exception_type
                    returncode = result.returncode
                    pieces = [f"status={status}; exception={exception_type or 'none'}"]
                    session_summary = None
                    if tool_name is None:
                        session_summary = "python_session=" + json.dumps({
                            "generation": self.session.generation, "cells": self.session.cells,
                            "variables": self.session.variables,
                            "state_lost": exception_type in ("TimeoutExpired", "WorkerExit"),
                            "failed_cell_may_have_changed_variables": status == "error",
                            "next_step": "Continue from these variables; avoid repeating completed inspection."
                        }, ensure_ascii=False)
                    names = ("stderr.txt", "stdout.txt") if status != "ok" else ("stdout.txt", "stderr.txt")
                    for name in (*names, "reads.jsonl"):
                        path = workspace / name
                        save(name, path)
                        if name != "reads.jsonl":
                            # Read a bounded preview; stream the full log into the artifact store.
                            with path.open("rb") as stream:
                                preview = stream.read(self.store.config.tool_return_tokens * 8 + 1)
                            pieces.append(name + ":\n" + preview.decode("utf-8", errors="replace"))
                            if path.stat().st_size > len(preview):
                                pieces.append("[remaining output saved in artifact]")
                    output_dir = workspace / "outputs"
                    if output_dir.resolve() != output_dir:
                        raise PermissionError("output_dir must not be redirected")
                    for path in sorted(output_dir.iterdir()):
                        if path.is_file() and not path.is_symlink() and (
                                previous_outputs.get(path.name) != hashlib.sha256(path.read_bytes()).hexdigest()):
                            suffix = path.suffix if re.fullmatch(r"\.[A-Za-z0-9]+", path.suffix) else ""
                            reference = save(f"output-{len(outputs) + 1:03d}{suffix}", path)
                            outputs.append({"name": path.name, "path": reference.path})
                    pieces.append("output_files=" + json.dumps([
                        {**item, "artifact_id": artifact_id(next(ref for ref in artifacts if ref.path == item["path"]))}
                        for item in outputs
                    ], ensure_ascii=False))
                    computed = workspace / "computed_result.json"
                    if computed.exists() and tool_name is None:
                        checked_result = ComputedResult.model_validate_json(computed.read_text())
                        reference = save("computed_result.json", checked_result.model_dump_json())
                        pieces.append("computed_result=" + json.dumps({
                            "execution_status": status, "partial": status != "ok",
                            **checked_result.model_dump(), "artifact_id": artifact_id(reference),
                            "evidence_ref": reference.path}, ensure_ascii=False))
                    if status == "ok" and not outputs and (workspace / "stdout.txt").stat().st_size == 0:
                        pieces.append("No printed values or output files were returned. This is a Python script, "
                                      "not a notebook: a bare final expression is not displayed. "
                                      "Run computation with print(result) or save the result in output_dir. "
                                      "Execution metadata does not contain the computed answer.")
                    if session_summary is not None:
                        pieces.append(session_summary)
                    summary = "\n".join(pieces)
                except (OSError, ValueError) as error:
                    status, exception_type, summary = "error", type(error).__name__, str(error)
                    save("host_error.txt", summary)
        duration = monotonic() - started
        metadata = save("result.json", json.dumps({
            "status": status, "exception_type": exception_type, "duration_seconds": duration,
            "returncode": returncode,
            "output_files": outputs,
            "artifacts": [item.path for item in artifacts],
        }, ensure_ascii=False))
        return call.finish(status, summary, metadata, exception_type,
                           truncated="[remaining output saved in artifact]" in summary)



class ToolExecution:
    """Share call budgeting, event provenance and artifact publication across adapters."""

    def __init__(self, python: PythonTool, name: str, arguments: dict, **details) -> None:
        """Bind an authorized task context and optional MCP/source event metadata."""
        self.python, self.name, self.arguments, self.details = python, name, arguments, details
        self.stage = "react" if python.variant == "A" else "reflection"
        self.artifacts = []
        self.available = 0.0
        self.event = None
        self.started = None

    def begin(self, remaining_seconds: Optional[float]) -> Optional[Observation]:
        """Count one attempt before execution; exhausted budgets return a denied observation."""
        python = self.python
        state = python.store.load_state(python.task_id)
        budget = getattr(python.store.config, f"budget_{python.variant}")
        counter = state.counters.setdefault(python.variant, UsageCounters())
        self.available = budget.max_seconds - counter.tool_duration_seconds
        if remaining_seconds is not None:
            if not math.isfinite(remaining_seconds):
                raise ValueError("remaining_seconds must be finite")
            self.available = min(self.available, remaining_seconds)
        exhausted = counter.tool_calls >= budget.max_tool_calls or self.available <= 0
        self.event = python.store.append_event(
            python.challenge_id, self.stage, "denied" if exhausted else "started",
            task_id=python.task_id, variant=python.variant, tool_name=self.name,
            arguments=self.arguments, **self.details,
            result_summary="Tool budget exhausted." if exhausted else None,
        )
        if exhausted:
            observation = Observation(event_id=self.event.event_id, status="denied",
                                      result_summary="Tool budget exhausted.")
            state.observations.append(observation)
            python.store.save_state(state)
            return observation
        counter.tool_calls += 1
        state.stage = self.stage
        python.store.save_state(state)
        self.started = monotonic()
        return None

    def save(self, name: str, content) -> ArtifactRef:
        """Register a complete event-owned artifact before exposing its ID."""
        reference = self.python.store.write_artifact(self.event.event_id, name, content)
        self.artifacts.append(reference)
        return reference

    def remaining(self, limit: float) -> float:
        """Include input preparation and service initialization in the same call timeout."""
        return min(limit, self.available) - (monotonic() - self.started)

    def finish(self, status: str, summary: str, result_ref: ArtifactRef,
               exception_type: Optional[str] = None, truncated: bool = False,
               image_ref: Optional[ArtifactRef] = None) -> Observation:
        """Append the completion event and publish counters, artifacts and short feedback."""
        python = self.python
        duration = monotonic() - self.started
        preview, clipped = python._clip(summary)
        completed = python.store.append_event(
            python.challenge_id, self.stage, status, task_id=python.task_id, variant=python.variant,
            parent_event_id=self.event.event_id, tool_name=self.name, **self.details,
            exception_type=exception_type, duration_seconds=duration,
            result_summary=preview, result_ref=result_ref.path,
        )
        observation = Observation(event_id=completed.event_id, status=status, result_summary=preview,
                                  result_ref=result_ref, truncated=truncated or clipped, image_ref=image_ref)
        state = python.store.load_state(python.task_id)
        state.counters[python.variant].tool_duration_seconds += duration
        state.artifacts.extend(self.artifacts)
        state.observations.append(observation)
        python.store.save_state(state)
        return observation


class FileArguments(Record):
    """Read an authorized original image or a registered artifact by opaque ID."""

    file_id: Optional[Identifier] = Field(default=None, description="Original image attachment ID; choose only one ID.")
    artifact_id: Optional[Identifier] = Field(default=None, description="Saved image ID; do not also provide file_id.")
    offset: int = Field(default=0, ge=0, description="UTF-8 byte offset for a text artifact.")


class ArtifactArguments(Record):
    """Read saved text by artifact ID; original attachment IDs are not accepted."""

    artifact_id: Identifier
    offset: int = Field(default=0, ge=0, description="UTF-8 byte offset.")


class TableInspectArguments(Record):
    """Select one table source and optionally one worksheet for bounded layout inspection."""

    file_id: Optional[Identifier] = None
    artifact_id: Optional[Identifier] = None
    sheet: Optional[str] = Field(default=None, min_length=1, max_length=31)

    @model_validator(mode="after")
    def check_source(self):
        """Require exactly one source without accepting model-generated filesystem paths."""
        if (self.file_id is None) == (self.artifact_id is None):
            raise ValueError("provide exactly one file_id or artifact_id")
        return self


class TableInspectTool:
    """Inspect sheet names and bounded cell samples in a single authorized subprocess call."""

    name = "inspect_table"

    def __init__(self, python: PythonTool) -> None:
        """Reuse task authorization, copying, timeouts and artifact registration."""
        self.python = python

    def definition(self) -> dict:
        """Describe combined layout/data inspection instead of repeated metadata-only calls."""
        return {"type": "function", "function": {"name": self.name,
                "description": "Inspect sheets and cell samples once; use Python for complete calculations.",
                "parameters": TableInspectArguments.model_json_schema()}}

    def run(self, arguments: dict, remaining_seconds: Optional[float] = None) -> Observation:
        """Return row numbers and samples, charging inspection rather than computation evidence."""
        try:
            request = TableInspectArguments.model_validate(arguments)
        except ValueError as error:
            call = ToolExecution(self.python, self.name, arguments)
            denied = call.begin(remaining_seconds)
            if denied:
                return denied
            return call.finish("denied", str(error), call.save("error.txt", str(error)), type(error).__name__)
        key = request.file_id or request.artifact_id
        code = f"key = {key!r}\nrequested_sheet = {request.sheet!r}\n" + '''import json
from contextlib import nullcontext
import pandas as pd
path = files[key]
engines = {".xlsx": "openpyxl", ".xlsm": "openpyxl", ".xlsb": "calamine"}
if path.suffix.lower() == ".csv":
    sheets = []
    sheet = 0
elif path.suffix.lower() in engines:
    with pd.ExcelFile(path, engine=engines[path.suffix.lower()]) as workbook:
        sheets = workbook.sheet_names
    sheet = requested_sheet or sheets[0]
else:
    raise ValueError("inspect_table requires CSV/XLSX/XLSM/XLSB")
frame = read_table(key, sheet=sheet, header=None, nrows=100)
column_count = len(frame.columns)
frame = frame.iloc[:, :16]
frame = frame.astype(object).where(pd.notna(frame), None)
rows = [{"row": int(index) + 1, "pandas_index": int(index), "cells": row.tolist()} for index, row in frame.iterrows() if any(value is not None for value in row)]
headers = []
for first, second in zip(rows, rows[1:]):
    text_count = sum(isinstance(value, str) for value in first["cells"])
    typed_count = sum(value is not None and not isinstance(value, str) for value in second["cells"])
    if second["row"] == first["row"] + 1 and text_count >= 2 and typed_count >= 2:
        headers.append({**first, "header_row_excel": first["row"], "header_index_pandas": first["pandas_index"],
                        "data_start_row_excel": second["row"], "data_start_index_pandas": second["pandas_index"]})
headers = headers[:4]
regions = []
for row in rows:
    positions = [i + 1 for i, value in enumerate(row["cells"]) if value is not None]
    if regions and row["row"] == regions[-1]["end_row"] + 1 and positions == regions[-1]["columns"]:
        regions[-1]["end_row"] = row["row"]
    else:
        regions.append({"start_row": row["row"], "end_row": row["row"], "columns": positions})
regions = sorted(regions, key=lambda block: block["end_row"] - block["start_row"], reverse=True)[:8]
for region in regions:
    region.update(pandas_start_index=region["start_row"]-1, pandas_end_index_inclusive=region["end_row"]-1,
                  iloc_stop_exclusive=region["end_row"], row_count=region["end_row"]-region["start_row"]+1)
selected = {row["row"]: row for row in rows[:3] + rows[-3:]}
for header in headers:
    for row in rows:
        if header["row"] <= row["row"] <= header["row"] + 2:
            selected[row["row"]] = row
sample = rows if len(rows) <= 12 else [selected[index] for index in sorted(selected)]
print(json.dumps({"source_id": key, "sheets": sheets, "sheet": sheet,
    "candidate_headers": headers, "data_regions": regions,
    "column_positions": list(range(1, len(frame.columns) + 1)), "column_count": column_count, "sample_rows": sample,
    "sample_is_partial": len(rows) > 12 or len(frame) >= 100 or column_count > 16,
    "coordinate_system": {"excel": "1-based rows and columns", "pandas": "0-based indices with header=None",
                          "iloc": "stop is exclusive; Excel row r equals pandas index r-1"},
    "note": "Layout sample only. Read full rows, check first/last dates and row count; missing assumptions may be supplied in Introduction."}, default=str))
'''
        return self.python.run({"file_ids": [request.file_id] if request.file_id else [],
                                "artifact_ids": [request.artifact_id] if request.artifact_id else [],
                                "code": code}, remaining_seconds, tool_name=self.name)


class FileTool:
    """Bounded artifact reading or image viewing using the same phase tool budget."""

    def __init__(self, python: PythonTool, image: bool = False) -> None:
        """Share task identity, authorization and feedback clipping with PythonTool."""
        self.python, self.image = python, image
        self.name = "view_image" if image else "read_artifact"

    def definition(self) -> dict:
        """Return the file-ID schema; filesystem paths are never tool arguments."""
        return {"type": "function", "function": {"name": self.name,
                "description": "View an authorized image." if self.image else "Read a bounded part of a saved text artifact.",
                "parameters": (FileArguments if self.image else ArtifactArguments).model_json_schema()}}

    def run(self, arguments: dict, remaining_seconds: Optional[float] = None) -> Observation:
        """Record one read attempt, returning errors as observations and images by reference."""
        python = self.python
        store, task_id, variant = python.store, python.task_id, python.variant
        call = ToolExecution(python, self.name, arguments)
        denied = call.begin(remaining_seconds)
        if denied:
            return denied
        image_ref, result_ref, failure, truncated = None, None, None, False
        try:
            checked = (FileArguments if self.image else ArtifactArguments).model_validate(arguments)
            request = FileArguments.model_validate(checked.model_dump())
            if (request.file_id is None) == (request.artifact_id is None):
                raise ValueError("provide exactly one file_id or artifact_id")
            if request.artifact_id:
                authorized = python.available_artifacts()
                if request.artifact_id not in authorized:
                    raise PermissionError("artifact_id was not authorized")
                source = store.artifact_path(authorized[request.artifact_id])
            elif self.image:
                source = store.dataset.resolve_asset(task_id, request.file_id)
            else:
                raise PermissionError("read_artifact requires a saved artifact_id")
            if self.image:
                if source.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
                    raise ValueError("selected file is not a supported image")
                image_ref = call.save("image" + source.suffix.lower(), source)
                summary = "Authorized image attached to the next model input."
            else:
                if source.suffix.lower() not in (".txt", ".log", ".json", ".jsonl", ".csv", ".md", ".py", ".yaml", ".yml"):
                    raise ValueError("read_artifact accepts text only; images are already attached or use view_image")
                with source.open("rb") as stream:
                    prefix = stream.read(512)
                    if b"\x00" in prefix:
                        raise ValueError("binary data cannot be read as a text artifact")
                    stream.seek(request.offset)
                    content = stream.read(store.config.tool_return_tokens * 8 + 1)
                # A byte window can end in the middle of a UTF-8 character.
                import codecs
                preview, clipped = python._clip(codecs.getincrementaldecoder("utf-8")().decode(content, final=False))
                truncated = clipped or request.offset + len(content) < source.stat().st_size
                summary = f"byte_offset={request.offset}; bytes_read={len(content)}\n" + preview
            result_ref = call.save("read.json", json.dumps({
                "arguments": arguments, "source": str(source), "summary": summary,
                "truncated": truncated, "image_ref": image_ref.model_dump() if image_ref else None,
            }, ensure_ascii=False))
            status = "ok"
        except (ValueError, PermissionError, OSError) as error:
            status, summary, failure = "error", str(error), type(error).__name__
            result_ref = call.save("error.txt", summary)
        if call.remaining(python.store.config.tool_timeout_seconds) <= 0:
            status, summary, image_ref, failure = "timeout", "Read exceeded call deadline.", None, "TimeoutError"
        return call.finish(status, summary, result_ref, failure, truncated, image_ref)



class ToolRegistry:
    """Dispatch registered tool names; task/phase scope is bound by the host."""

    def __init__(self, python: PythonTool) -> None:
        """Register local tools and configured Excel tools without starting a service."""
        self.python = python
        self._resources = []
        self._tools = {python.name: python, "read_artifact": FileTool(python),
                       "view_image": FileTool(python, image=True)}
        assets = python.store.dataset.list_assets(python.task_id)
        if any(Path(asset.reference.path).suffix.lower() in (".csv", ".xlsx", ".xlsm", ".xlsb") for asset in assets) or any(
                Path(ref.path).suffix.lower() in (".csv", ".xlsx", ".xlsm", ".xlsb") for ref in python.selected_artifacts):
            self._tools["inspect_table"] = TableInspectTool(python)
        has_excel = any(Path(asset.reference.path).suffix.lower() in (".xlsx", ".xlsm") for asset in assets)
        has_excel = has_excel or any(Path(ref.path).suffix.lower() in (".xlsx", ".xlsm")
                                     for ref in python.selected_artifacts)
        if python.store.config.mcp.excel.enabled and has_excel:
            from .mcp_tools import ExcelMCPTools

            excel = ExcelMCPTools(python)
            self._resources.append(excel)
            self._tools.update({tool.name: tool for tool in excel.tools()})

    def system_prompt(self) -> str:
        """Describe enabled services without mixing the Python code contract into action selection."""
        return "\n".join(
            resource.system_prompt() for resource in self._resources
        )

    def close(self) -> None:
        """Release task-owned MCP connections and temporary directories."""
        try:
            for resource in self._resources:
                resource.close()
        finally:
            self.python.close()

    def definitions(self) -> list[dict]:
        """Return model-facing schemas for available tools.
        Args: None.
        """
        return [tool.definition() for tool in self._tools.values()]

    def call(self, name: str, arguments: dict, remaining_seconds: Optional[float] = None) -> Observation:
        """Dispatch an available tool; reject unknown names before execution.
        Args: name: Requested tool; arguments: JSON arguments; remaining_seconds: Host phase budget.
        """
        if name not in self._tools:
            raise ValueError(f"tool is not available: {name}")
        return self._tools[name].run(arguments, remaining_seconds=remaining_seconds)
