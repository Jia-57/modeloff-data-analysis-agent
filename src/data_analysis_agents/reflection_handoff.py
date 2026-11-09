"""Bounded evidence extraction and retry guards for Reflection."""
from __future__ import annotations

import json
from pathlib import Path

from .schema import is_inspection_cell

INSPECTION_TOOLS = {"inspect_table", "excel_describe_workbook", "excel_describe_sheet",
                    "excel_read_range", "excel_find_cells"}
LAYOUT_TOOLS = {"inspect_table", "excel_describe_workbook", "excel_describe_sheet"}
REPEAT_INSPECTION = "This exact inspection already succeeded in an earlier attempt. Use its result in the recovery handoff."
CURRENT_INSPECTION = "This exact inspection already succeeded in this round. Use the previous observation."
KNOWN_LAYOUT = "Layout for this workbook version was already inspected. Use the saved sample and compute next."
COMPUTE_RESERVE = "Keep the last tool call for execute_python to calculate the missing value."
RECOVERY_GUARDS = {REPEAT_INSPECTION, CURRENT_INSPECTION, KNOWN_LAYOUT, COMPUTE_RESERVE}


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _answered(row: dict | None) -> bool:
    answer = (row or {}).get("answer") or {}
    return answer.get("status") == "answered" and (answer.get("answer") or {}).get("value") is not None


def _recovery_priority(baseline: "Baseline", task_id: str) -> int:
    """Finish existing calculations first, then partial work, then untouched failures."""
    for _, path in baseline.artifacts(task_id):
        if path.name == "computed_result.json":
            try:
                result = json.loads(path.read_text(encoding="utf-8"))
                if result.get("value") is not None:
                    return 0
            except (OSError, ValueError):
                pass
    events = baseline.events(task_id)
    if any(item.get("tool_name") == "execute_python" and item.get("status") == "ok"
           for item in events):
        return 1
    if any(item.get("tool_name") for item in events):
        return 2
    return 3


def _inspection_signatures(events: list[dict]) -> set[str]:
    """Identify successful inspections by exact tool arguments, including file version IDs."""
    starts = {item["event_id"]: item for item in events
              if item.get("status") == "started" and item.get("tool_name")}
    completed = [item for item in events if item.get("status") == "ok" and
                 item.get("parent_event_id") in starts and item.get("tool_name")]
    signatures = set()
    for item in completed[-3:]:
        start = starts.get(item["parent_event_id"])
        if not start:
            continue
        name, arguments = start["tool_name"], start.get("arguments") or {}
        if name in INSPECTION_TOOLS or (name == "execute_python" and
                                        is_inspection_cell(str(arguments.get("code") or ""))):
            signatures.add(json.dumps([name, arguments], sort_keys=True, ensure_ascii=False))
    return signatures


def _layout_sources(events: list[dict]) -> set[str]:
    """Record workbook versions whose layout was already inspected successfully."""
    starts = {item["event_id"]: item for item in events
              if item.get("status") == "started" and item.get("tool_name")}
    completed = [item for item in events if item.get("status") == "ok" and
                 item.get("parent_event_id") in starts and item.get("tool_name")]
    sources = set()
    for item in completed[-3:]:
        if starts[item["parent_event_id"]]["tool_name"] not in LAYOUT_TOOLS:
            continue
        arguments = starts[item["parent_event_id"]].get("arguments") or {}
        source = arguments.get("file_id") or arguments.get("artifact_id")
        if source:
            sources.add(str(source))
    return sources


def _attempt_handoff(events: list[dict], artifacts: list[tuple[str, Path]], *,
                     label: str, status: str, explanation: str, limit: int) -> list[str]:
    """Build a bounded, deterministic handoff from saved execution evidence."""
    starts = {item["event_id"]: item for item in events
              if item.get("status") == "started" and item.get("tool_name")}
    completed = [item for item in events if item.get("parent_event_id") in starts
                 and item.get("tool_name")]
    completed_event_ids = {item["event_id"] for item in completed}
    successful = [item for item in completed if item.get("status") == "ok"]
    completed_ids = {item["parent_event_id"] for item in successful}
    lines = [f"{label} terminal status: {status}"]
    if explanation:
        lines.append(f"{label} explanation: {explanation[:450]}")
    results = []
    outputs = []
    for relative, path in artifacts:
        if path.name == "computed_result.json" and path.parent.name in completed_ids:
            try:
                result = json.loads(path.read_text(encoding="utf-8"))
                results.append(f"{label} computed result ({relative}): " +
                               json.dumps(result, ensure_ascii=False, allow_nan=False)[:800])
            except (OSError, UnicodeError, ValueError):
                continue
        elif path.name.startswith("output-") and path.parent.name in completed_ids:
            outputs.append(relative)
    lines.extend(results[-2:])
    if outputs:
        lines.append(f"{label} saved output files: " + ", ".join(outputs[-3:]))
    for item in successful[-3:]:
        start = starts[item["parent_event_id"]]
        summary = str(item.get("result_summary") or "")[:700]
        lines.append(f"{label} completed {start['tool_name']}: {summary}")
    if not successful:
        lines.append(f"{label} has no completed tool result to reuse.")
    for item in [row for row in completed if row.get("status") in ("error", "denied", "timeout")][-2:]:
        start = starts[item["parent_event_id"]]
        lines.append(f"{label} failed {start['tool_name']} ({item['status']}): " +
                     str(item.get("result_summary") or "")[:500])
    for item in [row for row in events if row.get("status") in ("error", "denied", "timeout")
                 and row.get("event_id") not in completed_event_ids and row.get("result_summary")][-2:]:
        lines.append(f"{label} other failure ({item['status']}): " +
                     str(item["result_summary"])[:500])
    lines.append("Python starts a fresh session in this round; use recorded values or authorized files, not old variables.")
    bounded, remaining = [], limit
    for line in lines:
        if remaining <= 0:
            break
        part = line[:remaining]
        bounded.append(part)
        remaining -= len(part)
    return bounded


