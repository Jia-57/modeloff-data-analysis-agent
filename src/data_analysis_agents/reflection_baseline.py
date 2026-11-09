"""Read-only access to a frozen first-round run for Reflection."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from .dataset import Dataset
from .types import Answer, Plan, validate_relative_path
from .reflection_handoff import _read_jsonl, _attempt_handoff, _inspection_signatures, _layout_sources

class Baseline:
    """Read-only first-round evidence, with no labels, answer rules or scoring files."""

    def __init__(self, root: Path, dataset: Dataset):
        self.root = root.resolve(strict=True)
        self.db = sqlite3.connect((self.root / "runtime.sqlite").as_uri() + "?mode=ro", uri=True)
        self.db.row_factory = sqlite3.Row
        manifest = {row["task_id"]: row for row in _read_jsonl(self.root / "manifest.jsonl")}
        if any(key not in manifest or manifest[key] != dataset.get_task(key).model_dump()
               for key in dataset.task_ids):
            raise ValueError("baseline manifest differs from active inputs; recovery refused")
        self.rows = {row["task_id"]: row for row in _read_jsonl(self.root / "answers_A.jsonl")}
        if any(key not in self.rows for key in dataset.task_ids):
            raise ValueError("baseline is missing active answer rows")
        self.manifest_hash = hashlib.sha256((self.root / "manifest.jsonl").read_bytes()).hexdigest()
        self.answer_hash = hashlib.sha256((self.root / "answers_A.jsonl").read_bytes()).hexdigest()
        self.db_hash = hashlib.sha256((self.root / "runtime.sqlite").read_bytes()).hexdigest()

    def close(self):
        self.db.close()

    def state(self, task_id: str) -> dict:
        row = self.db.execute("SELECT payload FROM states WHERE state_id=?", (task_id,)).fetchone()
        return json.loads(row["payload"]) if row else {}

    def plan(self, task_id: str, dataset: Dataset) -> Plan:
        saved = self.state(task_id).get("plan")
        if saved:
            return Plan.model_validate(saved)
        question = dataset.read_task(task_id).question
        return Plan(task_id=task_id, goal=question[:400],
                    steps=["Read the original question and authorized material", "Compute and check the requested answer"],
                    answer_format="as requested", data_sources=["question", "introduction"])

    def answer(self, task_id: str) -> Answer | None:
        raw = self.rows[task_id].get("answer")
        return Answer.model_validate(raw) if raw else None

    def artifacts(self, task_id: str) -> list[tuple[str, Path]]:
        refs = self.state(task_id).get("artifacts") or []
        completed = {row["parent_event_id"] for row in self.db.execute(
            "SELECT parent_event_id FROM events WHERE task_id=? AND variant='A' "
            "AND status='ok' AND parent_event_id IS NOT NULL", (task_id,))}
        found = []
        for ref in refs:
            if ref.get("variant") != "A":
                continue
            if ref.get("event_id") not in completed:
                continue
            relative = validate_relative_path(ref["path"])
            path = (self.root / relative).resolve()
            if self.root not in path.parents or not path.is_file():
                continue
            found.append((relative, path))
        return found

    def events(self, task_id: str) -> list[dict]:
        rows = self.db.execute(
            "SELECT payload FROM events WHERE task_id=? AND variant='A' ORDER BY seq", (task_id,)
        )
        return [json.loads(row["payload"]) for row in rows]

    def evidence(self, task_id: str, limit: int) -> list[str]:
        saved = self.state(task_id)
        answer = saved.get("answer_A") or self.rows[task_id].get("answer") or {}
        return _attempt_handoff(
            self.events(task_id), self.artifacts(task_id), label="first-round",
            status=str(self.rows[task_id].get("terminal_status", "not_run")),
            explanation=str(answer.get("brief_explanation") or ""), limit=limit)

    def inspections(self, task_id: str) -> set[str]:
        return _inspection_signatures(self.events(task_id))

    def layout_sources(self, task_id: str) -> set[str]:
        return _layout_sources(self.events(task_id))

    def prior_signatures(self, task_id: str) -> tuple[set[str], set[str]]:
        codes, observations = set(), set()
        for row in self.db.execute("SELECT payload FROM events WHERE task_id=? AND variant='A'", (task_id,)):
            event = json.loads(row["payload"])
            if event.get("tool_name") == "execute_python" and event.get("status") == "started":
                code = (event.get("arguments") or {}).get("code")
                if code:
                    codes.add(code.strip())
            if event.get("tool_name") and event.get("status") in ("ok", "error", "timeout"):
                observations.add(str(event.get("result_summary") or "").strip())
        return codes, observations

    def failed_codes(self, task_id: str) -> set[str]:
        rows = [json.loads(row["payload"]) for row in self.db.execute(
            "SELECT payload FROM events WHERE task_id=? AND variant='A'", (task_id,))]
        failed = {row["parent_event_id"] for row in rows
                  if row.get("tool_name") == "execute_python" and
                  row.get("status") in ("error", "denied", "timeout") and row.get("parent_event_id")}
        return {str((row.get("arguments") or {}).get("code")).strip() for row in rows
                if row.get("event_id") in failed and (row.get("arguments") or {}).get("code")}


