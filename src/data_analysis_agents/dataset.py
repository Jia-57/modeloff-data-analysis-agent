"""Read-only dataset access for challenge planning and individual ReAct tasks.

This module checks host-mediated reads. Arbitrary Python can bypass these
checks unless a separate operating-system sandbox is supplied.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Collection, Union

from .config import ResolvedPaths, resolve_paths
from .types import (
    AssetInfo, ChallengeContext, ChallengeInput, FileRef, RunConfig, Task, TaskInput,
)


def _check_file_declarations(task: Task) -> None:
    """Check file names and prevent attachments from aliasing task material.
    Args: task: Validated manifest row.
    """
    if Path(task.question_file.path).name.casefold() != f"{task.question_id}.txt".casefold():
        raise ValueError("question file name must match question_id")
    if Path(task.introduction_file.path).name.casefold() != "introduction.txt":
        raise ValueError("introduction file must be introduction.txt")
    paths = [item.path for item in task.allowed_assets]
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate attachment paths")
    for reference in task.allowed_assets:
        name = Path(reference.path).name
        if (
            name.startswith(".")
            or name.casefold() == "introduction.txt"
            or re.fullmatch(r"question\d+\.txt", name, re.IGNORECASE)
            or Path(name).suffix.lower() == ".ipynb"
        ):
            raise ValueError(f"excluded task/solution material in attachments: {name}")


def _load_manifest(path: Path) -> dict[str, Task]:
    """Load ordered task records, reporting the source line for malformed rows.
    Args: path: Manifest JSONL path.
    """
    tasks: dict[str, Task] = {}
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                task = Task.model_validate(json.loads(line))
                _check_file_declarations(task)
                if task.task_id in tasks:
                    raise ValueError(f"duplicate task_id: {task.task_id}")
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            tasks[task.task_id] = task
    if not tasks:
        raise ValueError(f"{path}: manifest is empty")
    return tasks


class Dataset:
    """Index manifest records; only declared inputs are available for reading."""

    def __init__(self, paths: ResolvedPaths) -> None:
        """Load the manifest and build consistent challenge indexes.
        Args: paths: Paths produced by resolve_paths().
        """
        self._paths = paths
        self._tasks = _load_manifest(paths.manifest)
        self._contexts: dict[str, ChallengeContext] = {}
        self._assets: dict[str, dict[str, FileRef]] = {}
        for task in self._tasks.values():
            assets = sorted(task.allowed_assets, key=lambda item: item.path)
            context = self._contexts.get(task.challenge_id)
            if context is None:
                context = ChallengeContext(
                    challenge_id=task.challenge_id, introduction_file=task.introduction_file,
                    task_ids=[], allowed_assets=assets,
                )
                self._contexts[task.challenge_id] = context
                self._assets[task.challenge_id] = {
                    f"asset-{i:03d}": item for i, item in enumerate(assets, start=1)
                }
            elif context.introduction_file != task.introduction_file or context.allowed_assets != assets:
                raise ValueError(f"{task.task_id}: inconsistent shared introduction or attachments")
            context.task_ids.append(task.task_id)

    @classmethod
    def from_config(cls, config: RunConfig, project_root: Union[str, Path]) -> Dataset:
        """Build the dataset using validated solver configuration.
        Args: config: Run settings; project_root: Project directory.
        """
        return cls(resolve_paths(config, project_root))

    @property
    def task_ids(self) -> tuple[str, ...]:
        """Return task IDs in manifest order; no explicit arguments."""
        return tuple(self._tasks)

    @property
    def challenge_ids(self) -> tuple[str, ...]:
        """Return challenge IDs in first-occurrence order; no explicit arguments."""
        return tuple(self._contexts)

    def get_task(self, task_id: str) -> Task:
        """Return a detached task record, protecting the internal whitelist.
        Args: task_id: Manifest task ID.
        """
        return self._tasks[task_id].model_copy(deep=True)

    def get_context(self, challenge_id: str) -> ChallengeContext:
        """Return a detached shared context for trusted planning/scheduling code.
        Args: challenge_id: Manifest challenge ID.
        """
        return self._contexts[challenge_id].model_copy(deep=True)

    def _resolve_input(self, task_id: str, reference: FileRef) -> Path:
        """Check the whitelist, exact location, and existence before every read.
        Args: task_id: Owning task; reference: Declared input file.
        """
        task = self._tasks[task_id]
        permitted = [task.question_file, task.introduction_file, *task.allowed_assets]
        if reference not in permitted:
            raise PermissionError(f"{task_id}: file is not in the input whitelist")
        declared = self._paths.project_root / reference.path
        try:
            resolved = declared.resolve()
        except (OSError, RuntimeError) as error:
            raise PermissionError(f"{task_id}: cannot safely resolve {reference.path}") from error
        challenge_root = self._paths.raw_data / task.challenge_id
        if challenge_root not in resolved.parents or resolved != declared:
            raise PermissionError(f"{task_id}: redirected or out-of-scope input: {reference.path}")
        if not resolved.is_file():
            raise FileNotFoundError(f"{task_id}: missing input file: {reference.path}")
        return resolved

    def _read_text(self, task_id: str, reference: FileRef) -> str:
        """Read UTF-8 text without changing line endings or writing source files.
        Args: task_id: Owning task; reference: Question/introduction file.
        """
        path = self._resolve_input(task_id, reference)
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                return stream.read()
        except UnicodeError as error:
            raise ValueError(f"{task_id}: invalid UTF-8 in {reference.path}") from error

    def resolve_asset(self, task_id: str, file_id: str) -> Path:
        """Map an allowed file ID to a host path for subsequent tool adapters.
        Args: task_id: Current task; file_id: Attachment ID, not an arbitrary path.
        """
        task = self._tasks[task_id]
        reference = self._assets[task.challenge_id].get(file_id)
        if reference is None:
            raise PermissionError(f"{task_id}: unknown or unauthorized file_id: {file_id}")
        return self._resolve_input(task_id, reference)

    def list_assets(self, task_id: str) -> list[AssetInfo]:
        """Inspect attachment sizes without interpreting workbook/image contents.
        Args: task_id: Task whose attachment whitelist is used.
        """
        task = self._tasks[task_id]
        return [
            AssetInfo(
                file_id=file_id, reference=reference,
                size_bytes=self.resolve_asset(task_id, file_id).stat().st_size,
            )
            for file_id, reference in self._assets[task.challenge_id].items()
        ]

    def read_task(self, task_id: str) -> TaskInput:
        """Build the current-question input for ReAct, excluding other questions.
        Args: task_id: Current task ID.
        """
        task = self.get_task(task_id)
        return TaskInput(
            task=task,
            question=self._read_text(task_id, task.question_file),
            introduction=self._read_text(task_id, task.introduction_file),
            assets=self.list_assets(task_id),
        )

    def read_challenge(self, challenge_id: str) -> ChallengeInput:
        """Build the complete question input for challenge-level planning only.
        Args: challenge_id: Challenge to plan.
        """
        context = self.get_context(challenge_id)
        first_id = context.task_ids[0]
        questions = {
            task_id: self._read_text(task_id, self._tasks[task_id].question_file)
            for task_id in context.task_ids
        }
        return ChallengeInput(
            context=context,
            introduction=self._read_text(first_id, context.introduction_file),
            questions=questions,
            assets=self.list_assets(first_id),
        )

    def read_previous_question(
        self, task_id: str, previous_task_id: str, authorized_task_ids: Collection[str]
    ) -> str:
        """Read predecessor text using trusted Scheduler authorization.
        Args: task_id: Current task; previous_task_id: Source; authorized_task_ids: Host-approved IDs.
        """
        current = self._tasks[task_id]
        if isinstance(authorized_task_ids, (str, bytes)):
            raise ValueError("predecessor authorization must be a collection of task IDs")
        if previous_task_id == task_id or previous_task_id not in authorized_task_ids:
            raise PermissionError(f"{task_id}: predecessor read is not authorized")
        previous = self._tasks[previous_task_id]
        if previous.challenge_id != current.challenge_id:
            raise PermissionError(f"{task_id}: cross-challenge predecessor read is forbidden")
        return self._read_text(previous_task_id, previous.question_file)

    def validate_inputs(self) -> dict[str, list[str]]:
        """Report inaccessible inputs per task while retaining every manifest entry.
        Args: None. File contents and gold labels are not read.
        """
        issues: dict[str, list[str]] = {}
        for task_id, task in self._tasks.items():
            for reference in [task.question_file, task.introduction_file, *task.allowed_assets]:
                try:
                    self._resolve_input(task_id, reference)
                except OSError as error:
                    issues.setdefault(task_id, []).append(str(error))
        return issues
