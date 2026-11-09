"""Host-only SQLite state and memory, with immutable file artifacts."""

from __future__ import annotations

import fcntl
import json
import math
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Optional, Union
from uuid import uuid4

from .config import ResolvedPaths, resolve_paths
from .dataset import Dataset
from .storage import SQLiteStore
from .types import (
    Answer, ArtifactRef, ChallengeContext, ChallengeState, Event, EventStatus, RunConfig,
    GenerationResult, SnapshotRef, Stage, TaskState, UsageCounters, Variant,
    validate_component, validate_relative_path,
)


State = Union[ChallengeState, TaskState]


def _json_bytes(value: object) -> bytes:
    """Encode finite JSON deterministically.
    Args: value: JSON-compatible data.
    """
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _atomic_write(path: Path, content: Union[bytes, Path], replace: bool = False) -> None:
    """Publish a complete file, keeping existing files unless replacement is explicit.
    Args: path: Checked destination; content: Bytes or source file; replace: Allow replacement.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            if isinstance(content, Path):
                with content.open("rb") as source:
                    shutil.copyfileobj(source, stream)
            else:
                stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class RuntimeStore:
    """One writer per run, with serialized thread access and immutable phase records."""

    def __init__(self, config: RunConfig, paths: ResolvedPaths, dataset: Dataset, *, create: bool = False) -> None:
        """Lock one run and open its database without loading historical records."""
        self.config, self.root, self.dataset = config, paths.run_dir, dataset
        self.project_root = paths.project_root
        self._mutex = RLock()
        self._writer = self._path(".writer.lock").open("a")
        self._db = None
        try:
            fcntl.flock(self._writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._db = SQLiteStore(self._path("runtime.sqlite"), create=create)
        except BaseException:
            self._writer.close()
            raise

    def _get_event(self, event_id: str) -> Optional[Event]:
        """Read one event by primary key rather than retaining every event."""
        row = self._db.one("SELECT payload FROM events WHERE event_id=?", (event_id,))
        return self._check_event(Event.model_validate(json.loads(row["payload"]))) if row else None

    def _phase_record(self, task_id: str, variant: Variant):
        """Fetch one immutable phase record from the indexed snapshot table."""
        return self._db.one("SELECT * FROM snapshots WHERE task_id=? AND variant=?", (task_id, variant))

    def _put_state(self, state: State) -> None:
        """Write a validated state inside the caller's transaction."""
        key = state.task_id if isinstance(state, TaskState) else state.challenge_id
        kind = "task" if isinstance(state, TaskState) else "challenge"
        self._db.execute(
            "INSERT INTO states VALUES (?,?,?,?,?) ON CONFLICT(state_id) DO UPDATE SET stage=excluded.stage,payload=excluded.payload",
            (key, state.challenge_id, kind,
             state.stage if isinstance(state, TaskState) else state.planning_status,
             state.model_dump_json()),
        )

    @classmethod
    def create(cls, config: RunConfig, project_root: Union[str, Path]) -> RuntimeStore:
        """Create a fresh run; existing directories are never replaced.
        Args: config: Run settings; project_root: Repository root.
        """
        config = RunConfig.model_validate(config)
        paths = resolve_paths(config, project_root)
        dataset = Dataset.from_config(config, project_root)
        paths.run_dir.mkdir(parents=True, exist_ok=False)
        store = cls(config, paths, dataset, create=True)
        try:
            manifest = b"".join(_json_bytes(dataset.get_task(key).model_dump()) for key in dataset.task_ids)
            _atomic_write(store._path("manifest.jsonl"), manifest)
            for challenge_id in dataset.challenge_ids:
                store.save_context(dataset.get_context(challenge_id))
                store.save_state(ChallengeState(
                    challenge_id=challenge_id, thread_id=f"{config.run_id}:{challenge_id}",
                    challenge_context_ref=f"contexts/{challenge_id}.json",
                ))
            for task_id in dataset.task_ids:
                task = dataset.get_task(task_id)
                store.save_state(TaskState(
                    task_id=task_id, challenge_id=task.challenge_id,
                    challenge_context_ref=f"contexts/{task.challenge_id}.json",
                    allowed_assets=task.allowed_assets,
                    counters={"A": UsageCounters(), "B": UsageCounters()},
                ))
            # The metadata transaction marks completed initialization.
            with store._db.transaction():
                store._db.execute("INSERT INTO runs VALUES (?,?)", ("config", config.model_dump_json()))
                store._db.execute("INSERT INTO runs VALUES (?,?)", ("manifest", manifest.decode("utf-8")))
        except BaseException:
            store.close()
            raise
        return store

    @classmethod
    def resume(cls, config: RunConfig, project_root: Union[str, Path]) -> RuntimeStore:
        """Open saved progress without restarting interrupted model or tool calls.
        Args: config: Unchanged run settings; project_root: Repository root.
        """
        config = RunConfig.model_validate(config)
        paths = resolve_paths(config, project_root)
        saved_paths = ResolvedPaths(paths.project_root, paths.run_dir / "manifest.jsonl", paths.raw_data, paths.run_dir)
        saved = Dataset(saved_paths)
        store = cls(config, paths, saved)
        try:
            row = store._db.one("SELECT payload FROM runs WHERE key='config'")
            if row is None:
                raise ValueError("runtime initialization or legacy migration is incomplete")
            previous = RunConfig.model_validate(json.loads(row["payload"]))
            manifest_row = store._db.one("SELECT payload FROM runs WHERE key='manifest'")
            if manifest_row is None or (paths.run_dir / "manifest.jsonl").read_text(encoding="utf-8") != manifest_row["payload"]:
                raise ValueError("saved manifest changed")
            if previous != config:
                # Recovery may resume in a later Slurm allocation with a new
                # loopback service port. All inference settings must still match.
                transport_only = (previous.model_copy(update={"model": previous.model.model_copy(
                    update={"api_base": config.model.api_base})}) == config)
                if not transport_only:
                    raise ValueError("run configuration changed; use a new run_id")
            current = Dataset.from_config(config, project_root)
            if current.task_ids != saved.task_ids or any(
                current.get_task(key) != saved.get_task(key) for key in saved.task_ids
            ):
                raise ValueError("manifest changed; use a new run_id")
            for row in store._db.execute("SELECT task_id,variant FROM snapshots"):
                store._seal_phase(row["task_id"], row["variant"])
        except BaseException:
            store.close()
            raise
        return store

    def close(self) -> None:
        """Release the run writer lock; no explicit arguments."""
        with self._mutex:
            if not self._writer.closed:
                if self._db is not None:
                    self._db.close()
                self._writer.close()

    @classmethod
    def migrate_legacy(cls, config: RunConfig, project_root: Union[str, Path]) -> RuntimeStore:
        """Import an existing JSON run atomically, preserving every original file."""
        config = RunConfig.model_validate(config)
        paths = resolve_paths(config, project_root)
        if (paths.run_dir / "runtime.sqlite").exists():
            raise FileExistsError("runtime.sqlite already exists; use resume")
        saved = RunConfig.model_validate(json.loads((paths.run_dir / "run_config.json").read_text()))
        if saved != config:
            raise ValueError("run configuration changed; use a new run_id")
        dataset = Dataset(ResolvedPaths(paths.project_root, paths.run_dir / "manifest.jsonl",
                                        paths.raw_data, paths.run_dir))
        current = Dataset.from_config(config, project_root)
        if current.task_ids != dataset.task_ids or any(current.get_task(key) != dataset.get_task(key) for key in dataset.task_ids):
            raise ValueError("manifest changed; use a new run_id")
        store = cls(config, paths, dataset, create=True)
        try:
            with store._db.transaction():
                store._db.execute("PRAGMA defer_foreign_keys=ON")
                for key in dataset.challenge_ids:
                    context = ChallengeContext.model_validate(store._read_json(f"contexts/{key}.json"))
                    store.save_context(context)
                    store._put_state(ChallengeState.model_validate(store._read_json(f"checkpoints/challenges/{key}.json")))
                for key in dataset.task_ids:
                    store._put_state(TaskState.model_validate(store._read_json(f"checkpoints/tasks/{key}.json")))
                for path in sorted(store.root.glob("events/**/*.jsonl")):
                    checked = store._path(path.relative_to(store.root).as_posix())
                    for number, line in enumerate(checked.open(encoding="utf-8"), 1):
                        if not line.endswith("\n"):
                            raise ValueError(f"incomplete event record: {path}:{number}")
                        event = store._check_event(Event.model_validate(json.loads(line)))
                        expected = (f"events/{event.task_id}/{event.variant}.jsonl" if event.task_id else
                                    f"events/{event.challenge_id}/{'scheduling' if event.stage == 'scheduling' else 'planning'}.jsonl")
                        if checked != store._path(expected):
                            raise ValueError("event identity does not match its legacy log")
                        store._db.execute(
                            "INSERT INTO events(event_id,challenge_id,task_id,variant,stage,status,timestamp,parent_event_id,payload) VALUES (?,?,?,?,?,?,?,?,?)",
                            (event.event_id, event.challenge_id, event.task_id, event.variant, event.stage,
                             event.status, event.timestamp, event.parent_event_id, event.model_dump_json()),
                        )
                        scope = f"{event.task_id}/{event.variant}" if event.task_id else f"{event.challenge_id}/planning"
                        folder = store._path(f"artifacts/{scope}/{event.event_id}")
                        if folder.exists():
                            for artifact in folder.iterdir():
                                relative = artifact.relative_to(store.root).as_posix()
                                if store._path(relative).is_file():
                                    store._db.execute("INSERT INTO artifacts VALUES (?,?,?,?,?)",
                                                      (relative, event.event_id, event.task_id, event.variant, artifact.stat().st_size))
                            request = folder / "request.json"
                            if event.stage == "planning" and event.status == "started" and request.exists():
                                store.begin_model_call(event.event_id, json.loads(request.read_text()))
                                response, failure = folder / "response.json", folder / "error.txt"
                                if response.exists():
                                    store.finish_model_call(event.event_id, result=GenerationResult.model_validate(json.loads(response.read_text())))
                                elif failure.exists():
                                    store.finish_model_call(event.event_id, error=failure.read_text())
                for task_id in dataset.task_ids:
                    for variant in ("A", "B"):
                        relative = f"snapshots/{task_id}/{variant}"
                        if not store._path(relative).exists():
                            continue
                        frozen = TaskState.model_validate(store._read_json(relative + "/state.json"))
                        reference = SnapshotRef.model_validate(store._read_json(relative + "/snapshot.json"))
                        answer = getattr(frozen, f"answer_{variant}")
                        if reference.task_id != task_id or reference.variant != variant or reference.path != relative or frozen.task_id != task_id or answer is None or variant not in frozen.terminal_status:
                            raise ValueError("invalid legacy frozen phase")
                        if Answer.model_validate(store._read_json(relative + "/answer.json")) != answer:
                            raise ValueError("legacy answer does not match its frozen state")
                        if variant == "B" and reference.source_snapshot_id != store.load_snapshot(task_id, "A").snapshot_id:
                            raise ValueError("B snapshot does not reference its frozen A")
                        store._db.execute("INSERT INTO snapshots VALUES (?,?,?,?,?,?)", (
                            reference.snapshot_id, task_id, variant, reference.source_snapshot_id,
                            reference.model_dump_json(), frozen.model_dump_json(),
                        ))
                        store._db.execute("INSERT INTO answers VALUES (?,?,?,?)",
                                          (task_id, variant, frozen.terminal_status[variant], answer.model_dump_json()))
                for key in (*dataset.challenge_ids, *dataset.task_ids):
                    store.load_state(key)
                for row in store._db.execute("SELECT task_id,variant FROM snapshots"):
                    phase = store.load_phase(row["task_id"], row["variant"])
                    for reference in phase.artifacts:
                        store._check_artifact(reference, phase.task_id)
                for row in store._db.execute("SELECT payload FROM events WHERE parent_event_id IS NOT NULL"):
                    event = Event.model_validate(json.loads(row["payload"]))
                    parent = store._get_event(event.parent_event_id)
                    if parent is None or parent.challenge_id != event.challenge_id or parent.task_id != event.task_id or (event.variant == "A" and parent.variant == "B"):
                        raise ValueError("legacy event parent is outside its authorized history")
                store._db.execute("INSERT INTO runs VALUES (?,?)", ("config", config.model_dump_json()))
                store._db.execute("INSERT INTO runs VALUES (?,?)", ("manifest", (paths.run_dir / "manifest.jsonl").read_text(encoding="utf-8")))
            return store
        except BaseException:
            # A failed import leaves legacy records intact and permits a clean retry.
            store._db.close()
            store._db = None
            store._path("runtime.sqlite").unlink(missing_ok=True)
            store.close()
            raise

    def __enter__(self) -> RuntimeStore:
        """Return the open store for a with block; no explicit arguments."""
        return self

    def __exit__(self, *exception: object) -> None:
        """Close the store after a with block.
        Args: exception: Context-manager exception details.
        """
        self.close()

    def _path(self, relative: str) -> Path:
        """Reject unsafe paths and all symlink redirections within the run.
        Args: relative: Run-relative path.
        """
        validate_relative_path(relative)
        path = self.root / relative
        if self.root.resolve() != self.root or path.resolve() != path:
            raise PermissionError(f"redirected run path: {relative}")
        return path

    def _read_json(self, relative: str) -> object:
        """Read a checked JSON file.
        Args: relative: Run-relative file path.
        """
        with self._path(relative).open(encoding="utf-8") as stream:
            return json.load(stream)

    def _check_open(self) -> None:
        """Reject access after the writer lock is released; no explicit arguments."""
        if self._writer.closed:
            raise ValueError("runtime store is closed")

    def _check_event(self, event: Event) -> Event:
        """Validate run membership and source identity.
        Args: event: Event to read or append.
        """
        event = Event.model_validate(event)
        if (event.experiment_id, event.run_id) != (self.config.experiment_id, self.config.run_id):
            raise ValueError("event belongs to another run")
        self.dataset.get_context(event.challenge_id)
        if event.task_id is not None:
            if self.dataset.get_task(event.task_id).challenge_id != event.challenge_id:
                raise ValueError("event task belongs to another challenge")
        return event

    def save_context(self, context: ChallengeContext) -> str:
        """Persist immutable manifest metadata; return its logical context reference."""
        with self._mutex:
            self._check_open()
            context = ChallengeContext.model_validate(context)
            expected = self.dataset.get_context(context.challenge_id)
            if context.model_dump(exclude={"workbook_metadata"}) != expected.model_dump(exclude={"workbook_metadata"}):
                raise ValueError("context does not match the run manifest")
            row = self._db.one("SELECT payload FROM contexts WHERE challenge_id=?", (context.challenge_id,))
            if row and ChallengeContext.model_validate(json.loads(row["payload"])) != context:
                raise ValueError("saved challenge context cannot be replaced")
            if not row:
                with self._db.transaction():
                    self._db.execute("INSERT INTO contexts VALUES (?,?)", (context.challenge_id, context.model_dump_json()))
            return f"contexts/{context.challenge_id}.json"

    def load_context(self, challenge_id: str) -> ChallengeContext:
        """Load and validate only the requested shared context."""
        with self._mutex:
            self._check_open()
            expected = self.dataset.get_context(challenge_id)
            row = self._db.one("SELECT payload FROM contexts WHERE challenge_id=?", (challenge_id,))
            if not row:
                raise FileNotFoundError(challenge_id)
            context = ChallengeContext.model_validate(json.loads(row["payload"]))
            if context.model_dump(exclude={"workbook_metadata"}) != expected.model_dump(exclude={"workbook_metadata"}):
                raise ValueError("saved context does not match the run manifest")
            return context

    def _check_state(self, state: State) -> State:
        """Revalidate mutable state and preserve frozen A/B outcomes.
        Args: state: Challenge or task state.
        """
        state = type(state).model_validate(state)
        self.dataset.get_context(state.challenge_id)
        if state.challenge_context_ref != f"contexts/{state.challenge_id}.json":
            raise ValueError("state must reference its saved challenge context")
        if isinstance(state, ChallengeState):
            previous = self._db.one("SELECT payload FROM states WHERE state_id=?", (state.challenge_id,))
            if previous:
                saved = ChallengeState.model_validate(json.loads(previous["payload"]))
                if saved.plan is not None and (
                    state.plan != saved.plan or state.plan_ref != saved.plan_ref
                    or state.planning_status != saved.planning_status
                ):
                    raise ValueError("an installed challenge plan cannot be replaced")
        if isinstance(state, TaskState):
            task = self.dataset.get_task(state.task_id)
            if set(state.allowed_assets) != set(task.allowed_assets):
                raise ValueError("task assets do not match the run manifest")
            if not self._phase_record(state.task_id, "A") and (
                state.answer_B is not None or state.reflection is not None
                or state.stage in ("reflection", "freeze_B")
                or state.counters.get("B", UsageCounters()) != UsageCounters()
                or any(item.variant == "B" for item in state.artifacts)
            ):
                raise ValueError("B progress requires a frozen A outcome")
            for variant in ("A", "B"):
                row = self._phase_record(state.task_id, variant)
                if row is None:
                    continue
                frozen = TaskState.model_validate(json.loads(row["payload"]))
                fields = [f"answer_{variant}"]
                if variant == "A":
                    fields += ["plan", "dependency_ids", "workspace_policy", "prior_A_exports",
                               "dependency_exports", "reusable_artifact_refs", "allowed_assets"]
                else:
                    fields += ["reflection"]
                if any(getattr(state, key) != getattr(frozen, key) for key in fields):
                    raise ValueError(f"frozen {variant} fields cannot be changed")
                if state.counters.get(variant) != frozen.counters.get(variant) or state.terminal_status.get(variant) != frozen.terminal_status.get(variant):
                    raise ValueError(f"frozen {variant} usage/status cannot be changed")
                if [item for item in state.artifacts if item.variant == variant] != [item for item in frozen.artifacts if item.variant == variant]:
                    raise ValueError(f"frozen {variant} artifact references cannot be changed")
                if state.observations[:len(frozen.observations)] != frozen.observations:
                    raise ValueError("frozen observations cannot be changed")
        return state

    def save_state(self, state: State) -> None:
        """Validate and atomically replace one checkpoint in SQLite."""
        if not isinstance(state, (ChallengeState, TaskState)):
            raise TypeError("expected a challenge or task state")
        with self._mutex:
            self._check_open()
            with self._db.transaction():
                self._put_state(self._check_state(state))

    def load_state(self, key: str) -> State:
        """Read one checkpoint and verify it against immutable phase records."""
        with self._mutex:
            self._check_open()
            if "/" in key:
                self.dataset.get_task(key)
                model = TaskState
            else:
                self.dataset.get_context(key)
                model = ChallengeState
            row = self._db.one("SELECT payload FROM states WHERE state_id=?", (key,))
            if row is None:
                raise FileNotFoundError(key)
            state = model.model_validate(json.loads(row["payload"]))
            if (state.task_id if isinstance(state, TaskState) else state.challenge_id) != key:
                raise ValueError("saved state identity does not match its key")
            return self._check_state(state)

    def append_event(
        self, challenge_id: str, stage: Stage, status: EventStatus, *,
        task_id: Optional[str] = None, variant: Optional[Variant] = None,
        event_id: Optional[str] = None, **details: object,
    ) -> Event:
        """Validate and append a scoped event in a short database transaction."""
        with self._mutex:
            self._check_open()
            event = self._check_event(Event(
                experiment_id=self.config.experiment_id, run_id=self.config.run_id,
                challenge_id=challenge_id, task_id=task_id, variant=variant,
                event_id=f"event-{uuid4().hex}" if event_id is None else event_id,
                timestamp=datetime.now(timezone.utc).isoformat(), stage=stage, status=status,
                **details,
            ))
            with self._db.transaction():
                if self._get_event(event.event_id):
                    raise ValueError(f"duplicate event_id: {event.event_id}")
                if task_id and self._phase_record(task_id, variant):
                    raise PermissionError("cannot append to a frozen phase")
                if variant == "B" and not self._phase_record(task_id, "A"):
                    raise ValueError("B events require a frozen A outcome")
                if event.parent_event_id:
                    parent = self._get_event(event.parent_event_id)
                    if parent is None:
                        raise ValueError("parent event does not exist")
                    if parent.challenge_id != challenge_id or parent.task_id != task_id or (variant == "A" and parent.variant == "B"):
                        raise ValueError("event parent is outside the permitted history")
                self._db.execute(
                    "INSERT INTO events(event_id,challenge_id,task_id,variant,stage,status,timestamp,parent_event_id,payload) VALUES (?,?,?,?,?,?,?,?,?)",
                    (event.event_id, challenge_id, task_id, variant, stage, status, event.timestamp,
                     event.parent_event_id, event.model_dump_json()),
                )
            return event

    def iter_events(self, challenge_id: str, task_id: Optional[str] = None, variant: Optional[Variant] = None, *, stage: Stage = "planning") -> tuple[Event, ...]:
        """Read only the requested event scope in insertion order."""
        with self._mutex:
            self._check_open()
            self._check_event(Event(
                experiment_id=self.config.experiment_id, run_id=self.config.run_id,
                challenge_id=challenge_id, task_id=task_id, variant=variant,
                event_id="scope", timestamp=datetime.now(timezone.utc).isoformat(), stage=stage, status="ok",
            ))
            if task_id:
                cursor = self._db.execute("SELECT payload FROM events WHERE challenge_id=? AND task_id=? AND variant=? ORDER BY seq", (challenge_id, task_id, variant))
            else:
                cursor = self._db.execute("SELECT payload FROM events WHERE challenge_id=? AND task_id IS NULL AND stage=? ORDER BY seq", (challenge_id, stage))
            return tuple(self._check_event(Event.model_validate(json.loads(row["payload"]))) for row in cursor)

    def begin_model_call(self, event_id: str, request: dict) -> None:
        """Persist the complete JSON request before attempting model inference."""
        with self._mutex:
            self._check_open()
            event = self._get_event(event_id)
            if event is None or event.status != "started":
                raise ValueError("model call requires a started event")
            if event.task_id and self._phase_record(event.task_id, event.variant):
                raise PermissionError("phase is frozen")
            with self._db.transaction():
                self._db.execute("INSERT INTO model_calls(event_id,request,status) VALUES (?,?,'started')",
                                 (event_id, _json_bytes(request).decode("utf-8")))

    def finish_model_call(self, event_id: str, *, result: Optional[GenerationResult] = None,
                          error: Optional[str] = None, duration_seconds: float = 0.0) -> None:
        """Complete one attempted call, retaining raw output even when parsing later fails."""
        with self._mutex:
            self._check_open()
            if (result is None) == (error is None) or not math.isfinite(duration_seconds) or duration_seconds < 0:
                raise ValueError("provide one result or error and a nonnegative duration")
            result = GenerationResult.model_validate(result) if result is not None else None
            with self._db.transaction():
                changed = self._db.execute(
                    "UPDATE model_calls SET response=?,error=?,duration_seconds=?,status=?,input_tokens=?,output_tokens=? WHERE event_id=? AND status='started'",
                    (result.model_dump_json() if result else None, error, duration_seconds,
                     "ok" if result else "error", result.tokens.input_tokens if result else None,
                     result.tokens.output_tokens if result else None, event_id),
                ).rowcount
                if changed != 1:
                    raise ValueError("model call is missing or already completed")

    def load_model_call(self, event_id: str) -> dict:
        """Return one complete request/response for trusted host inspection."""
        with self._mutex:
            self._check_open()
            row = self._db.one("SELECT * FROM model_calls WHERE event_id=?", (event_id,))
            if row is None:
                raise KeyError(event_id)
            record = dict(row)
            for field in ("request", "response"):
                if record[field] is not None:
                    record[field] = json.loads(record[field])
            return record

    def read_planning_artifact(self, challenge_id: str, relative: str) -> str:
        """Read a stored Planner request/response after checking event ownership.
        Args: challenge_id: Expected owner; relative: Artifact path returned by write_artifact.
        """
        with self._mutex:
            self._check_open()
            parts = relative.split("/")
            if len(parts) != 5 or parts[:3] != ["artifacts", challenge_id, "planning"]:
                raise PermissionError("artifact is outside this challenge's planning log")
            event = self._get_event(parts[3])
            if event is None or event.challenge_id != challenge_id or event.task_id is not None or event.stage != "planning":
                raise PermissionError("artifact has no matching planning event")
            if self._db.one("SELECT 1 FROM artifacts WHERE path=? AND event_id=?", (relative, event.event_id)) is None:
                raise PermissionError("artifact is not registered")
            return self._path(relative).read_text(encoding="utf-8")

    def has_phase(self, task_id: str, variant: Variant) -> bool:
        """Check whether a task outcome is frozen, including failed outcomes."""
        with self._mutex:
            self._check_open()
            self.dataset.get_task(task_id)
            if variant not in ("A", "B"):
                raise ValueError("variant must be A or B")
            return self._db.one("SELECT 1 FROM snapshots WHERE task_id=? AND variant=?",
                                (task_id, variant)) is not None

    def load_snapshot(self, task_id: str, variant: Variant) -> SnapshotRef:
        """Return a frozen record reference; it does not imply a saved workbook."""
        with self._mutex:
            if not self.has_phase(task_id, variant):
                raise ValueError("requested phase is not frozen")
            return SnapshotRef.model_validate(json.loads(self._phase_record(task_id, variant)["reference"]))

    def artifact_path(self, reference: ArtifactRef) -> Path:
        """Resolve a checked event-owned artifact for host use, not arbitrary model access."""
        with self._mutex:
            self._check_open()
            reference = ArtifactRef.model_validate(reference)
            self._check_artifact(reference, reference.task_id)
            return self._path(reference.path)

    def write_artifact(self, event_id: str, name: str, content: Union[str, bytes, Path]) -> Union[ArtifactRef, str]:
        """Save a new event-owned artifact; shared planning returns a path only.
        Args: event_id: Existing event; name: File name; content: Text, bytes, or host source file.
        """
        with self._mutex:
            self._check_open()
            validate_component(name)
            event = self._get_event(event_id)
            if event is None:
                raise KeyError(event_id)
            if event.task_id and self._phase_record(event.task_id, event.variant):
                raise PermissionError("cannot write artifacts for a frozen phase")
            scope = f"{event.task_id}/{event.variant}" if event.task_id else f"{event.challenge_id}/planning"
            relative = f"artifacts/{scope}/{event_id}/{name}"
            if not isinstance(content, (str, bytes, Path)):
                raise TypeError("artifact content must be text, bytes, or a source Path")
            if isinstance(content, Path) and (content.is_symlink() or not content.is_file()):
                raise ValueError("artifact source must be a regular file, not a symlink")
            destination = self._path(relative)
            _atomic_write(destination, content.encode("utf-8") if isinstance(content, str) else content)
            with self._db.transaction():
                self._db.execute("INSERT INTO artifacts VALUES (?,?,?,?,?)",
                                 (relative, event_id, event.task_id, event.variant, destination.stat().st_size))
            if event.task_id is None:
                return relative
            return ArtifactRef(path=relative, task_id=event.task_id, event_id=event_id, variant=event.variant)

    def unregistered_artifacts(self) -> tuple[str, ...]:
        """List interrupted file publications for inspection without deleting any data."""
        with self._mutex:
            self._check_open()
            orphaned = []
            directory = self._path("artifacts")
            for path in directory.rglob("*"):
                relative = path.relative_to(self.root).as_posix()
                if self._path(relative).is_file() and not self._db.one("SELECT 1 FROM artifacts WHERE path=?", (relative,)):
                    orphaned.append(relative)
            return tuple(sorted(orphaned))

    def _seal_phase(self, task_id: str, variant: Variant) -> None:
        """Mark frozen records, phase artifacts, and the source log read-only.
        Args: task_id: Manifest ID; variant: Published phase.
        """
        for relative in (f"snapshots/{task_id}/{variant}", f"artifacts/{task_id}/{variant}"):
            directory = self._path(relative)
            if directory.exists():
                for path in sorted(directory.rglob("*"), reverse=True):
                    self._path(path.relative_to(self.root).as_posix()).chmod(0o555 if path.is_dir() else 0o444)
                directory.chmod(0o555)
        log = self._path(f"events/{task_id}/{variant}.jsonl")
        if log.exists():
            log.chmod(0o444)

    def _check_artifact(self, artifact: ArtifactRef, task_id: str) -> None:
        """Check ownership, source event, and the saved artifact file.
        Args: artifact: Output reference; task_id: Owning task.
        """
        event = self._get_event(artifact.event_id)
        expected = f"artifacts/{artifact.task_id}/{artifact.variant}/{artifact.event_id}/"
        if event is None or artifact.task_id != task_id or (
            event.task_id, event.variant
        ) != (artifact.task_id, artifact.variant) or not artifact.path.startswith(expected):
            raise ValueError("artifact provenance does not match its source event")
        if self._db.one("SELECT path FROM artifacts WHERE path=? AND event_id=?", (artifact.path, artifact.event_id)) is None:
            raise ValueError("artifact has no database registration")
        if not self._path(artifact.path).is_file():
            raise FileNotFoundError(artifact.path)

    def freeze_phase(self, state: TaskState, variant: Variant) -> SnapshotRef:
        """Atomically freeze checkpoint, answer and snapshot; files remain external artifacts."""
        with self._mutex:
            self._check_open()
            if variant not in ("A", "B"):
                raise ValueError("variant must be A or B")
            with self._db.transaction():
                state = self._check_state(TaskState.model_validate(state))
                if self._phase_record(state.task_id, variant):
                    raise FileExistsError("phase is already frozen")
                if variant == "B" and not self._phase_record(state.task_id, "A"):
                    raise ValueError("freeze A before freezing B")
                answer = getattr(state, f"answer_{variant}")
                if answer is None or variant not in state.terminal_status:
                    raise ValueError("freezing requires an explicit answer outcome and terminal status")
                if variant == "A" and (state.answer_B is not None or state.reflection is not None):
                    raise ValueError("A cannot contain reflection results")
                if variant == "A" and any(item.variant != "A" for item in state.artifacts):
                    raise ValueError("A cannot contain B artifacts")
                references = list(state.artifacts)
                for export in state.dependency_exports:
                    references.extend(export.artifact_refs)
                for observation in state.observations:
                    event = self._get_event(observation.event_id)
                    if event is None or event.task_id != state.task_id or (variant == "A" and event.variant != "A"):
                        raise ValueError("observation does not reference an event in this task's history")
                    if observation.result_ref:
                        references.append(observation.result_ref)
                    if observation.image_ref:
                        references.append(observation.image_ref)
                for artifact in references:
                    self._check_artifact(artifact, state.task_id)
                pending = self._db.one("SELECT 1 FROM model_calls m JOIN events e ON e.event_id=m.event_id WHERE e.task_id=? AND e.variant=? AND m.status='started' LIMIT 1", (state.task_id, variant))
                if pending:
                    raise ValueError("cannot freeze an unfinished model call")
                snapshot = SnapshotRef(
                    snapshot_id=f"snapshot-{uuid4().hex}", path=f"snapshots/{state.task_id}/{variant}",
                    task_id=state.task_id, variant=variant,
                    source_snapshot_id=self.load_snapshot(state.task_id, "A").snapshot_id if variant == "B" else None,
                )
                self._put_state(state)
                self._db.execute("INSERT INTO snapshots VALUES (?,?,?,?,?,?)", (
                    snapshot.snapshot_id, state.task_id, variant, snapshot.source_snapshot_id,
                    snapshot.model_dump_json(), state.model_dump_json(),
                ))
                self._db.execute("INSERT INTO answers VALUES (?,?,?,?)", (
                    state.task_id, variant, state.terminal_status[variant], answer.model_dump_json(),
                ))
            self._seal_phase(state.task_id, variant)
            return snapshot

    def load_phase(self, task_id: str, variant: Variant) -> TaskState:
        """Read one detached immutable phase state; no full-run state cache is kept."""
        with self._mutex:
            self._check_open()
            if not self.has_phase(task_id, variant):
                raise ValueError("requested phase is not frozen")
            return TaskState.model_validate(json.loads(self._phase_record(task_id, variant)["payload"]))

    def progress(self) -> dict[str, dict[str, str]]:
        """Classify frozen outcomes and interrupted attempts using indexed SQL queries."""
        with self._mutex:
            self._check_open()
            result = {}
            for task_id in self.dataset.task_ids:
                state = self.load_state(task_id)
                phases = {}
                for variant in ("A", "B"):
                    row = self._db.one("SELECT terminal_status FROM answers WHERE task_id=? AND variant=?", (task_id, variant))
                    if row:
                        phases[variant] = row["terminal_status"]
                    elif variant in state.terminal_status or state.stage in (("react", "freeze_A") if variant == "A" else ("reflection", "freeze_B")) or self._db.one("SELECT 1 FROM events WHERE task_id=? AND variant=? LIMIT 1", (task_id, variant)):
                        phases[variant] = "interrupted"
                    else:
                        phases[variant] = "pending"
                result[task_id] = phases
            return result
