"""SQLite persistence for one run, with indexed JSON payloads and short transactions."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (key TEXT PRIMARY KEY, payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS contexts (challenge_id TEXT PRIMARY KEY, payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS states (
    state_id TEXT PRIMARY KEY, challenge_id TEXT NOT NULL, kind TEXT NOT NULL,
    stage TEXT NOT NULL, payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY, event_id TEXT NOT NULL UNIQUE, challenge_id TEXT NOT NULL,
    task_id TEXT, variant TEXT, stage TEXT NOT NULL, status TEXT NOT NULL,
    timestamp TEXT NOT NULL, parent_event_id TEXT REFERENCES events(event_id), payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_scope ON events(challenge_id, task_id, variant, seq);
CREATE INDEX IF NOT EXISTS events_shared ON events(challenge_id, stage, seq) WHERE task_id IS NULL;
CREATE INDEX IF NOT EXISTS events_task ON events(task_id, variant, seq);
CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id TEXT PRIMARY KEY, task_id TEXT NOT NULL, variant TEXT NOT NULL,
    source_snapshot_id TEXT REFERENCES snapshots(snapshot_id),
    reference TEXT NOT NULL, payload TEXT NOT NULL, UNIQUE(task_id, variant)
);
CREATE TABLE IF NOT EXISTS answers (
    task_id TEXT NOT NULL, variant TEXT NOT NULL, terminal_status TEXT NOT NULL,
    payload TEXT NOT NULL, PRIMARY KEY(task_id, variant)
);
CREATE TABLE IF NOT EXISTS artifacts (
    path TEXT PRIMARY KEY, event_id TEXT NOT NULL REFERENCES events(event_id),
    task_id TEXT, variant TEXT, size_bytes INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS model_calls (
    event_id TEXT PRIMARY KEY REFERENCES events(event_id), request TEXT NOT NULL,
    response TEXT, error TEXT, duration_seconds REAL, status TEXT NOT NULL,
    input_tokens INTEGER, output_tokens INTEGER
);
CREATE TRIGGER IF NOT EXISTS snapshots_no_update BEFORE UPDATE ON snapshots
BEGIN SELECT RAISE(ABORT, 'frozen snapshot is immutable'); END;
CREATE TRIGGER IF NOT EXISTS snapshots_no_delete BEFORE DELETE ON snapshots
BEGIN SELECT RAISE(ABORT, 'frozen snapshot is immutable'); END;
CREATE TRIGGER IF NOT EXISTS answers_no_update BEFORE UPDATE ON answers
BEGIN SELECT RAISE(ABORT, 'frozen answer is immutable'); END;
CREATE TRIGGER IF NOT EXISTS answers_no_delete BEFORE DELETE ON answers
BEGIN SELECT RAISE(ABORT, 'frozen answer is immutable'); END;
CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events
BEGIN SELECT RAISE(ABORT, 'events are append only'); END;
CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events
BEGIN SELECT RAISE(ABORT, 'events are append only'); END;
CREATE TRIGGER IF NOT EXISTS events_frozen BEFORE INSERT ON events
WHEN EXISTS (SELECT 1 FROM snapshots WHERE task_id=NEW.task_id AND variant=NEW.variant)
BEGIN SELECT RAISE(ABORT, 'phase is frozen'); END;
CREATE TRIGGER IF NOT EXISTS artifacts_frozen BEFORE INSERT ON artifacts
WHEN EXISTS (SELECT 1 FROM snapshots WHERE task_id=NEW.task_id AND variant=NEW.variant)
BEGIN SELECT RAISE(ABORT, 'phase is frozen'); END;
CREATE TRIGGER IF NOT EXISTS calls_frozen BEFORE UPDATE ON model_calls
WHEN EXISTS (SELECT 1 FROM events e JOIN snapshots s
    ON s.task_id=e.task_id AND s.variant=e.variant WHERE e.event_id=OLD.event_id)
BEGIN SELECT RAISE(ABORT, 'phase is frozen'); END;
CREATE TRIGGER IF NOT EXISTS calls_request_immutable BEFORE UPDATE ON model_calls
WHEN OLD.status!='started' OR OLD.request!=NEW.request OR OLD.event_id!=NEW.event_id
BEGIN SELECT RAISE(ABORT, 'model request and completed response are immutable'); END;
CREATE TRIGGER IF NOT EXISTS calls_no_delete BEFORE DELETE ON model_calls
BEGIN SELECT RAISE(ABORT, 'model calls are retained'); END;
CREATE TRIGGER IF NOT EXISTS artifacts_no_update BEFORE UPDATE ON artifacts
BEGIN SELECT RAISE(ABORT, 'artifact registration is immutable'); END;
CREATE TRIGGER IF NOT EXISTS artifacts_no_delete BEFORE DELETE ON artifacts
BEGIN SELECT RAISE(ABORT, 'artifact registration is immutable'); END;
"""


class SQLiteStore:
    """A single connection; RuntimeStore serializes access across worker threads."""

    def __init__(self, path: Path, create: bool = False) -> None:
        """Open a run database, using rollback journaling for the shared filesystem."""
        uri = path.as_uri() + ("?mode=rwc" if create else "?mode=rw")
        self.connection = sqlite3.connect(uri, uri=True, timeout=30, isolation_level=None,
                                          check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        try:
            self.connection.execute("PRAGMA foreign_keys=ON")
            self.connection.execute("PRAGMA busy_timeout=30000")
            self.connection.execute("PRAGMA synchronous=FULL")
            if create:
                self.connection.execute("PRAGMA journal_mode=DELETE")
                self.connection.executescript(SCHEMA + "\nPRAGMA user_version=1;")
            if self.connection.execute("PRAGMA user_version").fetchone()[0] != 1:
                raise ValueError("unsupported runtime database version")
        except BaseException:
            self.connection.close()
            raise

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Commit a complete mutation; nested calls join the caller's transaction."""
        if self.connection.in_transaction:
            yield
            return
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield
            self.connection.execute("COMMIT")
        except BaseException:
            self.connection.execute("ROLLBACK")
            raise

    def execute(self, sql: str, values: tuple = ()) -> sqlite3.Cursor:
        """Run a parameterized statement without retaining records in memory."""
        return self.connection.execute(sql, values)

    def one(self, sql: str, values: tuple = ()) -> Optional[sqlite3.Row]:
        """Fetch one indexed record, or None when no record matches."""
        return self.execute(sql, values).fetchone()

    def close(self) -> None:
        """Close the connection after the runtime has finished using it."""
        self.connection.close()
