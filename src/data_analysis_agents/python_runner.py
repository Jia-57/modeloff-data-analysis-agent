"""Run bounded legacy scripts or task-scoped persistent Python sessions."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Optional

from .types import EventStatus


@dataclass(frozen=True)
class ExecutionResult:
    """Host execution metadata; stdout, stderr, and generated files stay on disk."""

    status: EventStatus
    duration_seconds: float
    returncode: int
    exception_type: Optional[str]


def run_python(workspace: Path, files: dict[str, Path], code: str, timeout: float, *, seed: int = 0) -> ExecutionResult:
    """Copy selected inputs and execute one script in an independent process.
    Args: workspace: Temporary directory; files: Checked inputs; code: Script; timeout: Seconds.
    """
    started = monotonic()
    inputs, outputs = workspace / "inputs", workspace / "outputs"
    inputs.mkdir()
    outputs.mkdir()
    copies = {}
    for file_id, source in files.items():
        copy = inputs / (file_id + source.suffix.lower())
        shutil.copyfile(source, copy)
        copies[file_id] = str(copy)
    (workspace / "request.json").write_text(json.dumps({"files": copies, "seed": seed}), encoding="utf-8")
    (workspace / "code.py").write_text(code, encoding="utf-8")
    (workspace / "reads.jsonl").touch()
    worker = Path(__file__).with_name("python_worker.py")
    remaining = timeout - (monotonic() - started)
    if remaining <= 0:
        (workspace / "stdout.txt").touch()
        (workspace / "stderr.txt").write_text("Timeout while preparing input copies.\n", encoding="utf-8")
        return ExecutionResult("timeout", monotonic() - started, -1, "TimeoutExpired")
    with (workspace / "stdout.txt").open("wb") as stdout, (workspace / "stderr.txt").open("wb") as stderr:
        process = subprocess.Popen(
            [sys.executable, "-I", "-u", str(worker), str(workspace)],
            cwd=workspace, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
            start_new_session=True,
        )
        timed_out = False
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            # Clean up ordinary descendants too; this does not contain hostile code.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    result_path = workspace / "worker_result.json"
    exception_type = "TimeoutExpired" if timed_out else None
    if not timed_out:
        try:
            exception_type = json.loads(result_path.read_text(encoding="utf-8"))["exception_type"]
        except (OSError, ValueError, KeyError):
            exception_type = "WorkerExit"
    status = "timeout" if timed_out else ("error" if exception_type or process.returncode else "ok")
    return ExecutionResult(status, monotonic() - started, process.returncode, exception_type)


class PythonSession:
    """Keep one task namespace and workspace until close or a timed-out execution."""

    def __init__(self):
        """Allocate task-owned storage without starting a worker before it is needed."""
        import tempfile
        self._directory = tempfile.TemporaryDirectory(prefix="agent-session-")
        self.workspace = Path(self._directory.name)
        (self.workspace / "inputs").mkdir()
        (self.workspace / "outputs").mkdir()
        self.process = None
        self.files = {}
        self.generation = 0
        self.cells = 0
        self.last_code = None
        self.variables = {}

    def prepare(self, files: dict[str, Path]) -> dict[str, str]:
        """Stage authorized inputs once and expose stable, directly usable paths."""
        for key, source in files.items():
            target = self.workspace / "inputs" / (key + source.suffix.lower())
            if key not in self.files:
                shutil.copyfile(source, target)
                self.files[key] = str(target)
        return dict(self.files)

    def run(self, files: dict[str, Path], code: str, timeout: float, *, seed: int = 0) -> ExecutionResult:
        """Run one cell within its timeout; terminate the namespace on worker loss."""
        import selectors
        started = monotonic()
        self.prepare(files)
        for name in ("computed_result.json", "worker_result.json"):
            (self.workspace / name).unlink(missing_ok=True)
        for name in ("reads.jsonl", "stdout.txt", "stderr.txt"):
            (self.workspace / name).write_text("")
        (self.workspace / "code.py").write_text(code, encoding="utf-8")
        (self.workspace / "request.json").write_text(json.dumps({"files": self.files, "seed": seed}))
        if self.process is None or self.process.poll() is not None:
            self.stop()
            self.variables = {}
            self.generation += 1
            self.process = subprocess.Popen(
                [sys.executable, "-I", "-u", str(Path(__file__).with_name("python_worker.py")),
                 str(self.workspace), "--session"], cwd=self.workspace,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, start_new_session=True,
            )
        error, returncode = None, -1
        try:
            self.process.stdin.write("execute\n")
            self.process.stdin.flush()
            with selectors.DefaultSelector() as selector:
                selector.register(self.process.stdout, selectors.EVENT_READ)
                if not selector.select(max(0, timeout - (monotonic() - started))):
                    error = "TimeoutExpired"
                else:
                    response = self.process.stdout.readline()
                    returncode = json.loads(response)["returncode"]
            if error is None:
                result = json.loads((self.workspace / "worker_result.json").read_text())
                error = result["exception_type"]
                self.variables = result.get("variables", {})
        except (OSError, ValueError, KeyError):
            error = "WorkerExit"
        self.cells += 1
        self.last_code = code
        if error in ("TimeoutExpired", "WorkerExit"):
            self.stop()
            self.variables = {}
            (self.workspace / "stderr.txt").write_text(
                f"{error}: Python session was reset; all in-memory variables were lost. Saved output files remain.\n")
        status = "timeout" if error == "TimeoutExpired" else "error" if error else "ok"
        return ExecutionResult(status, monotonic() - started, returncode, error)

    def stop(self):
        """Stop the worker and descendants, including after a task deadline."""
        if self.process is not None:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.process.wait()
            self.process.stdin.close()
            self.process.stdout.close()
            self.process = None

    def close(self):
        """Release process and task storage deterministically."""
        self.stop()
        self._directory.cleanup()

    def __del__(self):
        """Clean up a session whose owner exited without an explicit close."""
        try:
            self.close()
        except (AttributeError, OSError):
            pass
