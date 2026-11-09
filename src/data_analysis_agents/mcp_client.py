"""Run a task-scoped MCP client and Excel server in isolated local processes."""

from __future__ import annotations

import asyncio
from contextlib import suppress
import json
import os
import selectors
import signal
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from threading import RLock
from time import monotonic


class MCPClient:
    """Own one lazy local service connection; no MCP imports enter the model environment."""

    def __init__(self, python: Path, command: Path, workspace: Path, settings: dict) -> None:
        """Bind trusted executable paths, one task workspace and frozen service settings."""
        self.python, self.command, self.workspace = python, command, workspace
        self.settings = settings
        self._process = None
        self._log = None
        self._buffer = b""
        self._lock = RLock()
        self._sequence = 0
        self.info = None

    def _receive(self, timeout: float) -> dict:
        """Read one bounded bridge message without blocking beyond the call deadline."""
        deadline = monotonic() + timeout
        with selectors.DefaultSelector() as selector:
            selector.register(self._process.stdout, selectors.EVENT_READ)
            while b"\n" not in self._buffer:
                remaining = deadline - monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise TimeoutError("MCP bridge exceeded the call deadline")
                chunk = os.read(self._process.stdout.fileno(), 65536)
                if not chunk:
                    raise ConnectionError("MCP bridge exited; see saved stderr")
                self._buffer += chunk
                if len(self._buffer) > 16 * 1024 * 1024:
                    raise ValueError("MCP bridge response exceeds 16 MiB")
        line, self._buffer = self._buffer.split(b"\n", 1)
        response = json.loads(line)
        if not isinstance(response, dict):
            raise ValueError("MCP bridge response must be an object")
        if response.get("error"):
            raise RuntimeError(response["error"])
        return response

    def open(self, timeout: float) -> dict:
        """Start the bridge, verify installed versions and discover server tools once."""
        with self._lock:
            if self.info is not None:
                return self.info
            if timeout <= 0:
                raise TimeoutError("No time remains for MCP initialization")
            for executable in (self.python, self.command):
                if not executable.is_file() or not os.access(executable, os.X_OK):
                    raise FileNotFoundError(f"MCP executable is unavailable: {executable}")
            request = {**self.settings, "command": str(self.command),
                       "workspace": str(self.workspace), "timeout_seconds": timeout}
            path = self.workspace / "client.json"
            path.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
            self._log = (self.workspace / "service-stderr.txt").open("ab", buffering=0)
            self._process = subprocess.Popen(
                [str(self.python), "-I", "-u", str(Path(__file__).resolve()), "--worker", str(path)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
                cwd=self.workspace, start_new_session=True,
            )
            try:
                self.info = self._receive(timeout)
                missing = set(self.settings["tools"]) - set(self.info["tools"])
                if missing:
                    raise ValueError(f"Excel MCP is missing configured tools: {sorted(missing)}")
                return self.info
            except Exception:
                self.close()
                raise

    def call_tool(self, name: str, arguments: dict, timeout: float) -> dict:
        """Call one whitelisted MCP tool; failures close the connection without replaying writes."""
        with self._lock:
            started = monotonic()
            self.open(timeout)
            remaining = timeout - (monotonic() - started)
            if remaining <= 0:
                self.close()
                raise TimeoutError("MCP initialization exhausted the call deadline")
            self._sequence += 1
            request = {"id": self._sequence, "name": name, "arguments": arguments, "timeout": remaining}
            try:
                self._process.stdin.write((json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8"))
                self._process.stdin.flush()
                response = self._receive(remaining)
                if response.get("id") != self._sequence:
                    raise ValueError("MCP bridge response has the wrong request ID")
                return response["result"]
            except Exception:
                self.close()
                raise

    def close(self) -> None:
        """Close transport and reap the bridge and its separate server process group."""
        with self._lock:
            process = self._process
            if process is not None:
                children = []
                try:
                    children = [int(value) for value in Path(
                        f"/proc/{process.pid}/task/{process.pid}/children"
                    ).read_text().split()]
                except OSError:
                    pass
                if process.stdin:
                    try:
                        process.stdin.close()
                    except BrokenPipeError:
                        pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                for child in children:
                    try:
                        # The SDK starts its server in a separate session.
                        if os.getpgid(child) == child:
                            os.killpg(child, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                if process.stdout:
                    process.stdout.close()
            if self._log:
                self._log.close()
            self._process, self._log, self.info = None, None, None
            self._buffer = b""


def _send(payload: dict) -> None:
    """Write one bridge reply; stdout is reserved for this protocol."""
    print(json.dumps(payload, ensure_ascii=False, allow_nan=False), flush=True)


async def _serve(settings: dict) -> None:
    """Run the standard MCP SDK inside Python 3.11 with one task-confined Excel service."""
    from mcp import Client, StdioServerParameters

    for package, expected in (("excel-mcp-server", settings["version"]), ("mcp", settings["sdk_version"])):
        actual = version(package)
        if actual != expected:
            raise ValueError(f"{package} version mismatch: expected {expected}, installed {actual}")
    workspace = Path(settings["workspace"]).resolve()
    args = ["-I", "-u", str(Path(__file__).resolve()), "--excel-server", str(workspace),
            str(settings["max_file_mb"]), "1" if settings["allow_write"] else "0"]
    # Never inherit an additional Excel directory or an opt-in VBA write setting.
    parameters = StdioServerParameters(command=sys.executable, args=args, cwd=workspace,
                                       env={"EXCEL_FILES_PATH": "", "EXCEL_MCP_ALLOW_VBA_WRITE": "0",
                                            "EXCEL_MCP_READ_ONLY": "0"})
    async with Client(parameters, mode="legacy", cache=None,
                      read_timeout_seconds=settings["timeout_seconds"]) as client:
        names, cursor = [], None
        for _ in range(100):
            page = await client.list_tools(cursor=cursor)
            names.extend(tool.name for tool in page.tools)
            cursor = page.next_cursor
            if not cursor:
                break
        else:
            raise ValueError("MCP tool discovery exceeded 100 pages")
        _send({"tools": names, "server": client.server_info.model_dump(mode="json")
               if client.server_info else None,
               "versions": {"excel-mcp-server": settings["version"], "mcp": settings["sdk_version"]}})
        while line := await asyncio.to_thread(sys.stdin.readline):
            request = json.loads(line)
            try:
                if request["name"] not in settings["tools"]:
                    raise PermissionError("MCP tool was not configured")
                source = Path(request["arguments"]["path"]).resolve(strict=True)
                if workspace not in source.parents:
                    raise PermissionError("MCP input escapes the task workspace")
                result = await client.call_tool(request["name"], request["arguments"],
                                                read_timeout_seconds=request["timeout"])
                _send({"id": request["id"], "result": result.model_dump(mode="json", by_alias=True)})
            except Exception as error:
                _send({"id": request.get("id"), "error": f"{type(error).__name__}: {error}"})
                # A timed-out write must never continue alongside another call.
                break


async def _wake_loop() -> None:
    """Bound callback latency on nodes where worker-thread wakeups do not wake asyncio."""
    while True:
        await asyncio.sleep(0.05)


async def _run_with_wakeup(coroutine) -> None:
    wakeup = asyncio.create_task(_wake_loop())
    try:
        await coroutine
    finally:
        wakeup.cancel()
        with suppress(asyncio.CancelledError):
            await wakeup


async def _run_excel_server(workspace: Path, max_file_mb: int, allow_write: bool) -> None:
    """Start the installed Excel MCP service with a responsive local stdio loop."""
    from excel_mcp.config import Limits, Settings
    from excel_mcp.server import create_server

    settings = Settings(allowed_dirs=[workspace.resolve()], read_only=not allow_write,
                        allow_vba_write=False,
                        limits=Limits(max_file_bytes=max_file_mb * 1024 * 1024))
    await create_server(settings).run_stdio_async()


def main() -> int:
    """Launch the bridge worker or its task-confined Excel server process."""
    try:
        if len(sys.argv) == 3 and sys.argv[1] == "--worker":
            settings = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
            asyncio.run(_run_with_wakeup(_serve(settings)))
        elif len(sys.argv) == 5 and sys.argv[1] == "--excel-server":
            workspace = Path(sys.argv[2]).resolve(strict=True)
            asyncio.run(_run_with_wakeup(_run_excel_server(
                workspace, int(sys.argv[3]), sys.argv[4] == "1")))
        else:
            raise ValueError("usage: mcp_client.py --worker CLIENT_JSON | "
                             "--excel-server WORKSPACE MAX_FILE_MB ALLOW_WRITE")
        return 0
    except Exception as error:
        _send({"error": f"{type(error).__name__}: {error}"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
