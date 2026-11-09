"""Execute bounded script cells in a host-owned namespace; this is not a sandbox."""

from __future__ import annotations

import json
import contextlib
import os
import random
import sys
import traceback
from pathlib import Path


def execute(workspace: Path, namespace: dict) -> int:
    """Load prepared copies, expose helpers, and record execution errors.
    Args: CLI supplies one workspace directory containing request.json and code.py.
    """
    os.chdir(workspace)
    request = json.loads((workspace / "request.json").read_text(encoding="utf-8"))
    if not namespace:
        random.seed(request.get("seed", 0))
    files = {file_id: Path(path) for file_id, path in request["files"].items()}
    reads_path = workspace / "reads.jsonl"
    result = {"exception_type": None}

    def read_table(file_id: str, sheet=0, header=None, **kwargs):
        """Read a selected CSV or workbook without recalculating formulas.
        Args: file_id: Supplied ID; sheet: Excel sheet; header/kwargs: pandas read options.
        """
        import pandas as pd

        if file_id not in files:
            raise PermissionError(f"file_id was not supplied: {file_id}")
        path = files[file_id]
        record = {"file_id": file_id, "sheet": sheet, "header": header, "options": kwargs}
        with reads_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, default=repr) + "\n")
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path, header=header, **kwargs)
        engines = {".xlsx": "openpyxl", ".xlsm": "openpyxl", ".xlsb": "calamine"}
        if path.suffix.lower() not in engines:
            raise ValueError(f"read_table does not support {path.suffix}; use files[file_id]")
        engine = kwargs.pop("engine", engines[path.suffix.lower()])
        return pd.read_excel(path, sheet_name=sheet, header=header, engine=engine, **kwargs)

    def emit_result(value, unit=None, checks=None):
        """Publish the value; malformed optional metadata is a warning, not a computation failure."""
        if value is None:
            raise ValueError("computed result cannot be null")
        if unit is not None and not isinstance(unit, str):
            raise ValueError("result unit must be a string or null")
        supplied = checks or {}
        if not isinstance(supplied, dict):
            print("emit_result warning: ignored non-mapping checks", file=sys.stderr)
            supplied = {}
        if any(type(passed) is not bool for passed in supplied.values()):
            print("emit_result warning: ignored non-boolean check metadata", file=sys.stderr)
        checks = {str(key): passed for key, passed in supplied.items() if type(passed) is bool}
        def json_value(item):
            """Convert NumPy scalars/arrays while rejecting unsupported result objects."""
            if hasattr(item, "tolist"):
                return item.tolist()
            if hasattr(item, "item"):
                return item.item()
            raise TypeError(f"unsupported computed value: {type(item).__name__}")
        text = json.dumps({"value": value, "unit": unit, "checks": checks},
                          allow_nan=False, ensure_ascii=False, default=json_value)
        (workspace / "computed_result.json").write_text(text, encoding="utf-8")
        print(text)

    namespace.update({
        "__name__": "__main__", "__file__": str(workspace / "code.py"),
        "files": files, "read_table": read_table, "emit_result": emit_result,
        "random_seed": request.get("seed", 0),
        "output_dir": workspace / "outputs",
    })
    try:
        code_path = workspace / "code.py"
        exec(compile(code_path.read_text(encoding="utf-8"), str(code_path), "exec"), namespace)
    except BaseException as error:
        result["exception_type"] = type(error).__name__
        traceback.print_exc()
    variables = {}
    for name, value in list(namespace.items()):
        if name.startswith("_") or name in {"files", "read_table", "emit_result", "output_dir", "random_seed"}:
            continue
        info = {"type": type(value).__name__}
        if isinstance(value, (str, int, float, bool, type(None))):
            info["value"] = str(value)[:80]
        else:
            try:
                shape = getattr(value, "shape", None)
                if isinstance(shape, tuple):
                    info["shape"] = str(shape)[:100]
            except Exception:
                pass
        variables[name] = info
        if len(variables) >= 20:
            break
    result["variables"] = variables
    (workspace / "worker_result.json").write_text(json.dumps(result), encoding="utf-8")
    return 1 if result["exception_type"] else 0


def main() -> int:
    """Execute one legacy script or accept sequential cells in one task namespace."""
    workspace = Path(sys.argv[1])
    if "--session" not in sys.argv:
        return execute(workspace, {})
    namespace = {}
    for command in sys.stdin:
        if command.strip() != "execute":
            break
        with (workspace / "stdout.txt").open("w") as stdout, (workspace / "stderr.txt").open("w") as stderr:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                status = execute(workspace, namespace)
        print(json.dumps({"returncode": status}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
