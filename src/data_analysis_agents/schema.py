"""Shared Pydantic settings; validate external inputs and state checkpoints."""

import ast
import json
import re
from copy import deepcopy

from pydantic import BaseModel, ConfigDict


def generation_schema(schema: dict) -> dict:
    """Expand unrestricted JSON value schemas into equivalent types understood by the decoder."""
    def expand(node):
        """Visit schema positions without changing empty property maps or annotation data."""
        if not isinstance(node, dict):
            return node
        if not node:
            return {"type": ["string", "number", "boolean", "null", "object", "array"]}
        result = deepcopy(node)
        for key in ("$defs", "definitions", "properties", "patternProperties"):
            if key in result:
                result[key] = {name: expand(child) for name, child in result[key].items()}
        for key in ("items", "additionalProperties", "contains", "not"):
            if key in result:
                result[key] = expand(result[key])
        for key in ("anyOf", "oneOf", "allOf"):
            if key in result:
                result[key] = [expand(child) for child in result[key]]
        return result

    return expand(schema)


def _object_value(content: str) -> dict:
    """Decode complete objects; allow literal quotes/trailing commas without inventing fields."""
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        cleaned, quote, escaped = [], None, False
        for index, character in enumerate(content):
            if quote:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == quote:
                    quote = None
            elif character in ('"', "'"):
                quote = character
            elif character == "," and content[index + 1:].lstrip().startswith(("}", "]")):
                continue
            cleaned.append(character)
        try:
            value = json.loads("".join(cleaned))
        except json.JSONDecodeError:
            try:
                value = ast.literal_eval("".join(cleaned))
            except (SyntaxError, ValueError):
                raise ValueError("No complete JSON object; missing data is not synthesized") from None
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError("model output must be an object with string keys")
    # Reject non-JSON literal types and nonfinite numbers before downstream validation.
    encoded = json.dumps(value, allow_nan=False)
    if json.loads(encoded) != value:
        raise ValueError("literal contains non-JSON values")
    return value


def json_candidates(text: str) -> list[dict]:
    """Find balanced outer objects, respecting quoted braces and refusing truncated structures."""
    candidates, start, depth, quote, escaped = [], None, 0, None, False
    for index, character in enumerate(text):
        if start is None:
            if character == "{":
                start, depth = index, 1
            continue
        if quote:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
        elif character in ('"', "'"):
            quote = character
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                try:
                    candidates.append(_object_value(text[start:index + 1]))
                except (ValueError, TypeError):
                    pass
                start = None
    # A complete field list may omit only its outer object; never close partial parameters.
    body = re.sub(r"</?(?:tool_call|think)>|```(?:json)?", "", text.strip(), flags=re.I).strip()
    if re.match(r"^[\"'][A-Za-z_][\w]*[\"']\s*:", body):
        try:
            candidates.append(_object_value("{" + body + "}"))
        except (ValueError, TypeError):
            pass
    return candidates


def repair_action_closers(text: str) -> str:
    """Close containers only after complete JSON values; never complete strings or fields."""
    start = text.find("{")
    if start < 0:
        return text
    raw = text[start:].strip()
    stack, quote, escaped = [], False, False
    for char in raw:
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quote = False
        elif char == '"':
            quote = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]":
            if not stack or stack.pop() != ("{" if char == "}" else "["):
                return text
    if quote or not stack or len(stack) > 3:
        return text
    candidate = raw + "".join("}" if char == "{" else "]" for char in reversed(stack))
    try:
        value = json.loads(candidate)
    except ValueError:
        return text
    return candidate if isinstance(value, dict) else text


def parse_model_json(text: str, required_keys: set[str] | None = None) -> dict:
    """Return a unique complete object, optionally filtering by the phase's identifying keys."""
    values = json_candidates(text)
    if required_keys:
        values = [value for value in values if required_keys <= value.keys()]
    unique = {json.dumps(value, sort_keys=True, ensure_ascii=False): value for value in values}
    if len(unique) != 1 or len(values) != 1:
        raise ValueError("Expected one unambiguous complete object; found " + str(len(unique)))
    return next(iter(unique.values()))


def normalize_tool_action(payload: dict, tool_names: set[str]) -> dict:
    """Adapt protocol shapes using registered names; preserve arguments and reject conflicts."""
    normalized = deepcopy(payload)
    for key in ("thought", "reasoning", "analysis"):
        normalized.pop(key, None)
    if "function" in normalized:
        if set(normalized) - {"function", "type", "id"}:
            raise ValueError("Conflicting function and action fields")
        normalized = deepcopy(normalized["function"])
        if not isinstance(normalized, dict):
            raise ValueError("function must identify an object, not free text")
    if isinstance(normalized.get("action"), dict):
        action = normalized["action"]
        normalized = deepcopy(action.get("payload", action))
        if not isinstance(normalized, dict):
            raise ValueError("nested action payload must be an object")
        if action.get("type") in ("tool", "tool_use"):
            normalized.setdefault("action", "tool")
    if isinstance(normalized.get("name"), str) and normalized["name"] in tool_names:
        if normalized.get("tool_name", normalized["name"]) != normalized["name"]:
            raise ValueError("name and tool_name identify different tools")
        normalized["tool_name"] = normalized.pop("name")
    for alias in ("parameters", "input"):
        if alias in normalized:
            if "arguments" in normalized and normalized["arguments"] != normalized[alias]:
                raise ValueError("Conflicting argument fields")
            normalized["arguments"] = normalized.pop(alias)
    if isinstance(normalized.get("arguments"), str):
        normalized["arguments"] = _object_value(normalized["arguments"])
    if "tool_name" in normalized and "action" not in normalized:
        normalized["action"] = "tool"
    elif "value" in normalized and "action" not in normalized:
        normalized["action"] = "final"
    if normalized.get("tool_name") == "execute_python" and isinstance(normalized.get("arguments"), dict):
        args = normalized["arguments"]
        if "brief_reason" in args and isinstance(args["brief_reason"], str):
            if "brief_reason" not in normalized or normalized["brief_reason"] == args["brief_reason"]:
                normalized["brief_reason"] = args.pop("brief_reason")
    name = normalized.get("action")
    if isinstance(name, str) and name in tool_names:
        if "tool_name" in normalized and normalized["tool_name"] != name:
            raise ValueError("action and tool_name identify different tools")
        normalized.update(action="tool", tool_name=name)
    elif isinstance(name, str) and name in ("tool_use", "function_call"):
        normalized["action"] = "tool"
    if normalized.get("action") == "tool" and (not isinstance(normalized.get("tool_name"), str) or normalized["tool_name"] not in tool_names):
        raise ValueError("Missing or unregistered tool_name")
    return normalized


def _tagged_tool_candidates(text: str, tool_names: set[str]) -> list[dict]:
    """Read complete function/parameter tags without treating surrounding prose as arguments."""
    found = []
    pattern = r"<function=([A-Za-z_][\w.-]*)>(.*?)</function>"
    for match in re.finditer(pattern, text, flags=re.S | re.I):
        name, body = match.groups()
        if name == "python" and "execute_python" in tool_names:
            name = "execute_python"
        if name not in tool_names:
            continue
        parameters = {}
        for parameter in re.finditer(r"<parameter=([A-Za-z_][\w.-]*)>(.*?)</parameter>", body,
                                     flags=re.S | re.I):
            key, raw = parameter.groups()
            if key in parameters:
                parameters = None
                break
            raw = raw.strip()
            try:
                parameters[key] = json.loads(raw)
            except json.JSONDecodeError:
                parameters[key] = raw
        if parameters:
            found.append({"action": "tool", "tool_name": name, "arguments": parameters})
    return found


def action_candidates(text: str, tool_names: set[str], native_calls: list[dict] | None = None,
                      *, errors: list[str] | None = None) -> list[dict]:
    """Extract valid candidates independently; irrelevant malformed fragments cannot hide one."""
    without_code = re.sub(r"<parameter=code>.*?</parameter>", "", text, flags=re.S | re.I)
    values = native_calls if native_calls else [*json_candidates(without_code),
                                                 *_tagged_tool_candidates(text, tool_names)]
    actions, failures = [], []
    for value in values:
        if not any(key in value for key in ("action", "tool_name", "function", "name", "value")):
            continue
        try:
            actions.append(normalize_tool_action(value, tool_names))
        except (ValueError, TypeError) as error:
            failures.append(str(error))
    if errors is not None:
        errors.extend(failures)
    if not actions and failures:
        raise ValueError("No usable action: " + failures[0][:400])
    return actions


def is_inspection_cell(code: str) -> bool:
    """Recognize simple read/preview cells without treating state-changing computation as repeats."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    readers = {"read_table", "read_csv", "read_excel", "ExcelFile"}
    previews = {"head", "tail", "columns", "sheet_names"}
    seen_preview = False
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.Assign) and all(isinstance(target, ast.Name) for target in node.targets):
            call = node.value
            name = (call.func.attr if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    else call.func.id if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) else None)
            if name in readers:
                continue
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name) and node.value.func.id == "print"):
            calls = [item for item in ast.walk(node.value) if isinstance(item, ast.Call)]
            if all((isinstance(item.func, ast.Name) and item.func.id == "print") or
                   (isinstance(item.func, ast.Attribute) and item.func.attr in
                    {"head", "tail", "to_string", "tolist"}) for item in calls):
                seen_preview |= any(isinstance(item, ast.Attribute) and item.attr in previews
                                    for item in ast.walk(node.value))
                continue
        return False
    return seen_preview


def _narrative_line(line: str) -> bool:
    """Identify an obvious prose line; ambiguous lines remain code for model repair."""
    value = line.strip()
    if not value or value.startswith("#"):
        return False
    code_words = r"import|from|def|class|async|for|while|if|elif|else|try|except|finally|with|return|raise|assert|yield|print|pass|break|continue|del|global|nonlocal"
    if re.match(r"^(?:" + code_words + r")\b", value):
        return False
    if re.search(r"[=()\[\]{}+*/\\]", value):
        return False
    return bool(re.match(r"^(?:here(?:'s| is)|the following|this code|explanation|note|output|i will|let(?:'s| us))\b", value, re.I)
                or (" " in value and value.endswith((":", "."))))


def _validate_python_code(code: str, input_paths: dict[str, str] | None = None) -> str:
    """Check syntax and obvious misuse of opaque asset IDs without executing code."""
    if not code:
        raise ValueError("Python code is empty")
    try:
        tree = ast.parse(code, filename="<model-code>")
        if not tree.body:
            raise ValueError("Python code has no executable statements")
        bound_paths = False
        for node in ast.walk(tree):
            if (input_paths and isinstance(node, ast.Call) and node.args
                    and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)
                    and node.args[0].value in input_paths
                    and ((isinstance(node.func, ast.Attribute) and node.func.attr in
                          ("read_csv", "read_excel", "ExcelFile", "load_workbook"))
                         or isinstance(node.func, ast.Name) and node.func.id == "open")):
                node.args[0].value = input_paths[node.args[0].value]
                bound_paths = True
                if (str(node.args[0].value).endswith(".xlsb") and isinstance(node.func, ast.Attribute)
                        and node.func.attr in ("read_excel", "ExcelFile")
                        and not any(item.arg == "engine" for item in node.keywords)):
                    node.keywords.append(ast.keyword(arg="engine", value=ast.Constant(value="calamine")))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("read_csv", "read_excel", "ExcelFile", "load_workbook"):
                if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str) and re.fullmatch(
                        r"(?:asset|artifact)-[A-Za-z0-9_.-]+", node.args[0].value):
                    raise ValueError("Asset IDs are not filenames; use read_table(id) or files[id]")
        if len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Dict):
            raise ValueError("Return Python code only, without a JSON parameter wrapper")
        if bound_paths:
            ast.fix_missing_locations(tree)
            code = ast.unparse(tree)
        compile(tree, "<model-code>", "exec")
    except SyntaxError as error:
        raise ValueError(f"Python syntax error at line {error.lineno}: {error.msg}") from error
    return code


def parse_python_code(text: str, *, input_paths: dict[str, str] | None = None) -> str:
    """Recover one validated Python script from plain text or common model wrappers."""
    raw = re.sub(r"^\s*<think>.*?</think>\s*", "", text.strip(), flags=re.S)
    raw = re.sub(r"\s*</think>\s*$", "", raw)
    blocks = re.findall(r"```(?:python|py)?[ \t]*\r?\n(.*?)\r?\n[ \t]*```", raw,
                        flags=re.S | re.I)
    tagged = re.findall(r"<parameter=code>\s*(.*?)\s*</parameter>", raw, flags=re.S | re.I)
    if "```" in raw and (not blocks or raw.count("```") != 2 * len(blocks)):
        raise ValueError("Python code block is incomplete")
    if "<parameter=code>" in raw.lower() and not tagged:
        raise ValueError("Python code parameter is incomplete")
    candidates = [item.strip() for item in (blocks or tagged)]
    if not candidates and raw.startswith("{"):
        try:
            wrapper = _object_value(raw)
            if isinstance(wrapper.get("code"), str):
                candidates.append(wrapper["code"].strip())
        except ValueError:
            pass
    if not candidates:
        candidates = [raw]
    unique = list(dict.fromkeys(candidates))
    if len(unique) != 1:
        raise ValueError("Multiple different Python code candidates; request a targeted repair")
    code = unique[0]
    try:
        return _validate_python_code(code, input_paths)
    except ValueError as initial:
        if blocks or tagged or code != raw or not str(initial).startswith("Python syntax error"):
            raise
        # Comment only recognizable prose at the syntax-error line; never remove code-like lines.
        lines = code.splitlines()
        for _ in range(min(3, len(lines))):
            try:
                ast.parse("\n".join(lines), filename="<model-code>")
                break
            except SyntaxError as syntax:
                index = (syntax.lineno or 1) - 1
                if index >= len(lines) or not _narrative_line(lines[index]):
                    raise initial from syntax
                lines[index] = "# " + lines[index].lstrip()
        return _validate_python_code("\n".join(lines).strip(), input_paths)


class Record(BaseModel):
    """Strict contracts; frozen attributes do not freeze nested lists or files."""

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        frozen=True,
        validate_default=True,
        allow_inf_nan=False,
        revalidate_instances="always",
    )


class MutableRecord(Record):
    """Mutable runtime state; call model_validate(state) before saving it."""

    model_config = ConfigDict(frozen=False)
