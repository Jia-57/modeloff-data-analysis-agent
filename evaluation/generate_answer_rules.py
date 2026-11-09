"""Generate per-task draft scoring rules from the manifest, questions, and gold labels.

The output is a reviewable draft. It must not be treated as the final scoring
specification until units, tolerances, and composite fields are checked.
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
import os
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1.0"
OPTION_RE = re.compile(r"(?:^|\s)([A-I])\s*[.)]\s+", re.IGNORECASE)
NUMBER_RE = re.compile(
    r"^\s*(?P<currency>[$€£])?\s*"
    r"(?P<number>[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*"
    r"(?P<scale>[kKmMbB])?\s*(?P<percent>%)?\s*$"
)
DATE_FORMATS = ("%Y-%m-%d", "%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            rows.append(row)
    return rows


def parse_number(value: Any) -> tuple[Decimal, dict[str, str] | None, int] | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return Decimal(value), None, 0
    if isinstance(value, float):
        decimal_value = Decimal(str(value))
        decimals = max(0, -decimal_value.as_tuple().exponent)
        return decimal_value, None, decimals
    if not isinstance(value, str):
        return None

    match = NUMBER_RE.fullmatch(value)
    if not match:
        return None
    number_text = match.group("number").replace(",", "")
    number = Decimal(number_text)
    decimals = len(number_text.partition(".")[2]) if "." in number_text else 0
    unit: dict[str, str] = {}
    if match.group("currency"):
        unit["symbol"] = match.group("currency")
    if match.group("scale"):
        unit["scale"] = {
            "k": "thousand", "m": "million", "b": "billion"
        }[match.group("scale").lower()]
    if match.group("percent"):
        unit["symbol"] = "%"
    return number, (unit or None), decimals


def parse_date(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    for date_format in DATE_FORMATS:
        try:
            return datetime.strptime(candidate, date_format).date().isoformat()
        except ValueError:
            continue
    return None


def infer_leaf(value: Any) -> dict[str, Any]:
    if isinstance(value, (dict, list)):
        return {
            "answer_kind": "composite",
            "normalization_profile": "review_field_structure",
            "unit": None,
            "tolerance_candidate": None,
        }
    if parse_date(value):
        return {
            "answer_kind": "date",
            "normalization_profile": "date_iso",
            "unit": None,
            "tolerance_candidate": None,
        }
    parsed = parse_number(value)
    if parsed:
        number, unit, decimals = parsed
        tolerance = Decimal("0.5") * (Decimal(10) ** -decimals) if decimals else Decimal(0)
        return {
            "answer_kind": "numeric",
            "normalization_profile": "decimal_numeric",
            "unit": unit,
            "tolerance_candidate": {
                "mode": "absolute",
                "value": format(tolerance, "f"),
                "basis": "displayed_unit",
                "source": "half_last_displayed_digit" if decimals else "exact_integer_candidate",
            },
        }
    if isinstance(value, str):
        return {
            "answer_kind": "text",
            "normalization_profile": "unicode_nfc_trim_exact",
            "unit": None,
            "tolerance_candidate": None,
        }
    return {
        "answer_kind": "unknown",
        "normalization_profile": "manual_review",
        "unit": None,
        "tolerance_candidate": None,
    }


def build_rule(
    task: dict[str, Any], gold: Any, gold_ref: Any, question_text: str
) -> dict[str, Any]:
    choice_labels = sorted({m.group(1).upper() for m in OPTION_RE.finditer(question_text)})
    reasons: list[str] = []

    if isinstance(gold, dict):
        fields = {str(key): infer_leaf(value) for key, value in gold.items()}
        rule = {
            "answer_kind": "composite",
            "normalization_profile": "compare_required_fields",
            "required_fields": list(fields),
            "fields": fields,
            "unit": None,
            "tolerance_candidate": None,
        }
        reasons.append("confirm_required_fields_and_each_field_rule")
    elif isinstance(gold, list):
        rule = {
            "answer_kind": "composite",
            "normalization_profile": "compare_ordered_items",
            "required_fields": [str(index) for index in range(len(gold))],
            "fields": {str(index): infer_leaf(value) for index, value in enumerate(gold)},
            "unit": None,
            "tolerance_candidate": None,
        }
        reasons.append("confirm_item_order_and_each_item_rule")
    elif (
        isinstance(gold, str)
        and re.fullmatch(r"\s*[A-Ia-i]\s*", gold)
        and len(choice_labels) >= 2
        and gold.strip().upper() in choice_labels
    ):
        rule = {
            "answer_kind": "choice",
            "normalization_profile": "choice_label_exact",
            "allowed_labels": choice_labels,
            "unit": None,
            "tolerance_candidate": None,
        }
        reasons.append("verify_option_parser_and_gold_label")
    else:
        rule = infer_leaf(gold)
        if rule["answer_kind"] == "numeric":
            reasons.append("verify_unit_scale_and_tolerance_against_question")
            if rule["unit"] is None:
                reasons.append("unit_not_explicit_in_gold_value")
        elif rule["answer_kind"] == "date":
            reasons.append("verify_required_date_precision_and_format")
        elif rule["answer_kind"] == "text":
            if re.fullmatch(r"\s*[A-Ia-i]\s*", str(gold)):
                reasons.append("single_letter_gold_but_options_not_confirmed")
            else:
                reasons.append("verify_text_case_and_whitespace_policy")
        else:
            reasons.append("answer_type_not_inferred")

    # Keep all generated rules explicitly provisional until a human reviews them.
    return {
        "task_id": task["task_id"],
        **rule,
        "review_status": "needs_review",
        "review_reasons": reasons,
        "evidence": {
            "question_path": task["question_file"]["path"],
            "gold_ref": gold_ref,
        },
    }


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--manifest", type=Path, default=project_root / "data/preprocessed/task_inputs/manifest.jsonl")
    parser.add_argument("--labels", type=Path, default=project_root / "data/preprocessed/evaluation_labels/labels.jsonl")
    parser.add_argument("--output", type=Path, default=project_root / "data/preprocessed/evaluation_rules/answer_rules.json")
    args = parser.parse_args()

    root = args.project_root.resolve()
    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    labels_path = args.labels if args.labels.is_absolute() else root / args.labels
    output_path = args.output if args.output.is_absolute() else root / args.output
    manifest = read_jsonl(manifest_path)
    labels = read_jsonl(labels_path)

    task_by_id = {task["task_id"]: task for task in manifest}
    gold_by_id = {label["task_id"]: label for label in labels}
    if len(task_by_id) != len(manifest) or len(gold_by_id) != len(labels):
        raise ValueError("duplicate task_id in manifest or labels")
    if set(task_by_id) != set(gold_by_id):
        missing_labels = sorted(set(task_by_id) - set(gold_by_id))
        missing_tasks = sorted(set(gold_by_id) - set(task_by_id))
        raise ValueError(f"task_id mismatch; missing_labels={missing_labels}, missing_tasks={missing_tasks}")

    rules = {}
    for task_id, task in task_by_id.items():
        question_path = root / task["question_file"]["path"]
        if not question_path.is_file():
            raise FileNotFoundError(f"question file missing for {task_id}: {question_path}")
        question_text = question_path.read_text(encoding="utf-8-sig", errors="replace")
        label = gold_by_id[task_id]
        rules[task_id] = build_rule(
            task, label.get("gold_raw"), label.get("gold_ref"), question_text
        )

    result = {
        "schema_version": SCHEMA_VERSION,
        "generation_status": "draft_requires_human_review",
        "source_manifest": {
            "path": manifest_path.resolve().relative_to(root).as_posix(),
        },
        "source_labels": {
            "path": labels_path.resolve().relative_to(root).as_posix(),
        },
        "task_count": len(rules),
        "rules": rules,
        "notes": [
            "Rules are heuristic drafts and must be reviewed before formal scoring.",
            "Numeric tolerance is a candidate derived from displayed gold precision, not an approved acceptance threshold.",
            "Unit, date precision, choice parsing, and composite fields require question-level review where flagged.",
        ],
    }
    atomic_write_json(output_path, result)
    print(f"Generated {len(rules)} draft rules: {output_path.resolve()}")


if __name__ == "__main__":
    main()
