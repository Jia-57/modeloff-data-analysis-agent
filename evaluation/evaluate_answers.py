"""Offline, provisional scoring of frozen answers; never called by the solver.

Uses explicit format normalization, not an LLM judge. Generated answer rules
remain drafts; this report is an initial analysis, not an approved benchmark.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import unicodedata
from collections import Counter
from decimal import Decimal
from pathlib import Path

from generate_answer_rules import parse_date, parse_number, read_jsonl


def text(value):
    return unicodedata.normalize("NFC", str(value)).strip()


def indexed(rows, source):
    result = {row["task_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate task_id in {source}")
    return result


def number_parts(value, unit=None):
    parsed = parse_number(value)
    if not parsed:
        return None
    number, embedded, _ = parsed
    embedded = embedded or {}
    suffix = (unit or "").strip().lower()
    scale = embedded.get("scale")
    unit_scale = next((v for k, v in [("billion", "billion"), ("million", "million"),
                                     ("thousand", "thousand")] if k in suffix), None)
    if suffix in ("k", "m", "b"):
        unit_scale = {"k": "thousand", "m": "million", "b": "billion"}[suffix]
    if scale and unit_scale and scale != unit_scale:
        return None
    scale = scale or unit_scale
    symbol = embedded.get("symbol")
    unit_symbol = None
    if suffix in ("%", "percent", "percentage"):
        unit_symbol = "%"
    elif re.search(r"usd|dollar|\$", suffix):
        unit_symbol = "$"
    elif re.search(r"eur|euro|€", suffix):
        unit_symbol = "€"
    elif re.search(r"gbp|pound|£", suffix):
        unit_symbol = "£"
    if symbol and unit_symbol and symbol != unit_symbol:
        return None
    symbol = symbol or unit_symbol
    recognized = not suffix or unit_scale or unit_symbol or suffix in ("fraction", "ratio", "probability")
    if not recognized:
        return None
    return number, symbol, scale, suffix in ("fraction", "ratio", "probability")


def compare_numeric(pred, gold, rule, unit, question):
    actual, expected = number_parts(pred, unit), number_parts(gold)
    if not actual or not expected:
        return False, "numeric_format_or_unit_mismatch"
    av, au, scale, fraction = actual
    gv, gu, gold_scale, _ = expected
    if au and au != gu:
        return False, "numeric_unit_mismatch"
    if fraction:
        if gu != "%":
            return False, "numeric_unit_mismatch"
        av *= 100
    multipliers = {None: Decimal(1), "thousand": Decimal(1000),
                   "million": Decimal(1000000), "billion": Decimal(1000000000)}
    # A bare answer may use the displayed unit explicitly required by the question.
    # Otherwise a scale-bearing gold and a bare prediction require manual review.
    if gold_scale and not scale and not unit and au is None:
        hint = {"thousand": r"\bthousands?\b|\$k\b", "million": r"\bmillions?\b|\$m\b",
                "billion": r"\bbillions?\b|\$b\b"}[gold_scale]
        if not re.search(hint, question, re.I):
            return None, "ambiguous_numeric_scale"
        scale = gold_scale
    tolerance = Decimal(str((rule.get("tolerance_candidate") or {}).get("value", "0")))
    correct = abs(av * multipliers[scale] - gv * multipliers[gold_scale]) <= tolerance * multipliers[gold_scale]
    return correct, "numeric_display_precision" if correct else "numeric_mismatch"


def compare(pred, gold, rule, *, unit=None, task_id="", question=""):
    kind = rule["answer_kind"]
    if kind == "choice":
        match = re.fullmatch(r"(?:(?:option|choice)\s+)?([A-I])[.)]?", text(pred), re.I)
        correct = bool(match and match[1].upper() == text(gold).upper())
        return correct, "choice_match" if correct else "choice_mismatch"
    if kind == "numeric":
        return compare_numeric(pred, gold, rule, unit, question)
    if kind == "date":
        correct = parse_date(pred) is not None and parse_date(pred) == parse_date(gold)
        return correct, "date_match" if correct else "date_mismatch"
    if kind == "text":
        actual, expected = text(pred), text(gold)
        # Excel function-name questions accept an empty call, e.g. PI() = PI.
        if task_id.split("/")[0] == "00000042":
            actual = re.sub(r"\(\s*\)$", "", actual).upper()
            expected = expected.upper()
        correct = actual == expected
        return correct, "text_match" if correct else "text_mismatch"
    if kind == "composite":
        expected = gold
        if isinstance(gold, dict):
            expected = dict(gold)
            # Q is the source answer-sheet row number, not a requested answer.
            if expected.get("Q") == int(task_id.rsplit("question", 1)[-1]):
                expected.pop("Q")
            actual = pred
            # Dataset adapter: the question and introduction specify this order.
            fields = ["# of words", "Xth word", "Highest Word", "Highest Score"]
            if isinstance(pred, str) and set(expected) == set(fields) and all(
                marker in question for marker in ("# of words", "Xth word", "Highest Score")
            ):
                values = next(csv.reader(io.StringIO(pred)))
                if len(values) != len(fields):
                    return None, "unparsed_composite"
                actual = dict(zip(fields, [v.strip() for v in values]))
            if not isinstance(actual, dict) or not set(expected) <= set(actual):
                return None, "unparsed_composite"
            results = [compare(actual[k], v, rule["fields"][k], task_id=task_id, question=question)[0]
                       for k, v in expected.items()]
        elif isinstance(gold, list) and isinstance(pred, list) and len(gold) == len(pred):
            results = [compare(p, g, rule["fields"][str(i)], task_id=task_id, question=question)[0]
                       for i, (p, g) in enumerate(zip(pred, gold))]
        else:
            return None, "unparsed_composite"
        result = False if False in results else None if None in results else True
        return result, "composite_all_fields_match" if result else "composite_mismatch_or_review"
    return None, "unsupported_rule"


def summarize(rows):
    total = len(rows)
    answered = sum(r["answered"] for r in rows)
    correct = sum(r["correct"] is True for r in rows)
    return {"total": total, "answered": answered, "missing": total - answered,
            "correct": correct, "incorrect_answered": sum(r["answered"] and r["correct"] is False for r in rows),
            "needs_review": sum(r["correct"] is None for r in rows),
            "accuracy": correct / total if total else None,
            "answered_accuracy": correct / answered if answered else None,
            "answer_rate": answered / total if total else None,
            "raw_string_exact_correct": sum(r["raw_string_exact"] for r in rows),
            "terminal_statuses": dict(Counter(r["terminal_status"] for r in rows))}


def percent(value):
    return f"{value:.2%}" if value is not None else "—"


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--rules", type=Path)
    parser.add_argument("--active-manifest", type=Path,
                        help="Current eligible training/evaluation tasks; defaults to the project's active manifest")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve()
    run = args.run_dir.resolve()
    output = (args.output_dir or run / "evaluation").resolve()
    paths = {"manifest": run / "manifest.jsonl", "answers": run / "answers_A.jsonl",
             "active_manifest": args.active_manifest or root / "data/preprocessed/task_inputs/manifest.jsonl",
             "labels": args.labels or root / "data/preprocessed/evaluation_labels/labels.jsonl",
             "rules": args.rules or root / "data/preprocessed/evaluation_rules/answer_rules.json"}
    manifest = indexed(read_jsonl(paths["manifest"]), "manifest")
    active = indexed(read_jsonl(paths["active_manifest"]), "active manifest")
    excluded = [task for key, task in manifest.items() if key not in active]
    run_ids = set(manifest)
    manifest = {key: task for key, task in manifest.items() if key in active}
    if not manifest:
        raise ValueError("No eligible tasks in this run")
    answers = indexed(read_jsonl(paths["answers"]), "answers")
    labels = indexed(read_jsonl(paths["labels"]), "labels")
    rule_file = json.loads(paths["rules"].read_text())
    rules = rule_file["rules"]
    if set(answers) - run_ids or set(manifest) - set(labels) or set(manifest) - set(rules):
        raise ValueError("Task IDs do not align across manifest/answers/labels/rules")
    rows = []
    for key, task in manifest.items():
        row = answers.get(key, {})
        answer = row.get("answer") or {}
        value = answer.get("answer") or {}
        answered = answer.get("status") == "answered" and value.get("value") is not None
        pred, gold = value.get("value"), labels[key]["gold_raw"]
        question = (root / task["question_file"]["path"]).read_text(encoding="utf-8-sig")
        correct, reason = compare(pred, gold, rules[key], unit=value.get("unit"), task_id=key, question=question) if answered else (False, "missing_answer")
        rows.append({"task_id": key, "challenge_id": task["challenge_id"], "challenge_name": task["challenge_name"],
                     "answer_kind": rules[key]["answer_kind"], "terminal_status": row.get("terminal_status", "not_run"),
                     "answered": answered, "prediction": pred, "prediction_unit": value.get("unit"), "gold": gold,
                     "correct": correct, "reason": reason,
                     "raw_string_exact": answered and text(pred) == text(gold),
                     "rule_review_status": rules[key]["review_status"]})
    projects = []
    for cid in sorted({r["challenge_id"] for r in rows}):
        group = [r for r in rows if r["challenge_id"] == cid]
        projects.append({"challenge_id": cid, "challenge_name": group[0]["challenge_name"], **summarize(group)})
    report = {"evaluation_status": "provisional_offline_analysis", "variant": "A",
              "source_rule_status": rule_file["generation_status"],
              "scope": {"run_task_count": len(run_ids), "evaluated_task_count": len(manifest),
                        "excluded_task_count": len(excluded),
                        "excluded_challenge_ids": sorted({t["challenge_id"] for t in excluded}),
                        "excluded_task_ids": [t["task_id"] for t in excluded]},
              "policy": {"missing_answers": "incorrect; included in total denominator",
                         "composite": "all requested fields must match; Q is row metadata",
                         "text": "NFC and trim; Excel function names case-insensitive and empty parentheses removed",
                         "numeric": "Decimal comparison; draft half-last-displayed-digit tolerance; explicit unit scales",
                         "unresolved": "needs_review; included in total denominator without correctness credit"},
              "sources": {k: {"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                          for k, p in paths.items()},
              "overall": summarize(rows), "projects": projects,
              "macro_project_accuracy": sum(p["accuracy"] for p in projects) / len(projects),
              "by_answer_kind": {kind: summarize([r for r in rows if r["answer_kind"] == kind])
                                 for kind in sorted({r["answer_kind"] for r in rows})}}
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (output / "task_scores.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    for filename, records in [("per_project.csv", projects), ("task_scores.csv", rows)]:
        with (output / filename).open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                             for k, v in r.items()} for r in records)
    s = report["overall"]
    lines = ["# Preliminary offline evaluation of first-round answers", "",
             f"Historical run: {len(run_ids)} tasks; excluded from the current active dataset: "
             f"{len(excluded)} tasks (projects: "
             f"{', '.join(report['scope']['excluded_challenge_ids']) or 'none'}); "
             f"evaluated: {len(manifest)} tasks.", "",
             f"Accuracy over all tasks: **{s['correct']}/{s['total']} = {percent(s['accuracy'])}**; "
             f"accuracy among answered tasks: **{s['correct']}/{s['answered']} = "
             f"{percent(s['answered_accuracy'])}**.", "",
             f"Answered: {s['answered']}; unanswered: {s['missing']}; "
             f"answered incorrectly: {s['incorrect_answered']}; needs review: {s['needs_review']}.", "",
             f"Exact raw-string matches: {s['raw_string_exact_correct']}/{s['total']}; "
             f"additional correct answers after normalization: "
             f"{s['correct'] - s['raw_string_exact_correct']}.", "",
             "Scoring: tasks have equal weight; unanswered tasks count as incorrect; "
             "composite answers must match in every requested field. Q is question-number metadata. "
             "Numeric comparisons use draft precision tolerances; text is stripped; Excel function names "
             "ignore case and empty-parentheses differences. Composite fields follow question order. "
             "No model judge is used. The rules are drafts, so these results are provisional.", "",
             "| Project ID | Project | Correct/total | Overall accuracy | Answered | Answered accuracy | Unanswered |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for p in projects:
        lines.append(f"| {p['challenge_id']} | {p['challenge_name']} | {p['correct']}/{p['total']} | "
                     f"{percent(p['accuracy'])} | {p['answered']} | {percent(p['answered_accuracy'])} | {p['missing']} |")
    lines += ["", "By answer type:", "",
              "| Type | Correct/total | Overall accuracy | Answered accuracy |",
              "|---|---:|---:|---:|"]
    for kind, group in report["by_answer_kind"].items():
        lines.append(f"| {kind} | {group['correct']}/{group['total']} | {percent(group['accuracy'])} | "
                     f"{percent(group['answered_accuracy'])} |")
    lines += ["", f"Mean project accuracy: {percent(report['macro_project_accuracy'])}.", "",
              f"Unanswered tasks ({s['missing']}) and incorrect answered tasks "
              f"({s['incorrect_answered']}) both reduce overall accuracy. "
              "Unanswered terminal statuses: " + ", ".join(
                  f"{k}={v}" for k, v in s['terminal_statuses'].items() if k != 'completed') + ".", "",
              "Additional tasks counted correct after normalization:", ""]
    lines += [f"- {r['task_id']}: {r['prediction']!r} -> {r['gold']!r} ({r['reason']})"
              for r in rows if r["correct"] and not r["raw_string_exact"]]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report["overall"], ensure_ascii=False, indent=2))
    print(f"Report: {output / 'report.md'}")


if __name__ == "__main__":
    main()
