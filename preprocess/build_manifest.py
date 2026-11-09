"""Build a reproducible task manifest and a separate gold-label file.

The source tree is read-only. Outputs are written to data/preprocessed/ by
default. The answer source is line-delimited JSON despite its .json suffix.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "1.4.0"
QUESTION_RE = re.compile(r"^question(\d+)\.txt$", re.IGNORECASE)
IGNORED_NAMES = {".DS_Store", "Thumbs.db", "desktop.ini"}
SOLUTION_SUFFIXES = {".ipynb"}
# Freeze exclusions before evaluation; archives use the same eligible tasks.
EXCLUDED_CHALLENGES = {
    "00000007": "Incorrect workbook supplied for this challenge; excluded from training/task inputs and evaluation at the user's request.",
    "00000010": "Missing the 380-match dataset required by the questions; only Check_Sum is available.",
    "00000043": "Missing the debt-calculation worksheets required by the questions; only a small example sheet is available.",
}


def read_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected one JSON object per line")
            rows.append((line_no, value))
    return rows


def relpath(path: Path, project_root: Path) -> str:
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def asset_role(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm", ".xlsb"}:
        return "workbook"
    if suffix == ".csv":
        return "table"
    if suffix in {".jpg", ".jpeg", ".png", ".webp"}:
        return "image"
    return "supplemental"


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


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


def build(project_root: Path, raw_data: Path, answers_path: Path, output_dir: Path) -> dict[str, Any]:
    """Build matching active inputs/labels and an archive of the eligible records.
    Args: project_root: Path base; raw_data: Inputs; answers_path: Gold source;
        output_dir: Destination for active records, archives, and the report.
    """
    answer_rows = read_jsonl(answers_path)
    challenge_by_id: dict[str, tuple[int, dict[str, Any]]] = {}
    for line_no, row in answer_rows:
        challenge_id = str(row.get("id", ""))
        if not challenge_id:
            raise ValueError(f"{answers_path}:{line_no}: missing challenge id")
        if challenge_id in challenge_by_id:
            raise ValueError(f"duplicate challenge id: {challenge_id}")
        questions, gold = row.get("questions"), row.get("answers")
        if not isinstance(questions, list) or not isinstance(gold, list):
            raise ValueError(f"{answers_path}:{line_no}: questions/answers must be arrays")
        if not all(isinstance(question_id, str) for question_id in questions):
            raise ValueError(f"{answers_path}:{line_no}: question ids must be strings")
        if len(questions) != len(gold):
            raise ValueError(
                f"{answers_path}:{line_no}: {challenge_id} has "
                f"{len(questions)} questions but {len(gold)} answers"
            )
        if len(set(questions)) != len(questions):
            raise ValueError(f"{answers_path}:{line_no}: duplicate question id in {challenge_id}")
        challenge_by_id[challenge_id] = (line_no, row)

    if not raw_data.is_dir():
        raise FileNotFoundError(f"raw data directory not found: {raw_data}")
    source_challenges = {p.name for p in raw_data.iterdir() if p.is_dir()}
    listed_challenges = set(challenge_by_id)
    if source_challenges != listed_challenges:
        missing_dirs = sorted(listed_challenges - source_challenges)
        extra_dirs = sorted(source_challenges - listed_challenges)
        raise ValueError(f"challenge directory mismatch; missing={missing_dirs}, unlisted={extra_dirs}")

    manifest: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []

    def file_record(path: Path, role: str) -> dict[str, Any]:
        return {
            "role": role,
            "path": relpath(path, project_root),
        }

    for challenge_id, (answer_line, challenge) in challenge_by_id.items():
        challenge_dir = raw_data / challenge_id
        intro_path = challenge_dir / "introduction.txt"
        if not intro_path.is_file():
            raise FileNotFoundError(f"missing introduction: {intro_path}")

        asset_paths: list[Path] = []
        for path in sorted(challenge_dir.iterdir(), key=lambda p: p.name.casefold()):
            if not path.is_file() or path.name in IGNORED_NAMES or path.name.startswith("."):
                continue
            if path.name == "introduction.txt" or QUESTION_RE.match(path.name):
                continue
            if path.suffix.lower() in SOLUTION_SUFFIXES:
                excluded.append({"challenge_id": challenge_id, "path": relpath(path, project_root),
                                 "reason": "notebook excluded from task inputs; review as possible solution material"})
                continue
            asset_paths.append(path)

        actual_questions = {
            match.group(1): path
            for path in challenge_dir.iterdir()
            if path.is_file() and (match := QUESTION_RE.match(path.name))
        }
        question_ids = [str(q) for q in challenge["questions"]]
        for question_id in question_ids:
            if not re.fullmatch(r"question\d+", question_id, re.IGNORECASE):
                raise ValueError(f"unsupported question id {question_id!r} in {challenge_id}")
        expected_names = {q.casefold() for q in question_ids}
        actual_names = {p.stem.casefold() for p in actual_questions.values()}
        if expected_names != actual_names:
            raise ValueError(
                f"question file mismatch for {challenge_id}; "
                f"missing={sorted(expected_names - actual_names)}, "
                f"unlisted={sorted(actual_names - expected_names)}"
            )

        intro_record = file_record(intro_path, "introduction")
        assets = [file_record(path, asset_role(path)) for path in asset_paths]
        for question_index, (question_id, gold) in enumerate(zip(challenge["questions"], challenge["answers"])):
            question_path = challenge_dir / f"{question_id}.txt"
            if not question_path.is_file():
                # Preserve the actual case of the file if its spelling differs.
                question_path = next(p for p in actual_questions.values()
                                     if p.stem.casefold() == question_id.casefold())
            question_record = file_record(question_path, "question")
            task_id = f"{challenge_id}/{question_id}"
            manifest.append({
                "task_id": task_id,
                "challenge_id": challenge_id,
                "question_id": question_id,
                "challenge_name": challenge.get("name"),
                "year": challenge.get("year"),
                "question_file": question_record,
                "introduction_file": intro_record,
                "allowed_assets": assets,
            })
            labels.append({
                "task_id": task_id,
                "gold_raw": gold,
                "gold_ref": {
                    "source_file": relpath(answers_path, project_root),
                    "source_line": answer_line,
                    "answer_index": question_index,
                },
            })

    # Keep task ordering deterministic and make accidental duplicate IDs fatal.
    task_ids = [row["task_id"] for row in manifest]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("duplicate task_id generated")

    source_task_count = len(manifest)
    excluded_challenges = [
        {"challenge_id": challenge_id, "reason": reason,
         "task_ids": [task["task_id"] for task in manifest
                      if task["challenge_id"] == challenge_id]}
        for challenge_id, reason in EXCLUDED_CHALLENGES.items()
        if challenge_id in challenge_by_id
    ]
    manifest = [task for task in manifest if task["challenge_id"] not in EXCLUDED_CHALLENGES]
    active_ids = {task["task_id"] for task in manifest}
    labels = [label for label in labels if label["task_id"] in active_ids]

    # Archives mirror the active dataset; archived labels remain host-only.
    archive_dir = output_dir / "source_archive"
    atomic_write_jsonl(archive_dir / "manifest.jsonl", manifest)
    atomic_write_jsonl(archive_dir / "labels.jsonl", labels)

    manifest_path = output_dir / "task_inputs" / "manifest.jsonl"
    labels_path = output_dir / "evaluation_labels" / "labels.jsonl"
    report_path = output_dir / "preprocessing_report.json"
    atomic_write_jsonl(manifest_path, manifest)
    atomic_write_jsonl(labels_path, labels)
    report = {
        "schema_version": "1.0",
        "script_version": SCRIPT_VERSION,
        "source_answers": {
            "path": relpath(answers_path, project_root),
            "format": "JSON Lines (one object per non-empty line)",
        },
        "source_raw_data": relpath(raw_data, project_root),
        "source_challenge_count": len(challenge_by_id),
        "source_task_count": source_task_count,
        "challenge_count": len({task["challenge_id"] for task in manifest}),
        "task_count": len(manifest),
        "excluded_task_count": source_task_count - len(manifest),
        "excluded_challenges": excluded_challenges,
        "exclusion_policy": [
            {"challenge_id": challenge_id, "reason": reason}
            for challenge_id, reason in EXCLUDED_CHALLENGES.items()
        ],
        "source_archive": {
            "manifest": relpath(archive_dir / "manifest.jsonl", project_root),
            "labels": relpath(archive_dir / "labels.jsonl", project_root),
        },
        "excluded_files": excluded,
        "notes": [
            "Source files were read only; task inputs are referenced by project-relative paths.",
            "Gold values are stored separately from the task manifest and must remain inaccessible to solving agents.",
            "Active task inputs and labels use the same predeclared challenge exclusions.",
            "Source archives mirror the active task manifest and labels; they are for host-side auditing only, not solving or scoring.",
            "Exclusion policy is retained even when excluded projects have been removed from the source directories and answer file.",
            "All non-hidden, non-question, non-introduction files are treated as allowed assets except Jupyter notebooks.",
        ],
    }
    atomic_write_json(report_path, report)
    return report


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--raw-data", type=Path, default=project_root / "data/raw_data")
    parser.add_argument("--answers", type=Path, default=project_root / "data/answer/data.json")
    parser.add_argument("--output-dir", type=Path, default=project_root / "data/preprocessed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    raw_data = args.raw_data if args.raw_data.is_absolute() else root / args.raw_data
    answers = args.answers if args.answers.is_absolute() else root / args.answers
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    report = build(root, raw_data.resolve(), answers.resolve(), output_dir.resolve())
    print(
        f"Generated {report['task_count']} tasks from {report['challenge_count']} challenges.\n"
        f"Manifest: {output_dir.resolve() / 'task_inputs' / 'manifest.jsonl'}\n"
        f"Evaluation labels: {output_dir.resolve() / 'evaluation_labels' / 'labels.jsonl'}\n"
        f"Report: {output_dir.resolve() / 'preprocessing_report.json'}"
    )


if __name__ == "__main__":
    main()
