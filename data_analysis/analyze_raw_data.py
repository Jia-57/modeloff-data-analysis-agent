"""Summarize raw data by project, using only the Python standard library."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import statistics
import sys
import xml.etree.ElementTree as ET
from zipfile import ZipFile


BASE_DIR = Path(__file__).resolve().parents[1]
EXCEL_SUFFIXES = {".xlsx", ".xlsm", ".xlsb"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff", ".webp", ".svg"}
QUESTION_PATTERN = re.compile(r"question(\d+)\.txt", re.IGNORECASE)
METRICS = (
    ("#Questions", "question_count"),
    ("Len_Intro", "introduction_word_count"),
    ("Len_Que", "question_word_count"),
    ("#Excel", "excel_count"),
    ("Excel Size (KB)", "excel_size_kb"),
    ("#Image", "image_count"),
    ("#Sheets", "sheet_count"),
)


def read_relationships(archive, member):
    """Read OOXML relationships without needing an Excel reader dependency."""
    return [dict(element.attrib) for element in ET.fromstring(archive.read(member))]


def workbook_statistics(path):
    """Count actual worksheet relationships, including hidden sheets and XLSB."""
    with ZipFile(path) as archive:
        workbook_name = "workbook.bin" if path.suffix.lower() == ".xlsb" else "workbook.xml"
        relationships = read_relationships(archive, "xl/_rels/" + workbook_name + ".rels")
        sheet_count = sum(rel["Type"].rsplit("/", 1)[-1] == "worksheet" for rel in relationships)
        return {
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "size_kb": path.stat().st_size / 1024,
            "sheet_count": sheet_count,
        }


def analyze_project(project_dir):
    files = sorted(
        (path for path in project_dir.rglob("*")
         if path.is_file() and not any(part.startswith(".") for part in path.relative_to(project_dir).parts)),
        key=lambda path: path.relative_to(project_dir).as_posix(),
    )
    unsupported = [path for path in files if path.suffix.lower() == ".xls"]
    if unsupported:
        raise ValueError("Legacy .xls files require an additional reader: " + str(unsupported))
    introductions = [path for path in files if path.name.lower() == "introduction.txt"]
    questions = sorted(
        (path for path in files if QUESTION_PATTERN.fullmatch(path.name)),
        key=lambda path: (int(QUESTION_PATTERN.fullmatch(path.name).group(1)), path.as_posix()),
    )
    intro_details = [
        {"path": path.relative_to(project_dir).as_posix(),
         "word_count": len(path.read_text(encoding="utf-8-sig").split())}
        for path in introductions
    ]
    question_details = [
        {"path": path.relative_to(project_dir).as_posix(),
         "question_number": int(QUESTION_PATTERN.fullmatch(path.name).group(1)),
         "word_count": len(path.read_text(encoding="utf-8-sig").split())}
        for path in questions
    ]
    workbooks = []
    for path in files:
        if path.suffix.lower() in EXCEL_SUFFIXES:
            detail = workbook_statistics(path)
            detail["path"] = path.relative_to(project_dir).as_posix()
            workbooks.append(detail)
    images = [path.relative_to(project_dir).as_posix()
              for path in files if path.suffix.lower() in IMAGE_SUFFIXES]
    warnings = []
    if not introductions:
        warnings.append("Missing introduction.txt; introduction length recorded as zero.")
    for detail in intro_details:
        if detail["word_count"] == 0:
            warnings.append("Empty introduction: " + detail["path"])
    return {
        "project_id": project_dir.name,
        "question_count": len(questions),
        "introduction_word_count": sum(detail["word_count"] for detail in intro_details),
        "question_word_count": sum(detail["word_count"] for detail in question_details),
        "excel_count": len(workbooks),
        "excel_size_bytes": sum(detail["size_bytes"] for detail in workbooks),
        "excel_size_kb": sum(detail["size_bytes"] for detail in workbooks) / 1024,
        "image_count": len(images),
        "sheet_count": sum(detail["sheet_count"] for detail in workbooks),
        "file_counts_by_extension": dict(sorted(Counter(path.suffix.lower() for path in files).items())),
        "introductions": intro_details,
        "questions": question_details,
        "workbooks": workbooks,
        "images": images,
        "warnings": warnings,
    }


def describe(values):
    return {"mean": statistics.mean(values), "max": max(values),
            "min": min(values), "total": sum(values)}


def analyze(raw_dir):
    if not raw_dir.is_dir():
        raise ValueError("Raw data directory does not exist: " + str(raw_dir))
    project_dirs = sorted(path for path in raw_dir.iterdir()
                          if path.is_dir() and not path.name.startswith("."))
    if not project_dirs:
        raise ValueError("No project directories found in " + str(raw_dir))
    projects = [analyze_project(path) for path in project_dirs]
    question_lengths = [question["word_count"] for project in projects for question in project["questions"]]
    summary = {
        label: describe(question_lengths) if label == "Len_Que" else
        describe([project[key] for project in projects])
        for label, key in METRICS
    }
    return {
        "schema_version": 2,
        "raw_data_dir": str(raw_dir.resolve()),
        "total_projects": len(projects),
        "total_questions": len(question_lengths),
        "definitions": {
            "aggregation_unit": "Len_Que: individual question; all other metrics: project; each visible immediate subdirectory is one project",
            "mean_denominator": {
                label: "total_questions" if label == "Len_Que" else "total_projects"
                for label, _ in METRICS
            },
            "text_length": "UTF-8 text split on whitespace; all content including headings and answer choices is counted",
            "Len_Intro": "total introduction.txt word count in each project",
            "Len_Que": "word count of each individual questionN.txt; Mean, Max and Min across all individual questions, Total across all question words",
            "#Questions": "number of files matching question[0-9]+.txt, not the largest question number",
            "#Excel": "all .xlsx, .xlsm and .xlsb files, including auxiliary workbooks",
            "Excel Size (KB)": "sum of Excel file sizes in bytes / 1024 per project",
            "#Image": "standalone image files; excludes images embedded in Excel",
            "#Sheets": "all Excel worksheets, including hidden worksheets; excludes chart sheets",
            "ignored_files": "hidden files and hidden directories, including .DS_Store",
        },
        "summary": summary,
        "projects": projects,
    }


def render_table(result):
    rows = [["Metric", "Mean", "Max", "Min", "Total"]]
    for label, _ in METRICS:
        metric = result["summary"][label]
        values = [f"{metric['mean']:,.2f}"]
        for key in ("max", "min", "total"):
            values.append(f"{metric[key]:,.2f}" if label == "Excel Size (KB)" else f"{metric[key]:,}")
        rows.append([label] + values)
    widths = [max(len(row[i]) for row in rows) for i in range(5)]
    lines = ["  ".join(value.ljust(widths[i]) if i == 0 else value.rjust(widths[i])
                       for i, value in enumerate(row)) for row in rows]
    lines.insert(1, "  ".join("-" * width for width in widths))
    return "Total projects: " + str(result["total_projects"]) + "\n" + "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=BASE_DIR / "data" / "raw_data")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "data_analysis" / "data_statistics.json")
    args = parser.parse_args()
    try:
        result = analyze(args.raw_dir)
    except (OSError, ValueError, ET.ParseError) as error:
        parser.exit(1, "Statistics failed: " + str(error) + "\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(render_table(result))
    print("\nJSON saved to: " + str(args.output.resolve()))
    for project in result["projects"]:
        for warning in project["warnings"]:
            print("Warning [" + project["project_id"] + "]: " + warning, file=sys.stderr)


if __name__ == "__main__":
    main()
