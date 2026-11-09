"""Check computed evidence and question options without reading evaluation labels."""

from __future__ import annotations

import math
import re
from decimal import Decimal
from pydantic import Field

from .schema import Record
from .types import Answer, FinalConsistencyCheck, JsonValue

# Unicode escapes preserve recognition of non-English answer text without
# embedding non-English literals in the published source file.


class ComputedResult(Record):
    """A value emitted by a successfully completed Python computation."""

    value: JsonValue
    unit: str | None = None
    checks: dict[str, bool] = Field(default_factory=dict)


class AnswerValidationError(ValueError):
    """A result mismatch that may require computation rather than JSON repair."""


def question_options(question: str) -> dict[str, str]:
    """Extract an ordered A-D style option list, rejecting partial or duplicate labels."""
    matches = list(re.finditer(r"(?:^|\s)([A-Za-z])[.)]\s+", question))
    labels = [match[1].upper() for match in matches]
    if len(labels) < 2 or labels != list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(labels)]):
        return {}
    return {match[1].upper(): question[match.end():matches[index + 1].start() if index + 1 < len(matches) else len(question)].strip()
            for index, match in enumerate(matches)}


def needs_computation(question: str) -> bool:
    """Identify explicit calculation requests, including numeric multiple-choice questions."""
    options = question_options(question)
    return bool(re.search(r"\b(calculate|compute|simulate|average|probability|total|cumulative|forecast|revenue|sum)\b",
                          question, re.IGNORECASE) or
                options and all(re.search(r"\d", text) for text in options.values()))


def _number(text: str, *, directional: bool = False) -> float | None:
    """Parse a single numeric option, preserving thousands separators and scale suffixes."""
    text = re.sub(r"(?<=\d)\s*\.\s*(?=\d)", ".", text).replace(",", "")
    matches = list(re.finditer(r"[-+]?\d+(?:\.\d+)?", text))
    if len(matches) != 1:
        return None
    match = matches[0]
    suffix = text[match.end():]
    scale = (1e6 if re.match(r"\s*(?:m\b|million\b)", suffix, re.IGNORECASE) else
             1e3 if re.match(r"\s*(?:k\b|thousand\b)", suffix, re.IGNORECASE) else 1)
    value = float(match[0]) * scale
    if directional and re.search(r'\b(decrease|decreasing|reduction|decline)\b|\u51cf\u5c11|\u4e0b\u964d', text, re.IGNORECASE):
        value = -abs(value)
    elif directional and re.search(r'\b(increase|increasing)\b|\u589e\u52a0|\u4e0a\u5347', text, re.IGNORECASE):
        value = abs(value)
    return value


def _directional_options(options: dict[str, str], value: float) -> bool:
    texts=' '.join(options.values())
    return value<0 or bool(re.search(r'\bincrease\b|\u589e\u52a0',texts,re.IGNORECASE)
                           and re.search(r'\b(decrease|reduction|decline)\b|\u51cf\u5c11|\u4e0b\u964d',texts,re.IGNORECASE))


def _scaled_value(value: float, unit: str | None, percent: bool) -> float:
    """Normalize explicitly stated percent/fraction and currency scale conventions."""
    normalized = (unit or "").lower().strip()
    if percent:
        if normalized in ("fraction", "probability", "ratio"):
            return value * 100
        if normalized not in ("%", "percent", "percentage"):
            raise AnswerValidationError("Percent choices require an explicit % or fraction computed-result unit")
    if "million" in normalized:
        value *= 1e6
    elif "thousand" in normalized:
        value *= 1e3
    return value


def _unit_family(unit: str | None) -> str:
    """Compare unit identity while allowing explicit magnitude and probability conversions."""
    text = re.sub(r"\b(millions?|thousands?)\b", "", (unit or "").lower()).strip()
    if text in ("%", "percent", "percentage", "fraction", "probability", "ratio"):
        return "%"
    text = text.replace(" ", "")
    singular = {"flights": "flight", "rolls": "roll", "days": "day", "years": "year",
                "months": "month", "units": "unit", "vehicles": "vehicle", "teams": "team",
                "people": "person", "persons": "person"}
    return singular.get(text, text)


def _choice_unit(text: str) -> str | None:
    """Read a simple explicit suffix unit; leave prose, ranges and currency symbols ambiguous."""
    match = re.fullmatch(r"\s*[-+]?\d[\d,]*(?:\.\d+)?\s+([A-Za-z%]+(?:/[A-Za-z]+)?)\s*", text)
    if match and match[1].lower() not in ("million", "thousand"):
        return match[1]
    prefix = re.fullmatch(r"\s*([A-Za-z]+(?:/[A-Za-z]+)?)\s*\$?\s*[-+]?\d[\d,]*(?:\.\d+)?\s*", text)
    return prefix[1] if prefix else None


def computed_option(question: str, result: ComputedResult, *, context: str = "") -> str | None:
    """Map a computed number to ranges or the nearest numeric choice without gold answers."""
    options = question_options(question)
    if not options or isinstance(result.value, bool) or not isinstance(result.value, (int, float)):
        return None
    percent = all("%" in text for text in options.values())
    units = {_unit_family(unit) if unit is not None else None
             for unit in (_choice_unit(text) for text in options.values())}
    if len(units - {None}) > 1:
        raise AnswerValidationError("Numeric options have different quantity kinds or units")
    if len(units) == 1 and None not in units and result.unit is not None and not percent:
        if _unit_family(next(iter(units))) != _unit_family(result.unit):
            raise AnswerValidationError("Computed unit conflicts with the numeric option unit")
    value = _scaled_value(float(result.value), result.unit, percent)
    ranges, points = [], {}
    for label, text in options.items():
        clean = text.replace("–", "-").replace("−", "-")
        range_match = re.search(r"(\d+(?:\.\d+)?)\s*%?\s*-\s*(\d+(?:\.\d+)?)", clean)
        bound = re.search(r"(?:less than|more than|[<>])\s*(\d+(?:\.\d+)?)", clean, re.IGNORECASE)
        if range_match:
            if float(range_match[1]) <= value <= float(range_match[2]):
                ranges.append(label)
        elif bound:
            lower = re.search(r"more than|>", clean, re.IGNORECASE)
            within_bound = value > float(bound[1]) if lower else value < float(bound[1])
            if within_bound:
                ranges.append(label)
        else:
            number = _number(clean, directional=_directional_options(options, value))
            if number is None:
                return None
            points[label] = number
    if points and len(points) == len(options):
        distances = {label: abs(number - value) for label, number in points.items()}
        winners = [label for label, distance in distances.items() if math.isclose(distance, min(distances.values()), abs_tol=1e-12)]
        if len(winners) == 1:
            winner = winners[0]
            approximate = re.search(r"closest|approximate|approximately", question + " " + context, re.IGNORECASE)
            normalized = re.sub(r"(?<=\d)\s*\.\s*(?=\d)", ".", options[winner]).replace(",", "")
            token = re.search(r"\d+(?:\.(\d+))?", normalized)
            scale = 1e6 if re.search(r"\d\s*(?:m\b|million\b)", normalized, re.IGNORECASE) else 1
            tolerance = .5 * 10 ** -len(token[1] or "") * scale
            if approximate or distances[winner] <= tolerance + 1e-10:
                return winner
    elif len(ranges) == 1:
        return ranges[0]
    raise AnswerValidationError("Computed value does not select one unambiguous question option")


def _selection_label(value: JsonValue, options: dict[str, str]) -> str | None:
    """Recognize a single supplied option label without interpreting arbitrary answer prose."""
    if not isinstance(value, str) or not options:
        return None
    labels = "|".join(re.escape(label) for label in options)
    match = re.fullmatch(r"\s*(?:(?:option|choice|\u9009\u9879)\s*)?(" + labels + r")[.)]?\s*", value, re.IGNORECASE)
    return match[1].upper() if match else None


def _explicit_option_claims(text: str, options: dict[str, str]) -> list[str]:
    """Find affirmative option claims as review hints, ignoring negated or hypothetical clauses."""
    if not options:
        return []
    labels = "(" + "|".join(re.escape(label) for label in options) + ")"
    patterns = [
        r"\b(?:matches?|matching|corresponds? to|closest to|choose|select|that['’]s|that is|therefore|thus)\s+(?:(?:the\s+)?(?:option|choice)\s+)?" + labels + r"\b",
        r"\b(?:option|choice)\s+" + labels + r"\s+(?:is|would be)\s+(?:the\s+)?(?:correct|answer|closest|matching)\b",
        r"(?:\u5e94\u9009|\u9009\u62e9|\u5bf9\u5e94|\u5339\u914d|\u7b54\u6848(?:\u662f|\u4e3a))\s*(?:\u9009\u9879\s*)?" + labels + r"\b",
    ]
    claims = []
    for clause in re.split(r"(?<=[!?\u3002\uff01\uff1f;\uff1b])\s*|\.(?!\d)\s+", text):
        if clause.rstrip().endswith(("?", "\uff1f")) or re.search(r"\b(?:if|would|could|might|suppose|example|hypothetical)\b|\u5982\u679c|\u5047\u8bbe|\u4f8b\u5982", clause, re.IGNORECASE):
            continue
        for pattern in patterns:
            for match in re.finditer(pattern, clause, re.IGNORECASE):
                prefix = clause[max(0, match.start() - 24):match.start()]
                suffix = clause[match.end():match.end() + 24]
                if re.search(r"\b(?:not|never|don't|do not|cannot|can't)\s*$|(?:\u4e0d|\u4e0d\u662f|\u5e76\u975e)\s*$", prefix, re.IGNORECASE):
                    continue
                if re.match(r"\s+(?:is\s+)?(?:not|incorrect|wrong)\b|\s*(?:\u4e0d\u5bf9|\u9519\u8bef)", suffix, re.IGNORECASE):
                    continue
                claims.append(match[1].upper())
    return list(dict.fromkeys(claims))


def _normalized_metric(value: float, unit: str) -> tuple[float, float]:
    """Normalize only explicit probability or magnitude conventions and retain their scale."""
    family = _unit_family(unit)
    factor = _scaled_value(1.0, unit, family == "%")
    return value * factor, factor


def assess_final_consistency(question: str, answer: Answer, computed_results: list[dict] | None = None,
                             *, context: str = "") -> FinalConsistencyCheck:
    """Audit number, unit, option and explanation agreement without gold or answer mutation."""
    if answer.answer is None:
        return FinalConsistencyCheck()
    options = question_options(question)
    selected = _selection_label(answer.answer.value, options)
    claims = _explicit_option_claims(answer.brief_explanation, options)
    conflicts, unresolved, agreements, refs = [], [], [], []
    if selected and claims:
        if any(label != selected for label in claims):
            conflicts.append(f"Answer selects {selected}, but the explanation explicitly claims {', '.join(claims)}")
        else:
            agreements.append("Explanation and selected option agree")
    rows = [row for row in (computed_results or [])
            if isinstance(row.get("value"), (int, float)) and not isinstance(row.get("value"), bool)
            and math.isfinite(row["value"]) and all(row.get("checks", {}).values())]
    unique = {(row["value"], row.get("unit")) for row in rows}
    tool = None
    if len(unique) == 1:
        tool = ComputedResult.model_validate({key: rows[-1][key] for key in ("value", "unit", "checks") if key in rows[-1]})
        refs = list(dict.fromkeys(row["evidence_ref"] for row in rows if row.get("evidence_ref")))
    elif len(unique) > 1:
        unresolved.append("Multiple tool metrics are available; the current metric is not unambiguous")
    model = (ComputedResult(value=answer.computed_value, unit=answer.computed_unit)
             if answer.computed_value is not None and math.isfinite(answer.computed_value) else None)
    if tool and model:
        if not tool.unit or not model.unit or _unit_family(tool.unit) != _unit_family(model.unit):
            unresolved.append("Tool and reported metric units do not establish one comparable quantity")
        else:
            actual, _ = _normalized_metric(float(tool.value), tool.unit)
            reported, scale = _normalized_metric(float(model.value), model.unit)
            precision = Decimal(str(model.value)).as_tuple().exponent
            tolerance = .5 * 10 ** precision * scale
            if not math.isclose(actual, reported, rel_tol=1e-6, abs_tol=max(1e-10, tolerance)):
                conflicts.append("Reported metric differs from the tool metric after explicit unit conversion and rounding")
            else:
                agreements.append("Tool and reported metric agree")
    metric = tool if tool is not None else model if not unique else None
    expected = None
    if metric is not None and options:
        explicit_units = {unit for text in options.values() if (unit := _choice_unit(text)) is not None}
        if metric.unit is None and (explicit_units or any("%" in text for text in options.values())):
            unresolved.append("Numeric option matching lacks an explicit metric unit")
        else:
            try:
                expected = computed_option(question, metric, context=context)
                if expected and selected:
                    if expected != selected:
                        conflicts.append(f"Metric maps to option {expected}, while the answer selects {selected}; verify metric and unit metadata")
                    else:
                        agreements.append("Metric and selected option agree")
            except AnswerValidationError as error:
                unresolved.append(str(error))
    status = "conflict" if conflicts else "unresolved" if unresolved else "consistent" if agreements else "not_checked"
    return FinalConsistencyCheck(status=status, reasons=list(dict.fromkeys(conflicts + unresolved)),
                                 evidence_refs=refs, option_claims=claims, expected_option=expected)


def validate_result_value(question: str, value: JsonValue, unit: str | None, result: ComputedResult, *, context: str = "") -> JsonValue:
    """Require the final value to agree with computation, allowing numeric choice rounding."""
    if isinstance(value, (int, float)) and unit is not None and _unit_family(unit) != _unit_family(result.unit):
        raise AnswerValidationError("Final unit conflicts with the computed-result unit")
    expected = computed_option(question, result, context=context)
    if expected:
        label = re.match(r"^\s*([A-Za-z])(?:[.)\s]|$)", value) if isinstance(value, str) else None
        if label:
            if label[1].upper() != expected:
                raise AnswerValidationError("Selected option conflicts with the emitted computed result; reconsider the choice")
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            percent = all("%" in text for text in question_options(question).values())
            final_number = _scaled_value(float(value), unit or result.unit, percent)
            actual_number = _scaled_value(float(result.value), result.unit, percent)
            option_number = _number(question_options(question)[expected],
                                    directional=_directional_options(question_options(question),actual_number))
            if not math.isclose(final_number, actual_number, rel_tol=1e-8, abs_tol=1e-8) and (
                option_number is None or not math.isclose(final_number, option_number, rel_tol=1e-8, abs_tol=1e-8)):
                raise AnswerValidationError("Final numeric value conflicts with the computed value and selected option")
        else:
            raise AnswerValidationError("Return the computed question option letter as the final value")
        return label[1].upper() if label else value
    if isinstance(value, bool) != isinstance(result.value, bool) or value != result.value or (
            unit is not None and _unit_family(unit) != _unit_family(result.unit)):
        raise AnswerValidationError("Final value/unit must match the emitted computed result")
    return value
