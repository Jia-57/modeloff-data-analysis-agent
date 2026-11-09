"""Bounded task-level ReAct with short-lived messages and durable SQLite records."""

from __future__ import annotations

import json
from copy import deepcopy
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from time import monotonic
from typing import Callable, Optional

from pydantic import TypeAdapter

from .prompts import action_correction, final_review_messages, format_repair_messages, option_review, python_correction, python_messages, react_messages
from .answer_validation import AnswerValidationError, ComputedResult, assess_final_consistency, computed_option, needs_computation, question_options, validate_result_value
from .runtime import RuntimeStore
from .schema import action_candidates, json_candidates, parse_python_code, repair_action_closers, is_inspection_cell
from .scheduling import Scheduler
from .tools import PythonArguments, PythonTool, ToolRegistry, artifact_id, TableInspectArguments, ArtifactArguments, FileArguments
from .types import (
    Answer, AnswerValue, ArtifactRef, DependencyExport, FinalAction, GenerationResult,
    ModelAgentAction, Observation, PythonSelectionArguments, ReportedValue, TaskDispatch, ToolAction, UsageCounters,
)


class PhaseLimitError(RuntimeError):
    """A bounded termination whose status is retained in the frozen task outcome."""

    def __init__(self, message: str, status: str = "step_limit") -> None:
        """Attach the phase termination reason without claiming a final answer."""
        super().__init__(message)
        self.status = status


class ReActAgent:
    """Solve one dispatched A task; every attempted model call is recorded in SQLite."""

    def __init__(self, store: RuntimeStore, generate_fn: Callable[..., GenerationResult],
                 count_tokens_fn: Callable[[list[dict]], int], tokenizer=None) -> None:
        """Inject model generation/counting; no weights are loaded by the Agent."""
        self.store, self.generate_fn = store, generate_fn
        self.count_tokens_fn, self.tokenizer = count_tokens_fn, tokenizer
        self.action_adapter = TypeAdapter(ModelAgentAction)
        self._read_queues = {}

    def _parse_action(self, payload: dict, task_id: str):
        """Validate compact output and attach host metadata; accept strict legacy actions for callers."""
        if payload.get("action") == "final" and "answer" in payload:
            return FinalAction.model_validate(payload)
        action = self.action_adapter.validate_python(payload)
        if isinstance(action, ToolAction):
            return action
        answer = Answer(task_id=task_id,
                        answer=AnswerValue(value=action.value, unit=action.unit) if action.value is not None else None,
                        brief_explanation=action.brief_explanation, evidence_refs=action.evidence_refs,
                        reported_values=action.reported_values,
                        computed_value=action.computed_value, computed_unit=action.computed_unit,
                        computed_value_source="model" if action.computed_value is not None else None,
                        status="answered" if action.value is not None else "no_answer")
        return FinalAction(action="final", answer=answer, exports=action.exports)

    def _remaining(self, task_id: str, started: float) -> float:
        """Measure the entire phase, including model preparation and shared-GPU waits."""
        counter = self.store.load_state(task_id).counters["A"]
        return self.store.config.budget_A.max_seconds - counter.duration_seconds - (monotonic() - started)

    def _output_limit(self, call_kind: str) -> int:
        """Resolve per-stage output limits while preserving legacy configuration defaults."""
        model = self.store.config.model
        if call_kind == "format_repair":
            return model.format_repair_max_new_tokens or min(512, model.max_new_tokens)
        selected = model.python_max_new_tokens if call_kind == "python_code" else model.action_max_new_tokens
        return selected or model.max_new_tokens

    def _pack(self, base: list[dict], history: list[list[dict]], controls: dict) -> list[dict]:
        """Discard oldest complete turns while preserving fixed input and the newest observation."""
        omitted = 0
        while True:
            completion = {"required_action": "final", "completion_rule":
                          "Return action=final now with a one-sentence brief_explanation; put calculations "
                          "in computed_value/computed_unit and the choice or exact result in value. "
                          "If evidence is insufficient, use value=null. Do not request a tool."} if controls.get("final_required") else {}
            cue = {"role": "user", "content": json.dumps(
                {**controls, **completion, "omitted_old_turns": omitted}, ensure_ascii=False)}
            messages = base + [item for pair in history for item in pair] + [cue]
            count = self.count_tokens_fn(messages)
            if type(count) is not int or count < 0:
                raise ValueError("count_tokens_fn must return a nonnegative integer")
            kind = "python_code" if controls.get("generation_task") == "python_code" else "action"
            if count + self._output_limit(kind) <= self.store.config.model.context_limit:
                return messages
            if len(history) <= 1:
                raise PhaseLimitError("fixed context and latest observation exceed context_limit", "material_unresolved")
            history.pop(0)
            omitted += 1

    def _call(self, task_id: str, messages: list[dict], started: float,
              output_schema: Optional[dict] = None, *, call_kind: str = "action",
              parent_event_id: Optional[str] = None) -> tuple[GenerationResult, str]:
        """Record requests, raw responses, errors and counters for every inference attempt."""
        limits = self.store.config.budget_A
        state = self.store.load_state(task_id)
        counter = state.counters.setdefault("A", UsageCounters())
        remaining = self._remaining(task_id, started)
        if remaining <= 0:
            raise PhaseLimitError("A time budget exhausted", "timeout")
        if counter.model_calls >= limits.max_model_calls:
            raise PhaseLimitError("A model-call budget exhausted")
        reserve = min(limits.final_reserve_seconds, limits.max_seconds / 5) if call_kind in ("python_code", "format_repair", "final_review") else 0
        if self.store.config.recovery_run and call_kind == "action" and messages:
            cue = messages[-1].get("content")
            if isinstance(cue, str):
                try:
                    if not json.loads(cue).get("final_required", False):
                        reserve = min(limits.final_reserve_seconds, limits.max_seconds / 5)
                except ValueError:
                    pass
        if call_kind in ("python_code", "format_repair") and (counter.model_calls >= limits.max_model_calls - 1 or remaining <= reserve):
            raise PhaseLimitError("Remaining budget is reserved for a final answer")
        if call_kind == "final_review" and remaining <= reserve:
            raise PhaseLimitError("Remaining time is reserved for accepting the existing answer")
        event = self.store.append_event(state.challenge_id, "react", "started", task_id=task_id,
                                       variant="A", parent_event_id=parent_event_id,
                                       result_summary="Model generation requested.")
        output_limit = self._output_limit(call_kind)
        self.store.begin_model_call(event.event_id, {
            "task_id": task_id, "variant": "A", "messages": messages,
            "output_schema": output_schema,
            "call_kind": call_kind,
            "output_format": "python_code" if call_kind == "python_code" else "json",
            "decoding_mode": "schema" if output_schema is not None and (self.store.config.model.structured_output or self.store.config.model.backend == "vllm") else "none",
            "generation": {**self.store.config.model.model_dump(), "max_new_tokens": output_limit}, "remaining_seconds": remaining - reserve,
        })
        counter.model_calls += 1
        self.store.save_state(state)
        call_started = monotonic()
        try:
            options = {"thinking": self.store.config.model.enable_thinking and call_kind == "python_code"} if self.store.config.model.backend == "vllm" else {}
            result = GenerationResult.model_validate(self.generate_fn(
                messages, max_new_tokens=output_limit,
                remaining_seconds=max(0, self._remaining(task_id, started) - reserve),
                output_schema=output_schema, **options,
            ))
        except Exception as error:
            duration = monotonic() - call_started
            self.store.finish_model_call(event.event_id, error=str(error), duration_seconds=duration)
            state = self.store.load_state(task_id)
            state.counters["A"].input_tokens = state.counters["A"].output_tokens = None
            self.store.save_state(state)
            self.store.append_event(state.challenge_id, "react", "error", task_id=task_id, variant="A",
                                    parent_event_id=event.event_id, exception_type=type(error).__name__,
                                    result_summary=str(error), duration_seconds=duration)
            raise
        duration = monotonic() - call_started
        self.store.finish_model_call(event.event_id, result=result, duration_seconds=duration)
        state = self.store.load_state(task_id)
        for field in ("input_tokens", "output_tokens"):
            previous, current = getattr(state.counters["A"], field), getattr(result.tokens, field)
            setattr(state.counters["A"], field, None if previous is None or current is None else previous + current)
        self.store.save_state(state)
        self.store.append_event(state.challenge_id, "react", "ok", task_id=task_id, variant="A",
                                parent_event_id=event.event_id, result_summary="Raw response saved in model_calls.",
                                tokens=result.tokens, duration_seconds=duration)
        if self._remaining(task_id, started) <= 0:
            raise PhaseLimitError("model response arrived after the A deadline", "timeout")
        return result, event.event_id

    def _generate(self, task_id: str, messages: list[dict], started: float,
                  output_schema: Optional[dict] = None, *, call_kind: str = "action",
                  parent_event_id: Optional[str] = None) -> tuple[GenerationResult, str]:
        """Retry only transient inference failures; each retry consumes calls and time."""
        for attempt in range(self.store.config.model_max_retries + 1):
            try:
                return self._call(task_id, messages, started, output_schema,
                                  call_kind=call_kind, parent_event_id=parent_event_id)
            except (TimeoutError, ConnectionError):
                if attempt == self.store.config.model_max_retries:
                    raise
        raise RuntimeError("unreachable retry state")

    def _parse_response(self, result: GenerationResult, definitions: list[dict], task_id: str,
                        event_id: str, *, allow_read_queue: bool = True):
        """Select one intended validated action while retaining malformed fragments for diagnostics."""
        message = ((result.raw_response or {}).get("choices") or [{}])[0].get("message", {})
        native = message.get("tool_calls")
        names = {entry["function"]["name"] for entry in definitions}
        fragment_errors = []
        repaired_closers = False
        candidates = action_candidates(result.text, names, native, errors=fragment_errors)
        if not candidates and not native and result.finish_reason != "length":
            recovered = repair_action_closers(result.text)
            if recovered != result.text:
                candidates = action_candidates(recovered, names, errors=fragment_errors)
                repaired_closers = bool(candidates)
        actions, errors = {}, []
        for payload in candidates:
            try:
                action = self._parse_action(payload, task_id)
                if isinstance(action, ToolAction) and action.tool_name == "execute_python" and "code" not in action.arguments:
                    PythonSelectionArguments.model_validate(action.arguments)
                if isinstance(action, ToolAction):
                    local_contract = {"inspect_table": TableInspectArguments, "read_artifact": ArtifactArguments,
                                      "view_image": FileArguments}.get(action.tool_name)
                    if local_contract is not None:
                        local_contract.model_validate(action.arguments)
                if isinstance(action, ToolAction) and not (action.tool_name == "execute_python" and "code" not in action.arguments):
                    from jsonschema import Draft202012Validator
                    schema = next(d["function"]["parameters"] for d in definitions
                                  if d["function"]["name"] == action.tool_name)
                    problem = next(Draft202012Validator(schema).iter_errors(action.arguments), None)
                    if problem is not None:
                        raise ValueError("Tool parameters: " + problem.message[:350])
                actions[json.dumps(action.model_dump(), sort_keys=True)] = action
            except ValueError as error:
                errors.append(str(error))
        if allow_read_queue and len(actions) > 1 and all(isinstance(item, ToolAction) and item.tool_name in
                {"inspect_table", "read_artifact", "excel_read_range"} for item in actions.values()):
            ordered = list(actions.values())
            self._read_queues[task_id] = ordered[1:]
            actions = {"first_read": ordered[0]}
        if len(actions) != 1:
            detail = (errors or fragment_errors)
            raise ValueError("Cannot determine one intended action: " +
                             (detail[0][:400] if len(actions) == 0 and detail else str(len(actions)) + " distinct valid candidates"))
        action = next(iter(actions.values()))
        self.store.append_event(self.store.load_state(task_id).challenge_id, "react", "ok", task_id=task_id,
                                variant="A", parent_event_id=event_id,
                                arguments={"normalization": "native_tool_calls" if native else "text_candidates",
                                           "normalized_action": self._action_content(action),
                                           "container_closers_repaired": repaired_closers,
                                           "ignored_invalid_fragments": len(errors) + len(fragment_errors)},
                                result_summary="Action recovered; original output remains in model_calls.")
        return action

    def _action_content(self, action) -> dict:
        """Return compact history content, excluding host identity and invalid model wrappers."""
        if isinstance(action, ToolAction):
            return action.model_dump(exclude_none=True)
        answer = action.answer
        return {"action": "final", "value": answer.answer.value if answer.answer else None,
                "unit": answer.answer.unit if answer.answer else None,
                "computed_value": answer.computed_value, "computed_unit": answer.computed_unit,
                "brief_explanation": answer.brief_explanation, "evidence_refs": answer.evidence_refs,
                "reported_values": [item.model_dump() for item in answer.reported_values],
                "exports": [item.model_dump() for item in action.exports]}

    def _repair_action(self, result: GenerationResult, definitions: list[dict], task_id: str,
                       started: float, event_id: str, error: ValueError, final_required: bool):
        """Attempt one separately prompted format repair, reserving a final call and time."""
        message = ((result.raw_response or {}).get("choices") or [{}])[0].get("message", {})
        native = message.get("tool_calls")
        raw_output = (json.dumps({"tool_calls": native, "text": result.text}, ensure_ascii=False)
                      if native else result.text)
        messages = format_repair_messages(raw_output, definitions, str(error), final_required=final_required)
        if self.count_tokens_fn(messages) + self._output_limit("format_repair") > self.store.config.model.context_limit:
            raise ValueError("Format repair input does not fit; original output was not truncated")
        repair_schema, patch_base = None, None
        try:
            originals = action_candidates(result.text, {d["function"]["name"] for d in definitions}, native)
        except ValueError:
            originals = []
        if (len(originals) == 1 and originals[0].get("action") == "tool"
                and self.store.config.model.repair_structured_output
                and (self.store.config.model.backend == "vllm" or self.store.config.model.structured_output)):
            patch_base = originals[0]
            definition = next((d["function"] for d in definitions
                               if d["function"]["name"] == patch_base["tool_name"]), None)
            if definition:
                parameters = (PythonSelectionArguments.model_json_schema()
                              if patch_base["tool_name"] == "execute_python" and "code" not in patch_base.get("arguments", {})
                              else deepcopy(definition["parameters"]))
                definitions_schema = parameters.pop("$defs", {})
                repair_schema = {"$defs": definitions_schema, "type": "object", "properties": {
                    "arguments": parameters, "unresolved": {"type": "string"}},
                    "additionalProperties": False, "minProperties": 1}
                messages = [{"role": "system", "content": (
                    "Repair only the supplied tool parameters using the reported error and original output. "
                    "Return only an arguments object or unresolved reason. Preserve valid input values. "
                    "Remove unsupported fields. Never solve the task or invent missing information.")},
                    {"role": "user", "content": json.dumps({"model_output": raw_output,
                        "parser_error": str(error)[:500], "tool": definition,
                        "repair_fields": ["arguments", "unresolved"]}, ensure_ascii=False)}]
        repaired, repaired_id = self._call(task_id, messages, started, output_schema=repair_schema,
                                          call_kind="format_repair", parent_event_id=event_id)
        if repair_schema is not None:
            patches = json_candidates(repaired.text)
            if len(patches) != 1 or "arguments" not in patches[0] or "unresolved" in patches[0]:
                raise ValueError("Parameter repair did not recover the missing inputs")
            for key, value in patch_base.get("arguments", {}).items():
                if key in patches[0]["arguments"] and patches[0]["arguments"][key] != value:
                    raise ValueError("Parameter repair changed a supplied input value")
            repaired = repaired.model_copy(update={"text": json.dumps({**patch_base,
                                                    "arguments": patches[0]["arguments"]}), "raw_response": None})
        action = self._parse_response(repaired, definitions, task_id, repaired_id, allow_read_queue=False)
        raw_objects = [] if native else json_candidates(result.text)
        if raw_objects and isinstance(action, ToolAction):
            fields = [item.get("function", item.get("action", item)) for item in raw_objects]
            fields = [item if isinstance(item, dict) else original for item, original in zip(fields, raw_objects)]
            if not any(item.get("tool_name", item.get("name", item.get("action"))) == action.tool_name for item in fields):
                raise ValueError("Format repair cannot invent a missing tool name")
            if action.arguments and not any(any(key in item for key in ("arguments", "parameters", "input")) for item in fields):
                raise ValueError("Format repair cannot invent missing arguments")
        # Ambiguous outputs may be resolved to one existing candidate, never an invented operation.
        try:
            original = action_candidates(result.text, {d["function"]["name"] for d in definitions}, native)
        except ValueError:
            original = []
        if original:
            canonical = self._action_content(action)
            fields = (("action", "tool_name") if repair_schema is not None else
                      ("action", "tool_name", "arguments", "value", "computed_value", "computed_unit"))
            if not any(all(item[key] == canonical.get(key) for key in fields if key in item)
                       for item in original):
                raise ValueError("Format repair changed the operation or value of every original candidate")
        return action, repaired_id

    def _option_assessment(self, action: FinalAction, dispatch: TaskDispatch, event_id: str):
        """Assemble uniquely matched numeric options and return a nonblocking review diagnostic."""
        question, answer = dispatch.task_input.question, action.answer
        options = question_options(question)
        if (not self.store.config.match_answer_options or not options or answer.answer is None
                or not all(any(character.isdigit() for character in text) for text in options.values())):
            return action, None
        results = [(ref, result) for ref, result in self._computed_results(dispatch)
                   if isinstance(result.value, (int, float)) and not isinstance(result.value, bool)]
        aliases = self._evidence_artifacts(dispatch)
        refs = {aliases[key].path if key in aliases else key for key in answer.evidence_refs}
        cited = [(ref, result) for ref, result in results if ref.path in refs or ref.event_id in refs
                 or any(item.event_id == ref.event_id and item.path in refs for item in aliases.values())]
        candidates = cited or results
        unique = {(result.value, result.unit) for _, result in candidates}
        reference, result, source, problem = None, None, None, None
        if len(unique) == 1:
            reference, result = candidates[-1]
            source = "tool"
        elif len(unique) > 1:
            problem = "Multiple computed metrics conflict; identify the current question's result"
        elif answer.computed_value is not None:
            result = ComputedResult(value=answer.computed_value, unit=answer.computed_unit)
            source = "model"
        elif isinstance(answer.answer.value, (int, float)) and not isinstance(answer.answer.value, bool):
            result = ComputedResult(value=answer.answer.value, unit=answer.answer.unit)
            source = "model"
        else:
            problem = "No explicit numeric metric was supplied; keep an evidence-supported choice if unresolved"
        updates = {"option_consistency": "unresolved"}
        if result is not None:
            updates.update(computed_value=result.value, computed_unit=result.unit, computed_value_source=source)
            try:
                expected = computed_option(question, result, context=dispatch.task_input.introduction)
                if expected is None:
                    raise AnswerValidationError("Options are not a uniquely comparable numeric list")
                label = str(answer.answer.value).strip().upper()
                updates.update(answer=AnswerValue(value=expected),
                               option_consistency="matched" if label == expected else "corrected")
                if label != expected:
                    updates["brief_explanation"] = f"Numeric value {result.value} {result.unit or ''} matches option {expected}."
                if reference:
                    updates.update(evidence_refs=list(dict.fromkeys([*answer.evidence_refs, reference.path])),
                                   reported_values=[ReportedValue(value=result.value, unit=result.unit,
                                                                  evidence_ref=reference.path)])
            except AnswerValidationError as error:
                problem = str(error)
        updated = action.model_copy(update={"answer": answer.model_copy(update=updates)})
        feedback = {"option_consistency": updated.answer.option_consistency, "source": source,
                    "computed_value": result.value if result else None, "computed_unit": result.unit if result else None,
                    "original_choice": answer.answer.value, "selected_choice": updated.answer.answer.value,
                    "model_explanation": answer.brief_explanation, "options": options, "reason": problem}
        self.store.append_event(dispatch.task_input.task.challenge_id, "react", "ok", task_id=answer.task_id,
                                variant="A", parent_event_id=event_id, arguments=feedback,
                                result_summary="Option assembly checks consistency, not correctness of the calculation.")
        return updated, feedback if updated.answer.option_consistency == "unresolved" else None

    def _retain_reviewed_answer(self, pending, dispatch: TaskDispatch) -> Optional[Answer]:
        """Keep a valid proposed choice when its optional review times out or cannot be parsed."""
        if pending is None or self.store.config.answer_validation == "strict":
            return None
        action, event_id = pending
        try:
            answer = self._accept(action, dispatch, event_id)
        except ValueError:
            return None
        self.store.append_event(dispatch.task_input.task.challenge_id, "react", "ok", task_id=answer.task_id,
                                variant="A", parent_event_id=event_id,
                                arguments={"option_consistency": "unresolved", "review_interrupted": True},
                                result_summary="Optional review failed; retained the authorized model choice.")
        return answer

    def _evidence_artifacts(self, dispatch: TaskDispatch) -> dict[str, ArtifactRef]:
        """Map IDs only for own A artifacts and explicitly authorized predecessor files."""
        state = self.store.load_state(dispatch.task_input.task.task_id)
        records = [ref for ref in state.artifacts if ref.variant == "A"]
        records.extend(dispatch.selected_artifacts)
        records.extend(ref for export in dispatch.prior_A_exports for ref in export.artifact_refs)
        return {artifact_id(ref): ref for ref in records}

    def _references(self, dispatch: TaskDispatch) -> set[str]:
        """Build the allowed evidence set from task inputs, own events and authorized A sources."""
        state = self.store.load_state(dispatch.task_input.task.task_id)
        allowed = {"question", "introduction", *(item.file_id for item in dispatch.task_input.assets)}
        allowed.add(state.task_id)
        allowed.update(item.path for item in state.allowed_assets)
        allowed.update((dispatch.task_input.task.question_file.path, dispatch.task_input.task.introduction_file.path))
        allowed.update(ref.path for ref in [*state.artifacts, *dispatch.selected_artifacts])
        allowed.update(event.event_id for event in self.store.iter_events(state.challenge_id, state.task_id, "A"))
        for source in dispatch.previous_answers:
            allowed.add(source)
            allowed.update(event.event_id for event in self.store.iter_events(state.challenge_id, source, "A"))
        for export in dispatch.prior_A_exports:
            allowed.add(export.event_id)
            allowed.update(ref.path for ref in export.artifact_refs)
        allowed.update(self._evidence_artifacts(dispatch))
        return allowed

    def _validated_final(self, action: FinalAction, dispatch: TaskDispatch) -> FinalAction:
        """Check final identity, evidence permissions and exports without publishing an answer."""
        task_id = dispatch.task_input.task.task_id
        if action.answer.task_id != task_id:
            raise ValueError("final Answer.task_id must match the current task")
        artifacts = self._evidence_artifacts(dispatch)
        aliases = {key: ref.path for key, ref in artifacts.items()}
        payload = action.model_dump()
        payload["answer"]["evidence_refs"] = [aliases.get(ref, ref) for ref in action.answer.evidence_refs]
        for item in payload["answer"]["reported_values"]:
            item["evidence_ref"] = aliases.get(item["evidence_ref"], item["evidence_ref"])
        for item in payload["exports"]:
            item["evidence_refs"] = [aliases.get(ref, ref) for ref in item["evidence_refs"]]
        action = FinalAction.model_validate(payload)
        allowed = self._references(dispatch)
        references = [*action.answer.evidence_refs, *(item.evidence_ref for item in action.answer.reported_values)]
        references.extend(ref for export in action.exports for ref in export.evidence_refs)
        if any(ref not in allowed for ref in references):
            raise ValueError("evidence references must be supplied IDs or real authorized events/artifacts")
        if len({item.name for item in action.exports}) != len(action.exports):
            raise ValueError("export names must be unique")
        if action.answer.status != "answered" and action.exports:
            raise ValueError("an unanswered task cannot publish reusable values")
        return action

    def _final_evidence(self, action: FinalAction, dispatch: TaskDispatch) -> list[dict]:
        """Prefer cited successful current-task metrics; keep multiple uncited metrics ambiguous."""
        results = self._computed_results(dispatch)
        references = set(action.answer.evidence_refs)
        artifacts = self._evidence_artifacts(dispatch)
        event_ids = {ref.event_id for ref in artifacts.values() if ref.path in references}
        cited = [(ref, result) for ref, result in results
                 if ref.path in references or ref.event_id in references or ref.event_id in event_ids]
        return [{**result.model_dump(), "evidence_ref": ref.path, "event_id": ref.event_id}
                for ref, result in (cited or results)]

    def _review_final_answer(self, action: FinalAction, dispatch: TaskDispatch, event_id: str,
                             started: float, base: list[dict], reviewed: bool):
        """Audit a final answer and perform at most one optional, non-destructive review call."""
        action = self._validated_final(action, dispatch)
        evidence = self._final_evidence(action, dispatch)
        check = assess_final_consistency(dispatch.task_input.question, action.answer, evidence,
                                         context=dispatch.task_input.introduction)
        self.store.append_event(dispatch.task_input.task.challenge_id, "react", "ok",
                                task_id=action.answer.task_id, variant="A", parent_event_id=event_id,
                                arguments={"final_consistency": check.model_dump(),
                                           "candidate": self._action_content(action)},
                                result_summary="Internal agreement check; no option was changed and no gold was read.")
        consistency = ("matched" if check.status == "consistent" else "unresolved"
                       if check.status in ("conflict", "unresolved") else action.answer.option_consistency)
        action = action.model_copy(update={"answer": action.answer.model_copy(update={"option_consistency": consistency})})
        if check.status != "conflict":
            return action, event_id, reviewed
        original, original_event = action, event_id
        counter, limits = self.store.load_state(action.answer.task_id).counters["A"], self.store.config.budget_A
        reserve = min(limits.final_reserve_seconds, limits.max_seconds / 5)
        if reviewed or counter.model_calls >= limits.max_model_calls or self._remaining(action.answer.task_id, started) <= reserve:
            self.store.append_event(dispatch.task_input.task.challenge_id, "react", "ok",
                                    task_id=action.answer.task_id, variant="A", parent_event_id=event_id,
                                    arguments={"final_review": "skipped", "original_choice": action.answer.answer.value},
                                    result_summary="Review already used or remaining calls/time are insufficient; original answer retained.")
            return action, event_id, reviewed
        images = [item for message in base if isinstance(message.get("content"), list)
                  for item in message["content"] if item.get("type") == "image"]
        try:
            events = {event.event_id: event for event in self.store.iter_events(
                dispatch.task_input.task.challenge_id, action.answer.task_id, "A")
                if event.tool_name == "execute_python" and event.status == "started"}
            observations = {item.event_id: item for item in self.store.load_state(action.answer.task_id).observations}
            for row in evidence:
                if row["event_id"] in events:
                    row["executed_code"] = events[row["event_id"]].arguments.get("code", "")
                if row["event_id"] in observations:
                    row["tool_observation"] = observations[row["event_id"]].result_summary
            messages = final_review_messages(dispatch, self._action_content(action), check, evidence, images=images)
            if self.count_tokens_fn(messages) + self._output_limit("final_review") > self.store.config.model.context_limit:
                raise ValueError("Final review context does not fit; source inputs were not truncated")
            # Use _call directly: retries and format repair must not multiply the optional review.
            result, event_id = self._call(action.answer.task_id, messages, started,
                                          call_kind="final_review", parent_event_id=original_event)
            candidate = self._parse_response(result, [], action.answer.task_id, event_id)
            if not isinstance(candidate, FinalAction) or candidate.answer.status != "answered":
                raise ValueError("Final review must return a non-null final answer; no tool was executed")
            # Review concerns the answer, not the previously authorized downstream exports.
            candidate = candidate.model_copy(update={"exports": original.exports})
            updates = {}
            if not candidate.answer.evidence_refs:
                updates["evidence_refs"] = original.answer.evidence_refs
            if not candidate.answer.brief_explanation:
                updates["brief_explanation"] = original.answer.brief_explanation
            candidate = candidate.model_copy(update={"answer": candidate.answer.model_copy(update=updates)})
            candidate = self._validated_final(candidate, dispatch)
            if self.store.config.verify_computed_answers and self.store.config.answer_validation == "strict":
                self._check_computed_answer(candidate, dispatch)
            after = assess_final_consistency(dispatch.task_input.question, candidate.answer, evidence,
                                             context=dispatch.task_input.introduction)
            resolved = after.status in ("consistent", "not_checked")
            self.store.append_event(dispatch.task_input.task.challenge_id, "react", "ok",
                                    task_id=action.answer.task_id, variant="A", parent_event_id=event_id,
                                    arguments={"final_review": "resolved" if resolved else "unresolved",
                                               "original_choice": original.answer.answer.value,
                                               "reviewed_choice": candidate.answer.answer.value,
                                               "final_consistency": after.model_dump()},
                                    result_summary="Model review recorded; unresolved review retains the original answer.")
            if resolved:
                candidate = candidate.model_copy(update={"answer": candidate.answer.model_copy(update={
                    "option_consistency": "matched" if after.status == "consistent" else "unresolved"})})
                return candidate, event_id, True
        except Exception as error:
            self.store.append_event(dispatch.task_input.task.challenge_id, "react", "error",
                                    task_id=action.answer.task_id, variant="A", parent_event_id=event_id,
                                    arguments={"final_review": "failed", "original_choice": original.answer.answer.value},
                                    exception_type=type(error).__name__, result_summary=f"Original answer retained: {error}")
        return original, original_event, True

    def _accept(self, action: FinalAction, dispatch: TaskDispatch, event_id: str) -> Answer:
        """Validate answer identity and real evidence, then attach export provenance."""
        action = self._validated_final(action, dispatch)
        task_id = dispatch.task_input.task.task_id
        mode = self.store.config.answer_validation if self.store.config.verify_computed_answers else "off"
        if mode != "off" and action.answer.status == "answered" and needs_computation(dispatch.task_input.question):
            try:
                checked = self._check_computed_answer(action, dispatch)
                if mode == "strict":
                    action = checked
                else:
                    # Keep the model's value and unit; add only actual computation provenance.
                    action = action.model_copy(update={"answer": action.answer.model_copy(update={
                        "reported_values": checked.answer.reported_values})})
                audit = {"validation_mode": mode, "consistent": True}
                summary = "Computed evidence is consistent; this does not prove correctness."
            except AnswerValidationError as error:
                if mode == "strict":
                    raise
                audit = {"validation_mode": mode, "consistent": False}
                summary = "Answer retained with an audit warning: " + str(error)
            self.store.append_event(dispatch.task_input.task.challenge_id, "react", "ok", task_id=task_id,
                                    variant="A", parent_event_id=event_id, arguments=audit,
                                    result_summary=summary)
        state = self.store.load_state(task_id)
        state.dependency_exports = [
            DependencyExport(name=item.name, value=item.value, unit=item.unit, task_id=task_id,
                             event_id=event_id, applicability=item.applicability,
                             evidence_refs=item.evidence_refs,
                             artifact_refs=[ref for ref in state.artifacts if ref.path in item.evidence_refs])
            for item in action.exports
        ]
        self.store.save_state(state)
        return action.answer

    def _check_computed_answer(self, action: FinalAction, dispatch: TaskDispatch) -> FinalAction:
        """Check internal consistency without reading labels or supplying another answer."""
        task_id = dispatch.task_input.task.task_id
        artifacts = self._evidence_artifacts(dispatch)
        results = self._computed_results(dispatch)
        cited_events = {ref.event_id for ref in artifacts.values() if ref.path in action.answer.evidence_refs
                        and Path(ref.path).name in ("stdout.txt", "computed_result.json")}
        cited_events.update(event.parent_event_id for event in self.store.iter_events(
            dispatch.task_input.task.challenge_id, task_id, "A") if event.event_id in action.answer.evidence_refs
            and event.tool_name == "execute_python" and event.status == "ok")
        candidates = [(ref, result) for ref, result in results if ref.path in action.answer.evidence_refs
                      or ref.event_id in action.answer.evidence_refs or ref.event_id in cited_events]
        if not candidates:
            raise AnswerValidationError("Calculation requires cited successful Python emit_result(value, unit) evidence; layout reads are not computed results")
        failures = []
        for reference, result in candidates:
            try:
                value = validate_result_value(dispatch.task_input.question, action.answer.answer.value,
                                              action.answer.answer.unit, result,
                                              context=dispatch.task_input.introduction)
            except AnswerValidationError as error:
                failures.append(str(error))
                continue
            checked = action.answer.model_dump()
            checked["answer"] = {"value": value, "unit": action.answer.answer.unit or (
                result.unit if value == result.value or isinstance(value, (int, float)) else None)}
            checked["evidence_refs"] = list(dict.fromkeys([*checked["evidence_refs"], reference.path]))
            checked["reported_values"] = [{"value": result.value, "unit": result.unit, "evidence_ref": reference.path}]
            action = action.model_copy(update={"answer": Answer.model_validate(checked)})
            break
        else:
            raise AnswerValidationError(failures[-1])
        return action

    def _computed_results(self, dispatch: TaskDispatch) -> list[tuple[ArtifactRef, ComputedResult]]:
        """Read only current-task result artifacts from successfully completed Python events."""
        task_id = dispatch.task_input.task.task_id
        completed = {event.parent_event_id for event in self.store.iter_events(
            dispatch.task_input.task.challenge_id, task_id, "A")
            if event.status == "ok" and event.tool_name == "execute_python"}
        results = []
        for reference in self.store.load_state(task_id).artifacts:
            if reference.event_id in completed and Path(reference.path).name == "computed_result.json":
                result = ComputedResult.model_validate_json(self.store.artifact_path(reference).read_text())
                if result.value is not None and all(result.checks.values()):
                    results.append((reference, result))
        return results

    def _initial_images(self, dispatch: TaskDispatch) -> list[dict]:
        """Copy authorized original images into durable context without spending tool calls."""
        task_id = dispatch.task_input.task.task_id
        assets = [item for item in dispatch.task_input.assets if item.reference.role == "image"]
        if not assets:
            return []
        event = self.store.append_event(dispatch.task_input.task.challenge_id, "react", "started",
                                       task_id=task_id, variant="A",
                                       arguments={"context_kind": "initial_images",
                                                  "file_ids": [item.file_id for item in assets]},
                                       result_summary="Preparing authorized images for the initial context.")
        images = []
        for index, asset in enumerate(assets, 1):
            source = self.store.dataset.resolve_asset(task_id, asset.file_id)
            if source.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
                raise ValueError("original image is not a supported image")
            reference = self.store.write_artifact(event.event_id, f"image-{index:03d}" + source.suffix.lower(), source)
            state = self.store.load_state(task_id)
            state.artifacts.append(reference)
            self.store.save_state(state)
            images.append({"file_id": asset.file_id, "source_path": asset.reference.path,
                           "artifact_id": artifact_id(reference), "evidence_ref": reference.path,
                           "event_id": event.event_id, "image": str(self.store.artifact_path(reference))})
        provenance = self.store.write_artifact(event.event_id, "image_sources.json",
                                               json.dumps(images, ensure_ascii=False))
        self.store.append_event(dispatch.task_input.task.challenge_id, "react", "ok",
                                task_id=task_id, variant="A", parent_event_id=event.event_id,
                                arguments={"context_kind": "initial_images", "image_count": len(images)},
                                result_ref=provenance.path,
                                result_summary="Authorized original images attached to the initial model context.")
        return images

    def _denied(self, task_id: str, name: str, arguments: dict, reason: str) -> Observation:
        """Record a failed dispatch, counting it only when tool budget remains."""
        state = self.store.load_state(task_id)
        counter = state.counters["A"]
        if counter.tool_calls < self.store.config.budget_A.max_tool_calls:
            counter.tool_calls += 1
        event = self.store.append_event(state.challenge_id, "react", "denied", task_id=task_id, variant="A",
                                       tool_name=name, arguments=arguments, result_summary=reason)
        observation = Observation(event_id=event.event_id, status="denied", result_summary=reason)
        state.observations.append(observation)
        self.store.save_state(state)
        return observation

    def _blocked_tool(self, python: PythonTool, action: ToolAction, failed: bool) -> Optional[str]:
        """Block identical failed requests and empty saved-file reads before executing a tool."""
        if failed:
            return "Identical request previously produced an error or no output. Change the code/arguments; do not repeat it."
        if action.tool_name == "read_artifact":
            key = action.arguments.get("artifact_id")
            reference = python.available_artifacts().get(key) if isinstance(key, str) else None
            if reference and self.store.artifact_path(reference).stat().st_size == 0:
                return "Saved artifact is empty. Do not reread it; correct execute_python to print(result) or save an output."
        return None

    def _tool_has_result(self, python: PythonTool, action: ToolAction, observation: Observation) -> bool:
        """Distinguish a successful Python process from an actual printed value or saved output."""
        if observation.status != "ok":
            return False
        if action.tool_name != "execute_python":
            return True
        if observation.result_ref is None:
            return False
        return any(ref.event_id == observation.result_ref.event_id and
                   (Path(ref.path).name == "stdout.txt" or Path(ref.path).name.startswith("output-")) and
                   self.store.artifact_path(ref).stat().st_size > 0
                   for ref in python.available_artifacts().values())

    def _python_action(self, python: PythonTool, selection: ToolAction, base: list[dict],
                       history: list[list[dict]], controls: dict, started: float,
                       parent_event_id: str) -> tuple[ToolAction | Observation, list[dict]]:
        """Generate plain Python code with host-bound inputs and one bounded syntax repair."""
        task_id = python.task_id
        turns, draft = [], None
        try:
            if "code" in selection.arguments:
                # Strict legacy requests provide a draft, never bypassing the second generation.
                checked = PythonArguments.model_validate(selection.arguments)
                draft = checked.code
                inputs = PythonSelectionArguments(file_ids=checked.file_ids, artifact_ids=checked.artifact_ids,
                                                  instruction=selection.brief_reason or "Perform the supplied computation.")
            else:
                inputs = PythonSelectionArguments.model_validate(selection.arguments)
            session = python.session_context(PythonArguments(file_ids=inputs.file_ids, artifact_ids=inputs.artifact_ids, code="pass"))
        except (ValueError, PermissionError, FileNotFoundError) as error:
            return self._denied(task_id, selection.tool_name, selection.arguments, str(error)), turns
        pybase = python_messages(base, {**inputs.model_dump(), **session}, python.system_prompt(), draft)
        observations = [[message for message in turn if message["role"] == "user"
                         and isinstance(message["content"], list)] for turn in history]
        turns.append({"role": "user", "content": json.dumps({
            "generation_task": "python_code", "selected_operation": inputs.model_dump(),
            "instruction": "Return Python code only; compute the requested metric and emit_result(value, unit). Save files only when required.",
        }, ensure_ascii=False)})
        for repair in range(self.store.config.budget_A.max_format_repairs + 1):
            counter = self.store.load_state(task_id).counters["A"]
            reserve = min(self.store.config.budget_A.final_reserve_seconds,
                          self.store.config.budget_A.max_seconds / 5)
            if counter.model_calls >= self.store.config.budget_A.max_model_calls - 1 or self._remaining(task_id, started) <= reserve:
                return self._denied(task_id, selection.tool_name, selection.arguments,
                                    "Python code was not completed; remaining budget is reserved for a final answer."), turns
            cue = {**controls, "final_required": False, "generation_task": "python_code",
                   "remaining_model_calls": self.store.config.budget_A.max_model_calls - counter.model_calls,
                   "remaining_seconds": max(0, self._remaining(task_id, started))}
            messages = self._pack(pybase, [*observations, turns.copy()], cue)
            try:
                result, event_id = self._generate(task_id, messages, started,
                                                call_kind="python_code", parent_event_id=parent_event_id)
            except (TimeoutError, PhaseLimitError):
                if self._remaining(task_id, started) <= 0:
                    raise
                return self._denied(task_id, selection.tool_name, selection.arguments,
                                    "Python generation budget ended; remaining budget is reserved for a final answer."), turns
            turns.append({"role": "assistant", "content": result.text})
            try:
                if result.finish_reason == "length":
                    raise ValueError("Python code was truncated; return shorter complete code")
                code = parse_python_code(result.text, input_paths=session["input_paths"])
                if code != result.text.strip():
                    source_format = ("parameter_tag" if "<parameter=code>" in result.text.lower()
                                     else "code_fence" if "```" in result.text else "input_path_or_prose")
                    self.store.append_event(python.challenge_id, "react", "ok", task_id=task_id, variant="A",
                                            parent_event_id=event_id,
                                            arguments={"normalization": "python_code_extracted",
                                                       "source_format": source_format},
                                            result_summary="Python code extracted and syntax checked before execution.")
                arguments = PythonArguments(file_ids=inputs.file_ids, artifact_ids=inputs.artifact_ids,
                                            code=code)
                return ToolAction(action="tool", tool_name="execute_python", arguments=arguments.model_dump(),
                                  brief_reason=selection.brief_reason), turns
            except ValueError as error:
                self.store.append_event(python.challenge_id, "react", "error", task_id=task_id, variant="A",
                                        parent_event_id=event_id, exception_type=type(error).__name__,
                                        arguments={"error_kind": "python_format", "repair_number": repair},
                                        result_summary=str(error))
                turns.append({"role": "user", "content": python_correction(
                    error, raw_output=result.text, selected_operation=inputs.model_dump())})
        return self._denied(task_id, selection.tool_name, selection.arguments,
                            "Python code repair allowance exhausted; no code was executed."), turns

    def run(self, dispatch: TaskDispatch, *, finalize: bool = True) -> Answer:
        """Execute one A attempt and freeze success or an explicit failed outcome."""
        dispatch = TaskDispatch.model_validate(dispatch)
        if dispatch.variant != "A":
            raise ValueError("ReActAgent implements Benchmark A only")
        task_id = dispatch.task_input.task.task_id
        if self.store.has_phase(task_id, "A"):
            return self.store.load_phase(task_id, "A").answer_A
        state = self.store.load_state(task_id)
        if state.plan != dispatch.plan or state.stage != "react":
            raise ValueError("Scheduler.prepare_task must publish this dispatch first")
        if state.counters["A"].model_calls or state.counters["A"].tool_calls:
            raise RuntimeError("A was interrupted; preserve the attempt and use a new run")
        started, history = monotonic(), []
        answer, terminal, pending_final = None, "no_final_answer", None
        failed_tools, force_final = {}, False
        last_error_signature, consecutive_errors = None, 0
        format_repairs_used, action_retry_used, final_format_attempted, option_reviewed = 0, False, False, False
        registry = None
        runtime_repairs = 0
        python_continuation = None
        previous_cell = None
        try:
            source = dispatch.workspace_policy.source_task_id
            versions = [ref for ref in dispatch.selected_artifacts if ref.task_id == source and
                        Path(ref.path).suffix.lower() in (".xlsx", ".xlsm", ".xlsb", ".csv")]
            if source and not versions:
                raise PhaseLimitError("required saved workspace is unavailable; original inputs were not substituted",
                                      "material_unresolved")
            python = PythonTool(self.store, task_id, tokenizer=self.tokenizer,
                                selected_artifacts=dispatch.selected_artifacts,
                                workspace_policy=dispatch.workspace_policy)
            registry = ToolRegistry(python)
            base = react_messages(dispatch, registry.definitions(), registry.system_prompt(),
                                  images=self._initial_images(dispatch))
            while answer is None:
                counter = self.store.load_state(task_id).counters["A"]
                remaining = max(0, self._remaining(task_id, started))
                reserve = min(self.store.config.budget_A.final_reserve_seconds,
                              self.store.config.budget_A.max_seconds / 5)
                controls = {
                    "remaining_model_calls": self.store.config.budget_A.max_model_calls - counter.model_calls,
                    "remaining_tool_calls": self.store.config.budget_A.max_tool_calls - counter.tool_calls,
                    "remaining_seconds": remaining,
                    "python_action_allowed": self.store.config.budget_A.max_model_calls - counter.model_calls >= (2 if python_continuation else 3),
                    "final_required": counter.model_calls >= self.store.config.budget_A.max_model_calls - 1
                                      or counter.tool_calls >= self.store.config.budget_A.max_tool_calls
                                      or remaining <= reserve or force_final,
                    "available_artifacts": [
                        {"artifact_id": key, "name": Path(ref.path).name, "event_id": ref.event_id,
                         "evidence_ref": ref.path, "empty": self.store.artifact_path(ref).stat().st_size == 0}
                        for key, ref in python.available_artifacts().items()
                        if Path(ref.path).name not in {"dispatch.json", "request.json", "inputs.json",
                                                       "result.json", "service.json", "code.py"}
                    ],
                }
                computation_required = self.store.config.verify_computed_answers and self.store.config.answer_validation == "strict" and needs_computation(dispatch.task_input.question)
                computed = self._computed_results(dispatch) if self.store.config.verify_computed_answers else []
                controls.update(computation_required=computation_required, computation_ready=bool(computed),
                                computed_results=[{**result.model_dump(), "artifact_id": artifact_id(ref),
                                                   "evidence_ref": ref.path} for ref, result in computed])
                if computation_required and not computed:
                    controls["budget_hint"] = "Inspect layout at most once, then execute_python; keep 3 model calls and 1 tool call for computation plus final. If computation cannot be completed, return value=null."
                if python_continuation is not None and not controls["final_required"]:
                    action, event_id = python_continuation
                    python_continuation = None
                    result = None
                elif self._read_queues.get(task_id) and not controls["final_required"]:
                    action = self._read_queues[task_id].pop(0)
                    result = None
                else:
                    messages = self._pack(base, history, controls)
                    result, event_id = self._generate(task_id, messages, started)
                    action = None
                try:
                    if action is None:
                        action = self._parse_response(result, registry.definitions(), task_id, event_id)
                except ValueError as error:
                    self.store.append_event(state.challenge_id, "react", "error", task_id=task_id, variant="A",
                                            parent_event_id=event_id, exception_type=type(error).__name__,
                                            arguments={"error_kind": "format"}, result_summary=str(error))
                    counter = self.store.load_state(task_id).counters["A"]
                    if (self.store.config.llm_format_repair and format_repairs_used < min(1, self.store.config.budget_A.max_format_repairs)
                            and counter.model_calls < self.store.config.budget_A.max_model_calls - 1
                            and self._remaining(task_id, started) > reserve):
                        format_repairs_used += 1
                        try:
                            action, event_id = self._repair_action(result, registry.definitions(), task_id,
                                                                 started, event_id, error, controls["final_required"])
                        except (ValueError, TimeoutError, PhaseLimitError) as repair_error:
                            self.store.append_event(state.challenge_id, "react", "error", task_id=task_id, variant="A",
                                                    parent_event_id=event_id, arguments={"error_kind": "format_repair"},
                                                    result_summary=str(repair_error))
                    if action is None:
                        if controls["final_required"] or final_format_attempted:
                            raise PhaseLimitError("No complete action after bounded format recovery", "no_final_answer")
                        if (not action_retry_used and self.store.load_state(task_id).counters["A"].model_calls <
                                self.store.config.budget_A.max_model_calls - 1 and self._remaining(task_id, started) > reserve):
                            action_retry_used = True
                            history.append([{"role": "user", "content": action_correction(
                                error, task_id, raw_output=result.text)}])
                            continue
                        force_final, final_format_attempted = True, True
                        history.append([{"role": "user", "content": action_correction(
                            error, task_id, final_required=True, raw_output=result.text)}])
                        continue
                error_kind = "format"
                try:
                    if isinstance(action, FinalAction):
                        error_kind = "evidence"
                        if self.store.config.final_consistency_review:
                            action, event_id, option_reviewed = self._review_final_answer(
                                action, dispatch, event_id, started, base, option_reviewed)
                            review = None
                        else:
                            action, review = self._option_assessment(action, dispatch, event_id)
                        if (review and not option_reviewed and
                                self.store.load_state(task_id).counters["A"].model_calls < self.store.config.budget_A.max_model_calls
                                and self._remaining(task_id, started) > 0):
                            option_reviewed = True
                            pending_final = (action, event_id)
                            history.append([{"role": "assistant", "content": json.dumps(self._action_content(action), ensure_ascii=False)},
                                            {"role": "user", "content": option_review(review)}])
                            continue
                        answer = self._accept(action, dispatch, event_id)
                        terminal = "completed" if answer.status == "answered" else answer.status
                        break
                except ValueError as error:
                    if isinstance(error, AnswerValidationError):
                        error_kind = "computation"
                    signature = (error_kind, str(error))
                    same_format_sequence = (error_kind == "format" and last_error_signature is not None
                                            and last_error_signature[0] == "format")
                    consecutive_errors = consecutive_errors + 1 if same_format_sequence or signature == last_error_signature else 1
                    last_error_signature = signature
                    self.store.append_event(state.challenge_id, "react", "error", task_id=task_id, variant="A",
                                            parent_event_id=event_id, exception_type=type(error).__name__,
                                            arguments={"error_kind": error_kind,
                                                       "consecutive_errors": consecutive_errors},
                                            result_summary=str(error))
                    if consecutive_errors > self.store.config.budget_A.max_format_repairs:
                        raise PhaseLimitError(f"consecutive {error_kind} errors exhausted the repair allowance")
                    history.append([{"role": "user", "content": action_correction(
                                        error, task_id, final_required=controls["final_required"],
                                        error_kind=error_kind)}])
                    continue
                last_error_signature, consecutive_errors = None, 0
                turn = [{"role": "assistant", "content": json.dumps(self._action_content(action), ensure_ascii=False)}]
                preparation_failure = None
                if action.tool_name == "execute_python" and not controls["final_required"]:
                    selected_action = action
                    action, _ = self._python_action(python, action, base, history, controls, started, event_id)
                    if isinstance(action, Observation):
                        preparation_failure, force_final = action, True
                        action = selected_action
                signature = json.dumps([action.tool_name, action.arguments], sort_keys=True, ensure_ascii=False)
                if preparation_failure is not None:
                    observation = preparation_failure
                elif controls["final_required"]:
                    observation = self._denied(task_id, action.tool_name, action.arguments,
                                               "Final answer required; no tool was executed.")
                else:
                    try:
                        duplicate_inspection = (action.tool_name == "execute_python"
                            and action.arguments.get("code") == previous_cell
                            and is_inspection_cell(previous_cell or "")
                            and "emit_result" not in (previous_cell or ""))
                        blocked = ("This inspection cell was already executed; use the previous observation and session variables. "
                                   "Generate the next computation instead of repeating inspection." if duplicate_inspection else
                                   self._blocked_tool(python, action, signature in failed_tools))
                        if blocked:
                            if duplicate_inspection:
                                skipped = self.store.append_event(state.challenge_id, "react", "denied",
                                    task_id=task_id, variant="A", tool_name=action.tool_name,
                                    arguments={"duplicate_inspection": True}, result_summary=blocked)
                                observation = Observation(event_id=skipped.event_id, status="denied", result_summary=blocked)
                            else:
                                observation = self._denied(task_id, action.tool_name, action.arguments, blocked)
                        else:
                            observation = registry.call(action.tool_name, action.arguments,
                                                        remaining_seconds=self._remaining(task_id, started))
                    except ValueError as error:
                        observation = self._denied(task_id, action.tool_name, action.arguments, str(error))
                feedback = observation.model_dump(exclude={"image_ref"})
                if (self.store.config.direct_python_repair and action.tool_name == "execute_python"
                        and "code" in action.arguments and observation.status == "error"
                        and runtime_repairs < self.store.config.model.python_runtime_repairs):
                    runtime_repairs += 1
                    repaired = self._repair_python(python, action, observation, base, history, started, event_id)
                    if repaired is not None:
                        action = repaired
                        observation = registry.call(action.tool_name, action.arguments,
                                                    remaining_seconds=self._remaining(task_id, started))
                        feedback = observation.model_dump(exclude={"image_ref"})
                        feedback["runtime_repair_attempt"] = runtime_repairs
                if action.tool_name == "execute_python" and "code" in action.arguments:
                    # Preserve executed code as an observation, without leaking code-only repair instructions.
                    feedback["executed_code"] = action.arguments["code"]
                has_result = self._tool_has_result(python, action, observation)
                if computation_required and action.tool_name == "execute_python" and observation.status == "ok":
                    has_result = observation.result_ref is not None and any(
                        ref.event_id == observation.result_ref.event_id for ref, _ in self._computed_results(dispatch))
                    if not has_result:
                        feedback["recovery"] = "Return the actual computed value with emit_result(value, unit). Printing alone does not publish a bindable result."
                if not has_result:
                    failed_tools[signature] = failed_tools.get(signature, 0) + 1
                    force_final = force_final or failed_tools[signature] >= 2
                    feedback["error_kind"] = "missing_output" if observation.status == "ok" else "execution"
                    feedback.setdefault("recovery", "Change the code/arguments. Missing stdout requires print(result); an empty file cannot supply a value.")
                content = [{"type": "text", "text": json.dumps(feedback, ensure_ascii=False)}]
                if observation.image_ref:
                    content.append({"type": "image", "image": str(self.store.artifact_path(observation.image_ref))})
                history.append([*turn, {"role": "user", "content": content}])
                if action.tool_name == "execute_python" and observation.status == "ok":
                    code = action.arguments.get("code", "")
                    published = any(ref.event_id == observation.event_id for ref, _ in self._computed_results(dispatch))
                    inspection = (not published and is_inspection_cell(code))
                    if (not has_result or inspection) and code != previous_cell and preparation_failure is None:
                        arguments = {key: value for key, value in selected_action.arguments.items() if key != "code"}
                        arguments["instruction"] = ("Continue the current question from the existing session. "
                            "Use the last observation and variables; complete the next unfinished computation, "
                            "then emit_result. Do not repeat inspection already performed.")
                        python_continuation = (selected_action.model_copy(update={"arguments": arguments}), event_id)
                    previous_cell = code
        except PhaseLimitError as error:
            terminal, reason = error.status, str(error)
        except TimeoutError as error:
            terminal, reason = "timeout", str(error)
        except Exception as error:
            terminal, reason = "execution_error", str(error)
        finally:
            self._read_queues.pop(task_id, None)
            if registry is not None:
                registry.close()
        if answer is None:
            answer = self._retain_reviewed_answer(pending_final, dispatch)
            if answer is not None:
                terminal = "completed" if answer.status == "answered" else answer.status
        if answer is None:
            answer = Answer(task_id=task_id, answer=None, status="no_final_answer",
                            brief_explanation=reason)
            self.store.append_event(state.challenge_id, "react", "error", task_id=task_id, variant="A",
                                    result_summary=reason, exception_type=terminal)
        state = self.store.load_state(task_id)
        state.counters["A"].duration_seconds += monotonic() - started
        state.answer_A, state.terminal_status["A"], state.stage = answer, terminal, "freeze_A"
        if finalize:
            self.store.freeze_phase(state, "A")
        else:
            self.store.save_state(state)
        return answer

    def _repair_python(self, python: PythonTool, action: ToolAction, failure: Observation,
                       base: list[dict], history: list[list[dict]], started: float,
                       parent_event_id: str) -> Optional[ToolAction]:
        """Repair the original code once using its traceback without another tool-selection turn."""
        counter = self.store.load_state(python.task_id).counters["A"]
        limits = self.store.config.budget_A
        if counter.model_calls >= limits.max_model_calls - 1 or counter.tool_calls >= limits.max_tool_calls:
            return None
        if self._remaining(python.task_id, started) <= limits.final_reserve_seconds:
            return None
        arguments = PythonArguments.model_validate(action.arguments)
        selection = {**python.session_context(arguments), "file_ids": arguments.file_ids, "artifact_ids": arguments.artifact_ids,
                     "instruction": "Fix the runtime error while preserving the task's calculation and inputs."}
        pybase = python_messages(base, selection, python.system_prompt(), arguments.code)
        observations = [[message for message in turn if message["role"] == "user"
                         and isinstance(message["content"], list)] for turn in history]
        observations.append([{"role": "user", "content": json.dumps({
            "failed_execution": failure.model_dump(exclude={"image_ref"}),
            "instruction": "Return repaired Python only. Publish the result before optional file exports."},
            ensure_ascii=False)}])
        try:
            messages = self._pack(pybase, observations, {"generation_task": "python_code"})
            result, _ = self._generate(python.task_id, messages, started,
                                       call_kind="python_code", parent_event_id=parent_event_id)
            if result.finish_reason == "length":
                return None
            repaired = arguments.model_copy(update={"code": parse_python_code(
                result.text, input_paths=selection["input_paths"])})
            return action.model_copy(update={"arguments": repaired.model_dump()})
        except (ValueError, TimeoutError, PhaseLimitError):
            return None


class MinimalSolver(ReActAgent):
    """Diagnostic code-execute-final baseline, separate from the formal A/B protocol."""

    def run(self, dispatch: TaskDispatch) -> Answer:
        """Bypass free tool selection while preserving authorization, budgets and full storage."""
        task_id = dispatch.task_input.task.task_id
        state = self.store.load_state(task_id)
        if self.store.has_phase(task_id, "A"):
            return self.store.load_phase(task_id, "A").answer_A
        if dispatch.variant != "A" or state.stage != "react":
            raise ValueError("MinimalSolver requires a prepared A dispatch")
        if state.counters["A"].model_calls or state.counters["A"].tool_calls:
            raise RuntimeError("diagnostic attempt already started; use a new run")
        started, answer, terminal, pending_final = monotonic(), None, "no_final_answer", None
        python = PythonTool(self.store, task_id, tokenizer=self.tokenizer,
                            selected_artifacts=dispatch.selected_artifacts)
        registry = ToolRegistry(python)
        try:
            base = react_messages(dispatch, registry.definitions(), registry.system_prompt(),
                                  images=self._initial_images(dispatch))
            inputs = {"file_ids": [asset.file_id for asset in dispatch.task_input.assets],
                      "instruction": "Solve the current question. Inspect actual table layout in this script if needed; publish the computed result."}
            inputs.update(python.session_context(PythonArguments(file_ids=inputs["file_ids"], code="pass")))
            pybase = python_messages(base, inputs, python.system_prompt())
            messages = self._pack(pybase, [], {"generation_task": "python_code"})
            result, event_id = self._generate(task_id, messages, started, call_kind="python_code")
            if result.finish_reason == "length":
                raise PhaseLimitError("diagnostic Python code was truncated")
            action = ToolAction(action="tool", tool_name="execute_python", arguments={
                "file_ids": inputs["file_ids"], "code": parse_python_code(result.text, input_paths=inputs["input_paths"])})
            observation = registry.call(action.tool_name, action.arguments,
                                        remaining_seconds=self._remaining(task_id, started))
            if observation.status == "error" and self.store.config.direct_python_repair:
                repaired = self._repair_python(python, action, observation, base, [], started, event_id)
                if repaired:
                    action = repaired
                    observation = registry.call(action.tool_name, action.arguments,
                                                remaining_seconds=self._remaining(task_id, started))
            feedback = {**observation.model_dump(exclude={"image_ref"}), "executed_code": action.arguments["code"]}
            history = [[{"role": "user", "content": json.dumps(feedback, ensure_ascii=False)}]]
            option_reviewed = False
            for repair in range(self.store.config.budget_A.max_format_repairs + 2):
                messages = self._pack(base, history, {"final_required": True})
                final, event_id = self._generate(task_id, messages, started)
                try:
                    action = self._parse_response(final, registry.definitions(), task_id, event_id)
                    if not isinstance(action, FinalAction):
                        raise ValueError("diagnostic final stage requires action=final")
                    if self.store.config.final_consistency_review:
                        action, event_id, option_reviewed = self._review_final_answer(
                            action, dispatch, event_id, started, base, option_reviewed)
                        review = None
                    else:
                        action, review = self._option_assessment(action, dispatch, event_id)
                    if (review and not option_reviewed and
                            self.store.load_state(task_id).counters["A"].model_calls < self.store.config.budget_A.max_model_calls):
                        option_reviewed = True
                        pending_final = (action, event_id)
                        history.append([{"role": "user", "content": option_review(review)}])
                        continue
                    answer = self._accept(action, dispatch, event_id)
                    terminal = "completed" if answer.status == "answered" else answer.status
                    break
                except ValueError as error:
                    if repair >= self.store.config.budget_A.max_format_repairs:
                        raise
                    history.append([{"role": "user", "content": action_correction(error, task_id, final_required=True)}])
        except Exception as error:
            terminal = error.status if isinstance(error, PhaseLimitError) else "timeout" if isinstance(error, TimeoutError) else "execution_error"
            answer = self._retain_reviewed_answer(pending_final, dispatch)
            if answer is not None:
                terminal = "completed"
            else:
                answer = Answer(task_id=task_id, status="no_final_answer", brief_explanation=str(error))
            self.store.append_event(state.challenge_id, "react", "error", task_id=task_id, variant="A",
                                    exception_type=type(error).__name__, result_summary=str(error))
        finally:
            registry.close()
        state = self.store.load_state(task_id)
        state.counters["A"].duration_seconds += monotonic() - started
        state.answer_A, state.terminal_status["A"], state.stage = answer, terminal, "freeze_A"
        self.store.freeze_phase(state, "A")
        return answer


def select_workspace_artifacts(store: RuntimeStore, plan) -> list[ArtifactRef]:
    """Select an unambiguous saved source workbook; missing/ambiguous versions remain explicit."""
    source = plan.workspace_policy.source_task_id
    if source is None:
        return []
    frozen = store.load_phase(source, "A")
    records = frozen.artifacts
    candidates = [ref for ref in records if Path(ref.path).name.startswith("output-") and
                  Path(ref.path).suffix.lower() in (".xlsx", ".xlsm", ".xlsb", ".csv")]
    exported = {ref.path: ref for export in frozen.dependency_exports for ref in export.artifact_refs
                if ref in candidates}
    if len(exported) == 1:
        return list(exported.values())
    return candidates if len(candidates) == 1 else []


def select_task_artifacts(store: RuntimeStore, plan) -> list[ArtifactRef]:
    """Authorize exported predecessor files plus an unambiguous requested workspace."""
    references = select_workspace_artifacts(store, plan)
    for dependency in plan.dependencies:
        if dependency.kind == "prior_text":
            continue
        source = store.load_phase(dependency.source_task_id, "A")
        for export in source.dependency_exports:
            references.extend(export.artifact_refs)
    return list({ref.path: ref for ref in references}.values())


def run_benchmark_A(store: RuntimeStore, model, scheduler: Optional[Scheduler] = None, *,
                    can_start_fn: Optional[Callable[[str], bool]] = None,
                    on_finished_fn: Optional[Callable[[str], None]] = None,
                    allow_partial: bool = False) -> dict[str, dict[str, str]]:
    """Run A with optional admission/progress hooks; incomplete formal runs still fail."""
    scheduler = scheduler or Scheduler(store)
    agent = ReActAgent(store, model.generate, model.count_tokens, model.tokenizer)
    with ThreadPoolExecutor(max_workers=store.config.task_concurrency) as workers:
        running = {}
        while True:
            for task_id in scheduler.ready_tasks("A"):
                if can_start_fn is not None and not can_start_fn(task_id):
                    break
                plan = store.load_state(store.dataset.get_task(task_id).challenge_id).plan
                fragment = next(item for item in plan.plans if item.task_id == task_id)
                dispatch = scheduler.prepare_task(task_id, "A", selected_artifacts=select_task_artifacts(store, fragment))
                running[workers.submit(agent.run, dispatch)] = task_id
            if not running:
                break
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in finished:
                task_id = running.pop(future)
                future.result()
                scheduler.on_task_finished(task_id, "A")
                if on_finished_fn is not None:
                    on_finished_fn(task_id)
    progress = store.progress()
    unfinished = [key for key in store.dataset.task_ids if not store.has_phase(key, "A")]
    if unfinished and not allow_partial:
        raise RuntimeError(f"A has pending/interrupted tasks; no automatic restart: {unfinished}")
    return progress
