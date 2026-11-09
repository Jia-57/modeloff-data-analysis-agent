"""Generate bounded challenge plans with injectable, locally implemented model calls."""

from __future__ import annotations

import json
from threading import RLock
from time import monotonic
from typing import Callable

from .prompts import planning_messages
from .runtime import RuntimeStore
from .schema import parse_model_json
from .scheduling import validate_plan
from .types import ChallengeInput, ChallengePlan, GenerationResult, RunConfig


def planning_reserve_seconds(config: RunConfig, task_count: int) -> float:
    """Reserve the historical first-round project planning budget."""
    return config.planning.max_seconds


class PlanningLimitError(RuntimeError):
    """Planning cannot continue within the configured time, call, or context limits."""


class ChallengePlanner:
    """One planning task per challenge, potentially containing several output batches."""

    def __init__(
        self, store: RuntimeStore, generate_fn: Callable[..., GenerationResult],
        count_tokens_fn: Callable[[list[dict[str, str]]], int],
        material_context: dict | None = None,
    ) -> None:
        """Inject generation and exact prompt counting without loading a model here.
        Args: generate_fn: Accepts messages, max_new_tokens, remaining_seconds;
            count_tokens_fn: Counts the complete rendered request with its chat template.
        """
        self.store, self.generate_fn, self.count_tokens_fn = store, generate_fn, count_tokens_fn
        self.material_context = material_context or {}
        self._lock = RLock()

    def _call(
        self, challenge_id: str, messages: list[dict[str, str]], started: float,
    ) -> GenerationResult:
        """Save the actual request/response and count every attempted model invocation."""
        state = self.store.load_state(challenge_id)
        limits = self.store.config.planning
        remaining = limits.max_seconds - state.counters.duration_seconds - (monotonic() - started)
        if state.counters.model_calls >= limits.max_model_calls or remaining <= 0:
            raise PlanningLimitError("challenge planning budget exhausted")
        input_tokens = self.count_tokens_fn(messages)
        if type(input_tokens) is not int or input_tokens < 0:
            raise ValueError("count_tokens_fn must return a nonnegative integer")
        if input_tokens + limits.max_new_tokens > self.store.config.model.context_limit:
            raise PlanningLimitError("full planning input plus reserved output exceeds context; no text was truncated")
        remaining = limits.max_seconds - state.counters.duration_seconds - (monotonic() - started)
        if remaining <= 0:
            raise PlanningLimitError("planning deadline elapsed during prompt preparation")
        event = self.store.append_event(challenge_id, "planning", "started", arguments={
            "max_new_tokens": limits.max_new_tokens, "request_tokens": input_tokens,
        })
        request = {
            "challenge_id": challenge_id, "messages": messages,
            "output_schema": None, "decoding_mode": "none",
            "generation": {**self.store.config.model.model_dump(), "max_new_tokens": limits.max_new_tokens},
            "remaining_seconds": remaining,
        }
        self.store.begin_model_call(event.event_id, request)
        state.counters.model_calls += 1
        self.store.save_state(state)
        call_started = monotonic()
        try:
            options = {"thinking": self.store.config.model.enable_thinking} if self.store.config.model.backend == "vllm" else {}
            result = GenerationResult.model_validate(self.generate_fn(
                messages, max_new_tokens=limits.max_new_tokens, remaining_seconds=remaining,
                output_schema=None, **options,
            ))
        except Exception as error:
            state = self.store.load_state(challenge_id)
            state.counters.input_tokens = state.counters.output_tokens = None
            self.store.save_state(state)
            self.store.finish_model_call(event.event_id, error=str(error), duration_seconds=monotonic() - call_started)
            self.store.append_event(
                challenge_id, "planning", "error", parent_event_id=event.event_id,
                exception_type=type(error).__name__, result_summary=str(error),
                duration_seconds=monotonic() - call_started,
            )
            raise
        self.store.finish_model_call(event.event_id, result=result, duration_seconds=monotonic() - call_started)
        state = self.store.load_state(challenge_id)
        for field in ("input_tokens", "output_tokens"):
            previous, current = getattr(state.counters, field), getattr(result.tokens, field)
            setattr(state.counters, field, None if previous is None or current is None else previous + current)
        self.store.save_state(state)
        self.store.append_event(
            challenge_id, "planning", "ok", parent_event_id=event.event_id,
            result_summary="Planning model response saved in model_calls.",
            tokens=result.tokens, duration_seconds=monotonic() - call_started,
        )
        if state.counters.duration_seconds + monotonic() - started > limits.max_seconds:
            raise PlanningLimitError("model response arrived after the planning deadline")
        return result

    def _batch(
        self, challenge_id: str, messages: list[dict[str, str]], started: float,
    ) -> GenerationResult:
        """Retry only transient inference errors, with each attempt consuming budget."""
        for attempt in range(self.store.config.model_max_retries + 1):
            try:
                return self._call(challenge_id, messages, started)
            except (TimeoutError, ConnectionError):
                if attempt == self.store.config.model_max_retries:
                    raise
        raise RuntimeError("unreachable retry state")

    def _plan_batch(
        self, inputs: ChallengeInput, target: list[str], started: float,
        feedback: str, repairs: int = 0,
    ) -> tuple[ChallengePlan, int]:
        """Validate and repair only this batch, sharing its allowance with dependency repairs."""
        challenge_id = inputs.context.challenge_id
        while True:
            try:
                result = self._batch(challenge_id, planning_messages(
                    inputs, target, feedback, self.material_context), started)
                if result.finish_reason == "length":
                    raise ValueError("planning output was cut off; increase its limit or reduce batch size")
                batch = ChallengePlan.model_validate(parse_model_json(result.text, {"challenge_id", "plans"}))
                if batch.challenge_id != challenge_id or {item.task_id for item in batch.plans} != set(target):
                    raise ValueError("output batch must cover exactly target_task_ids")
                return batch, repairs
            except ValueError as error:
                self.store.append_event(
                    challenge_id, "planning", "error", exception_type=type(error).__name__,
                    result_summary=str(error),
                    arguments={"repair_scope": "batch", "target_task_ids": target},
                )
                if repairs >= self.store.config.planning.max_format_repairs:
                    raise
                repairs += 1
                feedback = "Correct only this output batch: " + str(error)

    def _generate_candidate(
        self, inputs: ChallengeInput, started: float, feedback: str,
    ) -> ChallengePlan:
        """Cache successful batches and repair only those implicated by merged validation."""
        limits = self.store.config.planning
        ids = inputs.context.task_ids
        targets = [ids[offset:offset + limits.batch_size] for offset in range(0, len(ids), limits.batch_size)]
        batches, repairs = {}, {}
        for index, target in enumerate(targets):
            batches[index], repairs[index] = self._plan_batch(inputs, target, started, feedback)
        while True:
            try:
                candidate = ChallengePlan(
                    challenge_id=inputs.context.challenge_id,
                    plans=[plan for index in range(len(targets)) for plan in batches[index].plans],
                    validation_notes=list(dict.fromkeys(
                        note for index in range(len(targets)) for note in batches[index].validation_notes
                    )),
                )
                return validate_plan(candidate, inputs.context, inputs.assets)
            except ValueError as error:
                affected = set(getattr(error, "task_ids", ()))
                pending = [index for index, target in enumerate(targets) if affected.intersection(target)]
                localized = bool(pending)
                if not localized:
                    pending = list(range(len(targets)))
                self.store.append_event(
                    inputs.context.challenge_id, "planning", "error",
                    exception_type=type(error).__name__, result_summary=str(error),
                    arguments={"repair_scope": "dependencies" if localized else "challenge",
                               "target_task_ids": [key for index in pending for key in targets[index]]},
                )
                if any(repairs[index] >= limits.max_format_repairs for index in pending):
                    raise
                for index in pending:
                    graph = {
                        plan.task_id: [dependency.model_dump() for dependency in plan.dependencies]
                        for batch in batches.values() for plan in batch.plans
                    }
                    correction = (
                        "Correct this merged-plan error in target_task_ids only: " + str(error)
                        + ". Preserve other batches. Current dependency graph: "
                        + json.dumps(graph, ensure_ascii=False)
                    )
                    batches[index], repairs[index] = self._plan_batch(
                        inputs, targets[index], started, correction, repairs[index] + 1,
                    )

    def generate(self, challenge_id: str, feedback: str = "") -> ChallengePlan:
        """Generate and validate one candidate, reusing saved plans after a clean restart.
        Args: challenge_id: Manifest challenge; feedback: Explicit correction after a failure.
        """
        with self._lock:
            state = self.store.load_state(challenge_id)
            if state.plan is not None:
                if feedback:
                    raise ValueError("an installed plan cannot be regenerated")
                return state.plan.model_copy(deep=True)
            if state.planning_status == "generated" and not feedback:
                data = json.loads(self.store.read_planning_artifact(challenge_id, state.plan_ref))
                inputs = self.store.dataset.read_challenge(challenge_id)
                return validate_plan(ChallengePlan.model_validate(data), inputs.context, inputs.assets)
            if state.planning_status == "running":
                raise RuntimeError("planning was interrupted; preserve the attempt and use a new run")
            if state.planning_status == "error" and not feedback:
                raise RuntimeError("planning failed; supply explicit feedback or use a new run")
            ids = self.store.dataset.get_context(challenge_id).task_ids
            if any(self.store.load_state(key).stage != "pending" or self.store.has_phase(key, "A") for key in ids):
                raise ValueError("plan before starting A")
            state.planning_status = "running"
            self.store.save_state(state)
            started = monotonic()
            try:
                inputs = self.store.dataset.read_challenge(challenge_id)
                candidate = self._generate_candidate(inputs, started, feedback)
                event = self.store.append_event(challenge_id, "planning", "ok", result_summary="Complete candidate validated; awaiting Scheduler installation.")
                state = self.store.load_state(challenge_id)
                state.plan_ref = self.store.write_artifact(event.event_id, "candidate_plan.json", candidate.model_dump_json())
                state.planning_status = "generated"
                self.store.save_state(state)
                return candidate
            except Exception as error:
                state = self.store.load_state(challenge_id)
                state.planning_status = "error"
                self.store.save_state(state)
                self.store.append_event(challenge_id, "planning", "error", exception_type=type(error).__name__, result_summary=str(error))
                raise
            finally:
                state = self.store.load_state(challenge_id)
                state.counters.duration_seconds += monotonic() - started
                self.store.save_state(state)
