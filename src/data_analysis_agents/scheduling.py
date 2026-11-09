"""Validate challenge plans and dispatch tasks against frozen A history."""

from __future__ import annotations

from threading import RLock
from pathlib import Path
from typing import Optional

from .runtime import RuntimeStore
from .types import (
    ArtifactRef, AssetInfo, ChallengeContext, ChallengePlan, Plan, TaskDispatch,
    Variant, WorkspacePolicy,
)


class PlanValidationError(ValueError):
    """A plan error with task IDs identifying the output batches that need repair."""

    def __init__(self, message: str, task_ids: tuple[str, ...] = ()) -> None:
        """Attach affected tasks; an empty tuple means the error cannot be localized."""
        super().__init__(message)
        self.task_ids = tuple(dict.fromkeys(task_ids))


def _cycle_members(parents: dict[str, set[str]]) -> tuple[str, ...]:
    """Find actual cycle members, excluding tasks that merely depend on a cycle."""
    members = []
    for target in sorted(parents):
        pending, seen = list(parents[target]), set()
        while pending:
            source = pending.pop()
            if source == target:
                members.append(target)
                break
            if source not in seen:
                seen.add(source)
                pending.extend(parents[source])
    return tuple(members)


def validate_plan(
    plan: ChallengePlan, context: ChallengeContext, assets: list[AssetInfo]
) -> ChallengePlan:
    """Check coverage, sources, dependencies, and workspace policies; return manifest order.
    Args: plan: Candidate output; context: Manifest task order; assets: Allowed file IDs.
    """
    plan = ChallengePlan.model_validate(plan)
    entries = {item.task_id: item for item in plan.plans}
    if plan.challenge_id != context.challenge_id or set(entries) != set(context.task_ids):
        raise PlanValidationError("plan must cover exactly this challenge's manifest tasks")
    positions = {key: index for index, key in enumerate(context.task_ids)}
    allowed_sources = {"question", "introduction", *(asset.file_id for asset in assets)}
    has_tables = any(asset.reference.role in ("table", "workbook") and
                     Path(asset.reference.path).suffix.lower() in (".csv", ".xlsx", ".xlsm", ".xlsb")
                     for asset in assets)
    parents = {}
    for item in plan.plans:
        # A current-task ID is an unambiguous alias for the current question only.
        sources = list(dict.fromkeys("question" if source == item.task_id else source
                                     for source in item.data_sources))
        item = item.model_copy(update={"data_sources": sources})
        entries[item.task_id] = item
        if not item.goal.strip() or not item.answer_format.strip() or not item.steps or any(not step.strip() for step in item.steps):
            raise PlanValidationError(f"{item.task_id}: goal, steps, and answer format must be nonempty", (item.task_id,))
        if not set(item.data_sources) <= allowed_sources:
            raise PlanValidationError(f"{item.task_id}: unknown data source; allowed sources are "
                                      f"{sorted(allowed_sources)}", (item.task_id,))
        edges = [(d.source_task_id, d.kind) for d in item.dependencies]
        if len(edges) != len(set(edges)) or any(source not in entries for source, _ in edges):
            raise PlanValidationError(f"{item.task_id}: duplicate or unknown dependency", (item.task_id,))
        parents[item.task_id] = {source for source, _ in edges}
        policy = item.workspace_policy
        if policy.snapshot_ref is not None:
            raise PlanValidationError(f"{item.task_id}: snapshot references are assigned by the host", (item.task_id,))
        workspace_edges = {edge for edge in edges if edge[1] in ("continue_workspace", "reset_workspace")}
        if workspace_edges and not has_tables:
            raise PlanValidationError(
                f"{item.task_id}: workspace dependencies require CSV/Excel attachments; "
                "use prior_text/prior_result for shared conditions, results or simulation logic", (item.task_id,))
        required = set()
        if policy.source_task_id is not None:
            kind = "continue_workspace" if policy.mode == "continue" else "reset_workspace"
            required.add((policy.source_task_id, kind))
        if workspace_edges != required:
            raise PlanValidationError(f"{item.task_id}: workspace dependency does not match its policy", (item.task_id,))
    # Separate the DAG check from execution waits: prior_text authorizes text only.
    remaining = {key: set(value) for key, value in parents.items()}
    while remaining:
        roots = {key for key, value in remaining.items() if not value}
        if not roots:
            cycle = _cycle_members(remaining)
            raise PlanValidationError("dependency cycle: " + ", ".join(cycle), cycle)
        remaining = {key: value - roots for key, value in remaining.items() if key not in roots}
    for target, sources in parents.items():
        if any(positions[source] >= positions[target] for source in sources):
            raise PlanValidationError(f"{target}: dependencies must reference earlier manifest questions", (target,))
    return ChallengePlan.model_validate({
        **plan.model_dump(), "plans": [entries[key].model_dump() for key in context.task_ids],
    })


class Scheduler:
    """One host dispatcher per run; workers freeze results before notifying it."""

    def __init__(self, store: RuntimeStore) -> None:
        """Restore accepted plans and classify outcomes without retrying interrupted tasks."""
        self.store = store
        self._lock = RLock()
        self._running: set[tuple[str, str]] = set()
        self._progress = store.progress()
        self._frozen = {
            (key, variant) for key in store.dataset.task_ids for variant in ("A", "B")
            if store.has_phase(key, variant)
        }
        self._plans: dict[str, Plan] = {}
        for challenge_id in store.dataset.challenge_ids:
            state = store.load_state(challenge_id)
            if state.plan is not None:
                accepted = self._validate(state.plan)
                self._plans.update({item.task_id: item for item in accepted.plans})
                self._update_challenge(challenge_id)

    def _validate(self, plan: ChallengePlan) -> ChallengePlan:
        """Validate using only this run's manifest and attachment metadata."""
        context = self.store.dataset.get_context(plan.challenge_id)
        return validate_plan(plan, context, self.store.dataset.list_assets(context.task_ids[0]))

    def install_plan(self, plan: ChallengePlan) -> None:
        """Publish a validated plan once; repeating the same installation is harmless."""
        with self._lock:
            plan = self._validate(plan)
            state = self.store.load_state(plan.challenge_id)
            if state.plan is not None:
                if state.plan != plan:
                    raise ValueError("installed plans cannot be replaced; use a new run")
            else:
                task_ids = self.store.dataset.get_context(plan.challenge_id).task_ids
                if any(self._progress[key]["A"] != "pending" for key in task_ids):
                    raise ValueError("install the plan before starting any A task")
                event = self.store.append_event(plan.challenge_id, "planning", "ok", result_summary="Challenge plan validated and installed.")
                state.plan_ref = self.store.write_artifact(event.event_id, "accepted_plan.json", plan.model_dump_json())
                state.plan, state.planning_status = plan, "validated"
                self.store.save_state(state)
            self._plans.update({item.task_id: item for item in plan.plans})
            self._update_challenge(plan.challenge_id)

    def _eligible(self, variant: Variant) -> list[str]:
        """Find pending tasks, enforcing waits for results and the global A/B barrier."""
        if variant not in ("A", "B"):
            raise ValueError("variant must be A or B")
        if variant == "B" and any((key, "A") not in self._frozen for key in self.store.dataset.task_ids):
            return []
        ready = []
        for key in self.store.dataset.task_ids:
            if key not in self._plans or (key, variant) in self._running or (key, variant) in self._frozen:
                continue
            if self._progress[key][variant] != "pending":
                continue
            waiting = [d.source_task_id for d in self._plans[key].dependencies if d.kind != "prior_text"]
            if all((source, "A") in self._frozen for source in waiting):
                ready.append(key)
        return ready

    def ready_tasks(self, variant: Variant = "A") -> list[str]:
        """Return stable manifest-ordered candidates within the remaining worker capacity."""
        with self._lock:
            capacity = max(0, self.store.config.task_concurrency - len(self._running))
            return self._eligible(variant)[:capacity]

    def _check_selection(self, reference: ArtifactRef, allowed_sources: set[str]) -> ArtifactRef:
        """Verify an explicitly selected artifact against its source's frozen A record."""
        reference = ArtifactRef.model_validate(reference)
        if reference.variant != "A" or reference.task_id not in allowed_sources:
            raise PermissionError("artifact is outside the authorized A history")
        source = self.store.load_phase(reference.task_id, "A")
        if reference not in source.artifacts:
            raise PermissionError("artifact is not listed in the frozen source state")
        self.store.artifact_path(reference)
        return reference

    def prepare_task(
        self, task_id: str, variant: Variant = "A", *,
        selected_artifacts: Optional[list[ArtifactRef]] = None,
    ) -> TaskDispatch:
        """Reserve one task and persist its exact authorized context before execution.
        Args: task_id: Ready task; variant: A/B; selected_artifacts: Explicit host choices.
        """
        with self._lock:
            if len(self._running) >= self.store.config.task_concurrency or task_id not in self._eligible(variant):
                raise ValueError("task is not ready or worker capacity is exhausted")
            plan = self._plans[task_id]
            state = self.store.load_state(task_id)
            result_sources = {d.source_task_id for d in plan.dependencies if d.kind != "prior_text"}
            selected = [self._check_selection(ref, result_sources | ({task_id} if variant == "B" else set()))
                        for ref in (selected_artifacts or [])]
            if len({ref.path for ref in selected}) != len(selected):
                raise ValueError("duplicate selected artifacts")
            questions, answers, statuses, exports, diagnostics = {}, {}, {}, [], []
            for dependency in plan.dependencies:
                source_id = dependency.source_task_id
                if dependency.kind == "prior_text":
                    questions[source_id] = self.store.dataset.read_previous_question(task_id, source_id, {source_id})
                    continue
                if source_id in answers:
                    continue
                source = self.store.load_phase(source_id, "A")
                answers[source_id], statuses[source_id] = source.answer_A, source.terminal_status["A"]
                if source.terminal_status["A"] != "completed":
                    diagnostics.append(f"{source_id}: A ended with {source.terminal_status['A']}; recompute if needed.")
                event_ids = {event.event_id for event in self.store.iter_events(source.challenge_id, source_id, "A")}
                for export in source.dependency_exports:
                    if export.event_id not in event_ids:
                        raise ValueError("dependency export has no matching source event")
                    for ref in export.artifact_refs:
                        self._check_selection(ref, {source_id})
                    if export.snapshot_ref is not None and export.snapshot_ref != self.store.load_snapshot(source_id, "A"):
                        raise ValueError("dependency export has a mismatched snapshot")
                    exports.append(export)
            policy = plan.workspace_policy
            if policy.source_task_id is not None:
                policy = WorkspacePolicy.model_validate({
                    **policy.model_dump(), "snapshot_ref": self.store.load_snapshot(policy.source_task_id, "A").model_dump(),
                })
                diagnostics.append("The A snapshot is a database record. Use the explicitly selected workbook artifact; missing or ambiguous versions cannot be replaced by original inputs.")
            own_A, own_snapshot = None, None
            if variant == "A":
                state.plan = plan
                state.dependency_ids = list(dict.fromkeys(d.source_task_id for d in plan.dependencies))
                state.workspace_policy, state.prior_A_exports = policy, exports
                state.reusable_artifact_refs = selected
            else:
                frozen = self.store.load_phase(task_id, "A")
                if frozen.plan != plan:
                    raise ValueError("B must use the plan frozen with this task's A")
                own_A, own_snapshot = frozen.answer_A, self.store.load_snapshot(task_id, "A")
                policy, exports = frozen.workspace_policy, frozen.prior_A_exports
            dispatch = TaskDispatch(
                task_input=self.store.dataset.read_task(task_id), variant=variant, plan=plan,
                workspace_policy=policy, previous_questions=questions, previous_answers=answers,
                previous_statuses=statuses, prior_A_exports=exports, selected_artifacts=selected,
                answer_A=own_A, source_A_snapshot=own_snapshot, diagnostics=diagnostics,
            )
            self._running.add((task_id, variant))
            self._progress[task_id][variant] = "interrupted"  # Never silently retry a failed dispatch publication.
            try:
                state.stage = "react" if variant == "A" else "reflection"
                self.store.save_state(state)
                event = self.store.append_event(state.challenge_id, state.stage, "started", task_id=task_id, variant=variant, result_summary="Task context dispatched.")
                reference = self.store.write_artifact(event.event_id, "dispatch.json", dispatch.model_dump_json())
                state.artifacts.append(reference)
                self.store.save_state(state)
            except BaseException:
                self._running.discard((task_id, variant))
                raise
            return dispatch

    def _update_challenge(self, challenge_id: str) -> None:
        """Rebuild A summaries from frozen outcomes, including recovery after a lost callback."""
        state = self.store.load_state(challenge_id)
        ids = self.store.dataset.get_context(challenge_id).task_ids
        for key in ids:
            if (key, "A") in self._frozen:
                source = self.store.load_phase(key, "A")
                state.terminal_statuses[key] = source.terminal_status["A"]
                state.task_snapshots[key] = self.store.load_snapshot(key, "A")
        state.ready_task_ids = [key for key in self._eligible("A") if key in ids]
        self.store.save_state(state)

    def on_task_finished(self, task_id: str, variant: Variant = "A") -> None:
        """Release a worker only after RuntimeStore has published its frozen outcome."""
        with self._lock:
            if not self.store.has_phase(task_id, variant):
                raise ValueError("freeze the task outcome before notifying Scheduler")
            was_frozen = (task_id, variant) in self._frozen
            self._frozen.add((task_id, variant))
            self._running.discard((task_id, variant))
            frozen = self.store.load_phase(task_id, variant)
            self._progress[task_id][variant] = frozen.terminal_status[variant]
            self._update_challenge(frozen.challenge_id)
            if not was_frozen:
                self.store.append_event(
                    frozen.challenge_id, "scheduling", "ok",
                    arguments={"task_id": task_id, "variant": variant},
                    snapshot_ref=self.store.load_snapshot(task_id, variant),
                    result_summary="Frozen task outcome acknowledged.",
                )
