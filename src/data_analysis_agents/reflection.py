"""Blind, separately persisted recovery of unanswered first-round tasks.

The baseline is read-only. Each recovery round has its own RuntimeStore, so the
first-round ReAct agent and its existing repair code can run without reopening frozen A.
"""
from __future__ import annotations

import hashlib
import json
import signal
import traceback
from collections import Counter
from pathlib import Path
from time import monotonic
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .agents import ReActAgent
from .answer_validation import needs_computation
from .cli import JobModel, RunControl, write_report
from .dataset import Dataset
from .model import LocalModel, warmup_model
from .runtime import RuntimeStore
from .types import Answer, Observation, RunConfig, TaskDispatch, validate_component, validate_relative_path
from .reflection_baseline import Baseline
from .reflection_handoff import (
    INSPECTION_TOOLS, LAYOUT_TOOLS, REPEAT_INSPECTION, CURRENT_INSPECTION,
    KNOWN_LAYOUT, COMPUTE_RESERVE, RECOVERY_GUARDS, _answered, _recovery_priority,
    _inspection_signatures, _layout_sources, _attempt_handoff,
)


WORKBOOK_SUFFIXES = {".csv", ".xlsx", ".xlsm", ".xlsb"}
def _round_config(base: RunConfig, round_number: int) -> RunConfig:
    data = base.model_dump()
    data["paths"]["runs_dir"] = "runs/reflection"
    data["experiment_id"] = base.run_id
    data["run_id"] = f"r{round_number}"
    data["recovery_run"] = True
    data["budget_A"] = base.recovery.round1.model_dump() if round_number == 1 else base.recovery.round2.model_dump()
    data["model"]["action_max_new_tokens"] = base.recovery.action_tokens
    data["model"]["python_max_new_tokens"] = base.recovery.python_tokens
    data["model"]["format_repair_max_new_tokens"] = base.recovery.format_repair_tokens
    # The output is audited after execution. No hidden label is available here.
    return RunConfig.model_validate(data)


def _store_artifacts(store: RuntimeStore, source: str) -> list:
    if not store.has_phase(source, "A"):
        return []
    completed = {event.parent_event_id for event in store.iter_events(source.split("/")[0], source, "A")
                 if event.status == "ok" and event.parent_event_id}
    imported = {event.event_id for event in store.iter_events(source.split("/")[0], source, "A")
                if event.status == "ok" and (event.arguments or {}).get("recovery_import")}
    refs = [ref for ref in store.load_phase(source, "A").artifacts
            if Path(ref.path).suffix.lower() in WORKBOOK_SUFFIXES
            and ref.event_id in completed | imported]
    return refs if len(refs) == 1 else []


def _import_artifacts(store: RuntimeStore, task_id: str, items: list[tuple[str, Path]], *,
                      selected_suffixes: set[str] | None = None) -> list:
    """Copy an authorized earlier artifact with explicit origin; never share error notes."""
    copied = []
    for relative, source in items:
        if selected_suffixes and source.suffix.lower() not in selected_suffixes:
            continue
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        event = store.append_event(task_id.split("/")[0], "react", "ok", task_id=task_id, variant="A",
                                   arguments={"recovery_import": relative, "sha256": digest},
                                   result_summary="Verified prior file copied into this recovery run")
        copied.append(store.write_artifact(event.event_id, source.name, source))
    if copied:
        state = store.load_state(task_id)
        state.artifacts.extend(copied)
        store.save_state(state)
    return copied


def _reusable_evidence(items: list[tuple[str, Path]]) -> list[tuple[str, Path]]:
    """Carry bounded calculation outputs into a fresh recovery tool session."""
    return [(relative, path) for relative, path in items
            if path.name in {"computed_result.json", "stdout.txt"} or
            path.name.startswith("output-")][-8:]


def _freeze_source(store: RuntimeStore, baseline: Baseline, source: str, dataset: Dataset,
                   previous: RuntimeStore | None = None) -> None:
    if store.has_phase(source, "A"):
        return
    original = baseline.artifacts(source)
    if previous is not None and previous.has_phase(source, "A"):
        original = [(ref.path, previous.artifact_path(ref)) for ref in _store_artifacts(previous, source)] or original
    workbooks = [(name, path) for name, path in original if path.suffix.lower() in WORKBOOK_SUFFIXES]
    if len(workbooks) != 1:
        return
    state = store.load_state(source)
    state.plan = baseline.plan(source, dataset)
    _import_artifacts(store, source, workbooks, selected_suffixes=WORKBOOK_SUFFIXES)
    state = store.load_state(source)
    answer = (previous.load_phase(source, "A").answer_A if previous and previous.has_phase(source, "A")
              else baseline.answer(source))
    state.answer_A = answer or Answer(task_id=source, answer=None, status="no_final_answer",
                                      brief_explanation="Prior calculation supplied an artifact, not an answer")
    state.terminal_status["A"] = "completed" if state.answer_A.status == "answered" else "no_answer"
    state.stage = "freeze_A"
    store.freeze_phase(state, "A")


def _diagnostics(baseline: Baseline, task_id: str, round_number: int,
                 previous: RuntimeStore | None, limit: int, artifact_only: bool) -> list[str]:
    lines = baseline.evidence(task_id, limit)
    if _recovery_priority(baseline, task_id) == 0:
        lines.append("A completed calculation is already saved. Check its scope and option matching, then finish; do not repeat inspection or computation unless the saved result is invalid.")
    lines.append("Recovery is blind to correctness labels and gold answers. Use original data and actual tool results.")
    if artifact_only:
        lines.append("A later question requires a saved workbook from this task. Save the required CSV/Excel file in output_dir and finish the original question; the baseline answer will remain unchanged.")
    if previous is not None and previous.has_phase(task_id, "A"):
        state = previous.load_phase(task_id, "A")
        events = [event.model_dump() for event in previous.iter_events(
            state.challenge_id, task_id, "A")]
        artifacts = [(ref.path, previous.artifact_path(ref)) for ref in state.artifacts]
        lines.extend(_attempt_handoff(
            events, artifacts, label="R1", status=state.terminal_status["A"],
            explanation=state.answer_A.brief_explanation, limit=limit))
        if round_number == 2:
            lines.append("Use the new observation or a genuinely different computation. Do not repeat an identical failed operation.")
    return lines


def _prepare_dispatch(store: RuntimeStore, baseline: Baseline, task_id: str, round_number: int,
                      previous: RuntimeStore | None, artifact_only: bool) -> TaskDispatch | None:
    dataset = store.dataset
    plan = baseline.plan(task_id, dataset)
    selected = []
    source = plan.workspace_policy.source_task_id
    if source:
        _freeze_source(store, baseline, source, dataset, previous)
        selected = _store_artifacts(store, source)
        if not selected:
            return None
    if round_number == 2 and previous is not None and previous.has_phase(task_id, "A"):
        prior = previous.load_phase(task_id, "A")
        files = [(ref.path, previous.artifact_path(ref)) for ref in prior.artifacts]
        _import_artifacts(store, task_id, _reusable_evidence(files))
    else:
        _import_artifacts(store, task_id, _reusable_evidence(baseline.artifacts(task_id)))
    state = store.load_state(task_id)
    state.plan = plan
    state.stage = "react"
    state.dependency_ids = [dep.source_task_id for dep in plan.dependencies]
    state.workspace_policy = plan.workspace_policy
    state.reusable_artifact_refs = selected
    store.save_state(state)
    previous_questions = {dep.source_task_id: dataset.read_previous_question(
        task_id, dep.source_task_id, {dep.source_task_id})
        for dep in plan.dependencies if dep.kind == "prior_text"}
    previous_statuses = {dep.source_task_id: baseline.rows[dep.source_task_id]["terminal_status"]
                         for dep in plan.dependencies if dep.kind != "prior_text"
                         and dep.source_task_id in baseline.rows}
    return TaskDispatch(task_input=dataset.read_task(task_id), variant="A", plan=plan,
                        workspace_policy=plan.workspace_policy, previous_questions=previous_questions,
                        previous_statuses=previous_statuses, selected_artifacts=selected,
                        diagnostics=_diagnostics(baseline, task_id, round_number, previous,
                                                 store.config.recovery.evidence_chars, artifact_only))


def _new_evidence(store: RuntimeStore, baseline: Baseline, task_id: str) -> bool:
    """R2 requires actual new execution evidence, not a repeated empty answer."""
    if not store.has_phase(task_id, "A"):
        return False
    state = store.load_phase(task_id, "A")
    imported = {event.event_id for event in store.iter_events(state.challenge_id, task_id, "A")
                if (event.arguments or {}).get("recovery_import")}
    if any(Path(ref.path).name.startswith("output-") and ref.event_id not in imported
           for ref in state.artifacts):
        return True
    old_codes, old_observations = baseline.prior_signatures(task_id)
    for event in store.iter_events(state.challenge_id, task_id, "A"):
        if event.tool_name == "execute_python" and event.status == "started":
            code = (event.arguments or {}).get("code")
            if code and code.strip() not in old_codes:
                return True
    return any(obs.status in ("ok", "error") and
               obs.result_summary.strip() not in old_observations
               for obs in state.observations)


class RecoveryAgent(ReActAgent):
    """first-round solver with guards for prior failed code and completed inspections."""

    def __init__(self, store, model, failed_codes: dict[str, set[str]],
                 seen_inspections: dict[str, set[str]],
                 seen_layout_sources: dict[str, set[str]]):
        super().__init__(store, model.generate, model.count_tokens, model.tokenizer)
        self.failed_codes = failed_codes
        self.seen_inspections = seen_inspections
        self.seen_layout_sources = seen_layout_sources

    def _blocked_tool(self, python, action, failed):
        if action.tool_name == "execute_python":
            code = action.arguments.get("code")
            if isinstance(code, str) and code.strip() in self.failed_codes.get(python.task_id, set()):
                return "The identical Python code already failed in an earlier attempt. Change the computation."
        signature = json.dumps([action.tool_name, action.arguments], sort_keys=True, ensure_ascii=False)
        if signature in self.seen_inspections.get(python.task_id, set()):
            return REPEAT_INSPECTION
        current_events = [event.model_dump() for event in self.store.iter_events(
            python.challenge_id, python.task_id, "A")]
        if signature in _inspection_signatures(current_events):
            return CURRENT_INSPECTION
        if action.tool_name in LAYOUT_TOOLS:
            source = action.arguments.get("file_id") or action.arguments.get("artifact_id")
            if source and (str(source) in self.seen_layout_sources.get(python.task_id, set()) or
                           str(source) in _layout_sources(current_events)):
                return KNOWN_LAYOUT
        if action.tool_name in INSPECTION_TOOLS:
            state = self.store.load_state(python.task_id)
            counter = state.counters.get("A")
            has_result = any(Path(ref.path).name == "computed_result.json" for ref in state.artifacts)
            if (counter and counter.tool_calls >= self.store.config.budget_A.max_tool_calls - 1
                    and not has_result and needs_computation(
                        self.store.dataset.read_task(python.task_id).question)):
                return COMPUTE_RESERVE
        return super()._blocked_tool(python, action, failed)

    def _denied(self, task_id: str, name: str, arguments: dict, reason: str) -> Observation:
        """Log host-blocked repeat inspections without spending the computation call."""
        if reason not in RECOVERY_GUARDS:
            return super()._denied(task_id, name, arguments, reason)
        state = self.store.load_state(task_id)
        event = self.store.append_event(state.challenge_id, "react", "denied", task_id=task_id,
                                        variant="A", tool_name=name, arguments=arguments,
                                        result_summary=reason)
        observation = Observation(event_id=event.event_id, status="denied", result_summary=reason)
        state.observations.append(observation)
        self.store.save_state(state)
        return observation


class RecoveryGraphState(TypedDict, total=False):
    stage: str
    rounds_finished: list[int]


class RecoveryWorkflow:
    """Project stages only; each question's attempt and counters live in its round store."""

    def __init__(self, base: RunConfig, project_root: Path, baseline: Baseline,
                 model, control: RunControl, selected: list[str], resume: bool):
        self.base, self.root, self.baseline = base, project_root, baseline
        self.model, self.control, self.selected, self.resume = model, control, selected, resume
        self.run_dir = project_root / "runs/reflection" / base.run_id
        self.stores: dict[int, RuntimeStore] = {}
        self.issues: dict[str, dict] = {}
        self.outcomes: dict[str, dict] = {}
        graph = StateGraph(RecoveryGraphState)
        graph.add_node("diagnose", self.diagnose)
        graph.add_node("round1", self.round1)
        graph.add_node("round2", self.round2)
        graph.add_node("finalize", self.finalize)
        graph.add_edge(START, "diagnose")
        graph.add_edge("diagnose", "round1")
        graph.add_edge("round1", "round2")
        graph.add_edge("round2", "finalize")
        graph.add_edge("finalize", END)
        self.graph = graph.compile()

    def close(self):
        for store in self.stores.values():
            store.close()

    def _store(self, round_number: int) -> RuntimeStore:
        if round_number not in self.stores:
            config = _round_config(self.base, round_number)
            path = self.run_dir / f"r{round_number}"
            self.stores[round_number] = (RuntimeStore.resume(config, self.root) if path.exists()
                                          else RuntimeStore.create(config, self.root))
        return self.stores[round_number]

    def diagnose(self, state: RecoveryGraphState) -> dict:
        dataset = Dataset.from_config(self.base, self.root)
        for cid in dataset.challenge_ids:
            ids = [key for key in dataset.get_context(cid).task_ids if key in self.selected]
            if not ids:
                continue
            rows = [self.baseline.rows[key] for key in dataset.get_context(cid).task_ids]
            absent = [key for key in ids if not self.baseline.rows[key].get("A_frozen")]
            missing_workbooks = {}
            for key in ids:
                source = self.baseline.plan(key, dataset).workspace_policy.source_task_id
                if source and not any(path.suffix.lower() in WORKBOOK_SUFFIXES
                                      for _, path in self.baseline.artifacts(source)):
                    missing_workbooks.setdefault(source, []).append(key)
            self.issues[cid] = {"selected_unanswered": ids, "unstarted": absent,
                                "all_unanswered": not any(_answered(row) for row in rows),
                                "missing_workbooks": missing_workbooks}
        self._export()
        return {"stage": "diagnosed"}

    def _round(self, number: int, selected: list[str]) -> None:
        if not selected or self.control.remaining() <= 0:
            return
        store = self._store(number)
        dataset = store.dataset
        selected_set = set(selected)
        previous = self.stores.get(1) if number == 2 else None
        # A selected downstream question may require several missing saved
        # workbooks. Include the entire source chain and execute it upstream first.
        work, visited, visiting = [], set(), set()

        def visit(task_id: str) -> None:
            if task_id in visited:
                return
            if task_id in visiting:
                raise ValueError(f"cyclic saved-workspace dependency: {task_id}")
            visiting.add(task_id)
            source = self.baseline.plan(task_id, dataset).workspace_policy.source_task_id
            if source:
                if source.split("/")[0] != task_id.split("/")[0]:
                    raise ValueError(f"cross-project saved-workspace dependency: {task_id}")
                earlier = previous and len(_store_artifacts(previous, source)) == 1
                original = [path for _, path in self.baseline.artifacts(source)
                            if path.suffix.lower() in WORKBOOK_SUFFIXES]
                if not earlier and len(original) != 1:
                    visit(source)
            visiting.remove(task_id)
            visited.add(task_id)
            work.append(task_id)

        order = (sorted(selected_set, key=lambda key: (_recovery_priority(self.baseline, key),
                                                        dataset.task_ids.index(key)))
                 if number == 1 else [key for key in dataset.task_ids if key in selected_set])
        for task_id in order:
            try:
                visit(task_id)
            except Exception as exc:
                self.outcomes.setdefault(task_id, {})[f"r{number}"] = (
                    f"dependency_planning_error:{type(exc).__name__}:{exc}")
                visiting.clear()
        failed_codes = {key: self.baseline.failed_codes(key) for key in work}
        seen_inspections = {key: self.baseline.inspections(key) for key in work}
        seen_layout_sources = {key: self.baseline.layout_sources(key) for key in work}
        if previous:
            for key in work:
                if not previous.has_phase(key, "A"):
                    continue
                events = list(previous.iter_events(key.split("/")[0], key, "A"))
                seen_inspections[key].update(_inspection_signatures(
                    [event.model_dump() for event in events]))
                seen_layout_sources[key].update(_layout_sources(
                    [event.model_dump() for event in events]))
                failures = {event.parent_event_id for event in events
                            if event.tool_name == "execute_python" and
                            event.status in ("error", "denied", "timeout")}
                failed_codes[key].update(str(event.arguments.get("code")).strip() for event in events
                                         if event.event_id in failures and event.arguments.get("code"))
        agent = RecoveryAgent(store, self.model, failed_codes,
                              seen_inspections, seen_layout_sources)
        for task_id in work:
            if self.control.remaining() <= 0:
                break
            if store.has_phase(task_id, "A"):
                frozen = store.load_phase(task_id, "A")
                self.outcomes.setdefault(task_id, {})[f"r{number}"] = frozen.answer_A.status
                continue
            old_state = store.load_state(task_id)
            if old_state.stage != "pending":
                self.outcomes.setdefault(task_id, {})[f"r{number}"] = "interrupted_attempt_not_restarted"
                continue
            artifact_only = task_id not in selected_set
            budget = store.config.budget_A
            if not self.control.can_start(budget.max_seconds):
                break
            print(f"R{number} START {task_id}" + (" artifact" if artifact_only else ""), flush=True)
            answer = None
            try:
                dispatch = _prepare_dispatch(store, self.baseline, task_id, number, previous, artifact_only)
                if dispatch is None:
                    outcome = "required_saved_workspace_unavailable"
                else:
                    answer = agent.run(dispatch)
                    outcome = answer.status
                    if artifact_only and not _store_artifacts(store, task_id):
                        outcome = "source_answered_without_required_workbook"
            except Exception as exc:
                outcome = f"recovery_exception:{type(exc).__name__}:{exc}"
            self.outcomes.setdefault(task_id, {})[f"r{number}"] = outcome
            value = None if answer is None or answer.answer is None else answer.answer.value
            reason = (answer.brief_explanation if answer else outcome)[:500]
            print(f"R{number} END {task_id}: {outcome}; answer=" +
                  json.dumps(value, ensure_ascii=False)[:600] +
                  "; reason=" + json.dumps(reason, ensure_ascii=False), flush=True)
            self._export()

    def round1(self, state: RecoveryGraphState) -> dict:
        self._round(1, self.selected)
        return {"stage": "round1_finished", "rounds_finished": [1]}

    def round2(self, state: RecoveryGraphState) -> dict:
        first = self.stores.get(1)
        if first is None:
            return {"stage": "round2_skipped"}
        retry = [key for key in self.selected if first.has_phase(key, "A")
                 and not _answered({"answer": first.load_phase(key, "A").answer_A.model_dump()})
                 and _new_evidence(first, self.baseline, key)]
        for issue in self.issues.values():
            for source, dependents in issue["missing_workbooks"].items():
                # A missing required workbook is a concrete, actionable blocker
                # even when the source's first attempt produced no new tool result.
                if first.has_phase(source, "A") and (
                        _new_evidence(first, self.baseline, source) or
                        not _store_artifacts(first, source)):
                    retry.extend(key for key in dependents if key in self.selected and
                                 key not in retry and not (first.has_phase(key, "A") and
                                 first.load_phase(key, "A").answer_A.status == "answered"))
        self._round(2, retry)
        return {"stage": "round2_finished", "rounds_finished": [1, 2]}

    def _export(self) -> None:
        rows = []
        for key in self.selected:
            row = {"task_id": key, "baseline_status": self.baseline.rows[key].get("terminal_status"),
                   "rounds": self.outcomes.get(key, {})}
            for number in (1, 2):
                store = self.stores.get(number)
                if store and store.has_phase(key, "A"):
                    phase = store.load_phase(key, "A")
                    row[f"r{number}"] = {"status": phase.terminal_status["A"],
                                         "answer": phase.answer_A.model_dump(),
                                         "counters": phase.counters["A"].model_dump()}
            rows.append(row)
        write_report(self.run_dir / "recovery_progress.json", {
            "baseline_run": str(self.baseline.root), "selected": self.selected,
            "project_issues": self.issues, "tasks": rows, "gold_read": False})

    def finalize(self, state: RecoveryGraphState) -> dict:
        dataset = Dataset.from_config(self.base, self.root)
        merged = []
        for key in dataset.task_ids:
            row = dict(self.baseline.rows[key])
            if key in self.selected and not _answered(row):
                for number in (2, 1):
                    store = self.stores.get(number)
                    if store and store.has_phase(key, "A"):
                        phase = store.load_phase(key, "A")
                        if phase.answer_A.status == "answered":
                            row.update(answer=phase.answer_A.model_dump(), terminal_status="completed",
                                       A_frozen=True, recovery_round=number,
                                       recovery_run=str(store.root))
                            break
            merged.append(row)
        write_report(self.run_dir / "answers_A.jsonl", merged, jsonl=True)
        write_report(self.run_dir / "run_summary.json", {
            "workflow": "reflection", "baseline_run": str(self.baseline.root),
            "selected": len(self.selected), "recovered_answered": sum("recovery_round" in r for r in merged),
            "answered_total": sum(_answered(r) for r in merged),
            "terminal_statuses": dict(Counter(r.get("terminal_status") for r in merged)),
            "status": "partial" if any(
                key not in self.outcomes or
                self.outcomes[key].get("r1") == "interrupted_attempt_not_restarted"
                for key in self.selected) else "completed",
            "gold_read": False, "job_remaining_seconds": round(self.control.remaining(), 2)})
        self._export()
        return {"stage": "finished"}


def run_recovery(base: RunConfig, project_root: Path, baseline_dir: Path,
                 max_seconds: float, *, resume: bool = False, task_ids: list[str] | None = None,
                 project_id: str | None = None, model=None) -> int:
    """Run recovery without writing to the baseline or loading evaluator labels."""
    root = project_root.resolve()
    dataset = Dataset.from_config(base, root)
    baseline = Baseline(baseline_dir, dataset)
    control = RunControl(max_seconds)
    previous_signals = {sig: signal.signal(sig, control.handle_signal)
                        for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT)}
    workflow = None
    try:
        if project_id and task_ids is not None:
            raise ValueError("select a project or explicit tasks, not both")
        if project_id:
            validate_component(project_id)
            if project_id not in dataset.challenge_ids:
                raise ValueError(f"project is not active: {project_id}")
        wanted = (task_ids if task_ids is not None else
                  [key for key in dataset.task_ids if baseline.rows[key].get("A_frozen")
                   and not _answered(baseline.rows[key])
                   and (project_id is None or key.split("/")[0] == project_id)])
        if not wanted:
            print("REFLECTION SKIPPED: no attempted unanswered tasks", flush=True)
            return 0
        if len(wanted) != len(set(wanted)) or any(key not in dataset.task_ids or
                not baseline.rows[key].get("A_frozen") or _answered(baseline.rows[key]) for key in wanted):
            raise ValueError("recovery selection must contain distinct attempted unanswered tasks")
        run_dir = root / "runs/reflection" / validate_component(base.run_id)
        if run_dir.exists() and not resume:
            raise FileExistsError(f"recovery run already exists: {run_dir}; use --resume")
        if resume and not run_dir.exists():
            raise FileNotFoundError(f"recovery run does not exist: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        inputs = hashlib.sha256()
        sources = {dataset.get_task(key).question_file.path for key in dataset.task_ids}
        sources.update(dataset.get_task(key).introduction_file.path for key in dataset.task_ids)
        sources.update(ref.path for key in dataset.task_ids for ref in dataset.get_task(key).allowed_assets)
        for relative in sorted(sources):
            inputs.update(relative.encode("utf-8"))
            inputs.update((root / relative).read_bytes())
        identity_config = base.model_dump()
        # The server's loopback port is allocation-specific, not a solver setting.
        identity_config["model"].pop("api_base", None)
        identity = {"baseline": str(baseline.root), "baseline_manifest_sha256": baseline.manifest_hash,
                    "baseline_answers_sha256": baseline.answer_hash,
                    "baseline_runtime_sha256": baseline.db_hash, "current_input_sha256": inputs.hexdigest(),
                    "active_task_ids": list(dataset.task_ids), "selected": wanted,
                    "config": identity_config, "workflow": "reflection-1"}
        identity_path = run_dir / "recovery_identity.json"
        if identity_path.exists():
            if json.loads(identity_path.read_text()) != identity:
                raise ValueError("recovery identity changed; use a new run_id")
        else:
            write_report(identity_path, identity)
            write_report(run_dir / "manifest.jsonl", [dataset.get_task(key).model_dump()
                                                         for key in dataset.task_ids], jsonl=True)
        print(f"RUN_DIR={run_dir}", flush=True)
        print(f"ENV OK recovery inputs: active={len(dataset.task_ids)}; unanswered_selected={len(wanted)}",
              flush=True)
        if model is None:
            model = LocalModel(base.model, base.seed)
        bound = JobModel(model, control)
        workflow = RecoveryWorkflow(base, root, baseline, bound, control, wanted, resume)
        if base.model_warmup and isinstance(model, LocalModel):
            print("ENV START model warmup", flush=True)
            warmup = warmup_model(workflow._store(1), model, min(90, control.remaining()))
            required = {"text"}
            if any(asset.reference.role == "image" for key in wanted
                   for asset in dataset.list_assets(key)):
                required.add("image")
            healthy = {item["kind"] for item in warmup if item["status"] == "ok"}
            if not required <= healthy:
                raise RuntimeError("model warmup failed; check the model service and resume this recovery run")
            print("ENV OK model warmup", flush=True)
        workflow.graph.invoke({"stage": "start", "rounds_finished": []})
        summary = json.loads((workflow.run_dir / "run_summary.json").read_text())
        return 1 if summary["status"] == "partial" else 0
    except Exception as exc:
        print(traceback.format_exc(), flush=True)
        if workflow is not None:
            write_report(workflow.run_dir / "run_summary.json", {
                "workflow": "reflection", "status": "failed", "baseline_run": str(baseline.root),
                "fatal_error": {"type": type(exc).__name__, "message": str(exc)}, "gold_read": False})
        return 2
    finally:
        if workflow is not None:
            workflow.close()
        baseline.close()
        for sig, handler in previous_signals.items():
            signal.signal(sig, handler)
