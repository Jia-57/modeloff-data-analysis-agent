"""Command-line entry for a local Benchmark A run or explicit legacy migration."""

from __future__ import annotations

import argparse
import json
import os
import signal
import traceback
from collections import Counter
from pathlib import Path
from time import monotonic

from .agents import run_benchmark_A
from .config import load_config
from .dataset import Dataset
from .model import LocalModel, warmup_model
from .planning import ChallengePlanner, planning_reserve_seconds
from .runtime import RuntimeStore
from .scheduling import Scheduler
from .types import ChallengePlan, Plan, RunConfig


def estimate_run(config: RunConfig, dataset: Dataset, workflow: str = "first_round") -> dict:
    """Estimate serial work from the selected workflow's configured phase limits."""
    tasks, challenges = len(dataset.task_ids), len(dataset.challenge_ids)
    work_seconds = tasks * config.budget_A.max_seconds + challenges * config.planning.max_seconds
    if workflow == "reflection":
        work_seconds = tasks * (config.recovery.round1.max_seconds + config.recovery.round2.max_seconds)
    return {"tasks": tasks, "challenges": challenges, "phase_budget_seconds": work_seconds,
            "margin_seconds": 7200, "recommended_job_seconds": work_seconds + 7200}


class RunControl:
    """Stop admitting work on a job signal or deadline without interrupting database writes."""

    def __init__(self, max_seconds: float) -> None:
        """Start the runner clock before connecting to the local model service."""
        self.started = monotonic()
        self.deadline = self.started + max_seconds
        self.stop_reason = None

    def handle_signal(self, signum, frame) -> None:
        """Request a graceful stop; the current bounded invocation may finish first."""
        self.stop_reason = f"signal:{signal.Signals(signum).name}"

    def remaining(self) -> float:
        """Return the time still available for model calls and new phases."""
        return 0.0 if self.stop_reason else max(0.0, self.deadline - monotonic())

    def can_start(self, required_seconds: float) -> bool:
        """Reserve a whole phase so the last task is not started with a partial budget."""
        if self.remaining() < required_seconds:
            self.stop_reason = self.stop_reason or "insufficient_time_for_next_phase"
            return False
        return True


class JobModel:
    """Use one existing model while capping its calls at the overall job deadline."""

    def __init__(self, model, control: RunControl) -> None:
        """Reuse the model and its tokenizer without creating another inference engine."""
        self.model, self.control, self.tokenizer = model, control, model.tokenizer

    def count_tokens(self, messages: list[dict]) -> int:
        """Count the actual context on CPU and check for a stop request on both sides."""
        if self.control.remaining() <= 0:
            raise TimeoutError("job deadline reached")
        result = self.model.count_tokens(messages)
        if self.control.remaining() <= 0:
            raise TimeoutError("job deadline reached during context preparation")
        return result

    def generate(self, messages: list[dict], *, remaining_seconds: float, **options):
        """Pass the smaller phase or job allowance to the local model adapter."""
        available = min(remaining_seconds, self.control.remaining())
        if available <= 0:
            raise TimeoutError("job deadline reached")
        return self.model.generate(messages, remaining_seconds=available, **options)


def write_report(path: Path, value, *, jsonl: bool = False) -> None:
    """Atomically replace a derived report; SQLite remains the authoritative record."""
    text = ("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in value)
            if jsonl else json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary = path.with_name(path.name + ".pending")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def fallback_plan(store: RuntimeStore, challenge_id: str, reason: str) -> ChallengePlan:
    """Permit original-input ReAct after a failed optional model plan, without invented dependencies."""
    ids = store.dataset.get_context(challenge_id).task_ids
    plans = []
    for key in ids:
        inputs = store.dataset.read_task(key)
        plans.append(Plan(task_id=key, goal="Solve the original question from authorized inputs",
                          steps=["Inspect only needed inputs, calculate and verify the result"],
                          answer_format="as requested by the question",
                          data_sources=["question", "introduction", *[a.file_id for a in inputs.assets]]))
    return ChallengePlan(challenge_id=challenge_id, plans=plans,
                         validation_notes=["Host fallback after planning failure: " + reason[:300]])


def run_A(store: RuntimeStore, max_seconds: float) -> int:
    """Run the complete manifest, export checkpoints and isolate challenge planning failures."""
    control = RunControl(max_seconds)
    previous = {sig: signal.signal(sig, control.handle_signal)
                for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT)}
    report = {"run_directory": str(store.root), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
              "partition": os.environ.get("SLURM_JOB_PARTITION"),
              "estimate": estimate_run(store.config, store.dataset), "planning": {}, "errors": [],
              "global_budget_seconds": max_seconds, "gold_read": False, "stage": "initialization"}
    rows = {}

    def update_task(key: str) -> None:
        """Read only this task's checkpoint, retaining its explicit failed or frozen outcome."""
        state = store.load_state(key)
        frozen = store.has_phase(key, "A")
        answer = store.load_phase(key, "A").answer_A if frozen else None
        terminal = state.terminal_status.get("A", "pending") if frozen else (
            "interrupted" if state.stage != "pending" else "not_run")
        rows[key] = {"task_id": key, "challenge_id": state.challenge_id, "A_frozen": frozen,
                     "terminal_status": terminal, "answer": answer.model_dump() if answer else None,
                     "counters": state.counters["A"].model_dump()}

    def export() -> None:
        """Save every selected task after each plan or task, without rereading full histories."""
        outputs = []
        for row in rows.values():
            reason = None if row["A_frozen"] else (
                report["planning"].get(row["challenge_id"], {}).get("error")
                or control.stop_reason or "not dispatched")
            outputs.append({**row, "not_run_reason": reason})
        frozen = sum(row["A_frozen"] for row in outputs)
        answered = sum(bool(row["answer"] and row["answer"]["status"] == "answered") for row in outputs)
        report.update(duration_seconds=round(monotonic() - control.started, 3),
                      stop_reason=control.stop_reason,
                      counts={"selected": len(outputs), "A_frozen": frozen, "answered": answered,
                              "terminal_statuses": dict(Counter(row["terminal_status"] for row in outputs))})
        report["status"] = ("failed" if report.get("fatal_error") else
                            "partial" if frozen != len(outputs) else
                            "completed" if answered == len(outputs) else "completed_with_missing_answers")
        write_report(store.root / "answers_A.jsonl", outputs, jsonl=True)
        write_report(store.root / "run_summary.json", report)

    def can_start(key: str) -> bool:
        """Admit a task only when its full configured A budget remains."""
        if not control.can_start(store.config.budget_A.max_seconds):
            return False
        report["stage"] = "react:" + key
        print(f"TASK START {key}", flush=True)
        return True

    def on_finished(key: str) -> None:
        """Export a newly frozen outcome, including legitimate null benchmark results."""
        update_task(key)
        row = rows[key]
        print(f"TASK END {key}: {row['terminal_status']}; answer=" +
              json.dumps(row["answer"], ensure_ascii=False), flush=True)
        export()

    try:
        for key in store.dataset.task_ids:
            update_task(key)
        export()
        issues = store.dataset.validate_inputs()
        if issues:
            raise ValueError(f"input validation failed: {issues}")
        report["stage"] = "model_loading"
        model = JobModel(LocalModel(store.config.model, store.config.seed), control)
        report["warmup"] = warmup_model(store, model.model, min(90, control.remaining()))
        planner = ChallengePlanner(store, model.generate, model.count_tokens)
        scheduler = Scheduler(store)
        for challenge_id in store.dataset.challenge_ids:
            # Reinstalling an accepted plan consumes no inference budget on resume.
            accepted = store.load_state(challenge_id).plan is not None
            reserve = planning_reserve_seconds(
                store.config, len(store.dataset.get_context(challenge_id).task_ids))
            if not accepted and not control.can_start(reserve):
                break
            report["stage"] = "planning:" + challenge_id
            print(f"PLAN START {challenge_id}", flush=True)
            try:
                candidate = planner.generate(challenge_id) if not accepted else store.load_state(challenge_id).plan
                scheduler.install_plan(candidate)
                report["planning"][challenge_id] = {"status": "validated"}
                print(f"PLAN OK {challenge_id}", flush=True)
            except Exception as error:
                trace = traceback.format_exc()
                report["planning"][challenge_id] = {"status": "fallback", "error": str(error)}
                report["errors"].append({"stage": report["stage"], "traceback": trace})
                print(trace, flush=True)
                try:
                    scheduler.install_plan(fallback_plan(store, challenge_id, str(error)))
                    print(f"PLAN FALLBACK {challenge_id}; first ReAct remains available", flush=True)
                except Exception:
                    report["planning"][challenge_id]["status"] = "failed"
                    report["errors"].append({"stage": "planning_fallback:" + challenge_id,
                                             "traceback": traceback.format_exc()})
            export()
        run_benchmark_A(store, model, scheduler, can_start_fn=can_start,
                        on_finished_fn=on_finished, allow_partial=True)
        report["stage"] = "finished"
    except Exception as error:
        report["fatal_error"] = {"stage": report["stage"], "message": str(error),
                                 "traceback": traceback.format_exc()}
        print(report["fatal_error"]["traceback"], flush=True)
    finally:
        try:
            # Include an interrupted dispatch if execution stopped outside the normal callback.
            for key in store.dataset.task_ids:
                update_task(key)
            export()
            print(json.dumps(report, ensure_ascii=False), flush=True)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    return 2 if report.get("fatal_error") else 1 if report["status"] == "partial" else 0


def main(argv=None) -> int:
    """Load configuration, acquire a run and execute on an allocated CUDA node."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/Project_planning_and_ReAct.yaml"))
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--run-id", help="Override the configured ID; existing runs are never overwritten.")
    parser.add_argument("--manifest", help="Use a project-scoped manifest under data/preprocessed/task_inputs/.")
    parser.add_argument("--max-seconds", type=float, help="Overall runner deadline, including model warmup.")
    parser.add_argument("--estimate-only", action="store_true", help="Count the full manifest and phase limits; no model or run files.")
    parser.add_argument("--workflow", choices=("first_round", "reflection"), default="first_round",
                        help="Select the first round or recovery of its unanswered tasks.")
    parser.add_argument("--baseline-run-dir", type=Path,
                        help="Read-only first-round run for --workflow reflection.")
    parser.add_argument("--recovery-task-ids", type=str,
                        help="Comma-separated unanswered task IDs for a recovery pilot; default all active unanswered tasks.")
    parser.add_argument("--recovery-project-id", type=str,
                        help="Recover all unanswered tasks in one active project; exclusive with --recovery-task-ids.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", help="Reuse frozen outcomes; interrupted attempts are not restarted.")
    mode.add_argument("--migrate-legacy", action="store_true", help="Import an existing JSON run and exit without loading a model.")
    args = parser.parse_args(argv)
    config = load_config(args.config, args.project_root)
    payload = config.model_dump()
    if args.run_id:
        payload["run_id"] = args.run_id
    if args.manifest:
        payload["paths"]["manifest"] = args.manifest
    if os.environ.get("AGENT_MODEL_PATH"):
        payload["model"]["model_path"] = os.environ["AGENT_MODEL_PATH"]
    if config.model.backend == "vllm" and os.environ.get("AGENT_MODEL_API_BASE"):
        payload["model"]["api_base"] = os.environ["AGENT_MODEL_API_BASE"]
    config = RunConfig.model_validate(payload)
    estimate = estimate_run(config, Dataset.from_config(config, args.project_root), args.workflow)
    if args.estimate_only:
        print(json.dumps(estimate, ensure_ascii=False))
        return 0
    max_seconds = args.max_seconds if args.max_seconds is not None else estimate["recommended_job_seconds"]
    if not 0 < max_seconds < float("inf"):
        parser.error("--max-seconds must be finite and positive")
    if args.workflow == "reflection":
        if args.baseline_run_dir is None or args.run_id is None:
            parser.error("reflection requires --baseline-run-dir and a fresh --run-id")
        if args.recovery_project_id and args.recovery_task_ids:
            parser.error("select either --recovery-project-id or --recovery-task-ids")
        from .reflection import run_recovery
        selection = args.recovery_task_ids.split(",") if args.recovery_task_ids else None
        return run_recovery(config, args.project_root, args.baseline_run_dir, max_seconds,
                            resume=args.resume, task_ids=selection,
                            project_id=args.recovery_project_id)
    if args.baseline_run_dir or args.recovery_task_ids or args.recovery_project_id:
        parser.error("recovery selection options require --workflow reflection")
    factory = RuntimeStore.migrate_legacy if args.migrate_legacy else (
        RuntimeStore.resume if args.resume else RuntimeStore.create
    )
    with factory(config, args.project_root) as store:
        if args.migrate_legacy:
            print(json.dumps({"database": str(store.root / "runtime.sqlite"), "status": "migrated"}))
            return 0
        print(f"RUN_DIR={store.root}", flush=True)
        return run_A(store, max_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
