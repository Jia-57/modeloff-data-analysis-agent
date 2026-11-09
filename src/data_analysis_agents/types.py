"""Dataset, plan, state, event, answer, and configuration contracts.

Contracts describe access boundaries. Dataset and tool adapters must enforce
filesystem permissions; frozen models do not freeze nested containers or files.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Annotated, Literal, Optional, Union

from pydantic import AfterValidator, BeforeValidator, Field, JsonValue, model_validator

from .schema import MutableRecord, Record


Variant = Literal["A", "B"]
Stage = Literal["pending", "planning", "scheduling", "react", "freeze_A", "reflection", "freeze_B", "completed"]
TerminalStatus = Literal[
    "completed", "no_answer", "no_final_answer", "timeout", "step_limit",
    "execution_error", "interrupted", "material_unresolved", "planning_failed",
]
EventStatus = Literal["started", "ok", "error", "denied", "timeout"]
DependencyKind = Literal["prior_text", "prior_result", "continue_workspace", "reset_workspace"]
AssetRole = Literal["question", "introduction", "workbook", "table", "image", "supplemental"]


def validate_component(value: str) -> str:
    """Check a safe identifier or directory name.
    Args: value: One path component.
    """
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"invalid identifier: {value!r}")
    return value


def validate_relative_path(path: str) -> str:
    """Check a lexical project/run-relative path.
    Args: path: POSIX path without absolute or traversal components.
    """
    if "\\" in path or "\x00" in path or any(part in ("", ".", "..") for part in path.split("/")):
        raise ValueError(f"unsafe relative path: {path!r}")
    return path


def make_task_id(challenge_id: str, question_id: str) -> str:
    """Build the stable manifest task ID without renaming its components.
    Args: challenge_id: Challenge key; question_id: question followed by digits.
    """
    validate_component(challenge_id)
    if not re.fullmatch(r"question\d+", question_id, flags=re.IGNORECASE):
        raise ValueError(f"invalid question_id: {question_id!r}")
    return f"{challenge_id}/{question_id}"


def validate_task_id(task_id: str, challenge_id: Optional[str] = None) -> str:
    """Check task identity and optional challenge membership.
    Args: task_id: Manifest ID; challenge_id: Expected challenge, if supplied.
    """
    parts = task_id.split("/")
    if len(parts) != 2 or make_task_id(*parts) != task_id:
        raise ValueError(f"invalid task_id: {task_id!r}")
    if challenge_id is not None and parts[0] != challenge_id:
        raise ValueError(f"task {task_id!r} does not belong to {challenge_id!r}")
    return task_id


def _strict_bool(value: object) -> bool:
    """Reject numeric lookalikes for fixed boolean policies.
    Args: value: Unparsed policy flag.
    """
    if type(value) is not bool:
        raise ValueError("policy flags must be booleans")
    return value


Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")]
TaskID = Annotated[str, AfterValidator(validate_task_id)]
RelativePath = Annotated[str, AfterValidator(validate_relative_path)]
InputPath = Annotated[RelativePath, Field(pattern=r"^data/raw_data/")]
NonNegativeInt = Annotated[int, Field(ge=0)]
NonNegativeFloat = Annotated[float, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(gt=0)]
One = Annotated[int, Field(ge=1, le=1)]
RequiredTrue = Annotated[Literal[True], BeforeValidator(_strict_bool)]
RequiredFalse = Annotated[Literal[False], BeforeValidator(_strict_bool)]


class FileRef(Record):
    """A project-relative input file listed in the manifest."""

    path: InputPath
    role: AssetRole


class Task(Record):
    """One manifest row, containing references rather than question contents."""

    task_id: TaskID
    challenge_id: Identifier
    question_id: str
    question_file: FileRef
    introduction_file: FileRef
    allowed_assets: list[FileRef] = Field(default_factory=list)
    challenge_name: Optional[str] = None
    year: Optional[int] = None

    @model_validator(mode="after")
    def check_protocol(self) -> Task:
        """Check manifest identity, roles, and challenge scope; no explicit arguments."""
        if self.task_id != make_task_id(self.challenge_id, self.question_id):
            raise ValueError("task_id must match challenge_id/question_id")
        if self.question_file.role != "question" or self.introduction_file.role != "introduction":
            raise ValueError("question/introduction file roles do not match")
        for reference in [self.question_file, self.introduction_file, *self.allowed_assets]:
            if not reference.path.startswith(f"data/raw_data/{self.challenge_id}/"):
                raise ValueError("input file belongs to another challenge")
        if any(item.role in ("question", "introduction") for item in self.allowed_assets):
            raise ValueError("allowed_assets must contain attachments only")
        return self


class ArtifactRef(Record):
    """A run-relative artifact with execution provenance."""

    path: RelativePath
    task_id: TaskID
    event_id: Identifier
    variant: Variant
    snapshot_id: Optional[Identifier] = None
    assumptions: list[str] = Field(default_factory=list)


class SnapshotRef(Record):
    """An immutable database record reference; it is not a workbook copy."""

    snapshot_id: Identifier
    path: RelativePath
    task_id: TaskID
    variant: Variant
    source_snapshot_id: Optional[Identifier] = None


class Dependency(Record):
    """A required predecessor; the containing Plan identifies the target task."""

    source_task_id: TaskID
    kind: DependencyKind


class WorkspacePolicy(Record):
    """Select original inputs, an upstream workspace, or a reset point."""

    mode: Literal["original", "continue", "reset"] = "original"
    source_task_id: Optional[TaskID] = None
    snapshot_ref: Optional[SnapshotRef] = None

    @model_validator(mode="after")
    def check_protocol(self) -> WorkspacePolicy:
        """Check the workspace source; no explicit arguments."""
        if self.snapshot_ref is not None:
            if self.snapshot_ref.variant != "A":
                raise ValueError("workspace sources must refer to A snapshots")
            if self.source_task_id is not None and self.source_task_id != self.snapshot_ref.task_id:
                raise ValueError("workspace task and snapshot source do not match")
        has_source = self.source_task_id is not None or self.snapshot_ref is not None
        if self.mode == "original" and has_source:
            raise ValueError("original workspace cannot specify an upstream source")
        if self.mode == "continue" and not has_source:
            raise ValueError("continue workspace requires a source")
        return self


class Plan(Record):
    """The current task's challenge-plan fragment."""

    task_id: TaskID
    goal: str
    steps: list[str]
    answer_format: str
    data_sources: list[str] = Field(default_factory=list)
    dependencies: list[Dependency] = Field(default_factory=list)
    workspace_policy: WorkspacePolicy = Field(default_factory=WorkspacePolicy)
    reusable_candidates: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_protocol(self) -> Plan:
        """Reject self-dependencies and cross-challenge sources; no explicit arguments."""
        challenge_id = self.task_id.split("/")[0]
        for dependency in self.dependencies:
            validate_task_id(dependency.source_task_id, challenge_id)
            if dependency.source_task_id == self.task_id:
                raise ValueError("a task cannot depend on itself")
        source = self.workspace_policy.source_task_id
        if self.workspace_policy.snapshot_ref is not None:
            source = self.workspace_policy.snapshot_ref.task_id
        if source is not None and source not in {item.source_task_id for item in self.dependencies}:
            raise ValueError("workspace source must be an explicit dependency")
        return self


class ChallengePlan(Record):
    """One challenge's plan; Scheduler later checks task existence and the DAG."""

    challenge_id: Identifier
    plans: list[Plan]
    validation_notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_protocol(self) -> ChallengePlan:
        """Check task scope and duplicate plan entries; no explicit arguments."""
        task_ids = [item.task_id for item in self.plans]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("duplicate task plans")
        for task_id in task_ids:
            validate_task_id(task_id, self.challenge_id)
        return self


class TokenUsage(Record):
    """Actual processed input and generated output counts; null means unknown."""

    input_tokens: Optional[NonNegativeInt] = None
    output_tokens: Optional[NonNegativeInt] = None


class GenerationResult(Record):
    """Model response used by an injected generator, without importing inference libraries."""

    text: str
    reasoning: Optional[str] = None
    raw_response: Optional[dict[str, JsonValue]] = None
    request_parameters: Optional[dict[str, JsonValue]] = None
    tokens: TokenUsage = Field(default_factory=TokenUsage)
    finish_reason: Literal["stop", "length"] = "stop"


class UsageCounters(MutableRecord):
    """One phase's usage; missing token usage invalidates the respective total."""

    model_calls: NonNegativeInt = 0
    tool_calls: NonNegativeInt = 0
    tool_duration_seconds: NonNegativeFloat = 0.0
    input_tokens: Optional[NonNegativeInt] = 0
    output_tokens: Optional[NonNegativeInt] = 0
    duration_seconds: NonNegativeFloat = 0.0


class AnswerValue(Record):
    """A scalar or composite JSON answer with its optional unit."""

    value: JsonValue
    unit: Optional[str] = None


class ReportedValue(Record):
    """A key result actually cited in the answer explanation."""

    value: JsonValue
    evidence_ref: str
    unit: Optional[str] = None


class Answer(Record):
    """The same public answer contract for A and B; no gold or score fields."""

    task_id: TaskID
    answer: Optional[AnswerValue]
    brief_explanation: str = ""
    evidence_refs: list[str] = Field(default_factory=list)
    reported_values: list[ReportedValue] = Field(default_factory=list)
    status: Literal["answered", "no_answer", "no_final_answer"] = "answered"
    computed_value: Optional[float] = None
    computed_unit: Optional[str] = None
    computed_value_source: Optional[Literal["tool", "model"]] = None
    option_consistency: Literal["not_checked", "matched", "corrected", "unresolved"] = "not_checked"

    @model_validator(mode="after")
    def check_protocol(self) -> Answer:
        """Check answer identity and presence; no explicit arguments."""
        if self.status == "answered" and (self.answer is None or self.answer.value is None):
            raise ValueError("answered status requires an answer value")
        if self.status != "answered" and self.answer is not None:
            raise ValueError("unanswered status requires a null answer")
        return self


class FinalConsistencyCheck(Record):
    """Describe internal answer agreement without changing a choice or claiming correctness."""

    status: Literal["consistent", "conflict", "unresolved", "not_checked"] = "not_checked"
    reasons: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    option_claims: list[str] = Field(default_factory=list)
    expected_option: Optional[str] = None


class DependencyExport(Record):
    """An internal A result authorized for downstream use."""

    name: str
    value: JsonValue
    task_id: TaskID
    event_id: Identifier
    applicability: str
    unit: Optional[str] = None
    evidence_refs: list[str] = Field(default_factory=list)
    snapshot_ref: Optional[SnapshotRef] = None
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_protocol(self) -> DependencyExport:
        """Require A-only provenance from the exporting task; no explicit arguments."""
        references = [*self.artifact_refs]
        if self.snapshot_ref is not None:
            references.append(self.snapshot_ref)
        if any(item.variant != "A" or item.task_id != self.task_id for item in references):
            raise ValueError("dependency exports must reference their source task's A artifacts")
        return self


class Reflection(Record):
    """One reflection's checks, decision, and supporting references."""

    decision: Literal["keep", "revise"]
    checks: list[str] = Field(default_factory=list)
    rationale: str = ""
    evidence_refs: list[str] = Field(default_factory=list)


class Observation(Record):
    """A short tool observation; the complete output remains in an artifact."""

    event_id: Identifier
    status: EventStatus
    result_summary: str
    result_ref: Optional[ArtifactRef] = None
    truncated: bool = False
    image_ref: Optional[ArtifactRef] = None


class Event(Record):
    """An append-only event payload; runtime assigns IDs and timestamps."""

    experiment_id: Identifier
    run_id: Identifier
    challenge_id: Identifier
    event_id: Identifier
    timestamp: str
    stage: Stage
    status: EventStatus
    task_id: Optional[TaskID] = None
    variant: Optional[Variant] = None
    parent_event_id: Optional[Identifier] = None
    tool_name: Optional[str] = None
    arguments: dict[str, JsonValue] = Field(default_factory=dict)
    source_path: Optional[str] = None
    source_range: Optional[str] = None
    dependency_ids: list[TaskID] = Field(default_factory=list)
    snapshot_ref: Optional[SnapshotRef] = None
    mcp_service: Optional[str] = None
    mcp_version: Optional[str] = None
    exception_type: Optional[str] = None
    result_summary: Optional[str] = None
    result_ref: Optional[RelativePath] = None
    tokens: Optional[TokenUsage] = None
    duration_seconds: Optional[NonNegativeFloat] = None
    trace_id: Optional[str] = None
    observation_id: Optional[str] = None

    @model_validator(mode="after")
    def check_protocol(self) -> Event:
        """Validate event scope and timing; no explicit arguments."""
        timestamp = datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
        if timestamp.utcoffset() != timedelta(0):
            raise ValueError("event timestamp must include a UTC offset")
        if self.task_id is not None:
            validate_task_id(self.task_id, self.challenge_id)
            if self.variant is None:
                raise ValueError("task events require an A/B variant")
        elif self.variant is not None:
            raise ValueError("shared challenge events cannot have an A/B variant")
        for task_id in self.dependency_ids:
            validate_task_id(task_id, self.challenge_id)
        if self.snapshot_ref is not None:
            validate_task_id(self.snapshot_ref.task_id, self.challenge_id)
        return self


class ChallengeContext(Record):
    """Shared background and directory; expose only authorized slices to solvers."""

    challenge_id: Identifier
    introduction_file: FileRef
    task_ids: list[TaskID]
    allowed_assets: list[FileRef] = Field(default_factory=list)
    workbook_metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_protocol(self) -> ChallengeContext:
        """Check shared input scope; no explicit arguments."""
        if self.introduction_file.role != "introduction":
            raise ValueError("challenge context requires an introduction file")
        for task_id in self.task_ids:
            validate_task_id(task_id, self.challenge_id)
        for reference in [self.introduction_file, *self.allowed_assets]:
            if not reference.path.startswith(f"data/raw_data/{self.challenge_id}/"):
                raise ValueError("challenge context contains another challenge's input")
        if any(item.role in ("question", "introduction") for item in self.allowed_assets):
            raise ValueError("shared allowed_assets must contain attachments only")
        return self


class AssetInfo(Record):
    """Attachment metadata; file IDs are mapped to paths by the host."""

    file_id: Identifier
    reference: FileRef
    size_bytes: NonNegativeInt

    @model_validator(mode="after")
    def check_protocol(self) -> AssetInfo:
        """Check attachment identity, role, and size; no explicit arguments."""
        if self.reference.role in ("question", "introduction"):
            raise ValueError("attachment metadata cannot describe question or introduction files")
        return self


class TaskInput(Record):
    """Current-question input for ReAct, without the challenge's question list."""

    task: Task
    question: str
    introduction: str
    assets: list[AssetInfo] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_protocol(self) -> TaskInput:
        """Require metadata to match the task's attachment whitelist; no explicit arguments."""
        _check_asset_metadata(self.assets, self.task.allowed_assets)
        return self


class ChallengeInput(Record):
    """All challenge questions for one Plan-and-Solve planning stage."""

    context: ChallengeContext
    introduction: str
    questions: dict[TaskID, str]
    assets: list[AssetInfo] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_protocol(self) -> ChallengeInput:
        """Require questions and assets to match the challenge context; no explicit arguments."""
        if set(self.questions) != set(self.context.task_ids):
            raise ValueError("planner input must contain the challenge's declared questions")
        _check_asset_metadata(self.assets, self.context.allowed_assets)
        return self


def _check_asset_metadata(assets: list[AssetInfo], allowed: list[FileRef]) -> None:
    """Require unique file IDs and exact declared file references.
    Args: assets: Exposed metadata; allowed: Manifest attachment whitelist.
    """
    identifiers = [item.file_id for item in assets]
    references = [item.reference for item in assets]
    if len(identifiers) != len(set(identifiers)) or len(references) != len(set(references)):
        raise ValueError("duplicate attachment metadata")
    if set(references) != set(allowed):
        raise ValueError("attachment metadata must match its whitelist, including file roles")


class ChallengeState(MutableRecord):
    """Challenge progress and shared planning usage, separate from task usage."""

    challenge_id: Identifier
    thread_id: str
    challenge_context_ref: RelativePath
    plan: Optional[ChallengePlan] = None
    planning_status: Literal["pending", "running", "generated", "validated", "error"] = "pending"
    plan_ref: Optional[RelativePath] = None
    ready_task_ids: list[TaskID] = Field(default_factory=list)
    terminal_statuses: dict[TaskID, TerminalStatus] = Field(default_factory=dict)
    baseline_workspace: Optional[RelativePath] = None
    task_snapshots: dict[TaskID, SnapshotRef] = Field(default_factory=dict)
    reusable_artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    counters: UsageCounters = Field(default_factory=UsageCounters)

    @model_validator(mode="after")
    def check_protocol(self) -> ChallengeState:
        """Check challenge-owned state and A baseline references; no explicit arguments."""
        if self.plan is not None and self.plan.challenge_id != self.challenge_id:
            raise ValueError("plan belongs to another challenge")
        for task_id in [*self.ready_task_ids, *self.terminal_statuses, *self.task_snapshots]:
            validate_task_id(task_id, self.challenge_id)
        for task_id, snapshot in self.task_snapshots.items():
            if snapshot.task_id != task_id or snapshot.variant != "A":
                raise ValueError("baseline snapshots must match their task and variant A")
        for artifact in self.reusable_artifact_refs:
            validate_task_id(artifact.task_id, self.challenge_id)
            if artifact.variant != "A":
                raise ValueError("shared artifacts must come from A")
        return self


class TaskState(MutableRecord):
    """Paired task records; persisted by runtime and filtered before model access."""

    task_id: TaskID
    challenge_id: Identifier
    challenge_context_ref: RelativePath
    dependency_ids: list[TaskID] = Field(default_factory=list)
    workspace_policy: WorkspacePolicy = Field(default_factory=WorkspacePolicy)
    snapshot_ref: Optional[SnapshotRef] = None
    prior_A_exports: list[DependencyExport] = Field(default_factory=list)
    reusable_artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    allowed_assets: list[FileRef] = Field(default_factory=list)
    plan: Optional[Plan] = None
    observations: list[Observation] = Field(default_factory=list)
    stage: Stage = "pending"
    counters: dict[Variant, UsageCounters] = Field(default_factory=dict)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    dependency_exports: list[DependencyExport] = Field(default_factory=list)
    answer_A: Optional[Answer] = None
    reflection: Optional[Reflection] = None
    answer_B: Optional[Answer] = None
    terminal_status: dict[Variant, TerminalStatus] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_protocol(self) -> TaskState:
        """Check task ownership and authorized predecessor inputs; no explicit arguments."""
        validate_task_id(self.task_id, self.challenge_id)
        for task_id in self.dependency_ids:
            validate_task_id(task_id, self.challenge_id)
            if task_id == self.task_id:
                raise ValueError("a task cannot authorize itself as a predecessor")
        if self.snapshot_ref is not None and self.snapshot_ref.task_id != self.task_id:
            if self.snapshot_ref.variant != "A" or self.snapshot_ref.task_id not in self.dependency_ids:
                raise ValueError("upstream snapshots require an authorized A dependency")
        source = self.workspace_policy.source_task_id
        if self.workspace_policy.snapshot_ref is not None:
            source = self.workspace_policy.snapshot_ref.task_id
        if source is not None and source not in self.dependency_ids:
            raise ValueError("workspace source requires an authorized dependency")
        for export in self.prior_A_exports:
            if export.task_id not in self.dependency_ids:
                raise ValueError("previous A export requires an authorized dependency")
        for export in self.dependency_exports:
            if export.task_id != self.task_id:
                raise ValueError("outgoing export belongs to another task")
        for reference in self.allowed_assets:
            if not reference.path.startswith(f"data/raw_data/{self.challenge_id}/"):
                raise ValueError("task assets belong to another challenge")
            if reference.role in ("question", "introduction"):
                raise ValueError("task allowed_assets must contain attachments only")
        for artifact in self.reusable_artifact_refs:
            if artifact.variant != "A" or artifact.task_id not in self.dependency_ids:
                raise ValueError("reusable inputs require an authorized predecessor A artifact")
        if self.plan is not None and self.plan.task_id != self.task_id:
            raise ValueError("plan belongs to another task")
        for answer in (self.answer_A, self.answer_B):
            if answer is not None and answer.task_id != self.task_id:
                raise ValueError("answer belongs to another task")
        if self.answer_B is not None and self.answer_A is None:
            raise ValueError("B requires a saved A outcome, including a failed A outcome")
        if self.reflection is not None and self.reflection.decision == "keep" and self.answer_B is not None:
            if (
                self.answer_A is None
                or self.answer_A.answer != self.answer_B.answer
                or self.answer_A.status != self.answer_B.status
            ):
                raise ValueError("keep must preserve A's answer value and answer status")
        return self


class Budget(Record):
    """Per-phase limits, distinct from measured usage."""

    max_model_calls: PositiveInt = 12
    max_tool_calls: NonNegativeInt = 12
    max_seconds: PositiveFloat = 300.0
    max_format_repairs: NonNegativeInt = 1
    final_reserve_seconds: NonNegativeFloat = 10.0


class PlanningConfig(Record):
    """Shared planning limits; max_format_repairs is a per-batch validation allowance."""

    batch_size: PositiveInt = 8
    max_new_tokens: PositiveInt = 4096
    max_model_calls: PositiveInt = 16
    max_seconds: PositiveFloat = 900.0
    max_format_repairs: NonNegativeInt = 1


class RecoveryConfig(Record):
    """Additional, separately recorded attempts for unanswered first-round tasks."""

    round1: Budget = Field(default_factory=lambda: Budget(
        max_model_calls=6, max_tool_calls=4, max_seconds=240,
        max_format_repairs=1, final_reserve_seconds=60))
    round2: Budget = Field(default_factory=lambda: Budget(
        max_model_calls=4, max_tool_calls=3, max_seconds=180,
        max_format_repairs=1, final_reserve_seconds=60))
    action_tokens: PositiveInt = 768
    python_tokens: PositiveInt = 2048
    format_repair_tokens: PositiveInt = 512
    evidence_chars: PositiveInt = 3500



class ModelConfig(Record):
    """Shared local Transformers settings; weights are loaded in a later stage."""

    backend: Literal["transformers", "vllm"] = "transformers"
    stream_responses: bool = False
    model_id: Annotated[str, Field(min_length=1)] = "Qwen/Qwen3-VL-8B-Instruct"
    model_path: Optional[str] = None
    revision: Optional[str] = None
    dtype: Literal["auto", "bfloat16", "float16", "float32"] = "auto"
    device_map: str = "auto"
    quantization: Optional[str] = None
    do_sample: bool = False
    temperature: Optional[PositiveFloat] = None
    max_new_tokens: PositiveInt = 1024
    action_max_new_tokens: Optional[PositiveInt] = None
    python_max_new_tokens: Optional[PositiveInt] = None
    python_runtime_repairs: NonNegativeInt = 1
    format_repair_max_new_tokens: Optional[PositiveInt] = None
    api_base: str = "http://127.0.0.1:8000/v1"
    served_model_name: str = "qwen38-int4"
    enable_thinking: bool = False
    context_limit: PositiveInt = 12000
    image_max_pixels: PositiveInt = 1048576
    structured_output: bool = False
    repair_structured_output: bool = True

    @model_validator(mode="after")
    def check_protocol(self) -> ModelConfig:
        """Check output/context limits and sampling settings; no explicit arguments."""
        if self.max_new_tokens >= self.context_limit:
            raise ValueError("output budget must fit the context limit")
        if self.do_sample and self.temperature is None:
            raise ValueError("sampling requires a positive temperature")
        if not self.do_sample and self.temperature is not None:
            raise ValueError("greedy generation uses do_sample=false and temperature=null")
        if any(value is not None and value >= self.context_limit for value in
               (self.action_max_new_tokens, self.python_max_new_tokens, self.format_repair_max_new_tokens)):
            raise ValueError("stage output budgets must fit the context limit")
        if self.backend == "vllm":
            from urllib.parse import urlparse
            endpoint = urlparse(self.api_base)
            if endpoint.scheme != "http" or endpoint.hostname not in ("127.0.0.1", "localhost", "::1"):
                raise ValueError("vLLM must be a local HTTP service")
            if self.structured_output:
                raise ValueError("vLLM adapter uses ordinary generation only")
        return self


class PathsConfig(Record):
    """Project-relative solver inputs and output root; labels are excluded."""

    manifest: RelativePath = "data/preprocessed/task_inputs/manifest.jsonl"
    raw_data: Literal["data/raw_data"] = "data/raw_data"
    runs_dir: RelativePath = "runs"

    @model_validator(mode="after")
    def check_protocol(self) -> PathsConfig:
        """Restrict configured input/output locations; no explicit arguments."""
        if not self.manifest.startswith("data/preprocessed/task_inputs/"):
            raise ValueError("manifest must be under task_inputs/")
        if self.runs_dir != "runs" and not self.runs_dir.startswith("runs/"):
            raise ValueError("outputs must be under runs/")
        return self


class AccessPolicy(Record):
    """Fixed experiment boundaries, enforced later by dataset and tool adapters."""

    read_only_inputs: RequiredTrue = True
    freeze_A: RequiredTrue = True
    solver_labels: RequiredFalse = False
    solver_repository: RequiredFalse = False
    solver_future_questions: RequiredFalse = False
    solver_prior_B: RequiredFalse = False
    cross_challenge_results: RequiredFalse = False
    B_writeback: RequiredFalse = False


class SearchConfig(Record):
    """Frozen concept-search scope; disabled until the tool is validated."""

    enabled: bool = False
    scope: Literal["concepts_only"] = "concepts_only"
    allowed_domains: list[str] = Field(default_factory=list)
    cache_scope: Literal["run"] = "run"


ExcelToolName = Literal["describe_workbook", "describe_sheet", "read_range", "find_cells", "write_range"]


class ExcelMCPConfig(Record):
    """Pinned local Excel service; only selected tools reach the model."""

    enabled: bool = False
    python: RelativePath = ".venv/bin/python"
    command: RelativePath = ".venv/bin/excel-mcp-server"
    version: Literal["2.0.0"] = "2.0.0"
    sdk_version: Literal["2.3.0"] = "2.3.0"
    transport: Literal["stdio"] = "stdio"
    tools: list[ExcelToolName] = Field(default_factory=lambda: [
        "describe_workbook", "describe_sheet", "read_range", "find_cells",
    ])
    allow_write: bool = False
    timeout_seconds: PositiveFloat = 30.0
    max_cells: Annotated[int, Field(ge=1, le=2000)] = 200
    max_results: Annotated[int, Field(ge=1, le=1000)] = 50
    max_file_mb: Annotated[int, Field(ge=1, le=1000)] = 100

    @model_validator(mode="after")
    def check_tools(self) -> ExcelMCPConfig:
        """Reject duplicate tools and writes without explicit configuration."""
        if not self.tools or len(self.tools) != len(set(self.tools)):
            raise ValueError("Excel MCP tools must be nonempty and unique")
        if "write_range" in self.tools and not self.allow_write:
            raise ValueError("write_range requires allow_write: true")
        return self


class MCPConfig(Record):
    """External services with task-scoped adapters."""

    excel: ExcelMCPConfig = Field(default_factory=ExcelMCPConfig)


class RunConfig(Record):
    """Solver-only settings and provenance shared across A and B."""

    experiment_id: Identifier
    run_id: Identifier
    schema_version: Literal["1.0"] = "1.0"
    seed: NonNegativeInt = 42
    paths: PathsConfig = Field(default_factory=PathsConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    planning: PlanningConfig = Field(default_factory=PlanningConfig)
    recovery: RecoveryConfig = Field(default_factory=RecoveryConfig)
    budget_A: Budget = Field(default_factory=Budget)
    budget_B: Budget = Field(
        default_factory=lambda: Budget(max_model_calls=6, max_tool_calls=4, max_seconds=180.0)
    )
    baseline_attempts: One = 1
    reflection_rounds: One = 1
    recovery_run: bool = False
    task_concurrency: PositiveInt = 1
    model_concurrency: One = 1
    model_max_retries: NonNegativeInt = 2
    tool_timeout_seconds: PositiveFloat = 30.0
    tool_return_tokens: PositiveInt = 2000
    verify_computed_answers: bool = True
    answer_validation: Literal["off", "audit", "strict"] = "audit"
    direct_python_repair: bool = True
    llm_format_repair: bool = True
    match_answer_options: bool = True
    final_consistency_review: bool = False
    model_warmup: bool = True
    access: AccessPolicy = Field(default_factory=AccessPolicy)
    search: SearchConfig = Field(default_factory=SearchConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    langfuse_enabled: bool = False
    git_commit: Optional[str] = None
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    mcp_versions: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_planning_window(self) -> RunConfig:
        """Reserve Planner input space and reject conflicting MCP version provenance."""
        if self.planning.max_new_tokens >= self.model.context_limit:
            raise ValueError("planning output must fit the model context limit")
        if max(self.recovery.action_tokens, self.recovery.python_tokens,
               self.recovery.format_repair_tokens) >= self.model.context_limit:
            raise ValueError("recovery output must fit the model context limit")
        if self.mcp.excel.enabled:
            expected = {"excel-mcp-server": self.mcp.excel.version, "mcp": self.mcp.excel.sdk_version}
            if any(self.mcp_versions.get(name, value) != value for name, value in expected.items()):
                raise ValueError("mcp_versions conflicts with the configured Excel service")
        return self


class TaskDispatch(Record):
    """Host-selected task context; artifact references are not automatic tool permissions."""

    task_input: TaskInput
    variant: Variant
    plan: Plan
    workspace_policy: WorkspacePolicy
    previous_questions: dict[TaskID, str] = Field(default_factory=dict)
    previous_answers: dict[TaskID, Answer] = Field(default_factory=dict)
    previous_statuses: dict[TaskID, TerminalStatus] = Field(default_factory=dict)
    prior_A_exports: list[DependencyExport] = Field(default_factory=list)
    selected_artifacts: list[ArtifactRef] = Field(default_factory=list)
    answer_A: Optional[Answer] = None
    source_A_snapshot: Optional[SnapshotRef] = None
    diagnostics: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_dispatch(self) -> TaskDispatch:
        """Keep context within the task's direct dependency types and frozen A history."""
        task_id = self.task_input.task.task_id
        if self.plan.task_id != task_id:
            raise ValueError("dispatch plan belongs to another task")
        policy = self.plan.workspace_policy
        if (self.workspace_policy.mode, self.workspace_policy.source_task_id) != (policy.mode, policy.source_task_id):
            raise ValueError("dispatch must preserve the plan's workspace selection")
        text_sources = {d.source_task_id for d in self.plan.dependencies if d.kind == "prior_text"}
        result_sources = {d.source_task_id for d in self.plan.dependencies if d.kind != "prior_text"}
        if not set(self.previous_questions) <= text_sources:
            raise ValueError("previous question text requires a prior_text dependency")
        if not (set(self.previous_answers) | set(self.previous_statuses)) <= result_sources:
            raise ValueError("previous results require a result or workspace dependency")
        if any(answer.task_id != key for key, answer in self.previous_answers.items()):
            raise ValueError("previous answer identity does not match its key")
        if any(export.task_id not in result_sources for export in self.prior_A_exports):
            raise ValueError("dependency export is outside the authorized results")
        allowed = result_sources | ({task_id} if self.variant == "B" else set())
        if any(ref.variant != "A" or ref.task_id not in allowed for ref in self.selected_artifacts):
            raise ValueError("selected artifacts must come from authorized A tasks")
        if self.variant == "A" and (self.answer_A is not None or self.source_A_snapshot is not None):
            raise ValueError("A dispatch cannot contain its own frozen A outcome")
        if self.variant == "B" and (
            self.answer_A is None or self.answer_A.task_id != task_id
            or self.source_A_snapshot is None
            or self.source_A_snapshot.task_id != task_id
            or self.source_A_snapshot.variant != "A"
        ):
            raise ValueError("B dispatch requires the current task's frozen A outcome")
        return self


class ToolAction(Record):
    """One model-selected tool call; task and phase are bound by the host."""

    action: Literal["tool"]
    tool_name: str = Field(min_length=1)
    arguments: dict[str, JsonValue]
    brief_reason: str = ""


class PythonSelectionArguments(Record):
    """Select authorized Python inputs and describe the operation before code generation."""

    file_ids: list[Identifier]
    artifact_ids: list[Identifier] = Field(default_factory=list)
    instruction: str = Field(min_length=1)


class ExportValue(Record):
    """A reusable value whose task identity and event provenance are assigned by the host."""

    name: str = Field(min_length=1)
    value: JsonValue
    unit: Optional[str] = None
    applicability: str = Field(min_length=1)
    evidence_refs: list[str] = Field(default_factory=list)


class FinalAction(Record):
    """A final public answer and optional intermediate values for downstream tasks."""

    action: Literal["final"]
    answer: Answer
    exports: list[ExportValue] = Field(default_factory=list)


AgentAction = Annotated[Union[ToolAction, FinalAction], Field(discriminator="action")]


class ModelFinalAction(Record):
    """Compact model output; the host supplies task identity and derives answer status."""

    action: Literal["final"]
    value: JsonValue
    unit: Optional[str] = None
    computed_value: Optional[float] = None
    computed_unit: Optional[str] = None
    brief_explanation: str = ""
    evidence_refs: list[str] = Field(default_factory=list)
    reported_values: list[ReportedValue] = Field(default_factory=list)
    exports: list[ExportValue] = Field(default_factory=list)


ModelAgentAction = Annotated[Union[ToolAction, ModelFinalAction], Field(discriminator="action")]
