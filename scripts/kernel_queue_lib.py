#!/usr/bin/env python3

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath

ROOT_DIR = Path(__file__).resolve().parents[1]
TASKS_DIR = ROOT_DIR / ".agents" / "kernel" / "tasks"

PRIORITY_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
STATUS_ORDER = {
    "claimed": 0,
    "in_progress": 1,
    "queued": 2,
    "blocked": 3,
    "completed": 4,
    "rejected": 5,
    "duplicate": 6,
}
VALID_PRIORITIES = set(PRIORITY_ORDER)
VALID_STATUSES = set(STATUS_ORDER)
VALID_KINDS = {"kernel", "benchmark", "research", "profiler", "autotune", "infra"}
VALID_RESOURCE_CLASSES = {"cpu_only", "gpu_light", "gpu_exclusive"}
ACTIVE_STATUSES = {"claimed", "in_progress"}
TASK_FILE_SUFFIX = ".json"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def to_utc_string(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def task_sort_key(task: dict) -> tuple[int, int, str]:
    return (
        STATUS_ORDER.get(task.get("status", "queued"), 999),
        PRIORITY_ORDER.get(task.get("priority", "P9"), 999),
        task.get("id", ""),
    )


def load_task(path: Path) -> dict:
    task = json.loads(path.read_text())
    task["_path"] = path
    return task


def iter_task_paths() -> list[Path]:
    if not TASKS_DIR.exists():
        return []
    return sorted(
        path
        for path in TASKS_DIR.glob(f"*{TASK_FILE_SUFFIX}")
        if path.is_file() and not path.name.startswith("_")
    )


def load_tasks() -> list[dict]:
    return [load_task(path) for path in iter_task_paths()]


def save_task(task: dict) -> None:
    path = Path(task["_path"])
    payload = {key: value for key, value in task.items() if key != "_path"}
    path.write_text(json.dumps(payload, indent=2) + "\n")


def get_task_by_id(tasks: list[dict], task_id: str) -> dict | None:
    for task in tasks:
        if task.get("id") == task_id:
            return task
    return None


def is_active(task: dict) -> bool:
    return task.get("status") in ACTIVE_STATUSES


def lease_expired(task: dict, now: datetime | None = None) -> bool:
    if not is_active(task):
        return False
    lease_until = parse_utc(task.get("lease_expires_utc"))
    if lease_until is None:
        return False
    if now is None:
        now = utc_now()
    return lease_until < now


def normalize_scope_entry(entry: str) -> PurePosixPath:
    return PurePosixPath(entry.strip().strip("/"))


def scope_entries(task: dict) -> list[PurePosixPath]:
    return [normalize_scope_entry(entry) for entry in task.get("scope_files", [])]


def scope_overlaps(left: str | PurePosixPath, right: str | PurePosixPath) -> bool:
    left_path = normalize_scope_entry(str(left))
    right_path = normalize_scope_entry(str(right))
    left_parts = left_path.parts
    right_parts = right_path.parts
    return (
        left_parts == right_parts
        or left_parts[: len(right_parts)] == right_parts
        or right_parts[: len(left_parts)] == left_parts
    )


def overlapping_active_tasks(target: dict, tasks: list[dict]) -> list[dict]:
    conflicts = []
    for other in tasks:
        if other.get("id") == target.get("id"):
            continue
        if not is_active(other):
            continue
        if lease_expired(other):
            continue
        for target_entry in scope_entries(target):
            for other_entry in scope_entries(other):
                if scope_overlaps(target_entry, other_entry):
                    conflicts.append(other)
                    target_entry = None
                    break
            if target_entry is None:
                break
    return conflicts


def validate_task(task: dict, known_ids: set[str]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    task_id = task.get("id", "<missing-id>")
    required_fields = [
        "schema_version",
        "id",
        "title",
        "priority",
        "status",
        "owner",
        "kind",
        "resource_class",
        "scope_files",
        "benchmark",
        "tests",
        "updated_utc",
    ]
    for field in required_fields:
        if field not in task:
            errors.append(f"{task_id}: missing required field '{field}'")

    if task.get("schema_version") != 1:
        errors.append(f"{task_id}: unsupported schema_version {task.get('schema_version')!r}")
    if task.get("priority") not in VALID_PRIORITIES:
        errors.append(f"{task_id}: invalid priority {task.get('priority')!r}")
    if task.get("status") not in VALID_STATUSES:
        errors.append(f"{task_id}: invalid status {task.get('status')!r}")
    if task.get("kind") not in VALID_KINDS:
        errors.append(f"{task_id}: invalid kind {task.get('kind')!r}")
    if task.get("resource_class") not in VALID_RESOURCE_CLASSES:
        errors.append(f"{task_id}: invalid resource_class {task.get('resource_class')!r}")

    owner = task.get("owner")
    if is_active(task) and (not owner or owner == "unclaimed"):
        errors.append(f"{task_id}: active task must have a real owner")
    if task.get("status") == "queued" and owner and owner != "unclaimed":
        warnings.append(f"{task_id}: queued task still has owner={owner!r}")

    scope = task.get("scope_files")
    if not isinstance(scope, list) or not scope:
        errors.append(f"{task_id}: scope_files must be a non-empty list")

    benchmark = task.get("benchmark", {})
    tests = task.get("tests", {})
    if not isinstance(benchmark, dict) or "command" not in benchmark:
        errors.append(f"{task_id}: benchmark.command is required")
    if not isinstance(tests, dict) or "command" not in tests:
        errors.append(f"{task_id}: tests.command is required")

    try:
        parse_utc(task.get("updated_utc"))
    except ValueError as exc:
        errors.append(f"{task_id}: updated_utc is not valid UTC ISO-8601: {exc}")

    try:
        parse_utc(task.get("lease_expires_utc"))
    except ValueError as exc:
        errors.append(f"{task_id}: lease_expires_utc is not valid UTC ISO-8601: {exc}")

    duplicate_of = task.get("duplicate_of")
    if duplicate_of is not None and duplicate_of not in known_ids:
        errors.append(f"{task_id}: duplicate_of references unknown task {duplicate_of!r}")

    for relation_field in ("depends_on", "blocked_by"):
        relation = task.get(relation_field, [])
        if not isinstance(relation, list):
            errors.append(f"{task_id}: {relation_field} must be a list")
            continue
        for related_id in relation:
            if related_id not in known_ids:
                errors.append(f"{task_id}: {relation_field} references unknown task {related_id!r}")

    if is_active(task) and task.get("lease_expires_utc") is None:
        warnings.append(f"{task_id}: active task has no lease_expires_utc")

    return errors, warnings


def validate_tasks(tasks: list[dict]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    ids = [task.get("id") for task in tasks]
    known_ids = {task_id for task_id in ids if task_id}
    if len(ids) != len(known_ids):
        errors.append("duplicate task ids detected")

    for task in tasks:
        task_errors, task_warnings = validate_task(task, known_ids)
        errors.extend(task_errors)
        warnings.extend(task_warnings)

        path = task.get("_path")
        if path is not None and Path(path).stem != task.get("id"):
            errors.append(f"{task.get('id')}: task filename must match id ({Path(path).name})")

    for index, left in enumerate(tasks):
        if not is_active(left) or lease_expired(left):
            continue
        for right in tasks[index + 1 :]:
            if not is_active(right) or lease_expired(right):
                continue
            if right.get("duplicate_of") == left.get("id") or left.get("duplicate_of") == right.get("id"):
                continue
            if any(scope_overlaps(a, b) for a in scope_entries(left) for b in scope_entries(right)):
                errors.append(
                    f"scope overlap between active tasks {left.get('id')} and {right.get('id')}"
                )

    active_gpu_exclusive = [
        task for task in tasks if is_active(task) and not lease_expired(task) and task.get("resource_class") == "gpu_exclusive"
    ]
    if len(active_gpu_exclusive) > 1:
        warnings.append(
            "multiple gpu_exclusive tasks are active; benchmark runs should be serialized on the shared GPU"
        )

    return errors, warnings


def claimable(task: dict) -> bool:
    return task.get("status") in {"queued", "blocked", "claimed", "in_progress"}


def lease_until(hours: float) -> str:
    return to_utc_string(utc_now() + timedelta(hours=hours))
