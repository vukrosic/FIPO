#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys

from kernel_queue_lib import (
    claimable,
    get_task_by_id,
    lease_expired,
    lease_until,
    load_tasks,
    overlapping_active_tasks,
    save_task,
    to_utc_string,
    utc_now,
    validate_tasks,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Claim a kernel workflow task card.")
    parser.add_argument("task_id", help="Task id, for example FIPO-004")
    parser.add_argument("--owner", required=True, help="Agent or user name claiming the task")
    parser.add_argument("--hours", type=float, default=4.0, help="Lease duration in hours")
    parser.add_argument(
        "--status",
        choices=["claimed", "in_progress"],
        default="claimed",
        help="Status to set after claiming",
    )
    parser.add_argument(
        "--ignore-deps",
        action="store_true",
        help="Allow claiming even if depends_on tasks are not completed",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the claim result without writing the task card")
    args = parser.parse_args()

    tasks = load_tasks()
    errors, warnings = validate_tasks(tasks)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)

    task = get_task_by_id(tasks, args.task_id)
    if task is None:
        print(f"error: unknown task id {args.task_id}", file=sys.stderr)
        return 1
    if not claimable(task):
        print(f"error: task {args.task_id} is not claimable from status={task.get('status')}", file=sys.stderr)
        return 1
    if task.get("duplicate_of"):
        print(f"error: task {args.task_id} is marked duplicate_of={task['duplicate_of']}", file=sys.stderr)
        return 1

    if not args.ignore_deps:
        incomplete = []
        for dependency_id in task.get("depends_on", []):
            dependency = get_task_by_id(tasks, dependency_id)
            if dependency is None or dependency.get("status") != "completed":
                incomplete.append(dependency_id)
        if incomplete:
            print(
                f"error: task {args.task_id} depends on incomplete tasks: {', '.join(incomplete)}",
                file=sys.stderr,
            )
            return 1

    current_owner = task.get("owner")
    if task.get("status") in {"claimed", "in_progress"} and current_owner not in {None, "unclaimed", args.owner}:
        if not lease_expired(task):
            print(
                f"error: task {args.task_id} is already owned by {current_owner} and the lease is still active",
                file=sys.stderr,
            )
            return 1

    conflicts = [
        conflict
        for conflict in overlapping_active_tasks(task, tasks)
        if conflict.get("owner") != args.owner
    ]
    if conflicts:
        for conflict in conflicts:
            print(
                f"error: scope conflict with active task {conflict['id']} owned by {conflict['owner']}",
                file=sys.stderr,
            )
        return 1

    now = to_utc_string(utc_now())
    task["owner"] = args.owner
    task["status"] = args.status
    task["claimed_utc"] = task.get("claimed_utc") or now
    task["lease_expires_utc"] = lease_until(args.hours)
    task["updated_utc"] = now

    print(
        f"{task['id']} status={task['status']} owner={task['owner']} lease_expires_utc={task['lease_expires_utc']}"
    )
    if args.dry_run:
        return 0

    save_task(task)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
