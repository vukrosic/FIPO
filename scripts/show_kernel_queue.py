#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import Counter

from kernel_queue_lib import lease_expired, load_tasks, task_sort_key, validate_tasks


def main():
    parser = argparse.ArgumentParser(description="Show the kernel task-card queue.")
    parser.add_argument("--active-only", action="store_true", help="Show only claimed and in-progress tasks")
    args = parser.parse_args()

    tasks = load_tasks()
    errors, warnings = validate_tasks(tasks)
    counts = Counter(task["status"] for task in tasks)
    summary = ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
    print(f"tasks={len(tasks)} status_summary={summary}")

    if warnings:
        print(f"warnings={len(warnings)}")
    if errors:
        print(f"errors={len(errors)}")

    shown = 0
    for task in sorted(tasks, key=task_sort_key):
        if args.active_only and task["status"] not in {"claimed", "in_progress"}:
            continue
        shown += 1
        lease_state = "expired" if lease_expired(task) else "active"
        if task.get("lease_expires_utc") is None:
            lease_state = "none"
        print(
            f'{task["id"]} [{task["priority"]}] {task["status"]} owner={task["owner"]} '
            f'lease={lease_state}'
        )
        print(f'  title: {task["title"]}')
        print(f'  kind: {task["kind"]} resource_class={task["resource_class"]}')
        print(f'  scope: {", ".join(task["scope_files"])}')
        print(f'  benchmark: {task["benchmark"]["command"]}')
        print(f'  tests: {task["tests"]["command"]}')
        if task.get("results"):
            print(f'  results: {task["results"]}')
        if task.get("duplicate_of"):
            print(f'  duplicate_of: {task["duplicate_of"]}')
        if task.get("depends_on"):
            print(f'  depends_on: {", ".join(task["depends_on"])}')
        if task.get("blocked_by"):
            print(f'  blocked_by: {", ".join(task["blocked_by"])}')
    if args.active_only and shown == 0:
        print("no active tasks")
    if warnings:
        for warning in warnings:
            print(f"warning: {warning}")
    if errors:
        for error in errors:
            print(f"error: {error}")


if __name__ == "__main__":
    main()
