#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from collections import Counter

from kernel_queue_lib import TASKS_DIR, load_tasks, validate_tasks


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate kernel task cards and ownership rules.")
    parser.add_argument("--quiet", action="store_true", help="Only print validation failures")
    args = parser.parse_args()

    tasks = load_tasks()
    errors, warnings = validate_tasks(tasks)

    if not args.quiet:
        print(f"task_dir={TASKS_DIR}")
        print(f"tasks={len(tasks)}")
        if tasks:
            statuses = Counter(task["status"] for task in tasks)
            summary = ", ".join(f"{name}={count}" for name, count in sorted(statuses.items()))
            print(f"status_summary={summary}")

    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    for error in errors:
        print(f"error: {error}", file=sys.stderr)

    if errors:
        return 1

    if not args.quiet:
        print("queue validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
