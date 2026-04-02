# Kernel Agent README

This directory is the structured entrypoint for agents working on GPU kernels.

## Read This First

1. [AGENTS.md](/workspace/FIPO/AGENTS.md)
2. [KERNEL_RULES.md](/workspace/FIPO/KERNEL_RULES.md)
3. Run `python scripts/validate_kernel_queue.py`
4. Inspect the queue with `python scripts/show_kernel_queue.py`
5. Read the specific task card under [.agents/kernel/tasks](/workspace/FIPO/.agents/kernel/tasks)
6. [Kernel Worker Prompt](/workspace/FIPO/prompts/KERNEL_WORKER_PROMPT.md)

## Files

- `tasks/`: one JSON task card per work item. This is the source of truth.
- `queue.json`: legacy queue snapshot. Do not treat it as authoritative.
- `../../KERNEL_QUEUE.md`: optional human-readable history and status board.
- `../../prompts/KERNEL_WORKER_PROMPT.md`: reusable prompt for worker agents.
- `../../prompts/KERNEL_COORDINATOR_PROMPT.md`: prompt for the coordinator who assigns parallel work.

## Task Card Schema

Each task card carries enough metadata for multiple agents to work without colliding:

- Identity: `id`, `title`, `summary`, `priority`, `kind`
- Ownership: `status`, `owner`, `claimed_utc`, `lease_expires_utc`, `updated_utc`
- Scope lock: `scope_files`
- Scheduling: `resource_class`
- Coordination: `depends_on`, `blocked_by`, `duplicate_of`
- Execution: `benchmark.command`, `tests.command`, `success_gate`
- Outcome: `results`, `notes`

## Agent Workflow

1. Validate the queue before starting: `python scripts/validate_kernel_queue.py`.
2. Inspect tasks: `python scripts/show_kernel_queue.py`.
3. Claim exactly one task with `python scripts/claim_kernel_task.py <TASK_ID> --owner <NAME>`.
4. Work only inside the claimed `scope_files`.
5. Run the task benchmark and test commands from the task card.
6. Update the task card first. If someone still uses `KERNEL_QUEUE.md`, treat it as a secondary summary only.
7. Re-run `python scripts/validate_kernel_queue.py` before handing off.
8. If the kernel regresses, mark it `rejected` or leave it experimental. Do not wire regressions into the hot path.

## Ownership Rule

If two active tasks overlap in `scope_files`, the queue is invalid. The validator checks this by path overlap, not just exact string match.
