# Agents

If you are an agent working in this repo, do not start by guessing. Read files in this order:

1. [KERNEL_RULES.md](/workspace/FIPO/KERNEL_RULES.md)
2. [Kernel Agent README](/workspace/FIPO/.agents/kernel/README.md)
3. [Kernel Worker Prompt](/workspace/FIPO/prompts/KERNEL_WORKER_PROMPT.md)
4. Optional human status board: [KERNEL_QUEUE.md](/workspace/FIPO/KERNEL_QUEUE.md)

## Scope

This repo currently has an active kernel-optimization track focused on PPO/FIPO hotspots.

## Working Rules

1. The canonical queue is the task-card set under [.agents/kernel/tasks](/workspace/FIPO/.agents/kernel/tasks).
2. Claim one task before writing code by running `python scripts/claim_kernel_task.py <TASK_ID> --owner <NAME>`.
3. Validate queue state before and after ownership changes with `python scripts/validate_kernel_queue.py`.
4. Do not duplicate another agent's active scope. The claim/validate scripts enforce overlapping-scope checks.
5. Keep a torch reference path for every kernelized section.
6. Benchmark with synchronized CUDA timings in milliseconds.
7. Add or update tests before closing a task.
8. If a kernel loses on the representative benchmark, do not integrate it into the trainer path.

## Canonical Files

- Task cards: [.agents/kernel/tasks](/workspace/FIPO/.agents/kernel/tasks)
- Queue validator: [validate_kernel_queue.py](/workspace/FIPO/scripts/validate_kernel_queue.py)
- Task claimer: [claim_kernel_task.py](/workspace/FIPO/scripts/claim_kernel_task.py)
- Queue viewer: [show_kernel_queue.py](/workspace/FIPO/scripts/show_kernel_queue.py)
- Human-readable queue/history: [KERNEL_QUEUE.md](/workspace/FIPO/KERNEL_QUEUE.md)
- Worker prompt: [KERNEL_WORKER_PROMPT.md](/workspace/FIPO/prompts/KERNEL_WORKER_PROMPT.md)
- Coordinator prompt: [KERNEL_COORDINATOR_PROMPT.md](/workspace/FIPO/prompts/KERNEL_COORDINATOR_PROMPT.md)

## Roles

- `kernel-implementer`: writes kernel code and integration switches
- `benchmarker`: measures before/after and updates results
- `validator`: writes tests and checks parity
- `profiler`: identifies next real hotspots before new kernel work starts

## Required Handoff

Every agent handoff should include:

1. Task ID
2. Task-card path
3. Files changed
4. Benchmark command used
5. Measured results in ms
6. Test command used
7. Recommended next task
