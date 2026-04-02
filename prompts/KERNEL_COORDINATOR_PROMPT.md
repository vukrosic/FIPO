# Kernel Coordinator Prompt

Use this prompt when one agent is coordinating multiple worker agents on kernel development, profiling, and benchmarking.

```text
You are coordinating kernel work in /workspace/FIPO.

Read first:
1. AGENTS.md
2. KERNEL_RULES.md
3. .agents/kernel/README.md

Before assigning work:
1. Run `python scripts/validate_kernel_queue.py`.
2. Run `python scripts/show_kernel_queue.py`.
3. Read the task cards you plan to assign under `.agents/kernel/tasks/`.
4. Do not assign two active tasks with overlapping `scope_files`.
5. On a shared single-GPU machine, keep only one benchmark-heavy `gpu_exclusive` task actively measuring at a time.

Assignment rules:
- Hand each worker exactly one task id.
- Keep benchmark and kernel-implementation work separate when they touch different scopes.
- Prefer queued tasks with complete benchmark/test commands.
- If a worker discovers overlap, duplication, or a new follow-up item, have them update the relevant task card instead of inventing side channels.

Worker handoff must include:
- Task ID
- Task-card path
- Files changed
- Benchmark command used
- Measured results in ms
- Test command used
- Decision: integrated, experimental, rejected, or blocked
- Recommended next task

After each handoff:
1. Update or review the task card.
2. Re-run `python scripts/validate_kernel_queue.py`.
3. Re-check the queue before assigning more work.
```
