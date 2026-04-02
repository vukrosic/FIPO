# Kernel Worker Prompt

Use this prompt when handing a kernel task to another agent.

```text
You are working in /workspace/FIPO on GPU kernel optimization.

Before editing anything:
1. Read KERNEL_RULES.md.
2. Read .agents/kernel/README.md.
3. Run `python scripts/validate_kernel_queue.py`.
4. Claim exactly one task with `python scripts/claim_kernel_task.py <TASK_ID> --owner <YOUR_NAME>`.
5. Read the claimed task card under `.agents/kernel/tasks/<TASK_ID>.json`.
6. Do not touch files outside `scope_files`.

Your task:
- Work only on: <TASK_ID>
- Ownership scope: <FILES_OR_MODULES>
- Goal: <SHORT_GOAL>

Required workflow:
1. Keep a correct torch reference path.
2. Add or update a benchmark command under scripts/.
3. Measure before/after with synchronized CUDA timings in milliseconds.
4. Add or update unittest coverage.
5. If the kernel loses on the benchmark, do not integrate it into the trainer path.
6. Update the task card with status, measured results, and next action.
7. Re-run `python scripts/validate_kernel_queue.py` before handoff.

Constraints:
- Do not revert someone else's changes.
- Use apply_patch for file edits.
- Prefer float32 accumulation unless you have measured evidence for another choice.
- Keep an `auto | torch | triton` switch for any new kernel helper.
- Avoid architecture-specific behavior unless benchmark data proves it is necessary.

Final response format:
- Files changed
- Task-card path
- Measured results in ms
- Whether the kernel was integrated, kept experimental, or rejected
- Any follow-up task to queue
```
