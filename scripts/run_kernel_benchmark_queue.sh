#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
LOG_DIR="${ROOT_DIR}/artifacts/kernel_benchmarks"
mkdir -p "${LOG_DIR}"

run_cmd() {
    local name="$1"
    shift
    local log_file="${LOG_DIR}/${name}.log"
    echo "[kernel-queue] running ${name}"
    (
        cd "${ROOT_DIR}"
        "$@"
    ) | tee "${log_file}"
}

run_cmd future_kl_fp32 python scripts/benchmark_future_kl.py --batch-size 32 --response-len 2048 --dtype float32 --warmup 10 --iters 30
run_cmd future_kl_bf16 python scripts/benchmark_future_kl.py --batch-size 32 --response-len 2048 --dtype bfloat16 --warmup 10 --iters 30
run_cmd kernel_tests python -m unittest discover -s tests -v
