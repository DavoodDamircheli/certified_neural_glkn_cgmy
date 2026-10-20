#!/usr/bin/env bash
# Launch the approved M6/M7 runs in the background (single-threaded, pinned).
# Timing-sensitive runs on P-cores 2,3,4 (same 6.9 GHz max; CPU 0 avoided for
# interrupts, CPUs 1/7 boost higher); non-timing studies on E-cores 8,9,10.
# Each run creates its own results/free_integrated_v5/<run_id>/ and is resumable
# with --resume <run_dir>.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=src:experiments:scripts:benchmarks:.
mkdir -p logs
nohup taskset -c 2 python3 benchmarks/run_gate_d_benchmark_v5.py --full --approved-full-campaign --modes cold  > logs/v5_m7_bench_cold.log  2>&1 &
nohup taskset -c 3 python3 benchmarks/run_gate_d_benchmark_v5.py --full --approved-full-campaign --modes reuse > logs/v5_m7_bench_reuse.log 2>&1 &
nohup taskset -c 4 python3 benchmarks/run_scaling_v5.py                                                       > logs/v5_m7_scaling.log     2>&1 &
nohup taskset -c 8 python3 scripts/v5_m6_audit.py                                                             > logs/v5_m6_audit.log       2>&1 &
nohup taskset -c 9 python3 scripts/v5_m6_riesz.py                                                             > logs/v5_m6_riesz.log       2>&1 &
nohup taskset -c 10 python3 scripts/v5_corner_refinement.py                                                   > logs/v5_corner.log         2>&1 &
sleep 1; jobs -l
