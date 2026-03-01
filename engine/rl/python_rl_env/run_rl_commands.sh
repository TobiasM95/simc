#!/usr/bin/env bash
set -euo pipefail

# Run this script from engine/rl/python_rl_env
# Example:
#   cd engine/rl/python_rl_env
#   ./run_rl_commands.sh demo

# ---------------------------
# Edit these paths as needed
# ---------------------------
SIMC="../../../out/build/x64-Debug/simc.exe"
PROFILE="../../../profiles/my_rl_profile.simc"
RUN_DIR="../../../training_runs/simc_ppo_YYYYMMDD_HHMMSS"
EVAL_ITERATIONS="2000"
EPISODES="3"

# Optional: use Release binary instead
# SIMC="../../../out/build/x64-Release/simc.exe"

MODE="${1:-help}"

cmd_demo() {
  uv run python rl_bridge.py \
    --mode demo \
    --simc "$SIMC" \
    --profile "$PROFILE" \
    --episodes "$EPISODES"
}

cmd_multi() {
  uv run python rl_bridge.py \
    --mode multi \
    --simc "$SIMC" \
    --profile "$PROFILE"
}

cmd_resume() {
  uv run python rl_bridge.py \
    --mode multi \
    --simc "$SIMC" \
    --profile "$PROFILE" \
    --resume "$RUN_DIR"
}

cmd_eval() {
  uv run python rl_bridge.py \
    --mode eval \
    --simc "$SIMC" \
    --profile "$PROFILE" \
    --model-dir "$RUN_DIR" \
    --iterations "$EVAL_ITERATIONS" \
    --output-dir "$RUN_DIR/evaluation"
}

cmd_single() {
  uv run python rl_bridge.py \
    --mode single \
    --simc "$SIMC" \
    --profile "$PROFILE"
}

case "$MODE" in
  demo) cmd_demo ;;
  multi) cmd_multi ;;
  resume) cmd_resume ;;
  eval) cmd_eval ;;
  single) cmd_single ;;
  all)
    cmd_demo
    cmd_multi
    cmd_resume
    cmd_eval
    cmd_single
    ;;
  *)
    cat <<'USAGE'
Usage: ./run_rl_commands.sh <mode>

Modes:
  demo    - run_demo
  multi   - run_multi_thread_sb3
  resume  - run_multi_thread_sb3 with --resume
  eval    - run_evaluation
  single  - run_single_thread_sb3
  all     - run all commands sequentially
USAGE
    ;;
esac

