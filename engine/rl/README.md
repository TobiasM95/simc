# SimulationCraft RL Bridge (engine/rl)

This directory contains the C++ side of the Reinforcement Learning bridge and a Python Gymnasium environment that communicates with simc over `stdin/stdout`.

## What It Does

The RL bridge replaces APL foreground action selection (root call only) with policy-driven action selection.

At each foreground decision point:
1. simc builds an observation and legal-action mask.
2. simc writes a JSON `step` message to stdout (when `rl_stdio=1`).
3. External code returns one action index on stdin.
4. simc executes that action and advances time to the next decision point.

At episode end, simc writes a JSON `done` message and exits.

## Current Behavior (Important)

- RL only controls **foreground** actions.
- Off-GCD / cast-while-casting decisions are not delegated to RL.
- Dense reward is `priority_iteration_dmg` delta between consecutive decision points.
- RL is ignored for enemies and pets.
- Pseudo-actions are appended to action space:
  - `wait_0.1`, `wait_0.2`, `wait_0.5`, `wait_1.0`, `wait_1.5`
  - `pass`

## Files

- `rl_interface.hpp/.cpp`: C++ RL bridge implementation.
- `python_rl_env/rl_train_v2.py`: maintained v2 trainer/eval entry point.
- `python_rl_env/rl_bridge.py`: legacy helper script, not kept in sync with the current v2 path.

## Simc Options

These are simulator options (set in profile or CLI):

- `rl_enable=1`
  - Enables RL action selection hook.
  - Default: `0`.
- `rl_teacher_enable=1`
  - Exposes active APL teacher action (`teacher_idx`, `teacher_label`) in each step payload.
  - Used by v2 teacher-reward curriculum mode.
  - Default: `0`.
- `rl_stdio=1`
  - Uses stdio JSON protocol for external policy control.
  - Forces `threads=1` internally for safe IO.
  - Default: `0`.
- `rl_trace=1`
  - Writes per-decision JSONL trace files.
  - Default: `0`.
- `rl_trace_file=path/to/file.jsonl`
  - Base path for trace output.
  - Thread index suffix is added when needed.
- `rl_observe_buffs=<list>`
  - Configures buff observation labels.
  - Repeatable option, comma-separated supported.
- `rl_observe_dots=<list>`
  - Configures dot observation labels.
  - Repeatable option, comma-separated supported.

### Buff/Dot Observation Config

`rl_observe_buffs` and `rl_observe_dots` are tokenized internally with simc token rules.

Examples:
- `rl_observe_buffs=tigers_fury,clearcasting`
- `rl_observe_buffs=incarnation_avatar_of_ashamane`
- `rl_observe_dots=rip,rake`

If not set, arrays are empty and no buff/dot features are sent.

## JSON Protocol

### `step` message (simc -> python)

Sent once per RL decision point.

Core fields:
- `type`: `"step"`
- `t`: current sim time (seconds)
- `fight_len`: expected fight length (seconds)
- `ttd`: target time-to-die (seconds)
- `gcd_rem`: gcd remaining (seconds)
- `reward`: dense reward for the previous decision
- `teacher_idx`: teacher action index in the raw action space (`-1` when unavailable)
- `teacher_label`: teacher action label (`""` when unavailable)
- `spec_id`: current specialization id
- `resource_pct`: resource percentages (`RESOURCE_MAX` length)
- `n`: number of actions
- `labels`: action label list (length `n`)
- `mask`: legal-action mask (0/1, length `n`)
- `cd_rem_s`: cooldown remaining per action
- `cd_rem_n`: cooldown remaining normalized per action
- `cd_charges_f`: fractional charges per action
- `buff_labels`, `buff_remains`, `buff_stacks`
- `dot_labels`, `dot_remains`, `dot_stacks`

### action input (python -> simc)

- Write one integer index followed by newline.
- Index must refer to a legal action (`mask[i] == 1`) or simc falls back to first legal action.

### `done` message (simc -> python)

At end of the episode:
- `type`: `"done"`
- `total_damage`: final damage for the main actor
- `fight_length`: final sim time in seconds

## Action Space Construction

The action space is built once per episode on first RL decision:

1. Start from `player.action_list` (APL-parsed + class-created actions).
2. Discover and create additional baseline class/spec/talent actions from DBC/talent data.
3. Filter out internal/non-castable/control-flow/background/off-gcd/cwc actions.
4. Deduplicate by action name.
5. Append wait/pass pseudo-actions.

Action legality (`mask`) is recomputed each decision using `action_t::ready()` + target checks + context rules.

## How To Use

## 1) Build simc with RL files

RL files are already wired into `source_files/*` and `cmake_engine.txt`.
Build simc normally for your platform.

All Python commands below assume you are in:

```bash
cd engine/rl/python_rl_env
```

and use `uv run python ...`.

## Quickstart Matrix

Use this as the fastest entry point.

| Goal | Command |
|---|---|
| Smoke-test bridge IO | `uv run python rl_bridge.py --mode demo --simc ../../../out/build/x64-Release/simc.exe --profile ./simc_profiles/druid_feral.simc --episodes 1` |
| Smoke-test with full action space | `uv run python rl_bridge.py --mode demo --simc ../../../out/build/x64-Release/simc.exe --profile ./simc_profiles/druid_feral.simc --episodes 1 --no-blacklist` |
| Start training | `uv run python rl_bridge.py --mode multi --simc ../../../out/build/x64-Release/simc.exe --profile ./simc_profiles/druid_feral.simc` |
| Resume training | `uv run python rl_bridge.py --mode multi --simc ../../../out/build/x64-Release/simc.exe --profile ./simc_profiles/druid_feral.simc --resume ./training_runs/simc_ppo_YYYYMMDD_HHMMSS` |
| Evaluate trained model | `uv run python rl_bridge.py --mode eval --simc ../../../out/build/x64-Release/simc.exe --profile ./simc_profiles/druid_feral.simc --model-dir ./training_runs/simc_ppo_YYYYMMDD_HHMMSS --iterations 2000` |
| Single-env SB3 test | `uv run python rl_bridge.py --mode single --simc ../../../out/build/x64-Release/simc.exe --profile ./simc_profiles/druid_feral.simc` |

### V2 Training/Eval Script (Python-side improvements)

`python_rl_env/rl_train_v2.py` adds:
- per-episode seed diversity across workers
- cooldown normalized features (`cd_rem_n`) in observations
- no-op masking policy (`wait_*`/`pass` masked when real actions are legal)
- held-out eval based checkpoint selection (`best_eval_model.zip`)
- optional APL teacher-to-DPS reward curriculum with linear or hard transition
- configurable DPS reward:
  - `delta` = dense per-step damage deltas (default and recommended for regular non-teacher training)
  - `sparse` = terminal-only `total_damage`

`rl_train_v2.py` is the supported workflow. `rl_bridge.py` is legacy.

Start regular non-teacher training (default dense reward):

```bash
uv run python rl_train_v2.py \
  --mode train \
  --simc ../../../out/build/x64-Release/simc.exe \
  --profile ./simc_profiles/druid_feral.simc
```

This default run is equivalent to:
- `--teacher-mode off`
- `--dps-reward-mode delta`

Supported profile variants for v2:
- `./simc_profiles/druid_feral.simc`: profile with an explicit custom APL.
- `./simc_profiles/druid_feral_no_apl.simc`: profile without explicit `actions=` lines.
- Non-teacher v2 should work with either profile style.
- No-APL profiles depend on SimC having a generated/spec-implemented default APL for that spec.
- In teacher mode, the teacher source is the active APL:
  - profile APL when the profile defines one
  - otherwise SimC's generated/spec-implemented default APL

Run evaluation:

```bash
uv run python rl_train_v2.py \
  --mode eval \
  --simc ../../../out/build/x64-Release/simc.exe \
  --profile ./simc_profiles/druid_feral.simc \
  --model-dir ./training_runs/simc_ppo_v2_YYYYMMDD_HHMMSS \
  --iterations 2000
```

Resume training:

```bash
uv run python rl_train_v2.py \
  --mode train \
  --resume ./training_runs/simc_ppo_v2_YYYYMMDD_HHMMSS \
  --resume-checkpoint auto \
  --simc ../../../out/build/x64-Release/simc.exe \
  --profile ./simc_profiles/druid_feral.simc \
  --total-timesteps 3000000
```

Resume notes:
- `--total-timesteps` is additional training steps for the resumed `learn()` call.
- Internal timestep counters are preserved on resume (teacher schedule/adaptive windows continue from prior progress).
- `--resume-checkpoint` controls which checkpoint is loaded:
  - `auto` (default): `latest -> interrupted -> final -> best_eval`
  - `best_eval`: load `best_eval_model.zip` directly (recommended for recovery after drift)
  - `latest`, `interrupted`, `final`: force a specific checkpoint file
- When resuming with `--resume-checkpoint best_eval`, v2 prefers `vec_normalize_best_eval.pkl` (if present), otherwise falls back to `vec_normalize.pkl`.

Recover from degraded overnight training:

```bash
uv run python rl_train_v2.py \
  --mode train \
  --resume ./training_runs/simc_ppo_v2_YYYYMMDD_HHMMSS \
  --resume-checkpoint best_eval \
  --simc ../../../out/build/x64-Release/simc.exe \
  --profile ./simc_profiles/druid_feral.simc \
  --teacher-mode off \
  --total-timesteps 20000000 \
  --lr-initial 8e-5 \
  --lr-final 3e-5 \
  --ent-coef 0.001 \
  --clip-range 0.15 \
  --gamma 0.999 \
  --gae-lambda 0.97 \
  --eval-drop-stop-pct 8 \
  --eval-drop-stop-patience 3
```

Teacher curriculum run on top of dense DPS deltas:

```bash
uv run python rl_train_v2.py \
  --mode train \
  --simc ../../../out/build/x64-Release/simc.exe \
  --profile ./simc_profiles/druid_feral.simc \
  --num-envs 12 \
  --total-timesteps 16000000 \
  --dps-reward-mode delta \
  --teacher-mode reward \
  --teacher-schedule linear \
  --teacher-start-weight 1.0 \
  --teacher-end-weight 0.0 \
  --teacher-anneal-steps 12800000 \
  --teacher-reward-scale 100000 \
  --teacher-update-freq 1000 \
  --teacher-off-stabilize \
  --teacher-off-adaptive \
  --teacher-off-adaptive-steps 1000000 \
  --teacher-off-min-lr 5e-5 \
  --teacher-off-ent-coef-scale 2.0 \
  --teacher-off-ent-coef-max 0.03 \
  --lr-initial 2.5e-4 \
  --lr-final 2.5e-5 \
  --clip-range 0.2 \
  --gamma 1.0 \
  --gae-lambda 1.0 \
  --eval-drop-stop-pct 8 \
  --eval-drop-stop-patience 3
```

Teacher curriculum (gradual linear transition, defaults shown):

```bash
uv run python rl_train_v2.py \
  --mode train \
  --simc ../../../out/build/x64-Release/simc.exe \
  --profile ./simc_profiles/druid_feral.simc \
  --teacher-mode reward \
  --teacher-schedule linear \
  --teacher-start-weight 1.0 \
  --teacher-end-weight 0.0 \
  --teacher-anneal-steps 0 \
  --teacher-reward-scale 250000 \
  --teacher-update-freq 1000
```

Teacher curriculum (sudden switch):

```bash
uv run python rl_train_v2.py \
  --mode train \
  --simc ../../../out/build/x64-Release/simc.exe \
  --profile ./simc_profiles/druid_feral.simc \
  --teacher-mode reward \
  --teacher-schedule hard \
  --teacher-start-weight 1.0 \
  --teacher-end-weight 0.0 \
  --teacher-switch-step 4000000 \
  --teacher-reward-scale 250000
```

Teacher curriculum flag semantics (v2):
- `--dps-reward-mode {delta,sparse}`
  - `delta`: default. Per-step SimC damage deltas are used and summed over the episode.
  - With `--gamma 1.0`, `delta` is return-equivalent to terminal total damage, but with much better credit assignment.
  - `sparse`: only terminal DPS reward (`total_damage`) is used.
- `--teacher-mode {off,reward}`
  - `off`: no teacher reward, policy trains on DPS reward mode only.
  - `reward`: enable teacher-match reward and blend it with DPS reward.
- `--teacher-schedule {linear,hard}`
  - `linear`: teacher weight `w` interpolates from start to end over `anneal_steps`.
  - `hard`: teacher weight is `start` before `switch_step`, then `end` after.
- `--teacher-start-weight` / `--teacher-end-weight`
  - Blend weight range for teacher signal. Both are clamped to `[0, 1]`.
  - Reward blend is: `r = (1 - w) * r_dps + w * r_teacher`.
- `--teacher-anneal-steps`
  - Used only for `linear`.
  - `0` means auto: `total_timesteps`.
  - Why `0` default: with sparse terminal DPS reward, full-run anneal avoids an abrupt handoff.
  - If you want a pure-DPS tail while keeping smooth handoff, set it below total timesteps (example: `0.8 * total_timesteps`).
- `--teacher-switch-step`
  - Used only for `hard`.
  - `0` means auto: `total_timesteps // 3`.
- `--teacher-reward-scale`
  - Exact-match bonus per decision when chosen action equals teacher action.
  - `r_teacher = teacher_reward_scale` on match, otherwise `0`.
- `--teacher-update-freq`
  - How often (global learner timesteps) the trainer pushes a new teacher weight to all subprocess envs via `env_method`.
  - Lower = closer tracking of the schedule but more IPC overhead.
  - Higher = cheaper but weight changes occur in larger jumps.
- `--teacher-off-stabilize`
  - Optional PPO stabilization at first timestep where teacher weight reaches `0`.
  - One-shot mode (default):
    - LR scaling by `--teacher-off-lr-scale` (default `0.2`)
    - clip range override to `--teacher-off-clip-range` (default `0.1`)
  - Adaptive mode (`--teacher-off-adaptive`):
    - lasts `--teacher-off-adaptive-steps` timesteps (default `1000000`)
    - clamps LR with floor `--teacher-off-min-lr` (default `5e-5`)
    - boosts entropy by `--teacher-off-ent-coef-scale` (default `2.0`) up to `--teacher-off-ent-coef-max` (default `0.03`)
    - restores original PPO schedules after the adaptive window ends
  - Intended to reduce collapse risk when switching from dense teacher signal to sparse terminal DPS-only learning.
- `--eval-drop-stop-pct`
  - Optional regression guard based on held-out eval damage.
  - If eval damage drops by this percentage from best and stays degraded for `--eval-drop-stop-patience` consecutive evals, training is stopped automatically.
  - `0` disables this guard.
- `--eval-drop-stop-patience`
  - Number of consecutive degraded eval passes required before stopping.

Reward behavior in v2 training:
- Reward blend at each step: `r = (1 - w) * r_dps + w * r_teacher`.
- Teacher term:
  - `r_teacher = teacher_reward_scale` on teacher match, else `0`.
- DPS term depends on `--dps-reward-mode`:
  - `sparse`:
    - non-terminal: `r_dps = 0`
    - terminal: `r_dps = total_damage`
  - `delta`:
    - non-terminal: `r_dps = simc_step_damage_delta`
    - terminal: tail delta so episode sum equals `total_damage`

Choosing `--teacher-reward-scale`:
- Start around `100000` for Feral-sized profiles.
- Increase toward `250000` if agent ignores teacher actions early.
- Decrease toward `50000` if teacher dominates for too long and DPS pickup is delayed after anneal.
- Keep schedule in mind: high start weight (`w`) plus high scale makes early learning almost pure imitation by design.

## 2) Prepare a profile

You can set RL options directly in the profile file:

```simc
# RL bridge settings
rl_enable=1
rl_stdio=1
rl_trace=0

# Optional observation features
rl_observe_buffs=tigers_fury,clearcasting
rl_observe_dots=rip,rake

# Typical RL run shape
iterations=1
threads=1
max_time=300
```

Notes:
- `iterations=1` is required for episode semantics.
- `threads=1` is still recommended explicitly, even though `rl_stdio=1` enforces it.

### Profile checklist before running Python

- Profile contains `rl_enable=1` and `rl_stdio=1`.
- Profile is valid in normal simc mode.
- Profile does not rely on unsupported external automation for action choice.
- `max_time` is set to your intended episode horizon.

## 3) Run from Python (`python_rl_env/rl_bridge.py`)

### Python environment setup

From `engine/rl/python_rl_env`:

```bash
uv sync
```

The script has multiple entry modes controlled by `--mode`.

Function mapping:
- `--mode demo` -> `run_demo(args)`
- `--mode multi` -> `run_multi_thread_sb3(args)` (main training flow)
- `--mode eval` -> `run_evaluation(args)` (important for final DPS validation)
- `--mode single` -> `run_single_thread_sb3(args)` (simple single-env SB3 run)

Base required arguments:
- `--simc` path to simc executable
- `--profile` path to `.simc` profile

### Demo mode (`run_demo`)

Good first smoke test for JSON bridge and action masks.

```bash
uv run python rl_bridge.py \
  --mode demo \
  --simc ../../../out/build/x64-Debug/simc.exe \
  --profile ../../../profiles/my_rl_profile.simc \
  --episodes 3
```

What it does:
- Spawns one `SimcEnv`.
- Uses random legal actions.
- Prints early-step action/reward/state debug output.

### Multi mode (`run_multi_thread_sb3`) - recommended training mode

This is the main training workflow and includes:
- parallel envs (`SubprocVecEnv`, currently hardcoded to 12)
- reward normalization (`VecNormalize`)
- checkpointing (`best_model`, `latest_model`, `final_model`)
- resume support

Start new training run:

```bash
uv run python rl_bridge.py \
  --mode multi \
  --simc ../../../out/build/x64-Debug/simc.exe \
  --profile ../../../profiles/my_rl_profile.simc
```

Resume existing run:

```bash
uv run python rl_bridge.py \
  --mode multi \
  --simc ../../../out/build/x64-Debug/simc.exe \
  --profile ../../../profiles/my_rl_profile.simc \
  --resume ./training_runs/simc_ppo_YYYYMMDD_HHMMSS
```

Artifacts are written under:
- `training_runs/simc_ppo_<timestamp>/logs`
- `training_runs/simc_ppo_<timestamp>/models`

Important model files:
- `best_model.zip`
- `latest_model.zip`
- `final_model.zip`
- `vec_normalize.pkl`

### Eval mode (`run_evaluation`) - important for report-quality output

Uses a trained model to control simc for many iterations and produces normal simc reports (HTML + text).

```bash
uv run python rl_bridge.py \
  --mode eval \
  --simc ../../../out/build/x64-Debug/simc.exe \
  --profile ../../../profiles/my_rl_profile.simc \
  --model-dir ./training_runs/simc_ppo_YYYYMMDD_HHMMSS \
  --iterations 2000 \
  --output-dir ./training_runs/simc_ppo_YYYYMMDD_HHMMSS/evaluation
```

Key eval args:
- `--model-dir` (required in eval mode): training run directory containing `models/`
- `--iterations`: number of sim iterations for statistical result
- `--output-dir`: where HTML/text reports are written

Model selection priority in eval:
1. `models/best_model.zip`
2. `models/final_model.zip`
3. `models/interrupted_model.zip`
4. `models/latest_model.zip`

### Single mode (`run_single_thread_sb3`)

Small SB3 sanity run with one environment:

```bash
uv run python rl_bridge.py \
  --mode single \
  --simc ../../../out/build/x64-Debug/simc.exe \
  --profile ../../../profiles/my_rl_profile.simc
```

### CLI argument summary (from current script)

- `--mode {demo,single,multi,eval}` (default `multi`)
- `--simc` (default `./simc`)
- `--profile` (required)
- `--episodes` (used by demo)
- `--resume` (used by multi)
- `--model-dir` (required for eval)
- `--iterations` (used by eval, default `2000`)
- `--output-dir` (optional eval report directory)
- `--no-blacklist` (disable Python-side action blacklist)

### Typical workflow loop

1. Run `demo` to validate bridge + masks.
2. Run `multi` to train and produce a run directory under `training_runs/`.
3. Run `eval` against that run directory for high-iteration statistical reports.
4. Repeat with adjusted profile/action blacklist/hyperparameters.

### Output locations and what to inspect

- Training logs: `training_runs/<run>/logs`
- Models: `training_runs/<run>/models`
- Eval reports (default): `training_runs/<run>/evaluation`

Inspect first:
- `models/best_model.zip`
- `models/vec_normalize.pkl`
- latest evaluation HTML report

### Ready-to-use command scripts

Template scripts are provided in this folder:
- `run_rl_commands.sh`
- `run_rl_commands.bat`

Both include the same modes as this README (`demo`, `multi`, `resume`, `eval`, `single`) and have editable variables at the top:
- simc binary path
- profile path
- training run directory
- evaluation iteration count

## 4) Optional tracing

Set:

```simc
rl_trace=1
rl_trace_file=rl_trace.jsonl
```

to emit decision-level traces for debugging legality, action choice, and state progression.

## Where To Put Settings

- Prefer profile-based settings when running many experiments with consistent config.
- Override per run from CLI for sweep scripts.

Example CLI override:

```bash
simc my_profile.simc rl_enable=1 rl_stdio=1 rl_trace=0 max_time=300 iterations=1
```

## Troubleshooting

- No JSON `step` output:
  - Check `rl_enable=1` and `rl_stdio=1`.
  - Confirm episode actor is a player (not pet/enemy).
- Process blocks waiting:
  - Python must write exactly one integer action per step.
- Unstable training:
  - For v2 teacher curriculum, tune `--teacher-reward-scale` and anneal settings before changing PPO hyperparameters.
- Dense delta reward is the default and recommended v2 training mode.
- If you explicitly experiment with sparse terminal reward and it collapses, switch back to `--dps-reward-mode delta` and use `--gamma 1.0 --gae-lambda 1.0`.
  - If running legacy `rl_bridge.py`, intermediate rewards can still help (`intermediate_rewards=True` in that script).
  - Start with a reduced action space via Python blacklist if needed.
- Buff/Dot features always zero:
  - Verify `rl_observe_buffs` / `rl_observe_dots` labels match tokenized simc names.
- Expected actions missing in Python action list:
  - Run with `--no-blacklist` to verify they are not being filtered client-side.
- Evaluation cannot find model:
  - Check `--model-dir` points to run root containing a `models/` folder.
  - Ensure at least one of `best_model.zip`, `final_model.zip`, `interrupted_model.zip`, `latest_model.zip` exists.
- Training appears too slow:
  - Confirm your simc executable is optimized/release build.
  - Reduce profile complexity and/or shorten `max_time` for early experiments.

## Notes For Future Extension

Likely next extension points:
- Multi-actor RL control.
- More observation channels (cooldown groups, target stats, encounter context).
- Optional different reward shaping modes.
- Optional off-gcd RL control mode.
