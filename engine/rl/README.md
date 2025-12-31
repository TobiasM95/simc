# engine/rl

This folder contains an **experimental reinforcement-learning (RL) control path** for the SimulationCraft engine.

At runtime, the engine can emit a **Gymnasium-like “step”** at each decision point:

- **Observation**: current game state (time, resources, spec-specific buffs/dots, …)
- **Action space**: a stable list of player-castable actions (plus a few pseudo-actions)
- **Action mask**: which actions are legal right now
- **Reward**: damage delta attributed to the previous decision

…and then call a **policy callback** that returns an action index.

This README documents what the code currently does (hook points, filtering rules, JSON protocol, and the Python bridge).

## Quick Start (Python / Gymnasium)

The Gymnasium wrapper lives at `engine/rl/python_rl_env/rl_bridge.py`.

From the repo root:

```bash
python -m pip install -r engine/rl/requirements.txt

# Random-action smoke test
python engine/rl/python_rl_env/rl_bridge.py --simc ./simc --profile ./profiles/TWW3_Raid.simc --mode demo --episodes 1

# Parallel training (MaskablePPO)
python engine/rl/python_rl_env/rl_bridge.py --simc ./simc --profile ./tww_st_feral.simc --mode multi

# Evaluate a trained run over many iterations (generates HTML/text reports)
python engine/rl/python_rl_env/rl_bridge.py --simc ./simc --profile ./tww_st_feral.simc --mode eval \
   --model-dir ./training_runs/simc_ppo_20251229_150600 --iterations 2000
```

## Safety / Defaults

In this fork, `sim_t` defaults `rl_enable=true`, `rl_stdio=true`, and `rl_trace=true`.

That means: if you run `simc` without overriding options, the sim can route action selection through the RL policy (default policy is `rl::dummy_policy`, which is effectively “random legal action”).

For **normal SimulationCraft usage**, explicitly disable RL:

```text
rl_enable=0 rl_stdio=0 rl_trace=0
```

## Where the RL hook runs

The hook point is:

- `player_t::select_action(const action_priority_list_t&, execute_type, const action_t*)`

The RL policy is invoked only when all of the following are true:

- `sim->rl_enable` is true
- This is the **root** `select_action` call (not inside `call_action_list` recursion)
  - Implemented as: `visited_apls_ == list.internal_id_mask`
- `execute_type` is **FOREGROUND**
  - OFF_GCD and CAST_WHILE_CASTING contexts do **not** call the RL policy
- The actor is a **player character** (not an enemy and not a pet)

## Policy API

Policies are simple C function pointers:

```cpp
using policy_fn_t = std::size_t (*)(const step_input_t&, void* user_data);

void set_policy(policy_fn_t fn, void* user_data = nullptr);
policy_fn_t get_policy();
void* get_policy_user_data();
```

Built-ins:

- `rl::dummy_policy`: round-robin selection over currently-legal actions
- `rl::stdio_policy`: JSON-over-stdio bridge for external policies (Python, etc.)

When `rl_stdio=1`, the player lazily installs `rl::stdio_policy` on the first RL decision.

## Action space (what the agent can choose)

### Source of actions

The “real” actions come from `player_t::action_list`, but there is an important extra step:

- On first use, `rl::build_action_list()` calls an internal helper that **discovers baseline class/spec actions from DBC + talent data** and creates missing actions via `player_t::create_action()`.
- These created actions are marked with a synthetic APL (`_rl_baseline`) so they pass the action filter.

The goal is that the RL agent sees **all castable abilities available to the spec**, not only whatever happened to be referenced in the APL.

### Filtering rules (`rl::is_exposed_action`)

An action is exposed to RL only if:

- It is not `background`, not `proc`, and not `dual` (i.e., not an internal tick/impact/proc action)
- It has a non-null `action_list` pointer
  - This intentionally excludes “secondary/internal” actions created during execution, unless they were created from APL parsing or the RL baseline APL
- It is not APL control flow (`ACTION_CALL`, `ACTION_VARIABLE`, `ACTION_SEQUENCE`)
- It is not a “wait barrier” (`option.wait_on_ready == 1`)
- It is not `wait` or `pool_resource` (those are replaced by fixed-duration pseudo-actions)
- It does not require a resource that the player doesn’t have active

The allowed action types are: `ACTION_SPELL`, `ACTION_ATTACK`, `ACTION_HEAL`, `ACTION_ABSORB`, `ACTION_USE`, `ACTION_OTHER`.

### Deduplication

Actions are deduplicated by `action_t::name_str` (first instance wins).

### Pseudo-actions: fixed waits + pass

After the real actions, the engine appends pseudo-actions with **null action pointers**:

- `wait_0.1`, `wait_0.2`, `wait_0.5`, `wait_1.0`, `wait_1.5`
- `pass`

How they behave when selected:

- **Wait** pseudo-actions return a dedicated `rl_wait_action_t` that advances time by the fixed duration.
- **Pass** returns `nullptr` from `select_action`, which triggers SimulationCraft’s normal “no action selected” wait behavior. This avoids infinite loops from a zero-duration no-op.

Masking rules for pseudo-actions:

- Wait pseudo-actions are legal only in `execute_type::FOREGROUND`.
- Pass is always considered legal.

## Action mask + per-action features

Every decision step computes arrays aligned with the action list:

- `mask[i]`: 1 if action is legal now, else 0
- `cd_rem_s[i]`: cooldown remaining in seconds
- `cd_rem_n[i]`: cooldown remaining normalized by expected fight length
- `cd_charges_f[i]`: fractional cooldown charges (or 1.0 when not applicable)

For “real” actions, legality is based on engine semantics:

- `action_t::ready()` and `action_t::target_ready(target)`

APL `if=` conditionals are **not** applied. The agent is expected to learn preferences.

## Observation

`rl::build_observation(const player_t&)` exports:

- `t`, `fight_len`
- `time_rem`, `time_rem_n` (normalized by fight length)
- `ttd`, `ttd_n` (normalized by fight length)
- `gcd_rem`, `gcd_rem_n` (normalized by the player’s base GCD; falls back to 1.5s)
- `resource_pct[RESOURCE_MAX]`

Spec-specific observation channels:

- `spec_id`
- `buff_remains[]` (normalized by 30s), `buff_stacks[]`, `buff_labels[]`
- `dot_remains[]` (normalized by 30s), `dot_stacks[]`, `dot_labels[]`

The lists of tracked buffs/dots are configured per spec in `engine/rl/rl_spec_config.cpp` and cached via pointer lookup.

## Reward

Reward is attributed to the _previous_ decision point:

```text
reward = priority_iteration_dmg(now) - priority_iteration_dmg(last_decision)
```

Optional potential-based shaping exists:

$$r' = r + \gamma\,\Phi(s') - \Phi(s)$$

- The engine only applies shaping if a potential function has been installed via `rl::set_potential_fn`.
- The hook currently uses $\gamma = 0.99$.

## Tracing (JSONL)

When `rl_trace=1`, each decision writes a JSON object (mask, labels, chosen index, cooldown features, and the basic observation fields) to `rl_trace_file`.

Note: if `sim.threads > 0`, the implementation appends a per-thread suffix:

- `"{rl_trace_file}.{thread_index}.jsonl"`

So the default `rl_trace_file=rl_trace.jsonl` becomes `rl_trace.jsonl.0.jsonl`, `rl_trace.jsonl.1.jsonl`, etc.

When using `rl_stdio=1`, it’s generally recommended to disable tracing (`rl_trace=0`) to avoid extra I/O.

## Stdio bridge protocol (simc ↔ external policy)

When `rl_stdio=1` and `rl_enable=1`:

1. **Step** (simc → policy): one JSON line

```json
{
  "type": "step",
  "t": 1.5,
  "fight_len": 300.0,
  "time_rem": 298.5,
  "time_rem_n": 0.995,
  "ttd": 298.5,
  "ttd_n": 0.995,
  "gcd_rem": 0.0,
  "gcd_rem_n": 0.0,
  "reward": 12345.67,
  "spec_id": 103,
  "resource_pct": [0.0, 0.0, 1.0],
  "buff_remains": [0.0],
  "buff_stacks": [0],
  "buff_labels": ["some_buff"],
  "dot_remains": [0.0],
  "dot_stacks": [0],
  "dot_labels": ["some_dot"],
  "n": 128,
  "mask": [1, 0, 1],
  "labels": ["thrash", "wait_0.1", "pass"],
  "cd_rem_s": [0.0, 0.0, 0.0],
  "cd_rem_n": [0.0, 0.0, 0.0],
  "cd_charges_f": [1.0, 1.0, 1.0]
}
```

2. **Action** (policy → simc): a single integer action index (one line)

```text
3
```

The stdio policy validates the action against the mask; if invalid/illegal, it falls back to the first legal action.

3. **Done** (simc → policy): one JSON line at the end of each iteration

```json
{ "type": "done", "total_damage": 1234567.89, "fight_length": 300.0 }
```

### Threading constraint

The stdio bridge forces `threads=1` in `sim_t::setup()` to avoid concurrent stdin/stdout usage.

## Python bridge details

`engine/rl/python_rl_env/rl_bridge.py` provides:

- `SimcEnv`: a Gymnasium `Env` that spawns `simc` and speaks the stdio protocol
- `make_vec_env`: helper to create `SubprocVecEnv` for parallel training

CLI modes:

- `--mode demo`: random action demo with masking
- `--mode single`: single-env MaskablePPO training
- `--mode multi`: parallel MaskablePPO training (default)
- `--mode eval`: run many `iterations` through a trained model and output reports

`--profile` is required. Use `--simc` to point at your built `simc` executable.

## Notes

Some handwritten notes

### Useful commands:

Training from scratch:

```
uv run python rl_bridge.py --simc "C:\Users\tobim\Documents\Programming\MachineLearning\simc\out\build\x64-Debug\simc.exe" --profile "./tww_st_feral.simc" --mode "multi"
```

Evaluation:

```
uv run python rl_bridge.py --simc "C:\Users\tobim\Documents\Programming\MachineLearning\simc\out\build\x64-Debug\simc.exe" --profile "./tww_st_feral.simc" --mode "eval" --model-dir "training_runs/simc_ppo_20251229_150600" --iterations 2000
```
