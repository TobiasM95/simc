# engine/rl

This folder contains a simple, self-contained **RL decision interface** for the SimulationCraft engine.

The goal is to let the sim produce a **Gym/Gymnasium-like step input** at each decision point:

- **Observation** (current game state)
- **Action space** (all player abilities, deduplicated)
- **Action mask** (which abilities are usable right now)
- **Reward** (damage dealt since last decision)

…and then let a **policy** pick an action index.

## Design Principles

1. **APL-independent**: The action space comes from `player_t::action_list` (all abilities the player has), not from APL structure
2. **Deduplicated**: One entry per unique ability name
3. **Unified space**: Single action space (no separate foreground/off_gcd/cast_while_casting)
4. **Simple**: abilities + mask (ready or not) + basic features

## Where it hooks in

The engine hook point is:

- `player_t::select_action(const action_priority_list_t&, execute_type, const action_t*)`

The RL path is enabled via a sim option and only triggers for the **root** `select_action` call.

## How the action set is built

The action set comes from `player_t::action_list`, which contains ALL actions created for the player. This includes abilities from all APL sub-lists (builder, finisher, cooldown, etc.).

### Filtering rules

`rl::build_action_list()` filters out:

- `background == true` (internal actions like DoT ticks)
- `ACTION_CALL`, `ACTION_VARIABLE`, `ACTION_SEQUENCE` (APL control flow)
- `option.wait_on_ready == 1` (wait barriers)

It keeps combat actions:

- `ACTION_SPELL`, `ACTION_ATTACK`, `ACTION_HEAL`, `ACTION_ABSORB`, `ACTION_USE`, `ACTION_OTHER`

### Deduplication

Actions are deduplicated by `name_str`. The first instance of each ability name is kept.

## Action mask: what "legal" means

The mask represents **game legality** (cooldowns, resources, target availability):

- `action_t::ready()` - cooldown ready, resources available, etc.
- `action_t::target_ready(target)` - target is valid

This intentionally does **not** apply APL `if=` conditions. The RL agent should learn preferences.

## Observations

`rl::build_observation(player_t&)` exports:

- Current time and expected fight length
- Time remaining (raw + normalized)
- Target TTD (raw + normalized)
- GCD remaining (raw + normalized)
- Player resource percentages (as a vector)

## Per-action features

For each action in the space:

- Legality mask (0 or 1)
- Cooldown remaining (seconds + normalized)
- Fractional charges

## Reward signal

Simple **damage delta** since last decision:

```
reward(t) = priority_iteration_dmg(t) - priority_iteration_dmg(last_step)
```

## Caching

The action list is built **once** on first RL decision and cached on the player:

- `player_t::rl_action_list` - action pointers
- `player_t::rl_action_labels` - ability names

Only the mask and per-action features are updated each decision.

## Tracing (validation)

JSON Lines trace for debugging:

- `rl_trace=1` enables tracing
- `rl_trace_file=<path>` sets output filename (default `rl_trace.jsonl`)

Each decision writes one JSON object containing observation, action labels, mask, chosen index, and reward.

## Enabling RL

Run with:

```
rl_enable=1
```

This routes action selection through the RL policy callback instead of the APL.

## Python Bridge (Gymnasium/Masked PPO)

The `rl_bridge.py` module provides a Gymnasium-compatible environment that communicates with simc via stdin/stdout.

### Quick Start

```bash
# Run simc with stdio bridge
./simc Feral.simc rl_enable=1 rl_stdio=1 rl_trace=0 iterations=1

# Or use the Python wrapper
python engine/rl/rl_bridge.py --simc ./simc --profile Feral.simc
```

### Using with Stable-Baselines3 MaskablePPO

```python
from sb3_contrib import MaskablePPO
from engine.rl.rl_bridge import SimcEnv, make_vec_env

# Single environment
env = SimcEnv(simc_path="./simc", profile="Feral.simc")

# Or parallel environments for faster training
envs = make_vec_env(8, simc_path="./simc", profile="Feral.simc")

# Train with MaskablePPO (action masking)
model = MaskablePPO("MlpPolicy", envs, verbose=1)
model.learn(total_timesteps=100000)
```

### Stdio Bridge Protocol

When `rl_stdio=1` is enabled:

1. **State message** (simc → Python): JSON line with observation

   ```json
   {"type":"step","t":1.5,"time_rem":298.5,"time_rem_n":0.995,"gcd_rem":0,"gcd_rem_n":0,"ttd":298.5,"ttd_n":0.995,"reward":12345.67,"resource_pct":[...],"n":25,"mask":[1,0,1,...],"labels":["fireball",...],"cd_rem_s":[...],"cd_rem_n":[...],"cd_charges_f":[...]}
   ```

2. **Action message** (Python → simc): Single integer (action index)

   ```
   3
   ```

3. **Episode end** (simc → Python): JSON line with done signal
   ```json
   { "type": "done", "total_damage": 1234567.89, "fight_length": 300.0 }
   ```

### Options

- `rl_stdio=1` - Enable stdin/stdout bridge (forces `threads=1`)
- `rl_trace=0` - Disable file tracing when using stdio (recommended)
- `iterations=N` - Run N episodes before simc exits

## Policy API

```cpp
using policy_fn_t = std::size_t (*)(const step_input_t&, void* user_data);

void set_policy(policy_fn_t fn, void* user_data = nullptr);
```

The policy receives the full step input and returns an action index. If the chosen action is illegal, the engine falls back to the first legal action.

### Built-in Policies

- `dummy_policy` - Round-robin through legal actions (default for testing)
- `stdio_policy` - Reads/writes JSON via stdin/stdout (enabled by `rl_stdio=1`)
