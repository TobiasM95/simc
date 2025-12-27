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

## Policy API

```cpp
using policy_fn_t = std::size_t (*)(const step_input_t&, void* user_data);

void set_policy(policy_fn_t fn, void* user_data = nullptr);
```

The policy receives the full step input and returns an action index. If the chosen action is illegal, the engine falls back to the first legal action.

## Intended next step (Python bridge)

The API is shaped for easy IPC:

- Serialize step_input_t to JSON/binary
- Send to Python over pipe/socket
- Receive action index
- Return to engine
