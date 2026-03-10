## APL Teacher-to-DPS Curriculum (SimC RL v2)
  ### Summary

  Implement a teacher-student training mode where the teacher is the active SimC APL (profile APL if provided, otherwise generated default), and rewards transition from teacher-match to DPS using a configurable schedule (linear or hard switch). Scope is v2 only (rl_train_v2.py + C++ RL bridge).

  ### Public API / Interface Changes

  1. Add new sim option in sim.hpp and sim.cpp:
      - rl_teacher_enable (bool, default 0).
  2. Extend RL step payload in rl_interface.hpp and rl_interface.cpp:
      - teacher_idx (int, raw action index in same space as labels/mask, -1 if unavailable).
      - teacher_label (string, empty if unavailable).
  3. Add v2 CLI options in rl_train_v2.py:
      - --teacher-mode {off,reward} default off
      - --teacher-schedule {linear,hard} default linear
      - --teacher-start-weight default 1.0
      - --teacher-end-weight default 0.0
      - --teacher-anneal-steps default 0 (auto: total_timesteps // 3)
      - --teacher-switch-step default 0 (auto: total_timesteps // 3)
      - --teacher-reward-scale default 250000.0
      - --teacher-update-freq default 1000

  ### Implementation Plan

  1. C++ teacher extraction in player.cpp:
      - In RL foreground hook (player_t::select_action RL branch), when sim->rl_teacher_enable is true:
      - Compute teacher action using active APL selection path (temporarily disable RL hook to avoid recursion).
      - Map teacher action to current RL action-space index (rl_action_list index); fallback by label match if pointer mismatch.
      - If unresolved, set teacher_idx=-1, teacher_label="".
      - Keep existing policy selection and legality fallback unchanged.
  2. C++ protocol emission in rl_interface.cpp:
      - Add teacher_idx/teacher_label to step JSON every step.
      - Include teacher fields in trace output (trace_decision) for debugging parity.
  3. Python env reward blending in rl_train_v2.py:
      - Extend ObsConfig with raw->filtered index mapping for teacher index translation.
      - Store current state’s filtered teacher index during _parse_state.
      - In step(action), compute teacher_match from the pre-action state (the one the action was chosen from).
      - Keep DPS reward extraction as-is.
      - Compute mixed reward:
          - r_teacher = teacher_reward_scale if teacher_match else 0.0
          - r_mixed = (1 - w) * r_dps + w * r_teacher
          - w is mutable env weight (set_teacher_weight), clamped [0,1].
      - Terminal step uses same mix for tail DPS reward and the final action’s teacher-match bonus.
      - If teacher is unavailable (teacher_idx < 0 or blacklisted out), teacher component is 0.
  4. Training schedule integration in rl_train_v2.py:
      - Add schedule calculator in training callback based on num_timesteps.
      - linear: interpolate from start_weight to end_weight over anneal_steps.
      - hard: start_weight before switch_step, else end_weight.
      - Push weight to all subprocess envs via envs.env_method("set_teacher_weight", w) every teacher_update_freq.
      - Eval env remains teacher-off so checkpoint selection stays pure DPS.
  5. Command wiring in rl_train_v2.py:
      - Add rl_teacher_enable=1 to simc command only when teacher-mode != off.
  6. Docs update in README.md:
      - Document rl_teacher_enable.
      - Document v2 teacher curriculum flags and examples for both gradual and sudden transition.

  ### Test Cases and Scenarios

  1. Protocol compatibility:
      - With teacher-mode=off, training/eval behavior remains unchanged and parsing still works.
  2. Teacher field presence:
      - With teacher-mode=reward, first step JSON contains teacher_idx and teacher_label.
  3. Reward math unit checks (Python):
      - Verify mixed reward formula for w=1, w=0.5, w=0.
      - Verify no teacher contribution when teacher_idx=-1.
  4. Schedule behavior:
      - Linear schedule reaches expected weights at t=0, mid, and end.
      - Hard schedule flips exactly at switch step.
  5. Integration smoke:
      - Short train run (~20k steps) with teacher mode enabled does not crash and logs changing teacher weight.
  6. DPS eval isolation:
      - run_eval uses teacher-off and reports pure DPS metrics.
  7. Regression guard:
      - Compare short A/B runs (teacher off vs teacher mode with start=end=0) and confirm no meaningful DPS drift (>0.5%) over multiple seeds.

  ### Assumptions and Defaults

  1. Teacher source is the active APL only (profile APL when provided; otherwise generated default).
  2. Scope is v2 training/eval script only; legacy rl_bridge.py is untouched.
  3. Teacher objective is exact-match bonus (no top-k/action-family shaping).
  4. Default transition is gradual linear anneal from 1.0 to 0.0.
  5. Sudden transition is available via --teacher-schedule hard.
  6. If teacher action is unavailable or filtered out, teacher reward is 0 for that decision.