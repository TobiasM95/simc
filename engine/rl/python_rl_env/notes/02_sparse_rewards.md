• Yes, absolutely. The agent can still discover 3-4 step “setup -> payoff” sequences with step rewards, but it gets much harder when reward is “damage delta” contaminated by past DoTs and unrelated ticking effects.

  What you’re seeing is classic credit assignment:

  - Dense noisy reward: learns something, but can misattribute payoff to wrong actions.
  - Pure terminal reward: correct objective, but exploration variance explodes, so PPO gets almost no signal.

  Practical middle ground usually works best:

  1. Keep dense reward, but add a small sequence-aware shaping term

  - Example: reward buff/condition creation that is known prerequisite for payoff (combo points, proc-enabler, window alignment), not just raw damage.

  2. Use a curriculum

  - Early: more teacher/shaping weight.
  - Later: anneal to pure DPS objective.

  3. Shorten horizon during training

  - Train on shorter fights first (or fixed short windows), then extend.
  - This dramatically helps discover delayed combos.

  4. Increase temporal credit capacity

  - Slightly higher gamma, use GAE tuning, maybe recurrent policy if state aliasing exists.

  5. Use event-triggered sparse components, not only terminal

  - Keep final damage reward, but also sparse rewards at key milestones (window executed correctly, finisher landed under buff, etc.).

  So: your intuition is right on both sides. Pure step-damage is noisy for combos, pure terminal is too sparse. Hybrid reward + curriculum is usually the stable path.


› as for 1. I'm kinda a purist and don't want complex reward functions since I want to be able to discover new playstyles etc. that might not be shaped properly by preconfigured reward functions. 2. that is something that might be a good approach, I already made a plan to implement a teacher/student model 3. that is also a good idea in
  general although in WoW these sequences most of the time don't happen at the beginning of a fight because of cooldown abilities 4. interesting training details 5. similar to 1.


• Then your direction is coherent: keep reward “clean” and move complexity into training procedure, not reward design.

  Given your constraints, best fit is:

  1. Objective remains pure DPS (or damage delta), no handcrafted action rewards.
  2. Use teacher->student curriculum only as a temporary optimization scaffold.
  3. Add exploration/optimization changes to recover long-horizon sequences without reward shaping:

  - Higher gamma (e.g. 0.997-0.999) for 300s fights.
  - Increase n_steps so returns include longer context.
  - Lower learning rate once policy starts stabilizing.
  - Slightly higher entropy early, anneal down.

  For your cooldown-window concern (important):
  You can still use shorter training horizons without losing those combos by randomizing episode start time/state (“mid-fight starts”) instead of always starting at t=0. That exposes the agent to post-opener windows while keeping horizons shorter.

  So the purist version is basically:

  - Clean reward
  - Curriculum via teacher guidance
  - Better horizon handling (gamma/n_steps + mid-fight starts)
  - Annealed exploration

  That keeps discovery potential high while avoiding brittle handcrafted rewards.