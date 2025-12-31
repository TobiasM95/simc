"""
SimulationCraft RL Bridge - Gymnasium-compatible environment for training RL agents.

This module provides a Gymnasium environment wrapper that communicates with the
SimulationCraft simulator via stdin/stdout JSON messages.

Usage:
    # Single environment
    env = SimcEnv(simc_path="./simc", profile="Feral.simc")
    obs, info = env.reset()
    while True:
        action = policy(obs, info["mask"])
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    # Vectorized environments (for parallel training)
    from stable_baselines3.common.vec_env import SubprocVecEnv
    envs = SubprocVecEnv([lambda: SimcEnv(...) for _ in range(num_envs)])

Requirements:
    - gymnasium
    - numpy
    - sb3-contrib (for MaskablePPO)
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# Base GCD for normalization (1.5 seconds is the default GCD before haste)
BASE_GCD_SECONDS = 1.5

# blacklist to exclude certaion actions inside the python gym environment
# these don't get masked out but instead are removed from the action space entirely
GLOBAL_ACTION_BLACKLIST = {
    "invoke_external_buff",
    "snapshot_stats",
    "cancel_buff",
    "use_item_arazs_ritual_forge",
    "do_treacherous_transmitter_task",
    "run_action_list",
    "entangling_roots",
    "shred",
    "rake",
    "skull_bash",
    "cat_form",
    "rip",
}


class SimcEnv(gym.Env):
    """
    Gymnasium environment that wraps SimulationCraft with RL stdio bridge.

    The environment spawns a simc subprocess and communicates via JSON over
    stdin/stdout. Each step sends an action index and receives the next state.

    Observation Space:
        A dictionary containing:
        - "obs": Box of continuous features:
            [gcd_rem_norm, ttd_norm] + resource_pct[18] + cd_charges_f[n_actions]
            where normalized values are computed using initial TTD and base GCD.
        - "mask": MultiBinary action mask (1 = legal, 0 = illegal)

    Action Space:
        Discrete(n) where n is the number of possible actions.
        Actions are masked - only actions with mask[i]=1 are valid.
        Actions can be filtered using the action_blacklist parameter.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        simc_path: str = "./simc",
        profile: Optional[str] = None,
        simc_args: Optional[list[str]] = None,
        fight_length: float = 300.0,
        iterations: int = 1,
        seed: Optional[int] = None,
        intermediate_rewards: bool = False,
        action_blacklist: Optional[set[str]] = None,
    ):
        """
        Initialize the SimulationCraft RL environment.

        Args:
            simc_path: Path to the simc executable.
            profile: Path to a .simc profile file to load.
            simc_args: Additional command-line arguments for simc.
            fight_length: Expected fight length in seconds (for normalization).
            iterations: Number of iterations per episode (typically 1 for RL).
            seed: Random seed for reproducibility.
            intermediate_rewards: Whether to use intermediate rewards.
            action_blacklist: Set of action label names to exclude from the action
                space (e.g., {"invoke_external_buff", "snapshot_stats"}).
        """
        super().__init__()

        self.simc_path = simc_path
        self.profile = profile
        self.simc_args = simc_args or []
        self.fight_length = fight_length
        self.iterations = iterations
        self.seed_value = seed
        self.intermediate_rewards = intermediate_rewards
        self._action_blacklist: set[str] = action_blacklist or set()

        self._process: Optional[subprocess.Popen] = None
        self._num_resources = 18  # RESOURCE_MAX in simc

        # Action space mapping (filtered -> raw simc index)
        self._action_index_map: list[int] = []  # Maps filtered index to raw simc index
        self._filtered_labels: list[str] = []  # Labels after blacklist filtering
        self._num_actions: int = 0  # Number of filtered actions
        self._raw_num_actions: int = 0  # Original number of actions from simc

        # Spec-specific observation tracking
        self._num_buffs: int = 0  # Number of tracked buffs for this spec
        self._num_dots: int = 0  # Number of tracked dots for this spec
        self._buff_labels: list[str] = []  # Names of tracked buffs
        self._dot_labels: list[str] = []  # Names of tracked dots
        self._spec_id: int = 0  # Specialization ID

        # Initial TTD for normalization (captured from first observation)
        self._initial_ttd: Optional[float] = None

        # Observation space: continuous features
        # [gcd_rem_norm, ttd_norm] + resource_pct[18] + cd_charges_f[n_actions]
        #   + buff_remains[num_buffs] + buff_stacks[num_buffs]
        #   + dot_remains[num_dots] + dot_stacks[num_dots]
        # We'll set the actual size after first step when we know num_actions
        self._obs_dim = (
            2 + self._num_resources
        )  # Base dimensions (before action features)

        self._spaces_initialized = False
        self._current_state: Optional[dict] = None
        self._total_reward = 0.0

        # Stderr capture thread and buffer
        self._stderr_buffer: list[str] = []
        self._stderr_lock = threading.Lock()
        self._stderr_thread: Optional[threading.Thread] = None

        # Probe the simc process once to discover the actual action space size
        # This is necessary because Gymnasium/SB3 expects fixed observation/action spaces
        self._probe_action_space()

    def _build_command(self) -> list[str]:
        """Build the simc command line."""
        cmd = [self.simc_path]

        if self.profile:
            cmd.append(self.profile)

        # Enable RL stdio bridge
        cmd.extend(
            [
                "rl_enable=1",
                "rl_stdio=1",
                "rl_trace=0",  # Disable file tracing when using stdio
                f"iterations={self.iterations}",
                f"max_time={self.fight_length}",
            ]
        )

        if self.seed_value is not None:
            cmd.append(f"seed={self.seed_value}")

        cmd.extend(self.simc_args)

        return cmd

    def _probe_action_space(self) -> None:
        """
        Probe simc to discover the actual action space size and spec observation config.

        This runs a quick simulation to get the first state message,
        which contains the action labels, space dimensions, and spec-specific
        buff/dot tracking configuration.
        The process is then closed - actual training uses fresh processes.
        """
        self._start_process()
        try:
            msg = self._read_message()
            if msg.get("type") == "done":
                raise RuntimeError("Episode ended immediately during probe")

            # Extract action space info and apply blacklist filtering
            raw_num_actions = msg["n"]
            raw_labels = msg.get(
                "labels", [f"action_{i}" for i in range(raw_num_actions)]
            )

            self._raw_num_actions = raw_num_actions
            self._action_index_map = []
            self._filtered_labels = []

            for i, label in enumerate(raw_labels):
                if label not in self._action_blacklist:
                    self._action_index_map.append(i)
                    self._filtered_labels.append(label)

            self._num_actions = len(self._filtered_labels)

            # Extract spec-specific observation configuration
            self._spec_id = msg.get("spec_id", 0)
            self._buff_labels = msg.get("buff_labels", [])
            self._dot_labels = msg.get("dot_labels", [])
            self._num_buffs = len(self._buff_labels)
            self._num_dots = len(self._dot_labels)

            # Calculate observation dimension:
            # 2 (gcd_rem_norm, ttd_norm) + 18 (resource_pct) + n_actions (cd_charges)
            # + 2*num_buffs (buff_remains + buff_stacks) + 2*num_dots (dot_remains + dot_stacks)
            obs_dim = (
                2
                + self._num_resources
                + self._num_actions
                + 2 * self._num_buffs
                + 2 * self._num_dots
            )

            self.observation_space = spaces.Dict(
                {
                    "obs": spaces.Box(
                        low=-1.0, high=100.0, shape=(obs_dim,), dtype=np.float32
                    ),
                    "mask": spaces.MultiBinary(self._num_actions),
                }
            )
            self.action_space = spaces.Discrete(self._num_actions)
            self._spaces_initialized = True

        finally:
            self._close_process()

    def _stderr_reader(self) -> None:
        """Background thread that reads stderr and accumulates output."""
        try:
            if self._process and self._process.stderr:
                for line in self._process.stderr:
                    with self._stderr_lock:
                        self._stderr_buffer.append(line)
                        # Also print to console in real-time for debugging
                        print(f"[simc stderr] {line}", end="", file=sys.stderr)
        except Exception:
            pass  # Process may have been closed

    def _get_stderr_output(self) -> str:
        """Get all captured stderr output."""
        with self._stderr_lock:
            return "".join(self._stderr_buffer)

    def _start_process(self) -> None:
        """Start the simc subprocess."""
        if self._process is not None:
            self._close_process()

        # Clear stderr buffer
        with self._stderr_lock:
            self._stderr_buffer.clear()

        cmd = self._build_command()
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line-buffered for responsive communication
        )

        # Start background thread to read stderr
        self._stderr_thread = threading.Thread(target=self._stderr_reader, daemon=True)
        self._stderr_thread.start()

    def _close_process(self) -> None:
        """Close the simc subprocess."""
        if self._process is not None:
            try:
                self._process.stdin.close()
                self._process.stdout.close()
                self._process.stderr.close()
                self._process.terminate()
                self._process.wait(timeout=5.0)
            except Exception:
                self._process.kill()
            finally:
                self._process = None
                # Wait for stderr thread to finish
                if self._stderr_thread and self._stderr_thread.is_alive():
                    self._stderr_thread.join(timeout=1.0)
                self._stderr_thread = None

    def _read_message(self) -> dict:
        """Read a JSON message from simc stdout."""
        if self._process is None or self._process.stdout is None:
            raise RuntimeError("Process not running")

        while True:
            line = self._process.stdout.readline()
            if not line:
                # Process ended unexpectedly - gather as much diagnostic info as possible
                exit_code = self._process.poll()

                # Give stderr thread a moment to capture final output
                import time

                time.sleep(0.1)

                # Get all captured stderr from our background thread
                stderr = self._get_stderr_output()

                # Provide more diagnostic information
                error_msg = f"simc process ended unexpectedly.\n"
                error_msg += f"  Exit code: {exit_code}\n"
                if exit_code is not None:
                    if exit_code < 0:
                        # Negative exit codes on Unix indicate signal
                        import signal

                        try:
                            sig_name = signal.Signals(-exit_code).name
                            error_msg += f"  Signal: {sig_name} ({-exit_code})\n"
                        except (ValueError, AttributeError):
                            error_msg += f"  Signal: {-exit_code}\n"
                    elif exit_code == 0xC0000005:
                        error_msg += (
                            "  Windows: Access Violation (EXCEPTION_ACCESS_VIOLATION)\n"
                        )
                    elif exit_code == 0xC0000094:
                        error_msg += "  Windows: Integer Divide by Zero\n"
                    elif exit_code == 0xC00000FD:
                        error_msg += "  Windows: Stack Overflow\n"
                    elif exit_code == 3:
                        error_msg += "  Likely: Assertion failure or abort()\n"
                error_msg += f"  stderr:\n{stderr if stderr else '(empty)'}"

                raise RuntimeError(error_msg)
            if not line.lstrip().startswith("{"):
                continue  # Skip debug or non-JSON lines
            try:
                return json.loads(line.strip())
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Failed to parse JSON from simc: {line!r}") from e

    def _send_action(self, action: int) -> None:
        """Send an action index to simc stdin."""
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("Process not running")

        self._process.stdin.write(f"{action}\n")
        self._process.stdin.flush()

    def _parse_state(
        self, msg: dict, is_first: bool = False
    ) -> tuple[dict, np.ndarray]:
        """
        Parse a state message from simc into observation dict and action mask.

        Args:
            msg: JSON message from simc containing state information.
            is_first: Whether this is the first state (used to capture initial TTD).

        Returns:
            Tuple of (observation_dict, action_mask)
        """
        # Capture initial TTD on first observation for normalization
        if is_first:
            self._initial_ttd = msg.get("ttd", self.fight_length)
            if self._initial_ttd <= 0:
                self._initial_ttd = self.fight_length

        # Get raw action data from simc
        raw_num_actions = msg["n"]
        raw_mask = msg["mask"]
        raw_cd_charges = msg["cd_charges_f"]

        # Validate that action count matches what we probed during init
        if raw_num_actions != self._raw_num_actions:
            raise RuntimeError(
                f"Action count mismatch: expected {self._raw_num_actions} but got "
                f"{raw_num_actions}. The simc profile may have changed."
            )

        # Build filtered mask and cd_charges
        filtered_mask = [raw_mask[i] for i in self._action_index_map]
        filtered_cd_charges = [raw_cd_charges[i] for i in self._action_index_map]

        # Compute normalized values internally using initial TTD and base GCD
        gcd_rem = msg.get("gcd_rem", 0.0)
        ttd = msg.get("ttd", self._initial_ttd)

        gcd_rem_norm = gcd_rem / BASE_GCD_SECONDS if BASE_GCD_SECONDS > 0 else 0.0
        ttd_norm = (
            ttd / self._initial_ttd
            if self._initial_ttd and self._initial_ttd > 0
            else 0.0
        )

        # Build observation vector
        obs_parts = [
            gcd_rem_norm,
            ttd_norm,
        ]
        obs_parts.extend(msg["resource_pct"])
        obs_parts.extend(filtered_cd_charges)

        # Add spec-specific buff observations (remains normalized, then stacks)
        buff_remains = msg.get("buff_remains", [0.0] * self._num_buffs)
        buff_stacks = msg.get("buff_stacks", [0.0] * self._num_buffs)
        obs_parts.extend(buff_remains)
        obs_parts.extend(buff_stacks)

        # Add spec-specific dot observations (remains normalized, then stacks)
        dot_remains = msg.get("dot_remains", [0.0] * self._num_dots)
        dot_stacks = msg.get("dot_stacks", [0.0] * self._num_dots)
        obs_parts.extend(dot_remains)
        obs_parts.extend(dot_stacks)

        obs = np.array(obs_parts, dtype=np.float32)
        mask = np.array(filtered_mask, dtype=np.int8)

        return {"obs": obs, "mask": mask}, mask

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[dict, dict]:
        """
        Reset the environment and start a new episode.

        Args:
            seed: Random seed (updates the seed for this and future episodes).
            options: Additional options (unused).

        Returns:
            Tuple of (observation, info_dict)
        """
        super().reset(seed=seed)

        if seed is not None:
            self.seed_value = seed

        # Reset initial TTD for new episode (used for normalization)
        self._initial_ttd = None

        self._start_process()
        self._total_reward = 0.0

        # Read first state message
        msg = self._read_message()

        if msg.get("type") == "done":
            # Episode ended immediately (shouldn't happen normally)
            raise RuntimeError("Episode ended immediately after reset")

        self._current_state = msg
        obs, mask = self._parse_state(msg, is_first=True)

        info = {
            "mask": mask,
            "action_labels": self._filtered_labels,
            "time": msg["t"],
            "initial_ttd": self._initial_ttd,
            "spec_id": self._spec_id,
            "buff_labels": self._buff_labels,
            "dot_labels": self._dot_labels,
        }

        return obs, info

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        """
        Execute one step in the environment.

        Args:
            action: Index of the action to take (must be legal per current mask).
                This is the filtered action index, not the raw simc index.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if self._process is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        # Translate filtered action index to raw simc action index
        if action < 0 or action >= len(self._action_index_map):
            raise ValueError(
                f"Invalid action {action}. Must be in range [0, {len(self._action_index_map)})"
            )
        raw_action = self._action_index_map[action]

        # Send raw action to simc
        self._send_action(raw_action)

        # Read response
        msg = self._read_message()

        if msg.get("type") == "done":
            # Episode ended
            total_damage = msg.get("total_damage", 0.0)
            if total_damage == 0.0:
                raise RuntimeError("Episode ended but total_damage is missing or zero.")
            fight_length = msg.get("fight_length", self.fight_length)

            # Final reward is the remaining damage delta
            if self.intermediate_rewards:
                reward = total_damage - self._total_reward
            else:
                reward = total_damage

            # Create terminal observation (zeros)
            if self._spaces_initialized:
                obs_dim = self.observation_space["obs"].shape[0]
                obs = {
                    "obs": np.zeros(obs_dim, dtype=np.float32),
                    "mask": np.zeros(self._num_actions, dtype=np.int8),
                }
            else:
                obs = {
                    "obs": np.zeros(128, dtype=np.float32),
                    "mask": np.zeros(64, dtype=np.int8),
                }

            info = {
                "mask": obs["mask"],
                "action_labels": self._filtered_labels,
                "total_damage": total_damage,
                "fight_length": fight_length,
                "episode_end": True,
            }

            self._close_process()
            return obs, reward, True, False, info

        # Normal step
        self._current_state = msg
        obs, mask = self._parse_state(msg)

        if self.intermediate_rewards:
            reward = msg.get("reward", 0.0)
        else:
            reward = 0.0
        self._total_reward += reward

        info = {
            "mask": mask,
            "action_labels": self._filtered_labels,
            "time": msg["t"],
            "chosen_label": (
                self._filtered_labels[action]
                if action < len(self._filtered_labels)
                else "unknown"
            ),
        }

        return obs, reward, False, False, info

    def close(self) -> None:
        """Clean up resources."""
        self._close_process()

    def action_masks(self) -> np.ndarray:
        """
        Return the current action mask (for compatibility with sb3-contrib MaskablePPO).

        Returns:
            Boolean array where True = legal action (using filtered action space).
        """
        if self._current_state is None:
            return np.ones(self._num_actions, dtype=bool)
        # Return filtered mask based on action index mapping
        raw_mask = self._current_state.get("mask", [])
        if len(raw_mask) == 0 or len(self._action_index_map) == 0:
            return np.ones(self._num_actions, dtype=bool)
        filtered_mask = [raw_mask[i] for i in self._action_index_map]
        return np.array(filtered_mask, dtype=bool)

    def get_action_label(self, action: int) -> str:
        """Get the human-readable label for a filtered action index."""
        if 0 <= action < len(self._filtered_labels):
            return self._filtered_labels[action]
        return f"action_{action}"


def make_simc_env(
    simc_path: str = "./simc",
    profile: Optional[str] = None,
    simc_args: Optional[list[str]] = None,
    action_blacklist: Optional[set[str]] = None,
    **kwargs,
) -> SimcEnv:
    """
    Factory function to create a SimcEnv with common defaults.

    Args:
        simc_path: Path to simc executable.
        profile: Path to .simc profile.
        simc_args: Additional simc arguments.
        action_blacklist: Set of action label names to exclude from the action space.
        **kwargs: Additional arguments passed to SimcEnv.

    Returns:
        Configured SimcEnv instance.
    """
    return SimcEnv(
        simc_path=simc_path,
        profile=profile,
        simc_args=simc_args,
        action_blacklist=action_blacklist,
        **kwargs,
    )


def make_vec_env(
    num_envs: int,
    simc_path: str = "./simc",
    profile: Optional[str] = None,
    simc_args: Optional[list[str]] = None,
    action_blacklist: Optional[set[str]] = None,
    **kwargs,
):
    """
    Create a vectorized environment with multiple simc subprocesses.

    This is the recommended way to train with parallel environments.
    Each subprocess runs its own simc instance.

    Args:
        num_envs: Number of parallel environments.
        simc_path: Path to simc executable.
        profile: Path to .simc profile.
        simc_args: Additional simc arguments.
        action_blacklist: Set of action label names to exclude from the action space.
        **kwargs: Additional arguments passed to SimcEnv.

    Returns:
        SubprocVecEnv with num_envs parallel SimcEnv instances.

    Example:
        >>> from sb3_contrib import MaskablePPO
        >>> envs = make_vec_env(8, simc_path="./simc", profile="Feral.simc",
        ...                     action_blacklist={"invoke_external_buff"})
        >>> model = MaskablePPO("MultiInputPolicy", envs, verbose=1)
        >>> model.learn(total_timesteps=100000)
    """
    try:
        from stable_baselines3.common.vec_env import SubprocVecEnv
    except ImportError:
        raise ImportError(
            "stable-baselines3 is required for vectorized environments. "
            "Install with: pip install stable-baselines3"
        )

    def make_env(seed: int):
        def _init():
            env = SimcEnv(
                simc_path=simc_path,
                profile=profile,
                simc_args=simc_args,
                action_blacklist=action_blacklist,
                seed=seed,
                **kwargs,
            )
            return env

        return _init

    return SubprocVecEnv([make_env(i) for i in range(num_envs)])


def run_demo(args):
    """Run a simple demo of the SimcEnv with random actions."""
    # Simple test/demo

    env = SimcEnv(
        simc_path=args.simc,
        profile=args.profile,
        action_blacklist=GLOBAL_ACTION_BLACKLIST,
    )

    for ep in range(args.episodes):
        print(f"\n=== Episode {ep + 1} ===")
        obs, info = env.reset()
        print(f"Action space: {env.action_space.n} actions")
        print(f"Actions: {info['action_labels']}")

        total_reward = 0.0
        step_count = 0

        while True:
            # Random policy with action masking
            mask = info["mask"]
            legal_actions = np.where(mask)[0]
            if len(legal_actions) == 0:
                print("No legal actions!")
                break

            action = np.random.choice(legal_actions)
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            step_count += 1

            if step_count <= 5 or step_count % 100 == 0:
                print(
                    f"  Step {step_count}: action={env.get_action_label(action)}, reward={reward:.2f}, obs={obs}, truncated={truncated}, info={info}"
                )

            if terminated or truncated:
                print(f"\nEpisode ended after {step_count} steps")
                print(f"Total damage: {info.get('total_damage', total_reward):.2f}")
                break

    env.close()
    print("\nDone!")


def run_single_thread_sb3(args):
    from sb3_contrib import MaskablePPO

    # Single environment
    env = SimcEnv(
        simc_path=args.simc,
        profile=args.profile,
        action_blacklist=GLOBAL_ACTION_BLACKLIST,
    )

    # Train with MaskablePPO (action masking)
    model = MaskablePPO("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=1000000)


def run_multi_thread_sb3(args):
    import os
    import pickle
    import time
    from pathlib import Path

    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import (
        BaseCallback,
        CallbackList,
        CheckpointCallback,
    )
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

    # === Configuration ===
    num_envs = 12
    total_timesteps = 100_000_000  # 100M steps for overnight training
    save_freq = 50_000  # Save checkpoint every 50k steps
    log_interval = 1  # Log every iteration

    # Output directory - use resume directory or create new with timestamp
    resume_dir = Path(args.resume) if args.resume else None
    if resume_dir:
        output_dir = resume_dir
        if not output_dir.exists():
            raise ValueError(f"Resume directory does not exist: {output_dir}")
        print(f"=== SimC RL Training (RESUMING) ===")
    else:
        run_name = f"simc_ppo_{time.strftime('%Y%m%d_%H%M%S')}"
        output_dir = Path("./training_runs") / run_name
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"=== SimC RL Training ===")

    log_dir = output_dir / "logs"
    model_dir = output_dir / "models"
    log_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Num envs: {num_envs}")
    print(f"Total timesteps: {total_timesteps:,}")

    def make_env(seed: int):
        def _init():
            env = SimcEnv(
                simc_path=args.simc,
                profile=args.profile,
                action_blacklist=GLOBAL_ACTION_BLACKLIST,
                seed=seed,
                intermediate_rewards=True,  # Enable intermediate rewards for learning signal
            )
            # Wrap with Monitor for episode stats (required for ep_rew_mean)
            env = Monitor(env)
            return env

        return _init

    print("Creating vectorized environments...")
    envs = SubprocVecEnv([make_env(i) for i in range(num_envs)])

    # Wrap with VecNormalize for reward normalization
    # This is critical - raw rewards range from 50k to 10M+ per step
    vec_normalize_path = model_dir / "vec_normalize.pkl"
    if resume_dir and vec_normalize_path.exists():
        print(f"Loading VecNormalize stats from {vec_normalize_path}")
        envs = VecNormalize.load(str(vec_normalize_path), envs)
        envs.training = True  # Ensure we continue updating stats
    else:
        envs = VecNormalize(
            envs,
            norm_obs=False,  # Don't normalize obs - already in [0,1] range
            norm_reward=True,  # Normalize rewards - critical for learning
            clip_reward=10.0,  # Clip to reasonable range
            gamma=0.99,  # Discount for reward normalization
        )

    # === Custom Callback for Best Model & Metrics ===
    class TrainingMetricsCallback(BaseCallback):
        """
        Custom callback that:
        1. Tracks episode rewards and saves metrics to CSV
        2. Saves the best model based on mean reward
        3. Keeps only the latest checkpoint to save space
        4. Saves VecNormalize stats alongside model checkpoints
        """

        def __init__(
            self,
            save_path: Path,
            vec_normalize_env: VecNormalize,
            check_freq: int = 1000,
            verbose: int = 1,
            resume: bool = False,
        ):
            super().__init__(verbose)
            self.save_path = save_path
            self.vec_normalize_env = vec_normalize_env
            self.check_freq = check_freq
            self.best_mean_reward = float("-inf")
            self.episode_rewards = []
            self.episode_lengths = []
            self.metrics_file = save_path / "training_metrics.csv"
            self._last_checkpoint_path = None

            # Load previous best reward if resuming
            if resume and self.metrics_file.exists():
                try:
                    import csv

                    with open(self.metrics_file, "r") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            if row.get("best_reward"):
                                self.best_mean_reward = float(row["best_reward"])
                    print(
                        f"Resuming with previous best reward: {self.best_mean_reward:.2f}"
                    )
                except Exception as e:
                    print(f"Warning: Could not load previous best reward: {e}")

            # Initialize or append to CSV file
            if not resume or not self.metrics_file.exists():
                with open(self.metrics_file, "w") as f:
                    f.write(
                        "timesteps,episodes,ep_rew_mean,ep_rew_std,ep_len_mean,best_reward\n"
                    )

        def _on_step(self) -> bool:
            # Collect episode info from monitor
            if self.locals.get("infos"):
                for info in self.locals["infos"]:
                    if "episode" in info:
                        self.episode_rewards.append(info["episode"]["r"])
                        self.episode_lengths.append(info["episode"]["l"])

            # Periodic logging and checkpointing
            if self.n_calls % self.check_freq == 0 and len(self.episode_rewards) > 0:
                mean_reward = np.mean(self.episode_rewards[-100:])  # Last 100 episodes
                std_reward = np.std(self.episode_rewards[-100:])
                mean_length = np.mean(self.episode_lengths[-100:])
                num_episodes = len(self.episode_rewards)

                # Log to CSV
                with open(self.metrics_file, "a") as f:
                    f.write(
                        f"{self.num_timesteps},{num_episodes},{mean_reward:.2f},"
                        f"{std_reward:.2f},{mean_length:.2f},{self.best_mean_reward:.2f}\n"
                    )

                # Save best model
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    best_path = self.save_path / "best_model"
                    self.model.save(best_path)
                    # Also save VecNormalize stats with best model
                    self.vec_normalize_env.save(
                        str(self.save_path / "vec_normalize.pkl")
                    )
                    if self.verbose > 0:
                        print(
                            f"  New best model! Mean reward: {mean_reward:.2f} "
                            f"(saved to {best_path})"
                        )

                # Save latest checkpoint (delete previous to save space)
                latest_path = self.save_path / "latest_model"
                self.model.save(latest_path)
                # Also save VecNormalize stats with latest
                self.vec_normalize_env.save(str(self.save_path / "vec_normalize.pkl"))

            return True

        def _on_training_end(self) -> None:
            # Final save
            final_path = self.save_path / "final_model"
            self.model.save(final_path)
            # Save VecNormalize stats
            self.vec_normalize_env.save(str(self.save_path / "vec_normalize.pkl"))
            print(f"\nTraining complete! Final model saved to {final_path}")
            print(f"Best mean reward achieved: {self.best_mean_reward:.2f}")
            print(f"Total episodes: {len(self.episode_rewards)}")
            print(f"Metrics saved to: {self.metrics_file}")

    # === Model Setup ===
    # Configure logger for TensorBoard + stdout + CSV
    logger = configure(str(log_dir), ["stdout", "csv", "tensorboard"])

    # Try to load existing model if resuming
    model = None
    if resume_dir:
        # Priority order: final > interrupted > latest
        model_candidates = [
            model_dir / "final_model.zip",
            model_dir / "interrupted_model.zip",
            model_dir / "latest_model.zip",
        ]
        for candidate in model_candidates:
            if candidate.exists():
                print(f"Loading model from {candidate}")
                model = MaskablePPO.load(
                    str(candidate),
                    env=envs,
                    tensorboard_log=str(log_dir),
                )
                model.set_logger(logger)
                print(f"Model loaded successfully!")
                break
        if model is None:
            print("Warning: No model found in resume directory, creating new model")

    if model is None:
        print("Initializing new MaskablePPO model...")
        model = MaskablePPO(
            "MultiInputPolicy",
            envs,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,  # Steps per env before update
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,  # Add entropy for exploration
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=str(log_dir),
        )
        model.set_logger(logger)

    # === Callbacks ===
    metrics_callback = TrainingMetricsCallback(
        save_path=model_dir,
        vec_normalize_env=envs,
        check_freq=5000,  # Check every 5k steps
        verbose=1,
        resume=resume_dir is not None,
    )

    # === Training ===
    print(f"\nStarting training for {total_timesteps:,} timesteps...")
    print("Press Ctrl+C to stop training early (model will be saved)\n")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=metrics_callback,
            log_interval=log_interval,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        interrupt_path = model_dir / "interrupted_model"
        model.save(interrupt_path)
        # Save VecNormalize stats
        envs.save(str(model_dir / "vec_normalize.pkl"))
        print(f"Model saved to {interrupt_path}")

    envs.close()
    print("\nDone!")


def run_evaluation(args):
    """
    Run evaluation of a trained RL model over many iterations.

    This runs simc with many iterations (default 2000) using the RL model
    for action selection, generating statistical output like a normal simc run.
    Produces HTML and text reports.

    Unlike training, this function directly manages the simc subprocess to
    handle multiple iterations in a single process run, allowing simc to
    aggregate statistics and generate proper reports.
    """
    import subprocess
    import time
    from pathlib import Path

    from sb3_contrib import MaskablePPO

    # === Configuration ===
    iterations = args.iterations if hasattr(args, "iterations") else 2000
    model_dir = Path(args.model_dir)

    if not model_dir.exists():
        raise ValueError(f"Model directory does not exist: {model_dir}")

    # Setup output paths
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir)
        if hasattr(args, "output_dir") and args.output_dir
        else model_dir / "evaluation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    html_output = output_dir / f"eval_{timestamp}.html"
    text_output = output_dir / f"eval_{timestamp}.txt"

    print(f"=== SimC RL Evaluation ===")
    print(f"Model directory: {model_dir}")
    print(f"Iterations: {iterations}")
    print(f"HTML output: {html_output}")
    print(f"Text output: {text_output}")

    # === Load Model ===
    model_candidates = [
        model_dir / "models" / "best_model.zip",
        model_dir / "models" / "final_model.zip",
        model_dir / "models" / "interrupted_model.zip",
        model_dir / "models" / "latest_model.zip",
    ]

    model_path = None
    for candidate in model_candidates:
        if candidate.exists():
            model_path = candidate
            break

    if model_path is None:
        raise ValueError(f"No model found in {model_dir / 'models'}")

    print(f"Loading model from {model_path}")

    # Create a probe env to get observation/action space dimensions
    probe_env = SimcEnv(
        simc_path=args.simc,
        profile=args.profile,
        action_blacklist=GLOBAL_ACTION_BLACKLIST,
        iterations=1,  # Just for probing
    )
    obs_space = probe_env.observation_space
    act_space = probe_env.action_space
    num_actions = probe_env._num_actions
    raw_num_actions = probe_env._raw_num_actions
    action_index_map = probe_env._action_index_map
    filtered_labels = probe_env._filtered_labels
    num_resources = probe_env._num_resources
    num_buffs = probe_env._num_buffs
    num_dots = probe_env._num_dots
    probe_env.close()

    print(f"Action space: {num_actions} actions (from {raw_num_actions} raw)")
    print(f"Action labels: {filtered_labels}")

    # Load the model
    # Note: VecNormalize was created with norm_obs=False during training,
    # so we don't need to load or apply any observation normalization.
    # The observations are already in a suitable range [0, ~10].
    model = MaskablePPO.load(str(model_path))
    print("Model loaded successfully!")

    # === Build simc command for evaluation ===
    cmd = [
        args.simc,
        args.profile,
        "rl_enable=1",
        "rl_stdio=1",
        "rl_trace=0",
        f"iterations={iterations}",
        f"html={html_output}",
        f"output={text_output}",
    ]

    print(f"\nSimC command: {' '.join(cmd)}")
    print(f"\nRunning {iterations} iterations with RL model...")
    print("=" * 60)

    # === Start simc subprocess ===
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    start_time = time.time()
    total_steps = 0
    iteration_count = 0
    iteration_damages = []
    non_json_lines = []

    def parse_state_to_obs(msg: dict) -> dict:
        """Parse simc message to observation dict compatible with model."""
        # Get raw action data
        raw_mask = msg["mask"]
        raw_cd_charges = msg["cd_charges_f"]

        # Build filtered mask and cd_charges using the same mapping as training
        filtered_mask = [raw_mask[i] for i in action_index_map]
        filtered_cd_charges = [raw_cd_charges[i] for i in action_index_map]

        # Compute normalized values
        gcd_rem = msg.get("gcd_rem", 0.0)
        ttd = msg.get("ttd", 300.0)
        initial_ttd = 300.0  # Assume default fight length

        gcd_rem_norm = gcd_rem / BASE_GCD_SECONDS if BASE_GCD_SECONDS > 0 else 0.0
        ttd_norm = ttd / initial_ttd if initial_ttd > 0 else 0.0

        # Build observation vector
        obs_parts = [gcd_rem_norm, ttd_norm]
        obs_parts.extend(msg["resource_pct"])
        obs_parts.extend(filtered_cd_charges)

        # Add spec-specific buff observations
        buff_remains = msg.get("buff_remains", [0.0] * num_buffs)
        buff_stacks = msg.get("buff_stacks", [0.0] * num_buffs)
        obs_parts.extend(buff_remains)
        obs_parts.extend(buff_stacks)

        # Add spec-specific dot observations
        dot_remains = msg.get("dot_remains", [0.0] * num_dots)
        dot_stacks = msg.get("dot_stacks", [0.0] * num_dots)
        obs_parts.extend(dot_remains)
        obs_parts.extend(dot_stacks)

        obs = np.array(obs_parts, dtype=np.float32)
        mask = np.array(filtered_mask, dtype=np.int8)

        return {"obs": obs, "mask": mask}

    try:
        while True:
            # Read line from simc
            line = process.stdout.readline()
            if not line:
                # Process ended
                break

            line = line.strip()
            if not line:
                continue

            # Try to parse as JSON
            if line.startswith("{"):
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    non_json_lines.append(line)
                    continue

                msg_type = msg.get("type")

                if msg_type == "done":
                    # End of one iteration
                    iteration_count += 1
                    iteration_damages.append(msg.get("total_damage", 0.0))

                    if iteration_count % 100 == 0:
                        elapsed = time.time() - start_time
                        avg_dmg = np.mean(iteration_damages[-100:])
                        print(
                            f"  Iteration {iteration_count}/{iterations} - "
                            f"Avg DPS (last 100): {avg_dmg:,.0f} - "
                            f"Elapsed: {elapsed:.1f}s"
                        )

                    # Don't break - continue to next iteration
                    continue

                # Regular step message - get action from model
                obs_dict = parse_state_to_obs(msg)

                # Prepare observation for model (needs to be in vectorized format)
                obs_array = obs_dict["obs"].reshape(1, -1)
                mask_array = obs_dict["mask"].reshape(1, -1)

                # Build observation dict in the format expected by MultiInputPolicy
                # Note: VecNormalize was created with norm_obs=False during training,
                # so we do NOT normalize observations here - they should match training exactly
                vec_obs = {"obs": obs_array, "mask": mask_array}

                # Get action from model (deterministic for evaluation)
                # CRITICAL: Must pass action_masks for MaskablePPO to respect legal actions
                action, _ = model.predict(
                    vec_obs, deterministic=True, action_masks=mask_array
                )
                action = int(action[0])

                # Map filtered action to raw simc action
                raw_action = action_index_map[action]

                # Send action to simc
                process.stdin.write(f"{raw_action}\n")
                process.stdin.flush()

                total_steps += 1

            else:
                # Non-JSON line - save for debugging
                non_json_lines.append(line)

    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Wait for process to complete
        try:
            process.stdin.close()
        except Exception:
            pass

        # Read any remaining stderr
        stderr_output = process.stderr.read() if process.stderr else ""

        process.wait(timeout=30.0)

    elapsed = time.time() - start_time

    # === Report Results ===
    print("=" * 60)
    print(f"\n=== Evaluation Complete ===")
    print(f"Iterations completed: {iteration_count}")
    print(f"Total decision steps: {total_steps:,}")
    print(f"Time elapsed: {elapsed:.1f}s")
    if elapsed > 0:
        print(f"Iterations/second: {iteration_count / elapsed:.1f}")
        print(f"Steps/second: {total_steps / elapsed:.1f}")

    if iteration_damages:
        print(f"\nDPS Statistics ({len(iteration_damages)} iterations):")
        print(f"  Mean:   {np.mean(iteration_damages):,.0f}")
        print(f"  Std:    {np.std(iteration_damages):,.0f}")
        print(f"  Min:    {np.min(iteration_damages):,.0f}")
        print(f"  Max:    {np.max(iteration_damages):,.0f}")
        print(f"  Median: {np.median(iteration_damages):,.0f}")

    print(f"\nOutput files:")
    if html_output.exists():
        print(f"  HTML report: {html_output} ({html_output.stat().st_size:,} bytes)")
    else:
        print(f"  HTML report: NOT CREATED")
    if text_output.exists():
        print(f"  Text report: {text_output} ({text_output.stat().st_size:,} bytes)")
    else:
        print(f"  Text report: NOT CREATED")

    # Show simc output for debugging
    if non_json_lines:
        print(f"\n--- SimC Output ({len(non_json_lines)} lines) ---")
        for line in non_json_lines[-50:]:  # Last 50 lines
            print(f"  {line}")

    if stderr_output.strip():
        print(f"\n--- SimC Stderr ---")
        print(stderr_output)

    print("\nDone!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="SimulationCraft RL Bridge - Train RL agents for WoW rotations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--simc", default="./simc", help="Path to simc executable")
    parser.add_argument("--profile", required=True, help="Path to .simc profile")
    parser.add_argument(
        "--episodes", type=int, default=1, help="Number of episodes (demo mode)"
    )
    parser.add_argument(
        "--mode",
        choices=["demo", "single", "multi", "eval"],
        default="multi",
        help="Mode: demo (random actions), single (1 env), multi (parallel training), eval (evaluate model)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to training run directory to resume (e.g., training_runs/simc_ppo_20251229_015900)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        dest="model_dir",
        help="Path to model directory for evaluation (e.g., training_runs/simc_ppo_20251229_015900)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=2000,
        help="Number of iterations for evaluation mode",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        dest="output_dir",
        help="Output directory for evaluation reports (defaults to model_dir/evaluation)",
    )
    args = parser.parse_args()

    if args.mode == "demo":
        run_demo(args)
    elif args.mode == "single":
        run_single_thread_sb3(args)
    elif args.mode == "eval":
        if not args.model_dir:
            parser.error("--model-dir is required for evaluation mode")
        run_evaluation(args)
    else:
        run_multi_thread_sb3(args)
