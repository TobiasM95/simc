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
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SimcEnv(gym.Env):
    """
    Gymnasium environment that wraps SimulationCraft with RL stdio bridge.

    The environment spawns a simc subprocess and communicates via JSON over
    stdin/stdout. Each step sends an action index and receives the next state.

    Observation Space:
        A dictionary containing:
        - "obs": Box of continuous features (time, resources, cooldowns, etc.)
        - "mask": MultiBinary action mask (1 = legal, 0 = illegal)

    Action Space:
        Discrete(n) where n is the number of possible actions.
        Actions are masked - only actions with mask[i]=1 are valid.
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
        """
        super().__init__()

        self.simc_path = simc_path
        self.profile = profile
        self.simc_args = simc_args or []
        self.fight_length = fight_length
        self.iterations = iterations
        self.seed_value = seed
        self.intermediate_rewards = intermediate_rewards

        self._process: Optional[subprocess.Popen] = None
        self._action_labels: list[str] = []
        self._num_actions: int = 0
        self._num_resources = 18  # RESOURCE_MAX in simc

        # Observation space: continuous features
        # [time_remaining_norm, gcd_remaining_norm, ttd_norm, 18 resources, n*3 action features]
        # We'll set the actual size after first step when we know num_actions
        self._obs_dim = (
            3 + self._num_resources
        )  # Base dimensions (before action features)

        # Placeholder spaces - will be updated after first message
        self.observation_space = spaces.Dict(
            {
                "obs": spaces.Box(low=0.0, high=1.0, shape=(128,), dtype=np.float32),
                "mask": spaces.MultiBinary(64),
            }
        )
        self.action_space = spaces.Discrete(64)

        self._spaces_initialized = False
        self._current_state: Optional[dict] = None
        self._total_reward = 0.0

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

    def _start_process(self) -> None:
        """Start the simc subprocess."""
        if self._process is not None:
            self._close_process()

        cmd = self._build_command()
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line-buffered for responsive communication
        )

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

    def _read_message(self) -> dict:
        """Read a JSON message from simc stdout."""
        if self._process is None or self._process.stdout is None:
            raise RuntimeError("Process not running")

        while True:
            line = self._process.stdout.readline()
            if not line:
                # Process ended unexpectedly
                stderr = self._process.stderr.read() if self._process.stderr else ""
                raise RuntimeError(f"simc process ended unexpectedly. stderr: {stderr}")
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

    def _parse_state(self, msg: dict) -> tuple[dict, np.ndarray]:
        """
        Parse a state message from simc into observation dict and action mask.

        Returns:
            Tuple of (observation_dict, action_mask)
        """
        # Update action space if needed
        num_actions = msg["n"]
        if not self._spaces_initialized or num_actions != self._num_actions:
            self._num_actions = num_actions
            self._action_labels = msg.get(
                "labels", [f"action_{i}" for i in range(num_actions)]
            )

            # Observation: base features + per-action features (cd_rem_n, cd_charges_f)
            obs_dim = 3 + self._num_resources + num_actions * 2
            self.observation_space = spaces.Dict(
                {
                    "obs": spaces.Box(
                        low=-1.0, high=10.0, shape=(obs_dim,), dtype=np.float32
                    ),
                    "mask": spaces.MultiBinary(num_actions),
                }
            )
            self.action_space = spaces.Discrete(num_actions)
            self._spaces_initialized = True

        # Build observation vector
        obs_parts = [
            msg["time_rem_n"],
            msg["gcd_rem_n"],
            msg["ttd_n"],
        ]
        obs_parts.extend(msg["resource_pct"])
        obs_parts.extend(msg["cd_rem_n"])
        obs_parts.extend(msg["cd_charges_f"])

        obs = np.array(obs_parts, dtype=np.float32)
        mask = np.array(msg["mask"], dtype=np.int8)

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

        self._start_process()
        self._total_reward = 0.0

        # Read first state message
        msg = self._read_message()

        if msg.get("type") == "done":
            # Episode ended immediately (shouldn't happen normally)
            raise RuntimeError("Episode ended immediately after reset")

        self._current_state = msg
        obs, mask = self._parse_state(msg)

        info = {
            "mask": mask,
            "action_labels": self._action_labels,
            "time": msg["t"],
            "fight_length": msg["fight_len"],
        }

        return obs, info

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        """
        Execute one step in the environment.

        Args:
            action: Index of the action to take (must be legal per current mask).

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if self._process is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        # Send action to simc
        self._send_action(action)

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
                "action_labels": self._action_labels,
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
            "action_labels": self._action_labels,
            "time": msg["t"],
            "chosen_label": (
                self._action_labels[action]
                if action < len(self._action_labels)
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
            Boolean array where True = legal action.
        """
        if self._current_state is None:
            return np.ones(self._num_actions, dtype=bool)
        return np.array(self._current_state.get("mask", []), dtype=bool)

    def get_action_label(self, action: int) -> str:
        """Get the human-readable label for an action index."""
        if 0 <= action < len(self._action_labels):
            return self._action_labels[action]
        return f"action_{action}"


def make_simc_env(
    simc_path: str = "./simc",
    profile: Optional[str] = None,
    simc_args: Optional[list[str]] = None,
    **kwargs,
) -> SimcEnv:
    """
    Factory function to create a SimcEnv with common defaults.

    Args:
        simc_path: Path to simc executable.
        profile: Path to .simc profile.
        simc_args: Additional simc arguments.
        **kwargs: Additional arguments passed to SimcEnv.

    Returns:
        Configured SimcEnv instance.
    """
    return SimcEnv(
        simc_path=simc_path,
        profile=profile,
        simc_args=simc_args,
        **kwargs,
    )


def make_vec_env(
    num_envs: int,
    simc_path: str = "./simc",
    profile: Optional[str] = None,
    simc_args: Optional[list[str]] = None,
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
        **kwargs: Additional arguments passed to SimcEnv.

    Returns:
        SubprocVecEnv with num_envs parallel SimcEnv instances.

    Example:
        >>> from sb3_contrib import MaskablePPO
        >>> envs = make_vec_env(8, simc_path="./simc", profile="Feral.simc")
        >>> model = MaskablePPO("MlpPolicy", envs, verbose=1)
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
                seed=seed,
                **kwargs,
            )
            return env

        return _init

    return SubprocVecEnv([make_env(i) for i in range(num_envs)])


if __name__ == "__main__":
    # Simple test/demo
    import argparse

    parser = argparse.ArgumentParser(description="Test SimulationCraft RL Bridge")
    parser.add_argument("--simc", default="./simc", help="Path to simc executable")
    parser.add_argument("--profile", required=True, help="Path to .simc profile")
    parser.add_argument(
        "--episodes", type=int, default=1, help="Number of episodes to run"
    )
    args = parser.parse_args()

    env = SimcEnv(simc_path=args.simc, profile=args.profile)

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
                    f"  Step {step_count}: action={env.get_action_label(action)}, reward={reward:.2f}"
                )

            if terminated or truncated:
                print(f"\nEpisode ended after {step_count} steps")
                print(f"Total damage: {info.get('total_damage', total_reward):.2f}")
                break

    env.close()
    print("\nDone!")
