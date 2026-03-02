
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

BASE_GCD_SECONDS = 1.5

GLOBAL_ACTION_BLACKLIST = {
    "invoke_external_buff",
    "snapshot_stats",
    "cancel_buff",
    "use_item_arazs_ritual_forge",
    "do_treacherous_transmitter_task",
    "run_action_list",
}

NOOP_LABELS = {
    "wait_0.1",
    "wait_0.2",
    "wait_0.5",
    "wait_1",
    "wait_1.0",
    "wait_1.5",
    "pass",
}


class EpisodeSeedManager:
    """Deterministic seed stream per worker, if base_seed is set."""

    def __init__(self, base_seed: Optional[int], worker_id: int):
        if base_seed is None:
            self._rng = np.random.default_rng()
        else:
            seed = int(base_seed) + int(worker_id) * 1_000_003
            self._rng = np.random.default_rng(seed)

    def next_seed(self) -> int:
        return int(self._rng.integers(1, 2_147_483_647))


@dataclass
class ObsConfig:
    num_resources: int
    num_actions: int
    raw_num_actions: int
    num_buffs: int
    num_dots: int
    action_index_map: list[int]
    filtered_labels: list[str]
    noop_filtered_indices: list[int]


class SimcEnvV2(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        simc_path: str = "./simc",
        profile: Optional[str] = None,
        simc_args: Optional[list[str]] = None,
        fight_length: float = 300.0,
        iterations: int = 1,
        seed: Optional[int] = None,
        worker_id: int = 0,
        intermediate_rewards: bool = True,
        action_blacklist: Optional[set[str]] = None,
        show_simc_stderr: bool = False,
        no_op_strategy: str = "mask_when_real",
    ):
        super().__init__()
        self.simc_path = simc_path
        self.profile = profile
        self.simc_args = simc_args or []
        self.fight_length = fight_length
        self.iterations = iterations
        self.intermediate_rewards = intermediate_rewards
        self.show_simc_stderr = show_simc_stderr
        self._action_blacklist = action_blacklist or set()
        self._no_op_strategy = no_op_strategy

        self._process: Optional[subprocess.Popen] = None
        self._current_state: Optional[dict[str, Any]] = None
        self._stderr_buffer: list[str] = []
        self._stderr_lock = threading.Lock()
        self._stderr_thread: Optional[threading.Thread] = None

        self._seed_manager = EpisodeSeedManager(seed, worker_id)
        self._current_simc_seed: Optional[int] = None

        self._config: Optional[ObsConfig] = None
        self._spec_id: int = 0
        self._buff_labels: list[str] = []
        self._dot_labels: list[str] = []
        self._spaces_initialized = False
        self._initial_ttd: Optional[float] = None
        self._total_reward = 0.0

        self._probe_action_space()

    def _build_command(self) -> list[str]:
        cmd = [self.simc_path]
        if self.profile:
            cmd.append(self.profile)

        cmd.extend(
            [
                "rl_enable=1",
                "rl_stdio=1",
                "rl_trace=0",
                f"iterations={self.iterations}",
                f"max_time={self.fight_length}",
            ]
        )
        if self._current_simc_seed is not None:
            cmd.append(f"seed={self._current_simc_seed}")

        cmd.extend(self.simc_args)
        return cmd

    def _probe_action_space(self) -> None:
        self._current_simc_seed = self._seed_manager.next_seed()
        self._start_process()
        try:
            msg = self._read_message()
            if msg.get("type") == "done":
                raise RuntimeError("Episode ended immediately during probe")

            resource_pct = msg.get("resource_pct", [])
            if not isinstance(resource_pct, list) or len(resource_pct) == 0:
                raise RuntimeError("Probe message missing valid resource_pct array.")

            raw_num_actions = int(msg["n"])
            raw_labels = msg.get("labels", [f"action_{i}" for i in range(raw_num_actions)])

            action_index_map: list[int] = []
            filtered_labels: list[str] = []
            noop_filtered_indices: list[int] = []
            for i, label in enumerate(raw_labels):
                if label in self._action_blacklist:
                    continue
                filtered_idx = len(filtered_labels)
                action_index_map.append(i)
                filtered_labels.append(label)
                if label in NOOP_LABELS:
                    noop_filtered_indices.append(filtered_idx)

            num_resources = len(resource_pct)
            num_actions = len(filtered_labels)
            num_buffs = len(msg.get("buff_labels", []))
            num_dots = len(msg.get("dot_labels", []))

            self._config = ObsConfig(
                num_resources=num_resources,
                num_actions=num_actions,
                raw_num_actions=raw_num_actions,
                num_buffs=num_buffs,
                num_dots=num_dots,
                action_index_map=action_index_map,
                filtered_labels=filtered_labels,
                noop_filtered_indices=noop_filtered_indices,
            )

            self._spec_id = int(msg.get("spec_id", 0))
            self._buff_labels = msg.get("buff_labels", [])
            self._dot_labels = msg.get("dot_labels", [])

            # + cd_rem_n + cd_charges_f
            obs_dim = (
                2
                + num_resources
                + num_actions
                + num_actions
                + 2 * num_buffs
                + 2 * num_dots
            )
            self.observation_space = spaces.Dict(
                {
                    "obs": spaces.Box(
                        low=-1.0, high=100.0, shape=(obs_dim,), dtype=np.float32
                    ),
                    "mask": spaces.MultiBinary(num_actions),
                }
            )
            self.action_space = spaces.Discrete(num_actions)
            self._spaces_initialized = True
        finally:
            self._close_process()

    def _stderr_reader(self) -> None:
        try:
            if self._process and self._process.stderr:
                for line in self._process.stderr:
                    with self._stderr_lock:
                        self._stderr_buffer.append(line)
                        if self.show_simc_stderr:
                            print(f"[simc stderr] {line}", end="", file=sys.stderr)
        except Exception:
            pass

    def _get_stderr_output(self) -> str:
        with self._stderr_lock:
            return "".join(self._stderr_buffer)

    def _start_process(self) -> None:
        if self._process is not None:
            self._close_process()

        with self._stderr_lock:
            self._stderr_buffer.clear()

        cmd = self._build_command()
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._stderr_thread = threading.Thread(target=self._stderr_reader, daemon=True)
        self._stderr_thread.start()

    def _close_process(self) -> None:
        if self._process is None:
            return
        try:
            if self._process.stdin:
                self._process.stdin.close()
            if self._process.stdout:
                self._process.stdout.close()
            if self._process.stderr:
                self._process.stderr.close()
            self._process.terminate()
            self._process.wait(timeout=5.0)
        except Exception:
            self._process.kill()
        finally:
            self._process = None
            if self._stderr_thread and self._stderr_thread.is_alive():
                self._stderr_thread.join(timeout=1.0)
            self._stderr_thread = None

    def _read_message(self) -> dict[str, Any]:
        if self._process is None or self._process.stdout is None:
            raise RuntimeError("Process not running")

        while True:
            line = self._process.stdout.readline()
            if not line:
                exit_code = self._process.poll()
                time.sleep(0.1)
                stderr = self._get_stderr_output()
                raise RuntimeError(
                    f"simc process ended unexpectedly (exit code={exit_code}). stderr:\n{stderr if stderr else '(empty)'}"
                )
            if not line.lstrip().startswith("{"):
                continue
            try:
                return json.loads(line.strip())
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Failed to parse JSON from simc: {line!r}") from e

    def _send_action(self, action: int) -> None:
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("Process not running")
        self._process.stdin.write(f"{action}\n")
        self._process.stdin.flush()

    def _apply_noop_mask_policy(
        self, filtered_mask: list[int] | np.ndarray
    ) -> np.ndarray:
        if self._config is None:
            return np.array(filtered_mask, dtype=np.int8)

        mask = np.array(filtered_mask, dtype=np.int8)
        if self._no_op_strategy != "mask_when_real":
            return mask

        if len(self._config.noop_filtered_indices) == 0:
            return mask

        real_legal = False
        noop_index_set = set(self._config.noop_filtered_indices)
        for i, value in enumerate(mask):
            if i not in noop_index_set and int(value) == 1:
                real_legal = True
                break

        if real_legal:
            for idx in self._config.noop_filtered_indices:
                mask[idx] = 0

        if int(mask.sum()) == 0:
            for idx in self._config.noop_filtered_indices:
                if idx < len(mask):
                    mask[idx] = 1

        return mask

    def _parse_state(
        self, msg: dict[str, Any], is_first: bool = False
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        if self._config is None:
            raise RuntimeError("Environment config is not initialized")

        if is_first:
            self._initial_ttd = msg.get("ttd", self.fight_length)
            if self._initial_ttd is None or self._initial_ttd <= 0:
                self._initial_ttd = self.fight_length

        raw_num_actions = int(msg["n"])
        if raw_num_actions != self._config.raw_num_actions:
            raise RuntimeError(
                f"Action count mismatch: expected {self._config.raw_num_actions}, got {raw_num_actions}"
            )

        raw_mask = msg["mask"]
        raw_cd_charges = msg["cd_charges_f"]
        raw_cd_rem_n = msg.get("cd_rem_n", [0.0] * raw_num_actions)

        filtered_mask = [raw_mask[i] for i in self._config.action_index_map]
        filtered_cd_charges = [raw_cd_charges[i] for i in self._config.action_index_map]
        filtered_cd_rem_n = [raw_cd_rem_n[i] for i in self._config.action_index_map]

        mask = self._apply_noop_mask_policy(filtered_mask)

        gcd_rem = msg.get("gcd_rem", 0.0)
        ttd = msg.get("ttd", self._initial_ttd)
        gcd_rem_norm = gcd_rem / BASE_GCD_SECONDS if BASE_GCD_SECONDS > 0 else 0.0
        ttd_norm = (
            ttd / self._initial_ttd
            if self._initial_ttd is not None and self._initial_ttd > 0
            else 0.0
        )

        obs_parts: list[float] = [gcd_rem_norm, ttd_norm]

        resource_pct = msg.get("resource_pct", [])
        if len(resource_pct) < self._config.num_resources:
            resource_pct = resource_pct + [0.0] * (
                self._config.num_resources - len(resource_pct)
            )
        elif len(resource_pct) > self._config.num_resources:
            resource_pct = resource_pct[: self._config.num_resources]
        obs_parts.extend(resource_pct)
        obs_parts.extend(filtered_cd_rem_n)
        obs_parts.extend(filtered_cd_charges)

        buff_remains = msg.get("buff_remains", [0.0] * self._config.num_buffs)
        buff_stacks = msg.get("buff_stacks", [0.0] * self._config.num_buffs)
        dot_remains = msg.get("dot_remains", [0.0] * self._config.num_dots)
        dot_stacks = msg.get("dot_stacks", [0.0] * self._config.num_dots)

        def fit(values: list[float], n: int) -> list[float]:
            if len(values) < n:
                return values + [0.0] * (n - len(values))
            if len(values) > n:
                return values[:n]
            return values

        obs_parts.extend(fit(buff_remains, self._config.num_buffs))
        obs_parts.extend(fit(buff_stacks, self._config.num_buffs))
        obs_parts.extend(fit(dot_remains, self._config.num_dots))
        obs_parts.extend(fit(dot_stacks, self._config.num_dots))

        obs = np.array(obs_parts, dtype=np.float32)
        expected = self.observation_space["obs"].shape[0]
        if obs.shape[0] != expected:
            raise RuntimeError(
                f"Observation size mismatch: expected {expected}, got {obs.shape[0]}"
            )

        return {"obs": obs, "mask": mask}, mask

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        if self._config is None:
            raise RuntimeError("Environment config is not initialized")

        # Respect Gym reset(seed=...) without collapsing diversity to one static seed.
        if seed is not None:
            self._seed_manager = EpisodeSeedManager(seed, worker_id=0)

        self._initial_ttd = None
        self._current_simc_seed = self._seed_manager.next_seed()
        self._start_process()
        self._total_reward = 0.0

        msg = self._read_message()
        if msg.get("type") == "done":
            raise RuntimeError("Episode ended immediately after reset")

        self._current_state = msg
        obs, mask = self._parse_state(msg, is_first=True)
        info = {
            "mask": mask,
            "action_labels": self._config.filtered_labels,
            "time": msg["t"],
            "initial_ttd": self._initial_ttd,
            "spec_id": self._spec_id,
            "buff_labels": self._buff_labels,
            "dot_labels": self._dot_labels,
            "simc_seed": self._current_simc_seed,
        }
        return obs, info

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self._process is None or self._config is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        if action < 0 or action >= len(self._config.action_index_map):
            raise ValueError(f"Invalid action {action}")

        raw_action = self._config.action_index_map[action]
        self._send_action(raw_action)
        msg = self._read_message()

        if msg.get("type") == "done":
            total_damage = float(msg.get("total_damage", 0.0))
            if total_damage == 0.0:
                raise RuntimeError("Episode ended but total_damage is missing or zero")
            fight_length = float(msg.get("fight_length", self.fight_length))

            reward = (
                total_damage - self._total_reward
                if self.intermediate_rewards
                else total_damage
            )

            obs_dim = self.observation_space["obs"].shape[0]
            obs = {
                "obs": np.zeros(obs_dim, dtype=np.float32),
                "mask": np.zeros(self._config.num_actions, dtype=np.int8),
            }
            info = {
                "mask": obs["mask"],
                "action_labels": self._config.filtered_labels,
                "total_damage": total_damage,
                "fight_length": fight_length,
                "episode_end": True,
                "simc_seed": self._current_simc_seed,
            }
            self._close_process()
            return obs, reward, True, False, info

        self._current_state = msg
        obs, mask = self._parse_state(msg)
        reward = float(msg.get("reward", 0.0)) if self.intermediate_rewards else 0.0
        self._total_reward += reward

        info = {
            "mask": mask,
            "action_labels": self._config.filtered_labels,
            "time": msg["t"],
            "chosen_label": (
                self._config.filtered_labels[action]
                if action < len(self._config.filtered_labels)
                else "unknown"
            ),
            "simc_seed": self._current_simc_seed,
        }
        return obs, reward, False, False, info

    def action_masks(self) -> np.ndarray:
        if self._config is None:
            return np.array([], dtype=bool)
        if self._current_state is None:
            return np.ones(self._config.num_actions, dtype=bool)

        raw_mask = self._current_state.get("mask", [])
        if len(raw_mask) == 0:
            return np.ones(self._config.num_actions, dtype=bool)

        filtered_mask = [raw_mask[i] for i in self._config.action_index_map]
        return self._apply_noop_mask_policy(filtered_mask).astype(bool)

    def get_action_label(self, action: int) -> str:
        if self._config and 0 <= action < len(self._config.filtered_labels):
            return self._config.filtered_labels[action]
        return f"action_{action}"

    def close(self) -> None:
        self._close_process()


def make_vec_env_v2(
    num_envs: int,
    simc_path: str,
    profile: str,
    action_blacklist: set[str],
    base_seed: Optional[int],
    show_simc_stderr: bool,
    no_op_strategy: str,
    fight_length: float,
):
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import SubprocVecEnv

    def make_env(worker_id: int):
        def _init():
            env = SimcEnvV2(
                simc_path=simc_path,
                profile=profile,
                action_blacklist=action_blacklist,
                seed=base_seed,
                worker_id=worker_id,
                intermediate_rewards=True,
                show_simc_stderr=show_simc_stderr,
                no_op_strategy=no_op_strategy,
                fight_length=fight_length,
            )
            return Monitor(env)

        return _init

    return SubprocVecEnv([make_env(i) for i in range(num_envs)])


class LinearSchedule:
    def __init__(self, initial_value: float, final_value: float):
        self.initial_value = initial_value
        self.final_value = final_value

    def __call__(self, progress_remaining: float) -> float:
        progress = 1.0 - progress_remaining
        return self.initial_value + progress * (self.final_value - self.initial_value)


class TrainingCallback:
    def __init__(
        self,
        model_dir: Path,
        vec_normalize_env,
        eval_env: SimcEnvV2,
        eval_freq: int,
        eval_episodes: int,
        save_freq: int,
        train_log_freq: int,
        resume: bool,
        verbose: int = 1,
    ):
        from stable_baselines3.common.callbacks import BaseCallback

        class _Callback(BaseCallback):
            def __init__(self, outer: "TrainingCallback"):
                super().__init__(verbose=outer.verbose)
                self.outer = outer

            def _on_step(self) -> bool:
                return self.outer.on_step(self)

            def _on_training_end(self) -> None:
                self.outer.on_end(self)

        self.verbose = verbose
        self.model_dir = model_dir
        self.vec_normalize_env = vec_normalize_env
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.save_freq = save_freq
        self.train_log_freq = train_log_freq
        self.best_eval = float("-inf")
        self._episode_rewards: list[float] = []
        self._episode_lengths: list[float] = []
        self._csv_train = model_dir / "training_metrics.csv"
        self._csv_eval = model_dir / "eval_metrics.csv"

        if resume and self._csv_eval.exists():
            try:
                rows = list(csv.DictReader(self._csv_eval.open("r", newline="")))
                if rows:
                    self.best_eval = float(rows[-1]["best_eval_damage"])
            except Exception:
                pass

        if not resume or not self._csv_train.exists():
            with self._csv_train.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timesteps",
                        "episodes",
                        "ep_rew_mean",
                        "ep_rew_std",
                        "ep_len_mean",
                        "best_eval_damage",
                    ]
                )

        if not resume or not self._csv_eval.exists():
            with self._csv_eval.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timesteps",
                        "eval_mean_damage",
                        "eval_std_damage",
                        "eval_episodes",
                        "best_eval_damage",
                    ]
                )

        self.callback = _Callback(self)

    def _run_eval(self, cb) -> tuple[float, float]:
        damages: list[float] = []
        for _ in range(self.eval_episodes):
            obs, info = self.eval_env.reset()
            done = False
            while not done:
                mask = info["mask"]
                action, _ = cb.model.predict(obs, deterministic=True, action_masks=mask)
                action = int(action)
                obs, _, term, trunc, info = self.eval_env.step(action)
                done = bool(term or trunc)
                if done:
                    damages.append(float(info.get("total_damage", 0.0)))

        return float(np.mean(damages)), float(np.std(damages))

    def on_step(self, cb) -> bool:
        infos = cb.locals.get("infos")
        if infos:
            for info in infos:
                ep = info.get("episode")
                if ep:
                    self._episode_rewards.append(float(ep["r"]))
                    self._episode_lengths.append(float(ep["l"]))

        if cb.n_calls % self.train_log_freq == 0 and self._episode_rewards:
            mean_reward = float(np.mean(self._episode_rewards[-100:]))
            std_reward = float(np.std(self._episode_rewards[-100:]))
            mean_len = float(np.mean(self._episode_lengths[-100:]))
            with self._csv_train.open("a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        cb.num_timesteps,
                        len(self._episode_rewards),
                        f"{mean_reward:.2f}",
                        f"{std_reward:.2f}",
                        f"{mean_len:.2f}",
                        f"{self.best_eval:.2f}",
                    ]
                )

        if cb.n_calls % self.save_freq == 0:
            cb.model.save(self.model_dir / "latest_model")
            self.vec_normalize_env.save(str(self.model_dir / "vec_normalize.pkl"))

        if cb.n_calls % self.eval_freq == 0:
            mean_damage, std_damage = self._run_eval(cb)
            with self._csv_eval.open("a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        cb.num_timesteps,
                        f"{mean_damage:.2f}",
                        f"{std_damage:.2f}",
                        self.eval_episodes,
                        f"{max(self.best_eval, mean_damage):.2f}",
                    ]
                )

            if self.verbose:
                print(
                    f"[eval] timesteps={cb.num_timesteps:,} mean_damage={mean_damage:,.0f} std={std_damage:,.0f}"
                )

            if mean_damage > self.best_eval:
                self.best_eval = mean_damage
                cb.model.save(self.model_dir / "best_eval_model")
                self.vec_normalize_env.save(str(self.model_dir / "vec_normalize.pkl"))
                if self.verbose:
                    print(f"[eval] New best checkpoint at {mean_damage:,.0f} damage")

        return True

    def on_end(self, cb) -> None:
        # Always run one final held-out eval so end-of-run updates are not missed.
        mean_damage, std_damage = self._run_eval(cb)
        with self._csv_eval.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    cb.num_timesteps,
                    f"{mean_damage:.2f}",
                    f"{std_damage:.2f}",
                    self.eval_episodes,
                    f"{max(self.best_eval, mean_damage):.2f}",
                ]
            )
        if self.verbose:
            print(
                f"[eval-final] timesteps={cb.num_timesteps:,} mean_damage={mean_damage:,.0f} std={std_damage:,.0f}"
            )

        if mean_damage > self.best_eval:
            self.best_eval = mean_damage
            cb.model.save(self.model_dir / "best_eval_model")
            self.vec_normalize_env.save(str(self.model_dir / "vec_normalize.pkl"))
            if self.verbose:
                print(
                    f"[eval-final] New best checkpoint at {mean_damage:,.0f} damage"
                )

        cb.model.save(self.model_dir / "final_model")
        self.vec_normalize_env.save(str(self.model_dir / "vec_normalize.pkl"))
        if self.verbose:
            print(f"Training complete. Best held-out mean damage: {self.best_eval:,.0f}")


@dataclass
class EvalProbe:
    config: ObsConfig


class EvalEncoder:
    def __init__(self, probe: EvalProbe, no_op_strategy: str):
        self.probe = probe
        self.no_op_strategy = no_op_strategy
        self.initial_ttd: Optional[float] = None

    def on_new_iteration(self) -> None:
        self.initial_ttd = None

    def _apply_noop_mask(self, filtered_mask: list[int]) -> np.ndarray:
        mask = np.array(filtered_mask, dtype=np.int8)
        if self.no_op_strategy != "mask_when_real":
            return mask

        noop_set = set(self.probe.config.noop_filtered_indices)
        real_legal = any((i not in noop_set and int(v) == 1) for i, v in enumerate(mask))
        if real_legal:
            for idx in self.probe.config.noop_filtered_indices:
                mask[idx] = 0

        if int(mask.sum()) == 0:
            for idx in self.probe.config.noop_filtered_indices:
                if idx < len(mask):
                    mask[idx] = 1
        return mask

    def encode(self, msg: dict[str, Any]) -> tuple[dict[str, np.ndarray], np.ndarray]:
        if self.initial_ttd is None:
            t0 = float(msg.get("ttd", 0.0))
            self.initial_ttd = t0 if t0 > 0 else 300.0

        cfg = self.probe.config
        raw_mask = msg["mask"]
        raw_cd_charges = msg["cd_charges_f"]
        raw_cd_rem_n = msg.get("cd_rem_n", [0.0] * cfg.raw_num_actions)

        filtered_mask = [raw_mask[i] for i in cfg.action_index_map]
        filtered_cd_charges = [raw_cd_charges[i] for i in cfg.action_index_map]
        filtered_cd_rem_n = [raw_cd_rem_n[i] for i in cfg.action_index_map]
        mask = self._apply_noop_mask(filtered_mask)

        gcd_rem = float(msg.get("gcd_rem", 0.0))
        ttd = float(msg.get("ttd", self.initial_ttd))
        gcd_rem_norm = gcd_rem / BASE_GCD_SECONDS if BASE_GCD_SECONDS > 0 else 0.0
        ttd_norm = ttd / self.initial_ttd if self.initial_ttd > 0 else 0.0

        obs_parts: list[float] = [gcd_rem_norm, ttd_norm]

        resource_pct = msg.get("resource_pct", [])
        if len(resource_pct) < cfg.num_resources:
            resource_pct = resource_pct + [0.0] * (cfg.num_resources - len(resource_pct))
        elif len(resource_pct) > cfg.num_resources:
            resource_pct = resource_pct[: cfg.num_resources]
        obs_parts.extend(resource_pct)
        obs_parts.extend(filtered_cd_rem_n)
        obs_parts.extend(filtered_cd_charges)

        def fit(values: list[float], n: int) -> list[float]:
            if len(values) < n:
                return values + [0.0] * (n - len(values))
            if len(values) > n:
                return values[:n]
            return values

        obs_parts.extend(fit(msg.get("buff_remains", []), cfg.num_buffs))
        obs_parts.extend(fit(msg.get("buff_stacks", []), cfg.num_buffs))
        obs_parts.extend(fit(msg.get("dot_remains", []), cfg.num_dots))
        obs_parts.extend(fit(msg.get("dot_stacks", []), cfg.num_dots))

        obs = np.array(obs_parts, dtype=np.float32)
        return {"obs": obs, "mask": mask}, mask


def load_model_for_eval(model_dir: Path) -> Path:
    candidates = [
        model_dir / "models" / "best_eval_model.zip",
        model_dir / "models" / "best_model.zip",
        model_dir / "models" / "final_model.zip",
        model_dir / "models" / "interrupted_model.zip",
        model_dir / "models" / "latest_model.zip",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise ValueError(f"No model checkpoint found in {model_dir / 'models'}")


def run_train(args: argparse.Namespace) -> None:
    import torch
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.vec_env import VecNormalize

    action_blacklist = set() if args.no_blacklist else GLOBAL_ACTION_BLACKLIST

    resume_dir = Path(args.resume) if args.resume else None
    if resume_dir:
        output_dir = resume_dir
        if not output_dir.exists():
            raise ValueError(f"Resume directory does not exist: {output_dir}")
    else:
        run_name = f"simc_ppo_v2_{time.strftime('%Y%m%d_%H%M%S')}"
        output_dir = Path("./training_runs") / run_name
        output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = output_dir / "logs"
    model_dir = output_dir / "models"
    log_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Num envs: {args.num_envs}")
    print(f"Total timesteps: {args.total_timesteps:,}")

    envs = make_vec_env_v2(
        num_envs=args.num_envs,
        simc_path=args.simc,
        profile=args.profile,
        action_blacklist=action_blacklist,
        base_seed=args.seed,
        show_simc_stderr=args.show_simc_stderr,
        no_op_strategy=args.no_op_strategy,
        fight_length=args.fight_length,
    )

    vec_norm_path = model_dir / "vec_normalize.pkl"
    if resume_dir and vec_norm_path.exists():
        print(f"Loading VecNormalize stats from {vec_norm_path}")
        envs = VecNormalize.load(str(vec_norm_path), envs)
        envs.training = True
    else:
        envs = VecNormalize(
            envs,
            norm_obs=False,
            norm_reward=True,
            clip_reward=10.0,
            gamma=args.gamma,
        )

    logger = configure(str(log_dir), ["stdout", "csv", "tensorboard"])

    model = None
    if resume_dir:
        model_candidates = [
            model_dir / "best_eval_model.zip",
            model_dir / "final_model.zip",
            model_dir / "interrupted_model.zip",
            model_dir / "latest_model.zip",
        ]
        for candidate in model_candidates:
            if candidate.exists():
                print(f"Loading model from {candidate}")
                model = MaskablePPO.load(
                    str(candidate), env=envs, tensorboard_log=str(log_dir)
                )
                model.set_logger(logger)
                break

    if model is None:
        print("Initializing new MaskablePPO v2 model")
        model = MaskablePPO(
            "MultiInputPolicy",
            envs,
            verbose=1,
            learning_rate=LinearSchedule(args.lr_initial, args.lr_final),
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            target_kl=args.target_kl,
            policy_kwargs={
                "net_arch": {"pi": [256, 256], "vf": [256, 256]},
                "activation_fn": torch.nn.SiLU,
                "ortho_init": False,
            },
            tensorboard_log=str(log_dir),
        )
        model.set_logger(logger)

    eval_env = SimcEnvV2(
        simc_path=args.simc,
        profile=args.profile,
        action_blacklist=action_blacklist,
        seed=(args.seed + 2_000_000) if args.seed is not None else None,
        worker_id=0,
        intermediate_rewards=True,
        show_simc_stderr=args.show_simc_stderr,
        no_op_strategy=args.no_op_strategy,
        fight_length=args.fight_length,
    )

    callback_wrapper = TrainingCallback(
        model_dir=model_dir,
        vec_normalize_env=envs,
        eval_env=eval_env,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        save_freq=args.save_freq,
        train_log_freq=args.train_log_freq,
        resume=resume_dir is not None,
        verbose=1,
    )

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback_wrapper.callback,
            log_interval=1,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("Training interrupted by user")
        model.save(model_dir / "interrupted_model")
        envs.save(str(model_dir / "vec_normalize.pkl"))
    finally:
        eval_env.close()
        envs.close()


def run_eval(args: argparse.Namespace) -> None:
    from sb3_contrib import MaskablePPO

    action_blacklist = set() if args.no_blacklist else GLOBAL_ACTION_BLACKLIST
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise ValueError(f"Model directory does not exist: {model_dir}")

    model_path = load_model_for_eval(model_dir)
    print(f"Loading model from {model_path}")
    model = MaskablePPO.load(str(model_path))

    probe_env = SimcEnvV2(
        simc_path=args.simc,
        profile=args.profile,
        action_blacklist=action_blacklist,
        seed=args.seed,
        worker_id=0,
        intermediate_rewards=True,
        no_op_strategy=args.no_op_strategy,
        fight_length=args.fight_length,
        show_simc_stderr=args.show_simc_stderr,
    )
    if probe_env._config is None:
        raise RuntimeError("Probe failed: missing config")
    probe = EvalProbe(config=probe_env._config)
    probe_env.close()

    output_dir = Path(args.output_dir) if args.output_dir else (model_dir / "evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    html_output = output_dir / f"eval_{timestamp}.html"
    text_output = output_dir / f"eval_{timestamp}.txt"

    cmd = [
        args.simc,
        args.profile,
        "rl_enable=1",
        "rl_stdio=1",
        "rl_trace=0",
        f"iterations={args.iterations}",
        f"max_time={args.fight_length}",
        f"html={html_output}",
        f"output={text_output}",
    ]
    if args.seed is not None:
        cmd.append(f"seed={args.seed}")

    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    encoder = EvalEncoder(probe=probe, no_op_strategy=args.no_op_strategy)
    encoder.on_new_iteration()

    start = time.time()
    total_steps = 0
    iteration_count = 0
    damages: list[float] = []
    non_json_lines: list[str] = []

    try:
        while True:
            line = process.stdout.readline() if process.stdout else ""
            if not line:
                break
            line = line.strip()
            if not line:
                continue

            if not line.startswith("{"):
                non_json_lines.append(line)
                continue

            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                non_json_lines.append(line)
                continue

            if msg.get("type") == "done":
                iteration_count += 1
                damages.append(float(msg.get("total_damage", 0.0)))
                encoder.on_new_iteration()
                if iteration_count % 100 == 0:
                    print(
                        f"Iteration {iteration_count}/{args.iterations} - mean(last100)={np.mean(damages[-100:]):,.0f}"
                    )
                continue

            obs_dict, mask = encoder.encode(msg)
            vec_obs = {
                "obs": obs_dict["obs"].reshape(1, -1),
                "mask": obs_dict["mask"].reshape(1, -1),
            }
            mask_vec = mask.reshape(1, -1)
            action, _ = model.predict(vec_obs, deterministic=True, action_masks=mask_vec)
            action = int(action[0])
            raw_action = probe.config.action_index_map[action]
            if process.stdin is None:
                raise RuntimeError("simc stdin not available")
            process.stdin.write(f"{raw_action}\n")
            process.stdin.flush()
            total_steps += 1
    finally:
        try:
            if process.stdin:
                process.stdin.close()
        except Exception:
            pass

        stderr_output = process.stderr.read() if process.stderr else ""
        process.wait(timeout=30.0)

    elapsed = time.time() - start

    print("=== Evaluation Complete ===")
    print(f"Iterations completed: {iteration_count}")
    print(f"Total decision steps: {total_steps:,}")
    print(f"Time elapsed: {elapsed:.1f}s")
    if damages:
        print(f"Mean damage: {np.mean(damages):,.0f}")
        print(f"Std damage: {np.std(damages):,.0f}")
        print(f"Min damage: {np.min(damages):,.0f}")
        print(f"Max damage: {np.max(damages):,.0f}")

    print(f"HTML report: {html_output}")
    print(f"Text report: {text_output}")

    if non_json_lines:
        print(f"Non-JSON lines captured: {len(non_json_lines)}")

    if stderr_output.strip():
        print("--- SimC stderr ---")
        print(stderr_output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SimulationCraft RL v2 - PPO training/evaluation with improved masking and model selection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--simc", default="./simc", help="Path to simc executable")
    parser.add_argument("--profile", required=True, help="Path to .simc profile")

    parser.add_argument("--resume", type=str, default=None, help="Training run dir to resume")
    parser.add_argument("--model-dir", type=str, default=None, help="Training run dir for eval")
    parser.add_argument("--output-dir", type=str, default=None, help="Eval output directory")

    parser.add_argument("--iterations", type=int, default=2000, help="Iterations for eval mode")
    parser.add_argument("--fight-length", type=float, default=300.0, help="max_time passed to simc")

    parser.add_argument("--num-envs", type=int, default=12)
    parser.add_argument("--total-timesteps", type=int, default=16_000_000)
    parser.add_argument("--seed", type=int, default=12345)

    parser.add_argument("--eval-freq", type=int, default=200_000)
    parser.add_argument("--eval-episodes", type=int, default=32)
    parser.add_argument("--save-freq", type=int, default=50_000)
    parser.add_argument("--train-log-freq", type=int, default=5_000)

    parser.add_argument("--lr-initial", type=float, default=2.5e-4)
    parser.add_argument("--lr-final", type=float, default=2.5e-5)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.98)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.005)
    parser.add_argument("--vf-coef", type=float, default=0.7)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.03)

    parser.add_argument("--no-blacklist", action="store_true")
    parser.add_argument("--show-simc-stderr", action="store_true")
    parser.add_argument(
        "--no-op-strategy",
        choices=["mask_when_real", "keep_all"],
        default="mask_when_real",
        help="How to handle wait/pass pseudo-actions",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "eval":
        if not args.model_dir:
            parser.error("--model-dir is required for eval mode")
        run_eval(args)
        return

    run_train(args)


if __name__ == "__main__":
    main()
