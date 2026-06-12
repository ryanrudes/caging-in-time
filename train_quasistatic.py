from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Callable

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from torch.utils.tensorboard import SummaryWriter

from quasistatic_env import CagingEnvParams, PolygonCagingEnv
from train_dynamic import SmoothRandomTrajectory


RUNS_DIR = Path("runs")
MODELS_DIR = Path("models")
LOGS_DIR = Path("tb_logs")


class RandomTrajectoryPolygonCagingEnv(PolygonCagingEnv):
    def __init__(
        self,
        params: CagingEnvParams,
        random_trajectory: bool,
        trajectory_seed: int,
        max_position: float,
        max_velocity: float,
        max_acceleration: float,
        render_mode: str | None = None,
    ):
        self.random_trajectory = random_trajectory
        self.trajectory_rng = np.random.default_rng(trajectory_seed)
        self.random_max_position = max_position
        self.random_max_velocity = max_velocity
        self.random_max_acceleration = max_acceleration
        super().__init__(params=params, render_mode=render_mode)

    def reset(self, *args, **kwargs):
        if self.random_trajectory:
            self._sample_generated_trajectory()
        return super().reset(*args, **kwargs)

    def _sample_generated_trajectory(self) -> None:
        dt = 0.02
        horizon_seconds = self.max_episode_steps * dt
        traj = SmoothRandomTrajectory(
            rng=self.trajectory_rng,
            n=2,
            horizon_seconds=horizon_seconds,
            max_position=self.random_max_position,
            max_velocity=self.random_max_velocity,
            max_acceleration=self.random_max_acceleration,
        )
        trajectory_points = []
        for step in range(self.max_episode_steps + 1):
            pos, _, _ = traj.evaluate(step * dt)
            trajectory_points.append((float(pos[0]), float(pos[1])))
        self.trajectory = trajectory_points


def make_env(
    rank: int,
    seed: int,
    trajectory_mode: str,
    random_max_position: float,
    random_max_velocity: float,
    random_max_acceleration: float,
    render_mode: str | None = None,
) -> Callable[[], gym.Env]:
    def _init() -> gym.Env:
        # Limited information, as in a real deployment: the agent sees only the
        # PSS/POA belief, the prescribed cage trajectory, and its own actions.
        params = CagingEnvParams(
            include_noop_action=True,
            include_polygon_state_in_obs=False,
        )
        env = RandomTrajectoryPolygonCagingEnv(
            params=params,
            random_trajectory=trajectory_mode == "random",
            trajectory_seed=seed + 100_000 * rank,
            max_position=random_max_position,
            max_velocity=random_max_velocity,
            max_acceleration=random_max_acceleration,
            render_mode=render_mode,
        )
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    return _init


class VisualizationCallback(BaseCallback):
    def __init__(
        self,
        make_eval_env: Callable[[], gym.Env],
        log_dir: Path,
        run_dir: Path,
        render_freq: int,
        n_frames: int,
        deterministic: bool = True,
    ):
        super().__init__()
        self.make_eval_env = make_eval_env
        self.log_dir = log_dir
        self.run_dir = run_dir
        self.video_dir = run_dir / "videos"
        self.render_freq = render_freq
        self.n_frames = n_frames
        self.deterministic = deterministic
        self.writer: SummaryWriter | None = None

    def _on_training_start(self) -> None:
        self.writer = SummaryWriter(str(self.log_dir / "visualizations"))
        self.video_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.render_freq <= 0 or self.n_calls % self.render_freq != 0:
            return True
        if self.writer is None:
            return True

        env = self.make_eval_env()
        obs, info = env.reset(seed=10_000 + self.n_calls)
        frames = []
        rewards = []
        max_vertex_dists = []
        pss_sizes = []

        terminated = False
        truncated = False
        for _ in range(self.n_frames):
            frame = env.render()
            if frame is not None:
                h, w = frame.shape[:2]
                new_h = h - (h % 2)
                new_w = w - (w % 2)
                if new_h != h or new_w != w:
                    frame = frame[:new_h, :new_w]
            frames.append(frame)
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
            obs, reward, terminated, truncated, info = env.step(int(action))
            rewards.append(float(reward))
            if "max_vertex_dist" in info:
                max_vertex_dists.append(float(info["max_vertex_dist"]))
            if "pss_size" in info:
                pss_sizes.append(float(info["pss_size"]))
            if terminated or truncated:
                break

        env.close()

        if frames:
            video_path = self.video_dir / f"rollout_{self.num_timesteps:09d}.mp4"
            imageio.mimsave(
                video_path,
                frames,
                fps=30,
                macro_block_size=1,
            )
            print(f"Saved rollout video to {video_path}")
            self.writer.add_image(
                "rollout/final_frame",
                np.transpose(frames[-1], (2, 0, 1)),
                global_step=self.num_timesteps,
            )

        if rewards:
            self.writer.add_scalar("rollout/return", float(np.sum(rewards)), self.num_timesteps)
            self.writer.add_scalar("rollout/mean_reward", float(np.mean(rewards)), self.num_timesteps)
        if max_vertex_dists:
            self.writer.add_scalar("rollout/final_max_vertex_dist", max_vertex_dists[-1], self.num_timesteps)
        if pss_sizes:
            self.writer.add_scalar("rollout/final_pss_size", pss_sizes[-1], self.num_timesteps)
        self.writer.flush()
        return True

    def _on_training_end(self) -> None:
        if self.writer is not None:
            self.writer.close()


def train(args: argparse.Namespace) -> None:
    run_dir = RUNS_DIR / args.run_name
    model_dir = MODELS_DIR / args.run_name
    log_dir = LOGS_DIR / args.run_name
    checkpoint_dir = run_dir / "checkpoints"

    run_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_check:
        check_env(
            RandomTrajectoryPolygonCagingEnv(
                params=CagingEnvParams(),
                random_trajectory=args.trajectory_mode == "random",
                trajectory_seed=args.seed,
                max_position=args.random_max_position,
                max_velocity=args.random_max_velocity,
                max_acceleration=args.random_max_acceleration,
            ),
            warn=True,
            skip_render_check=True,
        )

    vec_cls = DummyVecEnv if args.num_envs == 1 else SubprocVecEnv
    env = vec_cls(
        [
            make_env(
                rank=i,
                seed=args.seed,
                trajectory_mode=args.trajectory_mode,
                random_max_position=args.random_max_position,
                random_max_velocity=args.random_max_velocity,
                random_max_acceleration=args.random_max_acceleration,
            )
            for i in range(args.num_envs)
        ]
    )
    env = VecMonitor(env)

    eval_env = DummyVecEnv(
        [
            make_env(
                rank=10_000,
                seed=args.seed,
                trajectory_mode=args.eval_trajectory_mode,
                random_max_position=args.random_max_position,
                random_max_velocity=args.random_max_velocity,
                random_max_acceleration=args.random_max_acceleration,
            )
        ]
    )
    eval_env = VecMonitor(eval_env)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.num_envs, 1),
        save_path=str(checkpoint_dir),
        name_prefix="ppo_caging",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir),
        log_path=str(run_dir / "eval"),
        eval_freq=max(args.eval_freq // args.num_envs, 1),
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )

    viz_callback = VisualizationCallback(
        make_eval_env=make_env(
            rank=20_000,
            seed=args.seed,
            trajectory_mode=args.eval_trajectory_mode,
            random_max_position=args.random_max_position,
            random_max_velocity=args.random_max_velocity,
            random_max_acceleration=args.random_max_acceleration,
            render_mode="rgb_array",
        ),
        log_dir=log_dir,
        run_dir=run_dir,
        render_freq=args.visualization_freq,
        n_frames=args.visualization_frames,
        deterministic=True,
    )

    policy_kwargs = {
        "net_arch": {
            "pi": [256, 256],
            "vf": [256, 256],
        }
    }

    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(log_dir),
        seed=args.seed,
        verbose=1,
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_callback, eval_callback, viz_callback],
        progress_bar=True,
    )

    final_path = model_dir / "final_model"
    model.save(str(final_path))
    env.close()
    eval_env.close()

    print(f"Saved final model to {final_path}.zip")


def enjoy(args: argparse.Namespace) -> None:
    params = CagingEnvParams(
        include_noop_action=True,
        include_polygon_state_in_obs=False,
    )
    env = RandomTrajectoryPolygonCagingEnv(
        params=params,
        random_trajectory=args.trajectory_mode == "random",
        trajectory_seed=args.seed,
        max_position=args.random_max_position,
        max_velocity=args.random_max_velocity,
        max_acceleration=args.random_max_acceleration,
        render_mode="human",
    )
    model = PPO.load(args.model_path)

    obs, info = env.reset(seed=args.seed)
    terminated = False
    truncated = False
    total_reward = 0.0

    while not terminated and not truncated:
        action, _ = model.predict(obs, deterministic=not args.stochastic)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += float(reward)
        env.render()

    env.close()
    print(f"Episode reward: {total_reward:.3f}")
    print(f"Terminated: {terminated}, truncated: {truncated}")
    print(f"Final info: {info}")


def heuristic_rollout(args: argparse.Namespace) -> None:
    params = CagingEnvParams(
        include_noop_action=True,
        include_polygon_state_in_obs=False,
    )
    env = RandomTrajectoryPolygonCagingEnv(
        params=params,
        random_trajectory=args.trajectory_mode == "random",
        trajectory_seed=args.seed,
        max_position=args.random_max_position,
        max_velocity=args.random_max_velocity,
        max_acceleration=args.random_max_acceleration,
        render_mode="human",
    )
    obs, info = env.reset(seed=args.seed)

    terminated = False
    truncated = False
    total_reward = 0.0

    while not terminated and not truncated:
        action = env.heuristic_policy_action()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        env.render()

    env.close()
    print(f"Heuristic episode reward: {total_reward:.3f}")
    print(f"Terminated: {terminated}, truncated: {truncated}")
    print(f"Final info: {info}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "enjoy", "heuristic"], default="train")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", type=str, default="ppo_caging")
    parser.add_argument("--model-path", type=str, default="models/ppo_caging/best_model.zip")
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--stochastic", action="store_true")

    parser.add_argument("--trajectory-mode", choices=["fixed", "random"], default="fixed")
    parser.add_argument("--eval-trajectory-mode", choices=["fixed", "random"], default="fixed")
    parser.add_argument("--random-max-position", type=float, default=0.055)
    parser.add_argument("--random-max-velocity", type=float, default=0.08)
    parser.add_argument("--random-max-acceleration", type=float, default=0.25)
    parser.add_argument("--checkpoint-freq", type=int, default=10_000)
    parser.add_argument("--visualization-freq", type=int, default=25_000)
    parser.add_argument("--visualization-frames", type=int, default=300)
    parser.add_argument("--eval-freq", type=int, default=25_000)
    parser.add_argument("--n-eval-episodes", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "enjoy":
        enjoy(args)
    else:
        heuristic_rollout(args)