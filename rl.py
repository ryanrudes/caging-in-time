from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from env import CagingEnvParams, PolygonCagingEnv


RUNS_DIR = Path("runs")
MODELS_DIR = Path("models")
LOGS_DIR = Path("tb_logs")


def make_env(rank: int, seed: int, render_mode: str | None = None):
    def _init() -> gym.Env:
        params = CagingEnvParams(
            include_noop_action=True,
            include_polygon_state_in_obs=True,
        )
        env = PolygonCagingEnv(params=params, render_mode=render_mode)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    return _init


def train(
    total_timesteps: int,
    num_envs: int,
    seed: int,
    run_name: str,
    check: bool,
) -> None:
    run_dir = RUNS_DIR / run_name
    model_dir = MODELS_DIR / run_name
    log_dir = LOGS_DIR / run_name
    checkpoint_dir = run_dir / "checkpoints"

    run_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if check:
        check_env(PolygonCagingEnv(params=CagingEnvParams()), warn=True, skip_render_check=True)

    vec_cls = DummyVecEnv if num_envs == 1 else SubprocVecEnv
    env = vec_cls([make_env(rank=i, seed=seed) for i in range(num_envs)])
    env = VecMonitor(env)

    eval_env = DummyVecEnv([make_env(rank=10_000, seed=seed)])
    eval_env = VecMonitor(eval_env)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(10_000 // num_envs, 1),
        save_path=str(checkpoint_dir),
        name_prefix="ppo_caging",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir),
        log_path=str(run_dir / "eval"),
        eval_freq=max(25_000 // num_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
        render=False,
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
        seed=seed,
        verbose=1,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    final_path = model_dir / "final_model"
    model.save(str(final_path))
    env.close()
    eval_env.close()

    print(f"Saved final model to {final_path}.zip")


def enjoy(model_path: str, seed: int, deterministic: bool) -> None:
    env = PolygonCagingEnv(
        params=CagingEnvParams(
            include_noop_action=True,
            include_polygon_state_in_obs=True,
        ),
        render_mode="human",
    )
    model = PPO.load(model_path)

    obs, info = env.reset(seed=seed)
    terminated = False
    truncated = False
    total_reward = 0.0

    while not terminated and not truncated:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += float(reward)
        env.render()

    env.close()
    print(f"Episode reward: {total_reward:.3f}")
    print(f"Terminated: {terminated}, truncated: {truncated}")
    print(f"Final info: {info}")


def heuristic_rollout(seed: int) -> None:
    env = PolygonCagingEnv(render_mode="human")
    obs, info = env.reset(seed=seed)

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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "train":
        train(
            total_timesteps=args.total_timesteps,
            num_envs=args.num_envs,
            seed=args.seed,
            run_name=args.run_name,
            check=not args.no_check,
        )
    elif args.mode == "enjoy":
        enjoy(
            model_path=args.model_path,
            seed=args.seed,
            deterministic=not args.stochastic,
        )
    else:
        heuristic_rollout(seed=args.seed)