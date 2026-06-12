

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Callable, Literal

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from torch.utils.tensorboard import SummaryWriter

from dynamic_env import DynamicPlateTilt, DynamicPlateTiltConfig


RUNS_DIR = Path("runs")
MODELS_DIR = Path("models")
LOGS_DIR = Path("tb_logs")

TrajectoryMode = Literal["fixed", "random"]


class SmoothRandomTrajectory:
    """
    Band-limited analytic trajectory with bounded position, velocity, and acceleration.

    This is intentionally not a random walk. The plate center follows a sum of low-frequency
    sinusoids, then all coefficients are scaled so the generated path satisfies practical
    position, velocity, and acceleration limits over the episode horizon.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        n: int,
        horizon_seconds: float,
        max_position: float,
        max_velocity: float,
        max_acceleration: float,
        min_frequency_hz: float = 0.04,
        max_frequency_hz: float = 0.35,
        num_harmonics: int = 3,
    ):
        self.n = n
        self.horizon_seconds = horizon_seconds
        self.max_position = max_position
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.num_harmonics = num_harmonics

        self.omega = 2.0 * np.pi * rng.uniform(
            min_frequency_hz,
            max_frequency_hz,
            size=(n, num_harmonics),
        )
        self.phase = rng.uniform(0.0, 2.0 * np.pi, size=(n, num_harmonics))

        raw_amplitudes = rng.normal(0.0, 1.0, size=(n, num_harmonics))
        harmonic_decay = 1.0 / np.arange(1, num_harmonics + 1, dtype=float) ** 1.5
        self.amplitudes = raw_amplitudes * harmonic_decay[np.newaxis, :]

        # Randomly rotate 2D trajectories so training does not overfit to world axes.
        self.rotation = np.eye(n)
        if n == 2:
            angle = rng.uniform(0.0, 2.0 * np.pi)
            c = np.cos(angle)
            s = np.sin(angle)
            self.rotation = np.array([[c, -s], [s, c]], dtype=float)

        self.offset = np.zeros(n, dtype=float)
        self._center_initial_position()
        self._scale_to_limits()

    def evaluate(self, time: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        phases = self.omega * time + self.phase
        position = np.sum(self.amplitudes * np.sin(phases), axis=1)
        velocity = np.sum(self.amplitudes * self.omega * np.cos(phases), axis=1)
        acceleration = np.sum(-self.amplitudes * self.omega * self.omega * np.sin(phases), axis=1)

        position = self.rotation @ position + self.offset
        velocity = self.rotation @ velocity
        acceleration = self.rotation @ acceleration
        return position, velocity, acceleration

    def _center_initial_position(self) -> None:
        position0, _, _ = self.evaluate(0.0)
        self.offset = -position0

    def _scale_to_limits(self) -> None:
        times = np.linspace(0.0, self.horizon_seconds, 256)
        positions = []
        velocities = []
        accelerations = []
        for time in times:
            position, velocity, acceleration = self.evaluate(float(time))
            positions.append(position)
            velocities.append(velocity)
            accelerations.append(acceleration)

        positions = np.asarray(positions)
        velocities = np.asarray(velocities)
        accelerations = np.asarray(accelerations)

        max_observed_position = max(float(np.max(np.linalg.norm(positions, axis=1))), 1e-9)
        max_observed_velocity = max(float(np.max(np.linalg.norm(velocities, axis=1))), 1e-9)
        max_observed_acceleration = max(float(np.max(np.linalg.norm(accelerations, axis=1))), 1e-9)

        scale = min(
            self.max_position / max_observed_position,
            self.max_velocity / max_observed_velocity,
            self.max_acceleration / max_observed_acceleration,
        )
        scale = min(scale, 1.0)
        self.amplitudes *= scale
        self.offset *= scale


class RandomTrajectoryDynamicPlateTilt(DynamicPlateTilt):
    def __init__(
        self,
        config: DynamicPlateTiltConfig,
        random_trajectory: bool,
        trajectory_seed: int,
        max_position: float,
        max_velocity: float,
        max_acceleration: float,
        render_mode: str | None = None,
    ):
        # DynamicPlateTilt.__init__ computes render bounds and calls plate_trajectory(),
        # so subclass trajectory attributes must exist before super().__init__().
        self.random_trajectory = random_trajectory
        self.trajectory_rng = np.random.default_rng(trajectory_seed)
        self.random_max_position = max_position
        self.random_max_velocity = max_velocity
        self.random_max_acceleration = max_acceleration
        self.generated_trajectory: SmoothRandomTrajectory | None = None
        super().__init__(config=config, render_mode=render_mode)

    def reset(self, *args, **kwargs):
        if self.random_trajectory:
            self._sample_generated_trajectory()
            self._render_bounds = self._compute_render_bounds()
        return super().reset(*args, **kwargs)

    def plate_trajectory(self, time: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        generated_trajectory = getattr(self, "generated_trajectory", None)
        if generated_trajectory is not None:
            return generated_trajectory.evaluate(time)
        return super().plate_trajectory(time)

    def _sample_generated_trajectory(self) -> None:
        horizon_seconds = self.config.max_episode_steps * self.config.dt
        self.generated_trajectory = SmoothRandomTrajectory(
            rng=self.trajectory_rng,
            n=self.n,
            horizon_seconds=horizon_seconds,
            max_position=self.random_max_position,
            max_velocity=self.random_max_velocity,
            max_acceleration=self.random_max_acceleration,
        )


def make_config(args: argparse.Namespace) -> DynamicPlateTiltConfig:
    return DynamicPlateTiltConfig(
        dimensionality=args.dimensionality,
        trajectory=args.trajectory,
        boundary_energy_method=args.boundary_energy_method,
        plate_size=args.plate_size,
        virtual_elastic_stiffness=args.virtual_elastic_stiffness,
        dt=args.dt,
        max_episode_steps=args.max_episode_steps,
        max_tilt_angle=args.max_tilt_angle,
        max_tilt_rate=args.max_tilt_rate,
        trajectory_radius=args.trajectory_radius,
        trajectory_speed=args.trajectory_speed,
        line_amplitude=args.line_amplitude,
        pss_sample_count=args.pss_sample_count,
        pss_energy_percentile=args.pss_energy_percentile,
        terminate_on_pss_energy_violation=args.terminate_on_pss_energy_violation,
        terminate_on_pss_position_escape=args.terminate_on_pss_position_escape,
        render_pss=args.render_pss,
        include_pss_particles_in_obs=not args.no_pss_particles,
        trajectory_preview_step_offsets=tuple(args.trajectory_preview_offsets),
    )


def make_env(
    rank: int,
    seed: int,
    config: DynamicPlateTiltConfig,
    trajectory_mode: TrajectoryMode,
    random_max_position: float,
    random_max_velocity: float,
    random_max_acceleration: float,
    render_mode: str | None = None,
) -> Callable[[], gym.Env]:
    def _init() -> gym.Env:
        env = RandomTrajectoryDynamicPlateTilt(
            config=config,
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
        energy_margin = []
        pss_energy_margin = []

        terminated = False
        truncated = False
        for _ in range(self.n_frames):
            frame = env.render()
            frames.append(frame)
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(float(reward))
            energy_margin.append(float(info.get("energy_margin", 0.0)))
            pss_energy_margin.append(float(info.get("pss_energy_margin", 0.0)))
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
            # TensorBoard add_video depends on moviepy.editor, which is brittle across
            # MoviePy versions. Save MP4s directly and keep TensorBoard image/scalar logs.
            self.writer.add_image(
                "rollout/final_frame",
                np.transpose(frames[-1], (2, 0, 1)),
                global_step=self.num_timesteps,
            )

        if rewards:
            self.writer.add_scalar("rollout/return", float(np.sum(rewards)), self.num_timesteps)
            self.writer.add_scalar("rollout/mean_reward", float(np.mean(rewards)), self.num_timesteps)
        if energy_margin:
            self.writer.add_scalar("rollout/final_energy_margin", energy_margin[-1], self.num_timesteps)
            self.writer.add_scalar("rollout/final_pss_energy_margin", pss_energy_margin[-1], self.num_timesteps)
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

    config = make_config(args)

    if args.check_env:
        check_env(
            RandomTrajectoryDynamicPlateTilt(
                config=config,
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
                config=config,
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
                config=config,
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
        name_prefix="ppo_dynamic_caging",
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
            config=replace(config, render_pss=args.render_pss),
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

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
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
    import cv2

    config = make_config(args)
    env = RandomTrajectoryDynamicPlateTilt(
        config=config,
        random_trajectory=args.trajectory_mode == "random",
        trajectory_seed=args.seed,
        max_position=args.random_max_position,
        max_velocity=args.random_max_velocity,
        max_acceleration=args.random_max_acceleration,
        render_mode="rgb_array",
    )
    model = PPO.load(args.model_path)

    obs, info = env.reset(seed=args.seed)
    terminated = False
    truncated = False
    total_reward = 0.0

    while not terminated and not truncated:
        action, _ = model.predict(obs, deterministic=not args.stochastic)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        frame = env.render()
        cv2.imshow("Dynamic RL", frame)
        if cv2.waitKey(20) & 0xFF == ord("q"):
            break

    env.close()
    cv2.destroyAllWindows()
    print(f"Episode reward: {total_reward:.3f}")
    print(f"Terminated: {terminated}, truncated: {truncated}")
    print(f"Final info: {info}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "enjoy"], default="train")
    parser.add_argument("--run-name", type=str, default="ppo_dynamic_caging")
    parser.add_argument("--model-path", type=str, default="models/ppo_dynamic_caging/best_model.zip")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--dimensionality", type=int, choices=[2, 3], default=2)
    parser.add_argument("--trajectory-mode", choices=["fixed", "random"], default="fixed")
    parser.add_argument("--eval-trajectory-mode", choices=["fixed", "random"], default="fixed")
    parser.add_argument("--trajectory", choices=["static", "line", "circle", "figure8"], default="line")
    parser.add_argument("--boundary-energy-method", choices=["corners", "edges"], default="corners")

    parser.add_argument("--plate-size", type=float, default=0.16)
    parser.add_argument("--virtual-elastic-stiffness", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--max-tilt-angle", type=float, default=np.pi / 9.0)
    parser.add_argument("--max-tilt-rate", type=float, default=np.pi / 3.0)
    parser.add_argument("--trajectory-radius", type=float, default=0.05)
    parser.add_argument("--trajectory-speed", type=float, default=0.5)
    parser.add_argument("--line-amplitude", type=float, default=0.05)

    parser.add_argument("--random-max-position", type=float, default=0.055)
    parser.add_argument("--random-max-velocity", type=float, default=0.08)
    parser.add_argument("--random-max-acceleration", type=float, default=0.25)

    parser.add_argument("--pss-sample-count", type=int, default=256)
    parser.add_argument("--pss-energy-percentile", type=float, default=95.0)
    parser.add_argument("--no-pss-particles", action="store_true")
    parser.add_argument("--trajectory-preview-offsets", type=int, nargs="+", default=[1, 5, 10, 25])
    parser.add_argument("--terminate-on-pss-energy-violation", action="store_true")
    parser.add_argument("--terminate-on-pss-position-escape", action="store_true")

    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)

    parser.add_argument("--eval-freq", type=int, default=25_000)
    parser.add_argument("--n-eval-episodes", type=int, default=10)
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--visualization-freq", type=int, default=50_000)
    parser.add_argument("--visualization-frames", type=int, default=300)
    parser.add_argument("--render-pss", action="store_true")
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument("--stochastic", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        train(args)
    else:
        enjoy(args)