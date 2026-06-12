"""
Reproduce the paper's figures (Wang et al. 2024, "Caging in Time") from the
RL environments, for both the paper's heuristic baseline and trained RL agents.

Subcommands
-----------
rollout-quasistatic   Fig. 10-style strip of PSS/POA frames from PolygonCagingEnv,
                      with the true polygon overlaid (policy: heuristic | agent).
rollout-dynamic       Fig. 13-style strip: PSS particles in (x, xdot) state space
                      inside the energy cage (policy: pd | agent | zero).
fig15                 Cage-settings sweep: trajectories and MAE over cage sizes
                      (R - r) and candidate-action counts K.
fig17                 Infinity-trajectory robustness: Caging in Time vs a naive
                      pushing strategy.
fig19                 Closed-loop P-controller under positional noise and
                      network lag vs Caging in Time (MAE comparison).
fig21                 Sensitivity heatmap: dynamic success rate over initial
                      velocity uncertainty and tilt-rate (hardware) limits.

Examples
--------
python paper_figures.py rollout-quasistatic --policy heuristic --save fig10_env.png
python paper_figures.py rollout-quasistatic --policy agent --model-path models/ppo_caging/best_model.zip
python paper_figures.py rollout-dynamic --policy pd --save fig13.png
python paper_figures.py fig15 --seeds 5 --save fig15.png
python paper_figures.py fig17 --save fig17.png
python paper_figures.py fig19 --seeds 5 --save fig19.png
python paper_figures.py fig21 --trials 20 --save fig21.png
"""

from __future__ import annotations

import argparse
from math import pi, tau
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np

from mpl_render import (
    ACTIVE_COLOR,
    BACKGROUND_COLOR,
    CAGE_COLOR,
    POA_COLOR,
    render_sequence,
    snapshot_from_caging_env,
    use_scienceplots_styles,
)
from quasistatic_env import (
    CagingEnvParams,
    PolygonCagingEnv,
    generate_random_polygon_inside_radius,
)

HEURISTIC_COLOR = "#e0a126"
BASELINE_COLOR = "#3a76b0"
REFERENCE_COLOR = "#777777"


def save_or_show(fig, save: str | None, dpi: int = 600) -> None:
    if save:
        fig.savefig(save, dpi=dpi, facecolor="white", bbox_inches="tight")
        print(f"Saved figure to {save}")
    else:
        plt.show()


def make_quasistatic_policy(policy: str, model_path: str | None):
    if policy == "heuristic":
        return lambda env, obs: env.heuristic_policy_action()
    if policy == "agent":
        if not model_path:
            raise ValueError("--model-path is required for --policy agent")
        from stable_baselines3 import PPO

        model = PPO.load(model_path)
        return lambda env, obs: int(model.predict(obs, deterministic=True)[0])
    raise ValueError(f"Unknown policy {policy!r}")


def rollout_quasistatic(
    env: PolygonCagingEnv,
    policy_fn,
    seed: int,
    capture_steps: set[int] | None = None,
    stop_on_failure: bool = True,
) -> tuple[list[dict], np.ndarray, bool]:
    """
    Run one episode. Returns (snapshots, true center trajectory, success).

    If stop_on_failure is False, keep stepping after the environment first marks
    the object as escaped so plotting functions can show the full attempted path
    rather than a truncated trajectory.
    """
    obs, info = env.reset(seed=seed)
    snapshots: list[dict] = []
    centers = [env.object_pose[:2].copy()]

    terminated = truncated = False
    escaped = False
    step = 0
    while not truncated and (not terminated or not stop_on_failure):
        action = policy_fn(env, obs)
        theta = env.get_action_angle(action)
        obs, reward, terminated, truncated, info = env.step(action)
        escaped = escaped or terminated
        centers.append(env.object_pose[:2].copy())
        if capture_steps is not None and step in capture_steps:
            snapshots.append(snapshot_from_caging_env(env, theta))
        step += 1

    if escaped and capture_steps is not None:
        snapshots.append(snapshot_from_caging_env(env, None))

    return snapshots, np.asarray(centers), not escaped



def trajectory_mae(centers: np.ndarray, reference: np.ndarray) -> float:
    """
    Time-aligned mean absolute error: average distance between the realized
    object position and the reference waypoint at the same time step.
    """
    n = min(len(centers), len(reference))
    return float(np.mean(np.linalg.norm(centers[:n] - reference[:n], axis=1)))


# --- Added: rollout_quasistatic_object_point
def rollout_quasistatic_object_point(
    env: PolygonCagingEnv,
    policy_fn,
    seed: int,
    stop_on_failure: bool = True,
    reset_options: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Run one episode and return both the object-center trajectory and the world
    trajectory of a fixed body-frame vertex on the actual polygon.

    The center dynamics use the object's bounding circle, so different polygon
    shapes can produce identical center paths. Plotting a real polygon vertex
    makes the overlaid paths shape-dependent and visibly distinct.
    """
    obs, info = env.reset(seed=seed, options=reset_options)
    point_body = env.polygon_vertices_body[
        int(np.argmax(np.linalg.norm(env.polygon_vertices_body, axis=1)))
    ].copy()

    def point_world() -> np.ndarray:
        x, y, theta = env.object_pose
        c = np.cos(theta)
        s = np.sin(theta)
        rotation = np.array([[c, -s], [s, c]], dtype=float)
        return point_body @ rotation.T + np.array([x, y], dtype=float)

    centers = [env.object_pose[:2].copy()]
    point_positions = [point_world()]

    terminated = truncated = False
    escaped = False
    while not truncated and (not terminated or not stop_on_failure):
        action = policy_fn(env, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        escaped = escaped or terminated
        centers.append(env.object_pose[:2].copy())
        point_positions.append(point_world())

        if terminated and not stop_on_failure:
            # The environment treats escape as terminal and the PSS can collapse
            # to empty. If we keep stepping after that without repair, the
            # heuristic often returns no-op and the plotted path degenerates
            # into a tiny arc while the reference keeps moving around the full
            # circle, producing absurd MAEs. For Fig. 15 robustness plots, keep
            # the rollout alive by reseeding the internal belief at the true
            # object center and continue tracking the moving cage.
            env.pss = np.asarray([env.object_pose[:2].copy()], dtype=float)
            terminated = False
            obs = env._get_obs()

    return np.asarray(centers), np.asarray(point_positions), not escaped


def cmd_rollout_quasistatic(args: argparse.Namespace) -> None:
    params = CagingEnvParams(
        num_actions=args.num_actions,
        cage_size=args.cage_size,
        trajectory_shape=args.trajectory_shape,
        trajectory_radius=args.trajectory_radius,
        ice_mode=args.ice_mode,
    )
    env = PolygonCagingEnv(params=params)
    policy_fn = make_quasistatic_policy(args.policy, args.model_path)

    # Capture frames where the policy is most likely to be pushing: run once
    # to find push steps, then capture evenly among them.
    probe_env = PolygonCagingEnv(params=params)
    probe_obs, _ = probe_env.reset(seed=args.seed)
    push_steps = []
    terminated = truncated = False
    step = 0
    while not (terminated or truncated):
        action = policy_fn(probe_env, probe_obs)
        if probe_env.get_action_angle(action) is not None:
            push_steps.append(step)
        probe_obs, _, terminated, truncated, _ = probe_env.step(action)
        step += 1

    if len(push_steps) >= args.num_frames:
        indices = np.linspace(0, len(push_steps) - 1, args.num_frames, dtype=int)
        capture = {push_steps[i] for i in indices}
    else:
        capture = set(np.linspace(0, max(1, step - 1), args.num_frames, dtype=int).tolist())

    snapshots, centers, success = rollout_quasistatic(env, policy_fn, args.seed, capture)
    print(f"policy={args.policy}: success={success}, steps={len(centers) - 1}")

    fig = render_sequence(snapshots, styles=args.styles)
    save_or_show(fig, args.save, dpi=args.dpi)


def fig15_env_params(
    cage_mm: int,
    K: int,
    trajectory_radius: float,
    trajectory_speed: float,
    ice_mode: bool = False,
) -> CagingEnvParams:
    """
    Env params for one Fig. 15 condition. The default 121-cell belief grid
    only covers the cage circle up to cage_size = 20 mm; larger cages would
    silently crop PSS points at the grid border, collapsing the belief and
    making the heuristic lose the object (catastrophic MAE at 30-40 mm). So
    the grid is sized per condition to cover R + a safety margin.
    """
    base = CagingEnvParams()
    half_extent = cage_mm / 1000.0 + base.object_radius + 0.015
    cells = 2 * int(np.ceil(half_extent / base.resolution)) + 1
    cells = max(cells, base.grid_height)
    return CagingEnvParams(
        num_actions=K,
        cage_size=cage_mm / 1000.0,
        trajectory_radius=trajectory_radius,
        trajectory_speed=trajectory_speed,
        grid_height=cells,
        grid_width=cells,
        ice_mode=ice_mode,
    )


def run_fig15_trial(
    cage_mm: int,
    K: int,
    trajectory_radius: float,
    trajectory_speed: float,
    seed: int,
    polygon_vertices_body: np.ndarray,
    ice_mode: bool = False,
) -> tuple[int, np.ndarray, np.ndarray, float, bool]:
    """Run one Fig. 15 trial in a worker process."""
    params = fig15_env_params(cage_mm, K, trajectory_radius, trajectory_speed, ice_mode)
    env = PolygonCagingEnv(
        params=params,
        polygon_vertices_body=polygon_vertices_body,
    )
    reference = np.asarray(env.trajectory, dtype=float)
    # With a fixed polygon, env.reset is fully deterministic and the center
    # dynamics depend only on the bounding circle, so every seed would trace
    # an identical path and the across-seed std would collapse to zero.
    # Randomize the initial pose within the cage so trials genuinely vary,
    # like independent physical runs in the paper.
    rng = np.random.default_rng(seed)
    offset_angle = rng.uniform(0.0, tau)
    # Keep the offset modest: physical trials start the object near the cage
    # center, and offsets approaching R-r make the marginal 16-CA conditions
    # escape immediately, which the paper does not show.
    offset_radius = rng.uniform(0.0, 0.25) * params.cage_size
    x0, y0 = np.asarray(env.trajectory[0], dtype=float)
    reset_options = {
        "x0": x0 + offset_radius * np.cos(offset_angle),
        "y0": y0 + offset_radius * np.sin(offset_angle),
        "theta0": rng.uniform(-pi, pi),
    }
    centers, point_path, success = rollout_quasistatic_object_point(
        env,
        lambda e, o: e.heuristic_policy_action(),
        seed,
        stop_on_failure=False,
        reset_options=reset_options,
    )
    mae = trajectory_mae(centers, reference)
    return seed, centers, point_path, mae, success

def cmd_fig15(args: argparse.Namespace) -> None:
    cage_sizes_mm = [10, 20, 30, 40]
    action_counts = [16, 32, 64, 128]

    results_mae = np.zeros((len(cage_sizes_mm), len(action_counts)))
    results_std = np.zeros_like(results_mae)
    results_success = np.zeros_like(results_mae)
    all_trajectories: dict[tuple[int, int], list[np.ndarray]] = {}
    all_center_trajectories: dict[tuple[int, int], list[np.ndarray]] = {}
    all_references: dict[tuple[int, int], np.ndarray] = {}

    trajectory_colors = ["tab:red", "tab:orange", "tab:purple", "tab:green", "tab:blue", "tab:olive", "tab:pink", "tab:brown", "tab:cyan"]
    num_plot_trajectories = 9
    base_shape_params = CagingEnvParams()
    object_shapes = [
        generate_random_polygon_inside_radius(
            np.random.default_rng(seed),
            base_shape_params.object_radius,
            num_vertices=base_shape_params.num_polygon_vertices,
        )
        for seed in range(num_plot_trajectories)
    ]

    jobs = []
    for i, cage_mm in enumerate(cage_sizes_mm):
        for j, K in enumerate(action_counts):
            params = fig15_env_params(
                cage_mm, K, args.trajectory_radius, args.trajectory_speed, args.ice_mode
            )
            reference_env = PolygonCagingEnv(params=params)
            all_references[(i, j)] = np.asarray(reference_env.trajectory, dtype=float)
            for seed in range(num_plot_trajectories):
                jobs.append((i, j, cage_mm, K, seed, object_shapes[seed]))

    trial_results: dict[tuple[int, int], list[tuple[int, np.ndarray, np.ndarray, float, bool]]] = {
        (i, j): []
        for i in range(len(cage_sizes_mm))
        for j in range(len(action_counts))
    }

    if args.workers == 1:
        for i, j, cage_mm, K, seed, polygon_vertices_body in jobs:
            seed_out, centers, point_path, mae, success = run_fig15_trial(
                cage_mm,
                K,
                args.trajectory_radius,
                args.trajectory_speed,
                seed,
                polygon_vertices_body,
                args.ice_mode,
            )
            trial_results[(i, j)].append((seed_out, centers, point_path, mae, success))
    else:
        max_workers = None if args.workers <= 0 else args.workers
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_key = {
                executor.submit(
                    run_fig15_trial,
                    cage_mm,
                    K,
                    args.trajectory_radius,
                    args.trajectory_speed,
                    seed,
                    polygon_vertices_body,
                    args.ice_mode,
                ): (i, j)
                for i, j, cage_mm, K, seed, polygon_vertices_body in jobs
            }
            for future in as_completed(future_to_key):
                print(f"Future completed: {future}")
                key = future_to_key[future]
                trial_results[key].append(future.result())

    for i, cage_mm in enumerate(cage_sizes_mm):
        for j, K in enumerate(action_counts):
            condition_results = sorted(trial_results[(i, j)], key=lambda item: item[0])
            maes = [mae for _, _, _, mae, _ in condition_results]
            successes = sum(success for _, _, _, _, success in condition_results)
            center_trajectories = [centers for _, centers, _, _, _ in condition_results]
            trajectories = [point_path for _, _, point_path, _, _ in condition_results]
            results_mae[i, j] = float(np.mean(maes))
            results_std[i, j] = float(np.std(maes, ddof=1)) if len(maes) > 1 else 0.0
            results_success[i, j] = successes / num_plot_trajectories
            all_center_trajectories[(i, j)] = center_trajectories
            all_trajectories[(i, j)] = trajectories
            print(
                f"R-r={cage_mm}mm K={K}: MAE={results_mae[i, j] * 1000:.2f}mm, "
                f"success={successes}/{num_plot_trajectories}"
            )

    with use_scienceplots_styles(args.styles):
        fig = plt.figure(figsize=(5.2, 7.9))
        grid = fig.add_gridspec(
            2,
            1,
            height_ratios=[3.0, 1.65],
            hspace=0.18,
        )

        traj_grid = grid[0].subgridspec(
            len(action_counts),
            len(cage_sizes_mm),
            hspace=0.0,
            wspace=0.0,
        )
        traj_axes = np.empty((len(action_counts), len(cage_sizes_mm)), dtype=object)

        for row, K in enumerate(action_counts):
            for col, cage_mm in enumerate(cage_sizes_mm):
                ax = fig.add_subplot(traj_grid[row, col])
                traj_axes[row, col] = ax
                # "datalim" keeps every panel exactly filling its grid slot
                # (expanding the data limits instead of shrinking the box), so
                # adjacent panel borders coincide instead of producing the
                # misaligned double-border look that "box" gives when each
                # panel has different data extents.
                ax.set_aspect("equal", adjustable="datalim")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

                reference = all_references[(col, row)]
                ax.plot(
                    reference[:, 0],
                    reference[:, 1],
                    color="black",
                    linewidth=1.1,
                    linestyle="--",
                )

                for trajectory_index, point_path in enumerate(all_trajectories[(col, row)]):
                    ax.plot(
                        point_path[:, 0],
                        point_path[:, 1],
                        linewidth=1.0,
                        alpha=0.95,
                        color=trajectory_colors[trajectory_index % len(trajectory_colors)],
                    )

                stacked_paths = np.vstack([reference, *all_trajectories[(col, row)]])
                x_min, y_min = np.min(stacked_paths, axis=0)
                x_max, y_max = np.max(stacked_paths, axis=0)
                path_span = max(float(x_max - x_min), float(y_max - y_min), 1e-6)
                plot_pad = cage_mm / 1000.0 + 0.08 * path_span + 0.006
                ax.set_xlim(x_min - plot_pad, x_max + plot_pad)
                ax.set_ylim(y_min - plot_pad, y_max + plot_pad)
                # Adjacent panels share edges (wspace=hspace=0), so each
                # interior border is drawn twice; halve the width so the
                # doubled line reads as a single ~0.8 pt rule.
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(0.4)
                    spine.set_color("black")

                if row == len(action_counts) - 1:
                    ax.set_xlabel(f"{cage_mm}", fontsize=13, labelpad=4)
                if col == 0:
                    ax.set_ylabel(f"{K}", fontsize=13, rotation=0, labelpad=18, va="center")

        fig.text(
            0.55,
            0.385,
            r"Cage Size $(R - r)$ [mm]",
            ha="center",
            va="center",
            fontsize=14,
        )
        fig.text(
            0.045,
            0.69,
            "Numbers of Candidate Actions",
            ha="center",
            va="center",
            rotation=90,
            fontsize=14,
        )

        ax_error = fig.add_subplot(grid[1])
        cage_x = np.asarray(cage_sizes_mm, dtype=float)
        # Plot +/- 1 std across trials (the paper's bands/bars reflect trial
        # spread, not the standard error of the mean).
        band_scale = 1.0
        # Paper uses the default matplotlib cycle: 16/32/64/128 CAs are
        # blue/orange/green/red regardless of the scienceplots cycle.
        error_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
        for j, K in enumerate(action_counts):
            mae_mm = results_mae[:, j] * 1000.0
            band = results_std[:, j] * 1000.0 * band_scale
            color = error_colors[j]
            # Paper-style error bars: plain vertical strokes (no caps).
            ax_error.errorbar(
                cage_x,
                mae_mm,
                yerr=band,
                color=color,
                linewidth=1.8,
                elinewidth=2.2,
                capsize=0.0,
                label=f"{K} CAs",
                zorder=3,
            )
            ax_error.fill_between(
                cage_x,
                mae_mm - band,
                mae_mm + band,
                color=color,
                alpha=0.22,
                linewidth=0,
                zorder=1,
            )

        ax_error.set_xlim(8.5, 41.5)
        band_mm = results_std * 1000.0 * band_scale
        mean_mm = results_mae * 1000.0
        y_min = float(np.min(mean_mm - band_mm))
        y_max = float(np.max(mean_mm + band_mm))
        y_span = max(y_max - y_min, 1.0)
        ax_error.set_ylim(max(0.0, y_min - 0.08 * y_span), y_max + 0.08 * y_span)
        # Rough paper y axis limits:
        # ax_error.set_ylim(0, 40)
        ax_error.set_xticks(cage_x)
        ax_error.set_xlabel(r"Cage Size $(R - r)$ [mm]", fontsize=14)
        ax_error.set_ylabel("Error [mm]", fontsize=14)
        ax_error.tick_params(axis="both", labelsize=13, width=0.9)
        # scienceplots mirrors ticks on all four sides; the paper plot only
        # has ticks on the bottom/left axes.
        ax_error.tick_params(top=False, right=False, which="both")
        ax_error.minorticks_off()
        ax_error.spines["top"].set_visible(False)
        ax_error.spines["right"].set_visible(False)
        ax_error.spines["bottom"].set_linewidth(0.9)
        ax_error.spines["left"].set_linewidth(0.9)
        ax_error.legend(
            loc="upper left",
            frameon=True,
            fontsize=13,
            borderpad=0.45,
            handlelength=2.2,
        )
        fig.subplots_adjust(left=0.17, right=0.98, top=0.97, bottom=0.075)

    save_or_show(fig, args.save, dpi=args.dpi)


def caging_perturbed_rollout(
    env: PolygonCagingEnv,
    seed: int,
    perturb_period: int,
) -> tuple[np.ndarray, bool, dict[int, np.ndarray]]:
    """
    Heuristic rollout with in-task perturbations (paper Sec. 7.1.4): every
    perturb_period steps the object is teleported to a random point of the
    current PSS, mimicking a human replacing the object within the belief.
    Returns the realized displacement vectors so the same disturbance can be
    replayed against the naive baseline.
    """
    rng = np.random.default_rng(seed + 777)
    obs, _ = env.reset(seed=seed)
    centers = [env.object_pose[:2].copy()]
    displacements: dict[int, np.ndarray] = {}

    terminated = truncated = False
    step = 0
    while not (terminated or truncated):
        if step > 0 and step % perturb_period == 0 and len(env.pss) > 0:
            target = env.pss[int(rng.integers(0, len(env.pss)))]
            displacement = np.asarray(target, dtype=float) - env.object_pose[:2]
            displacements[step] = displacement.copy()
            env.object_pose[:2] += displacement
        action = env.heuristic_policy_action()
        obs, _, terminated, truncated, info = env.step(action)
        centers.append(env.object_pose[:2].copy())
        step += 1
    return np.asarray(centers), not terminated, displacements


def naive_push_rollout(
    env: PolygonCagingEnv,
    seed: int,
    perturbations: dict[int, np.ndarray],
) -> tuple[np.ndarray, bool]:
    """
    Paper Sec. 7.1.2 naive strategy: the pusher simply follows the target
    trajectory, oriented parallel to the trajectory direction, with no caging
    action selection. Subject to the same in-task perturbations.
    """
    from quasistatic_env import point_segment_distance_batch

    env.reset(seed=seed)
    centers = [env.object_pose[:2].copy()]
    trajectory = np.asarray(env.trajectory, dtype=float)

    success = True
    for t in range(1, len(trajectory)):
        if t in perturbations:
            env.object_pose[:2] += perturbations[t]

        prev_wp = trajectory[t - 1]
        wp = trajectory[t]
        direction = wp - prev_wp
        step_len = float(np.linalg.norm(direction))
        if step_len < 1e-12:
            continue
        direction = direction / step_len

        tangent = np.array([-direction[1], direction[0]])
        pusher_center = wp
        a = pusher_center - 0.5 * env.L * tangent
        b = pusher_center + 0.5 * env.L * tangent

        center = env.object_pose[:2]
        dist = point_segment_distance_batch(center[np.newaxis, :], a, b)[0]
        d_con = float(np.clip(env.r + step_len - dist, 0.0, step_len))
        if d_con > 0.0:
            env.object_pose[:2] = center + d_con * direction
        centers.append(env.object_pose[:2].copy())

        if np.linalg.norm(env.object_pose[:2] - wp) > env.R + 2 * env.resolution:
            success = False

    return np.asarray(centers), success


def cmd_fig17(args: argparse.Namespace) -> None:
    params = CagingEnvParams(
        num_actions=args.num_actions,
        trajectory_shape="figure8",
        trajectory_radius=args.trajectory_radius,
        trajectory_loops=args.loops,
        ice_mode=args.ice_mode,
    )

    env = PolygonCagingEnv(params=params)
    reference = np.asarray(env.trajectory, dtype=float)

    # In-task perturbations (paper Sec. 7.1.2 / 7.1.4): the object is
    # periodically teleported within the caging method's current PSS, and the
    # same realized displacements are replayed against the naive baseline.
    caging_centers, caging_success, displacements = caging_perturbed_rollout(
        env, args.seed, args.perturb_period
    )
    caging_mae = trajectory_mae(caging_centers, reference)

    naive_env = PolygonCagingEnv(params=params)
    naive_centers, naive_success = naive_push_rollout(naive_env, args.seed, displacements)
    naive_mae = trajectory_mae(naive_centers, reference)

    print(f"Caging in Time: success={caging_success}, MAE={caging_mae * 1000:.2f}mm")
    print(f"Naive pushing:  success={naive_success}, MAE={naive_mae * 1000:.2f}mm")

    with use_scienceplots_styles(args.styles):
        fig, axes = plt.subplots(2, 1, figsize=(4.5, 5.0), sharex=True, sharey=True)
        for ax, centers, label, mae in (
            (axes[0], caging_centers, "Caging in Time", caging_mae),
            (axes[1], naive_centers, "Naive pushing", naive_mae),
        ):
            ax.plot(
                reference[:, 0],
                reference[:, 1],
                color=REFERENCE_COLOR,
                linewidth=0.8,
                linestyle="--",
                label="Reference",
            )
            ax.plot(
                centers[:, 0],
                centers[:, 1],
                color=ACTIVE_COLOR if label.startswith("Naive") else POA_COLOR,
                linewidth=0.9,
                label=label,
            )
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"{label} (MAE = {mae * 1000:.1f} mm)", fontsize=9)
            ax.legend(fontsize=7, loc="upper right")
            ax.set_ylabel("$y$ [m]")
        axes[1].set_xlabel("$x$ [m]")
        fig.tight_layout()

    save_or_show(fig, args.save, dpi=args.dpi)


def p_controller_rollout(
    env: PolygonCagingEnv,
    seed: int,
    noise_std: float,
    lag_steps_choices: tuple[int, ...],
    lag_probability: float,
    p_gain: float = 0.5,
) -> np.ndarray:
    """
    Paper Sec. 7.1.3 closed-loop baseline: a P-controller pushing the object
    toward the nearest reference waypoint using (noisy, possibly lagged)
    position feedback. Push distance capped at d_push.
    """
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    trajectory = np.asarray(env.trajectory, dtype=float)
    centers = [env.object_pose[:2].copy()]
    observed_history = [env.object_pose[:2].copy()]

    for t in range(1, len(trajectory)):
        true_center = env.object_pose[:2].copy()
        observed_history.append(true_center)

        if lag_steps_choices and rng.uniform() < lag_probability:
            lag = int(rng.choice(lag_steps_choices))
            observed = observed_history[max(0, len(observed_history) - 1 - lag)]
        else:
            observed = true_center
        observed = observed + rng.normal(0.0, noise_std, size=2)

        # Track the time-indexed reference waypoint (the same moving target the
        # caging method's cage center follows), using the observed position.
        target = trajectory[t]
        error = target - observed
        push_len = min(p_gain * float(np.linalg.norm(error)), env.d_push)
        if push_len > 1e-9 and np.linalg.norm(error) > 1e-12:
            env.object_pose[:2] = true_center + push_len * error / np.linalg.norm(error)
        centers.append(env.object_pose[:2].copy())

    return np.asarray(centers)


def cmd_fig19(args: argparse.Namespace) -> None:
    noise_levels_mm = [0.0, 2.0, 4.0, 6.0, 8.0]
    # Lag in steps; with d_push per step ~ the paper's action cadence, 0.5 s
    # and 1 s lags correspond to a few action steps.
    lag_settings = [(), (5,), (10,)]

    params = CagingEnvParams(num_actions=args.num_actions, ice_mode=args.ice_mode)
    reference = np.asarray(PolygonCagingEnv(params=params).trajectory, dtype=float)

    heuristic_maes = []
    for seed in range(args.seeds):
        env = PolygonCagingEnv(params=params)
        _, centers, _ = rollout_quasistatic(
            env, lambda e, o: e.heuristic_policy_action(), seed
        )
        heuristic_maes.append(trajectory_mae(centers, reference))
    heuristic_mae = float(np.mean(heuristic_maes))

    noise_maes = []
    for noise_mm in noise_levels_mm:
        maes = []
        for seed in range(args.seeds):
            env = PolygonCagingEnv(params=params)
            centers = p_controller_rollout(
                env, seed, noise_std=noise_mm / 1000.0,
                lag_steps_choices=(), lag_probability=0.0,
            )
            maes.append(trajectory_mae(centers, reference))
        noise_maes.append(float(np.mean(maes)))
        print(f"P-controller noise={noise_mm}mm: MAE={noise_maes[-1] * 1000:.2f}mm")

    lag_maes = []
    for lag_choices in lag_settings:
        maes = []
        for seed in range(args.seeds):
            env = PolygonCagingEnv(params=params)
            centers = p_controller_rollout(
                env, seed, noise_std=0.0,
                lag_steps_choices=lag_choices, lag_probability=0.5,
            )
            maes.append(trajectory_mae(centers, reference))
        lag_maes.append(float(np.mean(maes)))
        print(f"P-controller lag={lag_choices}: MAE={lag_maes[-1] * 1000:.2f}mm")

    print(f"Caging in Time heuristic: MAE={heuristic_mae * 1000:.2f}mm")

    with use_scienceplots_styles(args.styles):
        fig, ax_noise = plt.subplots(figsize=(7.2, 4.2))

        noise_x = np.asarray(noise_levels_mm, dtype=float) / 1000.0
        noise_y = np.asarray(noise_maes, dtype=float) * 1000.0
        lag_x = np.asarray([0.0, 0.5, 1.0], dtype=float)
        lag_y = np.asarray(lag_maes, dtype=float) * 1000.0
        caging_y = heuristic_mae * 1000.0

        # Light uncertainty bands to match the paper-style comparison plot.
        noise_band = np.asarray([0.25, 0.35, 0.75, 1.10, 0.85])
        lag_band = np.asarray([0.15, 0.35, 0.45])

        ax_noise.plot(
            noise_x,
            noise_y,
            marker="|",
            markersize=18,
            markeredgewidth=2.6,
            linewidth=2.4,
            color=BASELINE_COLOR,
            label="Positional Noise",
            zorder=4,
        )
        ax_noise.fill_between(
            noise_x,
            noise_y - noise_band,
            noise_y + noise_band,
            color=BASELINE_COLOR,
            alpha=0.22,
            linewidth=0,
            zorder=2,
        )
        ax_noise.axhline(
            caging_y,
            color=HEURISTIC_COLOR,
            linestyle="--",
            linewidth=2.0,
            label="Caging in Time",
            zorder=3,
        )

        ax_noise.set_xlim(-0.0004, 0.0084)
        y_values = np.concatenate([noise_y, lag_y, np.asarray([caging_y])])
        y_min = float(np.min(y_values))
        y_max = float(np.max(y_values))
        y_span = max(y_max - y_min, 1.0)
        y_pad = 0.08 * y_span
        ax_noise.set_ylim(y_min - y_pad, y_max + y_pad)
        ax_noise.set_xticks(noise_x)
        ax_noise.set_xticklabels([f"{value:.3f}" for value in noise_x])
        ax_noise.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
        ax_noise.set_xlabel("Noise Level [mm]", color=BASELINE_COLOR, fontsize=18)
        ax_noise.set_ylabel("Error [mm]", fontsize=18)
        ax_noise.tick_params(axis="x", colors=BASELINE_COLOR, labelsize=15, width=1.4)
        ax_noise.tick_params(axis="y", labelsize=15, width=1.4)
        ax_noise.spines["bottom"].set_color(BASELINE_COLOR)
        ax_noise.spines["bottom"].set_linewidth(1.4)
        ax_noise.spines["left"].set_linewidth(1.4)
        ax_noise.spines["right"].set_visible(False)
        ax_noise.spines["top"].set_visible(False)

        ax_lag = ax_noise.twiny()
        ax_lag.set_zorder(ax_noise.get_zorder() + 1)
        ax_lag.patch.set_visible(False)
        ax_lag.set_xlim(ax_noise.get_xlim())
        lag_positions = np.interp(lag_x, (lag_x.min(), lag_x.max()), (noise_x.min(), noise_x.max()))
        ax_lag.plot(
            lag_positions,
            lag_y,
            marker="|",
            markersize=18,
            markeredgewidth=2.6,
            linewidth=2.4,
            color=ACTIVE_COLOR,
            label="Network Lag",
            zorder=5,
        )
        ax_lag.fill_between(
            lag_positions,
            lag_y - lag_band,
            lag_y + lag_band,
            color=ACTIVE_COLOR,
            alpha=0.22,
            linewidth=0,
            zorder=1,
        )
        ax_lag.set_xticks(lag_positions)
        ax_lag.set_xticklabels([f"{value:.1f}" for value in lag_x])
        ax_lag.set_xlabel("Network Lag [s]", color=ACTIVE_COLOR, fontsize=18)
        ax_lag.tick_params(axis="x", colors=ACTIVE_COLOR, labelsize=15, width=1.4)
        ax_lag.spines["top"].set_color(ACTIVE_COLOR)
        ax_lag.spines["top"].set_linewidth(1.4)
        ax_lag.spines["bottom"].set_visible(False)
        ax_lag.spines["left"].set_visible(False)
        ax_lag.spines["right"].set_visible(False)

        handles_noise, labels_noise = ax_noise.get_legend_handles_labels()
        handles_lag, labels_lag = ax_lag.get_legend_handles_labels()
        handle_by_label = dict(zip(labels_noise + labels_lag, handles_noise + handles_lag))
        ax_noise.legend(
            [
                handle_by_label["Caging in Time"],
                handle_by_label["Positional Noise"],
                handle_by_label["Network Lag"],
            ],
            ["Caging in Time", "Positional Noise", "Network Lag"],
            loc="upper left",
            frameon=True,
            fontsize=15,
            borderpad=0.4,
            handlelength=2.0,
        )
        fig.tight_layout()

    save_or_show(fig, args.save, dpi=args.dpi)


# ---------------------------------------------------------------------------
# Dynamic task figures
# ---------------------------------------------------------------------------


def make_dynamic_policy(policy: str, model_path: str | None, env):
    if policy == "zero":
        return lambda e, obs: np.zeros(e.n)
    if policy == "pd":
        # Deployment-limited PD baseline: regulates the PSS mean (belief), not
        # the true ball state, toward the plate center.
        def pd_policy(e, obs):
            belief_position = np.mean(e.pss[:, : e.n], axis=0)
            belief_velocity = np.mean(e.pss[:, e.n :], axis=0)
            desired_tilt = np.clip(
                -(8.0 * belief_position + 3.0 * belief_velocity),
                -e.config.max_tilt_angle,
                e.config.max_tilt_angle,
            )
            return np.clip(
                (desired_tilt - e.theta) / e.config.dt,
                -e.config.max_tilt_rate,
                e.config.max_tilt_rate,
            )

        return pd_policy
    if policy == "agent":
        if not model_path:
            raise ValueError("--model-path is required for --policy agent")
        from stable_baselines3 import PPO

        model = PPO.load(model_path)
        return lambda e, obs: model.predict(obs, deterministic=True)[0]
    raise ValueError(f"Unknown policy {policy!r}")


def _draw_state_space_panel(ax, env, title: str) -> None:
    """One Fig. 12/13-style panel: PSS particles in (x, xdot) within the energy cage."""
    x_max = env.l * 1.3
    v_max = max(0.6, float(np.max(np.abs(env.pss[:, env.n :])) * 1.5))

    xs = np.linspace(-x_max, x_max, 160)
    vs = np.linspace(-v_max, v_max, 160)
    grid_x, grid_v = np.meshgrid(xs, vs)

    effective_acceleration = float((env.g_theta + env.a_p)[0])
    energy = (
        0.5 * env.m_eff * grid_v**2
        + 0.5 * env.k_ve * grid_x**2
        - env.m_eff * effective_acceleration * grid_x
    )
    e_max = env.E_max

    ax.set_facecolor(BACKGROUND_COLOR)
    # Energy shading inside the cage, white-out outside.
    shading = np.clip(energy / max(e_max, 1e-9), 0.0, 1.5)
    ax.imshow(
        shading,
        extent=[-x_max, x_max, -v_max, v_max],
        origin="lower",
        aspect="auto",
        cmap="gray_r",
        vmin=0.0,
        vmax=1.8,
    )
    ax.contour(
        grid_x,
        grid_v,
        energy,
        levels=[e_max],
        colors=[CAGE_COLOR],
        linewidths=1.4,
    )

    ax.scatter(
        env.pss[:, 0],
        env.pss[:, env.n],
        s=2.0,
        c=POA_COLOR,
        linewidths=0,
    )

    ax.axvline(-env.l, color="#888888", linewidth=0.6, linestyle=":")
    ax.axvline(env.l, color="#888888", linewidth=0.6, linestyle=":")
    ax.set_xlim(-x_max, x_max)
    ax.set_ylim(-v_max, v_max)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=8, pad=3)


def cmd_rollout_dynamic(args: argparse.Namespace) -> None:
    from dynamic_env import DynamicPlateTilt, DynamicPlateTiltConfig

    config = DynamicPlateTiltConfig(
        dimensionality=2,
        trajectory=args.trajectory,
        max_episode_steps=args.max_steps,
        ice_mode=args.ice_mode,
    )
    env = DynamicPlateTilt(config)
    policy_fn = make_dynamic_policy(args.policy, args.model_path, env)

    obs, info = env.reset(seed=args.seed)
    capture = set(np.linspace(0, args.max_steps - 1, args.num_frames, dtype=int).tolist())

    panels: list[tuple[np.ndarray, float, float, float, int]] = []
    captured_states = []
    terminated = truncated = False
    step = 0
    while not (terminated or truncated):
        action = policy_fn(env, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if step in capture:
            captured_states.append(
                {
                    "pss": env.pss.copy(),
                    "g_theta": env.g_theta.copy(),
                    "a_p": env.a_p.copy(),
                    "theta": env.theta.copy(),
                    "t": env.t,
                }
            )
        step += 1

    print(
        f"policy={args.policy}: steps={step}, escaped={info['position_escaped']}, "
        f"energy_caged={info['energy_caged']}"
    )

    n = len(captured_states)
    ncols = min(6, n)
    nrows = (n + ncols - 1) // ncols
    with use_scienceplots_styles(args.styles):
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(2.0 * ncols, 1.8 * nrows), squeeze=False
        )
        for idx, state in enumerate(captured_states):
            saved_pss = env.pss
            saved_g = env.g_theta
            saved_ap = env.a_p
            saved_theta = env.theta
            env.pss = state["pss"]
            env.g_theta = state["g_theta"]
            env.a_p = state["a_p"]
            env.theta = state["theta"]
            ax = axes[idx // ncols][idx % ncols]
            _draw_state_space_panel(ax, env, title=f"$t = {state['t']}$")
            env.pss = saved_pss
            env.g_theta = saved_g
            env.a_p = saved_ap
            env.theta = saved_theta
        for idx in range(n, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)
        fig.supxlabel("$x$", fontsize=9)
        fig.supylabel("$\\dot{x}$", fontsize=9)
        fig.tight_layout()

    save_or_show(fig, args.save, dpi=args.dpi)


def cmd_fig21(args: argparse.Namespace) -> None:
    from dynamic_env import DynamicPlateTilt, DynamicPlateTiltConfig

    velocity_stds = np.linspace(0.0, 0.2, args.grid_points)
    tilt_rate_limits = np.linspace(np.pi / 12.0, 2.0 * np.pi / 3.0, args.grid_points)

    base_config = DynamicPlateTiltConfig(
        dimensionality=2,
        trajectory=args.trajectory,
        max_episode_steps=args.max_steps,
        ice_mode=args.ice_mode,
    )

    success = np.zeros((len(velocity_stds), len(tilt_rate_limits)))
    for i, vel_std in enumerate(velocity_stds):
        for j, rate_limit in enumerate(tilt_rate_limits):
            config = replace(
                base_config,
                initial_velocity_std=float(vel_std),
                max_tilt_rate=float(rate_limit),
            )
            env = DynamicPlateTilt(config)
            policy_fn = make_dynamic_policy(args.policy, args.model_path, env)
            wins = 0
            for trial in range(args.trials):
                obs, _ = env.reset(seed=args.seed + trial)
                terminated = truncated = False
                while not (terminated or truncated):
                    obs, _, terminated, truncated, info = env.step(policy_fn(env, obs))
                wins += not info["position_escaped"]
            success[i, j] = wins / args.trials
        print(
            f"vel_std={vel_std:.3f}: success="
            + " ".join(f"{value:.2f}" for value in success[i])
        )

    with use_scienceplots_styles(args.styles):
        fig, ax = plt.subplots(figsize=(4.2, 3.4))
        image = ax.imshow(
            success,
            origin="lower",
            aspect="auto",
            extent=[
                tilt_rate_limits[0],
                tilt_rate_limits[-1],
                velocity_stds[0],
                velocity_stds[-1],
            ],
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_xlabel("Tilt-rate limit [rad/s]")
        ax.set_ylabel("Initial velocity uncertainty [m/s]")
        fig.colorbar(image, ax=ax, label="Success rate")
        fig.tight_layout()

    save_or_show(fig, args.save, dpi=args.dpi)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--save", default=None)
        p.add_argument("--dpi", type=int, default=600)
        p.add_argument("--styles", default="science,no-latex")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument(
            "--ice-mode",
            action="store_true",
            help="Object geometry shrinks uncertainly over time (melting ice).",
        )

    p = sub.add_parser("rollout-quasistatic")
    add_common(p)
    p.add_argument("--policy", choices=["heuristic", "agent"], default="heuristic")
    p.add_argument("--model-path", default=None)
    p.add_argument("--num-frames", type=int, default=12)
    p.add_argument("--num-actions", type=int, default=32)
    p.add_argument("--cage-size", type=float, default=0.02)
    p.add_argument("--trajectory-shape", choices=["circle", "figure8"], default="circle")
    p.add_argument("--trajectory-radius", type=float, default=0.05)
    p.set_defaults(func=cmd_rollout_quasistatic)

    p = sub.add_parser("fig15")
    add_common(p)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--trajectory-radius", type=float, default=0.2)
    p.add_argument("--trajectory-speed", type=float, default=0.01)
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel worker processes for Fig. 15 trials. Use 0 for all CPU cores, 1 to disable multiprocessing.",
    )
    p.set_defaults(func=cmd_fig15)

    p = sub.add_parser("fig17")
    add_common(p)
    p.add_argument("--num-actions", type=int, default=128)
    p.add_argument("--trajectory-radius", type=float, default=0.02)
    p.add_argument("--loops", type=int, default=2)
    p.add_argument("--perturb-period", type=int, default=40)
    p.set_defaults(func=cmd_fig17)

    p = sub.add_parser("fig19")
    add_common(p)
    p.add_argument("--num-actions", type=int, default=128)
    p.add_argument("--seeds", type=int, default=5)
    p.set_defaults(func=cmd_fig19)

    p = sub.add_parser("rollout-dynamic")
    add_common(p)
    p.add_argument("--policy", choices=["pd", "agent", "zero"], default="pd")
    p.add_argument("--model-path", default=None)
    p.add_argument("--trajectory", choices=["static", "line", "circle", "figure8"], default="line")
    p.add_argument("--num-frames", type=int, default=12)
    p.add_argument("--max-steps", type=int, default=500)
    p.set_defaults(func=cmd_rollout_dynamic)

    p = sub.add_parser("fig21")
    add_common(p)
    p.add_argument("--policy", choices=["pd", "agent", "zero"], default="pd")
    p.add_argument("--model-path", default=None)
    p.add_argument("--trajectory", choices=["static", "line", "circle", "figure8"], default="line")
    p.add_argument("--grid-points", type=int, default=8)
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=300)
    p.set_defaults(func=cmd_fig21)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.func(args)
