from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from math import pi
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True, kw_only=True)
class DynamicPlateTiltConfig:
    # The paper uses a plate of dimension n embedded in an (n + 1)-D workspace.
    # dimensionality=2 -> 1D track plate. dimensionality=3 -> 2D square plate.
    dimensionality: Literal[2, 3] = 2
    plate_size: float = 0.16
    virtual_elastic_stiffness: float = 1.0
    boundary_energy_method: Literal["corners", "edges"] = "corners"

    object_mass: float = 0.057
    object_radius: float = 0.033
    rolling_friction: float = 0.05
    moment_of_inertia: float | None = None

    gravity: float = 9.81
    dt: float = 0.02
    max_episode_steps: int = 500

    max_tilt_angle: float = pi / 9.0
    max_tilt_rate: float = pi / 3.0

    # Preset translational plate trajectory, as in the paper. The RL action only
    # controls tilt-rate; translational motion is externally prescribed.
    trajectory: Literal["static", "circle", "line", "figure8"] = "line"
    trajectory_radius: float = 0.05
    trajectory_speed: float = 0.5
    line_amplitude: float = 0.05

    # Initial-state randomization.
    initial_position_std: float = 0.01
    initial_velocity_std: float = 0.02

    # PSS approximation for the RL observation and energy-cage checks.
    pss_sample_count: int = 256
    pss_position_std: float = 0.004
    pss_velocity_std: float = 0.02
    pss_acceleration_std: float = 0.15
    pss_probability_threshold: float = 1e-6
    pss_escape_fraction_threshold: float = 0.05

    pss_energy_percentile: float = 95.0
    terminate_on_pss_energy_violation: bool = False
    terminate_on_pss_position_escape: bool = False

    # Observation normalization scales.
    velocity_scale: float = 1.0
    acceleration_scale: float = 9.81

    # Limited-observation policy inputs.
    include_pss_particles_in_obs: bool = True
    trajectory_preview_step_offsets: tuple[int, ...] = (1, 5, 10, 25)

    # Reward weights.
    energy_margin_reward_weight: float = 1.0
    position_penalty_weight: float = 2.0
    velocity_penalty_weight: float = 0.1
    tilt_penalty_weight: float = 0.02
    action_penalty_weight: float = 0.001
    failure_penalty: float = 20.0
    survival_reward: float = 0.05

    # Rendering.
    render_pss: bool = False

    ice_mode: bool = False
    ice_melt_rate: float = 0.001


class DynamicPlateTilt(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, config: DynamicPlateTiltConfig, render_mode: str | None = None):
        if config.dimensionality not in {2, 3}:
            raise ValueError("Dimensionality must be 2 or 3")
        if config.plate_size <= 0.0:
            raise ValueError("Plate size must be positive")
        if config.virtual_elastic_stiffness <= 0.0:
            raise ValueError("Virtual elastic stiffness must be positive")
        if config.boundary_energy_method not in {"corners", "edges"}:
            raise ValueError("Boundary energy method must be 'corners' or 'edges'")
        if config.dt <= 0.0:
            raise ValueError("dt must be positive")
        if config.pss_sample_count <= 0:
            raise ValueError("pss_sample_count must be positive")
        if not (0.0 <= config.pss_energy_percentile <= 100.0):
            raise ValueError("pss_energy_percentile must be between 0 and 100")
        if config.velocity_scale <= 0.0:
            raise ValueError("velocity_scale must be positive")
        if config.acceleration_scale <= 0.0:
            raise ValueError("acceleration_scale must be positive")
        if any(offset <= 0 for offset in config.trajectory_preview_step_offsets):
            raise ValueError("trajectory_preview_step_offsets must contain only positive step offsets")

        self.config = config
        self.render_mode = render_mode

        self.n = config.dimensionality - 1
        self.l = 0.5 * config.plate_size
        self.k_ve = config.virtual_elastic_stiffness
        self.m = config.object_mass
        self.r_b = config.object_radius
        self.mu_r = config.rolling_friction
        self.I_b = (
            config.moment_of_inertia
            if config.moment_of_inertia is not None
            else 0.4 * config.object_mass * config.object_radius * config.object_radius
        )

        self.action_space = spaces.Box(
            low=-config.max_tilt_rate,
            high=config.max_tilt_rate,
            shape=(self.n,),
            dtype=np.float32,
        )

        # Limited-observation policy input:
        # known plate/trajectory state + known future trajectory preview + PSS.
        # No true ball x, xdot, or true-state energy is exposed to the agent.
        obs_dim = 10 * self.n + 8
        obs_dim += 3 * self.n * len(config.trajectory_preview_step_offsets)
        if config.include_pss_particles_in_obs:
            obs_dim += 2 * self.n * config.pss_sample_count
        high = np.full(obs_dim, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.t = 0
        self.x = np.zeros(self.n, dtype=float)
        self.xdot = np.zeros(self.n, dtype=float)
        self.theta = np.zeros(self.n, dtype=float)
        self.plate_position = np.zeros(self.n, dtype=float)
        self.plate_velocity = np.zeros(self.n, dtype=float)
        self.plate_acceleration = np.zeros(self.n, dtype=float)
        self.g_theta = np.zeros(self.n, dtype=float)
        self.a_p = np.zeros(self.n, dtype=float)
        self.pss = np.zeros((self.config.pss_sample_count, 2 * self.n + 1), dtype=float)
        self.pss_weights = np.full(self.config.pss_sample_count, 1.0 / self.config.pss_sample_count)
        self._render_bounds = self._compute_render_bounds()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}

        self.t = 0
        self.x = np.asarray(
            options.get(
                "x0",
                self.np_random.normal(0.0, self.config.initial_position_std, size=self.n),
            ),
            dtype=float,
        )
        self.xdot = np.asarray(
            options.get(
                "xdot0",
                self.np_random.normal(0.0, self.config.initial_velocity_std, size=self.n),
            ),
            dtype=float,
        )
        self.theta = np.asarray(options.get("theta0", np.zeros(self.n)), dtype=float)
        self.theta = np.clip(self.theta, -self.config.max_tilt_angle, self.config.max_tilt_angle)

        self._update_plate_trajectory_terms()
        self._update_frame_terms()
        self._initialize_pss()

        return self._get_obs(), self._get_info()

    @cached_property
    def m_eff(self) -> float:
        """The effective rolling mass of the object in the plate frame."""
        return self.m + self.I_b / (self.r_b * self.r_b)

    @property
    def E_k(self) -> float:
        """The kinetic energy of the object in plate frame."""
        return self.kinetic_energy(self.xdot)

    @property
    def E_ve(self) -> float:
        """The virtual elastic energy of the object in plate frame."""
        return self.virtual_elastic_energy(self.x)

    @property
    def E_p(self) -> float:
        """The potential energy of the object in plate frame."""
        return self.potential_energy(self.x, self.g_theta, self.a_p)

    @property
    def E_max(self) -> float:
        """The maximum allowed static boundary energy of the object in plate frame."""
        return self.max_boundary_energy(self.g_theta, self.a_p)

    @property
    def E(self) -> float:
        """The total energy of the object in plate frame."""
        return self.total_energy(self.x, self.xdot, self.g_theta, self.a_p)

    def kinetic_energy(self, xdot: np.ndarray) -> float:
        """Compute kinetic energy for a velocity in plate coordinates."""
        xdot = np.asarray(xdot, dtype=float)
        self._check_vector_shape("xdot", xdot)
        return 0.5 * self.m_eff * float(np.dot(xdot, xdot))

    def virtual_elastic_energy(self, x: np.ndarray) -> float:
        """Compute the virtual elastic energy used by the energy cage."""
        x = np.asarray(x, dtype=float)
        self._check_vector_shape("x", x)
        return 0.5 * self.k_ve * float(np.dot(x, x))

    def potential_energy(self, x: np.ndarray, g_theta: np.ndarray, a_p: np.ndarray) -> float:
        """Compute potential energy from plate-frame gravity and pseudo-acceleration."""
        x = np.asarray(x, dtype=float)
        g_theta = np.asarray(g_theta, dtype=float)
        a_p = np.asarray(a_p, dtype=float)
        self._check_vector_shape("x", x)
        self._check_vector_shape("g_theta", g_theta)
        self._check_vector_shape("a_p", a_p)

        effective_acceleration = g_theta + a_p
        return -self.m_eff * float(np.dot(effective_acceleration, x))

    def max_boundary_energy(self, g_theta: np.ndarray, a_p: np.ndarray) -> float:
        """
        Compute E_max as the minimum static escape energy on the plate boundary.

        boundary_energy_method="corners" evaluates the paper-likely corner/end-point
        approximation. boundary_energy_method="edges" evaluates the conservative
        minimum over the entire square boundary.
        """
        g_theta = np.asarray(g_theta, dtype=float)
        a_p = np.asarray(a_p, dtype=float)
        self._check_vector_shape("g_theta", g_theta)
        self._check_vector_shape("a_p", a_p)

        if self.config.boundary_energy_method == "corners":
            return self.corner_boundary_energy(g_theta, a_p)
        return self.edge_boundary_energy(g_theta, a_p)

    def corner_boundary_energy(self, g_theta: np.ndarray, a_p: np.ndarray) -> float:
        """Compute E_max using only boundary endpoints/corners."""
        if self.n == 1:
            boundary_points = [
                np.array([-self.l], dtype=float),
                np.array([self.l], dtype=float),
            ]
        else:
            boundary_points = [
                np.array([sx * self.l, sy * self.l], dtype=float)
                for sx in (-1.0, 1.0)
                for sy in (-1.0, 1.0)
            ]

        return min(
            self.virtual_elastic_energy(x_corner)
            + self.potential_energy(x_corner, g_theta, a_p)
            for x_corner in boundary_points
        )

    def edge_boundary_energy(self, g_theta: np.ndarray, a_p: np.ndarray) -> float:
        """Compute conservative E_max by minimizing over the full plate boundary."""
        if self.n == 1:
            return self.corner_boundary_energy(g_theta, a_p)

        effective_acceleration = g_theta + a_p
        unconstrained_minimum = self.m_eff * effective_acceleration / self.k_ve

        boundary_points: list[np.ndarray] = []
        for axis in range(self.n):
            for side in (-self.l, self.l):
                x_edge = np.clip(unconstrained_minimum, -self.l, self.l)
                x_edge[axis] = side
                boundary_points.append(x_edge.copy())

        return min(
            self.virtual_elastic_energy(x_edge)
            + self.potential_energy(x_edge, g_theta, a_p)
            for x_edge in boundary_points
        )

    def energy_terms(
        self,
        x: np.ndarray | None = None,
        xdot: np.ndarray | None = None,
        g_theta: np.ndarray | None = None,
        a_p: np.ndarray | None = None,
    ) -> tuple[float, float, float]:
        """Compute kinetic, virtual elastic, and potential energy."""
        if x is None:
            x = self.x
        if xdot is None:
            xdot = self.xdot
        if g_theta is None:
            g_theta = self.g_theta
        if a_p is None:
            a_p = self.a_p

        return (
            self.kinetic_energy(xdot),
            self.virtual_elastic_energy(x),
            self.potential_energy(x, g_theta, a_p),
        )

    def total_energy(
        self,
        x: np.ndarray | None = None,
        xdot: np.ndarray | None = None,
        g_theta: np.ndarray | None = None,
        a_p: np.ndarray | None = None,
    ) -> float:
        return float(sum(self.energy_terms(x=x, xdot=xdot, g_theta=g_theta, a_p=a_p)))

    def is_energy_caged(self) -> bool:
        """Check if the realized object state is energy caged."""
        return self.E <= self.E_max

    def step(self, u: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        u = np.asarray(u, dtype=float)
        if u.shape == () and self.n == 1:
            u = np.array([float(u)], dtype=float)
        self._check_vector_shape("u", u)
        u = np.clip(u, -self.config.max_tilt_rate, self.config.max_tilt_rate)

        previous_theta = self.theta.copy()
        self.theta = np.clip(
            self.theta + self.config.dt * u,
            -self.config.max_tilt_angle,
            self.config.max_tilt_angle,
        )

        self.t += 1
        self._update_plate_trajectory_terms()
        self._update_frame_terms()

        if self.config.ice_mode:
            self.r_b *= (1.0 - float(self.np_random.uniform(0.0, self.config.ice_melt_rate)))

        acceleration = self.ball_acceleration(self.xdot, self.g_theta, self.a_p)
        self.x = self.x + self.config.dt * self.xdot
        self.xdot = self.xdot + self.config.dt * acceleration

        self._propagate_pss()

        info = self._get_info()
        terminated = bool(
            info["position_escaped"]
            or (self.config.terminate_on_pss_position_escape and info["pss_position_escaped"])
            or (self.config.terminate_on_pss_energy_violation and info["pss_energy_violated"])
        )
        truncated = self.t >= self.config.max_episode_steps
        reward = self._compute_reward(u=u, previous_theta=previous_theta, info=info)

        return self._get_obs(), reward, terminated, truncated, info

    def ball_acceleration(
        self,
        xdot: np.ndarray,
        g_theta: np.ndarray,
        a_p: np.ndarray,
        acceleration_noise: np.ndarray | None = None,
    ) -> np.ndarray:
        """Mean rolling-ball acceleration in the non-inertial plate frame."""
        xdot = np.asarray(xdot, dtype=float)
        g_theta = np.asarray(g_theta, dtype=float)
        a_p = np.asarray(a_p, dtype=float)
        self._check_vector_shape("xdot", xdot)
        self._check_vector_shape("g_theta", g_theta)
        self._check_vector_shape("a_p", a_p)

        rolling_gain = self.m / self.m_eff
        acceleration = rolling_gain * (g_theta + a_p) - self.mu_r * xdot
        if acceleration_noise is not None:
            acceleration = acceleration + acceleration_noise
        return acceleration

    def render(self) -> np.ndarray:
        """Return an RGB render in world frame with fixed bounds over the full trajectory."""
        if self.n == 1:
            return self._render_1d_scene()
        return self._render_3d_scene()

    def _render_1d_scene(self) -> np.ndarray:
        image_size = 512
        image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        image[:, :] = np.array([18, 18, 18], dtype=np.uint8)

        x_min, x_max, y_min, y_max = self._render_bounds
        span = max(x_max - x_min, y_max - y_min, 1e-9)
        pad_px = 28
        scale = (image_size - 2 * pad_px) / span

        def to_px(point: np.ndarray) -> tuple[int, int]:
            point = np.asarray(point, dtype=float)
            px = pad_px + scale * (point[0] - x_min)
            py = image_size - pad_px - scale * (point[1] - y_min)
            return int(round(px)), int(round(py))

        self._draw_world_trajectory(image, to_px)

        center = np.array([self.plate_position[0], 0.0], dtype=float)
        tangent = np.array([np.cos(self.theta[0]), np.sin(self.theta[0])], dtype=float)
        normal = np.array([-tangent[1], tangent[0]], dtype=float)

        endpoint_a = center - self.l * tangent
        endpoint_b = center + self.l * tangent
        self._draw_line(image, to_px(endpoint_a), to_px(endpoint_b), color=(90, 190, 90))
        self._draw_disk(image, to_px(endpoint_a), 4, color=(90, 190, 90))
        self._draw_disk(image, to_px(endpoint_b), 4, color=(90, 190, 90))

        if self.config.render_pss:
            for pss_state in self.pss:
                particle_world = center + pss_state[0] * tangent
                self._draw_disk(image, to_px(particle_world), 1, color=(90, 90, 150))

        ball_world = center + self.x[0] * tangent + self.r_b * normal
        radius_px = max(3, int(round(scale * self.r_b)))
        self._draw_disk(image, to_px(ball_world), radius_px, color=(225, 225, 225))
        return image

    def _render_3d_scene(self) -> np.ndarray:
        image_size = 720
        image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        image[:, :] = np.array([18, 18, 18], dtype=np.uint8)

        project = self._world3_to_px_projector(image_size)
        self._draw_3d_floor_grid(image, project)
        self._draw_world_trajectory(image, None)

        center3 = np.array([self.plate_position[0], self.plate_position[1], 0.0], dtype=float)
        rotation3 = self._plate_rotation_matrix_3d()

        corners_local = np.array(
            [
                [-self.l, -self.l, 0.0],
                [self.l, -self.l, 0.0],
                [self.l, self.l, 0.0],
                [-self.l, self.l, 0.0],
            ],
            dtype=float,
        )
        corners_world = center3[np.newaxis, :] + corners_local @ rotation3.T
        pts = np.array([project(corner) for corner in corners_world], dtype=np.int32)
        pts = self._sort_polygon_points(pts)
        self._fill_polygon(image, pts, color=(34, 82, 48))
        self._draw_polyline(image, pts, color=(95, 210, 105), closed=True)

        # Draw plate normal so tilt is visually unambiguous.
        plate_normal = rotation3 @ np.array([0.0, 0.0, 1.0], dtype=float)
        normal_start = center3
        normal_end = center3 + 0.06 * plate_normal
        self._draw_line(image, project(normal_start), project(normal_end), color=(240, 180, 80))

        if self.config.render_pss:
            for pss_state in self.pss:
                particle_local = np.array([pss_state[0], pss_state[1], 0.002], dtype=float)
                particle_world = center3 + rotation3 @ particle_local
                self._draw_disk(image, project(particle_world), 1, color=(90, 90, 150))

        ball_local = np.array([self.x[0], self.x[1], self.r_b], dtype=float)
        ball_world = center3 + rotation3 @ ball_local
        radius_px = max(5, int(round(self._render_scale(image_size) * self.r_b)))
        self._draw_disk(image, project(ball_world), radius_px, color=(225, 225, 225))
        return image

    def _draw_world_trajectory(self, image: np.ndarray, to_px) -> None:
        samples = max(2, self.config.max_episode_steps + 1)
        points = []

        if self.n == 2:
            project = self._world3_to_px_projector(image.shape[0])
            for step in range(samples):
                position, _, _ = self.plate_trajectory(step * self.config.dt)
                world_point = np.array([position[0], position[1], -0.003], dtype=float)
                points.append(project(world_point))
        else:
            if to_px is None:
                raise ValueError("to_px is required for 1D rendering")
            for step in range(samples):
                position, _, _ = self.plate_trajectory(step * self.config.dt)
                world_point = np.array([position[0], 0.0], dtype=float)
                points.append(to_px(world_point))

        pts = np.asarray(points, dtype=np.int32)
        self._draw_polyline(image, pts, color=(145, 70, 70), closed=False)

    def _plate_rotation_matrix_3d(self) -> np.ndarray:
        # theta[0] is pitch about world y, theta[1] is roll about world x.
        pitch = float(self.theta[0])
        roll = float(self.theta[1])

        cp = np.cos(pitch)
        sp = np.sin(pitch)
        cr = np.cos(roll)
        sr = np.sin(roll)

        rotation_y = np.array(
            [
                [cp, 0.0, sp],
                [0.0, 1.0, 0.0],
                [-sp, 0.0, cp],
            ],
            dtype=float,
        )
        rotation_x = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, cr, -sr],
                [0.0, sr, cr],
            ],
            dtype=float,
        )
        return rotation_x @ rotation_y

    def _world3_to_px_projector(self, image_size: int):
        x_min, x_max, y_min, y_max = self._render_bounds
        span = max(x_max - x_min, y_max - y_min, 1e-9)
        target = np.array(
            [
                0.5 * (x_min + x_max),
                0.5 * (y_min + y_max),
                0.0,
            ],
            dtype=float,
        )
        camera = target + np.array([0.95 * span, -1.35 * span, 1.25 * span], dtype=float)

        forward = target - camera
        forward = forward / max(np.linalg.norm(forward), 1e-12)
        world_up = np.array([0.0, 0.0, 1.0], dtype=float)
        right = np.cross(forward, world_up)
        right = right / max(np.linalg.norm(right), 1e-12)
        up = np.cross(right, forward)
        focal_length = 0.85 * image_size

        def project(point: np.ndarray) -> tuple[int, int]:
            rel = np.asarray(point, dtype=float) - camera
            depth = max(float(np.dot(rel, forward)), 1e-6)
            x_cam = float(np.dot(rel, right))
            y_cam = float(np.dot(rel, up))
            px = 0.5 * image_size + focal_length * x_cam / depth
            py = 0.55 * image_size - focal_length * y_cam / depth
            return int(round(px)), int(round(py))

        return project

    def _draw_3d_floor_grid(self, image: np.ndarray, project) -> None:
        x_min, x_max, y_min, y_max = self._render_bounds
        grid_count = 6
        for x in np.linspace(x_min, x_max, grid_count):
            p0 = np.array([x, y_min, -0.006], dtype=float)
            p1 = np.array([x, y_max, -0.006], dtype=float)
            self._draw_line(image, project(p0), project(p1), color=(42, 42, 42))
        for y in np.linspace(y_min, y_max, grid_count):
            p0 = np.array([x_min, y, -0.006], dtype=float)
            p1 = np.array([x_max, y, -0.006], dtype=float)
            self._draw_line(image, project(p0), project(p1), color=(42, 42, 42))

    @staticmethod
    def _sort_polygon_points(points: np.ndarray) -> np.ndarray:
        center = np.mean(points, axis=0)
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        return points[np.argsort(angles)]

    def _compute_render_bounds(self) -> tuple[float, float, float, float]:
        samples = max(2, self.config.max_episode_steps + 1)
        points: list[np.ndarray] = []
        extent = self.l + self.r_b + 0.03

        for step in range(samples):
            position, _, _ = self.plate_trajectory(step * self.config.dt)
            if self.n == 1:
                center = np.array([position[0], 0.0], dtype=float)
            else:
                center = position[:2]
            points.extend(
                [
                    center + np.array([-extent, -extent], dtype=float),
                    center + np.array([extent, extent], dtype=float),
                ]
            )

        pts = np.asarray(points, dtype=float)
        x_min, y_min = np.min(pts, axis=0)
        x_max, y_max = np.max(pts, axis=0)
        span = max(x_max - x_min, y_max - y_min, 1e-9)
        x_mid = 0.5 * (x_min + x_max)
        y_mid = 0.5 * (y_min + y_max)
        half_span = 0.5 * span
        return x_mid - half_span, x_mid + half_span, y_mid - half_span, y_mid + half_span

    def _render_scale(self, image_size: int = 512) -> float:
        x_min, x_max, y_min, y_max = self._render_bounds
        span = max(x_max - x_min, y_max - y_min, 1e-9)
        return (image_size - 2 * 28) / span

    def close(self) -> None:
        pass

    def _get_obs(self) -> np.ndarray:
        """Return limited-observation policy input.

        The policy does not observe the true ball state. It receives known plate
        state, a known future trajectory preview, and the PSS/belief.
        """
        positions = self.pss[:, : self.n]
        velocities = self.pss[:, self.n : 2 * self.n]

        pss_mean_position = np.mean(positions, axis=0)
        pss_std_position = np.std(positions, axis=0)
        pss_mean_velocity = np.mean(velocities, axis=0)
        pss_std_velocity = np.std(velocities, axis=0)

        avg_pss_energy = self.average_pss_energy()
        percentile_pss_energy = self.percentile_pss_energy(self.config.pss_energy_percentile)
        max_pss_energy = self.max_pss_energy()
        energy_scale = max(abs(self.E_max), 1e-6)
        pss_escape_fraction = float(np.mean(np.any(np.abs(positions) > self.l, axis=1)))

        obs_parts = [
            # Known current robot / plate state.
            self.theta / self.config.max_tilt_angle,
            self.plate_position / max(self.l, 1e-6),
            self.plate_velocity / self.config.velocity_scale,
            self.plate_acceleration / self.config.acceleration_scale,
            self.g_theta / self.config.acceleration_scale,
            self.a_p / self.config.acceleration_scale,

            # Known future trajectory preview. This is allowed because the plate
            # translation trajectory is prescribed, not inferred from hidden ball state.
            self._trajectory_preview_observation(),

            # PSS / belief summary.
            pss_mean_position / self.l,
            pss_std_position / self.l,
            pss_mean_velocity / self.config.velocity_scale,
            pss_std_velocity / self.config.velocity_scale,

            # PSS energy and risk summary.
            np.array(
                [
                    avg_pss_energy / energy_scale,
                    percentile_pss_energy / energy_scale,
                    max_pss_energy / energy_scale,
                    (self.E_max - avg_pss_energy) / energy_scale,
                    (self.E_max - percentile_pss_energy) / energy_scale,
                    (self.E_max - max_pss_energy) / energy_scale,
                    pss_escape_fraction,
                    self.t / max(1, self.config.max_episode_steps),
                ],
                dtype=float,
            ),
        ]

        if self.config.include_pss_particles_in_obs:
            normalized_particles = np.column_stack(
                (
                    positions / self.l,
                    velocities / self.config.velocity_scale,
                )
            )
            obs_parts.append(normalized_particles.reshape(-1))

        obs = np.concatenate(obs_parts)
        return obs.astype(np.float32)

    def _trajectory_preview_observation(self) -> np.ndarray:
        preview_parts = []
        for step_offset in self.config.trajectory_preview_step_offsets:
            preview_time = (self.t + step_offset) * self.config.dt
            position, velocity, acceleration = self.plate_trajectory(preview_time)
            preview_parts.extend(
                [
                    position / max(self.l, 1e-6),
                    velocity / self.config.velocity_scale,
                    acceleration / self.config.acceleration_scale,
                ]
            )
        if not preview_parts:
            return np.empty(0, dtype=float)
        return np.concatenate(preview_parts).astype(float)

    def _get_info(self) -> dict[str, Any]:
        max_pss_energy = self.max_pss_energy()
        percentile_pss_energy = self.percentile_pss_energy(self.config.pss_energy_percentile)
        escaped = bool(np.any(np.abs(self.x) > self.l))
        pss_escape_fraction = float(np.mean(np.any(np.abs(self.pss[:, : self.n]) > self.l, axis=1)))
        pss_position_escaped = pss_escape_fraction > self.config.pss_escape_fraction_threshold
        pss_energy_violated = percentile_pss_energy > self.E_max
        return {
            "t": self.t,
            "x": self.x.copy(),
            "xdot": self.xdot.copy(),
            "theta": self.theta.copy(),
            "plate_position": self.plate_position.copy(),
            "plate_velocity": self.plate_velocity.copy(),
            "plate_acceleration": self.plate_acceleration.copy(),
            "g_theta": self.g_theta.copy(),
            "a_p": self.a_p.copy(),
            "E": self.E,
            "E_max": self.E_max,
            "energy_margin": self.E_max - self.E,
            "max_pss_energy": max_pss_energy,
            "percentile_pss_energy": percentile_pss_energy,
            "pss_energy_percentile": self.config.pss_energy_percentile,
            "pss_energy_margin": self.E_max - percentile_pss_energy,
            "max_pss_energy_margin": self.E_max - max_pss_energy,
            "energy_caged": self.is_energy_caged(),
            "pss_energy_caged": bool(percentile_pss_energy <= self.E_max),
            "escaped": escaped or pss_position_escaped,
            "position_escaped": escaped,
            "pss_position_escaped": pss_position_escaped,
            "pss_escape_fraction": pss_escape_fraction,
            "pss_energy_violated": bool(pss_energy_violated),
        }

    def _compute_reward(self, u: np.ndarray, previous_theta: np.ndarray, info: dict[str, Any]) -> float:
        energy_margin_scale = max(abs(info["E_max"]), 1e-6)
        normalized_pss_margin = info["pss_energy_margin"] / energy_margin_scale

        reward = self.config.survival_reward
        reward += self.config.energy_margin_reward_weight * float(np.tanh(normalized_pss_margin))
        reward -= self.config.position_penalty_weight * float(np.dot(self.x, self.x))
        reward -= self.config.velocity_penalty_weight * float(np.dot(self.xdot, self.xdot))
        reward -= self.config.tilt_penalty_weight * float(np.dot(self.theta, self.theta))
        reward -= self.config.action_penalty_weight * float(np.dot(u, u))
        reward -= 0.01 * float(np.dot(self.theta - previous_theta, self.theta - previous_theta))

        if info["position_escaped"]:
            reward -= self.config.failure_penalty
        if info["pss_position_escaped"]:
            reward -= 0.25 * self.config.failure_penalty
        if info["pss_energy_violated"]:
            reward -= 0.25 * self.config.failure_penalty
        return float(reward)

    def _initialize_pss(self) -> None:
        position_noise = self.np_random.normal(
            0.0,
            self.config.pss_position_std,
            size=(self.config.pss_sample_count, self.n),
        )
        velocity_noise = self.np_random.normal(
            0.0,
            self.config.pss_velocity_std,
            size=(self.config.pss_sample_count, self.n),
        )
        self.pss = np.column_stack((self.x + position_noise, self.xdot + velocity_noise, np.full(self.config.pss_sample_count, self.r_b)))
        self.pss_weights = np.full(self.config.pss_sample_count, 1.0 / self.config.pss_sample_count)

    def _propagate_pss(self) -> None:
        positions = self.pss[:, : self.n]
        velocities = self.pss[:, self.n : 2 * self.n]
        radii = self.pss[:, 2 * self.n]
        
        if self.config.ice_mode:
            melt_factors = 1.0 - self.np_random.uniform(0.0, self.config.ice_melt_rate, size=len(radii))
            radii = radii * melt_factors

        acceleration_noise = self.np_random.normal(
            0.0,
            self.config.pss_acceleration_std,
            size=(self.config.pss_sample_count, self.n),
        )

        rolling_gain = self.m / self.m_eff
        accelerations = rolling_gain * (self.g_theta + self.a_p)[np.newaxis, :] - self.mu_r * velocities
        accelerations = accelerations + acceleration_noise
        positions = positions + self.config.dt * velocities
        velocities = velocities + self.config.dt * accelerations
        self.pss = np.column_stack((positions, velocities, radii))

        # The paper normalizes the discretized probability distribution after
        # propagation. Here the particle approximation keeps uniform weights.
        self.pss_weights = np.full(self.config.pss_sample_count, 1.0 / self.config.pss_sample_count)

    def max_pss_energy(self) -> float:
        energies = [
            self.total_energy(
                x=state[: self.n],
                xdot=state[self.n : 2 * self.n],
                g_theta=self.g_theta,
                a_p=self.a_p,
            )
            for state in self.pss
        ]
        return float(np.max(energies))

    def percentile_pss_energy(self, percentile: float) -> float:
        energies = np.array(
            [
                self.total_energy(
                    x=state[: self.n],
                    xdot=state[self.n : 2 * self.n],
                    g_theta=self.g_theta,
                    a_p=self.a_p,
                )
                for state in self.pss
            ],
            dtype=float,
        )
        return float(np.percentile(energies, percentile))

    def average_pss_energy(self) -> float:
        energies = np.array(
            [
                self.total_energy(
                    x=state[: self.n],
                    xdot=state[self.n : 2 * self.n],
                    g_theta=self.g_theta,
                    a_p=self.a_p,
                )
                for state in self.pss
            ],
            dtype=float,
        )
        return float(np.dot(self.pss_weights, energies))

    def pss_entropy(self) -> float:
        weights = self.pss_weights[self.pss_weights > self.config.pss_probability_threshold]
        return float(-np.sum(weights * np.log(weights)))

    def _update_frame_terms(self) -> None:
        # Tangential component of gravity in the tilted plate frame. For the small
        # angles used in the paper's plate task, sin(theta) is the relevant slope term.
        self.g_theta = self.config.gravity * np.sin(self.theta)

        # Non-inertial pseudo-acceleration in the plate frame induced by the
        # prescribed translational acceleration of the plate. Only the
        # component tangential to the tilted plate acts on the rolling ball
        # (cos(theta) projection, cf. Eq. 38 in the paper).
        self.a_p = -self.plate_acceleration * np.cos(self.theta)

    def _update_plate_trajectory_terms(self) -> None:
        time = self.t * self.config.dt
        position, velocity, acceleration = self.plate_trajectory(time)
        self.plate_position = position
        self.plate_velocity = velocity
        self.plate_acceleration = acceleration

    def plate_trajectory(self, time: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.config.trajectory == "static":
            return (
                np.zeros(self.n, dtype=float),
                np.zeros(self.n, dtype=float),
                np.zeros(self.n, dtype=float),
            )

        omega = self.config.trajectory_speed
        if self.config.trajectory == "line" or self.n == 1:
            amp = self.config.line_amplitude
            position = np.array([amp * np.sin(omega * time)], dtype=float)
            velocity = np.array([amp * omega * np.cos(omega * time)], dtype=float)
            acceleration = np.array([-amp * omega * omega * np.sin(omega * time)], dtype=float)
            if self.n == 2 and self.config.trajectory == "line":
                position = np.array([position[0], 0.0], dtype=float)
                velocity = np.array([velocity[0], 0.0], dtype=float)
                acceleration = np.array([acceleration[0], 0.0], dtype=float)
            return position, velocity, acceleration

        radius = self.config.trajectory_radius
        if self.config.trajectory == "circle":
            position = radius * np.array([np.cos(omega * time), np.sin(omega * time)], dtype=float)
            velocity = radius * omega * np.array([-np.sin(omega * time), np.cos(omega * time)], dtype=float)
            acceleration = -radius * omega * omega * np.array([np.cos(omega * time), np.sin(omega * time)], dtype=float)
            return position, velocity, acceleration

        if self.config.trajectory == "figure8":
            position = radius * np.array(
                [np.sin(omega * time), 0.5 * np.sin(2.0 * omega * time)],
                dtype=float,
            )
            velocity = radius * np.array(
                [omega * np.cos(omega * time), omega * np.cos(2.0 * omega * time)],
                dtype=float,
            )
            acceleration = radius * np.array(
                [-(omega**2) * np.sin(omega * time), -2.0 * (omega**2) * np.sin(2.0 * omega * time)],
                dtype=float,
            )
            return position, velocity, acceleration

        raise ValueError(f"Unknown trajectory {self.config.trajectory}")

    def _check_vector_shape(self, name: str, value: np.ndarray) -> None:
        if value.shape != (self.n,):
            raise ValueError(f"{name} must have shape {(self.n,)}, got {value.shape}")

    @staticmethod
    def _fill_polygon(image: np.ndarray, points: np.ndarray, color: tuple[int, int, int]) -> None:
        if len(points) < 3:
            return

        min_y = max(0, int(np.min(points[:, 1])))
        max_y = min(image.shape[0] - 1, int(np.max(points[:, 1])))
        fill_color = np.array(color, dtype=np.uint8)

        for y in range(min_y, max_y + 1):
            intersections = []
            for p0, p1 in zip(points, np.roll(points, -1, axis=0), strict=True):
                x0, y0 = p0
                x1, y1 = p1
                if y0 == y1:
                    continue
                if (y0 <= y < y1) or (y1 <= y < y0):
                    t = (y - y0) / (y1 - y0)
                    intersections.append(x0 + t * (x1 - x0))
            intersections.sort()
            for x_start, x_end in zip(intersections[0::2], intersections[1::2], strict=False):
                x0 = max(0, int(np.ceil(x_start)))
                x1 = min(image.shape[1] - 1, int(np.floor(x_end)))
                if x0 <= x1:
                    image[y, x0 : x1 + 1] = fill_color

    @staticmethod
    def _draw_disk(image: np.ndarray, center: tuple[int, int], radius: int, color: tuple[int, int, int]) -> None:
        yy, xx = np.ogrid[: image.shape[0], : image.shape[1]]
        mask = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius * radius
        image[mask] = np.array(color, dtype=np.uint8)

    @staticmethod
    def _draw_polyline(
        image: np.ndarray,
        points: np.ndarray,
        color: tuple[int, int, int],
        closed: bool,
    ) -> None:
        segments = zip(points, np.roll(points, -1, axis=0), strict=True) if closed else zip(points[:-1], points[1:], strict=True)
        for p0, p1 in segments:
            DynamicPlateTilt._draw_line(image, tuple(p0), tuple(p1), color)

    @staticmethod
    def _draw_line(
        image: np.ndarray,
        p0: tuple[int, int],
        p1: tuple[int, int],
        color: tuple[int, int, int],
    ) -> None:
        x0, y0 = p0
        x1, y1 = p1
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            if 0 <= x0 < image.shape[1] and 0 <= y0 < image.shape[0]:
                image[y0, x0] = np.array(color, dtype=np.uint8)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy


if __name__ == "__main__":
    import cv2

    env = DynamicPlateTilt(DynamicPlateTiltConfig(dimensionality=2))
    obs, info = env.reset(seed=0)
    terminated = False
    truncated = False
    total_reward = 0.0

    while not terminated and not truncated:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        frame = env.render()
        cv2.imshow("Dynamic Plate Tilt", frame)
        if cv2.waitKey(20) & 0xFF == ord("q"):
            break

    env.close()
    cv2.destroyAllWindows()
    print(f"Episode reward: {total_reward:.3f}")
    print(info)