from dataclasses import dataclass
from math import tau, sin, cos, pi
from typing import Any

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces


def create_circular_trajectory(radius: float, speed: float) -> list[tuple[float, float]]:
    circle_path_length = tau * radius
    num_waypoints = int(circle_path_length / speed)
    waypoints = []
    for i in range(num_waypoints):
        t = tau * i / num_waypoints
        x = radius * cos(t)
        y = radius * sin(t)
        waypoints.append((x, y))
    return waypoints


def create_figure8_trajectory(radius: float, speed: float) -> list[tuple[float, float]]:
    """
    Lemniscate-of-Gerono infinity trajectory, resampled to (approximately)
    uniform arc-length spacing so consecutive waypoints are `speed` apart.
    """
    dense_t = np.linspace(0.0, tau, 20_000, endpoint=False)
    dense = np.column_stack(
        (radius * np.sin(dense_t), radius * np.sin(dense_t) * np.cos(dense_t))
    )
    deltas = np.diff(dense, axis=0, append=dense[:1])
    arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(deltas, axis=1))[:-1]))
    total_length = arc[-1] + float(np.linalg.norm(dense[0] - dense[-1]))

    num_waypoints = max(2, int(total_length / speed))
    targets = np.linspace(0.0, total_length, num_waypoints, endpoint=False)
    indices = np.searchsorted(arc, targets, side="left").clip(0, len(dense) - 1)
    return [tuple(point) for point in dense[indices]]


def transform_polygon(vertices_body: np.ndarray, pose: np.ndarray) -> np.ndarray:
    x, y, theta = pose
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, -s], [s, c]], dtype=float)
    return vertices_body @ R.T + np.array([x, y], dtype=float)


# Generate a random polygon whose body-frame circumcircle has the requested radius.
def generate_random_polygon_inside_radius(
    rng: np.random.Generator,
    object_radius: float,
    num_vertices: int = 7,
    min_radius_fraction: float = 0.35,
    max_radius_fraction: float = 0.95,
) -> np.ndarray:
    angles = np.sort(rng.uniform(0.0, tau, size=num_vertices))
    radii = rng.uniform(
        min_radius_fraction * object_radius,
        max_radius_fraction * object_radius,
        size=num_vertices,
    )

    # Force the declared object radius to be the actual circumcircle radius.
    # The polygon's body-frame origin is the circumcircle center.
    radii[int(rng.integers(0, num_vertices))] = object_radius

    vertices = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    return vertices.astype(float)


def point_segment_distance_batch(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom == 0.0:
        return np.linalg.norm(points - a, axis=1)

    s = np.clip(((points - a) @ ab) / denom, 0.0, 1.0)
    closest = a + s[:, np.newaxis] * ab
    return np.linalg.norm(points - closest, axis=1)



def polygon_swept_push_contact(
    polygon_world: np.ndarray,
    segment_a: np.ndarray,
    segment_b: np.ndarray,
    push_dir: np.ndarray,
    push_distance: float,
    contact_margin: float,
) -> tuple[float, np.ndarray | None]:
    """
    Approximate contact between a polygon and a finite line pusher that sweeps
    inward by push_distance.

    Returns:
        remaining_push: how much pusher travel remains after first contact.
        contact_point: approximate world-frame contact point, or None.

    In the pusher frame, x is along the pusher tangent and y is along push_dir.
    The swept pusher region is the rectangle:
        -L/2 <= x <= L/2
         0   <= y <= push_distance
    with contact_margin slack.
    """
    segment = segment_b - segment_a
    segment_length = float(np.linalg.norm(segment))
    if segment_length == 0.0:
        return 0.0, None

    tangent = segment / segment_length
    pusher_center = 0.5 * (segment_a + segment_b)

    rel = polygon_world - pusher_center
    polygon_local = np.column_stack((rel @ tangent, rel @ push_dir))

    half_length = 0.5 * segment_length
    x_min = -half_length - contact_margin
    x_max = half_length + contact_margin
    y_min = -contact_margin
    y_max = push_distance + contact_margin

    def inside_rect(points: np.ndarray) -> np.ndarray:
        return (
            (x_min <= points[:, 0])
            & (points[:, 0] <= x_max)
            & (y_min <= points[:, 1])
            & (points[:, 1] <= y_max)
        )

    def point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
        x, y = point
        inside = False
        n = len(polygon)
        for i in range(n):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % n]
            crosses = (y1 > y) != (y2 > y)
            if crosses:
                x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                if x < x_cross:
                    inside = not inside
        return inside

    def segment_intersections_with_rect(p0: np.ndarray, p1: np.ndarray) -> list[np.ndarray]:
        intersections = []
        d = p1 - p0

        if abs(d[0]) > 1e-12:
            for x_boundary in (x_min, x_max):
                s = (x_boundary - p0[0]) / d[0]
                if 0.0 <= s <= 1.0:
                    p = p0 + s * d
                    if y_min <= p[1] <= y_max:
                        intersections.append(p)

        if abs(d[1]) > 1e-12:
            for y_boundary in (y_min, y_max):
                s = (y_boundary - p0[1]) / d[1]
                if 0.0 <= s <= 1.0:
                    p = p0 + s * d
                    if x_min <= p[0] <= x_max:
                        intersections.append(p)

        return intersections

    candidate_points = []

    vertex_mask = inside_rect(polygon_local)
    if np.any(vertex_mask):
        candidate_points.extend(polygon_local[vertex_mask])

    for i in range(len(polygon_local)):
        p0 = polygon_local[i]
        p1 = polygon_local[(i + 1) % len(polygon_local)]
        candidate_points.extend(segment_intersections_with_rect(p0, p1))

    rect_corners = np.array(
        [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
        ],
        dtype=float,
    )
    for corner in rect_corners:
        if point_in_polygon(corner, polygon_local):
            candidate_points.append(corner)

    if not candidate_points:
        return 0.0, None

    candidates = np.asarray(candidate_points, dtype=float)
    valid = (candidates[:, 1] >= y_min) & (candidates[:, 1] <= y_max)
    candidates = candidates[valid]
    if len(candidates) == 0:
        return 0.0, None

    first_contact_depth = max(0.0, float(np.min(candidates[:, 1])))
    if first_contact_depth > push_distance + contact_margin:
        return 0.0, None

    remaining_push = max(0.0, push_distance - first_contact_depth)
    if remaining_push <= 0.0:
        return 0.0, None

    near_first_contact = np.isclose(candidates[:, 1], np.min(candidates[:, 1]), atol=contact_margin)
    contact_local = np.mean(candidates[near_first_contact], axis=0)
    contact_world = pusher_center + contact_local[0] * tangent + contact_local[1] * push_dir

    return remaining_push, contact_world


def discretized_V_pixel_offsets(u: float, d_con: float, resolution: float) -> list[np.ndarray]:
    nx = int(np.ceil(d_con / resolution))
    ny = int(np.ceil((0.5 * d_con) / resolution))

    c = np.cos(u)
    s = np.sin(u)
    R = np.array([[c, -s], [s, c]], dtype=float)

    motions = []
    for ix in range(-nx, 1):
        dx = ix * resolution
        for iy in range(-ny, ny + 1):
            dy = iy * resolution
            ellipse_value = (dx / d_con) ** 2 + (dy / (0.5 * d_con)) ** 2
            if ellipse_value <= 1.0 + 1e-12 and dx <= 0.0:
                motions.append(R @ np.array([dx, dy], dtype=float))

    if not any(np.allclose(v, [0.0, 0.0]) for v in motions):
        motions.append(np.array([0.0, 0.0], dtype=float))

    return motions


@dataclass
class CagingEnvParams:
    object_radius: float = 0.025
    pusher_length: float = 0.10
    push_distance: float = 0.02
    cage_size: float = 0.020
    # The belief grid must cover at least the cage circle S_A (radius
    # R = cage_size + object_radius = 45 mm) so PSS points are not silently
    # cropped away. 121 cells at 1 mm covers +/- 60 mm around the cage center.
    resolution: float = 0.001
    grid_height: int = 121
    grid_width: int = 121
    # K candidate pusher angles. The paper evaluates K in {16, 32, 64, 128};
    # below 16 the angular spacing is too coarse for reliable caging.
    num_actions: int = 128
    num_polygon_vertices: int = 7

    ice_mode: bool = False
    ice_melt_rate: float = 0.013

    lambda_1: float = 1.0
    lambda_2: float = 1.0

    trajectory_shape: str = "circle"  # "circle" or "figure8"
    trajectory_radius: float = 0.05
    trajectory_speed: float = 0.001
    trajectory_loops: int = 1

    max_episode_steps: int | None = None
    include_noop_action: bool = True

    # When True, the true object pose and polygon vertices are appended to the
    # observation. This is privileged information unavailable in a real
    # deployment of Caging in Time; keep False for limited-information agents.
    include_polygon_state_in_obs: bool = False

    # Real-object dynamics knobs.
    contact_margin: float = 0.006
    rotation_gain: float = 0.3
    action_penalty: float = 0.001
    failure_penalty: float = 10.0


class PolygonCagingEnv(gym.Env):
    """
    Gymnasium environment for caging a real polygonal object.

    Important split:
      - self.pss is an internal belief/PSS used for heuristic planning/debugging.
      - self.object_pose is the actual simulated object state.
      - RL actions push the actual polygon.
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

    def __init__(
        self,
        params: CagingEnvParams | None = None,
        polygon_vertices_body: np.ndarray | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()

        self.params = params or CagingEnvParams()
        self.render_mode = render_mode

        if polygon_vertices_body is None:
            self._fixed_polygon_vertices_body = None
        else:
            fixed_polygon = np.asarray(polygon_vertices_body, dtype=float)
            max_vertex_radius = float(np.max(np.linalg.norm(fixed_polygon, axis=1)))
            if max_vertex_radius <= 0.0:
                raise ValueError("polygon_vertices_body must contain at least one nonzero vertex")
            self._fixed_polygon_vertices_body = fixed_polygon * (
                self.params.object_radius / max_vertex_radius
            )
        self.polygon_vertices_body = np.empty((0, 2), dtype=float)

        self.R = self.params.cage_size + self.params.object_radius
        self.r = self.params.object_radius
        self.L = self.params.pusher_length
        self.K = self.params.num_actions
        self.H = self.params.grid_height
        self.W = self.params.grid_width
        self.resolution = self.params.resolution
        self.cage_radius = self.params.cage_size
        self.d_push = self.params.push_distance

        self.lambda_1 = self.params.lambda_1
        self.lambda_2 = self.params.lambda_2

        if self.params.trajectory_shape == "figure8":
            single_loop = create_figure8_trajectory(
                radius=self.params.trajectory_radius,
                speed=self.params.trajectory_speed,
            )
        elif self.params.trajectory_shape == "circle":
            single_loop = create_circular_trajectory(
                radius=self.params.trajectory_radius,
                speed=self.params.trajectory_speed,
            )
        else:
            raise ValueError(f"Unknown trajectory_shape {self.params.trajectory_shape!r}")
        self.trajectory = single_loop * max(1, self.params.trajectory_loops)

        self.max_episode_steps = (
            self.params.max_episode_steps
            if self.params.max_episode_steps is not None
            else len(self.trajectory) - 1
        )

        self._action_angles = tau * np.arange(self.K, dtype=float) / self.K
        self._action_radials = np.column_stack(
            (np.cos(self._action_angles), np.sin(self._action_angles))
        )
        self._action_tangents = np.column_stack(
            (-np.sin(self._action_angles), np.cos(self._action_angles))
        )

        # PSS particles carry a radius hypothesis (ice mode melts it). Radii
        # are deduplicated to bins of one grid cell so distinct conditional
        # beliefs over radii survive the per-cell deduplication.
        self._radius_bin_size = self.resolution
        self._num_radius_bins = (
            int(np.ceil(self.params.object_radius / self._radius_bin_size)) + 2
        )
        self._poa_kernels: dict[int, np.ndarray] = {}
        self._poa_kernel = self._poa_kernel_for_radius(self.r)

        num_rl_actions = self.K + int(self.params.include_noop_action)
        self.action_space = spaces.Discrete(num_rl_actions)

        # RL observation: local cage-frame belief masks plus compact numeric state.
        # belief[0] = PSS occupancy, belief[1] = POA occupancy.
        meta_dim = 7
        if self.params.include_polygon_state_in_obs:
            meta_dim += 3 + 2 * self.params.num_polygon_vertices
        self.observation_space = spaces.Dict(
            {
                "belief": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(2, self.H, self.W),
                    dtype=np.float32,
                ),
                "meta": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(meta_dim,),
                    dtype=np.float32,
                ),
            }
        )

        self.t = 0
        self.object_pose = np.zeros(3, dtype=float)
        self.pss = np.empty((0, 3), dtype=float)
        self.last_action_angle: float | None = None
        self.viewer_open = False

    def _radius_bins(self, radii: np.ndarray) -> np.ndarray:
        return np.clip(
            np.round(np.asarray(radii, dtype=float) / self._radius_bin_size),
            0,
            self._num_radius_bins - 1,
        ).astype(np.int64)

    def _poa_kernel_for_radius(self, radius: float) -> np.ndarray:
        """Disk dilation kernel for a given radius hypothesis, cached per bin."""
        rbin = int(np.clip(round(radius / self._radius_bin_size), 0, self._num_radius_bins - 1))
        kernel = self._poa_kernels.get(rbin)
        if kernel is None:
            # Use the bin's upper edge so the dilation stays conservative for
            # every radius that falls into the bin.
            bin_radius = (rbin + 0.5) * self._radius_bin_size
            cells = max(1, int(np.ceil(bin_radius / self.resolution)))
            offsets = np.arange(-cells, cells + 1)
            dj, di = np.meshgrid(offsets, offsets, indexing="xy")
            kernel = (
                (dj * self.resolution) ** 2 + (di * self.resolution) ** 2
                <= bin_radius * bin_radius + 1e-12
            ).astype(np.uint8)
            self._poa_kernels[rbin] = kernel
        return kernel

    def get_action_angle(self, action: int | None) -> float | None:
        if action is None:
            return None

        if self.params.include_noop_action and action == self.K:
            return None

        return float(self._action_angles[int(action)])

    def world_to_grid_batch(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        cage_center: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        cage_cx, cage_cy = cage_center
        j = np.round((xs - cage_cx) / self.resolution + self.W / 2).astype(int)
        i = np.round((ys - cage_cy) / self.resolution + self.H / 2).astype(int)
        return i, j

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        self.t = 0
        self.last_action_angle = None

        # Ice mode melts self.r during the episode; restore it so episodes
        # are independent.
        self.r = self.params.object_radius

        x0, y0 = self.trajectory[0]
        theta0 = 0.0

        if options is not None:
            x0 = options.get("x0", x0)
            y0 = options.get("y0", y0)
            theta0 = options.get("theta0", theta0)

        if self._fixed_polygon_vertices_body is None:
            self.polygon_vertices_body = generate_random_polygon_inside_radius(
                self.np_random,
                self.params.object_radius,
                num_vertices=self.params.num_polygon_vertices,
            )
        else:
            self.polygon_vertices_body = self._fixed_polygon_vertices_body.copy()

        self.object_pose = np.array([x0, y0, theta0], dtype=float)

        # Internal belief starts with the true object center.
        self.pss = np.array([[x0, y0, self.r]], dtype=float)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def _get_obs(self) -> dict[str, np.ndarray]:
        cage_center = np.asarray(self.trajectory[self.t], dtype=float)
        next_idx = min(self.t + 1, len(self.trajectory) - 1)
        next_cage_center = np.asarray(self.trajectory[next_idx], dtype=float)

        belief = self._get_belief_observation(cage_center)

        cage_delta = next_cage_center - cage_center
        if self.last_action_angle is None:
            last_sin = 0.0
            last_cos = 0.0
            has_last_action = 0.0
        else:
            last_sin = float(np.sin(self.last_action_angle))
            last_cos = float(np.cos(self.last_action_angle))
            has_last_action = 1.0

        t_frac = self.t / max(1, len(self.trajectory) - 1)
        meta_values = [
            float(cage_delta[0]),
            float(cage_delta[1]),
            last_sin,
            last_cos,
            has_last_action,
            float(t_frac),
            float(len(self.pss)) / float(self.H * self.W),
        ]

        if self.params.include_polygon_state_in_obs:
            polygon_world = transform_polygon(self.polygon_vertices_body, self.object_pose)
            polygon_local = polygon_world - cage_center
            meta_values.extend(
                [
                    float(self.object_pose[0] - cage_center[0]),
                    float(self.object_pose[1] - cage_center[1]),
                    float(self.object_pose[2]),
                ]
            )
            meta_values.extend(polygon_local.astype(np.float32).reshape(-1).tolist())

        return {
            "belief": belief.astype(np.float32, copy=False),
            "meta": np.asarray(meta_values, dtype=np.float32),
        }

    def _pss_mask(self, cage_center: np.ndarray) -> np.ndarray:
        mask = np.zeros((self.H, self.W), dtype=np.uint8)
        if len(self.pss) == 0:
            return mask

        center_i, center_j = self.world_to_grid_batch(self.pss[:, 0], self.pss[:, 1], tuple(cage_center))
        in_bounds = (
            (0 <= center_i)
            & (center_i < self.H)
            & (0 <= center_j)
            & (center_j < self.W)
        )
        mask[center_i[in_bounds], center_j[in_bounds]] = 1
        return mask

    def _poa_mask(self, cage_center: np.ndarray) -> np.ndarray:
        """
        POA occupancy: PSS centers dilated by their own radius hypothesis.
        Particles are grouped by radius bin and each group is dilated with a
        disk kernel matching that bin, so melted (smaller) hypotheses claim
        less area than unmelted ones.
        """
        poa = np.zeros((self.H, self.W), dtype=np.uint8)
        if len(self.pss) == 0:
            return poa

        center_i, center_j = self.world_to_grid_batch(
            self.pss[:, 0], self.pss[:, 1], tuple(cage_center)
        )
        in_bounds = (
            (0 <= center_i)
            & (center_i < self.H)
            & (0 <= center_j)
            & (center_j < self.W)
        )
        center_i = center_i[in_bounds]
        center_j = center_j[in_bounds]
        radii = self.pss[in_bounds, 2]
        if len(center_i) == 0:
            return poa

        rbins = self._radius_bins(radii)
        for rbin in np.unique(rbins):
            group = rbins == rbin
            occupancy = np.zeros((self.H, self.W), dtype=np.uint8)
            occupancy[center_i[group], center_j[group]] = 1
            kernel = self._poa_kernel_for_radius(float(rbin) * self._radius_bin_size)
            poa |= cv2.dilate(occupancy, kernel, iterations=1)
        return poa

    def _get_belief_observation(self, cage_center: np.ndarray) -> np.ndarray:
        pss_mask = self._pss_mask(cage_center)
        poa_mask = self._poa_mask(cage_center)
        return np.stack((pss_mask, poa_mask), axis=0)

    def _get_info(self) -> dict[str, Any]:
        cage_center = np.asarray(self.trajectory[self.t], dtype=float)
        polygon_world = transform_polygon(self.polygon_vertices_body, self.object_pose)
        vertex_dists = np.linalg.norm(polygon_world - cage_center, axis=1)
        max_vertex_dist = float(np.max(vertex_dists))

        return {
            "t": self.t,
            "cage_center": tuple(cage_center),
            "object_pose": self.object_pose.copy(),
            "polygon_world": polygon_world,
            # Two grid cells of tolerance, matching the Alg. 2 verification in
            # main.py: the PSS propagation and the worst-case push displacement
            # are quantized to the grid, so the true state can legitimately sit
            # up to ~2 cells outside R-r while caging is still working.
            "contained": bool(max_vertex_dist <= self.R + 2.0 * self.resolution),
            "max_vertex_dist": max_vertex_dist,
            "pss_size": int(len(self.pss)),
            "heuristic_action": None,
        }

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action} is outside the action space {self.action_space}")
        u = self.get_action_angle(action)

        if self.t >= len(self.trajectory) - 1:
            obs = self._get_obs()
            info = self._get_info()
            return obs, 0.0, False, True, info

        prev_cage_center = np.asarray(self.trajectory[self.t], dtype=float)
        previous_action_angle = self.last_action_angle

        self.t += 1
        cage_center = np.asarray(self.trajectory[self.t], dtype=float)

        # Ice mode: the real object melts every step (not just on pushes),
        # using the seeded env RNG so rollouts stay reproducible. The belief
        # handles melt separately in _propagate_pss as a set-valued branch.
        if self.params.ice_mode:
            melt_factor = 1.0 - float(self.np_random.uniform(0.0, self.params.ice_melt_rate))
            self.r *= melt_factor
            self.polygon_vertices_body *= melt_factor

        # Move/propagate the PSS belief under the chosen action.
        self._propagate_pss(u, cage_center)

        # Simulate the real polygonal object under the chosen action.
        if u is not None:
            self._push_real_polygon(u, cage_center)
            # The paper's tie-breaking (Alg. 4) uses the angle of the last
            # executed push, so the stored angle survives no-op steps.
            self.last_action_angle = u

        obs = self._get_obs()
        info = self._get_info()

        contained = info["contained"]
        terminated = not contained
        truncated = self.t >= self.max_episode_steps or self.t >= len(self.trajectory) - 1

        reward = self._compute_reward(
            contained=contained,
            action_angle=u,
            previous_action_angle=previous_action_angle,
            cage_center=cage_center,
            prev_cage_center=prev_cage_center,
        )

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _compute_reward(
        self,
        contained: bool,
        action_angle: float | None,
        previous_action_angle: float | None,
        cage_center: np.ndarray,
        prev_cage_center: np.ndarray,
    ) -> float:
        # Training-time reward may use the true (privileged) object state; the
        # agent's observation remains limited to belief + trajectory.
        polygon_world = transform_polygon(self.polygon_vertices_body, self.object_pose)
        max_vertex_dist = np.max(np.linalg.norm(polygon_world - cage_center, axis=1))

        # Positive reward for staying contained with margin.
        margin = self.R - max_vertex_dist
        reward = 10.0 * float(margin)

        if action_angle is not None:
            reward -= self.params.action_penalty
            if previous_action_angle is not None:
                angular_change = abs((action_angle - previous_action_angle + pi) % tau - pi)
                reward -= 0.01 * angular_change

        if not contained:
            reward -= self.params.failure_penalty

        return reward

    def _pusher_segment(self, u: float, cage_center: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        radial = np.array([np.cos(u), np.sin(u)], dtype=float)
        tangent = np.array([-np.sin(u), np.cos(u)], dtype=float)
        push_dir = -radial

        pusher_center = cage_center + self.R * radial
        a = pusher_center - 0.5 * self.L * tangent
        b = pusher_center + 0.5 * self.L * tangent
        return a, b, radial, push_dir

    def _push_real_polygon(self, u: float, cage_center: np.ndarray) -> None:
        a, b, radial, push_dir = self._pusher_segment(u, cage_center)

        # Bounding-circle contact semantics (paper Sec. 5.1): the object is
        # "in collision" when the pusher intersects its bounding circle, and
        # the push displaces it so the circle resolves penetration with the
        # pusher's final pose. This is the worst-case displacement on the
        # boundary of the Peshkin motion set (Eq. 9), so the true state stays
        # covered by the propagated PSS. Modeling polygon-only contact would
        # let the pusher "miss" the real object while the belief assumes the
        # circle was pushed, silently breaking the caging guarantee.
        center = self.object_pose[:2]
        center_dist = point_segment_distance_batch(center[np.newaxis, :], a, b)[0]
        d_con = float(np.clip(self.r + self.d_push - center_dist, 0.0, self.d_push))
        if d_con <= 0.0:
            return

        self.object_pose[:2] = center + d_con * push_dir

        # Shape-dependent rotation: off-center swept contact with the actual
        # polygon induces rotation (crude torque model).
        polygon_world = transform_polygon(self.polygon_vertices_body, self.object_pose)
        push_amount, contact_point = polygon_swept_push_contact(
            polygon_world=polygon_world,
            segment_a=a,
            segment_b=b,
            push_dir=push_dir,
            push_distance=self.d_push,
            contact_margin=self.params.contact_margin,
        )
        if push_amount <= 0.0 or contact_point is None:
            return

        lever = contact_point - self.object_pose[:2]
        torque_z = lever[0] * push_dir[1] - lever[1] * push_dir[0]
        self.object_pose[2] += self.params.rotation_gain * torque_z * push_amount / max(
            self.r * self.r,
            1e-12,
        )
        self.object_pose[2] = (self.object_pose[2] + pi) % tau - pi

    def _propagate_pss(self, u: float | None, cage_center: np.ndarray) -> None:
        if self.params.ice_mode and len(self.pss) > 0:
            # Set-valued melt: the true radius shrinks by an unknown factor in
            # [1 - ice_melt_rate, 1]. Branch every particle on the interval
            # endpoints; intermediate radii stay covered up to the radius-bin
            # width used by the per-(cell, radius-bin) deduplication. Sampling
            # a single factor here would let the belief underestimate the true
            # radius and silently void the PSS containment guarantee.
            melted = self.pss.copy()
            melted[:, 2] *= 1.0 - self.params.ice_melt_rate
            self.pss = np.vstack((self.pss, melted))
            # Per-step melt (~1.3% of r) is smaller than the radius bin, so a
            # melted branch usually lands in its parent's bin. Sort ascending
            # by radius so the per-(cell, bin) dedup keeps the smallest radius
            # in each bin and cumulative melt can keep walking downward across
            # bins instead of being re-rounded up every step. Conservatism is
            # preserved because POA kernels use each bin's upper edge.
            self.pss = self.pss[np.argsort(self.pss[:, 2], kind="stable")]

        if u is None:
            self.pss = self._deduplicate_and_crop(self.pss, cage_center)
            return

        a, b, radial, push_dir = self._pusher_segment(u, cage_center)
        distances = point_segment_distance_batch(self.pss[:, :2], a, b)

        radii = self.pss[:, 2] if self.pss.shape[1] >= 3 else self.r
        d_cons = np.clip(radii + self.d_push - distances, 0.0, self.d_push)
        contacted = d_cons > 0.0
        uncontacted = ~contacted

        # Final pose of the pusher segment after sweeping d_push toward the
        # cage center. Propagated belief states whose bounding circle would
        # intersect this segment are not in the object's free space (Eq. 1)
        # and must be excluded, otherwise the PSS can only grow and pushes
        # never contract the belief.
        push_displacement = self.d_push * push_dir
        a_final = a + push_displacement
        b_final = b + push_displacement

        def distance_to_final_pusher(points: np.ndarray) -> np.ndarray:
            return point_segment_distance_batch(points, a_final, b_final)

        # Deduplicate per (grid cell, radius bin) so particles that share a
        # position but carry different radius hypotheses both survive.
        occupied_grid = np.zeros(self.H * self.W * self._num_radius_bins, dtype=bool)
        pss_next_batches: list[np.ndarray] = []

        def append_first_unoccupied(points: np.ndarray) -> None:
            if len(points) == 0:
                return

            grid_indices = np.round(
                (points[:, :2] - cage_center) / self.resolution
                + np.array([self.W / 2, self.H / 2])
            ).astype(int)

            in_bounds = (
                (0 <= grid_indices[:, 0])
                & (grid_indices[:, 0] < self.W)
                & (0 <= grid_indices[:, 1])
                & (grid_indices[:, 1] < self.H)
            )
            if not np.any(in_bounds):
                return

            points_in_bounds = points[in_bounds]
            grid_indices = grid_indices[in_bounds]

            flat_grid_indices = (
                grid_indices[:, 1] * self.W + grid_indices[:, 0]
            ) * self._num_radius_bins + self._radius_bins(points_in_bounds[:, 2])

            num_keys = self.H * self.W * self._num_radius_bins
            first_indices = np.full(num_keys, len(flat_grid_indices), dtype=np.int64)
            np.minimum.at(
                first_indices,
                flat_grid_indices,
                np.arange(len(flat_grid_indices), dtype=np.int64),
            )
            local_unique_indices = first_indices[first_indices < len(flat_grid_indices)]
            local_unique_indices.sort()

            local_flat = flat_grid_indices[local_unique_indices]
            not_seen = ~occupied_grid[local_flat]
            if not np.any(not_seen):
                return

            local_unique_indices = local_unique_indices[not_seen]
            local_flat = local_flat[not_seen]
            occupied_grid[local_flat] = True
            pss_next_batches.append(points_in_bounds[local_unique_indices])

        append_first_unoccupied(self.pss[uncontacted])

        contacted_points = self.pss[contacted]
        contacted_d_cons = d_cons[contacted]

        if len(contacted_points) > 0:
            nxs = np.ceil(contacted_d_cons / self.resolution).astype(int)
            nys = np.ceil((0.5 * contacted_d_cons) / self.resolution).astype(int)
            ellipse_bins = np.column_stack((nxs, nys))
            unique_bins, inverse_bins = np.unique(ellipse_bins, axis=0, return_inverse=True)

            c_u = np.cos(u)
            s_u = np.sin(u)
            R_u = np.array([[c_u, -s_u], [s_u, c_u]], dtype=float)

            max_candidates_per_chunk = 2_000_000

            for bin_idx, (nx, ny) in enumerate(unique_bins):
                batch_points = contacted_points[inverse_bins == bin_idx]
                d_con = float(contacted_d_cons[inverse_bins == bin_idx].max())

                ix = np.arange(-nx, 1, dtype=float)
                iy = np.arange(-ny, ny + 1, dtype=float)
                dx, dy = np.meshgrid(
                    ix * self.resolution,
                    iy * self.resolution,
                    indexing="ij",
                )

                ellipse_values = (dx / d_con) ** 2 + (dy / (0.5 * d_con)) ** 2
                valid = ellipse_values <= 1.0 + 1e-12
                local_offsets = np.column_stack((dx[valid], dy[valid]))

                if not np.any(np.all(np.isclose(local_offsets, 0.0), axis=1)):
                    local_offsets = np.vstack((local_offsets, np.array([[0.0, 0.0]])))

                offsets = local_offsets @ R_u.T
                chunk_size = max(1, max_candidates_per_chunk // len(offsets))

                for start_idx in range(0, len(batch_points), chunk_size):
                    chunk_points = batch_points[start_idx : start_idx + chunk_size]
                    moved_xy = (
                        chunk_points[:, np.newaxis, :2] + offsets[np.newaxis, :, :]
                    ).reshape(-1, 2)
                    radii_repeated = np.repeat(chunk_points[:, 2], len(offsets))
                    moved_points = np.column_stack((moved_xy, radii_repeated))

                    # Keep only motions that resolve contact with the pusher's
                    # final pose, best effort per state when d_con cannot fully
                    # clear the overlap, with one grid cell of slack for the
                    # discretized motion offsets.
                    dist_final = distance_to_final_pusher(moved_points[:, :2]).reshape(
                        len(chunk_points), len(offsets)
                    )
                    per_state_target = np.minimum(chunk_points[:, 2], dist_final.max(axis=1))
                    keep = dist_final >= (
                        per_state_target[:, np.newaxis] - self.resolution - 1e-9
                    )
                    append_first_unoccupied(moved_points[keep.reshape(-1)])

        if pss_next_batches:
            self.pss = np.vstack(pss_next_batches)
        else:
            self.pss = np.empty((0, 3), dtype=float)

    def _deduplicate_and_crop(self, points: np.ndarray, cage_center: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return np.empty((0, 3), dtype=float)

        # Keep the smallest radius per (cell, radius bin); see _propagate_pss.
        points = points[np.argsort(points[:, 2], kind="stable")]

        grid_indices = np.round(
            (points[:, :2] - cage_center) / self.resolution
            + np.array([self.W / 2, self.H / 2])
        ).astype(int)

        in_bounds = (
            (0 <= grid_indices[:, 0])
            & (grid_indices[:, 0] < self.W)
            & (0 <= grid_indices[:, 1])
            & (grid_indices[:, 1] < self.H)
        )

        points = points[in_bounds]
        grid_indices = grid_indices[in_bounds]

        if len(points) == 0:
            return np.empty((0, 3), dtype=float)

        flat_grid_indices = (
            grid_indices[:, 1] * self.W + grid_indices[:, 0]
        ) * self._num_radius_bins + self._radius_bins(points[:, 2])
        first_indices = np.full(
            self.H * self.W * self._num_radius_bins,
            len(flat_grid_indices),
            dtype=np.int64,
        )

        np.minimum.at(
            first_indices,
            flat_grid_indices,
            np.arange(len(flat_grid_indices), dtype=np.int64),
        )

        unique_indices = first_indices[first_indices < len(flat_grid_indices)]
        unique_indices.sort()

        return points[unique_indices]

    def potential_object_area_points(
        self,
        pss: np.ndarray,
        cage_center: tuple[float, float] | np.ndarray,
    ) -> np.ndarray:
        if len(pss) == 0:
            return np.empty((0, 2), dtype=np.float64)

        xs = pss[:, 0]
        ys = pss[:, 1]
        center_i, center_j = self.world_to_grid_batch(xs, ys, tuple(cage_center))

        in_bounds = (
            (0 <= center_i)
            & (center_i < self.H)
            & (0 <= center_j)
            & (center_j < self.W)
        )
        center_i = center_i[in_bounds]
        center_j = center_j[in_bounds]
        radii = (
            pss[in_bounds, 2]
            if pss.shape[1] >= 3
            else np.full(len(center_i), self.r, dtype=float)
        )

        if len(center_i) == 0:
            return np.empty((0, 2), dtype=np.float64)

        # Dilate per radius bin so each radius hypothesis claims only the
        # area its own bounding circle can occupy.
        rbins = self._radius_bins(radii)
        dilated = np.zeros((self.H, self.W), dtype=np.uint8)
        for rbin in np.unique(rbins):
            group = rbins == rbin
            occupancy = np.zeros((self.H, self.W), dtype=np.uint8)
            occupancy[center_i[group], center_j[group]] = 1
            kernel = self._poa_kernel_for_radius(float(rbin) * self._radius_bin_size)
            dilated |= cv2.dilate(occupancy, kernel, iterations=1)
        occupied_i, occupied_j = np.nonzero(dilated)

        cx, cy = cage_center
        points = np.empty((len(occupied_i), 2), dtype=np.float64)
        points[:, 0] = cx + (occupied_j - self.W / 2) * self.resolution
        points[:, 1] = cy + (occupied_i - self.H / 2) * self.resolution

        return points

    def find_push_action(self) -> int:
        """
        Internal heuristic policy using PSS. This does not look at the real polygon,
        except indirectly through the PSS initialized at reset.
        """
        if self.t >= len(self.trajectory) - 1:
            return self.K if self.params.include_noop_action else 0

        u = self.find_push_angle()

        if u is None:
            return self.K if self.params.include_noop_action else 0

        idx = int(np.argmin(np.abs(((self._action_angles - u + pi) % tau) - pi)))
        return idx

    def find_push_angle(self) -> float | None:
        next_cage_center = np.asarray(self.trajectory[self.t + 1], dtype=float)
        cage_center_arr = next_cage_center

        rel_pss_to_cage = self.pss[:, :2] - cage_center_arr
        pss_radius_sq = np.sum(rel_pss_to_cage * rel_pss_to_cage, axis=1)

        if np.all(pss_radius_sq <= self.cage_radius * self.cage_radius):
            return None

        poa_points = self.potential_object_area_points(self.pss, next_cage_center)
        cell_area = self.resolution * self.resolution

        angles = self._action_angles
        radials = self._action_radials
        tangents = self._action_tangents
        pusher_centers = cage_center_arr[np.newaxis, :] + self.R * radials

        if poa_points.shape[0] > 0:
            rel_poa = poa_points[np.newaxis, :, :] - pusher_centers[:, np.newaxis, :]
            poa_outward_depth = np.einsum("knd,kd->kn", rel_poa, radials)
            on_pusher_side = poa_outward_depth >= 0.0
            within_pusher_span = np.abs(
                np.einsum("knd,kd->kn", rel_poa, tangents)
            ) <= 0.5 * self.L
            poa_behind_pusher = on_pusher_side & within_pusher_span

            # S_out^k: POA area behind the line pusher (paper Fig. 7, right).
            poa_area_behind = (
                np.count_nonzero(poa_behind_pusher, axis=1).astype(float) * cell_area
            )

            # d_out^k: furthest distance from behind the pusher to the POA
            # boundary.
            poa_depth_behind_pusher = np.where(
                poa_behind_pusher,
                poa_outward_depth,
                0.0,
            )
            furthest_dist = np.max(poa_depth_behind_pusher, axis=1)
        else:
            poa_area_behind = np.zeros(self.K, dtype=float)
            furthest_dist = np.zeros(self.K, dtype=float)

        scores = self.lambda_1 * poa_area_behind + self.lambda_2 * furthest_dist * furthest_dist
        sorted_indices = np.argsort(scores)

        if self.last_action_angle is None:
            return float(angles[sorted_indices[-1]])

        top_n = min(5, len(angles))
        top_angles = angles[sorted_indices[-top_n:]]

        return float(
            min(
                top_angles,
                key=lambda angle: abs((angle - self.last_action_angle + pi) % tau - pi),
            )
        )

    def heuristic_policy_action(self) -> int:
        """
        Convenience method for non-RL rollouts:
            action = env.heuristic_policy_action()
        """
        return self.find_push_action()

    def draw_pss_poa_masks(
        self,
        image: np.ndarray,
        points: np.ndarray,
        points_to_image_fn,
        pss_color: tuple[int, int, int] = (255, 255, 255),
        poa_color: tuple[int, int, int] = (96, 169, 101),
    ) -> None:
        if len(points) == 0:
            return

        image_h, image_w = image.shape[:2]
        point_i, point_j = points_to_image_fn(points)

        in_bounds = (
            (0 <= point_i)
            & (point_i < image_h)
            & (0 <= point_j)
            & (point_j < image_w)
        )
        point_i = point_i[in_bounds]
        point_j = point_j[in_bounds]
        radii = (
            np.asarray(points)[in_bounds, 2]
            if np.asarray(points).ndim == 2 and np.asarray(points).shape[1] >= 3
            else np.full(len(point_i), self.r, dtype=float)
        )

        if len(point_i) == 0:
            return

        pss_mask = np.zeros((image_h, image_w), dtype=np.uint8)
        pss_mask[point_i, point_j] = 1

        # Dilate per radius bin so melted hypotheses are drawn with their own
        # (smaller) footprint.
        rbins = self._radius_bins(radii)
        poa_mask = np.zeros((image_h, image_w), dtype=np.uint8)
        for rbin in np.unique(rbins):
            group = rbins == rbin
            occupancy = np.zeros((image_h, image_w), dtype=np.uint8)
            occupancy[point_i[group], point_j[group]] = 1
            kernel = self._poa_kernel_for_radius(float(rbin) * self._radius_bin_size)
            poa_mask |= cv2.dilate(occupancy, kernel, iterations=1)
        poa_mask = poa_mask.astype(bool)
        pss_mask = pss_mask.astype(bool)

        image[poa_mask] = poa_color
        image[pss_mask] = pss_color

    def render(self) -> np.ndarray | None:
        image = self.render_world()

        if self.render_mode == "human":
            cv2.imshow("PolygonCagingEnv", image)
            cv2.waitKey(1)
            return None

        return image

    def render_world(self) -> np.ndarray:
        trajectory_arr = np.asarray(self.trajectory, dtype=float)

        # Include trajectory, cage, and possible polygon extent.
        margin = self.R + self.r + self.resolution

        min_x = float(np.min(trajectory_arr[:, 0]) - margin)
        max_x = float(np.max(trajectory_arr[:, 0]) + margin)
        min_y = float(np.min(trajectory_arr[:, 1]) - margin)
        max_y = float(np.max(trajectory_arr[:, 1]) + margin)

        world_W = int(np.ceil((max_x - min_x) / self.resolution)) + 1
        world_H = int(np.ceil((max_y - min_y) / self.resolution)) + 1
        image = np.zeros((world_H, world_W, 3), dtype=np.uint8)

        def world_to_image(x: float, y: float) -> tuple[int, int]:
            j = int(round((x - min_x) / self.resolution))
            i = int(round((max_y - y) / self.resolution))
            return i, j

        def world_to_image_batch(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            js = np.round((points[:, 0] - min_x) / self.resolution).astype(int)
            is_ = np.round((max_y - points[:, 1]) / self.resolution).astype(int)
            return is_, js

        cage_center = np.asarray(self.trajectory[self.t], dtype=float)
        xc, yc = cage_center

        # Trajectory.
        trajectory_i, trajectory_j = world_to_image_batch(trajectory_arr)
        in_bounds = (
            (0 <= trajectory_i)
            & (trajectory_i < world_H)
            & (0 <= trajectory_j)
            & (trajectory_j < world_W)
        )
        image[trajectory_i[in_bounds], trajectory_j[in_bounds]] = (0, 0, 255)

        # Cage.
        ic, jc = world_to_image(xc, yc)
        cv2.circle(
            image,
            (jc, ic),
            int(round(self.R / self.resolution)),
            (96, 169, 101),
            1,
        )

        # Real polygon.
        polygon_world = transform_polygon(self.polygon_vertices_body, self.object_pose)
        polygon_pixels = []
        for x, y in polygon_world:
            i, j = world_to_image(float(x), float(y))
            polygon_pixels.append([j, i])
        polygon_pixels = np.asarray([polygon_pixels], dtype=np.int32)
        cv2.fillPoly(image, polygon_pixels, (220, 220, 220))
        cv2.polylines(image, polygon_pixels, isClosed=True, color=(255, 255, 255), thickness=1)

        # Current heuristic or actual pusher angle.
        if self.last_action_angle is not None:
            u = self.last_action_angle
            a, b, _, _ = self._pusher_segment(u, cage_center)

            ai, aj = world_to_image(float(a[0]), float(a[1]))
            bi, bj = world_to_image(float(b[0]), float(b[1]))
            cv2.line(image, (aj, ai), (bj, bi), (255, 0, 0), 1)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def close(self) -> None:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    env = PolygonCagingEnv(render_mode="human")
    obs, info = env.reset(seed=0)
    terminated = False
    truncated = False

    while not terminated and not truncated:
        action = env.heuristic_policy_action()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

    env.close()