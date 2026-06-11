from dataclasses import dataclass
from math import tau, sin, cos, tau, pi
from typing import Iterator
from time import time

from matplotlib.image import resample
import numpy as np
import cv2


def d_q_L(x, y, u, cage_center, cage_radius, pusher_length):
    """
    Compute d(q, L(u)) where:
      q = (x, y)
      u = pusher angle theta in radians
      L(u) = finite tangent pusher segment on cage boundary

    cage_center: (x_c, y_c)
    cage_radius: R
    pusher_length: length of the finite line pusher
    """
    q = np.array([x, y], dtype=float)
    c = np.asarray(cage_center, dtype=float)

    # Radial direction from cage center to pusher center
    radial = np.array([np.cos(u), np.sin(u)], dtype=float)

    # Tangent direction of the pusher segment
    tangent = np.array([-np.sin(u), np.cos(u)], dtype=float)

    # Pusher segment center on cage boundary
    p0 = c + cage_radius * radial

    # Segment endpoints
    a = p0 - 0.5 * pusher_length * tangent
    b = p0 + 0.5 * pusher_length * tangent

    # Point-to-segment distance
    ab = b - a
    denom = np.dot(ab, ab)

    if denom == 0.0:
        return np.linalg.norm(q - a)

    s = np.dot(q - a, ab) / denom
    s = np.clip(s, 0.0, 1.0)

    closest = a + s * ab
    return np.linalg.norm(q - closest)


def discretized_V_pixel_offsets(u, d_con, resolution):
    nx = int(np.ceil(d_con / resolution))
    ny = int(np.ceil((0.5 * d_con) / resolution))
    c = np.cos(u)
    s = np.sin(u)
    R = np.array([[c, -s], [s, c]])
    motions = []
    for ix in range(-nx, 1):
        dx = ix * resolution
        for iy in range(-ny, ny + 1):
            dy = iy * resolution
            ellipse_value = (dx / d_con) ** 2 + (dy / (0.5 * d_con)) ** 2
            if ellipse_value <= 1.0 + 1e-12 and dx <= 0.0:
                motions.append(R @ np.array([dx, dy]))
    if not any(np.allclose(v, [0.0, 0.0]) for v in motions):
        motions.append(np.array([0.0, 0.0]))
    return motions


@dataclass
class SimParams:
    # we represent the object's geometry by a simplified bounding circle with radius r
    object_radius: float

    # The robot is equipped with a fixed-length line pusher, represented by a line
    # segment L in the 2D space
    pusher_length: float

    # d_push
    push_distance: float

    # R - r
    cage_size: float

    # grid resolution in world units per pixel
    resolution: float

    # Size of the discretized PSS
    grid_height: int
    grid_width: int

    # K
    num_actions: int = 12

    lambda_1: float = 100
    lambda_2: float = 1


class PSS:
    def __init__(self, params: SimParams, cage_center: tuple[float, float]):
        self.H = params.grid_height
        self.W = params.grid_width
        self.resolution = params.resolution
        self.cage_radius = params.cage_size
        self.cx = cage_center[0]
        self.cy = cage_center[1]

        self.pss = {}
        self.out_of_bounds = False
    
    def __len__(self):
        return len(self.pss)

    def add(self, x: float, y: float):
        dy = y - self.cy
        dx = x - self.cx
        i = int(round((dy / self.resolution + self.H / 2)))
        j = int(round((dx / self.resolution + self.W / 2)))
        if 0 <= i < self.H and 0 <= j < self.W:
            self.pss[(i, j)] = (x, y)
        
        # Check if the point is out of bounds
        if dx * dx + dy * dy > self.cage_radius * self.cage_radius:
            self.out_of_bounds = True

    def is_occupied(self, x: float, y: float) -> bool:
        i = int(round((y - self.cy) / self.resolution + self.H / 2))
        j = int(round((x - self.cx) / self.resolution + self.W / 2))
        return (i, j) in self.pss
    
    def grid_cells(self) -> Iterator[tuple[int, int]]:
        for i, j in self.pss.keys():
            yield i, j

    def __iter__(self):
        for x, y in self.pss.values():
            yield x, y


class Simulator:
    def __init__(self, params: SimParams, trajectory: list[tuple[float, float]]):
        self.params = params
        self.trajectory = trajectory

        self.R = params.cage_size + params.object_radius
        self.r = params.object_radius
        self.K = params.num_actions
        self.L = params.pusher_length

        self.H = params.grid_height
        self.W = params.grid_width

        self.d_push = params.push_distance
        self.resolution = params.resolution
        self.cage_radius = params.cage_size

        self.lambda_1 = params.lambda_1
        self.lambda_2 = params.lambda_2

        self._action_angles = tau * np.arange(self.K, dtype=float) / self.K
        self._action_radials = np.column_stack((np.cos(self._action_angles), np.sin(self._action_angles)))
        self._action_tangents = np.column_stack((-np.sin(self._action_angles), np.cos(self._action_angles)))

        self._poa_radius_cells = int(np.ceil(self.r / self.resolution))
        self._poa_radius_sq = self.r * self.r
        offsets = np.arange(-self._poa_radius_cells, self._poa_radius_cells + 1)
        dj, di = np.meshgrid(offsets, offsets, indexing="xy")
        self._poa_kernel = (
            (dj * self.resolution) ** 2 + (di * self.resolution) ** 2
            <= self._poa_radius_sq + 1e-12
        ).astype(np.uint8)
    
    def get_action_angle(self, k: int | None) -> float | None:
        if k is None:
            return None
        return tau * k / self.K

    def world_to_grid(self, x: float, y: float, cage_center: tuple[float, float]) -> tuple[int, int]:
        cage_cx, cage_cy = cage_center
        j = int(round((x - cage_cx) / self.resolution + self.W / 2))
        i = int(round((y - cage_cy) / self.resolution + self.H / 2))
        return i, j
    
    def world_to_grid_batch(self, xs: np.ndarray, ys: np.ndarray, cage_center: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
        cage_cx, cage_cy = cage_center
        j = np.round((xs - cage_cx) / self.resolution + self.W / 2).astype(int)
        i = np.round((ys - cage_cy) / self.resolution + self.H / 2).astype(int)
        return i, j

    def reset(self, x_object: float, y_object: float):
        self.t = 0
        self.last_action_angle = None
        #self.pss = PSS(self.params, self.trajectory[0])
        #self.pss.add(x_object, y_object)

        self.pss = np.array([[x_object, y_object]], dtype=float)

    def propagate(self, u: float | None, render: bool = False) -> np.ndarray | None:
        """
        Propagate the PSS one step under the selected action.

        Args:
            u: The action angle in radians.
        """
        self.t += 1
        self.last_action_angle = u
        cage_center = self.trajectory[self.t]

        # start = time()
        if u is None:
            pss_next = self.pss
        else:
            # Get the distances from each potential object position to the pusher
            # i.e. do d_q_L in batch for all potential object positions
            c = np.asarray(cage_center, dtype=float)
            radial = np.array([np.cos(u), np.sin(u)], dtype=float)
            tangent = np.array([-np.sin(u), np.cos(u)], dtype=float)
            p0 = c + self.R * radial
            a = p0 - 0.5 * self.L * tangent
            b = p0 + 0.5 * self.L * tangent
            ab = b - a
            denom = np.dot(ab, ab)
            if denom == 0.0:
                distances = np.linalg.norm(self.pss - a, axis=1)
            else:
                s = np.clip(np.dot(self.pss - a, ab) / denom, 0.0, 1.0)
                closest = a + s[:, np.newaxis] * ab
                distances = np.linalg.norm(self.pss - closest, axis=1)
            
            # Get the distances over which the object is pushed
            d_cons = np.clip(self.r + self.d_push - distances, 0, self.d_push)

            # Select the indices where d_con is positive
            d_cons_positive = d_cons > 0
            d_cons_negative = d_cons <= 0

            occupied_grid = np.zeros(self.H * self.W, dtype=bool)
            pss_next_batches = []

            def append_first_unoccupied(points: np.ndarray) -> None:
                if len(points) == 0:
                    return

                grid_indices = np.round(
                    (points - np.asarray(cage_center, dtype=float)) / self.resolution
                    + np.array([self.W / 2, self.H / 2])
                ).astype(int)
                in_bounds = (
                    (0 <= grid_indices[:, 0])
                    & (grid_indices[:, 0] < self.W)
                    & (0 <= grid_indices[:, 1])
                    & (0 <= grid_indices[:, 1])
                    & (grid_indices[:, 1] < self.H)
                )
                if not np.any(in_bounds):
                    return

                points_in_bounds = points[in_bounds]
                grid_indices = grid_indices[in_bounds]
                flat_grid_indices = grid_indices[:, 1] * self.W + grid_indices[:, 0]

                first_indices = np.full(self.H * self.W, len(flat_grid_indices), dtype=np.int64)
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

            append_first_unoccupied(self.pss[d_cons_negative])

            # For contacted states, add the discretized quasi-static motion set
            # V(u, d_con). Batch states with the same discretized ellipse size so
            # each offset set is generated once, then use broadcasting to expand
            # chunks of each batch. Deduplicate immediately to avoid materializing
            # millions of candidates that collapse onto the same grid cells.
            contacted_points = self.pss[d_cons_positive]
            contacted_d_cons = d_cons[d_cons_positive]

            if len(contacted_points) > 0:
                nxs = np.ceil(contacted_d_cons / self.resolution).astype(int)
                nys = np.ceil((0.5 * contacted_d_cons) / self.resolution).astype(int)
                ellipse_bins = np.column_stack((nxs, nys))
                unique_bins, inverse_bins = np.unique(ellipse_bins, axis=0, return_inverse=True)

                c_u = np.cos(u)
                s_u = np.sin(u)
                R_u = np.array([[c_u, -s_u], [s_u, c_u]])
                max_candidates_per_chunk = 2_000_000

                for bin_idx, (nx, ny) in enumerate(unique_bins):
                    batch_points = contacted_points[inverse_bins == bin_idx]
                    d_con = float(contacted_d_cons[inverse_bins == bin_idx].max())

                    ix = np.arange(-nx, 1, dtype=float)
                    iy = np.arange(-ny, ny + 1, dtype=float)
                    dx, dy = np.meshgrid(ix * self.resolution, iy * self.resolution, indexing="ij")

                    ellipse_values = (dx / d_con) ** 2 + (dy / (0.5 * d_con)) ** 2
                    valid = ellipse_values <= 1.0 + 1e-12
                    local_offsets = np.column_stack((dx[valid], dy[valid]))

                    if not np.any(np.all(np.isclose(local_offsets, 0.0), axis=1)):
                        local_offsets = np.vstack((local_offsets, np.array([[0.0, 0.0]])))

                    offsets = local_offsets @ R_u.T
                    chunk_size = max(1, max_candidates_per_chunk // len(offsets))
                    for start_idx in range(0, len(batch_points), chunk_size):
                        chunk_points = batch_points[start_idx : start_idx + chunk_size]
                        moved_points = (
                            chunk_points[:, np.newaxis, :] + offsets[np.newaxis, :, :]
                        ).reshape(-1, 2)
                        append_first_unoccupied(moved_points)

            if pss_next_batches:
                pss_next = np.vstack(pss_next_batches)
            else:
                pss_next = np.empty((0, 2), dtype=float)

        self.pss = pss_next
        # end = time()
        # print(f"Propagate time: {end - start} seconds")



        # Old unbatched code for reference:
        # pss_next = PSS(self.params, cage_center)

        # start = time()
        # for x_object_maybe, y_object_maybe in self.pss:
        #     if u is None:
        #         # No push action, so the object stays where it is.
        #         pss_next.add(x_object_maybe, y_object_maybe)
        #         continue

        #     # Get the distance from the potential object position to the pusher
        #     dist = d_q_L(x_object_maybe, y_object_maybe, u, cage_center, self.R, self.L)

        #     # Get the distance over which the object is pushed
        #     d_con = max(0, min(self.d_push, self.r + self.d_push - dist))

        #     if d_con <= 0:
        #         pss_next.add(x_object_maybe, y_object_maybe)
        #         continue
            
        #     V = discretized_V_pixel_offsets(u, d_con, self.resolution)

        #     for dx, dy in V:
        #         x_next = x_object_maybe + dx
        #         y_next = y_object_maybe + dy

        #         pss_next.add(x_next, y_next)
        
        # end = time()
        # print(f"Propagate time: {end - start} seconds")
        # self.pss = pss_next

        if render:
            # start = time()
            image = self.render(u)
            # end = time()
            # print(f"Render time: {end - start} seconds")
            return image

    def potential_object_area_points(
        self, pss: PSS, cage_center: tuple[float, float]
    ) -> np.ndarray:
        """
        Approximate the Potential Object Area by dilating each PSS cell by the
        object's bounding-circle radius.
        """
        if len(pss) == 0:
            return np.empty((0, 2), dtype=np.float64)

        xs = pss[:, 0]
        ys = pss[:, 1]
        center_i, center_j = self.world_to_grid_batch(xs, ys, cage_center)

        in_bounds = (
            (0 <= center_i)
            & (center_i < self.H)
            & (0 <= center_j)
            & (center_j < self.W)
        )
        center_i = center_i[in_bounds]
        center_j = center_j[in_bounds]

        if len(center_i) == 0:
            return np.empty((0, 2), dtype=np.float64)

        occupancy = np.zeros((self.H, self.W), dtype=np.uint8)
        occupancy[center_i, center_j] = 1

        dilated = cv2.dilate(occupancy, self._poa_kernel, iterations=1)
        occupied_i, occupied_j = np.nonzero(dilated)

        cx, cy = cage_center
        points = np.empty((len(occupied_i), 2), dtype=np.float64)
        points[:, 0] = cx + (occupied_j - self.W / 2) * self.resolution
        points[:, 1] = cy + (occupied_i - self.H / 2) * self.resolution

        return points
    
    def find_push(self):
        next_cage_center = self.trajectory[self.t + 1]
        cx, cy = next_cage_center
        cage_center_arr = np.array([cx, cy], dtype=float)

        rel_pss_to_cage = self.pss - cage_center_arr
        pss_radius_sq = np.sum(rel_pss_to_cage * rel_pss_to_cage, axis=1)

        # If the PSS is contained within the next cage, do nothing.
        if np.all(pss_radius_sq <= self.cage_radius * self.cage_radius):
            return None

        poa_points = self.potential_object_area_points(self.pss, next_cage_center)
        cell_area = self.resolution * self.resolution

        angles = self._action_angles
        radials = self._action_radials
        tangents = self._action_tangents
        pusher_centers = cage_center_arr[np.newaxis, :] + self.R * radials
        push_dirs = -radials

        if poa_points.shape[0] > 0:
            rel_poa = poa_points[np.newaxis, :, :] - pusher_centers[:, np.newaxis, :]
            poa_outward_depth = np.einsum("knd,kd->kn", rel_poa, radials)
            on_pusher_side = poa_outward_depth >= -self.r
            within_pusher_span = np.abs(np.einsum("knd,kd->kn", rel_poa, tangents)) <= 0.5 * self.L
            poa_behind_pusher = on_pusher_side & within_pusher_span

            # a_k: area of the POA behind the pusher, using the pusher-side
            # half-plane so the opposite-side pusher is not rewarded for seeing
            # the entire cage interior as being behind it.
            poa_area_behind = np.count_nonzero(poa_behind_pusher, axis=1).astype(float) * cell_area

            # b_k: furthest POA depth into the pusher-side half-plane.
            poa_depth_behind_pusher = poa_outward_depth + self.r
            poa_depth_behind_pusher = np.where(poa_behind_pusher, poa_depth_behind_pusher, 0.0)
            furthest_dist = np.max(poa_depth_behind_pusher, axis=1)
        else:
            poa_area_behind = np.zeros(self.K, dtype=float)
            furthest_dist = np.zeros(self.K, dtype=float)

        print(poa_area_behind.mean(), (furthest_dist * furthest_dist).mean())
        scores = self.lambda_1 * poa_area_behind + self.lambda_2 * furthest_dist * furthest_dist

        # Sort angles by score and choose among the top-scoring actions using
        # the paper's tie-breaking rule: prefer the action closest to the last
        # pusher angle to reduce unnecessary pusher reorientation.
        sorted_indices = np.argsort(scores)

        if self.last_action_angle is None:
            u = angles[sorted_indices[-1]]
        else:
            top_n = min(5, len(angles))
            top_angles = angles[sorted_indices[-top_n:]]
            u = min(
                top_angles,
                key=lambda angle: abs((angle - self.last_action_angle + pi) % tau - pi),
            )

        return u

    # @property
    # def I(self) -> np.ndarray:
    #     I = np.zeros((self.H, self.W), dtype=float)
    #     for i, j in self.pss.grid_cells():
    #         I[i, j] = 1.0
    #     return I
    
    def render(self, theta: float | None = None) -> np.ndarray:
        image = np.zeros((self.H, self.W, 3), dtype=np.uint8)

        cage_center = self.trajectory[self.t]
        xc, yc = cage_center

        # plt.imshow(image, cmap='gray')
        # plt.plot([p1_j, p2_j], [p1_i, p2_i], color="red", linewidth=3, label="Active Pusher")

        # Draw the cage boundary (outer circle)
        ic, jc = self.world_to_grid(xc, yc, cage_center)
        cv2.circle(image, (jc, ic), int(self.R / self.resolution), (96, 169, 101), 1)

        # Draw the trajectory
        for x, y in self.trajectory:
            i, j = self.world_to_grid(x, y, cage_center)
            if 0 <= i < self.H and 0 <= j < self.W:
                image[i, j] = (0, 0, 255)
        
        # Draw the POA
        # for i, j in self.pss.grid_cells():
        # Convert PSS to grid coordinates
        all_i, all_j = self.world_to_grid_batch(self.pss[:, 0], self.pss[:, 1], cage_center)
        for i, j in zip(all_i, all_j):
            # Draw a circle at the POA
            # First, convert the object radius to grid units
            radius_grid = int(self.params.object_radius / self.resolution)
            cv2.circle(image, (j, i), radius_grid, (96, 169, 101), -1)
        
        # Draw the PSS
        #for i, j in self.pss.grid_cells():
        for i, j in zip(all_i, all_j):
            image[i, j] = (255, 255, 255)

        # Draw the pusher
        if theta is not None:
            pusher_x = xc + self.R * cos(theta)
            pusher_y = yc + self.R * sin(theta)

            p1_x = pusher_x - self.L / 2 * sin(theta)
            p1_y = pusher_y + self.L / 2 * cos(theta)

            p2_x = pusher_x + self.L / 2 * sin(theta)
            p2_y = pusher_y - self.L / 2 * cos(theta)

            # Convert to grid coordinates
            p1_i, p1_j = self.world_to_grid(p1_x, p1_y, cage_center)
            p2_i, p2_j = self.world_to_grid(p2_x, p2_y, cage_center)

            # Draw cv2 line on image
            cv2.line(image, (p1_j, p1_i), (p2_j, p2_i), (255, 0, 0), 1)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (0, 0), fx=3, fy=3)
        return image

    def render_world(self, theta: float | None = None) -> np.ndarray:
        """
        Render the simulator in a fixed world frame instead of centering the
        image on the current cage position. The image bounds are chosen
        dynamically so the full trajectory and the outer cage at every trajectory
        point are visible.
        """
        trajectory_arr = np.asarray(self.trajectory, dtype=float)
        margin = self.R + self.resolution

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

        cage_center = self.trajectory[self.t]
        xc, yc = cage_center

        # Draw the cage boundary at the current world position.
        ic, jc = world_to_image(xc, yc)
        cv2.circle(image, (jc, ic), int(self.R / self.resolution), (96, 169, 101), 1)

        # Draw the full trajectory in the fixed world frame.
        trajectory_i, trajectory_j = world_to_image_batch(trajectory_arr)
        in_bounds = (
            (0 <= trajectory_i)
            & (trajectory_i < world_H)
            & (0 <= trajectory_j)
            & (trajectory_j < world_W)
        )
        image[trajectory_i[in_bounds], trajectory_j[in_bounds]] = (0, 0, 255)

        # Draw the POA induced by the current PSS.
        if len(self.pss) > 0:
            all_i, all_j = world_to_image_batch(self.pss)
            in_bounds = (
                (0 <= all_i)
                & (all_i < world_H)
                & (0 <= all_j)
                & (all_j < world_W)
            )
            all_i = all_i[in_bounds]
            all_j = all_j[in_bounds]

            radius_grid = int(self.params.object_radius / self.resolution)
            for i, j in zip(all_i, all_j):
                cv2.circle(image, (j, i), radius_grid, (96, 169, 101), -1)

            image[all_i, all_j] = (255, 255, 255)

        # Draw the active pusher at the current world position.
        if theta is not None:
            pusher_x = xc + self.R * cos(theta)
            pusher_y = yc + self.R * sin(theta)

            p1_x = pusher_x - self.L / 2 * sin(theta)
            p1_y = pusher_y + self.L / 2 * cos(theta)

            p2_x = pusher_x + self.L / 2 * sin(theta)
            p2_y = pusher_y - self.L / 2 * cos(theta)

            p1_i, p1_j = world_to_image(p1_x, p1_y)
            p2_i, p2_j = world_to_image(p2_x, p2_y)

            cv2.line(image, (p1_j, p1_i), (p2_j, p2_i), (255, 0, 0), 1)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image


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


if __name__ == "__main__":
    cage_size = 0.04 # R - r = 20 mm
    object_radius = 0.025 # r = 25 mm
    pusher_length = 0.1 # L = 100 mm
    push_distance = 0.02 # d_push = 20 mm

    params = SimParams(
        object_radius=object_radius,
        pusher_length=pusher_length,
        push_distance=push_distance,
        cage_size=cage_size,
        resolution=0.002,
        grid_height=201,
        grid_width=201,
        num_actions=128,
    )

    # Create a long trajectory of 100 waypoints
    waypoints = create_circular_trajectory(radius=0.2, speed=0.001)

    x0, y0 = waypoints[0]

    sim = Simulator(params, waypoints)
    sim.reset(x0, y0)

    for i in range(len(waypoints) - 1):
        # start = time()
        u = sim.find_push()
        # end = time()
        # print(f"Find push time: {end - start} seconds")
        local_image = sim.propagate(u, render=True)
        world_image = sim.render_world(u)

        if local_image is not None:
            local_h, local_w = local_image.shape[:2]
            world_h, world_w = world_image.shape[:2]
            resized_local_w = int(round(local_w * world_h / local_h))
            resized_local_image = cv2.resize(local_image, (resized_local_w, world_h))
            image = np.hstack((resized_local_image, world_image))
        else:
            image = world_image

        cv2.imshow('image', image)
        cv2.waitKey(1)

        # if sim.pss.out_of_bounds:
        #     print(f"Object is out of bounds at step {i}")
        #     break

        

        # plt.imshow(image, cmap='gray')
        # plt.plot([p1_j, p2_j], [p1_i, p2_i], color="red", linewidth=3, label="Active Pusher")
        # plt.xlim(0, sim.W)
        # plt.ylim(0, sim.H)
        # plt.show()