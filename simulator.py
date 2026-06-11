from dataclasses import dataclass
from math import tau, sin, cos, tau, pi
from typing import Iterator

import matplotlib.pyplot as plt
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
    
    def get_action_angle(self, k: int) -> float:
        return tau * k / self.K

    def world_to_grid(self, x: float, y: float, cage_center: tuple[float, float]) -> tuple[int, int]:
        cage_cx, cage_cy = cage_center
        j = int(round((x - cage_cx) / self.resolution + self.W / 2))
        i = int(round((y - cage_cy) / self.resolution + self.H / 2))
        return i, j

    def reset(self, x_object: float, y_object: float):
        self.t = 0
        self.pss = PSS(self.params, self.trajectory[0])
        self.pss.add(x_object, y_object)

    def propagate(self, u: float | None, render: bool = False) -> np.ndarray | None:
        """
        Propagate the PSS one step under the selected action.

        Args:
            u: The action angle in radians.
        """
        self.t += 1
        cage_center = self.trajectory[self.t]

        pss_next = PSS(self.params, cage_center)

        for x_object_maybe, y_object_maybe in self.pss:
            if u is None:
                # No push action, so the object stays where it is.
                pss_next.add(x_object_maybe, y_object_maybe)
                continue

            # Get the distance from the potential object position to the pusher
            dist = d_q_L(x_object_maybe, y_object_maybe, u, cage_center, self.R, self.L)

            # Get the distance over which the object is pushed
            d_con = max(0, min(self.d_push, self.r + self.d_push - dist))

            if d_con <= 0:
                pss_next.add(x_object_maybe, y_object_maybe)
                continue
            
            V = discretized_V_pixel_offsets(u, d_con, self.resolution)

            for dx, dy in V:
                x_next = x_object_maybe + dx
                y_next = y_object_maybe + dy

                pss_next.add(x_next, y_next)

        self.pss = pss_next

        if render:
            return self.render()
    
    def find_push(self):
        next_cage_center = self.trajectory[self.t + 1]
        cx, cy = next_cage_center

        # If the PSS is contained within the next cage, do nothing
        is_contained = True
        for x, y in self.pss:
            dx = x - cx
            dy = y - cy
            if dx * dx + dy * dy > self.cage_radius * self.cage_radius:
                is_contained = False
                break

        if is_contained:
            return None
        
        for k in range(self.K):
            # Compute the furthest distance from behind the line pusher
            # to the POA boundary
            # TODO: Implement this
        return None
    
    @property
    def I(self) -> np.ndarray:
        I = np.zeros((self.H, self.W), dtype=float)
        for i, j in self.pss.grid_cells():
            I[i, j] = 1.0
        return I
    
    def render(self) -> np.ndarray:
        image = np.zeros((self.H, self.W, 3), dtype=np.uint8)

        cage_center = self.trajectory[sim.t]
        xc, yc = cage_center

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
        for i, j in self.pss.grid_cells():
            # Draw a circle at the POA
            # First, convert the object radius to grid units
            radius_grid = int(self.params.object_radius / self.resolution)
            cv2.circle(image, (j, i), radius_grid, (96, 169, 101), 1)
        
        # Draw the PSS
        for i, j in self.pss.grid_cells():
            image[i, j] = (255, 255, 255)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (0, 0), fx=3, fy=3)
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
    params = SimParams(
        object_radius=0.025,
        pusher_length=0.1,
        push_distance=0.025,
        cage_size=0.05,
        resolution=0.002,
        grid_height=151,
        grid_width=151,
    )

    # Create a long trajectory of 100 waypoints
    waypoints = create_circular_trajectory(radius=1, speed=0.001)

    x0, y0 = waypoints[0]

    sim = Simulator(params, waypoints)
    sim.reset(x0, y0)

    for i in range(len(waypoints) - 1):
        # k = 9#np.random.randint(params.num_actions)
        if i < 30:
            k = 9
        else:
            k = 0

        theta = sim.get_action_angle(k)
        image = sim.propagate(theta, render=True)
        
        cv2.imshow('image', image)
        cv2.waitKey(1)

        if sim.pss.out_of_bounds:
            print(f"Object is out of bounds at step {i}")
            break

        

        # plt.imshow(image, cmap='gray')
        # plt.plot([p1_j, p2_j], [p1_i, p2_i], color="red", linewidth=3, label="Active Pusher")
        # plt.xlim(0, sim.W)
        # plt.ylim(0, sim.H)
        # plt.show()