from dataclasses import dataclass
from math import tau

import matplotlib.pyplot as plt
import numpy as np

# from pss import GridPSS


import numpy as np

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
    if d_con <= 0.0:
        return [np.array([0.0, 0.0])]
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


@dataclass(frozen=True)
class ObjectState:
    x: float
    y: float


@dataclass(frozen=True)
class Trajectory:
    waypoints: list[ObjectState]


class Simulator:
    def __init__(self, params: SimParams, trajectory: Trajectory):
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
    
    def get_action_angle(self, k: int) -> float:
        return tau * k / self.K
    
    def world_to_grid(self, x: float, y: float, cage_center: tuple[float, float]) -> tuple[int, int]:
        x_cage, y_cage = cage_center
        j = int(round((x - x_cage) / self.resolution + self.W / 2))
        i = int(round((y - y_cage) / self.resolution + self.H / 2))
        return i, j
    
    def reset(self):
        self.t = 0
        self.q = ObjectState(x=0, y=0)

        self.Q = {self.q}
        self.I = np.zeros((self.H, self.W), dtype=bool)

        c0 = self.trajectory.waypoints[0]

        i, j = self.world_to_grid(c0.x, c0.y, (c0.x, c0.y))
        if 0 <= i < self.H and 0 <= j < self.W:
            self.I[i, j] = True
    
    def propagate(self, u: float | None):
        """
        Propagate the PSS one step under the selected action.

        Args:
            u: The action angle in radians.
        """
        Q_next = set()
        I_next = np.zeros((self.H, self.W), dtype=bool)

        x_traj = self.trajectory.waypoints[self.t + 1].x
        y_traj = self.trajectory.waypoints[self.t + 1].y

        x_cage = x_traj
        y_cage = y_traj
        cage_center = (x_cage, y_cage)

        for q in self.Q:
            x_object = q.x
            y_object = q.y

            if u is None:
                # No push action, so the object stays where it is.
                V = [np.zeros(2)]
            else:
                dist = d_q_L(x_object, y_object, u, cage_center, self.R, self.L)
            
                if dist > self.r + self.d_push:
                    # no contact
                    d_con = 0
                elif dist <= self.r:
                    # already touching / intersecting
                    d_con = self.d_push
                else:
                    d_con = self.d_push - (dist - self.r)
                
                if d_con <= 0:
                    V = [np.zeros(2)]
                else:
                    V = discretized_V_pixel_offsets(u, d_con, self.resolution)
            
            for v in V:
                dx = v[0]
                dy = v[1]

                x_next = x_object + dx
                y_next = y_object + dy

                Q_next.add(ObjectState(x=x_next, y=y_next))

                # Get world to grid coordinates
                i, j = self.world_to_grid(x_next, y_next, cage_center)

                if 0 <= i < self.H and 0 <= j < self.W:
                    I_next[i, j] = True

        self.Q = Q_next
        self.I = I_next

        self.t += 1


if __name__ == "__main__":
    params = SimParams(
        object_radius=0.025,
        pusher_length=0.30,
        push_distance=0.015,
        cage_size=0.10,
        resolution=0.002,
        grid_height=151,
        grid_width=151,
    )

    # Create a long trajectory of 100 waypoints
    waypoints = []
    x = 0.0
    y = 0.0
    for i in range(100):
        x += 0.01
        y += 0.01
        waypoints.append(ObjectState(x=x, y=y))

    trajectory = Trajectory(waypoints=waypoints)

    sim = Simulator(params, trajectory)
    sim.reset()

    for _ in range(len(waypoints)):
        k = np.random.randint(params.num_actions)
        theta = sim.get_action_angle(k)

        sim.propagate(theta)

        I = sim.I.astype(np.uint8) * 255

        plt.imshow(I, cmap='gray')
        plt.show()