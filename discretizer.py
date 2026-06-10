import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Set, Tuple, List, Optional

@dataclass(frozen=True)
class ObjectState:
    x: float
    y: float

class CagingDiscretizer:
    def __init__(
        self, 
        grid_height: int = 151, # height of binary mask matrix
        grid_width: int = 151, # width of binary mask matrix
        resolution: float = 0.002, # meters per pixel
        cage_radius: float = 0.05 # radius of cage boundary
    ):
        self.H = grid_height
        self.W = grid_width
        self.res = resolution
        self.cage_rad = cage_radius
        
        # Center indices of grid
        self.cxidx = grid_width // 2
        self.cyidx = grid_height // 2
        
        # PSS state representation
        self.Q: Set[ObjectState] = set()
        self.I = np.zeros((self.H, self.W), dtype=float)

    def world2grid(self, x: float, y: float, center: Tuple[float, float]) -> Tuple[int, int]:
        dx = x - center[0]
        dy = y - center[1]
        j = int(round(dx / self.res)) + self.cxidx
        i = int(round(dy / self.res)) + self.cyidx
        return i, j

    def grid2world(self, i: int, j: int, center: Tuple[float, float]) -> Tuple[float, float]:
        dx = (j - self.cxidx) * self.res
        dy = (i - self.cyidx) * self.res
        return center[0] + dx, center[1] + dy

    def entropy(self) -> float:
        probs = self.I[self.I > 0.0]
        if len(probs) == 0:
            return 0.0
        probs = -np.sum(probs * np.log(probs))
        return float(probs)

    def set_pss(self, states: List[ObjectState], center: Tuple[float, float]):
        self.Q = set(states)
        active_cells = set()
        for state in states:
            i, j = self.world2grid(state.x, state.y, center)
            if 0 <= i < self.H and 0 <= j < self.W:
                active_cells.add((i, j))

        if not active_cells:
            print('No active cells found')
            return
        
        for (i, j) in active_cells:
            self.I[i, j] = 1.0 / len(active_cells)

    def visualize(
        self, 
        center: Tuple[float, float], # center of cage
        pusher_angle: Optional[float] = None, 
        pusher_length: float = 0.30,
    ):
        fig, ax = plt.subplots(figsize=(6, 6))
            
        # grid bounds
        x_min = center[0] - (self.cxidx * self.res)
        x_max = center[0] + ((self.W - self.cxidx) * self.res)
        y_min = center[1] - (self.cyidx * self.res)
        y_max = center[1] + ((self.H - self.cyidx) * self.res)
        
        # plot PSS matrix
        ax.imshow(
            self.I, 
            cmap="Greens", 
            alpha=0.6, 
            extent=[x_min, x_max, y_min, y_max], 
            origin="lower"
        )
        
        # plot PSS object states
        if self.Q:
            xs = [q.x for q in self.Q]
            ys = [q.y for q in self.Q]
            ax.scatter(xs, ys, color="darkgreen", s=10, label="PSS Objects")
            
        cage = plt.Circle(
            center, 
            self.cage_rad, 
            color="blue", 
            fill=False, 
            linestyle="--", 
            linewidth=1.5, 
            label="Cage boundary"
        )
        ax.add_patch(cage)
        ax.plot(center[0], center[1], 'bx', label="Cage Center")

        # draw pusher
        if pusher_angle is not None:
            u = pusher_angle
            # Radial direction vector towards pusher mid-point
            radial = np.array([np.cos(u), np.sin(u)])
            # Tangent segment direction
            tangent = np.array([-np.sin(u), np.cos(u)])
            
            # Midpoint anchor directly on cage boundary
            p0 = np.array(center) + self.cage_rad * radial
            
            # Segment boundaries
            a_end = p0 - 0.5 * pusher_length * tangent
            b_end = p0 + 0.5 * pusher_length * tangent
            
            ax.plot([a_end[0], b_end[0]], [a_end[1], b_end[1]], color="red", linewidth=3, label="Active Pusher")
            
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal', 'box')
        ax.set_xlabel("X Workspace (m)")
        ax.set_ylabel("Y Workspace (m)")
        ax.set_title("Discretized Caging State Space Visualizer")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="upper right")
        
        plt.savefig("caging_pss_visualization.png", dpi=300) 
        plt.close()

if __name__ == "__main__":
    # Create a system scenario matching default simulation parameters
    discretizer = CagingDiscretizer(
        grid_height=151, 
        grid_width=151, 
        resolution=0.002, 
        cage_radius=0.05
    )
    
    # Define an active cage center frame
    current_cage_center = (0.1, 0.1)
    
    # Generate an explicit patch cluster of object states (representing tracking uncertainties)
    simulated_states = []
    for dx in np.linspace(-0.02, 0.01, 15):
        for dy in np.linspace(-0.03, 0.00, 15):
            # Object state position in world frame
            simulated_states.append(ObjectState(x=0.1 + dx, y=0.1 + dy))
            
    # Feed states directly into the discrete tracker 
    discretizer.set_pss(simulated_states, center=current_cage_center)
    
    print(f"Total probability mass: {np.sum(discretizer.I)}") # Must equal 1.0
    print(f"Initial System Entropy S(Q_t): {discretizer.entropy():.4f}")

    # Visualize the tracked state matrix mapping an active pusher at 225 degrees (in radians)
    discretizer.visualize(
        center=current_cage_center,
        pusher_angle=np.deg2rad(50),
        pusher_length=0.08,
    )