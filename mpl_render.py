"""
Matplotlib rendering for the Caging in Time simulator, reproducing the
paper's figures (Wang et al. 2024, Figs. 9-10): cage-centered panels with
the POA in green, the PSS in white, candidate pushers as grey bars, the
selected pusher in red, and a yellow arrow showing the cage's motion.

SciencePlots styles are loaded directly from the installed package's
``.mplstyle`` files. This avoids ``import scienceplots``, which currently
breaks on Matplotlib 3.11+ because it calls the removed
``matplotlib.style.core.read_style_directory`` API.
"""

from __future__ import annotations

import importlib.util
from contextlib import contextmanager
from math import cos, sin
from pathlib import Path
from typing import Iterator, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.patches import Arc, Circle

DEFAULT_STYLES: tuple[str, ...] = ("science", "no-latex")
DEFAULT_PANEL_SIZE = 3.0
DEFAULT_SAVE_DPI = 600

# Palette matched to the paper's renderings (Figs. 9-10).
BACKGROUND_COLOR = "#000000"
CAGE_COLOR = "#1e6b34"        # dark green circle (S_A^t boundary)
POA_COLOR = "#5da662"         # green POA region
PSS_COLOR = "#ffffff"         # white PSS pixels
CANDIDATE_COLOR = "#7f7f7f"   # grey candidate bars
ACTIVE_COLOR = "#e8262b"      # red selected pusher
ARROW_COLOR = "#f5d327"       # yellow cage-motion arrow


def _hex_to_rgb01(hex_color: str) -> tuple[float, float, float]:
    value = hex_color.lstrip("#")
    return tuple(int(value[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def _scienceplots_styles_root() -> Path:
    spec = importlib.util.find_spec("scienceplots")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError(
            "scienceplots is not installed. Install it with: uv add scienceplots"
        )
    return Path(spec.submodule_search_locations[0]) / "styles"


def resolve_style_path(name: str, styles_root: Path | None = None) -> Path:
    root = styles_root or _scienceplots_styles_root()
    direct = root / f"{name}.mplstyle"
    if direct.is_file():
        return direct
    matches = sorted(root.rglob(f"{name}.mplstyle"))
    if not matches:
        raise ValueError(
            f"Unknown SciencePlots style {name!r}. "
            f"Look for .mplstyle files under {root}."
        )
    return matches[0]


def parse_style_names(styles: str | Sequence[str] | None) -> list[str]:
    if styles is None:
        return list(DEFAULT_STYLES)
    if isinstance(styles, str):
        return [part.strip() for part in styles.split(",") if part.strip()]
    return [str(part) for part in styles]


@contextmanager
def use_scienceplots_styles(
    styles: str | Sequence[str] | None = None,
) -> Iterator[None]:
    """Apply SciencePlots ``.mplstyle`` files for the duration of the context."""
    style_names = parse_style_names(styles)
    styles_root = _scienceplots_styles_root()
    style_paths = [str(resolve_style_path(name, styles_root)) for name in style_names]
    with plt.style.context(style_paths):
        yield


def snapshot(sim, theta: float | None) -> dict:
    """Capture the simulator state needed to draw one paper-style frame."""
    t = sim.t
    cage_center = np.asarray(sim.trajectory[t], dtype=float)
    if t + 1 < len(sim.trajectory):
        motion = np.asarray(sim.trajectory[t + 1], dtype=float) - cage_center
    else:
        motion = np.zeros(2)
    return {
        "t": t,
        "cage_center": cage_center,
        "motion": motion,
        "pss": np.array(sim.pss, dtype=float, copy=True),
        "theta": theta,
        "R": sim.R,
        "r": sim.r,
        "L": sim.L,
        "K": sim.K,
        "d_push": sim.d_push,
        "resolution": sim.resolution,
    }


def snapshot_from_caging_env(env, theta: float | None) -> dict:
    """
    Capture a PolygonCagingEnv state for paper-style rendering. Includes the
    true polygon outline so belief (PSS/POA) and reality can be compared.
    """
    t = env.t
    cage_center = np.asarray(env.trajectory[t], dtype=float)
    if t + 1 < len(env.trajectory):
        motion = np.asarray(env.trajectory[t + 1], dtype=float) - cage_center
    else:
        motion = np.zeros(2)

    from quasistatic_env import transform_polygon

    polygon_world = transform_polygon(env.polygon_vertices_body, env.object_pose)
    return {
        "t": t,
        "cage_center": cage_center,
        "motion": motion,
        "pss": np.array(env.pss, dtype=float, copy=True),
        "theta": theta,
        "R": env.R,
        "r": env.r,
        "L": env.L,
        "K": env.K,
        "d_push": env.d_push,
        "resolution": env.resolution,
        "polygon": polygon_world,
    }


def _pusher_segment(
    cage_center: np.ndarray, theta: float, outer_radius: float, pusher_length: float
) -> tuple[np.ndarray, np.ndarray]:
    center = cage_center + outer_radius * np.array([cos(theta), sin(theta)])
    tangent = 0.5 * pusher_length * np.array([-sin(theta), cos(theta)])
    return center - tangent, center + tangent


def _pss_poa_rgba(
    snap: dict,
    extent: float,
) -> tuple[np.ndarray, list[float]] | tuple[None, None]:
    """
    Rasterize the PSS and POA on the simulation grid, matching the paper's
    discretized representation (Alg. 3 / Fig. 9).
    """
    cx, cy = snap["cage_center"]
    r = snap["r"]
    resolution = snap["resolution"]
    pss = snap["pss"]
    if len(pss) == 0:
        return None, None

    n = int(np.ceil(2 * extent / resolution)) + 1
    occupancy = np.zeros((n, n), dtype=np.uint8)
    for point in pss:
        x, y = point[:2]
        j = int(round((x - cx) / resolution + n / 2))
        i = int(round((y - cy) / resolution + n / 2))
        if 0 <= i < n and 0 <= j < n:
            occupancy[i, j] = 1

    if not occupancy.any():
        return None, None

    radius_cells = int(np.ceil(r / resolution))
    offsets = np.arange(-radius_cells, radius_cells + 1)
    dj, di = np.meshgrid(offsets, offsets, indexing="xy")
    kernel = (
        (dj * resolution) ** 2 + (di * resolution) ** 2 <= r * r + 1e-12
    ).astype(np.uint8)
    poa = cv2.dilate(occupancy, kernel, iterations=1)

    # The pusher's final pose displaces the object, so no POA mass can
    # remain behind (outside of) the pushed-in pusher line. Clip raster
    # cells in the swept channel to reflect the deformed POA (Fig. 10).
    if snap["theta"] is not None:
        theta = snap["theta"]
        R = snap["R"]
        L = snap["L"]
        d_push = snap["d_push"]
        coords = (np.arange(n) - n / 2) * resolution
        gx, gy = np.meshgrid(coords, coords, indexing="xy")
        radial_proj = gx * cos(theta) + gy * sin(theta)
        tangential_proj = -gx * sin(theta) + gy * cos(theta)
        behind_pusher = (radial_proj > R - d_push) & (
            np.abs(tangential_proj) <= 0.5 * L
        )
        poa[behind_pusher] = 0
        occupancy[behind_pusher] = 0

    poa_rgb = _hex_to_rgb01(POA_COLOR)
    rgba = np.zeros((n, n, 4), dtype=np.float32)
    rgba[poa > 0] = (*poa_rgb, 1.0)
    rgba[occupancy > 0] = (1.0, 1.0, 1.0, 1.0)

    image_extent = [
        cx - extent,
        cx + extent,
        cy - extent,
        cy + extent,
    ]
    return rgba, image_extent


def _draw_frame(
    ax: Axes,
    snap: dict,
    *,
    show_candidates: bool = True,
    show_arrow: bool = True,
    show_time_label: bool = True,
) -> None:
    """Draw one cage-centered frame in the style of Fig. 10 of the paper."""
    cx, cy = snap["cage_center"]
    R, r, L, K = snap["R"], snap["r"], snap["L"], snap["K"]

    # Frame extent: the cage circle plus room for the pusher bars.
    extent = R + 0.55 * L

    ax.set_xlim(cx - extent, cx + extent)
    ax.set_ylim(cy - extent, cy + extent)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor(BACKGROUND_COLOR)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#444444")
        spine.set_linewidth(0.8)

    # Scale line widths to the panel's data extent so strokes stay visible.
    data_span = 2.0 * extent
    lw_scale = data_span / 0.20

    # Candidate pushers (grey bars surrounding the cage). When K is large,
    # subsample for display so the ring stays readable (Fig. 10 uses K=32).
    if show_candidates:
        display_count = min(K, 32)
        step = max(1, K // display_count)
        candidate_alpha = float(np.clip(12.0 / display_count, 0.2, 0.55))
        segments = []
        for k in range(0, K, step):
            angle = 2.0 * np.pi * k / K
            p1, p2 = _pusher_segment(snap["cage_center"], angle, R, L)
            segments.append([p1, p2])
        ax.add_collection(
            LineCollection(
                segments,
                colors=CANDIDATE_COLOR,
                linewidths=1.1 * lw_scale,
                alpha=candidate_alpha,
                capstyle="round",
                antialiaseds=True,
                zorder=2,
            )
        )

    # S_A^t boundary (dark green circle). When a push is executed, the
    # pusher sweeps through the boundary, so truncate the outline over the
    # arc spanned by the pusher (as in Fig. 10 of the paper).
    if snap["theta"] is not None:
        theta_deg = np.degrees(snap["theta"])
        # The circle is hidden where it lies outside the pushed-in pusher
        # line (radial depth) and within the pusher's tangential span.
        d_push = snap["d_push"]
        half_gap_radial = np.arccos(np.clip((R - d_push) / R, -1.0, 1.0))
        half_gap_span = np.arcsin(min(1.0, 0.5 * L / R))
        half_gap_deg = np.degrees(min(half_gap_radial, half_gap_span))
        ax.add_patch(
            Arc(
                (cx, cy),
                2 * R,
                2 * R,
                theta1=theta_deg + half_gap_deg,
                theta2=theta_deg - half_gap_deg + 360.0,
                edgecolor=CAGE_COLOR,
                linewidth=1.6 * lw_scale,
                zorder=3,
            )
        )
    else:
        ax.add_patch(
            Circle(
                (cx, cy),
                R,
                fill=False,
                edgecolor=CAGE_COLOR,
                linewidth=1.6 * lw_scale,
                zorder=3,
            )
        )

    # POA and PSS on the simulation grid (crisp at any output size).
    rgba, image_extent = _pss_poa_rgba(snap, extent)
    if rgba is not None:
        ax.imshow(
            rgba,
            extent=image_extent,
            origin="lower",
            interpolation="nearest",
            zorder=4,
        )

    # Selected pusher (red bar), drawn at its fully pushed-in position,
    # i.e. after sweeping d_push toward the cage center.
    if snap["theta"] is not None:
        p1, p2 = _pusher_segment(
            snap["cage_center"], snap["theta"], R - snap["d_push"], L
        )
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            color=ACTIVE_COLOR,
            linewidth=2.4 * lw_scale,
            solid_capstyle="round",
            zorder=6,
        )

    # True polygon outline (RL envs), drawn over the belief so the real
    # object can be compared against the PSS/POA.
    polygon = snap.get("polygon")
    if polygon is not None and len(polygon) >= 3:
        closed = np.vstack((polygon, polygon[:1]))
        ax.plot(
            closed[:, 0],
            closed[:, 1],
            color="#66ccff",
            linewidth=1.2 * lw_scale,
            zorder=5,
        )

    # Yellow arrow showing the cage's motion direction.
    motion = snap["motion"]
    norm = np.linalg.norm(motion)
    if show_arrow and norm > 0:
        direction = motion / norm
        arrow_len = 0.45 * R
        ax.annotate(
            "",
            xy=(cx + direction[0] * arrow_len, cy + direction[1] * arrow_len),
            xytext=(cx, cy),
            arrowprops=dict(
                arrowstyle="-|>",
                color=ARROW_COLOR,
                linewidth=1.8 * lw_scale,
                mutation_scale=12 * lw_scale,
            ),
            zorder=7,
        )

    if show_time_label:
        ax.set_title(f"$t = {snap['t']}$", fontsize=10, pad=4)


def render_frame(
    sim,
    theta: float | None = None,
    *,
    styles: str | Sequence[str] | None = None,
    show_candidates: bool = True,
    show_arrow: bool = True,
    panel_size: float = DEFAULT_PANEL_SIZE,
) -> Figure:
    """Render the current simulator state as a single paper-style panel."""
    snap = snapshot(sim, theta)
    with use_scienceplots_styles(styles):
        fig, ax = plt.subplots(figsize=(panel_size, panel_size))
        _draw_frame(
            ax, snap, show_candidates=show_candidates, show_arrow=show_arrow
        )
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.88)
    return fig


def render_sequence(
    snapshots: Sequence[dict],
    *,
    styles: str | Sequence[str] | None = None,
    show_candidates: bool = True,
    show_arrow: bool = True,
    max_cols: int = 6,
    panel_size: float = DEFAULT_PANEL_SIZE,
) -> Figure:
    """
    Render a strip of frames like Fig. 10 of the paper: the PSS and POA
    evolving relative to the moving cage, one panel per captured time step.
    """
    n = len(snapshots)
    if n == 0:
        raise ValueError("No snapshots to render.")

    ncols = min(max_cols, n)
    nrows = (n + ncols - 1) // ncols

    with use_scienceplots_styles(styles):
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(panel_size * ncols, (panel_size + 0.3) * nrows),
            squeeze=False,
        )
        for idx, snap in enumerate(snapshots):
            ax = axes[idx // ncols][idx % ncols]
            _draw_frame(
                ax, snap, show_candidates=show_candidates, show_arrow=show_arrow
            )
        for idx in range(n, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)
        fig.subplots_adjust(
            left=0.01,
            right=0.99,
            bottom=0.02,
            top=0.90,
            wspace=0.06,
            hspace=0.22,
        )
    return fig
