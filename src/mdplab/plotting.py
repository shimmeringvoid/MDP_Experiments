"""Shared figure style and the drawings used across the notebooks.

One palette is used everywhere, so that a colour means the same thing in every
figure: categorical hues in a fixed order for series, one blue ramp for
magnitudes, blue-grey-red for signed quantities, and the green/red "status"
colours reserved for the goal and the pits.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Affine2D

from .gridworld import FREE, GOAL, OBSTACLE, PIT, GridWorld

__all__ = [
    "SERIES",
    "INK",
    "MUTED",
    "GREY",
    "GOOD",
    "BAD",
    "SEQUENTIAL",
    "DIVERGING",
    "use_style",
    "repo_root",
    "savefig",
    "original_figure",
    "show_original",
    "band",
    "end_label",
    "plot_gridworld",
]

# --- palette -----------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
GREY = "#b4b2a9"  # de-emphasised series, obstacles

# categorical hues, always used in this order
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = SERIES

GOOD = "#0ca30c"  # the goal
BAD = "#d03b3b"  # pits

_BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQUENTIAL = LinearSegmentedColormap.from_list("mdplab_blue", _BLUE_RAMP)
DIVERGING = LinearSegmentedColormap.from_list(
    "mdplab_diverging", ["#a32a2a", "#d03b3b", "#eb9a8f", "#f0efec", "#9ec5f4", "#2a78d6", "#184f95"]
)


def use_style() -> None:
    """Apply the shared matplotlib style.  Call once at the top of a notebook."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "font.size": 10,
            "text.color": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.labelsize": 10,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRIDLINE,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "xtick.color": AXIS,
            "ytick.color": AXIS,
            "xtick.labelcolor": MUTED,
            "ytick.labelcolor": MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.prop_cycle": cycler(color=SERIES),
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "lines.markersize": 6.5,
            "lines.markeredgecolor": SURFACE,
            "lines.markeredgewidth": 1.4,
            "scatter.edgecolors": SURFACE,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "image.cmap": "mdplab_blue",
            "figure.constrained_layout.use": True,
        }
    )
    for cmap in (SEQUENTIAL, DIVERGING):
        if cmap.name not in mpl.colormaps:
            mpl.colormaps.register(cmap)


# --- files -------------------------------------------------------------------


def repo_root() -> Path:
    """Root of the repository (the folder holding ``pyproject.toml``)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    cwd = Path.cwd().resolve()
    for parent in (cwd, *cwd.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return cwd


def savefig(fig, name: str) -> Path:
    """Save a figure as ``figures/<name>.png`` in the repository and return the path."""
    folder = repo_root() / "figures"
    folder.mkdir(exist_ok=True)
    path = folder / f"{name}.png"
    fig.savefig(path)
    return path


_ORIGINALS = {
    "fig1a": "fig-001.png",
    "fig1b": "fig-000.png",
    "fig2_small": "fig-002.png",
    "fig2_all": "fig-003.png",
    "fig3": "fig-004.png",
    "fig4": "fig-005.png",
    "fig5": "fig-006.png",
    "fig6": "fig-007.png",
    "fig7": "fig-008.png",
    "fig8": "fig-009.png",
}


def original_figure(name: str) -> np.ndarray:
    """A figure of the 2014 report as an image array.  ``name`` is e.g. ``"fig7"``."""
    return plt.imread(repo_root() / "report" / "figures" / _ORIGINALS[name])


def show_original(name: str, ax=None, title: str | None = None):
    """Display a figure of the 2014 report on an axis, without ticks or frame."""
    if ax is None:
        _, ax = plt.subplots()
    ax.imshow(original_figure(name), interpolation="antialiased")
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    return ax


# --- small helpers -------------------------------------------------------------


def band(ax, x, samples, *, color=None, label=None, lo=25, hi=75, center="median", **line_kw):
    """Plot the centre of a set of runs with a shaded band between two percentiles.

    ``samples`` has shape (len(x), n_runs).
    """
    samples = np.asarray(samples, dtype=float)
    mid = np.nanmedian(samples, axis=1) if center == "median" else np.nanmean(samples, axis=1)
    (line,) = ax.plot(x, mid, color=color, label=label, **line_kw)
    ax.fill_between(
        x,
        np.nanpercentile(samples, lo, axis=1),
        np.nanpercentile(samples, hi, axis=1),
        color=line.get_color(),
        alpha=0.15,
        linewidth=0,
    )
    return line


def end_label(ax, x, y, text, *, dx=6, dy=0, **kw):
    """Write a label next to the last point of a line, in ink rather than the series colour."""
    kw.setdefault("fontsize", 9)
    kw.setdefault("color", INK_SECONDARY)
    kw.setdefault("va", "center")
    return ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", **kw)


# --- grid worlds ---------------------------------------------------------------

_STAR = MplPath.unit_regular_star(5, innerCircle=0.42)
_DCOL = np.array([0, 1, 0, -1])
_DROW = np.array([-1, 0, 1, 0])


def plot_gridworld(
    gw: GridWorld,
    policy: np.ndarray | None = None,
    *,
    values: np.ndarray | None = None,
    flagged: np.ndarray | None = None,
    ax=None,
    title: str | None = None,
    legend: bool = True,
    value_label: str = "value",
    vlim: float | None = None,
    arrow_color: str = BLUE,
    flagged_label: str = "differs from the optimal policy",
):
    """Draw a grid world, optionally with a policy and a value function.

    Parameters
    ----------
    policy : (S,) actions to draw as arrows in the free cells.
    values : (S,) values to shade the free cells with (blue positive, red negative).
    flagged : (S,) bool mask of cells whose arrow is drawn in the highlight colour.
    """
    rows, cols = gw.shape
    if ax is None:
        scale = min(0.55, 9.0 / max(rows, cols))
        _, ax = plt.subplots(figsize=(cols * scale + 0.6, rows * scale + 0.9))
    layout = gw.layout
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(AXIS)

    if values is not None:
        shade = np.where(layout == FREE, np.asarray(values, dtype=float).reshape(rows, cols), np.nan)
        lim = np.nanmax(np.abs(shade)) if vlim is None else vlim
        image = ax.imshow(shade, cmap=DIVERGING, vmin=-lim, vmax=lim, alpha=0.9, zorder=0)
        cbar = ax.figure.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label(value_label)
        cbar.outline.set_visible(False)

    # hairlines between cells
    ax.vlines(np.arange(0.5, cols - 0.5), -0.5, rows - 0.5, color=GRIDLINE, linewidth=0.6, zorder=1)
    ax.hlines(np.arange(0.5, rows - 0.5), -0.5, cols - 0.5, color=GRIDLINE, linewidth=0.6, zorder=1)

    r_obs, c_obs = np.nonzero(layout == OBSTACLE)
    ax.add_collection(
        PatchCollection(
            [Rectangle((c - 0.5, r - 0.5), 1, 1) for r, c in zip(r_obs, c_obs)],
            facecolor=GREY,
            edgecolor=SURFACE,
            linewidth=0.6,
            zorder=2,
        )
    )
    r_pit, c_pit = np.nonzero(layout == PIT)
    ax.add_collection(
        PatchCollection(
            [Circle((c, r), 0.33) for r, c in zip(r_pit, c_pit)], facecolor=BAD, edgecolor=SURFACE, linewidth=0.8, zorder=3
        )
    )
    for r, c in zip(*np.nonzero(layout == GOAL)):
        star = PathPatch(
            _STAR.transformed(Affine2D().scale(0.46, -0.46).translate(c, r)),
            facecolor=GOOD,
            edgecolor=SURFACE,
            linewidth=0.8,
            zorder=3,
        )
        ax.add_patch(star)

    handles = [
        Line2D([], [], marker="s", color="none", markerfacecolor=GREY, markeredgecolor="none", markersize=9, label="obstacle"),
        Line2D([], [], marker="o", color="none", markerfacecolor=BAD, markeredgecolor="none", markersize=9, label="pit (−)"),
        Line2D([], [], marker="*", color="none", markerfacecolor=GOOD, markeredgecolor="none", markersize=13, label="goal (+)"),
    ]

    if policy is not None:
        policy = np.asarray(policy).reshape(rows, cols)
        free = layout == FREE
        flag = np.zeros((rows, cols), dtype=bool) if flagged is None else np.asarray(flagged, dtype=bool).reshape(rows, cols)
        length = 0.62
        for mask, colour, z in ((free & ~flag, arrow_color, 4), (free & flag, ORANGE, 5)):
            r, c = np.nonzero(mask)
            if r.size == 0:
                continue
            a = policy[r, c]
            ax.quiver(
                c - 0.5 * length * _DCOL[a],
                r - 0.5 * length * _DROW[a],
                length * _DCOL[a],
                length * _DROW[a],
                angles="xy",
                scale_units="xy",
                scale=1,
                units="xy",
                width=0.085,
                headwidth=3.6,
                headlength=3.4,
                headaxislength=3.0,
                color=colour,
                zorder=z,
            )
        handles.append(Line2D([], [], marker=r"$\rightarrow$", color="none", markerfacecolor=arrow_color, markeredgecolor=arrow_color, markersize=11, label="policy"))
        if flag.any():
            handles.append(Line2D([], [], marker=r"$\rightarrow$", color="none", markerfacecolor=ORANGE, markeredgecolor=ORANGE, markersize=11, label=flagged_label))
            rr, cc = np.nonzero(free & flag)
            ax.add_collection(
                PatchCollection(
                    [Rectangle((c - 0.5, r - 0.5), 1, 1) for r, c in zip(rr, cc)],
                    facecolor="none",
                    edgecolor=ORANGE,
                    linewidth=1.6,
                    zorder=6,
                )
            )

    if title:
        ax.set_title(title)
    if legend:
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=min(len(handles), 5 if max(rows, cols) > 6 else 3),
            handletextpad=0.3,
            columnspacing=1.2,
        )
    return ax
