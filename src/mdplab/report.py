"""What the 2014 report published: its tables, and data read back off its figures.

The original code and raw results no longer exist.  The tables below are
transcribed from the report; the figure data were recovered from the images
embedded in the PDF by ``tools/digitize_report.py`` and are accurate to about
a pixel of the original bitmaps.
"""

from __future__ import annotations

from importlib import resources

import numpy as np
import pandas as pd

from .gridworld import GridWorld

__all__ = [
    "figure2",
    "figure4",
    "figure7",
    "figure8",
    "figure3_gridworld",
    "figure3_policy",
    "TABLE_GRIDWORLD",
    "TABLE_ELEVATOR_50",
    "TABLE_ELEVATOR",
]


def _data(name: str):
    return resources.files("mdplab") / "data" / name


def _csv(name: str) -> pd.DataFrame:
    with resources.as_file(_data(name)) as path:
        return pd.read_csv(path)


def figure2() -> pd.DataFrame:
    """Figure 2: value vs policy iteration on random grid worlds, by number of states.

    Value-iteration numbers are averages over 20 random worlds of each size.
    """
    return _csv("report_fig2_gridworld.csv")


def figure4() -> pd.DataFrame:
    """Figure 4: "mean reward" of the planned and the Q-learned policies on small grid worlds."""
    return _csv("report_fig4_gridworld.csv")


def figure7() -> pd.DataFrame:
    """Figure 7: value vs policy iteration on the crazy elevator, by number of floors."""
    return _csv("report_fig7_elevator.csv")


def figure8() -> pd.DataFrame:
    """Figure 8: simulated reward and utility of the planned and Q-learned elevator policies.

    The planned-policy markers are missing wherever the figure draws a
    Q-learning marker on top of them.
    """
    return _csv("report_fig8_elevator.csv")


def figure3_gridworld(**kwargs) -> GridWorld:
    """The random 39 x 39 world of Figure 3, decoded from the figure."""
    kwargs.setdefault("name", "report figure 3 (39x39)")
    return GridWorld.from_text(_data("report_fig3_layout.txt").read_text(encoding="utf-8"), **kwargs)


def figure3_policy() -> np.ndarray:
    """The policy drawn in Figure 3, decoded from the figure (one action per cell)."""
    gw = figure3_gridworld()
    return gw.policy_from_text(_data("report_fig3_policy.txt").read_text(encoding="utf-8"))


# Table on page 2: the 7 x 8 world of Figure 1.
TABLE_GRIDWORLD = pd.DataFrame(
    {
        "iterations": [152, 6, 1_337_860],
        "seconds": [1.78, 11.42, 115.03],
        "expected_reward": [199.9, 199.9, 193.9],
    },
    index=["Value iteration", "Policy iteration", "Q-learning"],
)

# Table on page 7: the 50-floor elevator, simulated with 100 restarts per starting floor.
TABLE_ELEVATOR_50 = pd.DataFrame(
    {
        "reward": [324.04, -214.29, -177.58],
        "utility": [285.45, -141.37, -119.71],
        "floors_visited": [27.6, 49.5, 48.1],
    },
    index=["Optimal", "Always press 0", "Random"],
)

# Table on page 8.  Numbers are printed as in the report, including the row
# for "Always press 0" at 50 floors, whose reward and utility appear swapped.
TABLE_ELEVATOR = pd.DataFrame(
    [
        (10, "Value & policy iteration", 180.8, 173.6, 5.2),
        (10, "Q-learning", 179.1, 171.5, 5.4),
        (10, "Always press 0", 155.3, 142.7, 9.9),
        (10, "Random", 151.7, 138.9, 10.5),
        (30, "Value & policy iteration", 372.9, 322.7, 16.3),
        (30, "Q-learning", 370.4, 321.5, 16.5),
        (30, "Always press 0", 188.0, 146.7, 28.4),
        (30, "Random", 170.7, 133.2, 29.5),
        (50, "Value & policy iteration", 322.86, 255.1, 27.6),
        (50, "Q-learning", 309.9, 242.1, 28.2),
        (50, "Always press 0", -133.8, -191.8, 48.7),
        (50, "Random", -171.8, -115.3, 47.9),
    ],
    columns=["floors", "policy", "reward", "utility", "floors_visited"],
)
