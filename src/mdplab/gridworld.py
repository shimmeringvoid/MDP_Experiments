"""The stochastic grid world from the 2014 report.

A move goes where it was aimed with probability 0.8 and slips to either side
with probability 0.1 each.  Bumping into the edge of the grid or into an
obstacle leaves the agent where it was.  Cells are free, obstacles
("exclusionary states" in the report), pits ("negative absorbing states") or
the goal.  Rewards are paid on arrival: a small cost for every ordinary move,
a large penalty for a pit and a large prize for the goal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import scipy.sparse as sp

from .core import MDP

__all__ = [
    "GridWorld",
    "FREE",
    "OBSTACLE",
    "PIT",
    "GOAL",
    "UP",
    "RIGHT",
    "DOWN",
    "LEFT",
    "ARROWS",
    "REPORT_FIG1_LAYOUT",
    "REPORT_FIG1A_POLICY",
    "REPORT_FIG1B_POLICY",
]

FREE, OBSTACLE, PIT, GOAL = 0, 1, 2, 3
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
ARROWS = "↑→↓←"

_DROW = np.array([-1, 0, 1, 0])
_DCOL = np.array([0, 1, 0, -1])
_CELL_CHARS = ".#xG"
_ASCII_ARROWS = "^>v<"

# The 7 x 8 world of the report's Figure 1, read off the published figure.
REPORT_FIG1_LAYOUT = """
....#..x
........
..#...#.
.#...#G.
.#..x#..
.x..#...
#......x
"""

# Figure 1a: the policy found by value iteration and policy iteration in 2014.
REPORT_FIG1A_POLICY = """
→→→↓#↓←x
→→→→→→→↓
↑↑#→↑↑#↓
↑#↓↑↑#G←
↑#↓←x#↑↑
←x→↓#→↑↑
#↓→→→↑←x
"""

# Figure 1b: the policy found by the 2014 Q-learner.
REPORT_FIG1B_POLICY = """
→→→↓#↓↓x
→→→→→→→↓
↑↑#→↑↑#↓
↑#↓↑↑#G←
↑#↓↓x#↑←
←x→↓#→↑↑
#→→→→↑←x
"""


@dataclass(frozen=True, eq=False)
class GridWorld:
    """A rectangular grid world.

    Parameters
    ----------
    layout : int array, shape (rows, cols)
        Cell codes: ``FREE``, ``OBSTACLE``, ``PIT`` or ``GOAL``.  Row 0 is the
        top row, as in the report's figures.
    step_reward, goal_reward, pit_reward : float
        Rewards paid on arriving in a free cell, the goal, or a pit.
    gamma : float
        Discount factor.
    p_intended : float
        Probability that a move goes where it was aimed.  The remainder is
        split evenly between the two perpendicular directions.
    """

    layout: np.ndarray
    step_reward: float = -0.01
    goal_reward: float = 200.0
    pit_reward: float = -200.0
    gamma: float = 0.99
    p_intended: float = 0.8
    name: str = field(default="gridworld", compare=False)

    def __post_init__(self) -> None:
        layout = np.array(self.layout, dtype=np.int8)
        if layout.ndim != 2:
            raise ValueError("layout must be two-dimensional")
        layout.setflags(write=False)
        object.__setattr__(self, "layout", layout)

    # ------------------------------------------------------------ construction

    @classmethod
    def random(
        cls,
        rows: int,
        cols: int,
        seed: int | np.random.Generator | None = None,
        p_pit: float = 0.1,
        p_obstacle: float = 0.1,
        **kwargs,
    ) -> "GridWorld":
        """Random world, generated the way the report describes.

        Each cell becomes a pit with probability ``p_pit``.  Then each cell
        becomes an obstacle with probability ``p_obstacle``, overwriting any
        pit already there, which is why worlds end up with slightly fewer pits
        than obstacles.  Finally one cell is chosen as the goal.  Nothing
        guarantees that the goal can be reached from everywhere.
        """
        rng = np.random.default_rng(seed)
        layout = np.full((rows, cols), FREE, dtype=np.int8)
        layout[rng.random((rows, cols)) < p_pit] = PIT
        layout[rng.random((rows, cols)) < p_obstacle] = OBSTACLE
        layout[rng.integers(rows), rng.integers(cols)] = GOAL
        return cls(layout, name=f"gridworld {rows}x{cols}", **kwargs)

    @classmethod
    def from_text(cls, text: str, **kwargs) -> "GridWorld":
        """Parse a layout written with ``.`` free, ``#`` obstacle, ``x`` pit, ``G`` goal."""
        lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        if len({len(line) for line in lines}) != 1:
            raise ValueError("all rows of a layout must have the same length")
        layout = np.array([[_CELL_CHARS.index(ch) for ch in line] for line in lines], dtype=np.int8)
        kwargs.setdefault("name", f"gridworld {layout.shape[0]}x{layout.shape[1]}")
        return cls(layout, **kwargs)

    @classmethod
    def report_figure1(cls, **kwargs) -> "GridWorld":
        """The 7 x 8 world shown in Figure 1 of the report."""
        kwargs.setdefault("name", "report figure 1 (7x8)")
        return cls.from_text(REPORT_FIG1_LAYOUT, **kwargs)

    def replace(self, **changes) -> "GridWorld":
        """Copy of this world with some parameters changed."""
        params = dict(
            layout=self.layout,
            step_reward=self.step_reward,
            goal_reward=self.goal_reward,
            pit_reward=self.pit_reward,
            gamma=self.gamma,
            p_intended=self.p_intended,
            name=self.name,
        )
        params.update(changes)
        return GridWorld(**params)

    # ----------------------------------------------------------------- shape

    @property
    def shape(self) -> tuple[int, int]:
        return self.layout.shape

    @property
    def n_states(self) -> int:
        return self.layout.size

    @property
    def n_actions(self) -> int:
        return 4

    def state(self, row: int, col: int) -> int:
        return row * self.shape[1] + col

    def cell(self, state: int) -> tuple[int, int]:
        return divmod(int(state), self.shape[1])

    @property
    def free(self) -> np.ndarray:
        """Flat mask of cells the agent can occupy and act in."""
        return (self.layout == FREE).ravel()

    @property
    def goal_state(self) -> int:
        goals = np.flatnonzero(self.layout.ravel() == GOAL)
        if goals.size != 1:
            raise ValueError("layout must contain exactly one goal")
        return int(goals[0])

    def mask(self, code: int) -> np.ndarray:
        return (self.layout == code).ravel()

    # ------------------------------------------------------------------ model

    def destinations(self) -> np.ndarray:
        """Where each of the four directions leads from each cell.

        Returns an integer array of shape (4, S).  A blocked move leads back
        to the cell it started from.
        """
        rows, cols = self.shape
        S = rows * cols
        here = np.arange(S)
        r, c = np.divmod(here, cols)
        blocked = (self.layout == OBSTACLE).ravel()
        dest = np.empty((4, S), dtype=np.int64)
        for d in range(4):
            r2, c2 = r + _DROW[d], c + _DCOL[d]
            inside = (r2 >= 0) & (r2 < rows) & (c2 >= 0) & (c2 < cols)
            target = np.where(inside, r2 * cols + c2, here)
            dest[d] = np.where(inside & ~blocked[target], target, here)
        return dest

    @cached_property
    def mdp(self) -> MDP:
        """This world as a sparse :class:`~mdplab.core.MDP`."""
        S, A = self.n_states, 4
        dest = self.destinations()
        slip = (1.0 - self.p_intended) / 2.0
        row_idx, col_idx, vals = [], [], []
        states = np.arange(S)
        for a in range(A):
            for direction, p in ((a, self.p_intended), ((a - 1) % 4, slip), ((a + 1) % 4, slip)):
                row_idx.append(states * A + a)
                col_idx.append(dest[direction])
                vals.append(np.full(S, p))
        P = sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(row_idx), np.concatenate(col_idx))),
            shape=(S * A, S),
        )
        P.sum_duplicates()
        flat = self.layout.ravel()
        r_next = np.zeros(S)
        r_next[flat == FREE] = self.step_reward
        r_next[flat == GOAL] = self.goal_reward
        r_next[flat == PIT] = self.pit_reward
        return MDP(
            P=P,
            r_next=r_next,
            gamma=self.gamma,
            terminal=flat != FREE,
            name=self.name,
            meta={"gridworld": self},
        )

    # --------------------------------------------------------------- analysis

    def connected_to_goal(self) -> np.ndarray:
        """Flat mask of free cells from which some path of free cells leads to the goal.

        Cells outside this set are walled in by obstacles and pits: no policy
        can reach the goal from them.
        """
        dest = self.destinations()
        free = self.free
        reached = np.zeros(self.n_states, dtype=bool)
        reached[self.goal_state] = True
        while True:
            # a free cell joins if one of its moves lands on a reached cell
            new = reached | (free & reached[dest].any(axis=0))
            if (new == reached).all():
                break
            reached = new
        return reached & free

    # ----------------------------------------------------------------- text

    def to_text(self) -> str:
        return "\n".join("".join(_CELL_CHARS[c] for c in row) for row in self.layout)

    def policy_to_text(self, policy: np.ndarray, arrows: str = ARROWS) -> str:
        """Render a policy as a grid of arrows, with the layout symbols elsewhere."""
        grid = np.asarray(policy).reshape(self.shape)
        lines = []
        for r in range(self.shape[0]):
            line = ""
            for c in range(self.shape[1]):
                code = self.layout[r, c]
                line += arrows[grid[r, c]] if code == FREE else _CELL_CHARS[code]
            lines.append(line)
        return "\n".join(lines)

    def policy_from_text(self, text: str) -> np.ndarray:
        """Inverse of :meth:`policy_to_text`.  Non-free cells get action 0."""
        lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        policy = np.zeros(self.shape, dtype=np.int64)
        for r, line in enumerate(lines):
            for c, ch in enumerate(line):
                if ch in ARROWS:
                    policy[r, c] = ARROWS.index(ch)
                elif ch in _ASCII_ARROWS:
                    policy[r, c] = _ASCII_ARROWS.index(ch)
        return policy.ravel()

    def __repr__(self) -> str:
        rows, cols = self.shape
        flat = self.layout.ravel()
        return (
            f"GridWorld({rows}x{cols}, pits={int((flat == PIT).sum())}, "
            f"obstacles={int((flat == OBSTACLE).sum())}, goal={self.cell(self.goal_state)})"
        )
