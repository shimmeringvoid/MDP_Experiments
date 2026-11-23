"""Scoring a way of choosing policies on a whole collection of grid worlds.

A learner that is meant to cope with worlds it has not seen cannot be judged on
one map.  :class:`WorldSet` holds many worlds together with their exact
solutions, and scores any rule that turns a world into a policy: a network, a
table trained on the spot, or a planner working from the wrong discount.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .evaluation import evaluate_policy, hitting_probability, optimal_actions, uniform_random_policy
from .gridworld import GOAL, PIT, GridWorld
from .planning import policy_iteration

__all__ = ["WorldSet"]


@dataclass
class WorldSet:
    """Grid worlds with their exact solutions.

    Attributes
    ----------
    worlds : the worlds.
    optimal : one row per world: the number of free cells, the value that
        acting at random gives up against the optimal policy
        (``lost_by_random``), and the optimal policy's chances of ending at
        the goal (``p_goal``) and in a pit (``p_pit``).  Like every figure in
        this class, these are averages over the free cells as starting points.
    """

    worlds: list[GridWorld]
    optimal: pd.DataFrame = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._solved, rows = [], []
        for world in self.worlds:
            mdp, free = world.mdp, world.free
            plan = policy_iteration(mdp)
            lost_by_random = (plan.V - evaluate_policy(mdp, uniform_random_policy(mdp)))[free].mean()
            self._solved.append((plan.V, lost_by_random, optimal_actions(mdp, plan.V)))
            rows.append(
                {
                    "free_cells": int(free.sum()),
                    "lost_by_random": lost_by_random,
                    "p_goal": hitting_probability(mdp, plan.policy, world.mask(GOAL))[free].mean(),
                    "p_pit": hitting_probability(mdp, plan.policy, world.mask(PIT))[free].mean(),
                }
            )
        self.optimal = pd.DataFrame(rows)

    def __len__(self) -> int:
        return len(self.worlds)

    @classmethod
    def random(cls, rows: int, cols: int, n: int, seed: int, **kwargs) -> "WorldSet":
        """``n`` random worlds (:meth:`GridWorld.random`) in which the choice of policy matters.

        Worlds without a free cell, and worlds in which acting at random is as
        good as acting optimally, are skipped, since nothing can be learned or
        measured there.
        """
        rng = np.random.default_rng(seed)
        worlds = []
        while len(worlds) < n:
            world = GridWorld.random(rows, cols, seed=rng, **kwargs)
            if not world.free.any():
                continue
            mdp = world.mdp
            lost = policy_iteration(mdp).V - evaluate_policy(mdp, uniform_random_policy(mdp))
            if lost[world.free].mean() > 1e-6:
                worlds.append(world)
        return cls(worlds)

    def evaluate(self, policy_for: Callable[[GridWorld], np.ndarray]) -> pd.DataFrame:
        """Score ``policy_for(world)`` exactly on every world.

        Returns one row per world with

        * ``value_lost``: optimal value minus the policy's value, averaged over the free cells;
        * ``score``: 1 minus ``value_lost`` as a fraction of what acting at
          random loses, so 0 is as bad as random and 1 is optimal;
        * ``wrong_cells``: share of free cells in which the action is not an optimal one;
        * ``p_goal``, ``p_pit``: chances of ending at the goal and in a pit.
        """
        rows = []
        for world, (V_star, lost_by_random, best) in zip(self.worlds, self._solved, strict=True):
            mdp, free = world.mdp, world.free
            policy = np.asarray(policy_for(world))
            lost = (V_star - evaluate_policy(mdp, policy))[free].mean()
            cells = np.flatnonzero(free)
            rows.append(
                {
                    "value_lost": lost,
                    "score": 1.0 - lost / lost_by_random,
                    "wrong_cells": 1.0 - best[cells, policy[cells]].mean(),
                    "p_goal": hitting_probability(mdp, policy, world.mask(GOAL))[free].mean(),
                    "p_pit": hitting_probability(mdp, policy, world.mask(PIT))[free].mean(),
                }
            )
        return pd.DataFrame(rows)
