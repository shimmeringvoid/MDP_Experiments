"""The crazy elevator from the 2014 report.

A building has ``N`` floors, numbered from 0.  On floor ``k``, pressing button
``a`` takes you to floor ``j`` with probability

    T(k, a, j) = Z * | sin(k + sin(k * a + sin(j ** 2))) |

where ``Z`` normalises the distribution over ``j``.  Arriving on floor ``j``
pays ``-j``, except for the ground floor, which pays ``20 * N`` and ends the
ride.  There is one state per floor and one action per button, so the action
space grows with the state space.

That is the formula as the report prints it.  The numbers the report
publishes, however, can only be reproduced with ``sin(j XOR 2)`` in place of
``sin(j ** 2)``: in Python ``j ^ 2`` is bitwise exclusive-or, not a square, and
the original code evidently used it.  With that reading, the 50-floor optimal
policy printed in the report is recovered exactly.  Both versions are
available here; ``formula="as_run"`` (the default) is the elevator that was
actually solved in 2014, ``formula="as_written"`` the one the text describes.
"""

from __future__ import annotations

import numpy as np

from .core import MDP

__all__ = ["crazy_elevator", "elevator_transitions", "REPORT_POLICY_50"]

# The optimal policy for 50 floors printed in the report (best button per floor).
REPORT_POLICY_50 = np.array(
    [0, 24, 45, 20, 6, 13, 33, 34, 5, 15, 32, 47, 6, 43, 8, 1, 22,
     4, 42, 2, 10, 36, 48, 30, 37, 21, 6, 18, 34, 42, 34, 23, 10, 37,
     40, 28, 8, 41, 42, 33, 20, 2, 10, 30, 23, 36, 1, 22, 31, 43]
)  # fmt: skip


def elevator_transitions(
    n_floors: int,
    buttons: np.ndarray | None = None,
    formula: str = "as_run",
) -> np.ndarray:
    """Transition tensor ``T[k, a, j]`` of the crazy elevator, shape (N, A, N).

    ``formula`` selects the innermost term: ``"as_run"`` uses ``sin(j XOR 2)``
    (what the 2014 code computed), ``"as_written"`` uses ``sin(j ** 2)`` (what
    the report says).
    """
    if buttons is None:
        buttons = np.arange(n_floors)
    floors = np.arange(n_floors)
    if formula == "as_run":
        inner = np.sin((floors ^ 2).astype(float))
    elif formula == "as_written":
        inner = np.sin(floors.astype(float) ** 2)
    else:
        raise ValueError("formula must be 'as_run' or 'as_written'")
    k = floors.astype(float)[:, None, None]
    a = np.asarray(buttons, dtype=float)[None, :, None]
    weight = np.abs(np.sin(k + np.sin(k * a + inner[None, None, :])))
    return weight / weight.sum(axis=2, keepdims=True)


def crazy_elevator(
    n_floors: int,
    *,
    gamma: float = 0.99,
    goal_reward: float | None = None,
    buttons: np.ndarray | int | None = None,
    trap_floor: int | None = None,
    trap_reward: float | None = None,
    formula: str = "as_run",
) -> MDP:
    """Build the crazy elevator as a dense :class:`~mdplab.core.MDP`.

    Parameters
    ----------
    n_floors : int
        Number of floors ``N`` (and of states).
    goal_reward : float, optional
        Reward for arriving at the ground floor.  Defaults to ``20 * N`` as in
        the report.
    buttons : int or array, optional
        Restrict the panel to a subset of buttons.  An integer ``m`` keeps
        buttons ``0 .. m-1``.  Used to vary the number of actions while
        holding the number of states fixed.
    trap_floor, trap_reward : optional
        Turn one floor into a second absorbing state with its own arrival
        reward.  The report muses that "a negative absorbing state for floor
        13 would have been interesting".
    formula : ``"as_run"`` or ``"as_written"``
        See the module docstring.
    """
    if n_floors < 2:
        raise ValueError("the building needs at least two floors")
    if isinstance(buttons, (int, np.integer)):
        buttons = np.arange(int(buttons))
    T = elevator_transitions(n_floors, buttons, formula)
    n_actions = T.shape[1]
    r_next = -np.arange(n_floors, dtype=float)
    r_next[0] = 20.0 * n_floors if goal_reward is None else goal_reward
    terminal = np.zeros(n_floors, dtype=bool)
    terminal[0] = True
    if trap_floor is not None:
        terminal[trap_floor] = True
        if trap_reward is not None:
            r_next[trap_floor] = trap_reward
    return MDP(
        P=T.reshape(n_floors * n_actions, n_floors),
        r_next=r_next,
        gamma=gamma,
        terminal=terminal,
        name=f"crazy elevator, {n_floors} floors",
        meta={
            "buttons": np.arange(n_floors) if buttons is None else np.asarray(buttons),
            "formula": formula,
        },
    )
