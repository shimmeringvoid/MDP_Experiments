"""Value and policy iteration the way a 2014 course project would write them.

This is *not* the original code, which is lost.  These are the textbook
algorithms in the state-utility form of Russell & Norvig that the course
taught,

    U(s) <- R(s) + gamma * max_a sum_s' T[s, a, s'] U(s'),

started from ``U = 0`` and stopped when no utility moves by more than a
tolerance.  Policy evaluation is done by sweeping, not by solving a linear
system.

They are kept for two reasons.  First, they reproduce the report's iteration
counts exactly, where the arrival-reward formulation used in the rest of the
package is one sweep ahead.  Second, with ``loops=True`` they run as explicit
Python loops over states and actions around a NumPy dot product, which is
what the report's timings point to, so they show how much of those timings
belonged to the coding style rather than to the algorithms.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .core import MDP

__all__ = ["LegacyResult", "value_iteration_2014", "policy_iteration_2014"]


@dataclass
class LegacyResult:
    U: np.ndarray
    policy: np.ndarray
    iterations: int
    seconds: float
    eval_sweeps: int = 0


def _dense(mdp: MDP):
    S, A = mdp.n_states, mdp.n_actions
    P = mdp.P.toarray() if mdp.is_sparse else mdp.P
    return P.reshape(S, A, S), mdp.r_next.copy()


def value_iteration_2014(mdp: MDP, tol: float = 1e-6, max_iter: int = 1_000_000, loops: bool = False) -> LegacyResult:
    """Textbook value iteration on state utilities, from ``U = 0``."""
    T, R = _dense(mdp)
    S, A = mdp.n_states, mdp.n_actions
    terminal = mdp.terminal
    U = np.zeros(S)
    start = time.perf_counter()
    for it in range(1, max_iter + 1):
        if loops:
            U_new = np.empty(S)
            for s in range(S):
                if terminal[s]:
                    U_new[s] = R[s]
                    continue
                best = -np.inf
                for a in range(A):
                    q = np.dot(T[s, a], U)
                    if q > best:
                        best = q
                U_new[s] = R[s] + mdp.gamma * best
        else:
            U_new = R + mdp.gamma * (T @ U).max(axis=1)
            U_new[terminal] = R[terminal]
        delta = np.abs(U_new - U).max()
        U = U_new
        if delta < tol:
            break
    seconds = time.perf_counter() - start
    return LegacyResult(U=U, policy=(T @ U).argmax(axis=1), iterations=it, seconds=seconds)


def policy_iteration_2014(
    mdp: MDP,
    tol: float = 1e-6,
    max_iter: int = 10_000,
    loops: bool = False,
    policy0: np.ndarray | None = None,
) -> LegacyResult:
    """Textbook policy iteration: evaluate by sweeping to ``tol``, then act greedily.

    Each evaluation starts again from ``U = 0``.  ``iterations`` counts
    evaluate-improve rounds, including the last one that changes nothing.
    """
    T, R = _dense(mdp)
    S, A = mdp.n_states, mdp.n_actions
    terminal = mdp.terminal
    states = np.arange(S)
    policy = np.zeros(S, dtype=np.int64) if policy0 is None else np.array(policy0, dtype=np.int64)
    sweeps = 0
    start = time.perf_counter()
    for it in range(1, max_iter + 1):
        U = np.zeros(S)
        while True:
            sweeps += 1
            if loops:
                U_new = np.empty(S)
                for s in range(S):
                    U_new[s] = R[s] if terminal[s] else R[s] + mdp.gamma * np.dot(T[s, policy[s]], U)
            else:
                U_new = R + mdp.gamma * (T[states, policy] @ U)
                U_new[terminal] = R[terminal]
            delta = np.abs(U_new - U).max()
            U = U_new
            if delta < tol:
                break
        if loops:
            Q = np.empty((S, A))
            for s in range(S):
                for a in range(A):
                    Q[s, a] = np.dot(T[s, a], U)
        else:
            Q = T @ U
        # keep the current action unless another is better beyond the evaluation error
        margin = 2 * tol * mdp.gamma / (1 - mdp.gamma)
        better = (Q.max(axis=1) > Q[states, policy] + margin) & ~terminal
        if not better.any():
            break
        policy = np.where(better, Q.argmax(axis=1), policy)
    seconds = time.perf_counter() - start
    return LegacyResult(U=U, policy=policy, iterations=it, seconds=seconds, eval_sweeps=sweeps)
