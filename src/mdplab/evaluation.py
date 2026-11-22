"""Measuring how good a policy is.

Because the model is known, a policy's value can be computed exactly by
solving a linear system.  The 2014 report estimated the same quantities by
simulation (100 restarts from every starting state); :func:`monte_carlo`
reproduces that protocol, and the exact routines say what the simulation was
converging to.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .core import MDP

__all__ = [
    "evaluate_policy",
    "iterative_policy_evaluation",
    "absorption",
    "Absorption",
    "hitting_probability",
    "monte_carlo",
    "MonteCarlo",
    "optimal_actions",
    "agreement",
    "value_gap",
    "uniform_random_policy",
]


def _solve(A, b):
    if sp.issparse(A):
        return spla.spsolve(sp.csc_matrix(A), b)
    return np.linalg.solve(A, b)


def _restrict(P_pi, idx):
    """Square block of ``P_pi`` on the states ``idx``."""
    if sp.issparse(P_pi):
        return sp.csr_matrix(P_pi)[idx][:, idx]
    return P_pi[np.ix_(idx, idx)]


def _identity_minus(M, scale: float = 1.0):
    n = M.shape[0]
    if sp.issparse(M):
        return sp.identity(n, format="csc") - scale * sp.csc_matrix(M)
    return np.eye(n) - scale * M


def evaluate_policy(mdp: MDP, policy: np.ndarray, gamma: float | None = None) -> np.ndarray:
    """Exact discounted value ``V^pi`` by solving ``(I - gamma P_pi) V = r_pi``.

    Terminal states have value zero, so the system is solved on the
    non-terminal states only.
    """
    g = mdp.gamma if gamma is None else gamma
    P_pi, r_pi = mdp.policy_model(policy)
    idx = np.flatnonzero(mdp.active)
    V = np.zeros(mdp.n_states)
    if idx.size:
        V[idx] = _solve(_identity_minus(_restrict(P_pi, idx), g), r_pi[idx])
    return V


def iterative_policy_evaluation(
    mdp: MDP,
    policy: np.ndarray,
    tol: float = 1e-6,
    v0: np.ndarray | None = None,
    max_sweeps: int = 1_000_000,
) -> tuple[np.ndarray, int]:
    """Evaluate a policy by repeated sweeps ``V <- r_pi + gamma P_pi V``.

    Stops when the largest change in a sweep falls below ``tol``.  Returns the
    value estimate and the number of sweeps used.
    """
    P_pi, r_pi = mdp.policy_model(policy)
    V = np.zeros(mdp.n_states) if v0 is None else np.array(v0, dtype=float)
    for sweep in range(1, max_sweeps + 1):
        V_new = r_pi + mdp.gamma * (P_pi @ V)
        delta = np.abs(V_new - V).max()
        V = V_new
        if delta < tol:
            break
    return V, sweep


# --------------------------------------------------------------------------
# Undiscounted quantities: does the policy ever finish, and how long does it take?
# --------------------------------------------------------------------------


@dataclass
class Absorption:
    """What happens in the long run when a policy is followed without discounting.

    Attributes
    ----------
    probability : (S,) probability of eventually reaching a terminal state.
    steps : (S,) expected number of moves until then (``inf`` if it may never happen).
    total_reward : (S,) expected undiscounted sum of rewards (``nan`` if it may never end).
    """

    probability: np.ndarray
    steps: np.ndarray
    total_reward: np.ndarray

    @property
    def proper(self) -> np.ndarray:
        """States from which the policy is certain to finish."""
        return self.probability > 1.0 - 1e-9


def _can_reach(P_pi, targets: np.ndarray) -> np.ndarray:
    """States with a positive-probability path into ``targets`` under ``P_pi``."""
    adj = sp.csr_matrix(P_pi) if not sp.issparse(P_pi) else P_pi.tocsr()
    adj = (adj > 0).astype(np.float64)
    reach = targets.copy()
    while True:
        new = reach | ((adj @ reach.astype(np.float64)) > 0)
        if (new == reach).all():
            return reach
        reach = new


def hitting_probability(mdp: MDP, policy: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Probability of ending up in the terminal states ``target`` (a bool mask)."""
    target = np.asarray(target, dtype=bool) & mdp.terminal
    P_pi, _ = mdp.policy_model(policy)
    can = _can_reach(P_pi, mdp.terminal.copy()) & mdp.active
    h = target.astype(float)
    idx = np.flatnonzero(can)
    if idx.size:
        P_csr = sp.csr_matrix(P_pi) if not sp.issparse(P_pi) else P_pi.tocsr()
        into_target = np.asarray(P_csr[idx] @ target.astype(float)).ravel()
        h[idx] = _solve(_identity_minus(_restrict(P_pi, idx)), into_target)
    return np.clip(h, 0.0, 1.0)


def absorption(mdp: MDP, policy: np.ndarray) -> Absorption:
    """Exact undiscounted analysis of a policy (see :class:`Absorption`)."""
    S = mdp.n_states
    P_pi, r_pi = mdp.policy_model(policy)
    prob = hitting_probability(mdp, policy, mdp.terminal)
    steps = np.full(S, np.inf)
    total = np.full(S, np.nan)
    steps[mdp.terminal] = 0.0
    total[mdp.terminal] = 0.0
    idx = np.flatnonzero(mdp.active & (prob > 1.0 - 1e-9))
    if idx.size:
        A = _identity_minus(_restrict(P_pi, idx))
        steps[idx] = _solve(A, np.ones(idx.size))
        total[idx] = _solve(A, r_pi[idx])
    return Absorption(probability=prob, steps=steps, total_reward=total)


# --------------------------------------------------------------------------
# Simulation, as in the report
# --------------------------------------------------------------------------


@dataclass
class MonteCarlo:
    """Per-episode outcomes of simulated rollouts, shape (n_starts, n_restarts)."""

    starts: np.ndarray
    total_reward: np.ndarray
    discounted: np.ndarray
    steps: np.ndarray
    finished: np.ndarray

    def summary(self) -> dict[str, float]:
        return {
            "reward": float(self.total_reward.mean()),
            "utility": float(self.discounted.mean()),
            "steps": float(self.steps.mean()),
            "finished": float(self.finished.mean()),
        }


def monte_carlo(
    mdp: MDP,
    policy: np.ndarray,
    n_restarts: int = 100,
    seed: int | np.random.Generator | None = None,
    starts: np.ndarray | None = None,
    max_steps: int | None = None,
) -> MonteCarlo:
    """Follow a policy from every starting state ``n_restarts`` times.

    ``policy`` may be deterministic (shape (S,)) or stochastic (shape (S, A)).
    Episodes that have not ended after ``max_steps`` moves are cut off; by
    default that happens once the discount has shrunk below 1e-10, which is
    the report's "stop once the utility has converged".
    """
    rng = np.random.default_rng(seed)
    policy = np.asarray(policy)
    if starts is None:
        starts = np.flatnonzero(mdp.active)
    if max_steps is None:
        max_steps = int(np.ceil(np.log(1e-10) / np.log(mdp.gamma))) if mdp.gamma < 1 else 100_000
    sampler = mdp.sampler
    n = starts.size * n_restarts
    s = np.repeat(starts, n_restarts)
    total = np.zeros(n)
    disc = np.zeros(n)
    steps = np.zeros(n, dtype=np.int64)
    alive = np.arange(n)
    discount = np.ones(n)
    if policy.ndim == 2:
        cum_pi = np.cumsum(policy, axis=1)
        cum_pi /= cum_pi[:, -1:]
    for _ in range(max_steps):
        if alive.size == 0:
            break
        cur = s[alive]
        if policy.ndim == 1:
            a = policy[cur]
        else:
            a = (rng.random(cur.size)[:, None] >= cum_pi[cur]).sum(axis=1)
            a = np.minimum(a, mdp.n_actions - 1)
        s2, r = sampler.step(cur, a, rng)
        total[alive] += r
        disc[alive] += discount[alive] * r
        discount[alive] *= mdp.gamma
        steps[alive] += 1
        s[alive] = s2
        alive = alive[~mdp.terminal[s2]]
    finished = np.ones(n, dtype=bool)
    finished[alive] = False
    shape = (starts.size, n_restarts)
    return MonteCarlo(
        starts=starts,
        total_reward=total.reshape(shape),
        discounted=disc.reshape(shape),
        steps=steps.reshape(shape),
        finished=finished.reshape(shape),
    )


# --------------------------------------------------------------------------
# Comparing a policy with the optimum
# --------------------------------------------------------------------------


def optimal_actions(mdp: MDP, V_star: np.ndarray, atol: float = 1e-6) -> np.ndarray:
    """Mask of shape (S, A): actions whose value is within ``atol`` of the best."""
    Q = mdp.q_values(V_star)
    return Q >= Q.max(axis=1, keepdims=True) - atol


def agreement(mdp: MDP, policy: np.ndarray, optimal_mask: np.ndarray) -> float:
    """Fraction of decision states in which ``policy`` picks an optimal action."""
    ok = optimal_mask[np.arange(mdp.n_states), policy]
    return float(ok[mdp.active].mean())


def value_gap(mdp: MDP, policy: np.ndarray, V_star: np.ndarray, weights: np.ndarray | None = None):
    """Average shortfall ``V*(s) - V^pi(s)`` over starting states (uniform by default)."""
    V = evaluate_policy(mdp, policy)
    if weights is None:
        return float((V_star - V)[mdp.active].mean())
    return float(((V_star - V) * weights).sum() / weights.sum())


def uniform_random_policy(mdp: MDP) -> np.ndarray:
    """The stochastic policy that picks every action with equal probability."""
    return np.full((mdp.n_states, mdp.n_actions), 1.0 / mdp.n_actions)
