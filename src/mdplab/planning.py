"""Planning with a known model.

All solvers return a :class:`Solution`.  Besides wall-clock time, each one
counts its work in *backups*: one backup is one evaluation of

    r(s, a) + gamma * sum_s' p(s' | s, a) V(s')

for a single state-action pair.  A sweep of value iteration costs ``S * A``
backups, a sweep of policy evaluation only ``S``.  Counting this way lets
algorithms be compared independently of how the code happens to be written,
which matters when revisiting timings from 2014.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from .core import MDP
from .evaluation import evaluate_policy, iterative_policy_evaluation

__all__ = [
    "Solution",
    "value_iteration",
    "policy_iteration",
    "modified_policy_iteration",
    "gauss_seidel_value_iteration",
    "prioritized_sweeping",
    "linear_programming",
]


@dataclass
class Solution:
    """Result of a planner.

    Attributes
    ----------
    V, Q, policy : value function, action values and greedy policy.
    iterations : number of outer iterations (sweeps for value iteration,
        policy evaluations for policy iteration).
    backups : state-action backups performed (see module docstring).
    solves : linear systems solved (exact policy iteration only).
    seconds : wall-clock time.
    history : per-iteration diagnostics, depending on the solver.
    """

    V: np.ndarray
    Q: np.ndarray
    policy: np.ndarray
    iterations: int
    converged: bool
    method: str
    backups: int = 0
    solves: int = 0
    seconds: float = 0.0
    history: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"Solution({self.method}, iterations={self.iterations}, backups={self.backups}, "
            f"converged={self.converged}, seconds={self.seconds:.4f})"
        )


def _finish(mdp: MDP, V: np.ndarray, **kwargs) -> Solution:
    Q = mdp.q_values(V)
    return Solution(V=V, Q=Q, policy=Q.argmax(axis=1), **kwargs)


# --------------------------------------------------------------------------
# Value iteration
# --------------------------------------------------------------------------


def value_iteration(
    mdp: MDP,
    tol: float = 1e-6,
    max_iter: int = 1_000_000,
    v0: np.ndarray | None = None,
) -> Solution:
    """Synchronous value iteration.

    Stops when the largest change of any state's value in one sweep drops
    below ``tol``.  ``history`` records that change for every sweep
    (``delta``) and how many states changed their greedy action (``changed``).
    """
    S, A = mdp.n_states, mdp.n_actions
    V = np.zeros(S) if v0 is None else np.array(v0, dtype=float)
    deltas, changed = [], []
    policy = np.full(S, -1)
    converged = False
    start = time.perf_counter()
    for it in range(1, max_iter + 1):
        Q = mdp.q_values(V)
        V_new = Q.max(axis=1)
        new_policy = Q.argmax(axis=1)
        changed.append(int((new_policy != policy)[mdp.active].sum()))
        policy = new_policy
        delta = float(np.abs(V_new - V).max())
        deltas.append(delta)
        V = V_new
        if delta < tol:
            converged = True
            break
    seconds = time.perf_counter() - start
    changed_arr = np.array(changed)
    nz = np.flatnonzero(changed_arr)
    return _finish(
        mdp,
        V,
        iterations=it,
        converged=converged,
        method="value iteration",
        backups=it * mdp.n_active * A,
        seconds=seconds,
        history={
            "delta": np.array(deltas),
            "changed": changed_arr,
            # sweep after which the greedy policy never changed again
            "policy_settled": int(nz[-1]) + 1 if nz.size else 0,
        },
    )


# --------------------------------------------------------------------------
# Policy iteration
# --------------------------------------------------------------------------


def policy_iteration(
    mdp: MDP,
    evaluation: str = "exact",
    eval_tol: float = 1e-6,
    warm_start: bool = True,
    policy0: np.ndarray | None = None,
    max_iter: int = 10_000,
    improve_tol: float | None = None,
) -> Solution:
    """Howard's policy iteration.

    Parameters
    ----------
    evaluation : ``"exact"`` or ``"iterative"``
        Evaluate each policy by solving its linear system, or by sweeping
        ``V <- r_pi + gamma P_pi V`` until the largest change is below
        ``eval_tol``.  The second is how textbook pseudo-code (and, by every
        indication, the 2014 implementation) does it.
    warm_start : bool
        With iterative evaluation, start each evaluation from the previous
        policy's values rather than from zero.
    policy0 : initial policy (default: action 0 everywhere).
    improve_tol : an action replaces the current one only if it is better by
        more than this margin.  Guards against flip-flopping between ties.

    ``iterations`` counts policy evaluations, including the last one that
    confirms no improvement is possible.
    """
    if evaluation not in ("exact", "iterative"):
        raise ValueError("evaluation must be 'exact' or 'iterative'")
    S, A = mdp.n_states, mdp.n_actions
    states = np.arange(S)
    policy = np.zeros(S, dtype=np.int64) if policy0 is None else np.array(policy0, dtype=np.int64)
    if improve_tol is None:
        # Iterative evaluation stopped at eval_tol is only accurate to about
        # eval_tol * gamma / (1 - gamma); differences below that are noise.
        improve_tol = 1e-9 if evaluation == "exact" else 2 * eval_tol * mdp.gamma / (1 - mdp.gamma)
    V = np.zeros(S)
    backups = solves = 0
    sweeps_per_eval, n_changed = [], []
    converged = False
    start = time.perf_counter()
    for it in range(1, max_iter + 1):
        if evaluation == "exact":
            V = evaluate_policy(mdp, policy)
            solves += 1
            sweeps_per_eval.append(0)
        else:
            V, sweeps = iterative_policy_evaluation(
                mdp, policy, tol=eval_tol, v0=V if warm_start else None
            )
            backups += sweeps * mdp.n_active
            sweeps_per_eval.append(sweeps)
        Q = mdp.q_values(V)
        backups += mdp.n_active * A
        better = (Q.max(axis=1) > Q[states, policy] + improve_tol) & mdp.active
        n_changed.append(int(better.sum()))
        if not better.any():
            converged = True
            break
        policy = np.where(better, Q.argmax(axis=1), policy)
    seconds = time.perf_counter() - start
    return Solution(
        V=V,
        Q=Q,
        policy=policy,
        iterations=it,
        converged=converged,
        method=f"policy iteration ({evaluation})",
        backups=backups,
        solves=solves,
        seconds=seconds,
        history={"eval_sweeps": np.array(sweeps_per_eval), "changed": np.array(n_changed)},
    )


def modified_policy_iteration(
    mdp: MDP,
    m: int = 10,
    tol: float = 1e-6,
    max_iter: int = 1_000_000,
    v0: np.ndarray | None = None,
) -> Solution:
    """Modified policy iteration (Puterman & Shin, 1978).

    Each iteration does one greedy improvement followed by ``m`` cheap
    evaluation sweeps of the new policy.  ``m = 0`` is value iteration; large
    ``m`` approaches policy iteration.
    """
    S, A = mdp.n_states, mdp.n_actions
    V = np.zeros(S) if v0 is None else np.array(v0, dtype=float)
    deltas = []
    converged = False
    start = time.perf_counter()
    for it in range(1, max_iter + 1):
        Q = mdp.q_values(V)
        V_new = Q.max(axis=1)
        delta = float(np.abs(V_new - V).max())
        deltas.append(delta)
        if delta < tol:
            V = V_new
            converged = True
            break
        P_pi, r_pi = mdp.policy_model(Q.argmax(axis=1))
        for _ in range(m):
            V_new = r_pi + mdp.gamma * (P_pi @ V_new)
        V = V_new
    seconds = time.perf_counter() - start
    return _finish(
        mdp,
        V,
        iterations=it,
        converged=converged,
        method=f"modified policy iteration (m={m})",
        backups=it * mdp.n_active * A + (it - 1) * m * mdp.n_active,
        seconds=seconds,
        history={"delta": np.array(deltas)},
    )


# --------------------------------------------------------------------------
# Asynchronous value iteration
# --------------------------------------------------------------------------


def _local_tables(mdp: MDP):
    """Per-state successor tables: ``next[s]`` and ``prob[s]`` have shape (A, K)."""
    sampler = mdp.sampler
    S, A = mdp.n_states, mdp.n_actions
    K = sampler.n_successors
    return sampler.next_state.reshape(S, A, K), sampler.probability.reshape(S, A, K)


def gauss_seidel_value_iteration(
    mdp: MDP,
    tol: float = 1e-6,
    max_iter: int = 100_000,
    order: np.ndarray | None = None,
) -> Solution:
    """In-place value iteration: each backup sees the values already updated in this sweep.

    ``order`` is the sequence in which states are swept (default: by index).
    """
    S, A = mdp.n_states, mdp.n_actions
    nxt, prob = _local_tables(mdp)
    R, gamma = mdp.R, mdp.gamma
    states = np.flatnonzero(mdp.active) if order is None else np.asarray(order)
    V = np.zeros(S)
    deltas = []
    converged = False
    start = time.perf_counter()
    for it in range(1, max_iter + 1):
        delta = 0.0
        for s in states:
            v = (R[s] + gamma * (prob[s] * V[nxt[s]]).sum(axis=1)).max()
            change = abs(v - V[s])
            if change > delta:
                delta = change
            V[s] = v
        deltas.append(delta)
        if delta < tol:
            converged = True
            break
    seconds = time.perf_counter() - start
    return _finish(
        mdp,
        V,
        iterations=it,
        converged=converged,
        method="Gauss-Seidel value iteration",
        backups=it * states.size * A,
        seconds=seconds,
        history={"delta": np.array(deltas)},
    )


def prioritized_sweeping(
    mdp: MDP,
    tol: float = 1e-6,
    max_backups: int = 500_000_000,
) -> Solution:
    """Value iteration that always backs up the state whose value is most out of date.

    After a state's value moves by ``d``, each predecessor ``p`` has its
    priority raised by ``gamma * max_a p(s | p, a) * d``, an upper bound on how
    far its own value can now be off.  States are backed up in order of
    priority until every priority is below ``tol``.  (Moore & Atkeson, 1993,
    for the case where the model is known.)
    """
    S, A = mdp.n_states, mdp.n_actions
    nxt, prob = _local_tables(mdp)
    R, gamma = mdp.R, mdp.gamma
    active = mdp.active

    # predecessors[s] = (states p that can reach s in one move, max_a p(s | p, a))
    Pc = sp.csc_matrix(mdp.P)
    predecessors = []
    for s in range(S):
        rows = Pc.indices[Pc.indptr[s] : Pc.indptr[s + 1]]
        vals = Pc.data[Pc.indptr[s] : Pc.indptr[s + 1]]
        p = rows // A
        keep = active[p] & (p != s)
        p, vals = p[keep], vals[keep]
        if p.size:
            uniq, inv = np.unique(p, return_inverse=True)
            w = np.zeros(uniq.size)
            np.maximum.at(w, inv, vals)
            predecessors.append((uniq, gamma * w))
        else:
            predecessors.append((np.empty(0, dtype=np.int64), np.empty(0)))

    start = time.perf_counter()
    V = np.zeros(S)
    # Initial priorities: the Bellman residual of every state at V = 0.
    priority = np.abs(mdp.q_values(V).max(axis=1) - V)
    priority[~active] = 0.0
    backups = mdp.n_active * A
    heap = [(-p, s) for s, p in enumerate(priority) if p >= tol]
    heapq.heapify(heap)
    n_updates = 0
    while heap and backups < max_backups:
        neg, s = heapq.heappop(heap)
        if -neg != priority[s]:
            continue  # stale entry
        v = (R[s] + gamma * (prob[s] * V[nxt[s]]).sum(axis=1)).max()
        backups += A
        n_updates += 1
        change = abs(v - V[s])
        V[s] = v
        priority[s] = 0.0
        # a self-loop keeps the state itself slightly out of date
        own = prob[s][nxt[s] == s]
        if own.size:
            priority[s] = gamma * own.max() * change
            if priority[s] >= tol:
                heapq.heappush(heap, (-priority[s], s))
        preds, weights = predecessors[s]
        if preds.size and change > 0.0:
            new = priority[preds] + weights * change
            priority[preds] = new
            for p, pr in zip(preds[new >= tol].tolist(), new[new >= tol].tolist()):
                heapq.heappush(heap, (-pr, p))
    seconds = time.perf_counter() - start
    return _finish(
        mdp,
        V,
        iterations=n_updates,
        converged=not heap,
        method="prioritized sweeping",
        backups=backups,
        seconds=seconds,
    )


# --------------------------------------------------------------------------
# Linear programming
# --------------------------------------------------------------------------


def linear_programming(mdp: MDP) -> Solution:
    """Solve the MDP as a linear program.

    The optimal value function is the smallest ``V`` that satisfies
    ``V(s) >= r(s, a) + gamma * sum_s' p(s'|s,a) V(s')`` for every state and
    action, so minimising ``sum_s V(s)`` under those constraints returns it.
    """
    S, A = mdp.n_states, mdp.n_actions
    idx = np.flatnonzero(mdp.active)
    n = idx.size
    rows = (idx[:, None] * A + np.arange(A)[None, :]).ravel()
    P_act = sp.csr_matrix(mdp.P[rows])[:, idx]
    select = sp.csr_matrix((np.ones(n * A), (np.arange(n * A), np.repeat(np.arange(n), A))), shape=(n * A, n))
    start = time.perf_counter()
    res = linprog(
        c=np.ones(n),
        A_ub=-(select - mdp.gamma * P_act),
        b_ub=-mdp.R[idx].ravel(),
        bounds=(None, None),
        method="highs",
    )
    seconds = time.perf_counter() - start
    V = np.zeros(S)
    if res.status == 0:
        V[idx] = res.x
    return _finish(
        mdp,
        V,
        iterations=int(getattr(res, "nit", 0)),
        converged=res.status == 0,
        method="linear programming",
        seconds=seconds,
    )
