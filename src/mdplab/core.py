"""Finite Markov decision processes in matrix form.

Everything in this package works on one representation: a transition matrix
``P`` with one row per (state, action) pair, so that a full Bellman backup is a
single matrix-vector product.  ``P`` may be a dense array (the crazy elevator,
where every floor is reachable from every floor) or a SciPy sparse matrix (the
grid worlds, where a move has at most three outcomes).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import scipy.sparse as sp

__all__ = ["MDP", "Sampler", "greedy", "random_argmax"]


@dataclass(eq=False)
class MDP:
    """A finite MDP with ``S`` states and ``A`` actions.

    Parameters
    ----------
    P : array or sparse matrix, shape (S * A, S)
        Row ``s * A + a`` is the distribution over next states after taking
        action ``a`` in state ``s``.
    r_next : array, shape (S,)
        Reward collected on *arriving* in a state.  Both MDPs in the 2014
        report pay rewards this way.
    gamma : float
        Discount factor.
    terminal : bool array, shape (S,)
        Absorbing states.  Nothing is earned after reaching one.  Their rows
        in ``P`` are forced to self-loops so that ``P`` stays stochastic.
    r_sa : array, shape (S, A), optional
        Additional reward paid for taking ``a`` in ``s`` regardless of outcome.
    """

    P: np.ndarray | sp.csr_matrix
    r_next: np.ndarray
    gamma: float
    terminal: np.ndarray
    r_sa: np.ndarray | None = None
    name: str = "mdp"
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if sp.issparse(self.P):
            self.P = sp.csr_matrix(self.P, dtype=float)
        else:
            self.P = np.ascontiguousarray(self.P, dtype=float)
        n_rows, n_states = self.P.shape
        if n_rows % n_states:
            raise ValueError("P must have shape (S * A, S)")
        self.r_next = np.asarray(self.r_next, dtype=float)
        self.terminal = np.asarray(self.terminal, dtype=bool)
        if self.r_next.shape != (n_states,) or self.terminal.shape != (n_states,):
            raise ValueError("r_next and terminal must have shape (S,)")
        n_actions = n_rows // n_states
        if self.r_sa is None:
            self.r_sa = np.zeros((n_states, n_actions))
        self.r_sa = np.asarray(self.r_sa, dtype=float)
        if self.r_sa.shape != (n_states, n_actions):
            raise ValueError("r_sa must have shape (S, A)")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must lie in [0, 1]")
        self._absorb_terminals()

    # ------------------------------------------------------------------ shape

    @property
    def n_states(self) -> int:
        return self.P.shape[1]

    @property
    def n_actions(self) -> int:
        return self.P.shape[0] // self.P.shape[1]

    @property
    def is_sparse(self) -> bool:
        return sp.issparse(self.P)

    @cached_property
    def active(self) -> np.ndarray:
        """States in which a decision is actually made (non-terminal)."""
        return ~self.terminal

    @property
    def n_active(self) -> int:
        return int(self.active.sum())

    # ---------------------------------------------------------------- rewards

    @cached_property
    def R(self) -> np.ndarray:
        """Expected immediate reward ``r(s, a)``, shape (S, A)."""
        R = self.r_sa + (self.P @ self.r_next).reshape(self.n_states, self.n_actions)
        R[self.terminal] = 0.0
        return R

    # ---------------------------------------------------------------- backups

    def q_values(self, V: np.ndarray, gamma: float | None = None) -> np.ndarray:
        """One-step lookahead ``Q(s, a) = r(s, a) + gamma * sum_s' p(s'|s,a) V(s')``."""
        g = self.gamma if gamma is None else gamma
        return self.R + g * (self.P @ V).reshape(self.n_states, self.n_actions)

    def policy_model(self, policy: np.ndarray):
        """Markov reward process induced by a policy.

        ``policy`` is either an integer array of shape (S,) (deterministic) or
        a row-stochastic array of shape (S, A).  Returns ``(P_pi, r_pi)``.
        """
        policy = np.asarray(policy)
        S, A = self.n_states, self.n_actions
        if policy.ndim == 1:
            rows = np.arange(S) * A + policy
            return self.P[rows], self.R[np.arange(S), policy]
        if policy.shape != (S, A):
            raise ValueError("a stochastic policy must have shape (S, A)")
        W = sp.csr_matrix(
            (policy.ravel(), (np.repeat(np.arange(S), A), np.arange(S * A))),
            shape=(S, S * A),
        )
        P_pi = W @ self.P
        return P_pi, (policy * self.R).sum(axis=1)

    # --------------------------------------------------------------- sampling

    @cached_property
    def sampler(self) -> "Sampler":
        return Sampler(self)

    # ------------------------------------------------------------- validation

    def _absorb_terminals(self) -> None:
        """Make every terminal state a zero-reward self-loop."""
        S, A = self.n_states, self.n_actions
        term = np.flatnonzero(self.terminal)
        if term.size == 0:
            return
        rows = (term[:, None] * A + np.arange(A)[None, :]).ravel()
        cols = np.repeat(term, A)
        if self.is_sparse:
            keep = np.ones(S * A)
            keep[rows] = 0.0
            loops = sp.csr_matrix((np.ones(rows.size), (rows, cols)), shape=self.P.shape)
            self.P = (sp.diags(keep) @ self.P + loops).tocsr()
            self.P.eliminate_zeros()
            self.P.sort_indices()
        else:
            self.P[rows] = 0.0
            self.P[rows, cols] = 1.0
        self.r_sa = self.r_sa.copy()
        self.r_sa[term] = 0.0

    def validate(self, atol: float = 1e-9) -> None:
        """Raise ``ValueError`` unless ``P`` is a proper transition matrix."""
        P = self.P
        data = P.data if self.is_sparse else P
        if (data < -atol).any():
            raise ValueError("negative transition probability")
        sums = np.asarray(P.sum(axis=1)).ravel()
        if not np.allclose(sums, 1.0, atol=atol):
            bad = int(np.abs(sums - 1.0).argmax())
            raise ValueError(f"row {bad} of P sums to {sums[bad]!r}")

    def __repr__(self) -> str:
        kind = "sparse" if self.is_sparse else "dense"
        return (
            f"MDP({self.name!r}, states={self.n_states}, actions={self.n_actions}, "
            f"gamma={self.gamma}, {kind})"
        )


class Sampler:
    """Draws transitions from an :class:`MDP`, vectorised over many draws.

    Successors are stored in a padded table with one row per (state, action)
    pair, so a batch of transitions costs one comparison per stored successor.
    """

    def __init__(self, mdp: MDP) -> None:
        self.mdp = mdp
        S, A = mdp.n_states, mdp.n_actions
        M = S * A
        if mdp.is_sparse:
            P = mdp.P
            counts = np.diff(P.indptr)
            K = int(counts.max())
            nxt = np.zeros((M, K), dtype=np.int64)
            prob = np.zeros((M, K))
            row = np.repeat(np.arange(M), counts)
            pos = np.arange(P.nnz) - np.repeat(P.indptr[:-1], counts)
            nxt[row, pos] = P.indices
            prob[row, pos] = P.data
            # Pad with the last real successor so a rounding slip cannot
            # land on a state the row never reaches.
            last = nxt[np.arange(M), counts - 1]
            pad = np.arange(K)[None, :] >= counts[:, None]
            nxt[pad] = np.broadcast_to(last[:, None], nxt.shape)[pad]
        else:
            K = S
            nxt = np.broadcast_to(np.arange(S), (M, S)).copy()
            prob = mdp.P
        cum = np.cumsum(prob, axis=1)
        cum /= cum[:, -1:]
        cum[cum > 1.0 - 1e-12] = 1.0
        self.n_successors = K
        self.next_state = nxt
        self.probability = prob
        self.cumulative = cum
        # Flat layout for sampling every row at once with one binary search.
        self._offset = np.arange(M, dtype=float)
        self._flat_cum = (cum + self._offset[:, None]).ravel()
        self._flat_next = nxt.ravel()

    def step(self, s: np.ndarray, a: np.ndarray, rng: np.random.Generator):
        """Sample ``s'`` and the reward for arrays of states and actions."""
        mdp = self.mdp
        row = s * mdp.n_actions + a
        u = rng.random(row.shape[0])
        k = (u[:, None] >= self.cumulative[row]).sum(axis=1)
        s2 = self.next_state[row, k]
        return s2, mdp.r_sa[s, a] + mdp.r_next[s2]

    def sweep(self, rng: np.random.Generator, n_runs: int = 1) -> np.ndarray:
        """One draw of ``s'`` for *every* (state, action) pair.

        Returns an integer array of shape (n_runs, S * A).
        """
        M, K = self.cumulative.shape
        u = rng.random((n_runs, M)) + self._offset[None, :]
        pos = np.searchsorted(self._flat_cum, u.ravel(), side="right")
        pos = np.minimum(pos, np.tile((np.arange(M) + 1) * K - 1, n_runs))
        return self._flat_next[pos].reshape(n_runs, M)


def greedy(Q: np.ndarray) -> np.ndarray:
    """Greedy policy for an array of action values (ties go to the lowest index)."""
    return Q.argmax(axis=-1)


def random_argmax(q: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Row-wise argmax with ties broken uniformly at random."""
    ties = q == q.max(axis=-1, keepdims=True)
    return np.where(ties, rng.random(q.shape), -1.0).argmax(axis=-1)
