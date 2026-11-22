"""Random MDPs with adjustable size, used as a controlled test bed.

"Garnet" problems (Archibald, McKinnon & Thomas, 1995) are the standard way
to generate MDPs with a chosen number of states, actions and successors per
action but no other structure.  They make it possible to change one of those
numbers while holding the others fixed, which neither the grid world (always
four actions) nor the crazy elevator (as many actions as states) allows.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .core import MDP

__all__ = ["garnet"]


def garnet(
    n_states: int,
    n_actions: int,
    branching: int,
    gamma: float = 0.95,
    seed: int | np.random.Generator | None = None,
) -> MDP:
    """A random MDP in which every action has ``branching`` possible next states.

    For each state-action pair the successors are drawn uniformly without
    replacement, their probabilities uniformly from the simplex, and the
    reward ``r(s, a)`` from a standard normal.  There are no terminal states.
    """
    if not 1 <= branching <= n_states:
        raise ValueError("branching must lie between 1 and n_states")
    rng = np.random.default_rng(seed)
    M = n_states * n_actions
    # `branching` distinct successors per row: rank random keys
    succ = np.argsort(rng.random((M, n_states)), axis=1)[:, :branching]
    prob = rng.dirichlet(np.ones(branching), size=M)
    rows = np.repeat(np.arange(M), branching)
    P = sp.csr_matrix((prob.ravel(), (rows, succ.ravel())), shape=(M, n_states))
    if branching > n_states // 4:
        P = P.toarray()
    return MDP(
        P=P,
        r_next=np.zeros(n_states),
        r_sa=rng.standard_normal((n_states, n_actions)),
        gamma=gamma,
        terminal=np.zeros(n_states, dtype=bool),
        name=f"garnet({n_states}, {n_actions}, {branching})",
    )
