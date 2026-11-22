"""Learning from sampled transitions.

Two learners live here.

:func:`synchronous_q_learning` updates every state-action pair once per sweep,
each from one freshly sampled transition.  It gives every pair exactly the
same number of visits, which is how the 2014 report's Q-learner was set up
("a minimum of 100 * m * n visits to each bin of the Q array").

:func:`td_control` is the usual online agent: it follows trajectories through
the MDP, choosing actions epsilon-greedily, and supports Q-learning, SARSA,
Expected SARSA, Double Q-learning and Dyna-Q.

Both run many independent agents at once as one batch of array operations.
The agents never interact; batching is only there so that a learning curve
with a confidence band costs little more than a single run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .core import MDP
from .planning import value_iteration

__all__ = [
    "constant",
    "harmonic",
    "polynomial",
    "linear_decay",
    "exponential_decay",
    "visit_decay",
    "PolicyScorer",
    "synchronous_q_learning",
    "td_control",
    "LearningResult",
    "empirical_model",
]

# --------------------------------------------------------------------------
# Schedules
# --------------------------------------------------------------------------
#
# A step-size schedule maps the number of times a state-action pair has been
# visited (1, 2, 3, ...) to the step size for that visit.  An exploration
# schedule maps the time step (0, 1, 2, ...) to epsilon.  Plain floats are
# accepted wherever a schedule is.

Schedule = Callable[[np.ndarray], np.ndarray]


def constant(value: float) -> Schedule:
    """The same value at every step."""
    return lambda n: np.full(np.shape(n), value) if np.ndim(n) else value


def harmonic() -> Schedule:
    """Step size ``1 / n``: the estimate is the plain average of all targets so far."""
    return lambda n: 1.0 / np.asarray(n, dtype=float)


def polynomial(omega: float, scale: float = 1.0) -> Schedule:
    """Step size ``scale / n ** omega`` (capped at 1).  ``omega = 1`` is harmonic.

    Any ``omega`` in (1/2, 1] satisfies the classical convergence conditions,
    but smaller exponents forget early, badly-informed targets much faster.
    """
    return lambda n: np.minimum(1.0, scale / np.asarray(n, dtype=float) ** omega)


def linear_decay(start: float, end: float, steps: int) -> Schedule:
    """Go linearly from ``start`` to ``end`` over ``steps`` steps, then stay."""
    return lambda t: start + (end - start) * np.minimum(np.asarray(t, dtype=float) / steps, 1.0)


def exponential_decay(start: float, end: float, half_life: float) -> Schedule:
    """Decay from ``start`` towards ``end``, halving the distance every ``half_life`` steps."""
    return lambda t: end + (start - end) * 0.5 ** (np.asarray(t, dtype=float) / half_life)


@dataclass(frozen=True)
class visit_decay:
    """Exploration rate ``n0 / (n0 + N(s))`` where ``N(s)`` counts visits to the current state.

    Explores a lot where the agent has rarely been and little where it knows
    its way around.  Every action is still tried infinitely often, and the
    policy becomes greedy in the limit.
    """

    n0: float = 10.0


def _as_schedule(x) -> Schedule:
    return x if callable(x) else constant(float(x))


# --------------------------------------------------------------------------
# Scoring learned policies against the optimum
# --------------------------------------------------------------------------


class PolicyScorer:
    """Scores greedy policies exactly, using the known model.

    A learner never sees the model, but we do, so a learned policy does not
    have to be judged by noisy rollouts: its value function is the solution of
    a linear system.  For a batch of action-value tables this returns, per
    table,

    ``value_gap``  the average of ``V*(s) - V^pi(s)`` over starting states, and
    ``agreement``  the fraction of states in which the greedy action is optimal.
    """

    def __init__(self, mdp: MDP, V_star: np.ndarray | None = None, atol: float = 1e-6) -> None:
        self.mdp = mdp
        if V_star is None:
            V_star = value_iteration(mdp, tol=1e-10).V
        self.V_star = V_star
        Q_star = mdp.q_values(V_star)
        self.Q_star = Q_star
        self.optimal = Q_star >= Q_star.max(axis=1, keepdims=True) - atol
        S, A = mdp.n_states, mdp.n_actions
        P = mdp.P.toarray() if mdp.is_sparse else mdp.P
        self._P3 = P.reshape(S, A, S)
        self._idx = np.flatnonzero(mdp.active)
        self._states = np.arange(S)

    def values(self, policies: np.ndarray) -> np.ndarray:
        """Exact value functions, shape (B, S), of a batch of deterministic policies (B, S)."""
        mdp, idx = self.mdp, self._idx
        policies = np.atleast_2d(policies)
        B = policies.shape[0]
        rows = idx[None, :, None]
        P_pi = self._P3[rows, policies[:, idx][:, :, None], idx[None, None, :]]  # (B, n, n)
        r_pi = mdp.R[idx[None, :], policies[:, idx]]  # (B, n)
        A = np.eye(idx.size)[None] - mdp.gamma * P_pi
        V = np.zeros((B, mdp.n_states))
        V[:, idx] = np.linalg.solve(A, r_pi[..., None])[..., 0]
        return V

    def __call__(self, Q: np.ndarray) -> dict[str, np.ndarray]:
        Q = Q if Q.ndim == 3 else Q[None]
        policies = Q.argmax(axis=2)
        V = self.values(policies)
        idx = self._idx
        gap = (self.V_star[None, idx] - V[:, idx]).mean(axis=1)
        agree = self.optimal[idx[None, :], policies[:, idx]].mean(axis=1)
        return {"value_gap": gap, "agreement": agree}


@dataclass
class LearningResult:
    """Outcome of a learning run.

    Attributes
    ----------
    Q : final action values, shape (n_runs, S, A).
    steps : the checkpoints (sweeps or environment steps) at which the runs were scored.
    history : dict of arrays of shape (len(steps), n_runs).
    visits : visit counts, shape (n_runs, S, A), where available.
    """

    Q: np.ndarray
    steps: np.ndarray
    history: dict = field(default_factory=dict)
    visits: np.ndarray | None = None

    @property
    def policies(self) -> np.ndarray:
        return self.Q.argmax(axis=2)


def _checkpoints(total: int, n: int, log_spaced: bool) -> np.ndarray:
    if log_spaced:
        pts = np.unique(np.rint(np.geomspace(1, total, n)).astype(np.int64))
    else:
        pts = np.unique(np.rint(np.linspace(total / n, total, n)).astype(np.int64))
    return pts[pts >= 1]


# --------------------------------------------------------------------------
# Uniform-visit ("synchronous") Q-learning
# --------------------------------------------------------------------------


def synchronous_q_learning(
    mdp: MDP,
    n_sweeps: int,
    *,
    alpha: Schedule | float = None,
    q_init: float = 0.0,
    n_runs: int = 1,
    seed: int | np.random.Generator | None = 0,
    scorer: PolicyScorer | None = None,
    n_checkpoints: int = 40,
    log_spaced: bool = True,
) -> LearningResult:
    """Q-learning in which every state-action pair is visited once per sweep.

    In each sweep, one next state is sampled for every pair and

        Q(s, a) <- Q(s, a) + alpha_n * (r + gamma * max_a' Q(s', a') - Q(s, a))

    is applied to all of them, with ``n`` the sweep number.  There is no
    exploration problem at all in this setting: coverage is perfectly uniform
    by construction.  What remains is the effect of the step size.

    The default step size is ``1 / n``.
    """
    alpha = harmonic() if alpha is None else _as_schedule(alpha)
    rng = np.random.default_rng(seed)
    S, A = mdp.n_states, mdp.n_actions
    sampler = mdp.sampler
    Q = np.full((n_runs, S, A), float(q_init))
    Q[:, mdp.terminal] = 0.0
    live = np.repeat(mdp.active, A)  # rows of (s, a) pairs that are ever updated
    r_sa = mdp.r_sa.ravel()[None, :]
    checkpoints = _checkpoints(n_sweeps, n_checkpoints, log_spaced) if scorer else np.empty(0, dtype=np.int64)
    history: dict[str, list] = {}
    next_cp = 0
    for n in range(1, n_sweeps + 1):
        s2 = sampler.sweep(rng, n_runs)  # (B, S * A)
        V = Q.max(axis=2)  # (B, S)
        boot = np.take_along_axis(V, s2, axis=1)
        target = r_sa + mdp.r_next[s2] + mdp.gamma * boot * ~mdp.terminal[s2]
        Qf = Q.reshape(n_runs, S * A)
        Qf[:, live] += float(alpha(n)) * (target - Qf)[:, live]
        if next_cp < checkpoints.size and n == checkpoints[next_cp]:
            for key, val in scorer(Q).items():
                history.setdefault(key, []).append(val)
            next_cp += 1
    return LearningResult(
        Q=Q,
        steps=checkpoints,
        history={k: np.array(v) for k, v in history.items()},
        visits=np.full((n_runs, S, A), n_sweeps) * mdp.active[None, :, None],
    )


# --------------------------------------------------------------------------
# Online temporal-difference control
# --------------------------------------------------------------------------

_METHODS = ("q_learning", "sarsa", "expected_sarsa", "double_q")


def td_control(
    mdp: MDP,
    n_steps: int,
    *,
    method: str = "q_learning",
    epsilon: Schedule | visit_decay | float = 0.1,
    alpha: Schedule | float = 0.1,
    q_init: float = 0.0,
    planning_steps: int = 0,
    n_runs: int = 16,
    seed: int | np.random.Generator | None = 0,
    max_episode_steps: int | None = None,
    scorer: PolicyScorer | None = None,
    n_checkpoints: int = 40,
    log_spaced: bool = False,
) -> LearningResult:
    """Learn action values online with an epsilon-greedy behaviour policy.

    Parameters
    ----------
    method : ``"q_learning"``, ``"sarsa"``, ``"expected_sarsa"`` or ``"double_q"``.
    epsilon : exploration rate.  A float, a schedule of the time step, or
        :class:`visit_decay` to tie it to how often the current state was seen.
    alpha : step size.  A float or a schedule of the pair's visit count.
    q_init : initial action values.  Values above anything attainable make
        even a greedy agent explore ("optimism in the face of uncertainty").
    planning_steps : if positive, Dyna-Q: after every real step, replay that
        many transitions drawn from a model built out of the agent's own
        experience (visited pairs, empirical next-state frequencies).
    n_runs : independent agents trained side by side.
    max_episode_steps : restart an episode that has run this long.
    scorer : if given, called at ``n_checkpoints`` points during training.

    Every episode starts in a state drawn uniformly from the non-terminal
    states.  ``history`` also records, per checkpoint interval, the mean
    return and length of the episodes that ended in it and the fraction that
    ended on a negative terminal reward (``failure_rate``).
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}")
    if planning_steps and method != "q_learning":
        raise ValueError("planning_steps (Dyna-Q) is only available with method='q_learning'")
    rng = np.random.default_rng(seed)
    S, A, B = mdp.n_states, mdp.n_actions, n_runs
    gamma = mdp.gamma
    sampler = mdp.sampler
    terminal = mdp.terminal
    starts = np.flatnonzero(mdp.active)
    eps_by_state = isinstance(epsilon, visit_decay)
    eps_fn = None if eps_by_state else _as_schedule(epsilon)
    alpha_fn = _as_schedule(alpha)
    double = method == "double_q"

    n_tables = 2 if double else 1
    Q = np.full((n_tables, B, S, A), float(q_init))
    Q[:, :, terminal] = 0.0
    Q2 = Q.reshape(n_tables, B * S, A)  # views: one row per (run, state) ...
    Qf = Q.reshape(n_tables, B * S * A)  # ... and one entry per (run, state, action)
    counts = np.zeros((n_tables, B * S * A), dtype=np.int64)
    state_visits = np.zeros(B * S, dtype=np.int64)
    base = np.arange(B) * S  # offset of each run's block of states
    runs = np.arange(B)

    if planning_steps:
        model = np.zeros((B, S * A, S), dtype=np.int32)  # empirical transition counts
        seen = np.zeros((B, S * A), dtype=bool)
        buffer = np.zeros((B, S * A), dtype=np.int64)  # pairs in order of first visit
        n_seen = np.zeros(B, dtype=np.int64)

    def current_eps(t, s):
        if eps_by_state:
            return epsilon.n0 / (epsilon.n0 + state_visits[base + s])
        return eps_fn(t)

    def choose(s, eps):
        """Epsilon-greedy action for every run, ties broken at random."""
        q = Q2[:, base + s]
        q = q[0] + q[1] if double else q[0]
        ties = q == q.max(axis=1, keepdims=True)
        a = np.where(ties, rng.random((B, A)), -1.0).argmax(axis=1)
        explore = rng.random(B) < eps
        if explore.any():
            a = np.where(explore, rng.integers(0, A, size=B), a)
        return a

    checkpoints = _checkpoints(n_steps, n_checkpoints, log_spaced)
    history: dict[str, list] = {}
    next_cp = 0
    ep_return = np.zeros(B)
    ep_length = np.zeros(B, dtype=np.int64)
    acc_return, acc_length = np.zeros(B), np.zeros(B)
    acc_episodes, acc_failures = np.zeros(B), np.zeros(B)

    s = starts[rng.integers(0, starts.size, size=B)]
    a = choose(s, current_eps(0, s))
    for t in range(1, n_steps + 1):
        state_visits[base + s] += 1
        s2, r = sampler.step(s, a, rng)
        done = terminal[s2]
        rows2 = base + s2

        # ---- bootstrap value of the next state ------------------------------
        table = 0
        if method == "q_learning":
            boot = Q2[0, rows2].max(axis=1)
        elif method == "sarsa":
            a2 = choose(s2, current_eps(t, s2))  # the action that will really be taken next
            boot = Q2[0, rows2, a2]
        elif method == "expected_sarsa":
            eps2 = current_eps(t, s2)
            q2 = Q2[0, rows2]
            boot = (1.0 - eps2) * q2.max(axis=1) + eps2 * q2.mean(axis=1)
        else:  # double Q-learning: one table picks the action, the other values it
            flip = rng.random(B) < 0.5
            q2 = Q2[:, rows2]  # (2, B, A)
            pick = np.where(flip[:, None], q2[0], q2[1]).argmax(axis=1)
            boot = np.where(flip, q2[1, runs, pick], q2[0, runs, pick])
            table = np.where(flip, 0, 1)

        # ---- update -----------------------------------------------------------
        idx = (base + s) * A + a
        counts[table, idx] += 1
        step = alpha_fn(counts[table, idx])
        target = r + gamma * boot * ~done
        Qf[table, idx] += step * (target - Qf[table, idx])

        # ---- Dyna-Q: also learn from remembered transitions -------------------
        if planning_steps:
            pair = s * A + a
            model[runs, pair, s2] += 1
            new = ~seen[runs, pair]
            if new.any():
                nr = runs[new]
                buffer[nr, n_seen[nr]] = pair[new]
                n_seen[nr] += 1
                seen[nr, pair[new]] = True
            pick = (rng.random((B, planning_steps)) * n_seen[:, None]).astype(np.int64)
            pairs = np.take_along_axis(buffer, pick, axis=1)  # (B, n)
            freq = np.cumsum(model[runs[:, None], pairs], axis=2)  # (B, n, S)
            u = rng.random((B, planning_steps, 1)) * freq[:, :, -1:]
            ps2 = (u >= freq).sum(axis=2)
            ps, pa = pairs // A, pairs % A
            pboot = Q[0, runs[:, None], ps2].max(axis=2)
            ptarget = mdp.r_sa[ps, pa] + mdp.r_next[ps2] + gamma * pboot * ~terminal[ps2]
            pidx = (base[:, None] + ps) * A + pa
            pstep = alpha_fn(np.maximum(counts[0, pidx], 1))
            Qf[0, pidx] += pstep * (ptarget - Qf[0, pidx])

        # ---- episode bookkeeping and the next action --------------------------
        ep_return += r
        ep_length += 1
        finished = done if max_episode_steps is None else done | (ep_length >= max_episode_steps)
        if finished.any():
            acc_return[finished] += ep_return[finished]
            acc_length[finished] += ep_length[finished]
            acc_episodes[finished] += 1
            acc_failures[finished & done & (mdp.r_next[s2] < 0)] += 1
            ep_return[finished] = 0.0
            ep_length[finished] = 0
            s2 = np.where(finished, starts[rng.integers(0, starts.size, size=B)], s2)
            fresh = choose(s2, current_eps(t, s2))
            a = np.where(finished, fresh, a2) if method == "sarsa" else fresh
        elif method == "sarsa":
            a = a2
        else:
            a = choose(s2, current_eps(t, s2))
        s = s2

        if next_cp < checkpoints.size and t == checkpoints[next_cp]:
            if scorer is not None:
                for key, val in scorer(Q.mean(axis=0)).items():
                    history.setdefault(key, []).append(val)
            with np.errstate(invalid="ignore", divide="ignore"):
                history.setdefault("episode_return", []).append(acc_return / acc_episodes)
                history.setdefault("episode_length", []).append(acc_length / acc_episodes)
                history.setdefault("failure_rate", []).append(acc_failures / acc_episodes)
            history.setdefault("episodes", []).append(acc_episodes.copy())
            acc_return[:] = acc_length[:] = acc_episodes[:] = acc_failures[:] = 0.0
            next_cp += 1

    return LearningResult(
        Q=Q.mean(axis=0),
        steps=checkpoints,
        history={k: np.array(v) for k, v in history.items()},
        visits=counts.sum(axis=0).reshape(B, S, A),
    )


# --------------------------------------------------------------------------
# Model-based learning: estimate the model, then plan
# --------------------------------------------------------------------------


def empirical_model(mdp: MDP, n_samples: int, seed: int | np.random.Generator | None = 0) -> MDP:
    """An MDP estimated from ``n_samples`` sampled transitions per state-action pair.

    The reward function and the terminal states are taken as known; only the
    transition probabilities are estimated, by counting.  Planning on the
    result is the "certainty-equivalent" way of using the same data a
    model-free learner would use.
    """
    rng = np.random.default_rng(seed)
    P = mdp.P.toarray() if mdp.is_sparse else mdp.P
    P = np.clip(P, 0.0, None)
    P = P / P.sum(axis=1, keepdims=True)
    counts = rng.multinomial(n_samples, P)
    return MDP(
        P=counts / n_samples,
        r_next=mdp.r_next,
        gamma=mdp.gamma,
        terminal=mdp.terminal,
        r_sa=mdp.r_sa,
        name=f"{mdp.name} (estimated from {n_samples} samples per pair)",
    )
