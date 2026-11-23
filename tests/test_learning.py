import numpy as np
import pytest

from mdplab import (
    GridWorld,
    PolicyScorer,
    crazy_elevator,
    empirical_model,
    evaluate_policy,
    exponential_decay,
    harmonic,
    linear_decay,
    polynomial,
    synchronous_q_learning,
    td_control,
    uniform_random_policy,
    value_iteration,
    visit_decay,
)


@pytest.fixture(scope="module")
def tiny():
    gw = GridWorld.from_text(
        """
        ...G
        .#.x
        ....
        """,
        step_reward=-1.0,
        goal_reward=10.0,
        pit_reward=-10.0,
        gamma=0.9,
    )
    return gw.mdp


def test_schedules():
    assert harmonic()(4) == 0.25
    assert polynomial(0.5)(4) == 0.5 and polynomial(1.0, scale=10)(2) == 1.0
    assert linear_decay(1.0, 0.0, 10)(5) == 0.5 and linear_decay(1.0, 0.1, 10)(50) == pytest.approx(0.1)
    assert exponential_decay(1.0, 0.0, 10)(10) == pytest.approx(0.5)
    assert exponential_decay(1.0, 0.2, 10)(1e9) == pytest.approx(0.2)


def test_scorer_recognises_the_optimal_policy(tiny):
    scorer = PolicyScorer(tiny)
    out = scorer(scorer.Q_star)
    assert out["value_gap"][0] == pytest.approx(0.0, abs=1e-9) and out["agreement"][0] == 1.0
    worse = scorer(-scorer.Q_star)
    assert worse["value_gap"][0] > 0 and worse["agreement"][0] < 1.0


def test_synchronous_q_learning_converges(tiny):
    scorer = PolicyScorer(tiny)
    res = synchronous_q_learning(tiny, 4000, alpha=polynomial(0.7), n_runs=4, seed=0, scorer=scorer)
    err = np.abs(res.Q - scorer.Q_star[None])[:, tiny.active].max()
    assert err < 1.0
    assert res.history["value_gap"][-1].max() < 0.05
    assert res.history["value_gap"].shape == (len(res.steps), 4)


def test_one_over_n_lags_behind_on_a_long_horizon(fig1):
    """The reconstruction of the 2014 learner: uniform visits, step size 1/n, 5600 visits per pair."""
    scorer = PolicyScorer(fig1.mdp)
    slow = synchronous_q_learning(fig1.mdp, 5600, alpha=harmonic(), n_runs=6, seed=0)
    fast = synchronous_q_learning(fig1.mdp, 5600, alpha=polynomial(0.7), n_runs=6, seed=0)
    assert scorer(slow.Q)["value_gap"].min() > 4.0
    assert scorer(fast.Q)["value_gap"].max() < 0.5
    # the three cells flagged in Figure 1b are among the mistakes of every slow run
    flagged = [fig1.state(0, 6), fig1.state(4, 3), fig1.state(6, 1)]
    wrong = ~scorer.optimal[np.arange(56)[None, :], slow.policies]
    assert wrong[:, flagged].mean() > 0.9


@pytest.mark.parametrize("method", ["q_learning", "sarsa", "expected_sarsa", "double_q"])
def test_td_control_learns(tiny, method):
    scorer = PolicyScorer(tiny)
    res = td_control(tiny, 30_000, method=method, epsilon=0.2, alpha=0.1, n_runs=6, seed=1, scorer=scorer, n_checkpoints=6)
    gap = res.history["value_gap"]
    assert gap.shape == (6, 6)
    # far better than acting at random (on-policy methods settle on a slightly
    # more cautious policy than the greedy optimum, so they are not held to zero)
    random_gap = (scorer.V_star - evaluate_policy(tiny, uniform_random_policy(tiny)))[tiny.active].mean()
    assert gap[-1].mean() < 0.1 * random_gap
    assert res.Q.shape == (6, tiny.n_states, tiny.n_actions)
    assert res.visits.sum() == 6 * 30_000
    assert (res.Q[:, tiny.terminal] == 0).all()


def test_td_control_is_reproducible(tiny):
    a = td_control(tiny, 2000, n_runs=3, seed=7)
    b = td_control(tiny, 2000, n_runs=3, seed=7)
    c = td_control(tiny, 2000, n_runs=3, seed=8)
    assert np.array_equal(a.Q, b.Q) and not np.array_equal(a.Q, c.Q)


def test_dyna_q_needs_fewer_real_steps(tiny):
    scorer = PolicyScorer(tiny)
    kw = dict(epsilon=0.2, alpha=0.2, n_runs=8, seed=0, scorer=scorer, n_checkpoints=4)
    plain = td_control(tiny, 1500, **kw)
    dyna = td_control(tiny, 1500, planning_steps=20, **kw)
    assert dyna.history["value_gap"][-1].mean() < plain.history["value_gap"][-1].mean()
    with pytest.raises(ValueError):
        td_control(tiny, 10, method="sarsa", planning_steps=5)


def test_schedules_and_optimism_run(tiny):
    res = td_control(tiny, 3000, epsilon=visit_decay(5.0), alpha=polynomial(0.7), n_runs=2, seed=0, max_episode_steps=50)
    assert np.isfinite(res.Q).all()
    res = td_control(tiny, 3000, epsilon=0.0, q_init=20.0, alpha=0.2, n_runs=2, seed=0, max_episode_steps=50)
    # with optimistic values even a greedy agent tries every action somewhere
    assert (res.visits.reshape(2, -1) > 0).mean() > 0.5


def test_empirical_model_approaches_the_truth():
    mdp = crazy_elevator(8)
    est = empirical_model(mdp, 20_000, seed=0)
    est.validate()
    assert np.abs(est.P - mdp.P).max() < 0.02
    assert np.abs(value_iteration(est).V - value_iteration(mdp).V).max() < 3.0
