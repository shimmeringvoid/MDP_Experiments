import numpy as np
import pytest

from mdplab import (
    GridWorld,
    absorption,
    agreement,
    crazy_elevator,
    evaluate_policy,
    hitting_probability,
    iterative_policy_evaluation,
    monte_carlo,
    optimal_actions,
    uniform_random_policy,
    value_gap,
    value_iteration,
)
from mdplab.gridworld import GOAL, PIT, REPORT_FIG1A_POLICY, REPORT_FIG1B_POLICY


def test_exact_and_iterative_evaluation_agree(small_mdp, rng):
    policy = rng.integers(0, small_mdp.n_actions, size=small_mdp.n_states)
    exact = evaluate_policy(small_mdp, policy)
    swept, sweeps = iterative_policy_evaluation(small_mdp, policy, tol=1e-12)
    assert sweeps > 1
    assert np.allclose(exact, swept, atol=1e-7 * max(1, np.abs(exact).max()))


def test_report_figure1_numbers(fig1):
    """Expected rewards in the report's table: 199.9 for the plan, 193.9 for the Q-learned policy."""
    mdp, free = fig1.mdp, fig1.free
    plan = fig1.policy_from_text(REPORT_FIG1A_POLICY)
    learned = fig1.policy_from_text(REPORT_FIG1B_POLICY)
    assert absorption(mdp, plan).total_reward[free].mean() == pytest.approx(199.905, abs=5e-3)
    assert absorption(mdp, learned).total_reward[free].mean() == pytest.approx(193.2, abs=0.1)
    # and Figure 1a is exactly the optimal policy of the rebuilt model
    vi = value_iteration(mdp, tol=1e-10)
    assert (vi.policy[free] == plan[free]).all()
    mask = optimal_actions(mdp, vi.V)
    assert agreement(mdp, plan, mask) == 1.0
    assert agreement(mdp, learned, mask) == pytest.approx(38 / 42)
    assert value_gap(mdp, plan, vi.V) == pytest.approx(0.0, abs=1e-8)
    assert value_gap(mdp, learned, vi.V) > 5


def test_absorption_matches_simulation():
    mdp = crazy_elevator(10)
    policy = value_iteration(mdp).policy
    exact = absorption(mdp, policy)
    mc = monte_carlo(mdp, policy, n_restarts=4000, seed=0)
    act = mdp.active
    assert mc.finished.all()
    assert np.allclose(mc.steps.mean(axis=1), exact.steps[act], rtol=0.08)
    assert np.allclose(mc.total_reward.mean(axis=1), exact.total_reward[act], atol=6.0)
    assert np.allclose(mc.discounted.mean(axis=1), evaluate_policy(mdp, policy)[act], atol=6.0)


def test_random_policy_statistics_are_exact_for_stochastic_policies():
    mdp = crazy_elevator(8)
    pi = uniform_random_policy(mdp)
    exact = absorption(mdp, pi)
    mc = monte_carlo(mdp, pi, n_restarts=3000, seed=1)
    assert np.allclose(mc.steps.mean(axis=1), exact.steps[mdp.active], rtol=0.1)


def test_hitting_probabilities_split_between_goal_and_pits(fig1):
    mdp = fig1.mdp
    policy = fig1.policy_from_text(REPORT_FIG1B_POLICY)
    goal = hitting_probability(mdp, policy, fig1.mask(GOAL))
    pit = hitting_probability(mdp, policy, fig1.mask(PIT))
    free = fig1.free
    assert np.allclose((goal + pit)[free], 1.0)
    assert pit[free].max() > 0.05 and pit[free].min() >= 0.0


def test_policy_that_never_finishes_is_detected():
    gw = GridWorld.from_text("...G")
    policy = np.full(4, 3)  # always left: bumps into the wall for ever, slips are blocked too
    info = absorption(gw.mdp, policy)
    assert info.probability[0] == pytest.approx(0.0)
    assert np.isinf(info.steps[0]) and np.isnan(info.total_reward[0])
    assert not info.proper[:3].any()
    mc = monte_carlo(gw.mdp, policy, n_restarts=3, seed=0, max_steps=50)
    assert not mc.finished.any() and (mc.steps == 50).all()


def test_world_set_scores_optimal_one_and_random_zero():
    from mdplab import WorldSet, policy_iteration

    worlds = WorldSet.random(4, 4, n=6, seed=1)
    assert len(worlds) == 6 and set(worlds.optimal.columns) == {"free_cells", "lost_by_random", "p_goal", "p_pit"}
    best = worlds.evaluate(lambda world: policy_iteration(world.mdp).policy)
    assert np.allclose(best["score"], 1.0) and np.allclose(best["value_lost"], 0.0, atol=1e-8)
    assert np.allclose(best["wrong_cells"], 0.0)
    assert np.allclose(best["p_goal"], worlds.optimal["p_goal"]) and np.allclose(best["p_pit"], worlds.optimal["p_pit"])
    always_up = worlds.evaluate(lambda world: np.zeros(world.n_states, dtype=int))
    assert (always_up["score"] < 1.0).all() and (always_up["value_lost"] > 0).all()
    assert ((always_up["p_goal"] + always_up["p_pit"]) <= 1 + 1e-9).all()
