import numpy as np
import pytest

from mdplab import (
    GridWorld,
    crazy_elevator,
    evaluate_policy,
    gauss_seidel_value_iteration,
    linear_programming,
    modified_policy_iteration,
    policy_iteration,
    prioritized_sweeping,
    value_iteration,
)
from mdplab.legacy import policy_iteration_2014, value_iteration_2014


def test_all_planners_agree(small_mdp):
    mdp = small_mdp
    ref = policy_iteration(mdp)
    assert ref.converged
    scale = max(1.0, np.abs(ref.V).max())
    solvers = {
        "vi": lambda: value_iteration(mdp, tol=1e-10),
        "pi-iterative": lambda: policy_iteration(mdp, evaluation="iterative", eval_tol=1e-11),
        "mpi": lambda: modified_policy_iteration(mdp, m=8, tol=1e-10),
        "gauss-seidel": lambda: gauss_seidel_value_iteration(mdp, tol=1e-10),
        "prioritized": lambda: prioritized_sweeping(mdp, tol=1e-10),
        "lp": lambda: linear_programming(mdp),
    }
    for name, solve in solvers.items():
        sol = solve()
        assert sol.converged, name
        assert np.abs(sol.V - ref.V).max() < 1e-5 * scale, name
        # the returned policy is optimal: its exact value matches V*
        assert np.abs(evaluate_policy(mdp, sol.policy) - ref.V).max() < 1e-5 * scale, name


def test_optimal_value_satisfies_bellman_equation(small_mdp):
    V = policy_iteration(small_mdp).V
    assert np.allclose(small_mdp.q_values(V).max(axis=1), V, atol=1e-8)


def test_value_iteration_history_and_backups(fig1):
    sol = value_iteration(fig1.mdp, tol=1e-6)
    assert sol.iterations == 159
    assert len(sol.history["delta"]) == 159 and sol.history["delta"][-1] < 1e-6
    assert sol.backups == 159 * fig1.free.sum() * 4
    assert 0 < sol.history["policy_settled"] < sol.iterations


def test_modified_policy_iteration_with_m_zero_is_value_iteration(fig1):
    a = value_iteration(fig1.mdp, tol=1e-8)
    b = modified_policy_iteration(fig1.mdp, m=0, tol=1e-8)
    assert a.iterations == b.iterations
    assert np.allclose(a.V, b.V)


def test_policy_iteration_counts_work(fig1):
    exact = policy_iteration(fig1.mdp)
    sweeping = policy_iteration(fig1.mdp, evaluation="iterative")
    assert exact.solves == exact.iterations and exact.backups == exact.iterations * 42 * 4
    assert sweeping.solves == 0
    assert sweeping.backups == sweeping.history["eval_sweeps"].sum() * 42 + sweeping.iterations * 42 * 4


def test_textbook_solvers_match_the_report():
    # iteration counts read off Figure 7 of the report (to within the reading error)
    for floors, count in ((10, 81), (30, 235), (50, 351)):
        mdp = crazy_elevator(floors)
        assert abs(value_iteration_2014(mdp).iterations - count) <= 1
        assert value_iteration_2014(mdp).iterations - value_iteration(mdp).iterations in (0, 1)
        assert policy_iteration_2014(mdp).iterations <= 4
    mdp = crazy_elevator(15)
    assert np.allclose(value_iteration_2014(mdp, loops=True).U, value_iteration_2014(mdp).U)
    assert (policy_iteration_2014(mdp, loops=True).policy == policy_iteration_2014(mdp).policy).all()


def test_stalling_state_sets_the_iteration_ceiling():
    # a walled-in cell can only bump into walls for ever: its value converges at rate gamma exactly
    gw = GridWorld.from_text(
        """
        .#...
        ##..G
        """
    )
    sol = value_iteration(gw.mdp, tol=1e-6)
    assert sol.V[0] == pytest.approx(-0.01 / (1 - 0.99), abs=1e-3)
    assert sol.iterations == 918
