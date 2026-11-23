import numpy as np
import pytest

from mdplab import REPORT_POLICY_50, GridWorld, crazy_elevator, elevator_transitions, value_iteration
from mdplab.gridworld import DOWN, GOAL, OBSTACLE, PIT, RIGHT, UP


def test_gridworld_moves_slips_and_bumps():
    gw = GridWorld.from_text(
        """
        ...
        .#.
        ..G
        """
    )
    mdp = gw.mdp
    mdp.validate()
    P = mdp.P.toarray().reshape(9, 4, 9)
    s = gw.state(0, 0)  # top-left corner
    # up: blocked by the edge (0.8), slip left blocked (0.1), slip right moves (0.1)
    assert np.isclose(P[s, UP, s], 0.9) and np.isclose(P[s, UP, gw.state(0, 1)], 0.1)
    # right: moves (0.8), slip up blocked (0.1), slip down moves (0.1)
    assert np.isclose(P[s, RIGHT, gw.state(0, 1)], 0.8)
    assert np.isclose(P[s, RIGHT, s], 0.1) and np.isclose(P[s, RIGHT, gw.state(1, 0)], 0.1)
    # walking into the obstacle from above leaves the agent in place
    s = gw.state(0, 1)
    assert np.isclose(P[s, DOWN, s], 0.8)
    # nothing ever enters the obstacle
    assert P[:, :, gw.state(1, 1)].sum() == pytest.approx(4.0)  # only its own self-loops


def test_gridworld_rewards_and_terminals():
    gw = GridWorld.from_text("..x\n.#G", step_reward=-0.5, goal_reward=7, pit_reward=-9)
    mdp = gw.mdp
    assert list(mdp.r_next) == [-0.5, -0.5, -9, -0.5, 0, 7]
    assert list(mdp.terminal) == [False, False, True, False, True, True]


def test_random_gridworld_follows_the_report_recipe():
    gw = GridWorld.random(60, 60, seed=0)
    flat = gw.layout.ravel()
    assert (flat == GOAL).sum() == 1
    # about 10% obstacles and slightly fewer pits, since obstacles overwrite pits
    assert 0.08 < (flat == OBSTACLE).mean() < 0.12
    assert 0.07 < (flat == PIT).mean() < 0.11
    assert GridWorld.random(5, 5, seed=4).to_text() == GridWorld.random(5, 5, seed=4).to_text()


def test_text_round_trip(fig1):
    assert GridWorld.from_text(fig1.to_text()).to_text() == fig1.to_text()
    policy = np.arange(fig1.n_states) % 4
    back = fig1.policy_from_text(fig1.policy_to_text(policy))
    assert (back[fig1.free] == policy[fig1.free]).all()


def test_connected_to_goal_finds_walled_in_cells():
    gw = GridWorld.from_text(
        """
        .#..
        ##.G
        """
    )
    connected = gw.connected_to_goal().reshape(gw.shape)
    assert not connected[0, 0]
    assert connected[0, 2] and connected[0, 3] and connected[1, 2]


@pytest.mark.parametrize("formula", ["as_run", "as_written"])
def test_elevator_is_a_proper_mdp(formula):
    mdp = crazy_elevator(25, formula=formula)
    mdp.validate()
    assert mdp.n_states == 25 and mdp.n_actions == 25
    assert mdp.terminal.sum() == 1 and mdp.terminal[0]
    assert mdp.r_next[0] == 500 and mdp.r_next[7] == -7


def test_elevator_formula_as_written():
    T = elevator_transitions(8, formula="as_written")
    k, a, j = 5, 3, 6

    def w(jj):
        return abs(np.sin(k + np.sin(k * a + np.sin(jj**2))))

    assert T[k, a, j] == pytest.approx(w(j) / sum(w(jj) for jj in range(8)))


def test_elevator_as_run_reproduces_the_report_policy():
    """The 50-floor policy printed in the report is optimal for sin(j XOR 2), not for sin(j ** 2)."""
    as_run = value_iteration(crazy_elevator(50), tol=1e-10).policy
    as_written = value_iteration(crazy_elevator(50, formula="as_written"), tol=1e-10).policy
    assert (as_run[1:] == REPORT_POLICY_50[1:]).all()
    assert (as_written[1:] == REPORT_POLICY_50[1:]).mean() < 0.1


def test_elevator_options():
    mdp = crazy_elevator(20, buttons=5)
    assert mdp.n_actions == 5
    trap = crazy_elevator(20, trap_floor=13, trap_reward=-500)
    assert trap.terminal[13] and trap.r_next[13] == -500
