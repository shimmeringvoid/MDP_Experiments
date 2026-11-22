import numpy as np

from mdplab import evaluate_policy, optimal_actions, report, value_iteration
from mdplab.gridworld import GridWorld


def test_tables_and_figures_load():
    assert list(report.figure7().floors) == list(range(3, 60))
    assert len(report.figure2()) == 72 and report.figure2().vi_iterations.notna().all()
    assert len(report.figure4()) == 14 and len(report.figure8()) == 57
    assert report.TABLE_GRIDWORLD.loc["Value iteration", "iterations"] == 152
    assert len(report.TABLE_ELEVATOR) == 12


def test_figure1_data_files_match_the_constants(fig1):
    from importlib import resources

    text = (resources.files("mdplab") / "data" / "report_fig1_layout.txt").read_text(encoding="utf-8")
    assert GridWorld.from_text(text).to_text() == fig1.to_text()


def test_figure3_policy_is_optimal_for_the_decoded_world():
    """All 1238 arrows drawn in the report's 39 x 39 figure are optimal moves in the rebuilt model."""
    gw = report.figure3_gridworld()
    assert gw.shape == (39, 39) and gw.free.sum() == 1238
    drawn = report.figure3_policy()
    vi = value_iteration(gw.mdp, tol=1e-9)
    mask = optimal_actions(gw.mdp, vi.V, atol=1e-4)
    assert mask[np.arange(gw.n_states), drawn][gw.free].all()
    assert (drawn == vi.policy)[gw.free].mean() > 0.99
    assert np.allclose(evaluate_policy(gw.mdp, drawn), vi.V, atol=1e-3)
