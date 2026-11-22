import numpy as np
import pytest

from mdplab import GridWorld, crazy_elevator, garnet


@pytest.fixture(scope="session")
def fig1():
    return GridWorld.report_figure1()


@pytest.fixture(scope="session", params=["gridworld", "elevator", "garnet"])
def small_mdp(request):
    if request.param == "gridworld":
        return GridWorld.random(6, 7, seed=3).mdp
    if request.param == "elevator":
        return crazy_elevator(12)
    return garnet(30, 4, 3, gamma=0.9, seed=1)


@pytest.fixture
def rng():
    return np.random.default_rng(0)
