"""The deep-learning module is optional; these tests are skipped without PyTorch."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mdplab import GridWorld  # noqa: E402
from mdplab.dqn import (  # noqa: E402
    GridBatch,
    QDenseNetwork,
    QMapNetwork,
    action_values,
    greedy_policy,
    map_planes,
    sample_sweep,
    train_dqn,
    train_on_sweeps,
)
from mdplab.gridworld import OBSTACLE, RIGHT  # noqa: E402


def test_map_planes_draws_a_wall_around_the_world():
    world = GridWorld.report_figure1()
    planes = map_planes(world.layout)
    assert planes.shape == (3, world.shape[0] + 2, world.shape[1] + 2)
    assert planes[0, 0].all() and planes[0, -1].all() and planes[0, :, 0].all() and planes[0, :, -1].all()
    assert np.array_equal(planes[0, 1:-1, 1:-1], world.layout == OBSTACLE)
    assert planes[2].sum() == 1 and planes.sum(axis=0).max() == 1
    assert map_planes(np.stack([world.layout] * 5)).shape == (5, *planes.shape)


def test_grid_batch_moves_like_the_model():
    """The simulator used for deep learning and the transition matrix must describe the same world."""
    world = GridWorld.report_figure1()
    mdp = world.mdp
    rows, cols = world.shape
    n = 60_000
    env = GridBatch(lambda rng: world, n, np.random.default_rng(0), resample=False)
    for s in (world.state(6, 1), world.state(1, 6), world.state(3, 4)):
        for a in range(4):
            env.row[:], env.col[:] = divmod(s, cols)
            reward, done = env.step(np.full(n, a))
            landed = np.bincount(env.row * cols + env.col, minlength=rows * cols) / n
            expected = np.asarray(mdp.P[s * 4 + a].todense()).ravel()
            assert np.abs(landed - expected).max() < 0.01
            assert reward.mean() == pytest.approx(float(expected @ mdp.r_next), abs=2.5)
            assert np.array_equal(done, mdp.terminal[env.row * cols + env.col])


@pytest.mark.parametrize("kind", ["map", "dense"])
def test_networks_give_four_values_per_cell(kind):
    world = GridWorld.random(5, 6, seed=3)
    net = QMapNetwork(channels=8, depth=2) if kind == "map" else QDenseNetwork(5, 6, channels=8, hidden=16)
    q = action_values(net, world)
    assert q.shape == (30, 4)
    planes = torch.from_numpy(map_planes(world.layout).astype(np.float32))[None]
    one = net(planes, torch.tensor([2]), torch.tensor([4])).detach().numpy()[0]
    assert np.allclose(one, q[world.state(2, 4)], atol=1e-5)
    assert greedy_policy(net, world).shape == (30,)


def test_map_network_accepts_any_map_size():
    net = QMapNetwork(channels=8, depth=2)
    assert action_values(net, GridWorld.random(4, 4, seed=0)).shape == (16, 4)
    assert action_values(net, GridWorld.random(9, 12, seed=0)).shape == (108, 4)


def test_sample_sweep_moves_like_the_model():
    world = GridWorld.report_figure1()
    mdp = world.mdp
    rows, cols = world.shape
    n = 20_000
    landing, reward, done = sample_sweep(np.stack([world.layout] * n), world, np.random.default_rng(0))
    assert landing.shape == reward.shape == done.shape == (n, rows, cols, 4)
    for s in np.flatnonzero(world.free):
        r, c = divmod(s, cols)
        for a in range(4):
            landed = np.bincount(landing[:, r, c, a], minlength=rows * cols) / n
            assert np.abs(landed - np.asarray(mdp.P[s * 4 + a].todense()).ravel()).max() < 0.02
    free = world.free.reshape(rows, cols)  # moves are only ever used from free cells
    assert np.array_equal(done[:, free], mdp.terminal[landing[:, free]])
    assert np.allclose(reward[:, free], mdp.r_next[landing[:, free]])


# A corridor with a pit behind and the goal ahead; the discount is lowered so that the
# right action stands out clearly (at 0.99 the actions of a cell differ by about 1%).
CORRIDOR = "x...G"


@pytest.fixture(autouse=True)
def _same_initial_weights_every_time():
    torch.manual_seed(0)


def _walks_right(net, world):
    return bool((greedy_policy(net, world)[world.free] == RIGHT).all())


@pytest.mark.parametrize("tied", [False, True])
def test_sweep_training_solves_a_corridor(tied):
    world = GridWorld.from_text(CORRIDOR, gamma=0.8)
    net = QMapNetwork(channels=16, depth=4 if tied else 3, tied=tied)
    result = train_on_sweeps(
        lambda rng: world, 600, net=net, resample=False, maps_per_update=8, target_sync=10,
        evaluate=lambda net: _walks_right(net, world), eval_every=200, seed=0,
    )  # fmt: skip
    assert result.updates == 600 and len(result.losses) == 600 and result.steps == [200, 400, 600]
    assert result.scores[-1]


def test_dqn_solves_a_corridor():
    world = GridWorld.from_text(CORRIDOR, gamma=0.8)
    result = train_dqn(
        lambda rng: world, 8_000, net=QDenseNetwork(1, 5, channels=16, hidden=32), resample=False, n_envs=16,
        updates_per_step=2, max_episode_steps=20, warmup=300, target_sync=25, seed=0,
    )  # fmt: skip
    assert result.updates > 900
    assert _walks_right(result.net, world)


def test_tied_network_can_be_run_for_more_steps():
    world = GridWorld.random(6, 6, seed=2)
    net = QMapNetwork(channels=8, depth=4, tied=True)
    assert action_values(net, world, depth=12).shape == (36, 4)
    assert not np.allclose(action_values(net, world, depth=12), action_values(net, world))
