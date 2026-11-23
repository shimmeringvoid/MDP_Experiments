"""Deep Q-learning on grid worlds.

A table of action values is tied to one world: change the layout and learning
starts again from nothing.  A network that reads the *map* can in principle
learn what pits, obstacles and goals mean, and act sensibly in a world it has
never seen.  This module holds what that experiment needs:

* :class:`GridBatch`, many grid worlds simulated side by side, each with its
  own layout;
* two networks from a map to action values: :class:`QMapNetwork`, which is
  convolutional throughout and returns four values for every cell at once, and
  :class:`QDenseNetwork`, which is told where the agent stands and ends in
  dense layers;
* :func:`train_dqn`, DQN with experience replay, a target network and the
  double-Q target (Mnih et al., 2015; van Hasselt et al., 2016);
* :func:`greedy_policy`, which reads a full policy for one world off a network
  so that it can be evaluated exactly with the planners of this package.

PyTorch is needed for this module only (``pip install -e ".[deep]"``).
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np

try:
    import torch
    from torch import nn
except ImportError as error:  # pragma: no cover
    raise ImportError("mdplab.dqn needs PyTorch: pip install torch") from error

from .gridworld import FREE, GOAL, OBSTACLE, PIT, GridWorld

__all__ = [
    "GridBatch",
    "QMapNetwork",
    "QDenseNetwork",
    "DQNResult",
    "train_dqn",
    "train_on_sweeps",
    "sample_sweep",
    "greedy_policy",
    "action_values",
    "map_planes",
]

_DROW = np.array([-1, 0, 1, 0])
_DCOL = np.array([0, 1, 0, -1])


def map_planes(layouts: np.ndarray) -> np.ndarray:
    """What the networks see of a world: three images, for obstacles, pits and the goal.

    ``layouts`` has shape (..., rows, cols).  The result has shape
    (..., 3, rows + 2, cols + 2): the map is drawn with a ring of obstacles
    around it, because running into the edge of the world and running into an
    obstacle are the same thing.
    """
    layouts = np.asarray(layouts)
    pad = [(0, 0)] * (layouts.ndim - 2) + [(1, 1), (1, 1)]
    padded = np.pad(layouts, pad, constant_values=OBSTACLE)
    return np.stack([padded == OBSTACLE, padded == PIT, padded == GOAL], axis=-3).astype(np.uint8)


class GridBatch:
    """``n`` grid worlds stepped in parallel, each with its own layout.

    Parameters
    ----------
    make_world : callable taking a ``numpy.random.Generator`` and returning a
        :class:`~mdplab.gridworld.GridWorld`.  Called whenever an environment
        needs a new world, so the agent keeps meeting layouts it has not seen.
        All worlds must have the same shape, rewards and slip probability.
    n : number of parallel environments.
    resample : draw a new world for every episode (otherwise all environments
        share the first world drawn, for good).
    """

    def __init__(self, make_world, n: int, rng: np.random.Generator, resample: bool = True) -> None:
        self.make_world, self.n, self.rng, self.resample = make_world, n, rng, resample
        first = self._draw()
        self.rows, self.cols = first.shape
        self.params = first  # rewards and slip probability, shared by all worlds
        self.layouts = np.empty((n, self.rows, self.cols), dtype=np.int8)
        self.row = np.zeros(n, dtype=np.int64)
        self.col = np.zeros(n, dtype=np.int64)
        self.steps = np.zeros(n, dtype=np.int64)
        for i in range(n):
            self._new_episode(i, first if (i == 0 or not resample) else None)

    def _draw(self) -> GridWorld:
        world = self.make_world(self.rng)
        while not world.free.any():  # a world with no free cell is of no use
            world = self.make_world(self.rng)
        return world

    def _new_episode(self, i: int, world: GridWorld | None = None) -> None:
        if world is None and self.resample:
            world = self._draw()
        if world is not None:
            self.layouts[i] = world.layout
        free_r, free_c = np.nonzero(self.layouts[i] == FREE)
        k = self.rng.integers(free_r.size)
        self.row[i], self.col[i] = free_r[k], free_c[k]
        self.steps[i] = 0

    def step(self, actions: np.ndarray):
        """Move every agent; returns (reward, terminal).  Finished worlds stay as they are until :meth:`restart`."""
        p, n = self.params.p_intended, self.n
        u = self.rng.random(n)
        slip = (1.0 - p) / 2.0
        direction = np.where(u < p, actions, np.where(u < p + slip, (actions - 1) % 4, (actions + 1) % 4))
        r2, c2 = self.row + _DROW[direction], self.col + _DCOL[direction]
        inside = (r2 >= 0) & (r2 < self.rows) & (c2 >= 0) & (c2 < self.cols)
        r2, c2 = np.clip(r2, 0, self.rows - 1), np.clip(c2, 0, self.cols - 1)
        idx = np.arange(n)
        ok = inside & (self.layouts[idx, r2, c2] != OBSTACLE)
        self.row, self.col = np.where(ok, r2, self.row), np.where(ok, c2, self.col)
        cell = self.layouts[idx, self.row, self.col]
        reward = np.where(
            cell == GOAL, self.params.goal_reward, np.where(cell == PIT, self.params.pit_reward, self.params.step_reward)
        )
        self.steps += 1
        return reward.astype(np.float32), cell != FREE

    def restart(self, mask: np.ndarray) -> None:
        """Start a new episode (from a random free cell) in the environments selected by ``mask``."""
        for i in np.flatnonzero(mask):
            self._new_episode(i)


def sample_sweep(layouts: np.ndarray, params: GridWorld, rng: np.random.Generator):
    """One sampled move from every cell of every map, for each of the four actions.

    This is the "visit every state-action pair once" protocol of the tabular
    learner in notebook 3, for a whole batch of maps at once.  Returns the
    landing cell (as a flat index ``row * cols + col``), the reward and whether
    the move ended the episode, each of shape (maps, rows, cols, 4).
    """
    n, rows, cols = layouts.shape
    p = params.p_intended
    slip = (1.0 - p) / 2.0
    u = rng.random((n, rows, cols, 4))
    a = np.arange(4)
    direction = np.where(u < p, a, np.where(u < p + slip, (a - 1) % 4, (a + 1) % 4))
    r0 = np.arange(rows)[None, :, None, None]
    c0 = np.arange(cols)[None, None, :, None]
    r2, c2 = r0 + _DROW[direction], c0 + _DCOL[direction]
    inside = (r2 >= 0) & (r2 < rows) & (c2 >= 0) & (c2 < cols)
    r2, c2 = np.clip(r2, 0, rows - 1), np.clip(c2, 0, cols - 1)
    b = np.arange(n)[:, None, None, None]
    ok = inside & (layouts[b, r2, c2] != OBSTACLE)
    r2, c2 = np.where(ok, r2, r0), np.where(ok, c2, c0)
    cell = layouts[b, r2, c2]
    reward = np.where(cell == GOAL, params.goal_reward, np.where(cell == PIT, params.pit_reward, params.step_reward))
    return r2 * cols + c2, reward.astype(np.float32), cell != FREE


class QMapNetwork(nn.Module):
    """Convolutions all the way: a map goes in, four action values *per cell* come out.

    The value of an action for an agent standing on a cell is read off the
    output at that cell.  Nothing in the network depends on the size of the
    map, so it can be applied to worlds of any shape.  Each layer lets
    information travel one cell further, so a cell's values depend on the map
    within ``depth`` cells of it.

    With ``tied=True`` all layers after the first share one set of weights and
    are shown the map again each time.  The network is then a single local
    update rule applied ``depth`` times, which is the form value iteration has;
    and the number of applications can be changed when the network is used
    (``q_map(planes, depth=...)``).
    """

    def __init__(self, channels: int = 32, depth: int = 6, tied: bool = False) -> None:
        super().__init__()
        self.depth, self.tied = depth, tied
        self.first = nn.Conv2d(3, channels, 3, padding=1)
        if tied:
            self.step = nn.Conv2d(channels + 3, channels, 3, padding=1)
        else:
            self.steps = nn.ModuleList(nn.Conv2d(channels, channels, 3, padding=1) for _ in range(depth - 1))
        self.last = nn.Conv2d(channels, 4, 1)

    def q_map(self, planes: torch.Tensor, depth: int | None = None) -> torch.Tensor:
        """Action values for every cell, shape (batch, 4, rows + 2, cols + 2)."""
        h = torch.relu(self.first(planes))
        if self.tied:
            for _ in range((depth or self.depth) - 1):
                h = torch.relu(self.step(torch.cat([h, planes], dim=1)))
        else:
            for layer in self.steps:
                h = torch.relu(layer(h))
        return self.last(h)

    def forward(self, planes: torch.Tensor, row: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
        return self.q_map(planes)[torch.arange(planes.shape[0]), :, row + 1, col + 1]


class QDenseNetwork(nn.Module):
    """The agent's position as a fourth image, a few convolutions, then dense layers to four action values.

    The dense layers tie the network to one map size.
    """

    def __init__(self, rows: int, cols: int, channels: int = 32, hidden: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
        )  # fmt: skip
        n_features = channels * (rows + 2) * (cols + 2)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(n_features, hidden), nn.ReLU(), nn.Linear(hidden, 4))

    def forward(self, planes: torch.Tensor, row: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
        agent = torch.zeros_like(planes[:, :1])
        agent[torch.arange(planes.shape[0]), 0, row + 1, col + 1] = 1.0
        return self.head(self.features(torch.cat([planes, agent], dim=1)))


@torch.no_grad()
def action_values(net: nn.Module, world: GridWorld, depth: int | None = None) -> np.ndarray:
    """The network's action values in every cell of a world, shape (S, 4), in units of the goal reward.

    ``depth`` overrides the number of layers applied by a weight-tied :class:`QMapNetwork`.
    """
    planes = torch.from_numpy(map_planes(world.layout).astype(np.float32))
    rows, cols = world.shape
    if isinstance(net, QMapNetwork):
        q = net.q_map(planes[None], depth)[0, :, 1:-1, 1:-1]
        return q.permute(1, 2, 0).reshape(-1, 4).numpy()
    row, col = np.divmod(np.arange(rows * cols), cols)
    return net(planes.expand(rows * cols, -1, -1, -1), torch.from_numpy(row), torch.from_numpy(col)).numpy()


def greedy_policy(net: nn.Module, world: GridWorld, depth: int | None = None) -> np.ndarray:
    """The network's greedy action in every cell of a world, as a policy array of shape (S,)."""
    return action_values(net, world, depth).argmax(axis=1)


@dataclass
class DQNResult:
    """A trained network, its learning curve (``steps``, ``scores``) and the training loss per update."""

    net: nn.Module
    steps: list = field(default_factory=list)
    scores: list = field(default_factory=list)
    losses: list = field(default_factory=list)
    updates: int = 0
    seconds: float = 0.0


def train_dqn(
    make_world,
    n_steps: int,
    *,
    net: nn.Module | None = None,
    resample: bool = True,
    n_envs: int = 32,
    batch_size: int = 128,
    buffer_size: int = 100_000,
    gamma: float | None = None,
    learning_rate: float = 5e-4,
    lr_final: float = 0.1,
    epsilon: tuple[float, float, float] = (1.0, 0.1, 0.6),
    target_sync: int = 100,
    updates_per_step: int = 1,
    warmup: int = 2_000,
    max_episode_steps: int = 100,
    seed: int = 0,
    evaluate=None,
    eval_every: int = 20_000,
) -> DQNResult:
    """Train a network with DQN.

    Parameters
    ----------
    make_world : world generator, see :class:`GridBatch`.
    n_steps : total number of moves, summed over the parallel worlds.
    net : the network to train (default: a new :class:`QMapNetwork`).
    resample : new random world for every episode (``False``: one fixed world).
    n_envs : worlds stepped side by side; one round of moves is followed by
        ``updates_per_step`` gradient steps on ``batch_size`` remembered moves.
    gamma : discount used in the learning targets (default: the worlds' own).
    learning_rate, lr_final : Adam's step size, and the fraction of it that is
        left at the end of training (it shrinks linearly; 1.0 keeps it constant).
    epsilon : (start, end, fraction of training over which it falls linearly).
    target_sync : gradient steps between copies of the network being trained
        into the target network.
    max_episode_steps : episodes are cut off after this many moves (the last
        move is still learned from as an ordinary, non-final move).
    seed : seeds everything random inside this function (exploration, slips,
        replay, and the initial weights when ``net`` is not given).  A network
        passed in was initialised before that: call ``torch.manual_seed``
        before creating it if the whole run is to be repeatable.
    evaluate : optional callable ``net -> anything`` run every ``eval_every``
        moves; what it returns is collected as the learning curve.

    Rewards are divided by the goal reward before learning, so that action
    values are of order one.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = GridBatch(make_world, n_envs, rng, resample=resample)
    scale = float(abs(env.params.goal_reward))
    gamma = env.params.gamma if gamma is None else gamma
    if net is None:
        net = QMapNetwork()
    target = copy.deepcopy(net)
    optimiser = torch.optim.Adam(net.parameters(), lr=learning_rate)
    share_forward = isinstance(net, QMapNetwork)  # one pass gives the values at s and at s'

    buf_planes = np.zeros((buffer_size, 3, env.rows + 2, env.cols + 2), dtype=np.uint8)
    buf_int = np.zeros((buffer_size, 5), dtype=np.int64)  # row, col, action, next row, next col
    buf_reward = np.zeros(buffer_size, dtype=np.float32)
    buf_done = np.zeros(buffer_size, dtype=np.float32)
    size = cursor = 0

    result = DQNResult(net=net)
    eps_start, eps_end, eps_fraction = epsilon
    start = time.perf_counter()
    next_eval = eval_every
    for step in range(0, n_steps, n_envs):
        eps = eps_start + (eps_end - eps_start) * min(1.0, step / (eps_fraction * n_steps))
        for group in optimiser.param_groups:
            group["lr"] = learning_rate * (1.0 + (lr_final - 1.0) * step / n_steps)

        planes = map_planes(env.layouts)
        row, col = env.row.copy(), env.col.copy()
        with torch.no_grad():
            q = net(torch.from_numpy(planes.astype(np.float32)), torch.from_numpy(row), torch.from_numpy(col))
        actions = np.where(rng.random(n_envs) < eps, rng.integers(0, 4, size=n_envs), q.argmax(dim=1).numpy())
        reward, done = env.step(actions)

        idx = (cursor + np.arange(n_envs)) % buffer_size
        buf_planes[idx] = planes
        buf_int[idx] = np.stack([row, col, actions, env.row, env.col], axis=1)
        buf_reward[idx], buf_done[idx] = reward / scale, done
        cursor = (cursor + n_envs) % buffer_size
        size = min(size + n_envs, buffer_size)
        env.restart(done | (env.steps >= max_episode_steps))

        if size >= warmup:
            for _ in range(updates_per_step):
                batch = rng.integers(0, size, size=batch_size)
                p = torch.from_numpy(buf_planes[batch].astype(np.float32))
                r1, c1, a, r2, c2 = torch.from_numpy(buf_int[batch]).unbind(dim=1)
                r, d = torch.from_numpy(buf_reward[batch]), torch.from_numpy(buf_done[batch])
                rows_idx = torch.arange(batch_size)
                if share_forward:
                    q_map = net.q_map(p)
                    q_now = q_map[rows_idx, a, r1 + 1, c1 + 1]
                    best = q_map[rows_idx, :, r2 + 1, c2 + 1].detach().argmax(dim=1)
                else:
                    q_now = net(p, r1, c1)[rows_idx, a]
                    with torch.no_grad():
                        best = net(p, r2, c2).argmax(dim=1)
                with torch.no_grad():  # double DQN: the network being trained picks the next action,
                    bootstrap = target(p, r2, c2)[rows_idx, best]  # the target network says what it is worth
                    y = r + gamma * (1.0 - d) * bootstrap
                loss = nn.functional.smooth_l1_loss(q_now, y)
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                optimiser.step()
                result.updates += 1
                result.losses.append(loss.item())
                if result.updates % target_sync == 0:
                    target.load_state_dict(net.state_dict())

        if evaluate is not None and step + n_envs >= next_eval:
            result.steps.append(step + n_envs)
            result.scores.append(evaluate(net))
            next_eval += eval_every
    result.seconds = time.perf_counter() - start
    return result


def train_on_sweeps(
    make_world,
    n_updates: int,
    *,
    net: QMapNetwork | None = None,
    resample: bool = True,
    maps_per_update: int = 32,
    gamma: float | None = None,
    learning_rate: float = 1e-3,
    lr_final: float = 0.05,
    target_sync: int = 50,
    seed: int = 0,
    evaluate=None,
    eval_every: int = 200,
) -> DQNResult:
    """Deep Q-learning from uniform sweeps instead of episodes.

    Every update draws ``maps_per_update`` worlds, samples one move from every
    free cell for each action (:func:`sample_sweep`), and takes one gradient
    step on all of those moves at once.  A :class:`QMapNetwork` makes this
    cheap, because one pass over a map gives the values of every cell.  It is
    the neural counterpart of :func:`mdplab.learning.synchronous_q_learning`:
    still learning from sampled moves only, but with a simulator that can be
    asked about any cell.  Since each batch of experience is fresh and
    uncorrelated, no replay memory is needed; the target network and the
    double-Q target are the same as in :func:`train_dqn`, and so is the meaning
    of the parameters they share.

    ``result.steps`` counts updates.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    first = make_world(rng)
    scale = float(abs(first.goal_reward))
    gamma = first.gamma if gamma is None else gamma
    rows, cols = first.shape
    if net is None:
        net = QMapNetwork()
    target = copy.deepcopy(net)
    optimiser = torch.optim.Adam(net.parameters(), lr=learning_rate)
    n = maps_per_update
    b = torch.arange(n)[:, None, None, None]
    fixed = np.stack([first.layout] * n)

    result = DQNResult(net=net)
    start = time.perf_counter()
    for update in range(1, n_updates + 1):
        for group in optimiser.param_groups:
            group["lr"] = learning_rate * (1.0 + (lr_final - 1.0) * (update - 1) / n_updates)
        layouts = np.stack([make_world(rng).layout for _ in range(n)]) if resample else fixed
        landing, reward, done = sample_sweep(layouts, first, rng)
        landing = torch.from_numpy(landing)
        planes = torch.from_numpy(map_planes(layouts).astype(np.float32))
        free = torch.from_numpy(layouts == FREE)

        q = net.q_map(planes)[:, :, 1:-1, 1:-1].permute(0, 2, 3, 1)  # (maps, rows, cols, actions)
        with torch.no_grad():
            q_next = q.detach().reshape(n, rows * cols, 4)[b, landing]  # (maps, rows, cols, actions, next actions)
            q_target = target.q_map(planes)[:, :, 1:-1, 1:-1].permute(0, 2, 3, 1)
            q_next_target = q_target.reshape(n, rows * cols, 4)[b, landing]
            bootstrap = q_next_target.gather(-1, q_next.argmax(dim=-1, keepdim=True)).squeeze(-1)
            y = torch.from_numpy(reward) / scale + gamma * torch.from_numpy(~done) * bootstrap
        loss = nn.functional.smooth_l1_loss(q[free], y[free])
        optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        optimiser.step()
        result.updates += 1
        result.losses.append(loss.item())
        if update % target_sync == 0:
            target.load_state_dict(net.state_dict())
        if evaluate is not None and update % eval_every == 0:
            result.steps.append(update)
            result.scores.append(evaluate(net))
    result.seconds = time.perf_counter() - start
    return result
