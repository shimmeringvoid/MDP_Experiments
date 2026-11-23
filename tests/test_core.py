import numpy as np
import pytest
import scipy.sparse as sp

from mdplab import MDP, GridWorld, crazy_elevator, garnet


def two_state_mdp(sparse=False):
    # state 0: action 0 stays (p=.5) or goes to 1; action 1 goes to 2 (terminal)
    P = np.array(
        [
            [0.5, 0.5, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.3, 0.3, 0.4],  # terminal rows are overwritten
            [0.3, 0.3, 0.4],
        ]
    )
    if sparse:
        P = sp.csr_matrix(P)
    return MDP(P=P, r_next=np.array([1.0, 2.0, 10.0]), gamma=0.9, terminal=np.array([False, False, True]))


@pytest.mark.parametrize("sparse", [False, True])
def test_terminal_states_become_zero_reward_self_loops(sparse):
    mdp = two_state_mdp(sparse)
    mdp.validate()
    P = mdp.P.toarray() if sparse else mdp.P
    assert np.allclose(P[4:], [[0, 0, 1], [0, 0, 1]])
    assert np.allclose(mdp.R[2], 0.0)
    # expected arrival rewards elsewhere
    assert np.allclose(mdp.R[0], [0.5 * 1 + 0.5 * 2, 10.0])
    assert np.allclose(mdp.R[1], [1.0, 2.0])


def test_q_values_match_brute_force(small_mdp, rng):
    mdp = small_mdp
    S, A = mdp.n_states, mdp.n_actions
    P = (mdp.P.toarray() if mdp.is_sparse else mdp.P).reshape(S, A, S)
    V = rng.normal(size=S)
    expected = np.empty((S, A))
    for s in range(S):
        for a in range(A):
            r = mdp.r_sa[s, a] + P[s, a] @ mdp.r_next
            expected[s, a] = (0.0 if mdp.terminal[s] else r) + mdp.gamma * P[s, a] @ V
    assert np.allclose(mdp.q_values(V), expected)


def test_policy_model_deterministic_and_stochastic_agree(small_mdp, rng):
    mdp = small_mdp
    S, A = mdp.n_states, mdp.n_actions
    policy = rng.integers(0, A, size=S)
    one_hot = np.eye(A)[policy]
    P1, r1 = mdp.policy_model(policy)
    P2, r2 = mdp.policy_model(one_hot)

    def dense(M):
        return M.toarray() if sp.issparse(M) else np.asarray(M)

    assert np.allclose(dense(P1), dense(P2))
    assert np.allclose(r1, r2)


def test_rejects_non_stochastic_rows():
    P = np.array([[0.5, 0.4], [0.0, 1.0]])
    mdp = MDP(P=P, r_next=np.zeros(2), gamma=0.9, terminal=np.zeros(2, dtype=bool))
    with pytest.raises(ValueError):
        mdp.validate()


def test_sampler_frequencies_match_probabilities(rng):
    for mdp in (crazy_elevator(6), GridWorld.random(4, 4, seed=1).mdp):
        S, A = mdp.n_states, mdp.n_actions
        P = (mdp.P.toarray() if mdp.is_sparse else mdp.P)
        n = 4000
        counts = np.zeros((S * A, S))
        rows = np.arange(S * A)
        for draw in mdp.sampler.sweep(rng, n):
            counts[rows, draw] += 1
        assert np.abs(counts / n - P).max() < 0.04
        # batched single steps agree with the same table
        s = np.repeat(np.flatnonzero(mdp.active)[:1], 5000)
        a = np.zeros(5000, dtype=int)
        s2, r = mdp.sampler.step(s, a, rng)
        freq = np.bincount(s2, minlength=S) / 5000
        assert np.abs(freq - P[s[0] * A]).max() < 0.04
        assert np.allclose(r, mdp.r_sa[s, a] + mdp.r_next[s2])


def test_garnet_shapes_and_branching():
    mdp = garnet(40, 6, 5, seed=0)
    mdp.validate()
    P = mdp.P.toarray() if mdp.is_sparse else mdp.P
    assert P.shape == (240, 40)
    assert ((P > 0).sum(axis=1) == 5).all()
