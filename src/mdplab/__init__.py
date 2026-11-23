"""mdplab: a small laboratory for finite Markov decision processes.

Built to revisit a 2014 course project on two MDPs, a random grid world and a
"crazy elevator", and to take its experiments further.
"""

from .benchmark import WorldSet
from .core import MDP, Sampler, greedy, random_argmax
from .elevator import REPORT_POLICY_50, crazy_elevator, elevator_transitions
from .evaluation import (
    absorption,
    agreement,
    evaluate_policy,
    hitting_probability,
    iterative_policy_evaluation,
    monte_carlo,
    optimal_actions,
    uniform_random_policy,
    value_gap,
)
from .gridworld import GridWorld
from .learning import (
    LearningResult,
    PolicyScorer,
    constant,
    empirical_model,
    exponential_decay,
    harmonic,
    linear_decay,
    polynomial,
    synchronous_q_learning,
    td_control,
    visit_decay,
)
from .planning import (
    Solution,
    gauss_seidel_value_iteration,
    linear_programming,
    modified_policy_iteration,
    policy_iteration,
    prioritized_sweeping,
    value_iteration,
)
from .random_mdp import garnet

__version__ = "0.1.0"

__all__ = [
    "MDP",
    "Sampler",
    "greedy",
    "random_argmax",
    "GridWorld",
    "WorldSet",
    "crazy_elevator",
    "elevator_transitions",
    "REPORT_POLICY_50",
    "Solution",
    "value_iteration",
    "policy_iteration",
    "modified_policy_iteration",
    "gauss_seidel_value_iteration",
    "prioritized_sweeping",
    "linear_programming",
    "evaluate_policy",
    "iterative_policy_evaluation",
    "absorption",
    "hitting_probability",
    "monte_carlo",
    "optimal_actions",
    "agreement",
    "value_gap",
    "uniform_random_policy",
    "PolicyScorer",
    "LearningResult",
    "synchronous_q_learning",
    "td_control",
    "empirical_model",
    "constant",
    "harmonic",
    "polynomial",
    "linear_decay",
    "exponential_decay",
    "visit_decay",
    "garnet",
]
