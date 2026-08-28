from types import MappingProxyType

import numpy as np
import pytest

from option_pricing_app.domain import MarketInputs, PutContract, SimulationConfig
from option_pricing_app.lsmc_policy import (
    FittedLSMCPolicy,
    evaluate_independent_policy,
    fit_lsmc_policy,
    generate_paths,
)

CONTRACT = PutContract(100, 1)
MARKET = MarketInputs(100, 0.2, 0.05)


class _ConstantContinuation:
    """Duck-typed stand-in for ``ContinuationRegression``: a fixed continuation
    value everywhere, so exercise/continue decisions are exactly controllable."""

    def __init__(self, value: float):
        self.value = value

    def predict(self, prices, variances=None):
        return np.full(len(prices), self.value, dtype=float)


def _hand_built_policy() -> FittedLSMCPolicy:
    return FittedLSMCPolicy(
        continuation_models=MappingProxyType(
            {
                1: _ConstantContinuation(5.0),
                2: _ConstantContinuation(-100.0),
            }
        ),
        strike=100.0,
        maturity=1.0,
        risk_free_rate=0.05,
        dt=1.0 / 3.0,
        n_steps=3,
        basis_terms=("1",),
        exercise_immediately_at_zero=False,
        same_sample_price=0.0,
        same_sample_standard_error=0.0,
    )


def test_frozen_policy_evaluates_at_the_first_qualifying_date_only():
    policy = _hand_built_policy()
    # Path 0 exercises at t=1 (ITM, payoff 10 > continuation 5). Path 1 is out of
    # the money at t=1 so it survives to t=2, where it exercises (ITM, payoff 5 >
    # continuation -100). Path 2 never has a fitted continuation decision favour
    # exercise until maturity. Path 3 is never in the money.
    price_paths = np.array(
        [
            [100.0, 100.0, 100.0, 100.0],
            [90.0, 100.5, 110.0, 110.0],
            [999.0, 95.0, 110.0, 110.0],
            [999.0, 999.0, 97.0, 110.0],
        ]
    )
    evaluation = evaluate_independent_policy(policy, price_paths)
    assert list(evaluation.exercise_steps) == [1, 2, 3, -1]


def test_at_most_once_exercise_even_when_a_later_date_would_also_qualify():
    policy = _hand_built_policy()
    price_paths = np.array(
        [
            [100.0, 100.0],
            [90.0, 100.5],
            [80.0, 95.0],  # path 0 already exercised at t=1; its t=2 price is ignored
            [70.0, 110.0],
        ]
    )
    evaluation = evaluate_independent_policy(policy, price_paths)
    # Path 0 exercises once, at t=1 -- not re-flagged at t=2 despite continuation_models[2]
    # being deeply negative there too (which would also favour exercise if re-evaluated).
    assert evaluation.exercise_steps[0] == 1


def test_payoffs_are_discounted_from_their_own_exercise_date():
    policy = _hand_built_policy()
    price_paths = np.array(
        [
            [100.0, 100.0, 100.0, 100.0],
            [90.0, 100.5, 110.0, 110.0],
            [999.0, 95.0, 110.0, 110.0],
            [999.0, 999.0, 97.0, 110.0],
        ]
    )
    evaluation = evaluate_independent_policy(policy, price_paths)
    expected = [
        10.0 * np.exp(-0.05 * (1.0 / 3.0) * 1),
        5.0 * np.exp(-0.05 * (1.0 / 3.0) * 2),
        3.0 * np.exp(-0.05 * (1.0 / 3.0) * 3),
        0.0,
    ]
    np.testing.assert_allclose(evaluation.discounted_payoffs, expected)
    assert evaluation.mean_estimate == pytest.approx(np.mean(expected))


def test_no_regression_at_a_date_means_the_policy_continues():
    """A date with no fitted continuation model (e.g. no in-the-money training
    paths there) is skipped, not treated as an automatic exercise or an error."""
    policy = FittedLSMCPolicy(
        continuation_models=MappingProxyType({1: _ConstantContinuation(-100.0)}),
        strike=100.0,
        maturity=1.0,
        risk_free_rate=0.05,
        dt=1.0 / 3.0,
        n_steps=3,
        basis_terms=("1",),
        exercise_immediately_at_zero=False,
        same_sample_price=0.0,
        same_sample_standard_error=0.0,
    )
    # No model at t=2, so even though this path is deep in the money there, nothing
    # forces an exercise decision -- it survives to maturity.
    price_paths = np.array([[100.0], [110.0], [50.0], [40.0]])
    evaluation = evaluate_independent_policy(policy, price_paths)
    assert evaluation.exercise_steps[0] == 3


def test_evaluation_never_refits_the_frozen_policy():
    training_config = SimulationConfig(300, 12, 11, 1)
    policy = fit_lsmc_policy(CONTRACT, MARKET, training_config, "GBM", None)
    original_coefficients = {
        step: model.coefficients.copy() for step, model in policy.continuation_models.items()
    }

    for seed in (222, 333):
        validation_config = SimulationConfig(300, 12, seed, 1)
        validation_paths, _ = generate_paths(CONTRACT, MARKET, validation_config, "GBM", None)
        evaluate_independent_policy(policy, validation_paths)

    for step, model in policy.continuation_models.items():
        np.testing.assert_array_equal(model.coefficients, original_coefficients[step])


def test_evaluations_on_different_validation_paths_differ():
    training_config = SimulationConfig(300, 12, 11, 1)
    policy = fit_lsmc_policy(CONTRACT, MARKET, training_config, "GBM", None)

    paths_a, _ = generate_paths(CONTRACT, MARKET, SimulationConfig(300, 12, 222, 1), "GBM", None)
    paths_b, _ = generate_paths(CONTRACT, MARKET, SimulationConfig(300, 12, 333, 1), "GBM", None)
    result_a = evaluate_independent_policy(policy, paths_a)
    result_b = evaluate_independent_policy(policy, paths_b)
    assert result_a.mean_estimate != pytest.approx(result_b.mean_estimate)


def test_evaluate_rejects_a_path_grid_mismatched_to_the_policy():
    policy = _hand_built_policy()
    wrong_shape_paths = np.full((5, 3), 100.0)
    with pytest.raises(ValueError, match="does not match"):
        evaluate_independent_policy(policy, wrong_shape_paths)


def test_fitted_policy_same_sample_price_matches_backward_induction():
    config = SimulationConfig(500, 20, 7, 1)
    policy = fit_lsmc_policy(CONTRACT, MARKET, config, "GBM", None)
    assert policy.same_sample_standard_error >= 0.0
    assert policy.n_steps == 20
    assert len(policy.continuation_models) > 0
