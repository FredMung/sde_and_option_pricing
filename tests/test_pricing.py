import numpy as np
import pytest

from option_pricing_app import (
    ExerciseStyle,
    HestonInputs,
    MarketInputs,
    PutContract,
    SimulationConfig,
    price_put,
)
from option_pricing_app.service import (
    black_scholes_put_price,
    estimate_exercise_boundary,
)
from option_pricing_app.stats import (
    GBMPriceSimulator,
    HestonPriceSimulator,
    LSMCPriceSimulator,
)


def test_gbm_seed_reproduces_paths():
    simulator = GBMPriceSimulator()
    np.random.seed(42)
    first = simulator.generate_price_paths(100, 1, 0.05, 0.2, 12, 8)
    np.random.seed(42)
    second = simulator.generate_price_paths(100, 1, 0.05, 0.2, 12, 8)
    assert first.shape == (9, 12)
    assert np.all(first[0] == 100)
    assert np.all(first > 0)
    np.testing.assert_array_equal(first, second)


def test_heston_seed_reproduces_price_and_variance_paths():
    simulator = HestonPriceSimulator()
    arguments = (100, 1, 0.05, 2.0, 0.04, 0.3, -0.7, 0.04, 100, 20)
    np.random.seed(42)
    first_prices, first_variances = simulator.generate_price_paths(*arguments)
    np.random.seed(42)
    second_prices, second_variances = simulator.generate_price_paths(*arguments)
    np.testing.assert_array_equal(first_prices, second_prices)
    np.testing.assert_array_equal(first_variances, second_variances)
    assert first_prices.shape == (21, 100)
    assert first_variances.shape == (21, 100)
    assert np.all(first_prices > 0)
    assert np.all(first_variances >= 0)


def test_heston_european_pricing_helper_returns_price_and_error():
    np.random.seed(42)
    price, error = HestonPriceSimulator().calculate_option_price(
        100, 1, 0.05, 2.0, 0.04, 0.3, -0.7, 0.04, 100, 500, 20
    )
    assert price >= 0
    assert error >= 0


@pytest.mark.parametrize("style", list(ExerciseStyle))
def test_service_prices_both_exercise_styles(style):
    result = price_put(
        PutContract(100, 1),
        MarketInputs(100, 0.2, 0.05),
        SimulationConfig(1_000, 25, 42, 20),
        style,
    )
    assert result.price >= 0
    assert result.standard_error >= 0
    assert result.confidence_interval[0] >= 0
    assert result.displayed_paths.shape == (26, 20)
    assert result.path_quantile_05.shape == (26,)
    assert result.path_quantile_95.shape == (26,)
    assert result.risk_neutral_expected_path.shape == (26,)
    assert np.all(result.path_quantile_05 <= result.path_quantile_95)
    assert result.risk_neutral_expected_path[-1] == pytest.approx(100 * np.exp(0.05))
    assert result.terminal_prices.shape == (1_000,)
    assert result.discounted_realised_cash_flows.shape == (1_000,)
    assert np.mean(result.discounted_realised_cash_flows) == pytest.approx(result.price)
    assert result.european_mc_price >= 0
    assert result.european_exact_price == pytest.approx(5.573526, abs=1e-6)
    assert result.convergence_path_counts.shape == (1_000,)
    assert result.convergence_estimates[-1] == pytest.approx(result.price)
    assert result.convergence_lower[-1] == pytest.approx(result.confidence_interval[0])
    assert result.convergence_upper[-1] == pytest.approx(result.confidence_interval[1])
    if style is ExerciseStyle.AMERICAN:
        assert result.exercise_percentages.shape == (26,)
        assert 0 <= result.exercise_percentages.sum() <= 100
        assert result.continuation_diagnostics
        assert all(
            diagnostic.price_grid.shape == (300,)
            for diagnostic in result.continuation_diagnostics
        )
    else:
        assert result.exercise_percentages is None
        assert result.continuation_diagnostics is None


def test_lsmc_handles_no_in_the_money_paths():
    paths = np.full((4, 10), 120.0)
    paths[0] = 100.0
    simulator = LSMCPriceSimulator()
    price, error = simulator.backward_induction(paths, 90, 0.05, 1 / 3)
    assert price == 0
    assert error == 0
    assert np.all(simulator.exercise_steps == -1)


def test_lsmc_builds_quadratic_price_variance_cross_terms():
    prices = np.array([2.0, 3.0])
    variances = np.array([0.1, 0.2])
    matrix = LSMCPriceSimulator.build_design_matrix(
        prices,
        ("1", "S", "S²", "v", "v²", "S·v", "S²·v", "S·v²", "S²·v²"),
        variances,
    )
    np.testing.assert_allclose(
        matrix[0],
        [1.0, 2.0, 4.0, 0.1, 0.01, 0.2, 0.4, 0.02, 0.04],
    )


def test_stored_continuation_model_reproduces_fitted_values():
    simulator = LSMCPriceSimulator()
    prices = np.array([80.0, 85.0, 90.0, 95.0])
    future_cash_flows = np.array([19.0, 14.0, 9.0, 4.0])
    fitted = simulator.estimate_continuation_value(
        prices, future_cash_flows, 0.05, 0.25
    )
    model = simulator.fit_continuation_model(
        prices, future_cash_flows, 0.05, 0.25, timestep=2
    )
    np.testing.assert_allclose(model.predict(prices), fitted)
    assert model.timestep == 2
    assert model.basis_terms == ("1", "S", "S²")


def test_boundary_uses_low_price_exercise_to_high_price_continuation_crossing():
    prices = np.linspace(0.0, 10.0, 1_001)
    payoff = 10.0 - prices
    difference = -(prices - 2.0) * (prices - 4.0) * (prices - 6.0)
    continuation = payoff - difference
    boundary = estimate_exercise_boundary(prices, payoff, continuation, 1_000, 10)
    assert boundary == pytest.approx(2.0, abs=0.02)


def test_boundary_is_missing_when_crossing_is_unsupported_or_sample_is_small():
    prices = np.linspace(60.0, 99.0, 300)
    payoff = 100.0 - prices
    assert np.isnan(
        estimate_exercise_boundary(prices, payoff, np.zeros_like(prices), 1_000, 10)
    )
    assert np.isnan(
        estimate_exercise_boundary(prices, payoff, np.full_like(prices, 10.0), 5, 10)
    )


@pytest.mark.parametrize("style", list(ExerciseStyle))
def test_service_prices_heston_and_returns_variance_diagnostics(style):
    result = price_put(
        PutContract(100, 1),
        MarketInputs(100, 0.2, 0.05),
        SimulationConfig(
            500,
            20,
            42,
            20,
            ("1", "S", "S²", "v", "v²", "S·v"),
        ),
        style,
        "Heston",
        HestonInputs(),
    )
    assert result.price >= 0
    assert result.european_exact_price is None
    assert result.displayed_variance_paths.shape == (21, 20)
    assert result.variance_quantile_05.shape == (21,)
    assert result.variance_quantile_95.shape == (21,)
    assert result.expected_variance_path.shape == (21,)
    assert result.long_run_variance == pytest.approx(0.04)
    if style is ExerciseStyle.AMERICAN:
        assert result.continuation_diagnostics
        assert all(
            diagnostic.representative_variance is not None
            for diagnostic in result.continuation_diagnostics
        )


def test_gbm_rejects_variance_basis_terms():
    with pytest.raises(ValueError, match="only available with the Heston"):
        price_put(
            PutContract(100, 1),
            MarketInputs(100, 0.2, 0.05),
            SimulationConfig(500, 10, 42, 10, ("1", "S", "v")),
            ExerciseStyle.AMERICAN,
            "GBM",
        )


def test_unseeded_european_trace_matches_displayed_simulation_price():
    result = price_put(
        PutContract(100, 1),
        MarketInputs(100, 0.2, 0.05),
        SimulationConfig(500, 10, None, 10),
        ExerciseStyle.EUROPEAN,
    )
    assert result.convergence_estimates[-1] == pytest.approx(result.price)


def test_black_scholes_put_matches_known_value():
    price = black_scholes_put_price(
        PutContract(100, 1), MarketInputs(100, 0.2, 0.05)
    )
    assert price == pytest.approx(5.573526, abs=1e-6)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: PutContract(0, 1),
        lambda: PutContract(100, 0),
        lambda: MarketInputs(0, 0.2, 0.05),
        lambda: MarketInputs(100, 0, 0.05),
        lambda: SimulationConfig(99, 100),
        lambda: SimulationConfig(1_000, 1_001),
    ],
)
def test_invalid_inputs_are_rejected(factory):
    with pytest.raises(ValueError):
        factory()
