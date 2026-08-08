import numpy as np
import pytest

from option_pricing_app import (
    ExerciseStyle,
    MarketInputs,
    PutContract,
    SimulationConfig,
    price_put,
)
from option_pricing_app.stats import GBMPriceSimulator, LSMCPriceSimulator


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
    assert result.terminal_prices.shape == (1_000,)


def test_lsmc_handles_no_in_the_money_paths():
    paths = np.full((4, 10), 120.0)
    paths[0] = 100.0
    price, error = LSMCPriceSimulator().backward_induction(paths, 90, 0.05, 1 / 3)
    assert price == 0
    assert error == 0


@pytest.mark.parametrize(
    "factory",
    [
        lambda: PutContract(0, 1),
        lambda: PutContract(100, 0),
        lambda: MarketInputs(0, 0.2, 0.05),
        lambda: MarketInputs(100, 0, 0.05),
        lambda: SimulationConfig(99, 100),
        lambda: SimulationConfig(1_000, 501),
    ],
)
def test_invalid_inputs_are_rejected(factory):
    with pytest.raises(ValueError):
        factory()
