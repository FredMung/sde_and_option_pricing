"""App-facing orchestration around the migrated pricing algorithms."""

from typing import Protocol, runtime_checkable

import numpy as np

from option_pricing_app.domain import (
    ExerciseStyle,
    MarketInputs,
    PricingResult,
    PutContract,
    SimulationConfig,
)
from option_pricing_app.stats import GBMPriceSimulator, LSMCPriceSimulator


@runtime_checkable
class PricePathSimulator(Protocol):
    """Extension point for a future single-asset process such as Heston."""

    def generate_price_paths(self, S0, T, r, sigma, n_paths, n_steps): ...


def create_path_simulator(model: str) -> PricePathSimulator:
    if model == "GBM":
        return GBMPriceSimulator()
    raise ValueError(f"Unsupported asset model: {model}")


def price_put(
    contract: PutContract,
    market: MarketInputs,
    config: SimulationConfig,
    exercise_style: ExerciseStyle,
    model: str = "GBM",
) -> PricingResult:
    """Run the original algorithms and prepare their output for the dashboard."""
    simulator = create_path_simulator(model)
    if config.seed is not None:
        np.random.seed(config.seed)
    paths = simulator.generate_price_paths(
        market.spot,
        contract.maturity,
        market.risk_free_rate,
        market.volatility,
        config.n_paths,
        config.n_steps,
    )

    if exercise_style is ExerciseStyle.EUROPEAN:
        # Resetting the seed makes calculate_option_price use the displayed paths.
        if config.seed is not None:
            np.random.seed(config.seed)
        price, standard_error = simulator.calculate_option_price(
            market.spot,
            contract.maturity,
            market.risk_free_rate,
            market.volatility,
            contract.strike,
            config.n_paths,
            config.n_steps,
        )
        method = "Terminal-payoff Monte Carlo"
    elif exercise_style is ExerciseStyle.AMERICAN:
        lsmc = LSMCPriceSimulator(simulator)
        price, standard_error = lsmc.backward_induction(
            paths,
            contract.strike,
            market.risk_free_rate,
            contract.maturity / config.n_steps,
        )
        method = "Least-squares Monte Carlo"
    else:
        raise ValueError(f"Unsupported exercise style: {exercise_style}")

    low = max(0.0, price - 1.96 * standard_error)
    high = price + 1.96 * standard_error
    shown = min(config.max_display_paths, config.n_paths)
    terminal_prices = paths[-1].copy()
    return PricingResult(
        price=price,
        standard_error=standard_error,
        confidence_interval=(low, high),
        time_grid=np.linspace(0.0, contract.maturity, config.n_steps + 1),
        displayed_paths=paths[:, :shown].copy(),
        terminal_prices=terminal_prices,
        terminal_payoffs=np.maximum(contract.strike - terminal_prices, 0.0),
        exercise_style=exercise_style,
        model_name=model,
        pricing_method=method,
    )
