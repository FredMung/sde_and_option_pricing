"""App-facing orchestration around the migrated pricing algorithms."""

from math import erf, exp, log, sqrt

import numpy as np

from option_pricing_app.domain import (
    PRICE_BASIS_TERMS,
    ExerciseStyle,
    HestonInputs,
    MarketInputs,
    PricingResult,
    PutContract,
    SimulationConfig,
)
from option_pricing_app.stats import (
    GBMPriceSimulator,
    HestonPriceSimulator,
    LSMCPriceSimulator,
)


def create_asset_path_simulator(model: str):
    if model == "GBM":
        return GBMPriceSimulator()
    if model == "Heston":
        return HestonPriceSimulator()
    raise ValueError(f"Unsupported asset model: {model}")


def price_put(
    contract: PutContract,
    market: MarketInputs,
    config: SimulationConfig,
    exercise_style: ExerciseStyle,
    model: str = "GBM",
    heston_inputs: HestonInputs | None = None,
) -> PricingResult:
    """Run the original algorithms and prepare their output for the dashboard."""
    simulator = create_asset_path_simulator(model)
    if config.seed is not None:
        np.random.seed(config.seed)
    if model == "GBM":
        unsupported_terms = set(config.basis_terms) - set(PRICE_BASIS_TERMS)
        if unsupported_terms:
            raise ValueError("Variance basis terms are only available with the Heston model")
        paths = simulator.generate_price_paths(
            market.spot,
            contract.maturity,
            market.risk_free_rate,
            market.volatility,
            config.n_paths,
            config.n_steps,
        )
        variance_paths = None
    else:
        if heston_inputs is None:
            raise ValueError("Heston parameters are required for the Heston model")
        paths, variance_paths = simulator.generate_price_paths(
            market.spot,
            contract.maturity,
            market.risk_free_rate,
            heston_inputs.mean_reversion_speed,
            heston_inputs.long_run_variance,
            heston_inputs.volatility_of_variance,
            heston_inputs.correlation,
            heston_inputs.initial_variance,
            config.n_paths,
            config.n_steps,
        )
    european_pathwise_values = np.exp(
        -market.risk_free_rate * contract.maturity
    ) * np.maximum(contract.strike - paths[-1], 0.0)
    european_mc_price = float(np.mean(european_pathwise_values))
    european_mc_standard_error = float(
        np.std(european_pathwise_values) / np.sqrt(config.n_paths)
    )
    european_exact_price = (
        black_scholes_put_price(contract, market) if model == "GBM" else None
    )

    if exercise_style is ExerciseStyle.EUROPEAN:
        pathwise_values = european_pathwise_values
        price = european_mc_price
        standard_error = european_mc_standard_error
        exercise_percentages = None
        method = "Terminal-payoff Monte Carlo"
    elif exercise_style is ExerciseStyle.AMERICAN:
        lsmc = LSMCPriceSimulator(simulator)
        price, standard_error = lsmc.backward_induction(
            paths,
            contract.strike,
            market.risk_free_rate,
            contract.maturity / config.n_steps,
            config.basis_terms,
            variance_paths,
        )
        pathwise_values = lsmc.option_value_paths[0].copy()
        exercise_counts = np.bincount(
            lsmc.exercise_steps[lsmc.exercise_steps >= 0], minlength=config.n_steps + 1
        )
        exercise_percentages = 100.0 * exercise_counts / config.n_paths
        method = "Least-squares Monte Carlo"
    else:
        raise ValueError(f"Unsupported exercise style: {exercise_style}")

    low = max(0.0, price - 1.96 * standard_error)
    high = price + 1.96 * standard_error
    shown = min(config.max_display_paths, config.n_paths)
    terminal_prices = paths[-1].copy()
    time_grid = np.linspace(0.0, contract.maturity, config.n_steps + 1)
    path_quantile_05, path_quantile_95 = np.quantile(paths, [0.05, 0.95], axis=1)
    if variance_paths is None:
        displayed_variance_paths = None
        variance_quantile_05 = None
        variance_quantile_95 = None
        expected_variance_path = None
        long_run_variance = None
    else:
        assert heston_inputs is not None
        displayed_variance_paths = variance_paths[:, :shown].copy()
        variance_quantile_05, variance_quantile_95 = np.quantile(
            variance_paths, [0.05, 0.95], axis=1
        )
        expected_variance_path = heston_inputs.long_run_variance + (
            heston_inputs.initial_variance - heston_inputs.long_run_variance
        ) * np.exp(-heston_inputs.mean_reversion_speed * time_grid)
        long_run_variance = heston_inputs.long_run_variance
    path_counts, estimates, convergence_low, convergence_high = _convergence_trace(
        pathwise_values
    )
    return PricingResult(
        price=price,
        standard_error=standard_error,
        confidence_interval=(low, high),
        time_grid=time_grid,
        displayed_paths=paths[:, :shown].copy(),
        path_quantile_05=path_quantile_05,
        path_quantile_95=path_quantile_95,
        risk_neutral_expected_path=market.spot * np.exp(market.risk_free_rate * time_grid),
        displayed_variance_paths=displayed_variance_paths,
        variance_quantile_05=variance_quantile_05,
        variance_quantile_95=variance_quantile_95,
        expected_variance_path=expected_variance_path,
        long_run_variance=long_run_variance,
        terminal_prices=terminal_prices,
        terminal_payoffs=np.maximum(contract.strike - terminal_prices, 0.0),
        discounted_realised_cash_flows=pathwise_values.copy(),
        exercise_style=exercise_style,
        model_name=model,
        pricing_method=method,
        european_mc_price=european_mc_price,
        european_mc_standard_error=european_mc_standard_error,
        european_exact_price=european_exact_price,
        convergence_path_counts=path_counts,
        convergence_estimates=estimates,
        convergence_lower=convergence_low,
        convergence_upper=convergence_high,
        exercise_percentages=exercise_percentages,
    )


def black_scholes_put_price(contract: PutContract, market: MarketInputs) -> float:
    """Return the no-dividend European put price under Black–Scholes assumptions."""
    volatility_time = market.volatility * sqrt(contract.maturity)
    d1 = (
        log(market.spot / contract.strike)
        + (market.risk_free_rate + 0.5 * market.volatility**2) * contract.maturity
    ) / volatility_time
    d2 = d1 - volatility_time
    discounted_strike = contract.strike * exp(-market.risk_free_rate * contract.maturity)
    return float(discounted_strike * _normal_cdf(-d2) - market.spot * _normal_cdf(-d1))


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def _convergence_trace(
    pathwise_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return cumulative estimates and normal-approximation Monte Carlo intervals."""
    counts = np.arange(1, pathwise_values.size + 1)
    cumulative_sum = np.cumsum(pathwise_values, dtype=float)
    estimates = cumulative_sum / counts
    second_moments = np.cumsum(np.square(pathwise_values), dtype=float) / counts
    variances = np.maximum(second_moments - np.square(estimates), 0.0)
    standard_errors = np.sqrt(variances / counts)
    margin = 1.96 * standard_errors
    return counts, estimates, np.maximum(0.0, estimates - margin), estimates + margin
