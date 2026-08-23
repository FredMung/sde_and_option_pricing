"""App-facing orchestration around the migrated pricing algorithms."""

from math import erf, exp, log, sqrt

import numpy as np

from option_pricing_app.domain import (
    PRICE_BASIS_TERMS,
    ExerciseStyle,
    HestonInputs,
    LSMCContinuationDiagnostic,
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

CONTINUATION_GRID_SIZE = 300
# The exercise-boundary crossing test is only evaluated over this central percentile
# range of each timestep's simulated in-the-money prices, not the full min/max.
# Diagnosed empirically on this codebase (not drawn from a paper): with a quadratic
# basis ("1", "S", "S**2"), payoff - continuation is itself a quadratic function of
# price, which can cross zero twice. In the deep-in-the-money tail -- thinly
# populated, so the regression is extrapolating rather than interpolating -- that
# curvature produces a razor-thin spurious second crossing sitting right at the edge
# of the sampled range, in addition to the genuine, well-supported crossing near the
# middle of the in-the-money range. The near-the-money end is not the problem: it
# consistently shows continuation cleanly dominating, as theory predicts. Trimming to
# the central 90% shrinks -- but does not fully eliminate -- this edge effect, since
# it is a property of the fitted curve's shape, not of any specific percentile cutoff.
BOUNDARY_GRID_PERCENTILES = (5.0, 95.0)


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

    lsmc = None
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
    continuation_diagnostics = (
        None
        if lsmc is None
        else _build_lsmc_continuation_diagnostics(
            lsmc,
            paths,
            variance_paths,
            contract,
            time_grid,
        )
    )
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
        continuation_diagnostics=continuation_diagnostics,
    )


def estimate_exercise_boundary(
    price_grid: np.ndarray,
    immediate_payoff: np.ndarray,
    continuation_value: np.ndarray,
    itm_path_count: int,
    minimum_required_paths: int,
) -> float:
    """Return one supported exercise-to-continuation crossing.

    A put boundary must move from exercise at lower prices to continuation at higher
    prices. The crossing is accepted only when exactly one such sign change occurs
    within the observed in-the-money price support. Multiple crossings are treated as
    an unclear regression result rather than resolved by an arbitrary selection rule.
    """
    prices = np.asarray(price_grid, dtype=float)
    payoff = np.asarray(immediate_payoff, dtype=float)
    continuation = np.asarray(continuation_value, dtype=float)
    # itm_path_count < minimum_required_paths: with too few in-the-money observations
    # relative to the number of regression coefficients, the fit is under-determined
    # and can wiggle. minimum_required_paths = max(10, 2 x basis terms) is a cheap
    # proxy for the same idea Glasserman & Yu (2004, "Number of Paths versus Number of
    # Basis Functions in American Option Pricing") make rigorous: for polynomial
    # bases under GBM, the path count needed for a reliable fit grows much faster than
    # linearly with the number of basis terms. Woo, Liu & Choi (2024, "Leave-one-out
    # least squares Monte Carlo...") show LSM's look-ahead/overfitting bias is
    # empirically linear in (basis terms / paths); Longstaff & Schwartz (2001) note
    # restricting to ITM-only paths keeps this ratio favorable versus using all paths.
    # The remaining shape checks (ndim, size, matching shapes) are defensive
    # programming with no literature analogue -- they guard against malformed inputs.
    if (
        itm_path_count < minimum_required_paths
        or prices.ndim != 1
        or prices.size < 2
        or payoff.shape != prices.shape
        or continuation.shape != prices.shape
    ):
        return float("nan")

    # Drop non-finite points (can appear near the edge of a polynomial fit's support)
    # and require a non-degenerate price range. Numerical hygiene, not a modeling rule.
    finite = np.isfinite(prices) & np.isfinite(payoff) & np.isfinite(continuation)
    prices = prices[finite]
    difference = (payoff - continuation)[finite]
    if prices.size < 2 or np.ptp(prices) <= np.finfo(float).eps:
        return float("nan")

    # Treat payoff - continuation as exactly zero within a magnitude-scaled tolerance,
    # so floating-point noise around a genuine tie doesn't register as a sign change.
    # A pure numerical-stability guard; if the whole grid is within tolerance of zero,
    # there is no detectable preference anywhere and no boundary can be identified.
    scale = max(
        1.0,
        float(np.max(np.abs(payoff[finite]))),
        float(np.max(np.abs(continuation[finite]))),
    )
    tolerance = 1e-8 * scale
    if np.all(np.abs(difference) <= tolerance):
        return float("nan")

    signs = np.where(
        difference > tolerance,
        1,
        np.where(difference < -tolerance, -1, 0),
    )
    nonzero_indices = np.flatnonzero(signs)
    if nonzero_indices.size < 2:
        return float("nan")
    nonzero_signs = signs[nonzero_indices]
    transitions = np.flatnonzero(nonzero_signs[:-1] != nonzero_signs[1:])
    # Exactly one sign transition, running exercise (+) at low price to continuation
    # (-) at high price: for a single-asset American put, the optimal stopping time
    # has the closed form tau* = inf{t : S(t) <= b*(t)} for one boundary b*(t)
    # (Glasserman, "Monte Carlo Methods in Financial Engineering", 2004, Ch. 8, eq.
    # 8.3 and Fig. 8.1) -- exercise strictly below the boundary, continuation strictly
    # above, with a single crossing. The same chapter (Sec. 8.2, p. 427) notes this
    # simple one-boundary structure is specific to a single underlying asset and does
    # not generalize to multi-asset options, where the exercise region "need not have
    # a simple structure." Rejecting >1 transitions instead of picking one is a
    # deliberate choice (see docstring), not something drawn from a paper -- the
    # papers above establish what the *correct* shape should be, not how to recover
    # it from a noisy regression that violates that shape.
    if transitions.size != 1:
        return float("nan")
    transition = int(transitions[0])
    left_index = int(nonzero_indices[transition])
    right_index = int(nonzero_indices[transition + 1])
    if signs[left_index] != 1 or signs[right_index] != -1:
        return float("nan")
    # Linear interpolation for the root, falling back to the midpoint if the two
    # endpoint differences are numerically equal (avoids a near-zero denominator).
    # Standard root-finding, not a modeling assumption.
    x0, x1 = prices[left_index], prices[right_index]
    d0, d1 = difference[left_index], difference[right_index]
    if np.isclose(d0, d1):
        return float((x0 + x1) / 2.0)
    return float(x0 - d0 * (x1 - x0) / (d1 - d0))


def _build_lsmc_continuation_diagnostics(
    lsmc: LSMCPriceSimulator,
    price_paths: np.ndarray,
    variance_paths: np.ndarray | None,
    contract: PutContract,
    time_grid: np.ndarray,
) -> tuple[LSMCContinuationDiagnostic, ...]:
    """Evaluate stored LSMC regressions on their observed in-the-money support."""
    diagnostics = []
    for timestep, model in sorted(lsmc.continuation_models.items()):
        # This is the same cross-sectional population used to fit the LSMC model:
        # out-of-the-money paths have no immediate exercise decision and are excluded.
        in_the_money = price_paths[timestep] < contract.strike
        itm_prices = price_paths[timestep, in_the_money]
        # No ITM paths at all, or all identical: there is nothing to fit a boundary
        # against at this timestep. Not a modeling rule, just "no data."
        if itm_prices.size < 2 or np.ptp(itm_prices) <= np.finfo(float).eps:
            continue

        grid_low, grid_high = np.percentile(itm_prices, BOUNDARY_GRID_PERCENTILES)
        # The trimmed range collapsed to a point (can happen with heavily duplicated
        # simulated prices). Same numerical-hygiene reasoning as
        # BOUNDARY_GRID_PERCENTILES above, not a separate modeling rule.
        if grid_high <= grid_low:
            continue
        price_grid = np.linspace(
            float(grid_low),
            float(grid_high),
            CONTINUATION_GRID_SIZE,
        )
        representative_variance = None
        grid_variances = None
        if model.uses_variance:
            # Defensive: a Heston-fitted model always has variance_paths available
            # from its caller, so this should not occur in practice.
            if variance_paths is None:
                continue
            # Heston continuation is C(S, v), so the one-dimensional chart is an
            # explicit conditional slice at the median variance of the ITM paths.
            representative_variance = float(
                np.median(variance_paths[timestep, in_the_money])
            )
            grid_variances = np.full_like(price_grid, representative_variance)

        continuation_value = np.asarray(
            model.predict(price_grid, grid_variances), dtype=float
        )
        # The regression extrapolated to a non-finite value somewhere on the grid.
        # Same overfitting/instability concern as the itm_path_count check inside
        # estimate_exercise_boundary below, just caught after evaluation rather than
        # before it.
        if not np.all(np.isfinite(continuation_value)):
            continue
        immediate_payoff = np.maximum(contract.strike - price_grid, 0.0)
        minimum_required_paths = max(10, 2 * len(model.basis_terms))
        boundary = estimate_exercise_boundary(
            price_grid,
            immediate_payoff,
            continuation_value,
            int(itm_prices.size),
            minimum_required_paths,
        )
        diagnostics.append(
            LSMCContinuationDiagnostic(
                timestep=timestep,
                time=float(time_grid[timestep]),
                time_to_maturity=float(contract.maturity - time_grid[timestep]),
                itm_path_count=int(itm_prices.size),
                minimum_required_paths=minimum_required_paths,
                basis_terms=model.basis_terms,
                price_grid=price_grid,
                immediate_payoff=immediate_payoff,
                continuation_value=continuation_value,
                boundary=boundary,
                representative_variance=representative_variance,
            )
        )
    return tuple(diagnostics)


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


CRR_BINOMIAL_STEPS = 4_000


def crr_american_put_price(
    contract: PutContract, market: MarketInputs, steps: int = CRR_BINOMIAL_STEPS
) -> float:
    """Return the American put price from a Cox–Ross–Rubinstein binomial tree.

    This is an independent numerical method to the simulated LSMC estimator, used as
    a converged reference value to validate LSMC American pricing under GBM. The
    binomial tree only models constant-volatility lognormal dynamics, so it is not a
    valid reference under Heston.
    """
    if steps < 1:
        raise ValueError("steps must be positive")
    dt = contract.maturity / steps
    up = exp(market.volatility * sqrt(dt))
    down = 1.0 / up
    growth = exp(market.risk_free_rate * dt)
    up_probability = (growth - down) / (up - down)
    if not 0.0 < up_probability < 1.0:
        raise ValueError(
            "CRR binomial parameters imply a non-arbitrage-free up-probability; "
            "reduce the time step or check the volatility and risk-free rate inputs."
        )
    discount = exp(-market.risk_free_rate * dt)

    down_steps = np.arange(steps + 1)
    terminal_prices = market.spot * up ** (steps - down_steps) * down**down_steps
    values = np.maximum(contract.strike - terminal_prices, 0.0)
    for step in range(steps - 1, -1, -1):
        down_steps = np.arange(step + 1)
        node_prices = market.spot * up ** (step - down_steps) * down**down_steps
        continuation = discount * (
            up_probability * values[:-1] + (1.0 - up_probability) * values[1:]
        )
        immediate_exercise = np.maximum(contract.strike - node_prices, 0.0)
        values = np.maximum(continuation, immediate_exercise)
    return float(values[0])


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
