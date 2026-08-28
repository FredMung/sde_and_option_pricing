"""Streamlit controls, orchestration, explanations, and result rendering."""

from dataclasses import dataclass
from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
import streamlit as st

from option_pricing_app.app.charts import (
    basis_sensitivity_study_figure,
    cash_flow_figure,
    convergence_figure,
    exercise_boundary_figure,
    exercise_figure,
    exercise_grid_study_figure,
    paired_replication_figure,
    path_count_study_figure,
    paths_figure,
    payoff_figure,
    terminal_figure,
    variance_paths_figure,
)
from option_pricing_app.app.static_export import (
    early_exercise_policy_pdf,
    numerical_studies_pdf,
    paired_validation_pdf,
)
from option_pricing_app.domain import (
    HESTON_BASIS_TERMS,
    PRICE_BASIS_TERMS,
    ExerciseStyle,
    HestonInputs,
    MarketInputs,
    PricingResult,
    PutContract,
    SimulationConfig,
)
from option_pricing_app.early_exercise import (
    build_early_exercise_policy_table,
    early_exercise_policy_to_csv,
)
from option_pricing_app.lsmc_policy import evaluate_independent_policy, generate_paths
from option_pricing_app.market_data import (
    MarketDataError,
    OptionChain,
    Quote,
    TreasuryClient,
    TreasuryCurve,
    YahooFinanceClient,
)
from option_pricing_app.numerical_studies import (
    NumericalStudiesResult,
    crr_convergence_table,
    derive_validation_seed,
    estimate_workload,
    generate_base_seed,
    make_exercise_grid,
    make_path_count_grid,
    run_numerical_studies,
    studies_to_csv,
)
from option_pricing_app.policy_validation import (
    PAIRED_VALIDATION_BASE_SEED,
    PairedValidationStudyResult,
    paired_validation_to_csv,
    run_paired_validation_study,
)
from option_pricing_app.service import CRR_BINOMIAL_STEPS, price_put

DISCLAIMER = (
    "This application is an educational proof of concept. Its purpose is to demonstrate the concepts behind "
    "stochastic asset-price models and option-pricing methods. It is not intended "
    "to provide production-grade valuations, trading signals, or investment advice."
)
MANUAL_MATURITY = "Manual maturity"
MIN_PLAUSIBLE_IMPLIED_VOLATILITY = 0.05
MAX_PLAUSIBLE_IMPLIED_VOLATILITY = 3.00
MAX_HOSTED_STUDY_PATH_STEPS = 200_000_000
# Raised from the original 100M once each replication started requiring a second,
# independently simulated validation path set on top of the training set (see
# lsmc_policy.py) -- out-of-sample evaluation roughly doubled the per-replication
# cost. 200M covers, e.g., 20 replications at 10,000 paths / 50 steps (GBM, the
# default 3-term basis), which needs ~185M path-steps.
DEFAULT_STUDY_REPLICATIONS = 20
# Chosen so a full numerical-studies run (path-count, exercise-grid where
# applicable, and basis sensitivity) at DEFAULT_STUDY_REPLICATIONS stays comfortably
# under MAX_HOSTED_STUDY_PATH_STEPS while still giving a visible convergence trend.
# GBM's exercise-grid study is the main cost driver there, so its step count stays
# moderate; Heston has no exercise-grid study but has time-discretisation error from
# its Euler scheme, so its budget goes to more time steps instead.
DEFAULT_SIMULATION_SETTINGS = {
    "GBM": {"n_paths": 5_000, "n_steps": 100},
    "Heston": {"n_paths": 5_000, "n_steps": 120},
}


@dataclass(frozen=True)
class DashboardInputs:
    contract: PutContract
    market: MarketInputs
    simulation: SimulationConfig
    exercise_style: ExerciseStyle
    model: str
    heston_inputs: HestonInputs | None
    market_put_price: float | None
    market_put_price_source: str
    volatility_label: str
    risk_free_rate_label: str
    sources: tuple[tuple[str, str], ...]


@st.cache_data(ttl=60, show_spinner=False)
def cached_quote(ticker: str) -> Quote:
    return YahooFinanceClient().get_spot(ticker)


@st.cache_data(ttl=900, show_spinner=False)
def cached_expiries(ticker: str) -> tuple[str, ...]:
    return YahooFinanceClient().get_expiries(ticker)


@st.cache_data(ttl=900, show_spinner=False)
def cached_chain(ticker: str, expiry: str) -> OptionChain:
    return YahooFinanceClient().get_put_chain(ticker, expiry)


@st.cache_data(ttl=900, show_spinner=False)
def cached_historical_volatility(ticker: str) -> tuple[float, datetime]:
    return YahooFinanceClient().get_historical_volatility(ticker)


@st.cache_data(ttl=900, show_spinner=False)
def cached_treasury_curve() -> TreasuryCurve:
    return TreasuryClient().get_latest_curve()


@st.cache_data(show_spinner=False)
def cached_numerical_studies(
    contract: PutContract,
    market: MarketInputs,
    config: SimulationConfig,
    model: str,
    heston_inputs: HestonInputs | None,
    base_seed: int,
    replications: int,
) -> NumericalStudiesResult:
    """Cache complete repeated-run studies for an identical numerical specification."""
    return run_numerical_studies(
        contract,
        market,
        config,
        model,
        heston_inputs,
        base_seed,
        replications,
    )


@st.cache_data(show_spinner=False)
def cached_paired_validation(
    contract: PutContract,
    market: MarketInputs,
    config: SimulationConfig,
    base_seed: int,
    replications: int,
) -> PairedValidationStudyResult:
    """Cache the paired same-sample/independent replication study for an identical spec."""
    return run_paired_validation_study(contract, market, config, base_seed, replications)


def run_dashboard() -> None:
    st.set_page_config(page_title="SDE Option-Pricing Lab", page_icon="📈", layout="wide")
    st.title("SDE Option-Pricing Lab")
    st.caption("Single-asset put pricing with GBM, Heston, Monte Carlo, and LSMC")
    st.warning(DISCLAIMER, icon="⚠️")

    inputs = _controls()
    if inputs and st.sidebar.button(
        "Generate simulation", type="primary", use_container_width=True
    ):
        try:
            with st.spinner("Simulating risk-neutral price paths..."):
                result = price_put(
                    inputs.contract,
                    inputs.market,
                    inputs.simulation,
                    inputs.exercise_style,
                    inputs.model,
                    inputs.heston_inputs,
                )
            st.session_state["pricing_output"] = (result, inputs)
            st.session_state["numerical_studies_base_seed"] = (
                inputs.simulation.seed
                if inputs.simulation.seed is not None
                else generate_base_seed()
            )
        except (ValueError, FloatingPointError) as exc:
            st.error(f"The simulation could not be completed: {exc}")

    output = st.session_state.get("pricing_output")
    if output:
        result, dashboard_inputs = output
        independent_evaluation = _independent_policy_evaluation(result, dashboard_inputs)
        _headline_results(result, dashboard_inputs, independent_evaluation)
        _numerical_studies(dashboard_inputs)
        _early_exercise_policy(result, dashboard_inputs, independent_evaluation)
        _numerical_validation(dashboard_inputs)
        _supplementary_results(result, dashboard_inputs)
    else:
        st.info("Choose assumptions in the sidebar, then generate a simulation.")
    _methodology()


def _independent_policy_evaluation(result: PricingResult, inputs: DashboardInputs):
    """Evaluate the fitted policy on one independent validation path set.

    Computed once per run and shared by the headline metrics and the
    Early-Exercise Policy section, so both report the same out-of-sample number
    instead of two independently simulated (and therefore slightly different)
    ones. Returns ``None`` for European exercise, where there is no fitted policy.
    """
    if inputs.exercise_style is not ExerciseStyle.AMERICAN or result.fitted_policy is None:
        return None
    base_seed = st.session_state.get("numerical_studies_base_seed")
    if base_seed is None:
        base_seed = generate_base_seed()
        st.session_state["numerical_studies_base_seed"] = base_seed
    validation_seed = derive_validation_seed(base_seed)
    policy = result.fitted_policy
    validation_config = SimulationConfig(
        inputs.simulation.n_paths, policy.n_steps, validation_seed, 1, policy.basis_terms
    )
    validation_paths, validation_variance_paths = generate_paths(
        inputs.contract, inputs.market, validation_config, inputs.model, inputs.heston_inputs
    )
    return evaluate_independent_policy(policy, validation_paths, validation_variance_paths)


def _controls() -> DashboardInputs | None:
    sidebar = st.sidebar
    sidebar.header("Model assumptions")
    if sidebar.button("Reset to live data", use_container_width=True):
        st.session_state["spot_source"] = "Live ticker"
        st.session_state.pop("maturity_choice", None)
        st.session_state.pop("pricing_output", None)

    model = sidebar.selectbox("Asset-price model", ["GBM", "Heston"])
    exercise_style = ExerciseStyle(
        sidebar.selectbox(
            "Exercise style",
            [style.value for style in ExerciseStyle],
            index=1,
        )
    )
    spot_source = sidebar.radio(
        "Initial price source",
        ["Manual price", "Live ticker"],
        index=1,
        key="spot_source",
        help="Manual spot or manual maturity switches all market inputs to manual mode.",
    )

    quote = None
    chain = None
    ticker = ""
    expiry = None
    market_error = None
    if spot_source == "Live ticker":
        ticker = sidebar.text_input("Ticker", value="AAPL").strip().upper()
        try:
            quote = cached_quote(ticker)
            expiries = tuple(
                item
                for item in cached_expiries(ticker)
                if date.fromisoformat(item) > datetime.now(UTC).date()
            )
            if not expiries:
                raise MarketDataError("No unexpired put maturities were available.")
            st.session_state["last_live_spot"] = quote.price
            sidebar.caption(f"Latest {ticker} price: {quote.price:,.2f}")
            options = [*expiries, MANUAL_MATURITY]
            if st.session_state.get("maturity_choice") not in options:
                st.session_state.pop("maturity_choice", None)
            expiry = sidebar.selectbox("Maturity", options, key="maturity_choice")
            if expiry != MANUAL_MATURITY:
                chain = cached_chain(ticker, expiry)
        except (MarketDataError, ValueError) as exc:
            market_error = str(exc)

    manual = determine_manual_mode(spot_source, expiry, market_error)
    if market_error:
        sidebar.warning(f"{market_error} Using manual inputs instead.")
    _manual_defaults(manual, quote, expiry, chain)

    if manual:
        sidebar.info("Manual market mode: enter all market inputs below.")
        spot = float(sidebar.number_input("Initial price, S₀", min_value=0.01, key="manual_spot"))
        maturity = float(
            sidebar.number_input(
                "Maturity (years)", min_value=0.001, format="%.4f", key="manual_maturity"
            )
        )
        strike = float(sidebar.number_input("Strike, K", min_value=0.01, key="manual_strike"))
        volatility = float(
            sidebar.number_input(
                "Volatility (%)", min_value=0.01, max_value=500.0, key="manual_volatility"
            )
        ) / 100
        rate = float(
            sidebar.number_input(
                "Risk-free rate (%)", min_value=-99.0, max_value=99.0, key="manual_rate"
            )
        ) / 100
        market_put_price = None
        market_put_price_source = "Unavailable in manual market mode"
        volatility_label = "Manually input volatility"
        risk_free_rate_label = "Manually input risk-free rate"
        sources = (
            ("Spot", "Manual"),
            ("Maturity", "Manual year fraction"),
            ("Strike", "Manual"),
            ("Volatility", "Manual"),
            ("Risk-free rate", "Manual"),
        )
    else:
        assert quote and chain and expiry
        spot = quote.price
        maturity = _maturity(expiry)
        strike, strike_source = _live_strike(chain, spot)
        market_put_price, market_put_price_source = _live_market_price(chain, strike)
        volatility, volatility_source, volatility_label = _live_volatility(
            chain, ticker, strike
        )
        rate, rate_source, risk_free_rate_label = _live_rate(maturity)
        sources = (
            ("Spot", f"Yahoo Finance; retrieved {quote.observed_at:%Y-%m-%d %H:%M UTC}"),
            ("Maturity", f"Yahoo listed expiry {expiry}; ACT/365.25"),
            ("Strike", strike_source),
            ("Volatility", volatility_source),
            ("Risk-free rate", rate_source),
        )

    heston_inputs = _heston_controls(volatility) if model == "Heston" else None

    sidebar.divider()
    sidebar.subheader("Simulation")
    simulation_defaults = DEFAULT_SIMULATION_SETTINGS[model]
    n_paths = int(
        sidebar.number_input(
            "Number of paths",
            100,
            25_000,
            simulation_defaults["n_paths"],
            step=500,
            key=f"n_paths_{model.lower()}",
        )
    )
    n_steps = int(
        sidebar.number_input(
            "Number of time steps",
            1,
            1_000,
            simulation_defaults["n_steps"],
            step=10,
            key=f"n_steps_{model.lower()}",
        )
    )
    if exercise_style is ExerciseStyle.AMERICAN:
        basis_options = HESTON_BASIS_TERMS if model == "Heston" else PRICE_BASIS_TERMS
        default_basis = (
            ("1", "S", "S²", "v", "v²", "S·v")
            if model == "Heston"
            else PRICE_BASIS_TERMS
        )
        basis_terms = tuple(
            sidebar.multiselect(
                "LSMC basis terms",
                basis_options,
                default=default_basis,
                key=f"lsmc_basis_{model.lower()}",
                help=(
                    "Regression terms for continuation value. Heston also permits "
                    "variance powers and price–variance cross-terms up to power two."
                ),
            )
        )
    else:
        basis_terms = PRICE_BASIS_TERMS
    with sidebar.expander("Advanced settings"):
        fixed_seed = st.checkbox("Use a reproducible random seed", value=True)
        seed = int(st.number_input("Random seed", 0, value=42)) if fixed_seed else None

    try:
        return DashboardInputs(
            PutContract(strike, maturity),
            MarketInputs(spot, volatility, rate),
            SimulationConfig(
                n_paths,
                n_steps,
                seed,
                min(100, n_paths),
                basis_terms,
            ),
            exercise_style,
            model,
            heston_inputs,
            market_put_price,
            market_put_price_source,
            volatility_label,
            risk_free_rate_label,
            sources,
        )
    except ValueError as exc:
        sidebar.error(str(exc))
        return None


def _heston_controls(reference_volatility: float) -> HestonInputs:
    sidebar = st.sidebar
    sidebar.divider()
    sidebar.subheader("Heston variance process")
    initial_volatility = float(
        sidebar.number_input(
            "Initial volatility, √v₀ (%)",
            min_value=0.01,
            max_value=500.0,
            value=float(reference_volatility * 100.0),
        )
    ) / 100.0
    long_run_volatility = float(
        sidebar.number_input(
            "Long-run volatility, √θ (%)",
            min_value=0.01,
            max_value=500.0,
            value=float(reference_volatility * 100.0),
        )
    ) / 100.0
    mean_reversion_speed = float(
        sidebar.number_input("Mean-reversion speed, κ", 0.01, 20.0, 2.0, step=0.1)
    )
    volatility_of_variance = float(
        sidebar.number_input("Volatility of variance, ξ", 0.001, 5.0, 0.30, step=0.05)
    )
    correlation = float(
        sidebar.number_input("Price–variance correlation, ρ", -1.0, 1.0, -0.70, step=0.05)
    )
    inputs = HestonInputs(
        mean_reversion_speed=mean_reversion_speed,
        long_run_variance=long_run_volatility**2,
        volatility_of_variance=volatility_of_variance,
        correlation=correlation,
        initial_variance=initial_volatility**2,
    )
    sidebar.caption(
        f"Initial variance v₀={inputs.initial_variance:.6f}; "
        f"long-run variance θ={inputs.long_run_variance:.6f}."
    )
    if not inputs.satisfies_feller_condition:
        sidebar.warning(
            "The Feller condition 2κθ ≥ ξ² is not satisfied; the full-truncation "
            "scheme keeps simulated variance non-negative."
        )
    return inputs


def _live_strike(chain: OptionChain, spot: float) -> tuple[float, str]:
    strikes = YahooFinanceClient.listed_strikes(chain)
    source = st.sidebar.radio("Strike source", ["Listed strike", "Manual strike"])
    if source == "Listed strike":
        index = min(range(len(strikes)), key=lambda position: abs(strikes[position] - spot))
        return float(st.sidebar.selectbox("Strike, K", strikes, index=index)), "Yahoo put chain"
    return float(st.sidebar.number_input("Strike, K", 0.01, value=round(spot, 2))), "Manual"


def _live_volatility(
    chain: OptionChain, ticker: str, strike: float
) -> tuple[float, str, str]:
    source = st.sidebar.selectbox(
        "Volatility source", ["Implied volatility", "Historical volatility", "Manual volatility"]
    )
    if source == "Manual volatility":
        return (
            float(st.sidebar.number_input("Volatility (%)", 0.01, 500.0, 20.0)) / 100,
            "Manual",
            "Manually input volatility",
        )
    try:
        if source == "Historical volatility":
            value, observed = cached_historical_volatility(ticker)
            st.sidebar.caption(f"1-year annualised volatility: {value:.2%}")
            return (
                value,
                f"Yahoo adjusted closes through {observed:%Y-%m-%d}; √252",
                "Historical volatility",
            )
        estimate = YahooFinanceClient.nearest_implied_volatility(chain, strike)
        st.sidebar.caption(f"Put-chain implied volatility: {estimate.volatility:.2%}")
        if estimate.is_approximation:
            st.sidebar.warning(
                f"Using IV from nearest listed strike K={estimate.matched_strike:,.2f}."
            )
        if not is_plausible_implied_volatility(estimate.volatility):
            value, observed = cached_historical_volatility(ticker)
            st.sidebar.warning(
                f"Yahoo's put IV of {estimate.volatility:.2%} is outside the "
                "5%–300% data-quality range. Using one-year historical volatility "
                f"of {value:.2%} instead."
            )
            return (
                value,
                (
                    f"Yahoo adjusted closes through {observed:%Y-%m-%d}; √252 "
                    f"(fallback from implausible {estimate.volatility:.2%} put IV)"
                ),
                "Historical volatility (IV fallback)",
            )
        return (
            estimate.volatility,
            f"Yahoo put IV at K={estimate.matched_strike:,.2f}",
            "Implied volatility",
        )
    except MarketDataError as exc:
        st.sidebar.warning(f"{exc} Enter volatility manually.")
        return (
            float(
                st.sidebar.number_input("Fallback volatility (%)", 0.01, 500.0, 20.0)
            )
            / 100,
            "Manual fallback",
            "Manually input volatility (fallback)",
        )


def _live_market_price(chain: OptionChain, strike: float) -> tuple[float | None, str]:
    try:
        quote = YahooFinanceClient.put_market_price(chain, strike)
    except MarketDataError as exc:
        st.sidebar.caption(f"Market-price comparison unavailable: {exc}")
        return None, str(exc)
    source = (
        f"Yahoo put {quote.basis} at K={quote.strike:,.2f}; "
        f"retrieved {quote.observed_at:%Y-%m-%d %H:%M UTC}"
    )
    return quote.price, source


def _live_rate(maturity: float) -> tuple[float, str, str]:
    source = st.sidebar.radio("Risk-free-rate source", ["U.S. Treasury curve", "Manual rate"])
    if source == "Manual rate":
        return (
            float(st.sidebar.number_input("Risk-free rate (%)", -99.0, 99.0, 4.0)) / 100,
            "Manual",
            "Manually input risk-free rate",
        )
    try:
        curve = cached_treasury_curve()
        value, outside = curve.rate_for_maturity(maturity)
        st.sidebar.caption(f"Treasury par-yield proxy: {value:.2%} ({curve.as_of})")
        if outside:
            boundary = "shortest" if maturity < curve.maturities[0] else "longest"
            message = (
                f"Selected expiry is outside the Treasury curve; the {boundary} "
                "available tenor is used."
            )
            if maturity < curve.maturities[0]:
                st.sidebar.caption(message)
            else:
                st.sidebar.warning(message)
        source = f"U.S. Treasury par curve dated {curve.as_of}; interpolated proxy"
        if outside:
            source += f" ({boundary} available tenor used)"
        return value, source, "U.S. Treasury risk-free rate"
    except MarketDataError as exc:
        st.sidebar.warning(f"{exc} Enter a rate manually.")
        return (
            float(
                st.sidebar.number_input("Fallback risk-free rate (%)", -99.0, 99.0, 4.0)
            )
            / 100,
            "Manual fallback",
            "Manually input risk-free rate (fallback)",
        )


def _manual_defaults(manual: bool, quote, expiry, chain) -> None:
    previous = st.session_state.get("_manual_mode", manual)
    if manual and not previous:
        spot = quote.price if quote else st.session_state.get("last_live_spot", 100.0)
        st.session_state.update(
            manual_spot=float(spot),
            manual_maturity=float(_maturity(expiry) if expiry and expiry != MANUAL_MATURITY else 1.0),
            manual_strike=float(spot),
            manual_volatility=20.0,
            manual_rate=4.0,
        )
    st.session_state.setdefault("manual_spot", 100.0)
    st.session_state.setdefault("manual_maturity", 1.0)
    st.session_state.setdefault("manual_strike", 100.0)
    st.session_state.setdefault("manual_volatility", 20.0)
    st.session_state.setdefault("manual_rate", 4.0)
    st.session_state["_manual_mode"] = manual


def determine_manual_mode(spot_source: str, expiry: str | None, error: str | None) -> bool:
    return spot_source == "Manual price" or expiry == MANUAL_MATURITY or error is not None


def is_plausible_implied_volatility(volatility: float) -> bool:
    """Apply a broad guard against Yahoo sentinel or malformed IV values."""
    return MIN_PLAUSIBLE_IMPLIED_VOLATILITY <= volatility <= MAX_PLAUSIBLE_IMPLIED_VOLATILITY


def _maturity(expiry: str) -> float:
    days = (date.fromisoformat(expiry) - datetime.now(UTC).date()).days
    if days <= 0:
        raise ValueError("expiry must be in the future")
    return days / 365.25


def _headline_results(
    result: PricingResult,
    inputs: DashboardInputs,
    independent_evaluation,
) -> None:
    st.subheader("Simulation results")
    st.caption("Results reflect the inputs captured when Generate simulation was pressed.")
    market = inputs.market_put_price
    known_facts, separator, simulation_metrics = st.columns([1, 0.04, 1])

    with known_facts:
        st.markdown("#### Contract & Market")
        st.metric("Spot price", f"{inputs.market.spot:,.4f}")
        st.metric("Strike", f"{inputs.contract.strike:,.4f}")
        st.metric("Maturity", f"{inputs.contract.maturity:.4f} years")
        st.metric(
            "Market put value",
            "N/A" if market is None else f"{market:,.4f}",
        )
        st.metric(inputs.volatility_label, f"{inputs.market.volatility:.2%}")
        st.metric(inputs.risk_free_rate_label, f"{inputs.market.risk_free_rate:.2%}")

    with separator:
        st.markdown(
            '<div style="border-left: 1px solid rgba(128, 128, 128, 0.45); '
            'height: 35rem; margin: 0 auto;"></div>',
            unsafe_allow_html=True,
        )

    with simulation_metrics:
        st.markdown("#### LSMC Estimate")
        if inputs.exercise_style is ExerciseStyle.AMERICAN:
            # independent_evaluation is always populated for American exercise (see
            # _independent_policy_evaluation); the price/SE shown here are the frozen
            # policy's out-of-sample estimate, not the same-sample training price,
            # since the latter is biased high (the stopping rule was chosen to fit
            # that exact data -- see the Numerical Validation section).
            price = independent_evaluation.mean_estimate
            standard_error = independent_evaluation.standard_error
            st.metric(
                "Estimated American put value (out-of-sample)",
                f"{price:,.4f}",
                delta=f"Same-sample: {result.price:,.4f}",
                delta_color="off",
                help=(
                    "The fitted LSMC stopping policy, evaluated on an independently "
                    "simulated validation path set rather than its own training "
                    "paths. The same-sample training price shown in the delta is "
                    "biased high, because that stopping rule was chosen to maximise "
                    "value on that exact data; see Numerical Validation below for "
                    "the full paired comparison across replications."
                ),
            )
            st.metric(
                "Estimated European put value",
                f"{result.european_mc_price:,.4f}",
            )
            early_exercise_premium = price - result.european_mc_price
            premium_standard_error = (
                standard_error**2 + result.european_mc_standard_error**2
            ) ** 0.5
            st.metric(
                "Early-exercise premium (American − European)",
                f"{early_exercise_premium:,.4f}",
                delta=f"± {premium_standard_error:,.4f} approx. SE",
                delta_color="off",
                help=(
                    "Out-of-sample American minus same-sample European Monte Carlo "
                    "estimate: the value the LSMC early-exercise feature adds. The "
                    "shown standard error treats the two estimates as independent."
                ),
            )
        else:
            price = result.price
            standard_error = result.standard_error
            st.metric("Estimated European put value", f"{price:,.4f}")
            if result.european_exact_price is not None:
                st.metric(
                    "Exact European put value",
                    f"{result.european_exact_price:,.4f}",
                    delta=(
                        "MC − exact: "
                        f"{result.european_mc_price - result.european_exact_price:+,.4f}"
                    ),
                    delta_color="off",
                )
            else:
                st.metric("Exact European put value", "Not implemented for Heston")
        low = max(0.0, price - 1.96 * standard_error)
        high = price + 1.96 * standard_error
        if inputs.exercise_style is ExerciseStyle.AMERICAN:
            se_label = "Monte Carlo standard error (out-of-sample)"
            interval_label = "Approximate 95% interval (out-of-sample American put value)"
            interval_help = (
                "Normal approximation around the out-of-sample estimate above: "
                "price ± 1.96 × its Monte Carlo standard error, clipped to be "
                "non-negative. This reflects one independent evaluation's own "
                "sampling noise, not the full spread across replications shown in "
                "Numerical Validation. Not a confidence interval for the market put "
                "value or for any other metric on this page."
            )
        else:
            se_label = "Monte Carlo standard error"
            interval_label = "Approximate 95% interval (European put value)"
            interval_help = (
                "Normal approximation for the estimated European put value: "
                "price ± 1.96 × Monte Carlo standard error, clipped to be "
                "non-negative. Not a confidence interval for the market put value "
                "or for any other metric on this page."
            )
        st.metric(se_label, f"{standard_error:,.4f}")
        st.metric(interval_label, f"[{low:,.4f}, {high:,.4f}]", help=interval_help)
        if market is not None:
            st.metric(
                "Simulated − Market",
                f"{price - market:+,.4f}",
                help=(
                    "The estimated value above (out-of-sample for American, the "
                    "Monte Carlo estimate for European) minus the observed market "
                    "put value. A difference may reflect dividends, "
                    "volatility-surface effects, model calibration, jumps, "
                    "liquidity, transaction costs, delayed quotes, or bid-ask "
                    "effects -- not necessarily a pricing error."
                ),
            )


def _supplementary_results(result: PricingResult, inputs: DashboardInputs) -> None:
    if not (result.discounted_realised_cash_flows > 0).any():
        st.warning(
            "No simulated path produced a positive option cash flow. This can occur "
            "for a short-dated out-of-the-money put when volatility is very low; "
            "check the maturity, strike and volatility inputs."
        )

    if (
        inputs.exercise_style is ExerciseStyle.EUROPEAN
        and result.european_exact_price is not None
    ):
        st.caption(
            "The European comparison validates terminal-payoff Monte Carlo against the "
            "closed-form Black–Scholes value under the same model inputs."
        )
    elif inputs.exercise_style is ExerciseStyle.EUROPEAN:
        st.caption(
            "The European Heston value is estimated by Monte Carlo; a semi-analytical "
            "Heston benchmark is not implemented in this application."
        )
    else:
        st.caption(
            "The American–European difference is the estimated value added by the "
            "LSMC early-exercise feature: an out-of-sample American estimate (the "
            "frozen policy evaluated on independent validation paths) minus the "
            "same-sample European estimate from this training run."
        )
    st.caption(f"Market reference: {inputs.market_put_price_source}")
    with st.expander("Single-run cumulative estimate", expanded=False):
        st.plotly_chart(convergence_figure(result), use_container_width=True)
        st.caption(
            "The line successively includes discounted cash flows from one completed "
            "simulation. For an American option, every point uses the stopping policy "
            "fitted with the full path pool, so this is not a path-count convergence "
            "study."
        )
    st.plotly_chart(paths_figure(result, inputs.contract), use_container_width=True)
    st.caption(
        "The shaded area contains the central 90% of simulated asset prices at each "
        "time step; it is a path-distribution band, not a confidence interval."
    )
    if result.displayed_variance_paths is not None:
        st.plotly_chart(variance_paths_figure(result), use_container_width=True)
        st.caption(
            "Variance is volatility squared. The shaded area contains the central "
            "90% of simulated Heston variance values at each time step."
        )
    left, right = st.columns(2)
    left.plotly_chart(terminal_figure(result, inputs.contract), use_container_width=True)
    right.plotly_chart(payoff_figure(result, inputs.contract), use_container_width=True)
    st.plotly_chart(cash_flow_figure(result), use_container_width=True)
    if result.exercise_style is ExerciseStyle.AMERICAN:
        st.caption(
            "American cash flows use each path's final LSMC stopping time and are "
            "discounted from that exercise time to time zero."
        )
    else:
        st.caption(
            "European cash flows are terminal put payoffs discounted from maturity "
            "to time zero."
        )

    rows = [
        ("Exercise style", inputs.exercise_style.value),
        ("Model / method", f"{inputs.model} / {result.pricing_method}"),
        ("Initial price", f"{inputs.market.spot:,.4f}"),
        ("Strike", f"{inputs.contract.strike:,.4f}"),
        ("Maturity", f"{inputs.contract.maturity:.6f} years"),
        (
            "Reference volatility" if inputs.model == "Heston" else "Volatility",
            f"{inputs.market.volatility:.4%}",
        ),
        ("Risk-free rate", f"{inputs.market.risk_free_rate:.4%}"),
        ("Paths / steps", f"{inputs.simulation.n_paths:,} / {inputs.simulation.n_steps:,}"),
        ("Seed", str(inputs.simulation.seed)),
        *(
            [("LSMC basis terms", ", ".join(inputs.simulation.basis_terms))]
            if inputs.exercise_style is ExerciseStyle.AMERICAN
            else []
        ),
        *inputs.sources,
    ]
    if inputs.heston_inputs is not None:
        rows.extend(
            [
                ("Heston initial variance, v₀", f"{inputs.heston_inputs.initial_variance:.6f}"),
                ("Heston long-run variance, θ", f"{inputs.heston_inputs.long_run_variance:.6f}"),
                ("Heston mean-reversion speed, κ", f"{inputs.heston_inputs.mean_reversion_speed:.4f}"),
                ("Heston volatility of variance, ξ", f"{inputs.heston_inputs.volatility_of_variance:.4f}"),
                ("Heston price–variance correlation, ρ", f"{inputs.heston_inputs.correlation:.4f}"),
                (
                    "Heston Feller condition",
                    "Satisfied" if inputs.heston_inputs.satisfies_feller_condition else "Not satisfied",
                ),
            ]
        )
    st.subheader("Inputs and provenance")
    st.dataframe(pd.DataFrame(rows, columns=["Input", "Value or source"]), hide_index=True, use_container_width=True)


def _early_exercise_policy(
    result: PricingResult,
    inputs: DashboardInputs,
    independent_evaluation,
) -> None:
    """Render the fitted stopping policy, its boundary, and the early-exercise value.

    The stopping-frequency chart and the early-exercise-value table both report
    out-of-sample statistics from ``independent_evaluation`` -- the training run's
    frozen policy (``result.fitted_policy``), evaluated without refitting on an
    independently simulated validation path set, computed once and shared with the
    headline metrics. The exercise-boundary chart is the deliberate exception -- it
    stays in-sample, since it is literally a picture of the training continuation
    regressions, not a statistic to validate out-of-sample.
    """
    st.divider()
    st.subheader("Early-Exercise Policy")
    if inputs.exercise_style is not ExerciseStyle.AMERICAN:
        st.info("The early-exercise policy is available when the exercise style is American.")
        return
    st.caption(
        "The fitted LSMC stopping rule for this run: when it exercises, where the "
        "boundary between exercise and continuation sits, and how much value the "
        "early-exercise feature is estimated to add."
    )

    evaluation = independent_evaluation
    exercise_counts = np.bincount(
        evaluation.exercise_steps[evaluation.exercise_steps >= 0],
        minlength=result.fitted_policy.n_steps + 1,
    )
    exercise_percentages = 100.0 * exercise_counts / inputs.simulation.n_paths
    out_of_the_money_percentage = 100.0 * float(np.mean(evaluation.exercise_steps == -1))

    st.plotly_chart(
        exercise_figure(result.time_grid, exercise_percentages, out_of_the_money_percentage),
        use_container_width=True,
    )
    st.caption(
        "Each independent validation path is counted once at its earliest selected "
        "exercise time under the frozen fitted policy; paths that expire out of the "
        "money are not counted. Being in the money does not automatically trigger "
        "exercise because continuation may be more valuable. These percentages are "
        "risk-neutral model estimates, not a forecast of how many investors will "
        "exercise."
    )
    if result.continuation_diagnostics:
        st.plotly_chart(
            exercise_boundary_figure(result, inputs.contract),
            use_container_width=True,
        )
        boundary_caption = (
            "This boundary is drawn from the training run's fitted continuation "
            "regressions (in-sample) -- it is a picture of the fitted policy itself, "
            "not a statistic to validate out-of-sample. Each point is obtained by "
            "solving immediate payoff = fitted continuation value on the central "
            "5th-95th percentile of that timestep's in-the-money prices; the thinly "
            "populated outer tails are excluded because the regression is "
            "extrapolating rather than interpolating there. A point is shown only "
            "when there are sufficient in-the-money observations and exactly one "
            "exercise-to-continuation sign change within that range. Missing "
            "timesteps had too few observations, no in-range crossing, or several "
            "crossings. No boundary is extrapolated beyond the trimmed range or "
            "selected arbitrarily from multiple roots. Gaps are retained between "
            "omitted timesteps."
        )
        if any(
            diagnostic.representative_variance is not None
            for diagnostic in result.continuation_diagnostics
        ):
            boundary_caption += (
                " Because Heston continuation depends on both price and variance, each "
                "displayed boundary is conditional on that timestep's median "
                "in-the-money variance."
            )
        st.caption(boundary_caption)
    elif result.continuation_diagnostics == ():
        st.info(
            "No LSMC continuation regression could be visualised. Increase the number "
            "of time steps or choose a contract with in-the-money simulated paths."
        )

    st.markdown("#### Estimated value from early exercise")
    try:
        policy_rows = build_early_exercise_policy_table(
            result,
            inputs.contract,
            inputs.market,
            inputs.model,
            independent_evaluation,
        )
    except (ValueError, FloatingPointError) as exc:
        st.error(f"The early-exercise policy table could not be computed: {exc}")
        return
    policy_table = pd.DataFrame(
        [
            {
                "Method": row.method,
                "American": row.american_value,
                "American (s.e.)": row.american_standard_error,
                "European": row.european_value,
                "Early exercise value": row.early_exercise_value,
            }
            for row in policy_rows
        ]
    )
    st.dataframe(policy_table, hide_index=True, use_container_width=True)
    if inputs.model == "GBM":
        difference = policy_rows[0].early_exercise_value - policy_rows[1].early_exercise_value
        st.caption(
            "The early exercise value is the American value minus the European value "
            "(Longstaff & Schwartz 2001, Table 1). The reference row uses a CRR "
            f"binomial tree ({CRR_BINOMIAL_STEPS:,} steps) and the closed-form "
            "Black–Scholes price in place of the paper's finite-difference benchmark; "
            "the LSMC row is the fitted policy's out-of-sample value (see above), not "
            "a same-sample estimate. "
            f"Difference in early exercise value (reference − out-of-sample): {difference:+,.4f}."
        )
    else:
        st.caption(
            "No independent reference row is shown for Heston: the CRR binomial tree "
            "and the closed-form European price both assume constant-volatility "
            "lognormal dynamics, so neither is a valid benchmark here."
        )

    download_csv, download_pdf = st.columns(2)
    download_csv.download_button(
        "Download early-exercise policy (CSV)",
        data=early_exercise_policy_to_csv(policy_rows),
        file_name="early_exercise_policy.csv",
        mime="text/csv",
        use_container_width=True,
    )
    try:
        policy_pdf_bytes = early_exercise_policy_pdf(
            result,
            inputs.contract,
            policy_rows,
            result.time_grid,
            exercise_percentages,
            out_of_the_money_percentage,
        )
    except (ValueError, RuntimeError) as exc:
        download_pdf.error(f"The PDF export could not be generated: {exc}")
    else:
        download_pdf.download_button(
            "Download early-exercise policy charts (PDF)",
            data=policy_pdf_bytes,
            file_name="early_exercise_policy.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
        st.caption(
            "The PDF is a vector figure regenerated directly from this run's summary "
            "data, not a rendering of the interactive Plotly figures."
        )


def _numerical_studies(inputs: DashboardInputs) -> None:
    """Automatically run and render the repeated-run convergence and sensitivity studies."""
    st.divider()
    st.subheader("Numerical studies")
    st.caption(
        "This is the key analysis: repeated, fully independent reruns of the "
        "simulation and LSMC fit, used to assess convergence and sensitivity rather "
        "than to report a single point estimate. Each point is the fitted policy's "
        "out-of-sample value -- evaluated on independently simulated validation "
        "paths, not the same-sample training price -- so a same-sample reuse "
        "artifact cannot be mistaken for a genuine sensitivity."
    )
    if inputs.exercise_style is not ExerciseStyle.AMERICAN:
        st.info("Numerical LSMC studies are available when the exercise style is American.")
        return

    replications = int(
        st.number_input(
            "Number of replications",
            min_value=3,
            max_value=20,
            value=DEFAULT_STUDY_REPLICATIONS,
            step=1,
            help=(
                "Each setting is fit on a fresh training path set and evaluated on a "
                "separate, independently simulated validation path set once per "
                "replication seed pair."
            ),
        )
    )
    pricing_runs, path_steps = estimate_workload(
        inputs.simulation, inputs.model, replications
    )
    path_grid = make_path_count_grid(inputs.simulation.n_paths)
    exercise_grid = (
        make_exercise_grid(inputs.simulation.n_steps) if inputs.model == "GBM" else ()
    )
    st.caption(
        f"Estimated workload: {pricing_runs:,} complete pricing runs and "
        f"approximately {path_steps:,} simulated path-steps. Runs are sequential and "
        "only scalar results are retained."
    )
    if len(path_grid) < 3:
        st.warning(
            "The selected maximum forms fewer than three path-count settings. Increase "
            "the number of paths for a more informative convergence study."
        )
    if inputs.model == "GBM" and len(exercise_grid) < 3:
        st.warning(
            "The selected maximum forms fewer than three exercise-grid settings. "
            "Increase the number of time steps for a more informative grid study."
        )
    if inputs.model == "Heston":
        st.info(
            "The exercise-grid study is disabled for Heston: changing the number of "
            "steps changes both the permitted exercise dates and the numerical "
            "discretisation of the Heston SDEs."
        )

    if path_steps > MAX_HOSTED_STUDY_PATH_STEPS:
        st.warning(
            "This study is too expensive for the hosted-app workload limit, so it has "
            "not run automatically. Reduce paths, time steps, basis complexity, or "
            "replications."
        )
        return

    base_seed = st.session_state.get("numerical_studies_base_seed")
    if base_seed is None:
        base_seed = generate_base_seed()
        st.session_state["numerical_studies_base_seed"] = base_seed
    try:
        with st.spinner("Running repeated simulations and LSMC fits..."):
            studies = cached_numerical_studies(
                inputs.contract,
                inputs.market,
                inputs.simulation,
                inputs.model,
                inputs.heston_inputs,
                base_seed,
                replications,
            )
    except (ValueError, FloatingPointError) as exc:
        st.error(f"The numerical studies could not be completed: {exc}")
        return

    st.session_state["numerical_studies_output"] = studies
    _render_numerical_studies(studies)


def _bias_rmse_rows(points, setting_column: str) -> list[dict]:
    return [
        {
            setting_column: point.setting_label,
            "Mean estimated value": point.mean_estimate,
            "Empirical standard deviation": point.empirical_standard_deviation,
            "Bias vs CRR binomial": point.bias_vs_binomial,
            "RMSE vs CRR binomial": point.rmse_vs_binomial,
            "Replications": len(point.runs),
        }
        for point in points
    ]


def _crr_convergence_rows(points) -> list[dict]:
    return [
        {
            "Binomial steps": point.steps,
            "CRR price": point.price,
            "Change from previous": point.change_from_previous,
        }
        for point in points
    ]


def _render_numerical_studies(studies: NumericalStudiesResult) -> None:
    """Present repeated-run summaries without retaining any experiment paths."""
    if studies.path_count.binomial_reference is not None:
        st.markdown("#### CRR binomial-tree convergence")
        crr_points = crr_convergence_table(studies.path_count.contract, studies.path_count.market)
        st.dataframe(
            pd.DataFrame(_crr_convergence_rows(crr_points)),
            hide_index=True,
            use_container_width=True,
            column_config={
                # Default float formatting rounds both columns to too few decimals
                # to show this table's point: the price and its step-to-step change
                # shrink into the 1e-4 to 1e-5 range as steps increase.
                "CRR price": st.column_config.NumberColumn(format="%.6f"),
                "Change from previous": st.column_config.NumberColumn(format="%.6f"),
            },
        )
        st.caption(
            f"The CRR reference used throughout this section is computed at "
            f"{CRR_BINOMIAL_STEPS:,} steps. This is a property of the binomial tree "
            "itself, independent of the LSMC simulation: it shows the price has "
            "already stabilised well before that step count, supporting its use as a "
            "converged reference value."
        )

    st.plotly_chart(path_count_study_figure(studies.path_count), use_container_width=True)
    st.caption(
        "Every point reruns path simulation, all LSMC continuation regressions and the "
        "stopping policy. The band is the 2.5–97.5% empirical replication interval, "
        "not a confidence interval containing all model and regression uncertainty."
    )
    if studies.path_count.binomial_reference is not None:
        st.dataframe(
            pd.DataFrame(_bias_rmse_rows(studies.path_count.points, "Path count")),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            f"CRR binomial-tree reference (American, {CRR_BINOMIAL_STEPS:,} steps): "
            f"{studies.path_count.binomial_reference:,.4f}. Bias is the mean LSMC "
            "estimate minus this independently computed reference; RMSE combines bias "
            "and replication variance and is never smaller than |bias|."
        )

    if studies.exercise_grid is not None:
        st.plotly_chart(
            exercise_grid_study_figure(studies.exercise_grid),
            use_container_width=True,
        )
        st.caption(
            "Under GBM the transition between selected dates is exact. Refining this "
            "grid therefore mainly increases the number of permitted exercise dates."
        )
        if studies.exercise_grid.binomial_reference is not None:
            st.dataframe(
                pd.DataFrame(
                    _bias_rmse_rows(studies.exercise_grid.points, "Exercise dates")
                ),
                hide_index=True,
                use_container_width=True,
            )
            st.caption(
                "Bias and RMSE against the same CRR binomial reference as the "
                "path-count study, isolating how the exercise-date grid alone affects "
                "accuracy."
            )

    st.plotly_chart(
        basis_sensitivity_study_figure(studies.basis_sensitivity),
        use_container_width=True,
    )
    basis_rows = [
        {
            "Basis terms": ", ".join(point.basis_terms),
            "Mean estimated value": point.mean_estimate,
            "Empirical standard deviation": point.empirical_standard_deviation,
            "Bias vs CRR binomial": point.bias_vs_binomial,
            "RMSE vs CRR binomial": point.rmse_vs_binomial,
            "Replications": len(point.runs),
        }
        for point in studies.basis_sensitivity.points
    ]
    st.dataframe(pd.DataFrame(basis_rows), hide_index=True, use_container_width=True)
    st.caption(
        "The basis study holds the paths, time grid and model inputs fixed by seed while "
        "refitting the complete stopping policy for each specification. Bias and RMSE "
        "are only populated under GBM, where the CRR binomial tree is a valid reference."
    )
    download_csv, download_pdf = st.columns(2)
    download_csv.download_button(
        "Download numerical studies (CSV)",
        data=studies_to_csv(studies),
        file_name="numerical_studies.csv",
        mime="text/csv",
        use_container_width=True,
    )
    try:
        studies_pdf_bytes = numerical_studies_pdf(studies)
    except (ValueError, RuntimeError) as exc:
        download_pdf.error(f"The PDF export could not be generated: {exc}")
    else:
        download_pdf.download_button(
            "Download numerical studies charts (PDF)",
            data=studies_pdf_bytes,
            file_name="numerical_studies.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
        st.caption(
            "The PDF is a vector figure regenerated directly from the summary "
            "statistics above (the same data as the CSV export), not a rendering of "
            "the interactive Plotly figures, which would require the optional "
            "Kaleido dependency."
        )


def _numerical_validation(inputs: DashboardInputs) -> None:
    """Paired same-sample vs independent replication validation under GBM.

    For each replication, fits the LSMC policy on one simulated path set (the
    training seed) and separately evaluates that frozen policy, without refitting,
    on an independently simulated path set (the valuation seed) -- Section 4.7's
    Table 2 design: same-sample vs independent policy value. The independent
    estimate is interpreted as a valid low estimator of the true price (Glasserman,
    *Monte Carlo Methods in Financial Engineering*, 2004, Sec. 8.6), not the biased
    same-sample estimate ordinarily reported. GBM only, since the CRR references
    assume constant-volatility lognormal dynamics.
    """
    st.divider()
    st.subheader("Numerical validation")
    st.caption(
        "Compares the same-sample LSMC estimate (fit and evaluated on the same "
        "training paths -- biased high, because the stopping rule was chosen to fit "
        "that exact data) against the frozen policy's independent, out-of-sample "
        "value (evaluated on separately simulated validation paths), across "
        "repeated paired replications, alongside a CRR binomial reference restricted "
        "to the same exercise grid. GBM only."
    )
    if inputs.exercise_style is not ExerciseStyle.AMERICAN or inputs.model != "GBM":
        st.info(
            "Numerical validation is available for American exercise under the GBM "
            "model, where an independent CRR binomial reference exists."
        )
        return

    replications = int(
        st.number_input(
            "Number of paired replications",
            min_value=3,
            max_value=20,
            value=DEFAULT_STUDY_REPLICATIONS,
            step=1,
            key="paired_validation_replications",
            help=(
                "Each replication derives a training seed and a valuation seed from "
                f"a fixed base seed ({PAIRED_VALIDATION_BASE_SEED}), so this table "
                "reproduces identically on every run, independent of the live "
                "simulation's seed."
            ),
        )
    )
    try:
        with st.spinner("Running the paired validation replications..."):
            study = cached_paired_validation(
                inputs.contract,
                inputs.market,
                inputs.simulation,
                PAIRED_VALIDATION_BASE_SEED,
                replications,
            )
    except (ValueError, FloatingPointError) as exc:
        st.error(f"The numerical validation could not be completed: {exc}")
        return

    st.plotly_chart(paired_replication_figure(study), use_container_width=True)
    st.caption(
        "Each pair of markers is joined because both come from the same training and "
        "valuation seed pair -- the line is not implying a continuous ordering across "
        "replications."
    )

    summary_rows = [
        {"Validation measure": "Continuous-exercise CRR", "Value": study.continuous_crr},
        {"Validation measure": "Matched-grid CRR", "Value": study.matched_grid_crr},
        {"Validation measure": "Mean same-sample LSMC", "Value": study.mean_same_sample_estimate},
        {
            "Validation measure": "Same-sample empirical SD",
            "Value": study.same_sample_empirical_sd,
        },
        {
            "Validation measure": "Mean independent policy value",
            "Value": study.mean_independent_estimate,
        },
        {
            "Validation measure": "Independent empirical SD",
            "Value": study.independent_empirical_sd,
        },
        {"Validation measure": "Mean paired gap", "Value": study.mean_paired_difference},
        {
            "Validation measure": "Independent mean error against matched CRR",
            "Value": study.independent_mean_error_vs_matched_crr,
        },
        {
            "Validation measure": "Independent RMSE against matched CRR",
            "Value": study.independent_rmse_vs_matched_crr,
        },
    ]
    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)
    st.caption(
        "The mean paired gap (same-sample minus independent) is an in-sample reuse "
        "gap for this contract and setting, not a formal high-bias estimate: a "
        "handful of replications shows whether reuse matters here, not a general "
        f"bias magnitude. Matched-grid CRR restricts exercise to the same "
        f"{inputs.simulation.n_steps} dates as the LSMC policy; continuous-exercise "
        f"CRR uses a {CRR_BINOMIAL_STEPS:,}-step tree as a near-continuous-exercise "
        "reference."
    )

    download_csv, download_pdf = st.columns(2)
    download_csv.download_button(
        "Download numerical validation (CSV)",
        data=paired_validation_to_csv(study),
        file_name="numerical_validation.csv",
        mime="text/csv",
        use_container_width=True,
    )
    try:
        validation_pdf_bytes = paired_validation_pdf(study)
    except (ValueError, RuntimeError) as exc:
        download_pdf.error(f"The PDF export could not be generated: {exc}")
    else:
        download_pdf.download_button(
            "Download numerical validation charts (PDF)",
            data=validation_pdf_bytes,
            file_name="numerical_validation.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
        st.caption(
            "The PDF is a vector figure regenerated directly from the summary "
            "statistics above (the same data as the CSV export)."
        )


def _methodology() -> None:
    with st.expander("Methodology and interpretation"):
        st.markdown("#### Model setting")
        st.markdown(
            """
            Valuation is performed under the risk-neutral measure, not the physical
            or real-world measure. The underlying is assumed to be a single
            non-dividend-paying stock in an arbitrage-free, frictionless market with
            a constant continuously compounded risk-free rate. The simulated paths
            are pricing inputs and should not be interpreted as forecasts of future
            stock returns.
            """
        )

        st.markdown("**GBM under the risk-neutral measure**")
        st.latex(
            r"S_{t+\Delta t}=S_t\exp\left[(r-\tfrac12\sigma^2)\Delta t"
            r"+\sigma\sqrt{\Delta t}Z\right],\quad Z\sim N(0,1)"
        )
        st.markdown(
            r"""
            With constant volatility and interest rates, this exponential recursion
            is the exact GBM transition between selected grid points and preserves
            positive prices. More time steps therefore do not improve a European
            option's terminal GBM distribution; they matter for an American option
            because they provide a finer set of possible exercise dates.
            """
        )

        st.markdown("**European put valuation and Monte Carlo uncertainty**")
        st.latex(
            r"P_0^E=e^{-rT}\mathbb{E}^{\mathbb{Q}}[(K-S_T)^+],\qquad "
            r"\widehat P_{0,n}^E=\frac1n\sum_{i=1}^{n}"
            r"e^{-rT}(K-S_T^{(i)})^+"
        )
        st.latex(
            r"\widehat{\operatorname{SE}}(\widehat P_{0,n})="
            r"\frac{s_X}{\sqrt n},\qquad "
            r"\text{approximate 95\% interval}="
            r"\widehat P_{0,n}\pm1.96\widehat{\operatorname{SE}}(\widehat P_{0,n})"
        )
        st.markdown(
            r"""
            The standard error decreases at rate \(n^{-1/2}\), so halving it requires
            approximately four times as many independent paths. For GBM, the exact
            no-dividend Black–Scholes put value evaluates the same risk-neutral payoff
            analytically and is used only to validate the European Monte Carlo
            implementation. Agreement with Black–Scholes does not establish agreement
            with an observed market price.
            """
        )

        st.markdown("**American put valuation with LSMC**")
        st.latex(
            r"H_j(s)=(K-s)^+,\qquad C_j(s)="
            r"\mathbb{E}^{\mathbb{Q}}[e^{-r\Delta t}V_{j+1}(S_{t_{j+1}})"
            r"\mid S_{t_j}=s]"
        )
        st.latex(
            r"Y_i^{(j)}=e^{-r\Delta t}F_i^{(j+1)},\qquad "
            r"Y_i^{(j)}\approx\sum_k\beta_k^{(j)}\phi_k(S_{t_j}^{(i)})"
        )
        st.markdown(
            r"""
            LSMC starts from the terminal put payoff and works backwards through the
            exercise grid. At each date it fits the discounted future cash flows using
            only in-the-money paths, then exercises when immediate value exceeds the
            fitted continuation value. The default GBM basis is \(1,S,S^2\). The app
            also allows selected variance and price–variance terms when Heston is used.
            After the final backward step, pathwise cash flows are discounted to time
            zero and averaged; immediate exercise at time zero is retained when it is
            more valuable. The theoretical early-exercise right implies
            \(P_0^A\geq P_0^E\), although a finite-sample LSMC estimate can be affected
            by regression and sampling error.
            """
        )

        st.markdown("**Heston stochastic variance under the risk-neutral measure**")
        st.latex(
            r"dS_t=rS_tdt+\sqrt{v_t}S_tdW_t^S,\qquad "
            r"dv_t=\kappa(\theta-v_t)dt+\xi\sqrt{v_t}dW_t^v,\qquad "
            r"dW_t^S dW_t^v=\rho dt"
        )
        st.markdown(
            r"""
            Here \(v_t\) is instantaneous variance, \(\kappa\) is its mean-reversion
            speed, \(\theta\) its long-run mean, \(\xi\) the volatility of variance,
            and \(\rho\) the price–variance correlation. Following the dissertation's
            simplified risk-neutral specification, the variance-risk premium is set to
            zero, so the physical and risk-neutral variance parameters are not separated.
            This assumption does not make the two measures identical: the stock drift
            still changes from the physical expected return to \(r\).
            """
        )
        st.latex(
            r"v_t^+=\max(v_t,0),\quad \widetilde v_{t+\Delta t}="
            r"v_t^++\kappa(\theta-v_t^+)\Delta t+\xi\sqrt{v_t^+\Delta t}Z_v"
        )
        st.latex(
            r"v_{t+\Delta t}=\max(\widetilde v_{t+\Delta t},0),\qquad "
            r"S_{t+\Delta t}=S_t\exp\left[(r-\tfrac12v_t^+)\Delta t"
            r"+\sqrt{v_t^+\Delta t}Z_S\right]"
        )
        st.latex(
            r"Z_S=\rho Z_v+\sqrt{1-\rho^2}Z_\perp,\qquad "
            r"Z_v,Z_\perp\sim N(0,1)"
        )
        st.markdown(
            """
            The variance equation uses a non-negative full-truncation Euler step,
            while the asset price uses an exponential log-Euler step. Unlike the exact
            GBM transition, Heston simulation has time-discretisation error, so a finer
            grid should more closely approximate the continuous-time processes. The
            Feller condition concerns strict positivity of the continuous variance
            process; truncation prevents negative simulated variance when that condition
            is not satisfied. Market volatility supplies initial Heston defaults rather
            than a calibration of all Heston parameters to an option surface.
            """
        )

        st.markdown("**Interpreting the results**")
        st.markdown(
            """
            - Increasing paths primarily reduces Monte Carlo sampling uncertainty.
            - Increasing exercise steps refines the American stopping opportunity;
              under Heston it also reduces time-discretisation error.
            - The displayed confidence interval measures Monte Carlo sampling
              uncertainty only. It does not include LSMC regression uncertainty,
              model risk or parameter uncertainty.
            - Model–market differences may reflect the no-dividend and frictionless-
              market assumptions, volatility-surface effects, Heston calibration,
              jumps, liquidity, transaction costs, delayed quotes and bid–ask effects.
            - Yahoo's bid–ask midpoint falls back to the last traded price when a valid
              bid and ask are unavailable.
            """
        )
