"""Streamlit controls, orchestration, explanations, and result rendering."""

from dataclasses import dataclass
from datetime import UTC, date, datetime

import pandas as pd
import streamlit as st

from option_pricing_app.app.charts import (
    cash_flow_figure,
    convergence_figure,
    exercise_figure,
    paths_figure,
    payoff_figure,
    terminal_figure,
    variance_paths_figure,
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
from option_pricing_app.market_data import (
    MarketDataError,
    OptionChain,
    Quote,
    TreasuryClient,
    TreasuryCurve,
    YahooFinanceClient,
)
from option_pricing_app.service import price_put

DISCLAIMER = (
    "This application is an educational proof of concept. Its purpose is to demonstrate the concepts behind "
    "stochastic asset-price models and option-pricing methods. It is not intended "
    "to provide production-grade valuations, trading signals, or investment advice."
)
MANUAL_MATURITY = "Manual maturity"
MIN_PLAUSIBLE_IMPLIED_VOLATILITY = 0.05
MAX_PLAUSIBLE_IMPLIED_VOLATILITY = 3.00


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
        except (ValueError, FloatingPointError) as exc:
            st.error(f"The simulation could not be completed: {exc}")

    output = st.session_state.get("pricing_output")
    if output:
        _results(*output)
    else:
        st.info("Choose assumptions in the sidebar, then generate a simulation.")
    _methodology()


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
    n_paths = int(
        sidebar.number_input("Number of paths", 100, 25_000, 1_000, step=500)
    )
    n_steps = int(sidebar.number_input("Number of time steps", 1, 1_000, 100, step=10))
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


def _results(result: PricingResult, inputs: DashboardInputs) -> None:
    st.subheader("Simulation results")
    st.caption("Results reflect the inputs captured when Generate simulation was pressed.")
    low, high = result.confidence_interval
    market = inputs.market_put_price
    known_facts, separator, simulation_metrics = st.columns([1, 0.04, 1])

    with known_facts:
        st.markdown("#### Known facts")
        st.metric(
            "Market put value",
            "N/A" if market is None else f"{market:,.4f}",
            delta=None if market is None else f"Model − market: {result.price - market:+,.4f}",
            delta_color="off",
        )
        st.metric(inputs.volatility_label, f"{inputs.market.volatility:.2%}")
        st.metric(inputs.risk_free_rate_label, f"{inputs.market.risk_free_rate:.2%}")

    with separator:
        st.markdown(
            '<div style="border-left: 1px solid rgba(128, 128, 128, 0.45); '
            'height: 25rem; margin: 0 auto;"></div>',
            unsafe_allow_html=True,
        )

    with simulation_metrics:
        st.markdown("#### Simulation metrics")
        if inputs.exercise_style is ExerciseStyle.AMERICAN:
            st.metric("Estimated American put value", f"{result.price:,.4f}")
            st.metric(
                "Estimated European put value",
                f"{result.european_mc_price:,.4f}",
                delta=f"American − European: {result.price - result.european_mc_price:+,.4f}",
                delta_color="off",
            )
        else:
            st.metric("Estimated European put value", f"{result.price:,.4f}")
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
        st.metric("Monte Carlo standard error", f"{result.standard_error:,.4f}")
        st.metric("Approximate 95% interval", f"[{low:,.4f}, {high:,.4f}]")

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
            "LSMC early-exercise feature under the same simulated paths."
        )
    st.caption(f"Market reference: {inputs.market_put_price_source}")
    st.plotly_chart(convergence_figure(result), use_container_width=True)
    st.caption(
        "The line is the cumulative mean of discounted pathwise payoffs; the band is "
        "the approximate 95% Monte Carlo confidence interval at each path count."
    )
    if result.exercise_percentages is not None:
        st.plotly_chart(exercise_figure(result), use_container_width=True)
        st.caption(
            "Each path is counted once at its earliest selected exercise time; paths "
            "that expire out of the money are not counted. Being in the money does "
            "not automatically trigger exercise because continuation may be more "
            "valuable. These percentages are risk-neutral model estimates, not a "
            "forecast of how many investors will exercise."
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


def _methodology() -> None:
    with st.expander("Methodology and interpretation"):
        st.markdown("**GBM under the risk-neutral measure**")
        st.latex(
            r"S_{t+\Delta t}=S_t\exp\left[(r-\tfrac12\sigma^2)\Delta t"
            r"+\sigma\sqrt{\Delta t}Z\right],\quad Z\sim N(0,1)"
        )
        st.markdown("**Heston stochastic variance under the risk-neutral measure**")
        st.latex(
            r"dS_t=rS_tdt+\sqrt{v_t}S_tdW_t^S,\qquad "
            r"dv_t=\kappa(\theta-v_t)dt+\xi\sqrt{v_t}dW_t^v,\qquad "
            r"dW_t^S dW_t^v=\rho dt"
        )
        st.markdown(
            """
            A European put is the average discounted terminal payoff. For an
            American put, LSMC works backwards through possible exercise dates,
            estimates continuation values by regression, and compares continuation
            with immediate exercise. Under Heston, continuation regression may
            include price, variance, their squares, and selected cross-products.
            The exact European reference is shown only for GBM and uses the
            no-dividend Black–Scholes put formula with the same model inputs.

            Heston variance is simulated with a non-negative full-truncation
            Euler step, while the asset uses an exponential log-Euler step. The
            market volatility supplies initial defaults rather than a calibrated
            Heston parameter set.

            The interval shown measures Monte Carlo sampling uncertainty only. It
            excludes model risk, dividends, volatility-surface effects, transaction
            costs, liquidity, and calibration error. Yahoo quotes may be delayed;
            the displayed midpoint falls back to the last trade when a valid bid
            and ask are unavailable.
            """
        )
