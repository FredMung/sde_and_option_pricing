"""Streamlit controls, orchestration, explanations, and result rendering."""

from dataclasses import dataclass
from datetime import UTC, date, datetime

import pandas as pd
import streamlit as st

from option_pricing_app.app.charts import (
    convergence_figure,
    exercise_figure,
    paths_figure,
    payoff_figure,
    terminal_figure,
)
from option_pricing_app.domain import (
    ExerciseStyle,
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


@dataclass(frozen=True)
class DashboardInputs:
    contract: PutContract
    market: MarketInputs
    simulation: SimulationConfig
    exercise_style: ExerciseStyle
    model: str
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
    st.caption("Single-asset put pricing with GBM, Monte Carlo, and LSMC")
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

    model = sidebar.selectbox("Asset-price model", ["GBM"])
    exercise_style = ExerciseStyle(
        sidebar.selectbox("Exercise style", [style.value for style in ExerciseStyle])
    )
    spot_source = sidebar.radio(
        "Initial price source",
        ["Manual price", "Live ticker"],
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
        volatility, volatility_source = _live_volatility(chain, ticker, strike)
        rate, rate_source = _live_rate(maturity)
        sources = (
            ("Spot", f"Yahoo Finance; retrieved {quote.observed_at:%Y-%m-%d %H:%M UTC}"),
            ("Maturity", f"Yahoo listed expiry {expiry}; ACT/365.25"),
            ("Strike", strike_source),
            ("Volatility", volatility_source),
            ("Risk-free rate", rate_source),
        )

    sidebar.divider()
    sidebar.subheader("Simulation")
    n_paths = int(
        sidebar.number_input("Number of paths", 100, 25_000, 5_000, step=500)
    )
    n_steps = int(sidebar.number_input("Number of time steps", 1, 500, 100, step=10))
    with sidebar.expander("Advanced settings"):
        fixed_seed = st.checkbox("Use a reproducible random seed", value=True)
        seed = int(st.number_input("Random seed", 0, value=42)) if fixed_seed else None

    try:
        return DashboardInputs(
            PutContract(strike, maturity),
            MarketInputs(spot, volatility, rate),
            SimulationConfig(n_paths, n_steps, seed, min(100, n_paths)),
            exercise_style,
            model,
            sources,
        )
    except ValueError as exc:
        sidebar.error(str(exc))
        return None


def _live_strike(chain: OptionChain, spot: float) -> tuple[float, str]:
    strikes = YahooFinanceClient.listed_strikes(chain)
    source = st.sidebar.radio("Strike source", ["Listed strike", "Manual strike"])
    if source == "Listed strike":
        index = min(range(len(strikes)), key=lambda position: abs(strikes[position] - spot))
        return float(st.sidebar.selectbox("Strike, K", strikes, index=index)), "Yahoo put chain"
    return float(st.sidebar.number_input("Strike, K", 0.01, value=round(spot, 2))), "Manual"


def _live_volatility(chain: OptionChain, ticker: str, strike: float) -> tuple[float, str]:
    source = st.sidebar.selectbox(
        "Volatility source", ["Implied volatility", "Historical volatility", "Manual volatility"]
    )
    if source == "Manual volatility":
        return float(st.sidebar.number_input("Volatility (%)", 0.01, 500.0, 20.0)) / 100, "Manual"
    try:
        if source == "Historical volatility":
            value, observed = cached_historical_volatility(ticker)
            st.sidebar.caption(f"1-year annualised volatility: {value:.2%}")
            return value, f"Yahoo adjusted closes through {observed:%Y-%m-%d}; √252"
        estimate = YahooFinanceClient.nearest_implied_volatility(chain, strike)
        st.sidebar.caption(f"Put-chain implied volatility: {estimate.volatility:.2%}")
        if estimate.is_approximation:
            st.sidebar.warning(
                f"Using IV from nearest listed strike K={estimate.matched_strike:,.2f}."
            )
        return estimate.volatility, f"Yahoo put IV at K={estimate.matched_strike:,.2f}"
    except MarketDataError as exc:
        st.sidebar.warning(f"{exc} Enter volatility manually.")
        return float(st.sidebar.number_input("Fallback volatility (%)", 0.01, 500.0, 20.0)) / 100, "Manual fallback"


def _live_rate(maturity: float) -> tuple[float, str]:
    source = st.sidebar.radio("Risk-free-rate source", ["U.S. Treasury curve", "Manual rate"])
    if source == "Manual rate":
        return float(st.sidebar.number_input("Risk-free rate (%)", -99.0, 99.0, 4.0)) / 100, "Manual"
    try:
        curve = cached_treasury_curve()
        value, outside = curve.rate_for_maturity(maturity)
        st.sidebar.caption(f"Treasury par-yield proxy: {value:.2%} ({curve.as_of})")
        if outside:
            st.sidebar.warning("Maturity is outside the curve; nearest boundary used.")
        return value, f"U.S. Treasury par curve dated {curve.as_of}; interpolated proxy"
    except MarketDataError as exc:
        st.sidebar.warning(f"{exc} Enter a rate manually.")
        return float(st.sidebar.number_input("Fallback risk-free rate (%)", -99.0, 99.0, 4.0)) / 100, "Manual fallback"


def _manual_defaults(manual: bool, quote, expiry, chain) -> None:
    previous = st.session_state.get("_manual_mode", manual)
    if manual and not previous:
        spot = quote.price if quote else st.session_state.get("last_live_spot", 100.0)
        st.session_state.update(
            manual_spot=float(spot),
            manual_maturity=float(_maturity(expiry) if expiry and expiry != MANUAL_MATURITY else 1.0),
            manual_strike=float(spot),
        )
    st.session_state.setdefault("manual_spot", 100.0)
    st.session_state.setdefault("manual_maturity", 1.0)
    st.session_state.setdefault("manual_strike", 100.0)
    st.session_state.setdefault("manual_volatility", 20.0)
    st.session_state.setdefault("manual_rate", 4.0)
    st.session_state["_manual_mode"] = manual


def determine_manual_mode(spot_source: str, expiry: str | None, error: str | None) -> bool:
    return spot_source == "Manual price" or expiry == MANUAL_MATURITY or error is not None


def _maturity(expiry: str) -> float:
    days = (date.fromisoformat(expiry) - datetime.now(UTC).date()).days
    if days <= 0:
        raise ValueError("expiry must be in the future")
    return days / 365.25


def _results(result: PricingResult, inputs: DashboardInputs) -> None:
    st.subheader("Simulation results")
    st.caption("Results reflect the inputs captured when Generate simulation was pressed.")
    columns = st.columns(3)
    columns[0].metric("Estimated put value", f"{result.price:,.4f}")
    columns[1].metric("Monte Carlo standard error", f"{result.standard_error:,.4f}")
    low, high = result.confidence_interval
    columns[2].metric("Approximate 95% interval", f"[{low:,.4f}, {high:,.4f}]")
    st.plotly_chart(convergence_figure(result), use_container_width=True)
    st.caption(
        "The line is the cumulative mean of discounted pathwise payoffs; the band is "
        "the approximate 95% Monte Carlo confidence interval at each path count."
    )
    if result.exercise_percentages is not None:
        st.plotly_chart(exercise_figure(result), use_container_width=True)
        st.caption(
            "Each path is counted once at its final LSMC stopping time. Paths that "
            "expire out of the money are not counted as exercised."
        )
    st.plotly_chart(paths_figure(result), use_container_width=True)
    left, right = st.columns(2)
    left.plotly_chart(terminal_figure(result, inputs.contract), use_container_width=True)
    right.plotly_chart(payoff_figure(result, inputs.contract), use_container_width=True)

    rows = [
        ("Exercise style", inputs.exercise_style.value),
        ("Model / method", f"{inputs.model} / {result.pricing_method}"),
        ("Initial price", f"{inputs.market.spot:,.4f}"),
        ("Strike", f"{inputs.contract.strike:,.4f}"),
        ("Maturity", f"{inputs.contract.maturity:.6f} years"),
        ("Volatility", f"{inputs.market.volatility:.4%}"),
        ("Risk-free rate", f"{inputs.market.risk_free_rate:.4%}"),
        ("Paths / steps", f"{inputs.simulation.n_paths:,} / {inputs.simulation.n_steps:,}"),
        ("Seed", str(inputs.simulation.seed)),
        *inputs.sources,
    ]
    st.subheader("Inputs and provenance")
    st.dataframe(pd.DataFrame(rows, columns=["Input", "Value or source"]), hide_index=True, use_container_width=True)


def _methodology() -> None:
    with st.expander("Methodology and interpretation"):
        st.markdown("**GBM under the risk-neutral measure**")
        st.latex(
            r"S_{t+\Delta t}=S_t\exp\left[(r-\tfrac12\sigma^2)\Delta t"
            r"+\sigma\sqrt{\Delta t}Z\right],\quad Z\sim N(0,1)"
        )
        st.markdown(
            """
            A European put is the average discounted terminal payoff. For an
            American put, LSMC works backwards through possible exercise dates,
            estimates continuation values by regression, and compares continuation
            with immediate exercise.

            The interval shown measures Monte Carlo sampling uncertainty only. It
            excludes model risk, dividends, volatility-surface effects, transaction
            costs, liquidity, and calibration error.
            """
        )
