from datetime import UTC, datetime

import numpy as np
import pandas as pd
import yfinance as yf

from option_pricing_app.market_data.models import (
    ImpliedVolatility,
    MarketDataError,
    OptionChain,
    Quote,
)


class YahooFinanceClient:
    """Fetch the small Yahoo Finance subset needed by the application."""

    def get_spot(self, ticker: str) -> Quote:
        symbol = _ticker(ticker)
        instrument = yf.Ticker(symbol)
        try:
            price = float(instrument.fast_info["lastPrice"])
        except (KeyError, TypeError, ValueError, AttributeError):
            price = float("nan")
        if not np.isfinite(price) or price <= 0:
            try:
                history = instrument.history(period="5d", auto_adjust=True)
                price = float(_close(history).dropna().iloc[-1])
            except (IndexError, KeyError, TypeError, ValueError) as exc:
                raise MarketDataError(f"No recent price was available for {symbol}.") from exc
        return Quote(symbol, price, datetime.now(UTC))

    def get_expiries(self, ticker: str) -> tuple[str, ...]:
        symbol = _ticker(ticker)
        try:
            expiries = tuple(yf.Ticker(symbol).options)
        except Exception as exc:
            raise MarketDataError(f"Could not fetch option expiries for {symbol}.") from exc
        if not expiries:
            raise MarketDataError(f"Yahoo Finance returned no put expiries for {symbol}.")
        return expiries

    def get_put_chain(self, ticker: str, expiry: str) -> OptionChain:
        symbol = _ticker(ticker)
        try:
            puts = yf.Ticker(symbol).option_chain(expiry).puts.copy()
        except Exception as exc:
            raise MarketDataError(f"Could not fetch the {expiry} put chain for {symbol}.") from exc
        required = {"strike", "impliedVolatility"}
        if puts.empty or not required.issubset(puts.columns):
            raise MarketDataError(f"Yahoo returned an incomplete put chain for {symbol}.")
        return OptionChain(symbol, expiry, puts, datetime.now(UTC))

    def get_historical_volatility(self, ticker: str) -> tuple[float, datetime]:
        symbol = _ticker(ticker)
        try:
            close = _close(
                yf.Ticker(symbol).history(period="1y", auto_adjust=True)
            ).dropna()
        except Exception as exc:
            raise MarketDataError(f"Could not fetch price history for {symbol}.") from exc
        if len(close) < 30:
            raise MarketDataError("At least 30 closes are required for historical volatility.")
        returns = np.log(close / close.shift(1)).dropna()
        volatility = float(returns.std(ddof=1) * np.sqrt(252.0))
        if not np.isfinite(volatility) or volatility <= 0:
            raise MarketDataError("Historical volatility was invalid.")
        observed = pd.Timestamp(close.index[-1])
        if observed.tzinfo is None:
            observed = observed.tz_localize("UTC")
        return volatility, observed.to_pydatetime()

    @staticmethod
    def listed_strikes(chain: OptionChain) -> tuple[float, ...]:
        strikes = pd.to_numeric(chain.puts["strike"], errors="coerce").dropna()
        values = tuple(sorted(float(value) for value in strikes.unique() if value > 0))
        if not values:
            raise MarketDataError("The put chain contained no valid strikes.")
        return values

    @staticmethod
    def nearest_implied_volatility(
        chain: OptionChain, strike: float
    ) -> ImpliedVolatility:
        frame = chain.puts[["strike", "impliedVolatility"]].copy()
        frame = frame.apply(pd.to_numeric, errors="coerce")
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
        frame = frame[(frame.strike > 0) & (frame.impliedVolatility > 0)]
        if frame.empty:
            raise MarketDataError("The put chain contained no valid implied volatilities.")
        index = (frame.strike - strike).abs().idxmin()
        row = frame.loc[index]
        return ImpliedVolatility(float(row.impliedVolatility), float(strike), float(row.strike))


def _ticker(value: str) -> str:
    symbol = value.strip().upper()
    if not symbol:
        raise ValueError("ticker cannot be empty")
    return symbol


def _close(history: pd.DataFrame) -> pd.Series:
    if history.empty or "Close" not in history:
        raise MarketDataError("Yahoo Finance returned no closing prices.")
    return history["Close"]
