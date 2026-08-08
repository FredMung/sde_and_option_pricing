from datetime import date

import numpy as np
import pandas as pd
import pytest

from option_pricing_app.market_data import MarketDataError, OptionChain, TreasuryClient
from option_pricing_app.market_data.yahoo import YahooFinanceClient


def test_nearest_strike_iv_is_reported_as_approximation():
    chain = OptionChain(
        "TEST",
        "2030-01-01",
        pd.DataFrame(
            {"strike": [90.0, 100.0, 110.0], "impliedVolatility": [0.30, 0.20, 0.25]}
        ),
        pd.Timestamp("2026-01-01", tz="UTC").to_pydatetime(),
    )
    result = YahooFinanceClient.nearest_implied_volatility(chain, 104.0)
    assert result.volatility == pytest.approx(0.20)
    assert result.matched_strike == 100.0
    assert result.is_approximation


def test_historical_volatility_uses_log_returns(monkeypatch):
    prices = 100 * np.exp(np.linspace(0, 0.1, 60) + np.sin(np.arange(60)) * 0.01)
    index = pd.date_range("2026-01-01", periods=60, freq="B", tz="UTC")

    class FakeTicker:
        def history(self, **kwargs):
            return pd.DataFrame({"Close": prices}, index=index)

    monkeypatch.setattr("option_pricing_app.market_data.yahoo.yf.Ticker", lambda _: FakeTicker())
    actual, observed = YahooFinanceClient().get_historical_volatility("TEST")
    expected = np.log(pd.Series(prices) / pd.Series(prices).shift(1)).std() * np.sqrt(252)
    assert actual == pytest.approx(expected)
    assert observed.date() == index[-1].date()


def test_treasury_curve_parsing_and_interpolation():
    xml = """<feed xmlns:m="http://schemas.microsoft.com/ado/2007/08/dataservices/metadata"
      xmlns:d="http://schemas.microsoft.com/ado/2007/08/dataservices"><entry><content>
      <m:properties><d:NEW_DATE>2026-08-07T00:00:00</d:NEW_DATE>
      <d:BC_1YEAR>4.00</d:BC_1YEAR><d:BC_2YEAR>4.50</d:BC_2YEAR>
      </m:properties></content></entry></feed>"""
    curve = TreasuryClient.parse_curve(xml)
    rate, outside = curve.rate_for_maturity(1.5)
    assert curve.as_of == date(2026, 8, 7)
    assert rate == pytest.approx(0.0425)
    assert not outside


def test_empty_treasury_feed_fails_cleanly():
    with pytest.raises(MarketDataError, match="no usable"):
        TreasuryClient.parse_curve("<feed />")
