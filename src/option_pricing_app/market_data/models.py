from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import pandas as pd


class MarketDataError(RuntimeError):
    """External data needed by the dashboard was unavailable or malformed."""


@dataclass(frozen=True)
class Quote:
    ticker: str
    price: float
    observed_at: datetime


@dataclass(frozen=True)
class OptionChain:
    ticker: str
    expiry: str
    puts: pd.DataFrame
    observed_at: datetime


@dataclass(frozen=True)
class PutMarketPrice:
    price: float
    strike: float
    basis: str
    observed_at: datetime


@dataclass(frozen=True)
class ImpliedVolatility:
    volatility: float
    requested_strike: float
    matched_strike: float

    @property
    def is_approximation(self) -> bool:
        return not np.isclose(self.requested_strike, self.matched_strike)


@dataclass(frozen=True)
class TreasuryCurve:
    as_of: date
    maturities: tuple[float, ...]
    rates: tuple[float, ...]

    def rate_for_maturity(self, maturity: float) -> tuple[float, bool]:
        if maturity <= 0:
            raise ValueError("maturity must be positive")
        value = float(np.interp(maturity, self.maturities, self.rates))
        outside = maturity < self.maturities[0] or maturity > self.maturities[-1]
        return value, outside
