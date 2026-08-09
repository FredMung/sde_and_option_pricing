# SDE application in option pricing — Streamlit

> This application is an educational proof of concept developed at
> master’s-student level. Its purpose is to demonstrate the concepts behind
> stochastic asset-price models and option-pricing methods. It is not intended
> to provide production-grade valuations, trading signals, or investment advice.

This standalone portfolio application presents GBM and Heston risk-neutral asset
simulation with European Monte Carlo and American LSMC put pricing. Application-
specific validation, market data, and visualisation live outside the statistical
modules.

## What the application demonstrates

- Risk-neutral geometric Brownian motion path simulation.
- Heston stochastic-variance price and variance path simulation.
- European put pricing from discounted terminal Monte Carlo payoffs.
- American put pricing through Least-Squares Monte Carlo backward induction.
- Configurable LSMC price, variance, and price–variance cross-term basis functions.
- Live Yahoo Finance spot, expiry, strike, implied-volatility, and historical-price data.
- A U.S. Treasury par-yield proxy for the risk-free rate.
- Monte Carlo standard errors, approximate confidence intervals, and path/payoff plots.

The calculations are deliberately illustrative. Yahoo data may be delayed or
incomplete, Treasury par yields are not bootstrapped zero rates, and the model does
not account for dividends, a volatility surface, liquidity, or transaction costs.

## Structure

```text
streamlit_app.py                  # Hosted application entry point
src/option_pricing_app/
├── stats/
│   ├── asset_pricing_simulator.py   # GBM and Heston asset/variance paths
│   └── option_pricing_simulator.py  # LSMC option pricing and basis terms
├── market_data/                  # Yahoo and Treasury adapters
├── app/                          # Streamlit controls and Plotly figures
├── domain.py                     # Validated input/result objects
└── service.py                    # App-facing pricing orchestration
tests/                            # Numerical, data-adapter, and app tests
```

Visualisations sit on the app side. The statistical modules do not know about
Streamlit or Plotly, which keeps the mathematical work reusable.

## Run locally

Install Poetry, select Python 3.11, and install the project:

```bash
poetry env use python3.11
poetry install
poetry run streamlit run streamlit_app.py
```

Run the automated checks:

```bash
poetry run pytest
poetry run ruff check .
```

## Inputs and data behaviour

The dashboard is put-only and single-asset. Manual spot or manual maturity switches
all market assumptions to manual entry. With a live ticker and listed expiry, strike,
volatility, and interest-rate sources can be selected independently. Historical
volatility uses one year of daily log returns and an annualisation factor of
`sqrt(252)`. A manual strike with implied volatility uses the nearest listed put
contract and identifies the approximation.

For Heston, the selected market volatility supplies the default initial and
long-run volatility levels; these remain user-configurable model assumptions and
are not a full Heston calibration. The app displays instantaneous variance
(`volatility²`) paths and permits LSMC basis terms in price and variance through
power two, including their cross-products.

The U.S. Treasury daily par-yield curve is linearly interpolated to the option
maturity. This is displayed as an educational risk-free-rate proxy, not an exact
option-pricing zero rate.

## Deploy on Streamlit Community Cloud

1. Push this directory as its own GitHub repository.
2. Create a Streamlit Community Cloud app from that repository.
3. Choose `streamlit_app.py` as the entry point and Python 3.11 as the runtime.

The root `pyproject.toml` and `poetry.lock` declare all dependencies. No API keys are
required for the public Yahoo Finance and Treasury sources.

## Future work

Calls, dividends, a semi-analytical Heston benchmark, parameter calibration,
Greeks, and multi-asset payoffs are intentionally left for future contributors.
