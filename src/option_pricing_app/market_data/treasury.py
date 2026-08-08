import xml.etree.ElementTree as ET
from datetime import UTC, date, datetime

import httpx

from option_pricing_app.market_data.models import MarketDataError, TreasuryCurve

FEED_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "pages/xml?data=daily_treasury_yield_curve&field_tdr_date_value={year}"
)
TENORS = {
    "BC_1MONTH": 1 / 12,
    "BC_1_5MONTH": 1.5 / 12,
    "BC_2MONTH": 2 / 12,
    "BC_3MONTH": 3 / 12,
    "BC_4MONTH": 4 / 12,
    "BC_6MONTH": 0.5,
    "BC_1YEAR": 1.0,
    "BC_2YEAR": 2.0,
    "BC_3YEAR": 3.0,
    "BC_5YEAR": 5.0,
    "BC_7YEAR": 7.0,
    "BC_10YEAR": 10.0,
    "BC_20YEAR": 20.0,
    "BC_30YEAR": 30.0,
}


class TreasuryClient:
    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout

    def get_latest_curve(self, year: int | None = None) -> TreasuryCurve:
        try:
            response = httpx.get(
                FEED_URL.format(year=year or datetime.now(UTC).year),
                timeout=self.timeout,
                follow_redirects=True,
                headers={"User-Agent": "masters-option-pricing-poc/0.1"},
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise MarketDataError("Could not fetch the U.S. Treasury yield curve.") from exc
        return self.parse_curve(response.text)

    @staticmethod
    def parse_curve(xml_text: str) -> TreasuryCurve:
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            raise MarketDataError("The Treasury feed returned invalid XML.") from exc
        curves: list[TreasuryCurve] = []
        for properties in root.iter():
            if _local(properties.tag) != "properties":
                continue
            fields = {_local(child.tag): child.text for child in properties}
            try:
                as_of = date.fromisoformat((fields.get("NEW_DATE") or "")[:10])
            except ValueError:
                continue
            points = []
            for field, maturity in TENORS.items():
                try:
                    value = float(fields[field]) / 100.0
                except (KeyError, TypeError, ValueError):
                    continue
                points.append((maturity, value))
            if len(points) >= 2:
                points.sort()
                curves.append(
                    TreasuryCurve(
                        as_of,
                        tuple(point[0] for point in points),
                        tuple(point[1] for point in points),
                    )
                )
        if not curves:
            raise MarketDataError("The Treasury feed contained no usable yield curve.")
        return max(curves, key=lambda curve: curve.as_of)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]
