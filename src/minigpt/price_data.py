"""Dutch day-ahead electricity prices: fetch, hourly aggregation, daily table, windows.

Source: Energy-Charts API (Fraunhofer ISE), `/price?bzn=NL`, licence CC BY 4.0 (the response's
`license_info` field credits Bundesnetzagentur | SMARD.de). No token is needed.

Conventions (all explicit, all tested):

* A timestamp marks the *start* of the delivery period it refers to (API documentation).
* Since 2025-10-01 the day-ahead market clears every 15 minutes. Those prices are averaged to
  hourly, and an hour is kept only if all four quarter-hours are present. Before the cutover an
  hour is kept only if its one value is present. Partial hours become gaps.
* Days are Europe/Amsterdam calendar days with 24 local hours. On the 25-hour fall-back day the
  two 02:00 hours are averaged; on the 23-hour spring-forward day the missing 02:00 is imputed as
  the mean of 01:00 and 03:00. Both are counted in `DayTable.dst_adjusted`.
* Any hour still missing leaves NaN in the day table. A forecasting window is built only if every
  value it uses (history and target) is present. Nothing else is interpolated.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

API_URL = "https://api.energy-charts.info/price"
BZN = "NL"
TZ = ZoneInfo("Europe/Amsterdam")
RAW_SUBDIR = "nl_dayahead"
DEFAULT_START = date(2024, 10, 1)
DEFAULT_END = date(2026, 10, 15)
CHUNK_DAYS = 90
LOOKBACK_DAYS = 7
HOURS = 24
CITATION = (
    "Energy-Charts API, Fraunhofer ISE (https://api.energy-charts.info), day-ahead prices for "
    "bidding zone NL, CC BY 4.0, data from Bundesnetzagentur | SMARD.de"
)


# --------------------------------------------------------------------------- fetching


def chunk_ranges(start: date, end: date, chunk_days: int = CHUNK_DAYS) -> list[tuple[date, date]]:
    """Split [start, end] (inclusive) into consecutive windows of at most `chunk_days` days."""
    if end < start:
        raise ValueError(f"end {end} is before start {start}")
    out: list[tuple[date, date]] = []
    s = start
    while s <= end:
        e = min(s + timedelta(days=chunk_days - 1), end)
        out.append((s, e))
        s = e + timedelta(days=1)
    return out


def _get_json(url: str, retries: int = 8) -> dict[str, Any]:
    """GET JSON; on HTTP 429 sleep for `Retry-After` (the /price limit is 2 requests/minute)."""
    for _ in range(retries):
        req = urllib.request.Request(url, headers={"User-Agent": "minigpt-price-forecast"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload: dict[str, Any] = json.loads(resp.read())
                return payload
        except urllib.error.HTTPError as err:
            if err.code != 429:
                raise
            wait = int(err.headers.get("Retry-After", "30"))
            print(f"rate limited, sleeping {wait + 1}s")
            time.sleep(wait + 1)
    raise RuntimeError(f"still rate limited after {retries} attempts: {url}")


def fetch_prices(
    start: date = DEFAULT_START, end: date = DEFAULT_END, raw_dir: Path = Path("data/raw")
) -> Path:
    """Download NL day-ahead prices for [start, end] into `raw_dir/nl_dayahead/`.

    One JSON file per chunk is cached, so a re-run only fetches missing chunks. Returns the
    directory. Raises if `end` is today or later, because that day is not complete.
    """
    if end >= datetime.now(TZ).date():
        raise ValueError(f"end {end} must be before today: the current day is incomplete")
    out_dir = raw_dir / RAW_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[str] = []
    license_info = ""
    for s, e in chunk_ranges(start, end):
        path = out_dir / f"price_{s.isoformat()}_{e.isoformat()}.json"
        if not path.exists():
            url = f"{API_URL}?bzn={BZN}&start={s.isoformat()}&end={e.isoformat()}"
            payload = _get_json(url)
            path.write_text(json.dumps(payload), encoding="utf-8")
            print(f"fetched {path.name}: {len(payload['unix_seconds'])} points")
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
        license_info = str(payload.get("license_info", license_info))
        chunks.append(path.name)
    manifest = {
        "source": API_URL,
        "bzn": BZN,
        "unit": "EUR/MWh",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "fetched_on": datetime.now(UTC).date().isoformat(),
        "license_info": license_info,
        "citation": CITATION,
        "chunks": chunks,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_dir


def read_manifest(raw_dir: Path = Path("data/raw")) -> dict[str, Any]:
    path = raw_dir / RAW_SUBDIR / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `minigpt price-data` first")
    manifest: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return manifest


# --------------------------------------------------------------------------- parsing


@dataclass
class HourlySeries:
    """Hourly prices; `hours` are unix seconds of hour starts, strictly increasing."""

    hours: np.ndarray
    price: np.ndarray
    partial_hours_dropped: int
    quarter_hour_from: int | None  # unix start of the first 15-minute hour, if any


def parse_payload(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """API payload -> (unix seconds int64, price float64 with NaN for null)."""
    unix = np.asarray(payload["unix_seconds"], dtype=np.int64)
    price = np.asarray([np.nan if p is None else p for p in payload["price"]], dtype=np.float64)
    if unix.shape != price.shape:
        raise ValueError("unix_seconds and price differ in length")
    return unix, price


def to_hourly(unix: np.ndarray, price: np.ndarray) -> HourlySeries:
    """Average to hourly. Hours need 1 value before the 15-minute cutover, 4 after."""
    uniq, first = np.unique(unix, return_index=True)  # sorts and drops duplicate timestamps
    price = price[first]
    unix = uniq
    off_hour = unix[unix % 3600 != 0]
    cutover = int(off_hour[0] // 3600 * 3600) if off_hour.size else None
    hour_of = unix // 3600 * 3600
    hours, inverse = np.unique(hour_of, return_inverse=True)
    finite = np.isfinite(price)
    count = np.bincount(inverse, weights=finite.astype(np.float64), minlength=hours.size)
    total = np.bincount(inverse, weights=np.where(finite, price, 0.0), minlength=hours.size)
    expected = np.ones(hours.size)
    if cutover is not None:
        expected[hours >= cutover] = 4.0
    keep = count == expected
    return HourlySeries(
        hours=hours[keep],
        price=total[keep] / count[keep],
        partial_hours_dropped=int((~keep).sum()),
        quarter_hour_from=cutover,
    )


def load_hourly(raw_dir: Path = Path("data/raw")) -> HourlySeries:
    """Read every cached chunk listed in the manifest and aggregate to hourly."""
    manifest = read_manifest(raw_dir)
    parts = [
        parse_payload(json.loads((raw_dir / RAW_SUBDIR / name).read_text(encoding="utf-8")))
        for name in manifest["chunks"]
    ]
    return to_hourly(np.concatenate([u for u, _ in parts]), np.concatenate([p for _, p in parts]))


# --------------------------------------------------------------------------- daily table


@dataclass
class DayTable:
    """One row per local calendar day, 24 local hours; NaN marks a missing hour."""

    dates: list[date]
    prices: np.ndarray  # (n_days, 24), EUR/MWh
    dst_adjusted: int  # hours imputed (spring) or averaged (fall) because of DST


def daily_table(series: HourlySeries, first_day: date, last_day: date) -> DayTable:
    """Reshape hourly UTC data into local-day rows (see module docstring for DST rules)."""
    lookup = dict(zip(series.hours.tolist(), series.price.tolist(), strict=True))
    dates: list[date] = []
    rows: list[list[float]] = []
    adjusted = 0
    d = first_day
    while d <= last_day:
        row = [np.nan] * HOURS
        spring_hole: int | None = None
        for h in range(HOURS):
            naive = datetime(d.year, d.month, d.day, h)
            first = naive.replace(tzinfo=TZ, fold=0)
            second = naive.replace(tzinfo=TZ, fold=1)
            roundtrip = first.astimezone(UTC).astimezone(TZ).replace(tzinfo=None)
            if roundtrip != naive:  # local time does not exist (spring forward)
                spring_hole = h
                continue
            t0, t1 = int(first.timestamp()), int(second.timestamp())
            v0 = lookup.get(t0, np.nan)
            if t0 == t1:
                row[h] = v0
            else:  # local time occurs twice (fall back): average both
                row[h] = (v0 + lookup.get(t1, np.nan)) / 2.0
                adjusted += 1
        if spring_hole is not None:
            row[spring_hole] = (row[spring_hole - 1] + row[spring_hole + 1]) / 2.0
            adjusted += 1
        dates.append(d)
        rows.append(row)
        d += timedelta(days=1)
    return DayTable(dates=dates, prices=np.asarray(rows, dtype=np.float64), dst_adjusted=adjusted)


# --------------------------------------------------------------------------- windows


@dataclass
class Windows:
    """Forecast samples: `x` is the 7 days before the target day, `y` the target day.

    For target day D (local), `x[i]` holds hours of D-7 ... D-1 (168 values) and nothing at or
    after D 00:00. The forecast is issued on D-1 at 12:00; every hour of D-1 is already public
    then (the D-1 auction cleared on D-2 around 13:00), so the full previous day is allowed.
    """

    dates: list[date]
    x: np.ndarray  # (N, 24 * lookback)
    y: np.ndarray  # (N, 24)

    def __len__(self) -> int:
        return len(self.dates)

    def subset(self, keep: np.ndarray) -> Windows:
        idx = np.flatnonzero(keep)
        return Windows([self.dates[i] for i in idx], self.x[idx], self.y[idx])


def make_windows(table: DayTable, lookback: int = LOOKBACK_DAYS) -> Windows:
    """Build windows for every target day whose history and target have no missing hour."""
    dates: list[date] = []
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for i in range(lookback, len(table.dates)):
        x = table.prices[i - lookback : i].reshape(-1)
        y = table.prices[i]
        if np.isnan(x).any() or np.isnan(y).any():
            continue
        dates.append(table.dates[i])
        xs.append(x)
        ys.append(y)
    n_in = HOURS * lookback
    return Windows(
        dates,
        np.asarray(xs, dtype=np.float64).reshape(len(dates), n_in),
        np.asarray(ys, dtype=np.float64).reshape(len(dates), HOURS),
    )


def parse_month(text: str) -> tuple[int, int]:
    year, month = text.split("-")
    if not 1 <= int(month) <= 12:
        raise ValueError(f"bad month {text!r}, expected YYYY-MM")
    return int(year), int(month)


def month_bounds(year: int, month: int) -> tuple[date, date]:
    """First and last day of a calendar month."""
    first = date(year, month, 1)
    nxt = date(year + (month == 12), month % 12 + 1, 1)
    return first, nxt - timedelta(days=1)


def last_full_month(last_day: date) -> tuple[int, int]:
    """The latest calendar month that ends on or before `last_day`."""
    _, month_end = month_bounds(last_day.year, last_day.month)
    if last_day == month_end:
        return last_day.year, last_day.month
    prev = date(last_day.year, last_day.month, 1) - timedelta(days=1)
    return prev.year, prev.month


@dataclass
class Split:
    train: Windows
    val: Windows
    test: Windows


def split_windows(windows: Windows, test_month: tuple[int, int], val_days: int = 28) -> Split:
    """Chronological split: test = `test_month`, val = the `val_days` before it, train = earlier.

    Test windows may use val/train days as *history* (they are in the past); no target of a
    later split is ever a training or validation target.
    """
    first, last = month_bounds(*test_month)
    val_start = first - timedelta(days=val_days)
    in_test = np.asarray([first <= d <= last for d in windows.dates])
    in_val = np.asarray([val_start <= d < first for d in windows.dates])
    in_train = np.asarray([d < val_start for d in windows.dates])
    return Split(windows.subset(in_train), windows.subset(in_val), windows.subset(in_test))


def fold_months(last_month: tuple[int, int], n: int) -> list[tuple[int, int]]:
    """The `n` consecutive calendar months ending at `last_month`, oldest first."""
    months: list[tuple[int, int]] = []
    year, month = last_month
    for _ in range(n):
        months.append((year, month))
        year, month = (year - 1, 12) if month == 1 else (year, month - 1)
    return months[::-1]


@dataclass
class PriceData:
    windows: Windows
    table: DayTable
    manifest: dict[str, Any]
    cutover: date | None  # first local day of 15-minute source data, if present
    partial_hours_dropped: int

    @property
    def last_day(self) -> date:
        return date.fromisoformat(self.manifest["end"])


def load_price_data(raw_dir: Path = Path("data/raw")) -> PriceData:
    """Cached chunks -> hourly -> local-day table -> forecast windows."""
    manifest = read_manifest(raw_dir)
    series = load_hourly(raw_dir)
    first, last = date.fromisoformat(manifest["start"]), date.fromisoformat(manifest["end"])
    table = daily_table(series, first, last)
    cutover = (
        datetime.fromtimestamp(series.quarter_hour_from, TZ).date()
        if series.quarter_hour_from is not None
        else None
    )
    return PriceData(make_windows(table), table, manifest, cutover, series.partial_hours_dropped)
