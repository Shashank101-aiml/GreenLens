"""Fetch daily adjusted closes for the ESG universe plus SPY, and FRED's 3-month T-bill rate.

Usage (from backend/): python -m scripts.fetch_market_data [--start 2019-01-01] [--end 2026-09-01]
"""
import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

from core.config import settings

log = logging.getLogger("fetch_market_data")

BENCHMARK = "SPY"
FRED_SERIES = "DGS3MO"
FRED_URL = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={FRED_SERIES}"
DEFAULT_START = "2019-01-01"
DEFAULT_END = "2026-09-01"
TRADING_DAYS = 252
EXTREME_MOVE = 0.5
STALE_RUN_DAYS = 5
# Yahoo serves history only under a company's current ticker; keys are symbols as written in the ESG file.
RENAMED = {"BK": "BNY", "FI": "FISV", "MMC": "MRSH", "PARA": "PSKY"}
ISSUES = ("low_coverage", "starts_late", "ends_early", "internal_gaps", "extreme_move", "stale", "non_positive", "no_data")


def to_yahoo_symbol(symbol: str) -> str:
    return symbol.strip().upper().replace(".", "-")


def load_universe(path: str) -> dict[str, str]:
    """Map Yahoo symbol -> symbol as written in the ESG file (e.g. BF-B -> BF.B, BNY -> BK)."""
    symbols = pd.read_csv(path, usecols=["Symbol"])["Symbol"].dropna().astype(str).str.strip().str.upper()
    mapping = {to_yahoo_symbol(RENAMED.get(s, s)): s for s in sorted(set(symbols))}
    mapping[BENCHMARK] = BENCHMARK
    return mapping


def _download_batch(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False, threads=True)
    if raw is None or raw.empty:
        return pd.DataFrame()
    close = raw["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    return close.dropna(axis=1, how="all")


def download_prices(
    tickers: list[str], start: str, end: str, batch_size: int, retries: int, pause: float
) -> tuple[pd.DataFrame, list[str]]:
    frames = []
    pending = list(tickers)
    for attempt in range(retries + 1):
        if attempt:
            wait = pause * 5 * attempt
            log.info("Retry round %d/%d: %d tickers after %.0fs", attempt, retries, len(pending), wait)
            time.sleep(wait)
        missing = []
        n_batches = -(-len(pending) // batch_size)
        for b, i in enumerate(range(0, len(pending), batch_size), start=1):
            batch = pending[i : i + batch_size]
            try:
                close = _download_batch(batch, start, end)
            except Exception as exc:  # network boundary: treat the whole batch as missing and retry it
                log.warning("Batch %d/%d failed: %s", b, n_batches, exc)
                close = pd.DataFrame()
            got = [t for t in batch if t in close.columns]
            missing += [t for t in batch if t not in close.columns]
            if got:
                frames.append(close[got])
            log.info("Batch %d/%d: %d/%d tickers", b, n_batches, len(got), len(batch))
            time.sleep(pause)
        pending = missing
        if not pending:
            break

    if not frames:
        return pd.DataFrame(), pending
    prices = pd.concat(frames, axis=1)
    idx = pd.DatetimeIndex(prices.index)
    prices.index = idx.tz_localize(None) if idx.tz is not None else idx
    return prices.sort_index(), pending


def quality_report(prices: pd.DataFrame, min_coverage: float) -> pd.DataFrame:
    calendar = prices.index
    head = calendar[min(5, len(calendar) - 1)]
    tail = calendar[max(len(calendar) - 6, 0)]
    rows = []
    for ticker in prices.columns:
        s = prices[ticker]
        first, last = s.first_valid_index(), s.last_valid_index()
        if first is None:
            rows.append({"ticker": ticker, "issues": "no_data"})
            continue
        span = s.loc[first:last]
        valid = span.dropna()
        rets = valid / valid.shift(1) - 1
        unchanged = valid.diff().eq(0)
        row = {
            "ticker": ticker,
            "first_date": first.date(),
            "last_date": last.date(),
            "coverage": round(float(s.notna().mean()), 4),
            "internal_missing": int(span.isna().sum()),
            "max_abs_return": round(float(rets.abs().max()), 4) if len(valid) > 1 else 0.0,
            "extreme_moves": int((rets.abs() > EXTREME_MOVE).sum()),
            "longest_flat_run": int(unchanged.groupby((~unchanged).cumsum()).sum().max()),
            "non_positive": int((valid <= 0).sum()),
        }
        checks = {
            "low_coverage": row["coverage"] < min_coverage,
            "starts_late": first > head,
            "ends_early": last < tail,
            "internal_gaps": row["internal_missing"] > 0,
            "extreme_move": row["extreme_moves"] > 0,
            "stale": row["longest_flat_run"] >= STALE_RUN_DAYS,
            "non_positive": row["non_positive"] > 0,
        }
        row["issues"] = ";".join(name for name, hit in checks.items() if hit)
        rows.append(row)
    return pd.DataFrame(rows).set_index("ticker")


def fetch_risk_free(retries: int, pause: float) -> pd.Series:
    for attempt in range(retries + 1):
        try:
            raw = pd.read_csv(FRED_URL, index_col=0, parse_dates=True, na_values=".")
            return pd.to_numeric(raw.iloc[:, 0], errors="coerce").dropna().rename(FRED_SERIES)
        except Exception as exc:
            if attempt == retries:
                raise
            log.warning("FRED fetch failed (%s); retrying", exc)
            time.sleep(pause * 5 * (attempt + 1))
    raise RuntimeError("unreachable")


def align_risk_free(rf_pct: pd.Series, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    # Bond-market holidays (e.g. Columbus Day) are NYSE trading days, so carry the last quote forward.
    annual = rf_pct.reindex(rf_pct.index.union(calendar)).ffill().reindex(calendar)
    return pd.DataFrame({"annual_pct": annual, "daily": annual / 100 / TRADING_DAYS}, index=calendar)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch prices and the risk-free rate for the ESG universe.")
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=DEFAULT_END, help="Exclusive. Pinned so reruns reproduce the same dataset.")
    p.add_argument("--out-dir", default=settings.PRICES_DIR)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--pause", type=float, default=2.0, help="Seconds between batches.")
    p.add_argument("--min-coverage", type=float, default=0.9)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    mapping = load_universe(settings.DATA_PATH)
    log.info("Universe: %d symbols + %s, %s to %s (exclusive)", len(mapping) - 1, BENCHMARK, args.start, args.end)
    prices, failed = download_prices(list(mapping), args.start, args.end, args.batch_size, args.retries, args.pause)
    if BENCHMARK not in prices.columns:
        log.error("%s did not download, so there is no trading calendar; aborting.", BENCHMARK)
        return 1

    prices = prices.rename(columns=mapping)
    prices = prices.reindex(prices[BENCHMARK].dropna().index).sort_index(axis=1)
    prices.index.name = "date"
    failed = sorted(mapping[t] for t in failed)
    report = quality_report(prices, args.min_coverage)
    rf = align_risk_free(fetch_risk_free(args.retries, args.pause), prices.index)
    if rf["annual_pct"].isna().any():
        log.warning("Risk-free rate missing for %d trading days", int(rf["annual_pct"].isna().sum()))

    prices.to_csv(out / "prices.csv", float_format="%.4f")
    rf.to_csv(out / "risk_free.csv", float_format="%.8f")
    report.to_csv(out / "quality_report.csv")

    exploded = report["issues"].str.split(";").explode()
    flagged = {issue: sorted(exploded.index[exploded == issue]) for issue in ISSUES}
    manifest = {
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "start": args.start,
        "end_exclusive": args.end,
        "first_date": prices.index[0].date(),
        "last_date": prices.index[-1].date(),
        "trading_days": len(prices),
        "universe_file": Path(settings.DATA_PATH).name,
        "symbols_requested": len(mapping),
        "symbols_with_data": prices.shape[1],
        "renamed": {s: y for s, y in RENAMED.items() if s in mapping.values()},
        "failed": failed,
        "flagged": flagged,
        "sources": {
            "prices": f"Yahoo Finance via yfinance {yf.__version__}, auto_adjust=True",
            "risk_free": FRED_URL,
        },
        "pandas_version": pd.__version__,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    log.info("Saved %d trading days x %d symbols to %s", len(prices), prices.shape[1], out)
    log.info("Failed (no data): %d %s", len(failed), failed)
    for issue, tickers in flagged.items():
        if tickers:
            log.info("Flagged %-13s %3d %s", issue, len(tickers), tickers[:10] + (["..."] if len(tickers) > 10 else []))
    return 0


if __name__ == "__main__":
    sys.exit(main())
