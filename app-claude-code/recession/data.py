"""FRED ingestion, caching and feature engineering."""
from __future__ import annotations

import io
import logging
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from . import config

log = logging.getLogger(__name__)

PUBLIC_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"


def _fetch_api(sid: str) -> pd.Series:
    from fredapi import Fred

    return Fred(api_key=config.FRED_API_KEY).get_series(sid)


def _fetch_public(sid: str) -> pd.Series:
    """Keyless fallback: FRED's public graph CSV endpoint."""
    r = requests.get(PUBLIC_CSV.format(sid=sid), timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), na_values=["."])
    return pd.Series(df.iloc[:, 1].values, index=pd.to_datetime(df.iloc[:, 0]), name=sid)


def _fetch(sid: str) -> pd.Series:
    fetch = _fetch_api if config.FRED_API_KEY else _fetch_public
    for attempt in range(3):
        try:
            s = pd.to_numeric(fetch(sid), errors="coerce").dropna()
            # Weekly/daily series -> monthly average, indexed to month start.
            return s.resample("MS").mean()
        except Exception as exc:  # network hiccups, rate limits
            if attempt == 2:
                raise RuntimeError(f"Failed to fetch FRED series {sid}: {exc}") from exc
            time.sleep(1.5 * (attempt + 1))
    raise AssertionError("unreachable")


def fetch_raw(force: bool = False) -> tuple[pd.DataFrame, dict]:
    """Return monthly raw series and fetch metadata, using a local cache."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not force and config.RAW_CACHE.exists():
        age_h = (time.time() - config.RAW_CACHE.stat().st_mtime) / 3600
        if age_h < config.CACHE_MAX_AGE_HOURS:
            with open(config.RAW_CACHE, "rb") as f:
                return pickle.load(f)

    with ThreadPoolExecutor(max_workers=6) as pool:
        series = dict(zip(config.SERIES, pool.map(_fetch, config.SERIES)))
    raw = pd.DataFrame(series).sort_index()
    meta = {
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "FRED API" if config.FRED_API_KEY else "FRED public CSV (no API key)",
        "last_observation": {k: v.last_valid_index().strftime("%Y-%m-%d") for k, v in series.items()},
    }
    with open(config.RAW_CACHE, "wb") as f:
        pickle.dump((raw, meta), f)
    return raw, meta


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Engineer model features from monthly raw series.

    Ragged-edge handling: monthly releases lag by 1–2 months, so each raw series
    is carried forward at most 2 months past its last print. Historical gaps are
    not filled beyond that.
    """
    r = raw.ffill(limit=2)
    log_ = lambda c: np.log(r[c])
    term = r["GS10"] - r["TB3MS"]
    f = pd.DataFrame(index=r.index)
    f["term_spread"] = term
    f["term_spread_min12"] = term.rolling(12, min_periods=12).min()
    f["credit_spread"] = r["BAA"] - r["GS10"]
    f["nfci"] = r["NFCI"]
    f["fedfunds_chg12"] = r["FEDFUNDS"].diff(12)
    f["sahm"] = r["SAHMREALTIME"]
    f["unrate_chg12"] = r["UNRATE"].diff(12)
    f["claims_chg6"] = log_("ICSA").diff(6)
    f["payrolls_yoy"] = log_("PAYEMS").diff(12)
    f["indpro_yoy"] = log_("INDPRO").diff(12)
    f["income_yoy"] = log_("W875RX1").diff(12)
    f["permits_yoy"] = log_("PERMIT").diff(12)
    f["oil_yoy"] = log_("WTISPLC").diff(12)
    f = f.loc[config.SAMPLE_START:, config.FEATURE_NAMES]
    return f.dropna()


def recession_flag(raw: pd.DataFrame) -> pd.Series:
    return raw["USREC"].dropna().astype(int)


def make_target(usrec: pd.Series, horizon: int) -> pd.Series:
    """1 if any month in (t, t+horizon] is an NBER recession month; NaN when unknown."""
    fwd = pd.concat([usrec.shift(-k) for k in range(1, horizon + 1)], axis=1)
    y = fwd.max(axis=1)
    y[fwd.isna().any(axis=1)] = np.nan
    return y


def load_dataset(force: bool = False, horizon: int = config.HORIZON):
    """Features (all months incl. unlabeled recent ones), target, USREC, fetch metadata."""
    raw, meta = fetch_raw(force=force)
    X = build_features(raw)
    # As-of month: latest month in which at least 75% of series actually printed,
    # so a mostly carried-forward partial month is never scored.
    observed = raw[list(config.SERIES)].notna().mean(axis=1)
    as_of = observed[observed >= 0.75].index.max()
    X = X.loc[:as_of]
    meta = {**meta, "as_of": as_of.strftime("%Y-%m-%d")}
    usrec = recession_flag(raw)
    y = make_target(usrec, horizon).reindex(X.index)
    return X, y, usrec, meta
