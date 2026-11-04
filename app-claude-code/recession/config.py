"""Central configuration: paths, FRED series, feature definitions, model settings."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

ARTIFACTS = ROOT / "artifacts"
DATA_DIR = ARTIFACTS / "data"
REGISTRY_DIR = ARTIFACTS / "registry"
PRODUCTION_POINTER = ARTIFACTS / "production.json"
RAW_CACHE = DATA_DIR / "fred_raw.pkl"

FRED_API_KEY = os.getenv("FRED_API_KEY", "").strip()
CACHE_MAX_AGE_HOURS = float(os.getenv("CACHE_MAX_AGE_HOURS", "12"))

HORIZON = 12                                  # headline forecast window (months)
TERM_STRUCTURE_HORIZONS = [3, 6, 9, 12, 18, 24]
SAMPLE_START = "1971-01-01"                   # NFCI begins Jan 1971
BACKTEST_START = "1985-01-01"                 # first out-of-sample year
RANDOM_STATE = 42

# FRED series pulled for the model. All are resampled to monthly averages.
SERIES = {
    "USREC": "NBER recession indicator",
    "GS10": "10-Year Treasury constant maturity",
    "TB3MS": "3-Month Treasury bill, secondary market",
    "BAA": "Moody's Baa corporate bond yield",
    "NFCI": "Chicago Fed National Financial Conditions Index",
    "SAHMREALTIME": "Sahm rule recession indicator (real-time)",
    "UNRATE": "Unemployment rate",
    "ICSA": "Initial jobless claims",
    "PAYEMS": "Total nonfarm payrolls",
    "INDPRO": "Industrial production index",
    "W875RX1": "Real personal income ex. transfer receipts",
    "PERMIT": "New private housing permits",
    "FEDFUNDS": "Effective federal funds rate",
    "WTISPLC": "WTI crude oil spot price",
}

# Engineered features. `direction` = sign of the move that raises recession risk.
FEATURES = {
    "term_spread": dict(label="Yield curve (10y − 3m)", unit="pp", group="Rates & credit",
                        source="GS10 − TB3MS", direction=-1,
                        desc="Slope of the Treasury curve. Inversion has preceded every US recession since 1970."),
    "term_spread_min12": dict(label="Curve inversion depth (12m min)", unit="pp", group="Rates & credit",
                              source="12-month rolling min of GS10 − TB3MS", direction=-1,
                              desc="Deepest point of the curve over the past year — captures recent inversions that have since re-steepened."),
    "credit_spread": dict(label="Credit spread (Baa − 10y)", unit="pp", group="Rates & credit",
                          source="BAA − GS10", direction=1,
                          desc="Default-risk premium on investment-grade corporate debt."),
    "nfci": dict(label="Financial conditions (NFCI)", unit="index", group="Rates & credit",
                 source="NFCI", direction=1,
                 desc="Chicago Fed index; positive values = tighter-than-average financial conditions."),
    "fedfunds_chg12": dict(label="Fed funds, 12m change", unit="pp", group="Rates & credit",
                           source="FEDFUNDS, 12-month difference", direction=1,
                           desc="Pace of monetary tightening over the past year."),
    "sahm": dict(label="Sahm rule indicator", unit="pp", group="Labor",
                 source="SAHMREALTIME", direction=1,
                 desc="3m avg unemployment rate vs. its 12m low. ≥ 0.50 has historically marked recession onset."),
    "unrate_chg12": dict(label="Unemployment rate, 12m change", unit="pp", group="Labor",
                         source="UNRATE, 12-month difference", direction=1,
                         desc="Year-over-year change in the headline unemployment rate."),
    "claims_chg6": dict(label="Initial claims, 6m change", unit="log Δ", group="Labor",
                        source="log(ICSA), 6-month difference", direction=1,
                        desc="Rising layoffs show up first in weekly initial claims."),
    "payrolls_yoy": dict(label="Nonfarm payrolls, YoY", unit="log Δ", group="Labor",
                         source="log(PAYEMS), 12-month difference", direction=-1,
                         desc="Employment growth across the nonfarm economy."),
    "indpro_yoy": dict(label="Industrial production, YoY", unit="log Δ", group="Activity",
                       source="log(INDPRO), 12-month difference", direction=-1,
                       desc="Real output of manufacturing, mining and utilities."),
    "income_yoy": dict(label="Real income ex. transfers, YoY", unit="log Δ", group="Activity",
                       source="log(W875RX1), 12-month difference", direction=-1,
                       desc="One of the NBER's core business-cycle dating series."),
    "permits_yoy": dict(label="Building permits, YoY", unit="log Δ", group="Activity",
                        source="log(PERMIT), 12-month difference", direction=-1,
                        desc="Housing is among the earliest sectors to turn down."),
    "oil_yoy": dict(label="WTI crude oil, YoY", unit="log Δ", group="Activity",
                    source="log(WTISPLC), 12-month difference", direction=1,
                    desc="Oil price shocks preceded the 1973, 1980, 1990 and 2008 recessions."),
}
FEATURE_NAMES = list(FEATURES)

MODEL_LABELS = {
    "logistic": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "ensemble": "Ensemble (soft vote)",
}
