"""Recession Probability Monitor — FastAPI service.

    uvicorn app.main:app --reload        →  http://127.0.0.1:8000
"""
from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from recession import config, data, inference, registry

log = logging.getLogger("uvicorn.error")
STATIC = Path(__file__).parent / "static"


class LiveState:
    """Production model + latest FRED data, scored once and served from memory."""

    def __init__(self):
        self.lock = threading.Lock()
        self.ready = False

    def load(self, force_refresh: bool = False) -> None:
        bundle, meta, report, pointer = registry.load_production()
        X, y, usrec, fetch_meta = data.load_dataset(force=force_refresh)
        live = inference.score(bundle, X)
        latest = X.iloc[[-1]]
        with self.lock:
            self.bundle, self.meta, self.report, self.pointer = bundle, meta, report, pointer
            self.X, self.y, self.usrec, self.fetch_meta = X, y, usrec, fetch_meta
            self.live, self.latest = live, latest
            self.ready = True
        log.info("Loaded production %s (%s), as-of %s, P=%.3f",
                 pointer["version"], meta["champion"], fetch_meta["as_of"], live.iloc[-1])

    def require(self):
        if not self.ready:
            raise HTTPException(503, "Model not loaded. Run `python train.py`, then restart.")
        return self


state = LiveState()


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        state.load()
    except Exception as exc:  # keep the server up so /api/health can report the problem
        log.error("Startup load failed: %s", exc)
        state.error = str(exc)
    yield


app = FastAPI(
    title="Recession Probability Monitor",
    description="12-month-ahead US recession probabilities from FRED leading indicators.",
    version="1.0.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.middleware("http")
async def cache_headers(request: Request, call_next):
    """API responses are never cached; app assets revalidate (ETag) so redeploys show up at once."""
    resp = await call_next(request)
    path = request.url.path
    if path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-store"
    elif not path.startswith("/static/vendor/"):
        resp.headers["Cache-Control"] = "no-cache"
    return resp


def recession_periods(usrec: pd.Series, start: str) -> list[dict]:
    u = usrec.loc[start:]
    starts = u[(u == 1) & (u.shift(1, fill_value=0) == 0)].index
    ends = u[(u == 1) & (u.shift(-1, fill_value=0) == 0)].index
    return [{"start": s.strftime("%Y-%m-%d"), "end": (e + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")}
            for s, e in zip(starts, ends)]


def _r(obj):
    return inference.round_floats(obj)


# --------------------------------------------------------------------------- pages
@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


# --------------------------------------------------------------------------- api
@app.get("/api/health", tags=["ops"])
def health():
    return {
        "status": "ok" if state.ready else "degraded",
        "model_version": getattr(state, "pointer", {}).get("version"),
        "error": getattr(state, "error", None),
    }


@app.get("/api/summary", tags=["forecast"])
def summary():
    s = state.require()
    p = s.live
    thr = s.bundle["threshold"]
    current = float(p.iloc[-1])
    return _r({
        "as_of": s.fetch_meta["as_of"],
        "probability": current,
        "change_1m": current - float(p.iloc[-2]),
        "change_3m": current - float(p.iloc[-4]),
        "change_12m": current - float(p.iloc[-13]),
        "peak_12m": float(p.iloc[-12:].max()),
        "threshold": thr,
        "band": inference.risk_band(current, thr),
        "model": {"name": s.meta["champion"], "label": s.meta["champion_label"],
                  "version": s.pointer["version"], "promoted_at": s.pointer["promoted_at"],
                  "pr_auc": s.meta["metrics"][s.meta["champion"]]["pr_auc"],
                  "roc_auc": s.meta["metrics"][s.meta["champion"]]["roc_auc"]},
        "data": {"source": s.fetch_meta["source"], "fetched_at": s.fetch_meta["fetched_at"],
                 "last_observation": s.fetch_meta["last_observation"]},
        "horizon_months": config.HORIZON,
    })


@app.get("/api/model", tags=["model"])
def model_card():
    s = state.require()
    return {**s.meta, "production": s.pointer, "labels": config.MODEL_LABELS,
            "versions": registry.list_versions()}


@app.get("/api/history", tags=["forecast"])
def history():
    """Walk-forward out-of-sample probabilities for every model, plus the live production fit."""
    s = state.require()
    return _r({
        "dates": s.report["dates"],
        "oos": s.report["oos"],
        "live": {"dates": [d.strftime("%Y-%m-%d") for d in s.live.index], "probability": s.live.tolist()},
        "recessions": recession_periods(s.usrec, config.SAMPLE_START),
        "threshold": s.bundle["threshold"],
        "champion": s.meta["champion"],
    })


@app.get("/api/forecast", tags=["forecast"])
def forecast():
    """Probability term structure (P(recession within h months)) now and at earlier vintages."""
    s = state.require()
    snaps = {}
    for label, back in [("Now", 1), ("3 months ago", 4), ("6 months ago", 7), ("12 months ago", 13)]:
        row = s.X.iloc[[-back]]
        snaps[label] = {"as_of": row.index[0].strftime("%Y-%m"),
                        "curve": inference.term_structure(s.bundle, row)}
    return _r({"horizons": config.TERM_STRUCTURE_HORIZONS, "snapshots": snaps})


@app.get("/api/drivers", tags=["forecast"])
def drivers():
    s = state.require()
    return _r({"probability": float(s.live.iloc[-1]),
               "drivers": inference.drivers(s.bundle, s.latest)})


@app.get("/api/importance", tags=["model"])
def importance():
    s = state.require()
    return {"importance": s.report["importance"], "features": config.FEATURES,
            "champion": s.meta["champion"]}


@app.get("/api/backtest", tags=["model"])
def backtest():
    s = state.require()
    return {"roc": s.report["roc"], "calibration": s.report["calibration"],
            "episodes": s.report["episodes"], "metrics": s.meta["metrics"],
            "backtest": s.meta["backtest"]}


@app.get("/api/indicators", tags=["data"])
def indicators():
    s = state.require()
    return _r({
        "dates": [d.strftime("%Y-%m-%d") for d in s.X.index],
        "series": {f: s.X[f].tolist() for f in config.FEATURE_NAMES},
        "context": inference.indicator_context(s.X, s.usrec),
        "recessions": recession_periods(s.usrec, config.SAMPLE_START),
    })


@app.get("/api/predict", tags=["forecast"])
def predict(month: str):
    """Production-model probability for a historical month (YYYY-MM)."""
    s = state.require()
    ts = pd.Timestamp(month + "-01") if len(month) == 7 else pd.Timestamp(month)
    if ts not in s.live.index:
        raise HTTPException(404, f"No feature data for {month}")
    row = s.X.loc[[ts]]
    return _r({"month": ts.strftime("%Y-%m"), "probability": float(s.live[ts]),
               "term_structure": inference.term_structure(s.bundle, row),
               "features": row.iloc[0].to_dict(),
               "recession_within_12m": None if pd.isna(s.y.get(ts)) else int(s.y[ts])})


class ScenarioIn(BaseModel):
    overrides: dict[str, float] = Field(default_factory=dict,
                                        examples=[{"term_spread": -1.0, "sahm": 0.6}])


@app.post("/api/scenario", tags=["forecast"])
def run_scenario(body: ScenarioIn):
    s = state.require()
    unknown = set(body.overrides) - set(config.FEATURE_NAMES)
    if unknown:
        raise HTTPException(422, f"Unknown features: {sorted(unknown)}")
    res = inference.scenario(s.bundle, s.latest, body.overrides)
    return _r({**res, "baseline_probability": float(s.live.iloc[-1])})


@app.post("/api/refresh", tags=["ops"])
def refresh():
    """Re-pull FRED data and re-score with the production model (no retraining)."""
    try:
        state.load(force_refresh=True)
    except Exception as exc:
        raise HTTPException(502, f"Refresh failed: {exc}")
    return summary()
