"""Recession probability dashboard (FastAPI).

Uses the model saved by train.py:
    models/recession_model.joblib
    models/recession_model_meta.json
and the FRED key in .env (FRED_API_KEY).

Run:  uvicorn app:app --reload      then open http://127.0.0.1:8000
"""
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fredapi import Fred
from sklearn.base import clone
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

load_dotenv()

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "recession_model.joblib"
META_PATH = MODELS_DIR / "recession_model_meta.json"
CACHE = ROOT / "data" / "fred_raw.pkl"   # created automatically
CACHE_MAX_AGE_H = 24
OOS_START = "1990-01-01"

SERIES = ["USREC", "GS10", "TB3MS", "BAA", "SAHMREALTIME", "ICSA", "PERMIT", "INDPRO"]

FEATURE_INFO = {
    "term": {"label": "Term spread (10y − 3m)", "unit": "pp", "source": "GS10 − TB3MS",
             "desc": "Yield curve slope. Inversion (below 0) has preceded most recessions."},
    "credit": {"label": "Credit spread (Baa − 10y)", "unit": "pp", "source": "BAA − GS10",
               "desc": "Extra yield on riskier corporate debt. Widens as default risk rises."},
    "sahm": {"label": "Sahm rule indicator", "unit": "pp", "source": "SAHMREALTIME",
             "desc": "3-mo avg unemployment vs its 12-mo low. Readings ≥ 0.5 signal recession."},
    "claims": {"label": "Jobless claims, 6-mo change", "unit": "log Δ", "source": "log(ICSA), 6-mo diff",
               "desc": "Rising initial claims point to layoffs picking up."},
    "permits": {"label": "Building permits, 12-mo change", "unit": "log Δ", "source": "log(PERMIT), 12-mo diff",
                "desc": "Housing tends to turn down well before the broader economy."},
    "ip_yoy": {"label": "Industrial production, YoY", "unit": "log Δ", "source": "log(INDPRO), 12-mo diff",
               "desc": "Real output growth in manufacturing, mining and utilities."},
}


# ---------- data ----------

def fetch_raw() -> pd.DataFrame:
    key = os.environ.get("FRED_API_KEY")
    if not key:
        raise RuntimeError("FRED_API_KEY is not set. Add it to your .env file.")
    fred = Fred(api_key=key)
    raw = {s: fred.get_series(s, frequency="m", aggregation_method="avg") for s in SERIES}
    return pd.DataFrame(raw).sort_index()


def load_raw(refresh: bool = False) -> pd.DataFrame:
    """FRED data with a 24h on-disk cache so the app starts fast."""
    if not refresh and CACHE.exists():
        if (time.time() - CACHE.stat().st_mtime) / 3600 < CACHE_MAX_AGE_H:
            return pd.read_pickle(CACHE)
    raw = fetch_raw()
    CACHE.parent.mkdir(exist_ok=True)
    raw.to_pickle(CACHE)
    return raw


def build_features(raw: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Same features as train.py, plus the forward-looking label y."""
    df = pd.DataFrame(index=raw.index)
    df["rec"] = raw["USREC"]
    df["term"] = raw["GS10"] - raw["TB3MS"]
    df["credit"] = raw["BAA"] - raw["GS10"]
    df["sahm"] = raw["SAHMREALTIME"]
    df["claims"] = np.log(raw["ICSA"]).diff(6)
    df["permits"] = np.log(raw["PERMIT"]).diff(12)
    df["ip_yoy"] = np.log(raw["INDPRO"]).diff(12)
    fwd = pd.concat([df["rec"].shift(-k) for k in range(1, horizon + 1)], axis=1)
    df["y"] = fwd.max(axis=1, skipna=False)   # NaN until the full window is known
    return df


# ---------- walk-forward evaluation (same procedure as train.py) ----------

def walk_forward(data: pd.DataFrame, feats: list[str], horizon: int) -> pd.Series:
    preds = []
    for test_start in pd.date_range(OOS_START, data.index[-1], freq="YS"):
        train = data[data.index < test_start - pd.DateOffset(months=horizon)]
        test = data[(data.index >= test_start) & (data.index < test_start + pd.DateOffset(years=1))]
        if train["y"].nunique() < 2 or test.empty:
            continue
        m = clone(S.model).fit(train[feats], train["y"])
        preds.append(pd.Series(m.predict_proba(test[feats])[:, 1], index=test.index))
    return pd.concat(preds)


def score(y, p) -> dict:
    base = brier_score_loss(y, np.full(len(y), y.mean()))
    brier = brier_score_loss(y, p)
    return {"auc": float(roc_auc_score(y, p)), "brier": float(brier),
            "brier_skill": float(1 - brier / base),
            "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))}


# ---------- state ----------

class State:
    model = None
    feats: list[str] = []
    horizon: int = 12
    meta: dict = {}
    df: pd.DataFrame | None = None
    scored: pd.DataFrame | None = None
    oos: pd.DataFrame | None = None
    data_updated: str | None = None


S = State()


def _steps():
    return S.model.named_steps["standardscaler"], S.model.named_steps["logisticregression"]


def contributions(X: pd.DataFrame):
    """Exact log-odds decomposition: logodds = intercept + sum(coef_j * z_j)."""
    scaler, clf = _steps()
    z = (X[S.feats] - scaler.mean_) / scaler.scale_
    return z * clf.coef_[0], float(clf.intercept_[0])


def load_state(refresh_data: bool = False):
    if not MODEL_PATH.exists():
        raise RuntimeError("models/recession_model.joblib not found. Run `python train.py` first.")
    S.model = joblib.load(MODEL_PATH)
    saved_meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
    S.horizon = int(saved_meta.get("horizon_months", 12))
    S.feats = list(saved_meta.get("features", list(S.model.feature_names_in_)))

    S.df = build_features(load_raw(refresh=refresh_data), S.horizon)
    S.data_updated = pd.Timestamp(CACHE.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M UTC")

    scored = S.df.dropna(subset=S.feats).copy()
    contrib, intercept = contributions(scored)
    scored["logodds"] = intercept + contrib.sum(axis=1)
    scored["prob"] = 1 / (1 + np.exp(-scored["logodds"]))
    for f in S.feats:
        scored[f"c_{f}"] = contrib[f]
    S.scored = scored

    # Out-of-sample: full model vs yield-curve-only vs base rate
    data = S.df.dropna(subset=S.feats + ["y"])
    p_full = walk_forward(data, S.feats, S.horizon)
    p_term = walk_forward(data, ["term"], S.horizon).reindex(p_full.index)
    y = data.loc[p_full.index, "y"]
    S.oos = pd.DataFrame({"y": y, "p_full": p_full, "p_term": p_term})
    rate = float(y.mean())
    metrics = {
        "full_model": score(y, p_full),
        "term_only": score(y, p_term.fillna(rate)),
        "base_rate": {"auc": 0.5, "brier": float(brier_score_loss(y, np.full(len(y), rate))),
                      "brier_skill": 0.0, "log_loss": float(log_loss(y, np.full(len(y), rate)))},
    }
    S.meta = {
        **saved_meta,
        "features": S.feats,
        "horizon_months": S.horizon,
        "train_start": saved_meta.get("train_start", str(data.index[0].date())),
        "train_end": saved_meta.get("train_end", str(data.index[-1].date())),
        "trained_at": pd.Timestamp(MODEL_PATH.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M UTC"),
        "oos_start": str(p_full.index[0].date()),
        "oos_end": str(p_full.index[-1].date()),
        "positive_rate_oos": rate,
        "metrics": metrics,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_state()
    yield


app = FastAPI(title="Recession Probability Dashboard", lifespan=lifespan)


# ---------- helpers ----------

def info(f: str) -> dict:
    return FEATURE_INFO.get(f, {"label": f, "unit": "", "source": f, "desc": ""})


def clean(values) -> list:
    return [None if v is None or not np.isfinite(v) else float(v) for v in values]


def dates(idx) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in idx]


def recession_periods() -> list[dict]:
    rec = S.df["rec"].dropna()
    periods, start = [], None
    for d, v in rec.items():
        if v == 1 and start is None:
            start = d
        elif v != 1 and start is not None:
            periods.append({"start": start.strftime("%Y-%m-%d"), "end": d.strftime("%Y-%m-%d")})
            start = None
    if start is not None:
        periods.append({"start": start.strftime("%Y-%m-%d"), "end": rec.index[-1].strftime("%Y-%m-%d")})
    return periods


def prob_months_ago(n: int):
    sc = S.scored
    sub = sc[sc.index <= sc.index[-1] - pd.DateOffset(months=n)]
    return None if sub.empty else float(sub["prob"].iloc[-1])


# ---------- routes ----------

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index():
    return INDEX_HTML


@app.get("/api/summary")
def summary():
    last = S.scored.iloc[-1]
    intercept = float(_steps()[1].intercept_[0])
    drivers = sorted(({"feature": f, "label": info(f)["label"], "contribution": float(last[f"c_{f}"])}
                      for f in S.feats), key=lambda d: -abs(d["contribution"]))
    return {
        "as_of": last.name.strftime("%Y-%m-%d"),
        "prob": float(last["prob"]), "logodds": float(last["logodds"]), "intercept": intercept,
        "prob_3m_ago": prob_months_ago(3), "prob_6m_ago": prob_months_ago(6),
        "prob_12m_ago": prob_months_ago(12),
        "drivers": drivers, "horizon_months": S.horizon,
        "data_updated": S.data_updated, "meta": S.meta,
    }


@app.get("/api/history")
def history():
    sc = S.scored
    return {
        "dates": dates(sc.index),
        "prob": clean(sc["prob"].round(4)),
        "logodds": clean(sc["logodds"].round(4)),
        "contributions": {f: clean(sc[f"c_{f}"].round(4)) for f in S.feats},
        "recessions": recession_periods(),
        "oos": {"dates": dates(S.oos.index), "p_full": clean(S.oos["p_full"].round(4)),
                "p_term": clean(S.oos["p_term"].round(4))},
    }


@app.get("/api/contributions")
def contributions_at(date: str | None = Query(None, description="YYYY-MM or YYYY-MM-DD")):
    sc = S.scored
    if date:
        try:
            ts = pd.Timestamp(date)
        except ValueError:
            raise HTTPException(400, f"Bad date: {date}")
        sub = sc[sc.index <= ts]
        if sub.empty:
            raise HTTPException(404, f"No data on or before {date}")
        row = sub.iloc[-1]
    else:
        row = sc.iloc[-1]
    scaler, clf = _steps()
    items = [{
        "feature": f, "label": info(f)["label"], "value": float(row[f]),
        "z": float((row[f] - scaler.mean_[i]) / scaler.scale_[i]),
        "coef": float(clf.coef_[0][i]), "contribution": float(row[f"c_{f}"]),
    } for i, f in enumerate(S.feats)]
    return {
        "date": row.name.strftime("%Y-%m-%d"), "intercept": float(clf.intercept_[0]), "items": items,
        "logodds": float(row["logodds"]), "prob": float(row["prob"]),
        "actual_rec": None if pd.isna(row["rec"]) else int(row["rec"]),
        "label_y": None if pd.isna(row["y"]) else int(row["y"]),
    }


@app.get("/api/importance")
def importance():
    _, clf = _steps()
    rows = [{
        "feature": f, "label": info(f)["label"],
        "coef_per_sd": float(clf.coef_[0][i]),
        "odds_ratio_per_sd": float(np.exp(clf.coef_[0][i])),
        "mean_abs_contribution": float(S.scored[f"c_{f}"].abs().mean()),
    } for i, f in enumerate(S.feats)]
    rows.sort(key=lambda r: -abs(r["coef_per_sd"]))
    return {"intercept": float(clf.intercept_[0]), "features": rows}


@app.get("/api/features")
def features():
    df = S.df
    out = {"recessions": recession_periods(), "features": []}
    for f in S.feats:
        s = df[f].dropna()
        cur = float(s.iloc[-1])
        out["features"].append({
            **info(f), "feature": f, "dates": dates(s.index), "values": clean(s.round(4)),
            "current": cur, "current_date": s.index[-1].strftime("%Y-%m-%d"),
            "percentile": float((s <= cur).mean()),
            "mean_in_recession_lead": float(df.loc[df["y"] == 1, f].mean()),
            "mean_otherwise": float(df.loc[df["y"] == 0, f].mean()),
        })
    return out


@app.get("/api/calibration")
def calibration(bins: int = Query(10, ge=3, le=20)):
    o = S.oos.dropna(subset=["p_full", "y"])
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(o["p_full"], edges) - 1, 0, bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        if m.sum():
            out.append({"bin_low": float(edges[b]), "bin_high": float(edges[b + 1]),
                        "mean_pred": float(o["p_full"][m].mean()),
                        "observed": float(o["y"][m].mean()), "count": int(m.sum())})
    return {"bins": out, "metrics": S.meta["metrics"]}


@app.post("/api/refresh")
def refresh():
    """Re-download FRED data and re-score with the saved model (does not retrain)."""
    try:
        load_state(refresh_data=True)
    except Exception as e:
        raise HTTPException(502, f"Refresh failed: {e}")
    return {"ok": True, "as_of": S.scored.index[-1].strftime("%Y-%m-%d"), "data_updated": S.data_updated}


# ---------- page ----------

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Recession Probability Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<style>
  :root {
    --bg: #f6f5f1; --panel: #ffffff; --ink: #1d1f23; --muted: #6b6f76; --line: #e3e1da;
    --accent: #2f5d8a; --accent2: #c26a2b; --accent3: #6f8f3a;
    --pos: #b8432f; --neg: #2f7a5a; --shade: rgba(120,120,120,0.18);
    --mono: ui-monospace, SFMono-Regular, Menlo, monospace;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #15171a; --panel: #1d2024; --ink: #e8e6e1; --muted: #9a9ea6; --line: #30343a;
      --accent: #7aa7d6; --accent2: #e39a5e; --accent3: #a8c271;
      --pos: #e2765f; --neg: #5fbf93; --shade: rgba(200,200,200,0.12);
    }
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--ink);
         font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, sans-serif; }
  .wrap { max-width: 1200px; margin: 0 auto; padding: 24px 16px 64px; }
  header { display: flex; flex-wrap: wrap; align-items: end; justify-content: space-between; gap: 12px; margin-bottom: 20px; }
  h1 { font-size: 24px; margin: 0; letter-spacing: -0.01em; }
  h2 { font-size: 16px; margin: 0 0 4px; }
  .sub { color: var(--muted); font-size: 13px; }
  button, select, input { font: inherit; color: var(--ink); background: var(--panel);
         border: 1px solid var(--line); border-radius: 8px; padding: 6px 12px; }
  button { cursor: pointer; } button:hover { border-color: var(--accent); }
  button:disabled { opacity: .5; cursor: wait; }
  .grid { display: grid; gap: 16px; }
  .kpis { grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); margin-bottom: 16px; }
  .two { grid-template-columns: 1fr 1fr; }
  @media (max-width: 860px) { .two { grid-template-columns: 1fr; } }
  .card { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 16px; min-width: 0; }
  .card + .card, .grid + .card, .card + .grid { margin-top: 16px; }
  .kpi .label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .06em; }
  .kpi .value { font-size: 28px; font-weight: 650; font-variant-numeric: tabular-nums; }
  .kpi .note { color: var(--muted); font-size: 13px; }
  .big .value { font-size: 44px; }
  .row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; justify-content: space-between; margin-bottom: 8px; }
  .controls { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
  .chart { width: 100%; height: 360px; }
  .chart.tall { height: 420px; }
  .chart.short { height: 300px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; }
  th, td { text-align: right; padding: 6px 8px; border-bottom: 1px solid var(--line); }
  th:first-child, td:first-child { text-align: left; }
  th { color: var(--muted); font-weight: 500; }
  .pos { color: var(--pos); } .neg { color: var(--neg); }
  .pill { display: inline-block; padding: 1px 8px; border-radius: 99px; font-size: 12px; border: 1px solid var(--line); color: var(--muted); }
  label.chk { font-size: 13px; color: var(--muted); display: inline-flex; gap: 4px; align-items: center; }
  .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 8px; margin-top: 8px; }
  .stats div { font-size: 13px; color: var(--muted); }
  .stats b { display: block; color: var(--ink); font-size: 18px; font-variant-numeric: tabular-nums; }
  .err { background: #fbe9e5; color: #8a2a17; padding: 12px 16px; border-radius: 8px; display: none; margin-bottom: 16px; }
  .notes { color: var(--muted); font-size: 13px; } .notes li { margin: 4px 0; }
  .scroll { overflow-x: auto; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <h1>U.S. Recession Probability</h1>
      <div class="sub" id="subtitle">Loading…</div>
    </div>
    <div class="controls">
      <span class="sub" id="updated"></span>
      <button id="refreshBtn" title="Re-download FRED data and re-score (no retraining)">Refresh data</button>
    </div>
  </header>

  <div class="err" id="err"></div>

  <!-- KPIs -->
  <div class="grid kpis">
    <div class="card kpi big"><div class="label" id="probLabel">P(recession, next 12 mo)</div><div class="value" id="kProb">–</div><div class="note" id="kAsOf"></div></div>
    <div class="card kpi"><div class="label">Log-odds</div><div class="value" id="kLogit">–</div><div class="note">intercept + Σ contributions</div></div>
    <div class="card kpi"><div class="label">Change vs 3 mo ago</div><div class="value" id="k3">–</div><div class="note" id="k3n"></div></div>
    <div class="card kpi"><div class="label">Change vs 12 mo ago</div><div class="value" id="k12">–</div><div class="note" id="k12n"></div></div>
    <div class="card kpi"><div class="label">Out-of-sample AUC</div><div class="value" id="kAuc">–</div><div class="note" id="kBss"></div></div>
  </div>

  <!-- Probability history -->
  <div class="card">
    <div class="row">
      <div><h2>Probability over time</h2><div class="sub">Grey bands are NBER recessions. Click any point to inspect that month below.</div></div>
      <div class="controls">
        <label class="chk"><input type="checkbox" id="showFit" checked> Final model (in-sample)</label>
        <label class="chk"><input type="checkbox" id="showOos" checked> Walk-forward (out-of-sample)</label>
        <label class="chk"><input type="checkbox" id="showTerm"> Yield curve only</label>
      </div>
    </div>
    <div id="probChart" class="chart tall"></div>
  </div>

  <!-- Log-odds decomposition over time -->
  <div class="card">
    <div class="row">
      <div><h2>What's driving the log-odds</h2><div class="sub">Each bar is a feature's contribution (coef × standardized value). Bars plus the intercept sum to the black line.</div></div>
    </div>
    <div id="contribChart" class="chart tall"></div>
  </div>

  <!-- Month inspector -->
  <div class="grid two">
    <div class="card">
      <div class="row">
        <div><h2>Month breakdown</h2><div class="sub" id="wfSub"></div></div>
        <div class="controls">
          <input type="month" id="monthPick">
          <button id="latestBtn">Latest</button>
        </div>
      </div>
      <div id="waterfall" class="chart"></div>
    </div>
    <div class="card">
      <h2>Feature values that month</h2>
      <div class="sub">z = standard deviations from the training mean</div>
      <div class="scroll"><table id="wfTable"></table></div>
      <div class="sub" id="wfFoot" style="margin-top:8px"></div>
    </div>
  </div>

  <!-- Importance -->
  <div class="grid two">
    <div class="card">
      <h2>Feature importance: coefficient per 1 SD</h2>
      <div class="sub">Change in log-odds when the feature rises by one standard deviation. Hover for the odds ratio.</div>
      <div id="coefChart" class="chart short"></div>
    </div>
    <div class="card">
      <h2>Feature importance: average impact</h2>
      <div class="sub">Mean |contribution| to log-odds across all months. Reflects both coefficient size and how much the feature moves.</div>
      <div id="impactChart" class="chart short"></div>
    </div>
  </div>

  <!-- Feature explorer -->
  <div class="card">
    <div class="row">
      <div><h2>Feature explorer</h2><div class="sub" id="featDesc"></div></div>
      <select id="featPick"></select>
    </div>
    <div id="featChart" class="chart"></div>
    <div class="stats" id="featStats"></div>
  </div>

  <!-- Evaluation -->
  <div class="grid two">
    <div class="card">
      <h2>Out-of-sample performance</h2>
      <div class="sub" id="evalSub"></div>
      <div class="scroll"><table id="metricsTable"></table></div>
      <ul class="notes">
        <li><b>AUC</b>: how well the model ranks risky months above safe ones (0.5 = coin flip).</li>
        <li><b>Brier</b>: mean squared error of the probabilities (lower is better).</li>
        <li><b>Brier skill</b>: improvement over always predicting the base rate (&gt; 0 beats it).</li>
      </ul>
    </div>
    <div class="card">
      <h2>Calibration</h2>
      <div class="sub">Predicted probability vs how often a recession actually followed (walk-forward predictions). Marker size = number of months.</div>
      <div id="calChart" class="chart short"></div>
    </div>
  </div>

  <div class="card notes">
    <h2 style="color:var(--ink)">Caveats</h2>
    <ul>
      <li>Only a handful of recessions are in the sample, so every metric here is noisy.</li>
      <li>NBER announces recession dates months after the fact, and some inputs (claims, permits, industrial production) get revised. The backtest uses today's data, which flatters it slightly.</li>
      <li>The in-sample line is fit on all of history, so it looks better than the model would have done in real time. The walk-forward line is the honest one.</li>
      <li>"Refresh data" re-scores with the saved model. Re-run <code>python train.py</code> to retrain.</li>
    </ul>
  </div>
</div>

<script>
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const C = () => ({ ink: css('--ink'), muted: css('--muted'), line: css('--line'), panel: css('--panel'),
  accent: css('--accent'), accent2: css('--accent2'), accent3: css('--accent3'),
  pos: css('--pos'), neg: css('--neg'), shade: css('--shade') });
const PALETTE = ['#2f5d8a', '#c26a2b', '#6f8f3a', '#8a4f9e', '#b8432f', '#3a8f8a'];
const pct = (x, d = 1) => x == null ? '–' : (x * 100).toFixed(d) + '%';
const signed = (x, d = 2) => x == null ? '–' : (x >= 0 ? '+' : '') + x.toFixed(d);
const ppChange = (now, then) => then == null ? '–' : signed((now - then) * 100, 1) + ' pp';

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
}
function showErr(e) { const el = document.getElementById('err'); el.textContent = e.message || e; el.style.display = 'block'; }

function baseLayout(extra = {}) {
  const c = C();
  return Object.assign({
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: c.ink, size: 12 }, margin: { l: 50, r: 16, t: 10, b: 40 },
    xaxis: { gridcolor: c.line, zerolinecolor: c.line },
    yaxis: { gridcolor: c.line, zerolinecolor: c.line },
    legend: { orientation: 'h', y: -0.15 }, hovermode: 'x unified',
  }, extra);
}
const config = { responsive: true, displaylogo: false, modeBarButtonsToRemove: ['lasso2d', 'select2d'] };

function recShapes(recs) {
  return recs.map(r => ({ type: 'rect', xref: 'x', yref: 'paper', x0: r.start, x1: r.end, y0: 0, y1: 1,
    fillcolor: C().shade, line: { width: 0 }, layer: 'below' }));
}

let HIST = null, SUMMARY = null, FEATS = null;

async function loadSummary() {
  SUMMARY = await api('/api/summary');
  const s = SUMMARY, m = s.meta.metrics?.full_model;
  const h = s.horizon_months;
  document.getElementById('subtitle').textContent =
    `Logistic model on ${s.meta.features.length} FRED indicators · trained ${s.meta.train_start.slice(0,4)}–${s.meta.train_end.slice(0,7)} · ${s.meta.trained_at}`;
  document.getElementById('updated').textContent = s.data_updated ? `FRED data: ${s.data_updated}` : '';
  document.getElementById('probLabel').textContent = `P(recession, next ${h} mo)`;
  const kp = document.getElementById('kProb');
  kp.textContent = pct(s.prob);
  kp.style.color = s.prob > 0.5 ? C().pos : s.prob > 0.25 ? C().accent2 : C().neg;
  document.getElementById('kAsOf').textContent = 'as of ' + s.as_of.slice(0, 7);
  document.getElementById('kLogit').textContent = signed(s.logodds);
  document.getElementById('k3').textContent = ppChange(s.prob, s.prob_3m_ago);
  document.getElementById('k3n').textContent = s.prob_3m_ago != null ? 'was ' + pct(s.prob_3m_ago) : '';
  document.getElementById('k12').textContent = ppChange(s.prob, s.prob_12m_ago);
  document.getElementById('k12n').textContent = s.prob_12m_ago != null ? 'was ' + pct(s.prob_12m_ago) : '';
  document.getElementById('kAuc').textContent = m ? m.auc.toFixed(3) : '–';
  document.getElementById('kBss').textContent = m ? `Brier ${m.brier.toFixed(3)} · skill ${signed(m.brier_skill)}` : '';
}

function drawProb() {
  const c = C(), h = HIST, traces = [];
  if (document.getElementById('showFit').checked)
    traces.push({ x: h.dates, y: h.prob, name: 'Final model (in-sample)', mode: 'lines',
      line: { color: c.accent, width: 2 }, hovertemplate: '%{y:.1%}' });
  if (h.oos && document.getElementById('showOos').checked)
    traces.push({ x: h.oos.dates, y: h.oos.p_full, name: 'Walk-forward (OOS)', mode: 'lines',
      line: { color: c.accent2, width: 1.5 }, hovertemplate: '%{y:.1%}' });
  if (h.oos && document.getElementById('showTerm').checked)
    traces.push({ x: h.oos.dates, y: h.oos.p_term, name: 'Yield curve only (OOS)', mode: 'lines',
      line: { color: c.accent3, width: 1.5, dash: 'dot' }, hovertemplate: '%{y:.1%}' });
  const layout = baseLayout({
    shapes: [...recShapes(h.recessions),
      { type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 0.5, y1: 0.5, line: { color: c.muted, width: 1, dash: 'dash' } }],
    yaxis: { tickformat: '.0%', range: [0, 1], gridcolor: c.line },
    xaxis: { gridcolor: c.line, rangeselector: { buttons: [
      { count: 10, label: '10y', step: 'year', stepmode: 'backward' },
      { count: 25, label: '25y', step: 'year', stepmode: 'backward' },
      { step: 'all', label: 'All' }], bgcolor: c.panel, activecolor: c.line } },
  });
  Plotly.react('probChart', traces, layout, config);
}

function drawContrib() {
  const c = C(), h = HIST;
  const feats = Object.keys(h.contributions);
  const labels = Object.fromEntries((FEATS?.features || []).map(f => [f.feature, f.label]));
  const traces = feats.map((f, i) => ({
    type: 'bar', x: h.dates, y: h.contributions[f], name: labels[f] || f,
    marker: { color: PALETTE[i % PALETTE.length] }, hovertemplate: '%{y:+.2f}',
  }));
  traces.push({ type: 'scatter', mode: 'lines', x: h.dates, y: h.logodds, name: 'Total log-odds',
    line: { color: c.ink, width: 1.5 }, hovertemplate: '%{y:+.2f}' });
  Plotly.react('contribChart', traces, baseLayout({
    barmode: 'relative', bargap: 0, shapes: recShapes(h.recessions),
    yaxis: { title: 'log-odds', gridcolor: c.line, zerolinecolor: c.muted },
  }), config);
}

async function drawMonth(date) {
  const d = await api('/api/contributions' + (date ? '?date=' + date : ''));
  const c = C();
  document.getElementById('monthPick').value = d.date.slice(0, 7);
  document.getElementById('wfSub').textContent =
    `${d.date.slice(0, 7)} · log-odds ${signed(d.logodds)} → probability ${pct(d.prob)}`;
  const items = [...d.items].sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
  Plotly.react('waterfall', [{
    type: 'waterfall', orientation: 'v',
    x: ['Intercept', ...items.map(i => i.label), 'Total'],
    y: [d.intercept, ...items.map(i => i.contribution), 0],
    measure: ['absolute', ...items.map(() => 'relative'), 'total'],
    text: [d.intercept, ...items.map(i => i.contribution), d.logodds].map(v => signed(v)),
    textposition: 'outside', cliponaxis: false,
    increasing: { marker: { color: c.pos } }, decreasing: { marker: { color: c.neg } },
    totals: { marker: { color: c.accent } }, connector: { line: { color: c.line } },
    hovertemplate: '%{x}: %{text}<extra></extra>',
  }], baseLayout({ hovermode: 'closest', margin: { l: 50, r: 16, t: 20, b: 110 },
     xaxis: { tickangle: -30, gridcolor: c.line }, yaxis: { title: 'log-odds', gridcolor: c.line, zerolinecolor: c.muted } }), config);

  const unitOf = Object.fromEntries((FEATS?.features || []).map(f => [f.feature, f.unit]));
  document.getElementById('wfTable').innerHTML =
    '<tr><th>Feature</th><th>Value</th><th>z</th><th>Coef</th><th>Contribution</th></tr>' +
    d.items.map(i => `<tr><td>${i.label}</td><td>${i.value.toFixed(3)} <span class="sub">${unitOf[i.feature] || ''}</span></td>
      <td>${signed(i.z)}</td><td>${signed(i.coef)}</td>
      <td class="${i.contribution >= 0 ? 'pos' : 'neg'}">${signed(i.contribution)}</td></tr>`).join('') +
    `<tr><td>Intercept</td><td></td><td></td><td></td><td>${signed(d.intercept)}</td></tr>
     <tr><td><b>Total log-odds</b></td><td></td><td></td><td></td><td><b>${signed(d.logodds)}</b></td></tr>`;
  const actual = d.label_y == null ? 'not yet known' : d.label_y ? 'yes' : 'no';
  document.getElementById('wfFoot').innerHTML =
    `Probability = 1 / (1 + e<sup>−${d.logodds.toFixed(2)}</sup>) = <b>${pct(d.prob)}</b>. ` +
    `In a recession that month: ${d.actual_rec == null ? '?' : d.actual_rec ? 'yes' : 'no'}. ` +
    `Recession within the following ${SUMMARY.horizon_months} months: ${actual}.`;
}

async function drawImportance() {
  const d = await api('/api/importance'), c = C();
  const f = [...d.features].reverse();
  Plotly.react('coefChart', [{
    type: 'bar', orientation: 'h', y: f.map(r => r.label), x: f.map(r => r.coef_per_sd),
    marker: { color: f.map(r => r.coef_per_sd >= 0 ? c.pos : c.neg) },
    customdata: f.map(r => r.odds_ratio_per_sd),
    hovertemplate: '%{y}<br>%{x:+.2f} log-odds per SD<br>odds × %{customdata:.2f}<extra></extra>',
  }], baseLayout({ hovermode: 'closest', margin: { l: 200, r: 16, t: 10, b: 40 },
     xaxis: { title: 'log-odds per +1 SD', gridcolor: c.line, zerolinecolor: c.muted } }), config);
  const g = [...d.features].sort((a, b) => a.mean_abs_contribution - b.mean_abs_contribution);
  Plotly.react('impactChart', [{
    type: 'bar', orientation: 'h', y: g.map(r => r.label), x: g.map(r => r.mean_abs_contribution),
    marker: { color: c.accent }, hovertemplate: '%{y}<br>%{x:.2f}<extra></extra>',
  }], baseLayout({ hovermode: 'closest', margin: { l: 200, r: 16, t: 10, b: 40 },
     xaxis: { title: 'mean |contribution|', gridcolor: c.line } }), config);
}

function drawFeature(name) {
  const f = FEATS.features.find(x => x.feature === name), c = C();
  document.getElementById('featDesc').textContent = `${f.desc} Source: ${f.source}.`;
  const shapes = recShapes(FEATS.recessions);
  if (name === 'sahm') shapes.push({ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 0.5, y1: 0.5, line: { color: c.pos, dash: 'dash', width: 1 } });
  Plotly.react('featChart', [{ x: f.dates, y: f.values, mode: 'lines', name: f.label,
    line: { color: c.accent, width: 1.5 }, hovertemplate: '%{y:.3f}' }],
    baseLayout({ shapes, showlegend: false, yaxis: { title: f.unit, gridcolor: c.line, zerolinecolor: c.muted } }), config);
  document.getElementById('featStats').innerHTML = `
    <div>Latest (${f.current_date.slice(0, 7)})<b>${f.current.toFixed(3)}</b></div>
    <div>Historical percentile<b>${pct(f.percentile, 0)}</b></div>
    <div>Avg. in the ${SUMMARY.horizon_months} mo before recessions<b>${f.mean_in_recession_lead.toFixed(3)}</b></div>
    <div>Avg. otherwise<b>${f.mean_otherwise.toFixed(3)}</b></div>`;
}

async function drawEval() {
  const c = C();
  const m = SUMMARY.meta;
  document.getElementById('evalSub').textContent =
    `Walk-forward, refit yearly with a ${m.horizon_months}-month embargo · ${m.oos_start.slice(0, 7)} to ${m.oos_end.slice(0, 7)} · positive rate ${pct(m.positive_rate_oos)}`;
  const names = { full_model: 'Full model', term_only: 'Yield curve only', base_rate: 'Base rate' };
  document.getElementById('metricsTable').innerHTML =
    '<tr><th>Model</th><th>AUC</th><th>Brier</th><th>Brier skill</th><th>Log loss</th></tr>' +
    Object.entries(m.metrics).map(([k, v]) => `<tr><td>${names[k] || k}</td><td>${v.auc.toFixed(3)}</td>
      <td>${v.brier.toFixed(3)}</td><td>${signed(v.brier_skill)}</td><td>${v.log_loss.toFixed(3)}</td></tr>`).join('');
  try {
    const cal = await api('/api/calibration');
    const maxN = Math.max(...cal.bins.map(b => b.count));
    Plotly.react('calChart', [
      { x: [0, 1], y: [0, 1], mode: 'lines', name: 'Perfect', line: { color: c.muted, dash: 'dash', width: 1 }, hoverinfo: 'skip' },
      { x: cal.bins.map(b => b.mean_pred), y: cal.bins.map(b => b.observed), mode: 'lines+markers', name: 'Model',
        line: { color: c.accent2 }, marker: { size: cal.bins.map(b => 8 + 22 * Math.sqrt(b.count / maxN)), color: c.accent2 },
        customdata: cal.bins.map(b => b.count),
        hovertemplate: 'predicted %{x:.0%}<br>observed %{y:.0%}<br>%{customdata} months<extra></extra>' },
    ], baseLayout({ hovermode: 'closest', showlegend: false,
      xaxis: { title: 'mean predicted', tickformat: '.0%', range: [0, 1], gridcolor: c.line },
      yaxis: { title: 'observed frequency', tickformat: '.0%', range: [0, 1], gridcolor: c.line } }), config);
  } catch (e) { document.getElementById('calChart').textContent = e.message; }
}

async function loadAll() {
  [HIST, FEATS] = await Promise.all([api('/api/history'), api('/api/features'), loadSummary()]);
  const pick = document.getElementById('featPick');
  pick.innerHTML = FEATS.features.map(f => `<option value="${f.feature}">${f.label}</option>`).join('');
  drawProb(); drawContrib(); drawFeature(pick.value);
  await Promise.all([drawMonth(), drawImportance(), drawEval()]);
}

function wire() {
  ['showFit', 'showOos', 'showTerm'].forEach(id => document.getElementById(id).addEventListener('change', drawProb));
  document.getElementById('featPick').addEventListener('change', e => drawFeature(e.target.value));
  document.getElementById('monthPick').addEventListener('change', e => e.target.value && drawMonth(e.target.value).catch(showErr));
  document.getElementById('latestBtn').addEventListener('click', () => drawMonth().catch(showErr));
  const jump = ev => { const x = ev.points?.[0]?.x; if (x) { drawMonth(String(x).slice(0, 10)).catch(showErr);
    document.getElementById('waterfall').scrollIntoView({ behavior: 'smooth', block: 'center' }); } };
  document.getElementById('probChart').on('plotly_click', jump);
  document.getElementById('contribChart').on('plotly_click', jump);
  document.getElementById('refreshBtn').addEventListener('click', async e => {
    const b = e.target; b.disabled = true; b.textContent = 'Refreshing…';
    try { await api('/api/refresh', { method: 'POST' }); await loadAll(); }
    catch (err) { showErr(err); } finally { b.disabled = false; b.textContent = 'Refresh data'; }
  });
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => loadAll().catch(showErr));
}

loadAll().then(wire).catch(showErr);
</script>
</body>
</html>
"""
