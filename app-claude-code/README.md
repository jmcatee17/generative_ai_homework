# Recession Monitor

A FastAPI app that estimates the probability of a US recession within the next 12 months. It uses 13 leading indicators from FRED.

Four candidate models are compared in a walk-forward backtest:
- logistic regression
- random forest
- XGBoost
- a soft-voting ensemble of the three

The best one is promoted to production through a small file-based model registry.

## Quick start

```bash
cd app-claude-code
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env          # then paste your FRED_API_KEY into .env
python train.py --refresh     # pull data, backtest, select and promote the champion (~30s)
uvicorn app.main:app --port 8000
```

Open http://127.0.0.1:8000. The interactive API docs are at `/docs`.

Without `FRED_API_KEY` the app still runs. It falls back to FRED's public CSV endpoint.

## How it works

| Stage | What happens |
|---|---|
| **Data** (`recession/data.py`) | 14 FRED series, resampled to monthly averages (1971 onward). The "as-of" month is the latest month in which at least 75% of series have actually been published. |
| **Features** | Yield curve (10y−3m) and its 12-month minimum, Baa credit spread, NFCI, 12-month change in fed funds, Sahm rule, unemployment change, initial claims, payrolls, industrial production, real income ex. transfers, building permits, WTI oil. |
| **Target** | 1 if any month in *t+1 … t+12* is an NBER recession month (`USREC`). |
| **Backtest** (`recession/evaluate.py`) | Expanding window from 1985, refit every January. A 12-month embargo keeps unresolved labels out of training. |
| **Selection** | Lowest average rank across out-of-sample PR-AUC, ROC-AUC, Brier score and log loss. |
| **Registry** (`recession/registry.py`) | Each run is saved to `artifacts/registry/<version>/`. `artifacts/production.json` points to the version that is serving. A new champion is promoted only if its PR-AUC is within 0.01 of the current production model's (use `--force` to override). |
| **Serving** (`app/main.py`) | Loads the production version, scores the latest data, and serves the dashboard and JSON API. `POST /api/refresh` re-pulls FRED data without retraining. |

### Current results (backtest 1985–2026, 4 recessions)

| Model | ROC-AUC | PR-AUC | Brier | Log loss | False alarms |
|---|---|---|---|---|---|
| **Logistic Regression (production)** | **0.875** | **0.460** | 0.122 | 0.393 | **6** |
| Ensemble | 0.872 | 0.421 | 0.124 | 0.366 | 10 |
| Random Forest | 0.866 | 0.387 | **0.119** | **0.357** | 11 |
| XGBoost | 0.847 | 0.416 | 0.149 | 0.438 | 12 |

All four models flagged all four recessions. The production model's signals came 15.5 months ahead of the NBER start on average. Numbers change after each retrain.

## Dashboard

- **Overview**: 12-month probability, risk band, changes over 1, 3 and 12 months, a forward probability curve (3–24 months), and the production model's credentials.
- **History**: out-of-sample probabilities for every model, with NBER recession shading and the signal threshold.
- **Drivers**: what is pushing today's reading up or down, plus out-of-sample permutation importance and model-native importance (coefficients, impurity, gain).
- **Models**: metrics table, ROC, calibration, precision/recall, and a recession-by-recession call record with lead times.
- **Indicators**: every input with its percentile, sparkline and full-history chart.
- **Scenarios**: shock any indicator, or replay conditions from 2000 or 2007, and re-score live.
- **Methodology**: model card, hyperparameters, data freshness and registry versions.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/summary` | Current probability, band, model and data status |
| GET | `/api/forecast` | Probability term structure now and 3, 6 and 12 months ago |
| GET | `/api/history` | Out-of-sample backtest series for all models |
| GET | `/api/drivers` | Attribution of today's reading |
| GET | `/api/importance` | Permutation and native importances |
| GET | `/api/backtest` | ROC, calibration, episodes, metrics |
| GET | `/api/indicators` | Feature history and context |
| GET | `/api/predict?month=YYYY-MM` | Score any historical month |
| POST | `/api/scenario` | `{"overrides": {"term_spread": -1.0}}` |
| POST | `/api/refresh` | Re-pull FRED data and re-score |
| GET | `/api/model`, `/api/health` | Model card and registry, health check |

## Deploy

```bash
python train.py --refresh
docker build -t recession-monitor .
docker run -p 8000:8000 -e FRED_API_KEY=... recession-monitor
```

The image contains the trained registry, so the container serves the promoted model without retraining. Retrain on a schedule (monthly, after the jobs report) and rebuild.

## Caveats

- Features use revised data, not the real-time vintages available at the time, so the backtest is somewhat optimistic.
- There are only four recessions in the out-of-sample window.
- Scenario shocks move one indicator at a time on correlated inputs.
- This is not investment advice.
