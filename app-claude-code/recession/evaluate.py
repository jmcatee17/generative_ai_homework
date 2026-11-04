"""Walk-forward backtesting, metrics and diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             precision_recall_fscore_support, roc_auc_score, roc_curve)

from . import config
from .models import BASE_MODELS, make_model


def walk_forward(X: pd.DataFrame, y: pd.Series, horizon: int = config.HORIZON,
                 start: str = config.BACKTEST_START) -> pd.DataFrame:
    """Expanding-window backtest, refit every January.

    A row's label looks `horizon` months ahead, so training rows must satisfy
    t + horizon < test start (an embargo) — otherwise the training set would
    contain outcomes that are not yet known at forecast time.

    Every month with features is scored (including recent, still-unlabelled
    months), giving a continuous out-of-sample probability series.
    """
    labelled = y.notna()
    out = []
    for test_start in pd.date_range(start, X.index[-1], freq="YS"):
        cutoff = test_start - pd.DateOffset(months=horizon)
        tr = labelled & (X.index < cutoff)
        te = (X.index >= test_start) & (X.index < test_start + pd.DateOffset(years=1))
        if not te.any():
            continue
        Xtr, ytr = X[tr], y[tr].astype(int)
        preds = {}
        fitted = {m: clone(make_model(m)).fit(Xtr, ytr) for m in BASE_MODELS}
        for m, est in fitted.items():
            preds[m] = est.predict_proba(X[te])[:, 1]
        preds["ensemble"] = np.mean([preds[m] for m in BASE_MODELS], axis=0)
        out.append(pd.DataFrame(preds, index=X.index[te]))
    return pd.concat(out)


def metrics(y_true: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
    base = brier_score_loss(y_true, np.full_like(p, y_true.mean()))
    return {
        "roc_auc": float(roc_auc_score(y_true, p)),
        "pr_auc": float(average_precision_score(y_true, p)),
        "brier": float(brier_score_loss(y_true, p)),
        "brier_skill": float(1 - brier_score_loss(y_true, p) / base),
        "log_loss": float(log_loss(y_true, np.clip(p, 1e-6, 1 - 1e-6))),
        "precision": float(prec), "recall": float(rec), "f1": float(f1),
    }


def best_threshold(y_true: np.ndarray, p: np.ndarray) -> float:
    """Threshold maximising F1 on the backtest (grid 0.10–0.80)."""
    grid = np.round(np.arange(0.10, 0.801, 0.01), 2)
    scores = [precision_recall_fscore_support(y_true, (p >= t).astype(int), average="binary",
                                              zero_division=0)[2] for t in grid]
    return float(grid[int(np.argmax(scores))])


def rank_models(table: dict[str, dict]) -> tuple[str, dict]:
    """Champion = best average rank across PR-AUC, ROC-AUC, Brier and log loss.

    Ties are broken by PR-AUC (the most relevant metric for a rare event).
    """
    df = pd.DataFrame(table).T
    ranks = pd.DataFrame({
        "pr_auc": df["pr_auc"].rank(ascending=False),
        "roc_auc": df["roc_auc"].rank(ascending=False),
        "brier": df["brier"].rank(ascending=True),
        "log_loss": df["log_loss"].rank(ascending=True),
    })
    avg = ranks.mean(axis=1)
    order = sorted(avg.index, key=lambda m: (avg[m], -df.loc[m, "pr_auc"]))
    return order[0], {m: float(avg[m]) for m in order}


def roc_points(y_true, p, n: int = 60) -> dict:
    fpr, tpr, _ = roc_curve(y_true, p)
    idx = np.unique(np.linspace(0, len(fpr) - 1, min(n, len(fpr))).astype(int))
    return {"fpr": fpr[idx].round(4).tolist(), "tpr": tpr[idx].round(4).tolist()}


def calibration(y_true, p, bins: int = 10) -> dict:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = [(float(p[idx == b].mean()), float(y_true[idx == b].mean()), int((idx == b).sum()))
            for b in range(bins) if (idx == b).any()]
    return {"predicted": [r[0] for r in rows], "observed": [r[1] for r in rows], "count": [r[2] for r in rows]}


def recession_episodes(usrec: pd.Series, prob: pd.Series, threshold: float, horizon: int) -> list[dict]:
    """For each recession inside the backtest: peak pre-recession probability and lead time."""
    starts = usrec[(usrec == 1) & (usrec.shift(1) == 0)].index
    ends = usrec[(usrec == 1) & (usrec.shift(-1) == 0)].index
    out = []
    for s in starts:
        if s < prob.index[0] + pd.DateOffset(months=horizon):
            continue
        e = ends[ends >= s].min()
        window = prob[(prob.index >= s - pd.DateOffset(months=horizon)) & (prob.index <= s)]
        hits = window[window >= threshold]
        lead = None
        if len(hits):
            # Walk back from the first in-window hit to the start of its contiguous
            # above-threshold run (up to 2x horizon), so lead time isn't capped at 12.
            first = hits.index[0]
            lookback = prob[(prob.index >= s - pd.DateOffset(months=2 * horizon)) & (prob.index <= first)]
            below = lookback[lookback < threshold]
            if len(below):
                first = lookback.index[lookback.index > below.index[-1]][0]
            else:
                first = lookback.index[0]
            lead = (s.year - first.year) * 12 + (s.month - first.month)
        out.append({
            "start": s.strftime("%Y-%m"), "end": e.strftime("%Y-%m") if pd.notna(e) else None,
            "duration_months": int(((e.year - s.year) * 12 + e.month - s.month + 1)) if pd.notna(e) else None,
            "peak_prob_before": float(window.max()) if len(window) else None,
            "prob_at_start": float(prob.get(s, np.nan)),
            "lead_months": lead, "detected": lead is not None,
        })
    return out


def false_alarms(y: pd.Series, prob: pd.Series, threshold: float) -> int:
    """Distinct runs of above-threshold readings with no recession in the following window."""
    flag = ((prob >= threshold) & (y == 0)).astype(int)
    return int(((flag == 1) & (flag.shift(1, fill_value=0) == 0)).sum())


def holdout_permutation_importance(X: pd.DataFrame, y: pd.Series, name: str,
                                   split: str = "2005-01-01", horizon: int = config.HORIZON) -> dict:
    """Out-of-sample permutation importance: train before `split` (with embargo), score after."""
    labelled = y.notna()
    cutoff = pd.Timestamp(split) - pd.DateOffset(months=horizon)
    tr = labelled & (X.index < cutoff)
    te = labelled & (X.index >= split)
    model = clone(make_model(name)).fit(X[tr], y[tr].astype(int))
    r = permutation_importance(model, X[te], y[te].astype(int), scoring="roc_auc",
                               n_repeats=25, random_state=config.RANDOM_STATE, n_jobs=1)
    return {
        "mean": dict(zip(X.columns, r.importances_mean.round(5).tolist())),
        "std": dict(zip(X.columns, r.importances_std.round(5).tolist())),
        "holdout": f"{split[:7]} → {X.index[te][-1]:%Y-%m}",
    }
