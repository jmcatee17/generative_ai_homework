"""Train, compare and promote the recession model.

    python train.py              # use cached FRED data if fresh
    python train.py --refresh    # force a fresh FRED pull
    python train.py --force      # promote even if the champion underperforms production

Pipeline
  1. Pull FRED series, engineer 13 leading indicators, build the 12-month target.
  2. Expanding-window walk-forward backtest (annual refits, 12-month embargo)
     for logistic regression, random forest, XGBoost and their soft-vote ensemble.
  3. Rank on out-of-sample PR-AUC, ROC-AUC, Brier score and log loss.
  4. Refit the champion on all labelled data (+ 3–24 month horizon variants),
     register the version and promote it to production if it clears the gate.
"""
from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.base import clone

from recession import config, data, evaluate, registry
from recession.inference import round_floats
from recession.models import BASE_MODELS, SoftVotingEnsemble, hyperparameters, make_model, native_importance

ALL_MODELS = [*BASE_MODELS, "ensemble"]
PROMOTION_TOLERANCE = 0.01  # allowed PR-AUC slippage vs. current production

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("train")


def fit_full(name: str, X: pd.DataFrame, y: pd.Series):
    m = y.notna()
    return clone(make_model(name)).fit(X[m], y[m].astype(int))


def main(refresh: bool, force: bool) -> None:
    t0 = time.time()
    X, y, usrec, fetch_meta = data.load_dataset(force=refresh)
    labelled = y.notna()
    log.info("Data: %d months (%s → %s), %d labelled, positive rate %.1f%%  [%s]",
             len(X), f"{X.index[0]:%Y-%m}", f"{X.index[-1]:%Y-%m}", labelled.sum(),
             100 * y[labelled].mean(), fetch_meta["source"])

    # ---- 1. walk-forward backtest -------------------------------------------------
    oos = evaluate.walk_forward(X, y)
    yo = y.reindex(oos.index)
    m = yo.notna()
    y_true = yo[m].astype(int).values
    log.info("Backtest: %d OOS months (%d labelled), %d refits",
             len(oos), m.sum(), oos.index.year.nunique())

    # ---- 2. metrics & champion selection ------------------------------------------
    table, thresholds = {}, {}
    for name in ALL_MODELS:
        p = oos.loc[m, name].values
        thresholds[name] = evaluate.best_threshold(y_true, p)
        table[name] = evaluate.metrics(y_true, p, thresholds[name])
        table[name]["threshold"] = thresholds[name]
        table[name]["false_alarms"] = evaluate.false_alarms(yo, oos[name], thresholds[name])
        eps = evaluate.recession_episodes(usrec, oos[name], thresholds[name], config.HORIZON)
        table[name]["recessions_flagged"] = f"{sum(e['detected'] for e in eps)}/{len(eps)}"
        table[name]["avg_lead_months"] = float(np.mean([e["lead_months"] for e in eps if e["detected"]] or [0]))

    champion, avg_rank = evaluate.rank_models(table)
    log.info("\n%s", pd.DataFrame(table).T[["roc_auc", "pr_auc", "brier", "log_loss", "f1", "threshold",
                                            "recessions_flagged", "false_alarms"]].round(3).to_string())
    log.info("Champion: %s (avg rank %.2f)", champion, avg_rank[champion])

    # ---- 3. final fits -------------------------------------------------------------
    full = {n: fit_full(n, X, y) for n in BASE_MODELS}
    ens = SoftVotingEnsemble()
    ens.estimators_, ens.classes_ = {n: full[n] for n in BASE_MODELS}, np.array([0, 1])
    full["ensemble"] = ens

    horizon_models = {}
    for h in config.TERM_STRUCTURE_HORIZONS:
        yh = data.make_target(usrec, h).reindex(X.index)
        horizon_models[h] = full[champion] if h == config.HORIZON else fit_full(champion, X, yh)

    expansion = X[usrec.reindex(X.index).fillna(0) == 0]
    bundle = {
        "name": champion, "model": full[champion], "horizon_models": horizon_models,
        "features": config.FEATURE_NAMES, "baseline": expansion.median().to_dict(),
        "threshold": thresholds[champion],
    }

    # ---- 4. importances -------------------------------------------------------------
    log.info("Computing permutation importance on 2005+ holdout…")
    importance = {
        n: {"native": native_importance(full[n], n, config.FEATURE_NAMES),
            "permutation": evaluate.holdout_permutation_importance(X, y, n)}
        for n in ALL_MODELS
    }
    if champion == "logistic" or "logistic" in ALL_MODELS:
        lr = full["logistic"]
        importance["logistic"]["coefficients"] = dict(
            zip(config.FEATURE_NAMES, lr.named_steps["clf"].coef_[0].round(4).tolist()))

    # ---- 5. register & promote ----------------------------------------------------
    version = registry.new_version()
    trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    metadata = round_floats({
        "version": version, "trained_at": trained_at, "champion": champion,
        "champion_label": config.MODEL_LABELS[champion], "horizon_months": config.HORIZON,
        "features": config.FEATURE_NAMES, "threshold": thresholds[champion],
        "metrics": table, "avg_rank": avg_rank,
        "selection_rule": "Lowest average rank across out-of-sample PR-AUC, ROC-AUC, Brier score and log loss; ties broken by PR-AUC.",
        "hyperparameters": {n: hyperparameters(n) for n in ALL_MODELS},
        "data": {**fetch_meta, "sample_start": f"{X.index[0]:%Y-%m}", "sample_end": f"{X.index[-1]:%Y-%m}",
                 "labelled_end": f"{y[labelled].index[-1]:%Y-%m}", "n_labelled": int(labelled.sum()),
                 "positive_rate": float(y[labelled].mean())},
        "backtest": {"start": config.BACKTEST_START[:7], "end": f"{oos.index[-1]:%Y-%m}",
                     "n_months": int(m.sum()), "refit": "annual (January), expanding window",
                     "embargo_months": config.HORIZON},
    })
    report = round_floats({
        "dates": [d.strftime("%Y-%m-%d") for d in oos.index],
        "oos": {n: oos[n].tolist() for n in ALL_MODELS},
        "target": [None if pd.isna(v) else int(v) for v in yo],
        "roc": {n: evaluate.roc_points(y_true, oos.loc[m, n].values) for n in ALL_MODELS},
        "calibration": {n: evaluate.calibration(y_true, oos.loc[m, n].values) for n in ALL_MODELS},
        "episodes": {n: evaluate.recession_episodes(usrec, oos[n], thresholds[n], config.HORIZON)
                     for n in ALL_MODELS},
        "importance": importance,
    })
    registry.save_version(version, bundle, metadata, report)

    prod = registry.production_version()
    new_score = table[champion]["pr_auc"]
    if prod is None or force:
        promote, why = True, "initial deployment" if prod is None else "forced promotion"
    else:
        old = registry.load_metadata(prod)
        old_score = old["metrics"][old["champion"]]["pr_auc"]
        promote = new_score >= old_score - PROMOTION_TOLERANCE
        why = (f"PR-AUC {new_score:.3f} vs production {old_score:.3f} "
               f"({'within' if promote else 'outside'} {PROMOTION_TOLERANCE} tolerance)")
    if promote:
        registry.promote(version, f"{config.MODEL_LABELS[champion]} champion — {why}")
        log.info("Promoted %s to production (%s)", version, why)
    else:
        log.warning("Registered %s but kept %s in production: %s", version, prod, why)
    log.info("Done in %.0fs", time.time() - t0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="force a fresh FRED download")
    ap.add_argument("--force", action="store_true", help="promote regardless of the promotion gate")
    args = ap.parse_args()
    main(args.refresh, args.force)
