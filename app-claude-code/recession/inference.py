"""Scoring with the production bundle: live probability, drivers, term structure, scenarios."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def risk_band(p: float, threshold: float) -> dict:
    if p >= threshold:
        return {"level": "high", "label": "Recession signal"}
    if p >= threshold / 2:
        return {"level": "elevated", "label": "Elevated"}
    return {"level": "low", "label": "Low"}


def score(bundle: dict, X: pd.DataFrame) -> pd.Series:
    return pd.Series(bundle["model"].predict_proba(X[bundle["features"]])[:, 1], index=X.index)


def term_structure(bundle: dict, row: pd.DataFrame) -> dict[int, float]:
    return {h: float(m.predict_proba(row[bundle["features"]])[:, 1][0])
            for h, m in sorted(bundle["horizon_models"].items())}


def drivers(bundle: dict, row: pd.DataFrame) -> list[dict]:
    """Occlusion attribution: change in probability when one feature is reset to its
    long-run expansion median, holding the others at today's values.

    Positive = this indicator is currently *raising* recession risk.
    """
    base_p = float(bundle["model"].predict_proba(row[bundle["features"]])[:, 1][0])
    out = []
    for f in bundle["features"]:
        alt = row[bundle["features"]].copy()
        alt[f] = bundle["baseline"][f]
        p = float(bundle["model"].predict_proba(alt)[:, 1][0])
        out.append({"feature": f, "label": config.FEATURES[f]["label"], "value": float(row[f].iloc[0]),
                    "baseline": float(bundle["baseline"][f]), "contribution": base_p - p})
    return sorted(out, key=lambda d: -abs(d["contribution"]))


def scenario(bundle: dict, row: pd.DataFrame, overrides: dict[str, float]) -> dict:
    alt = row[bundle["features"]].copy()
    for k, v in overrides.items():
        if k in alt.columns:
            alt[k] = float(v)
    return {
        "probability": float(bundle["model"].predict_proba(alt)[:, 1][0]),
        "term_structure": term_structure(bundle, alt),
    }


def indicator_context(X: pd.DataFrame, usrec: pd.Series) -> list[dict]:
    """Latest value of each feature with its historical percentile and recession-era norms."""
    latest = X.iloc[-1]
    rec = usrec.reindex(X.index).fillna(0).astype(bool)
    out = []
    for f, info in config.FEATURES.items():
        s = X[f]
        pct = float((s < latest[f]).mean())
        # Percentile on the "risky" side: high when the reading points toward recession.
        risk_pct = pct if info["direction"] > 0 else 1 - pct
        out.append({
            "feature": f, **info, "value": float(latest[f]), "prev_3m": float(s.iloc[-4]),
            "prev_12m": float(s.iloc[-13]), "percentile": pct, "risk_percentile": risk_pct,
            "expansion_median": float(s[~rec].median()), "recession_median": float(s[rec].median()),
            "p01": float(s.quantile(0.01)), "p99": float(s.quantile(0.99)),
        })
    return out


def round_floats(obj, nd: int = 5):
    if isinstance(obj, float):
        return None if np.isnan(obj) else round(obj, nd)
    if isinstance(obj, dict):
        return {k: round_floats(v, nd) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_floats(v, nd) for v in obj]
    return obj
