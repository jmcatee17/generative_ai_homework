"""Model zoo: logistic regression, random forest, XGBoost and a soft-voting ensemble."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .config import RANDOM_STATE

BASE_MODELS = ("logistic", "random_forest", "xgboost")


def make_model(name: str):
    if name == "logistic":
        return Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=0.3, max_iter=2000)),
        ])
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=400, max_depth=6, min_samples_leaf=12, max_features="sqrt",
            n_jobs=-1, random_state=RANDOM_STATE,
        )
    if name == "xgboost":
        return XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=5, reg_lambda=2.0,
            eval_metric="logloss", n_jobs=4, random_state=RANDOM_STATE,
        )
    if name == "ensemble":
        return SoftVotingEnsemble()
    raise ValueError(f"unknown model {name!r}")


class SoftVotingEnsemble(ClassifierMixin, BaseEstimator):
    """Equal-weight average of the three base models' probabilities."""

    def __init__(self, members: tuple[str, ...] = BASE_MODELS):
        self.members = members

    def fit(self, X, y):
        self.estimators_ = {m: clone(make_model(m)).fit(X, y) for m in self.members}
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return np.mean([est.predict_proba(X) for est in self.estimators_.values()], axis=0)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def hyperparameters(name: str) -> dict:
    """Readable hyperparameters for the model card."""
    if name == "ensemble":
        return {"members": list(BASE_MODELS), "weights": "equal (1/3 each)", "voting": "soft (probability average)"}
    m = make_model(name)
    params = m.named_steps["clf"].get_params() if name == "logistic" else m.get_params()
    keep = {
        "logistic": ["C", "penalty", "solver", "max_iter"],
        "random_forest": ["n_estimators", "max_depth", "min_samples_leaf", "max_features"],
        "xgboost": ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree",
                    "min_child_weight", "reg_lambda"],
    }[name]
    out = {k: params[k] for k in keep}
    if name == "logistic":
        out["preprocessing"] = "StandardScaler"
    return out


def native_importance(model, name: str, features: list[str]) -> dict[str, float]:
    """Model-native importance, normalised to sum to 1."""
    if name == "ensemble":
        parts = [native_importance(model.estimators_[m], m, features) for m in model.members]
        return {f: float(np.mean([p[f] for p in parts])) for f in features}
    if name == "logistic":
        raw = np.abs(model.named_steps["clf"].coef_[0])
    else:
        raw = np.asarray(model.feature_importances_, dtype=float)
    raw = raw / raw.sum() if raw.sum() > 0 else raw
    return dict(zip(features, map(float, raw)))
