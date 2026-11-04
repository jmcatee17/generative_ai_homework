"""Smoke tests against the production model. Run `python train.py` first."""
import pytest
from fastapi.testclient import TestClient

from app.main import app
from recession import config


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/api/health").json()["status"] == "ok"


def test_summary_probability_in_range(client):
    s = client.get("/api/summary").json()
    assert 0 <= s["probability"] <= 1
    assert s["band"]["level"] in {"low", "elevated", "high"}


def test_model_card_has_all_four_models(client):
    m = client.get("/api/model").json()
    assert set(m["metrics"]) == {"logistic", "random_forest", "xgboost", "ensemble"}
    assert m["champion"] in m["metrics"]


def test_term_structure_covers_horizons(client):
    f = client.get("/api/forecast").json()
    assert set(map(int, f["snapshots"]["Now"]["curve"])) == set(config.TERM_STRUCTURE_HORIZONS)


def test_scenario_without_overrides_matches_baseline(client):
    r = client.post("/api/scenario", json={"overrides": {}}).json()
    assert r["probability"] == pytest.approx(r["baseline_probability"], abs=1e-4)


def test_scenario_rejects_unknown_feature(client):
    assert client.post("/api/scenario", json={"overrides": {"bogus": 1}}).status_code == 422


def test_inverted_curve_raises_risk(client):
    base = client.post("/api/scenario", json={"overrides": {}}).json()["probability"]
    inv = client.post("/api/scenario", json={"overrides": {"term_spread": -1.5, "term_spread_min12": -1.8}}).json()
    assert inv["probability"] > base


def test_historical_predict(client):
    r = client.get("/api/predict", params={"month": "2007-06"}).json()
    assert r["recession_within_12m"] == 1
