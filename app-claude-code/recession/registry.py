"""Minimal file-based model registry with a production pointer."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib

from . import config


def new_version() -> str:
    return datetime.now(timezone.utc).strftime("v%Y%m%d-%H%M%S")


def save_version(version: str, bundle: dict, metadata: dict, report: dict) -> Path:
    d = config.REGISTRY_DIR / version
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, d / "model.joblib")
    (d / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (d / "report.json").write_text(json.dumps(report))
    return d


def production_version() -> str | None:
    if not config.PRODUCTION_POINTER.exists():
        return None
    return json.loads(config.PRODUCTION_POINTER.read_text())["version"]


def promote(version: str, reason: str) -> None:
    pointer = {
        "version": version,
        "promoted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reason": reason,
        "previous": production_version(),
    }
    config.PRODUCTION_POINTER.write_text(json.dumps(pointer, indent=2))


def load_metadata(version: str) -> dict:
    return json.loads((config.REGISTRY_DIR / version / "metadata.json").read_text())


def load_production() -> tuple[dict, dict, dict, dict]:
    """Return (bundle, metadata, report, pointer) for the production model."""
    if not config.PRODUCTION_POINTER.exists():
        raise FileNotFoundError("No production model. Run `python train.py` first.")
    pointer = json.loads(config.PRODUCTION_POINTER.read_text())
    d = config.REGISTRY_DIR / pointer["version"]
    bundle = joblib.load(d / "model.joblib")
    return bundle, load_metadata(pointer["version"]), json.loads((d / "report.json").read_text()), pointer


def list_versions() -> list[dict]:
    prod = production_version()
    out = []
    for d in sorted(config.REGISTRY_DIR.glob("v*"), reverse=True):
        m = json.loads((d / "metadata.json").read_text())
        out.append({
            "version": d.name, "champion": m["champion"], "trained_at": m["trained_at"],
            "pr_auc": m["metrics"][m["champion"]]["pr_auc"], "production": d.name == prod,
        })
    return out
