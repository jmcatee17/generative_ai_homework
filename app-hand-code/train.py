from fredapi import Fred
import pandas as pd, numpy as np
import os
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import brier_score_loss, roc_auc_score
import joblib
from pathlib import Path
import json

load_dotenv()      
fred = Fred(api_key=os.environ["FRED_API_KEY"])
get = lambda s: fred.get_series(s, frequency="m", aggregation_method="avg")

df = pd.DataFrame({
    "rec":     get("USREC"),
    "term":    get("GS10") - get("TB3MS"),
    "credit":  get("BAA") - get("GS10"),
    "sahm":    get("SAHMREALTIME"),
    "claims":  np.log(get("ICSA")).diff(6),          # 6-mo log change
    "permits": np.log(get("PERMIT")).diff(12),
    "ip_yoy":  np.log(get("INDPRO")).diff(12),
})

H = 12
fwd = pd.concat([df["rec"].shift(-k) for k in range(1, H + 1)], axis=1)
df["y"] = fwd.max(axis=1)        # NaN for the last H months (unknown yet)

feats = ["term", "credit", "sahm", "claims", "permits", "ip_yoy"]
data = df.dropna(subset=feats + ["y"])

# Walk-forward: refit each year, predict the next year, with a 12-month embargo
preds = []
for test_start in pd.date_range("1990-01-01", data.index[-1], freq="YS"):
    train = data[data.index < test_start - pd.DateOffset(months=H)]
    test  = data[(data.index >= test_start) & (data.index < test_start + pd.DateOffset(years=1))]
    if train["y"].nunique() < 2 or test.empty:
        continue
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.5))
    model.fit(train[feats], train["y"])
    preds.append(pd.Series(model.predict_proba(test[feats])[:, 1], index=test.index))

p = pd.concat(preds)
y = data.loc[p.index, "y"]
print("AUC:", roc_auc_score(y, p), " Brier:", brier_score_loss(y, p))


models_dir = Path(__file__).parent / "models"

joblib.dump(model, models_dir / "recession_model.joblib")

# Save what's needed to use the model correctly later
meta = {
    "features": feats,
    "horizon_months": H,
    "train_start": str(data.index[0].date()),
    "train_end": str(data.index[-1].date()),
    "oos_auc": float(roc_auc_score(y, p)),
    "oos_brier": float(brier_score_loss(y, p)),
}
(models_dir / "recession_model_meta.json").write_text(json.dumps(meta, indent=2))
print("Saved to", models_dir)