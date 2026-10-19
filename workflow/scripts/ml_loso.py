"""Shows how random K-fold CV leaks participant identity and inflates
performance relative to leave-one-subject-out (LOSO) CV."""
import sys; sys.path.insert(0, snakemake.params.src)  # noqa: E702
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

cfg = snakemake.params.cfg
panel = pd.read_csv(snakemake.input.panel)
rows = []
for outcome in cfg["outcomes"]:
    d = panel.dropna(subset=[outcome])
    X, y, groups = d[cfg["features"]], d[outcome].to_numpy(), d["uid"]
    models = {
        "ridge": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))),
        "random_forest": make_pipeline(SimpleImputer(strategy="median"),
                                       RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=0, n_jobs=-1)),
    }
    for mname, model in models.items():
        for cvname, cv in (("random_5fold", KFold(5, shuffle=True, random_state=0)), ("LOSO", LeaveOneGroupOut())):
            pred = cross_val_predict(model, X, y, cv=cv, groups=groups if cvname == "LOSO" else None)
            rows.append({"outcome": outcome, "model": mname, "cv": cvname,
                         "r2": r2_score(y, pred), "mae": mean_absolute_error(y, pred),
                         "n_obs": len(d), "n_persons": groups.nunique()})
pd.DataFrame(rows).to_csv(snakemake.output[0], index=False)
