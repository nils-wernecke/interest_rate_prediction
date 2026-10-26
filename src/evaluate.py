import pandas as pd
from sklearn.model_selection import KFold, cross_validate

CV = KFold(n_splits=5, shuffle=True, random_state=9)
SCORING = {"r2": "r2",
           "rmse": "neg_root_mean_squared_error",
           "mae": "neg_mean_absolute_error"}


def cv_report(models: dict, X, y) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        res = cross_validate(model, X, y, cv=CV, scoring=SCORING)
        rows.append({
            "model": name,
            "r2": res["test_r2"].mean(),
            "r2_std": res["test_r2"].std(),
            "rmse": -res["test_rmse"].mean(),
            "mae": -res["test_mae"].mean(),
        })
    return pd.DataFrame(rows).set_index("model").round(4)