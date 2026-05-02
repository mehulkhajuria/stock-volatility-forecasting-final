from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_SEED


def feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"target_future_vol"}
    return [col for col in df.columns if col not in excluded]


def _xy(df: pd.DataFrame, columns: list[str]):
    return df[columns], df["target_future_vol"]


def train_random_forest(train_validation: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.Series, float, str, object]:
    start = time.perf_counter()
    columns = feature_columns(train_validation)
    x_train, y_train = _xy(train_validation, columns)
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=5,
        random_state=RANDOM_SEED,
        n_jobs=1,
    )
    model.fit(x_train, y_train)
    pred = pd.Series(model.predict(test[columns]), index=test.index, name="Random Forest")
    return pred, time.perf_counter() - start, "RandomForestRegressor with fixed seed.", model


def train_gradient_boosting(train_validation: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.Series, float, str, object]:
    start = time.perf_counter()
    columns = feature_columns(train_validation)
    x_train, y_train = _xy(train_validation, columns)
    try:
        from xgboost import XGBRegressor

        model = XGBRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=RANDOM_SEED,
            n_jobs=1,
        )
        note = "XGBoost XGBRegressor."
    except Exception as exc:
        model = HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.04,
            max_leaf_nodes=31,
            l2_regularization=0.01,
            random_state=RANDOM_SEED,
        )
        note = f"XGBoost unavailable; used sklearn HistGradientBoostingRegressor. Error: {exc}"
    model.fit(x_train, y_train)
    pred = pd.Series(model.predict(test[columns]), index=test.index, name="Gradient Boosting")
    return pred, time.perf_counter() - start, note, model


def permutation_importance_fallback(model: object, train_validation: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    columns = feature_columns(train_validation)
    if hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_, dtype=float)
        return pd.DataFrame({"feature": columns, "importance": importance}).sort_values("importance", ascending=False)
    # HistGradientBoosting does not expose importances. A compact correlation proxy is enough for interpretation.
    corrs = train_validation[columns].corrwith(train_validation["target_future_vol"]).abs().fillna(0)
    return pd.DataFrame({"feature": corrs.index, "importance": corrs.values}).sort_values("importance", ascending=False)
