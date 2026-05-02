from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
from scipy import stats


def regression_metrics(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    mse = float(np.mean(err**2))
    mae = float(np.mean(np.abs(err)))
    return {"MSE": mse, "MAE": mae, "RMSE": float(math.sqrt(mse))}


def directional_accuracy(
    y_true: pd.Series,
    y_pred: pd.Series | np.ndarray,
    current_vol: pd.Series,
) -> float:
    pred_delta = np.asarray(y_pred) - np.asarray(current_vol)
    true_delta = np.asarray(y_true) - np.asarray(current_vol)
    if np.allclose(pred_delta, 0):
        return float("nan")
    return float(np.mean(np.sign(pred_delta) == np.sign(true_delta)))


def diebold_mariano(
    y_true: pd.Series | np.ndarray,
    pred_a: pd.Series | np.ndarray,
    pred_b: pd.Series | np.ndarray,
    power: int = 2,
) -> dict[str, float]:
    """Small-sample Diebold-Mariano style paired loss-difference test."""
    y_true = np.asarray(y_true, dtype=float)
    loss_a = np.abs(y_true - np.asarray(pred_a, dtype=float)) ** power
    loss_b = np.abs(y_true - np.asarray(pred_b, dtype=float)) ** power
    diff = loss_a - loss_b
    diff = diff[np.isfinite(diff)]
    if len(diff) < 3 or np.isclose(np.std(diff, ddof=1), 0):
        return {"dm_stat": np.nan, "p_value": np.nan}
    dm_stat = float(diff.mean() / (diff.std(ddof=1) / math.sqrt(len(diff))))
    p_value = float(2 * stats.t.sf(abs(dm_stat), df=len(diff) - 1))
    return {"dm_stat": dm_stat, "p_value": p_value}


def build_metrics_table(
    y_true: pd.Series,
    predictions: dict[str, pd.Series],
    current_vol: pd.Series,
    runtimes: dict[str, float],
    notes: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    direction_rows = []
    for model, pred in predictions.items():
        aligned = pd.concat([y_true.rename("actual"), pred.rename("pred"), current_vol.rename("current")], axis=1).dropna()
        scores = regression_metrics(aligned["actual"], aligned["pred"])
        metric_rows.append(
            {
                "model": model,
                **scores,
                "runtime_seconds": runtimes.get(model, np.nan),
                "notes": notes.get(model, ""),
            }
        )
        direction_rows.append(
            {
                "model": model,
                "directional_accuracy": directional_accuracy(
                    aligned["actual"], aligned["pred"], aligned["current"]
                ),
            }
        )

    dm_rows = []
    for model_a, model_b in itertools.combinations(predictions.keys(), 2):
        aligned = pd.concat(
            [
                y_true.rename("actual"),
                predictions[model_a].rename("a"),
                predictions[model_b].rename("b"),
            ],
            axis=1,
        ).dropna()
        dm = diebold_mariano(aligned["actual"], aligned["a"], aligned["b"])
        dm_rows.append({"model_a": model_a, "model_b": model_b, **dm})

    return pd.DataFrame(metric_rows), pd.DataFrame(direction_rows), pd.DataFrame(dm_rows)
