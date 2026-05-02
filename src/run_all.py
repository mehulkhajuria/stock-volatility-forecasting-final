from __future__ import annotations

import json
import math
from datetime import datetime

import pandas as pd

from .config import (
    FIGURES_DIR,
    PROCESSED_DIR,
    RESULTS_DIR,
    TARGET_SYMBOL,
    TRAIN_END,
    VALIDATION_END,
    ensure_directories,
)
from .download_data import data_inventory, download_all
from .evaluate import build_metrics_table
from .features import build_feature_frames, build_spy_model_frame, split_by_date
from .plots import (
    create_contact_sheet,
    plot_correlation_heatmap,
    plot_error_distribution,
    plot_feature_importance,
    plot_model_comparison,
    plot_predicted_vs_actual,
    plot_volatility_timeseries,
)
from .train_baselines import garch_forecast, naive_persistence, rolling_mean
from .train_lstm import train_lstm_or_fallback
from .train_ml import permutation_importance_fallback, train_gradient_boosting, train_random_forest


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def main() -> None:
    ensure_directories()
    raw_frames = download_all()
    feature_frames = build_feature_frames(raw_frames)
    model_df = build_spy_model_frame(feature_frames, TARGET_SYMBOL)
    train, validation, test = split_by_date(model_df, TRAIN_END, VALIDATION_END)
    train_validation = pd.concat([train, validation])

    predictions = {}
    runtimes = {}
    notes = {}

    for name, result in [
        ("Naive Persistence", naive_persistence(test)),
        ("Rolling Mean", rolling_mean(test)),
        ("GARCH(1,1)", garch_forecast(train_validation, test)),
    ]:
        predictions[name], runtimes[name], notes[name] = result

    rf_pred, rf_time, rf_note, rf_model = train_random_forest(train_validation, test)
    predictions["Random Forest"], runtimes["Random Forest"], notes["Random Forest"] = rf_pred, rf_time, rf_note

    gb_pred, gb_time, gb_note, gb_model = train_gradient_boosting(train_validation, test)
    predictions["Gradient Boosting"], runtimes["Gradient Boosting"], notes["Gradient Boosting"] = gb_pred, gb_time, gb_note

    lstm_pred, lstm_time, lstm_note = train_lstm_or_fallback(train, validation, test)
    neural_name = "MLP neural fallback"
    predictions[neural_name], runtimes[neural_name], notes[neural_name] = lstm_pred, lstm_time, lstm_note

    y_true = test["target_future_vol"]
    current_vol = test["realized_vol_5"]
    metrics, directional, dm = build_metrics_table(y_true, predictions, current_vol, runtimes, notes)
    metrics = metrics.sort_values("RMSE")
    metrics.to_csv(RESULTS_DIR / "metrics.csv", index=False)
    directional.to_csv(RESULTS_DIR / "directional_accuracy.csv", index=False)
    dm.to_csv(RESULTS_DIR / "dm_test_results.csv", index=False)

    pred_frame = pd.DataFrame({"actual": y_true, **{k: v for k, v in predictions.items()}})
    pred_frame.to_csv(RESULTS_DIR / "predictions.csv")
    data_inventory().to_csv(RESULTS_DIR / "data_inventory.csv", index=False)

    importances = permutation_importance_fallback(rf_model, train_validation, test)
    importances.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)

    best_model = metrics.iloc[0]["model"]
    plot_volatility_timeseries(feature_frames[TARGET_SYMBOL])
    plot_correlation_heatmap(feature_frames)
    plot_predicted_vs_actual(y_true, predictions, best_model)
    plot_model_comparison(metrics)
    plot_error_distribution(y_true, predictions)
    plot_feature_importance(importances)
    create_contact_sheet()

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "target_symbol": TARGET_SYMBOL,
        "rows": int(len(model_df)),
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "test_start": test.index.min().date().isoformat(),
        "test_end": test.index.max().date().isoformat(),
        "best_model_by_rmse": best_model,
        "metrics": metrics.to_dict(orient="records"),
        "directional_accuracy": directional.to_dict(orient="records"),
        "model_notes": notes,
    }
    summary = _json_safe(summary)
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
