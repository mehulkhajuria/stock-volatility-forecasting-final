from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image

from .config import FIGURES_DIR, PAPER_FIGURES_DIR, TARGET_WINDOW, ensure_directories

sns.set_theme(style="whitegrid", context="paper")


def _save(fig: plt.Figure, name: str) -> None:
    ensure_directories()
    for directory in [FIGURES_DIR, PAPER_FIGURES_DIR]:
        path = directory / name
        fig.savefig(path, dpi=180, bbox_inches="tight")
        if not path.exists() or path.stat().st_size < 5_000:
            raise RuntimeError(f"Figure {path} looks empty or was not written.")
    plt.close(fig)


def plot_volatility_timeseries(spy: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.5))
    spy[["realized_vol_5", "realized_vol_20"]].plot(ax=ax, linewidth=1.1)
    ax.set_title("SPY Realized Volatility")
    ax.set_ylabel("Annualized volatility")
    ax.set_xlabel("Date")
    ax.legend(["5-day", "20-day"], frameon=False)
    _save(fig, "volatility_timeseries.png")


def plot_correlation_heatmap(feature_frames: dict[str, pd.DataFrame]) -> None:
    vols = pd.DataFrame(
        {symbol: frame[f"realized_vol_{TARGET_WINDOW}"] for symbol, frame in feature_frames.items()}
    ).dropna()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(vols.corr(), cmap="vlag", center=0, annot=True, fmt=".2f", square=True, ax=ax)
    ax.set_title("5-Day Realized Volatility Correlation")
    _save(fig, "volatility_correlation_heatmap.png")


def plot_predicted_vs_actual(y_true: pd.Series, predictions: dict[str, pd.Series], best_model: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.8))
    y_true.plot(ax=ax, color="black", linewidth=1.5, label="Actual")
    predictions[best_model].reindex(y_true.index).plot(ax=ax, linewidth=1.1, label=best_model)
    ax.set_title(f"SPY Future Volatility: Actual vs {best_model}")
    ax.set_ylabel("Annualized volatility")
    ax.set_xlabel("Date")
    ax.legend(frameon=False)
    _save(fig, "predicted_vs_actual.png")


def plot_model_comparison(metrics: pd.DataFrame) -> None:
    order = metrics.sort_values("RMSE")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=order, x="RMSE", y="model", hue="model", palette="crest", legend=False, ax=ax)
    ax.set_title("Held-Out Test RMSE by Model")
    ax.set_xlabel("RMSE, annualized volatility")
    ax.set_ylabel("")
    _save(fig, "model_comparison.png")


def plot_error_distribution(y_true: pd.Series, predictions: dict[str, pd.Series]) -> None:
    rows = []
    for model, pred in predictions.items():
        aligned = pd.concat([y_true.rename("actual"), pred.rename("pred")], axis=1).dropna()
        rows.extend({"model": model, "error": e} for e in (aligned["actual"] - aligned["pred"]))
    errors = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.kdeplot(data=errors, x="error", hue="model", common_norm=False, linewidth=1.2, ax=ax)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Forecast Error Distribution")
    ax.set_xlabel("Actual minus predicted volatility")
    _save(fig, "error_distribution.png")


def plot_feature_importance(importances: pd.DataFrame) -> None:
    top = importances.head(15).sort_values("importance")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top["feature"], top["importance"], color="#2f6f73")
    ax.set_title("Top Tree-Based Feature Importances")
    ax.set_xlabel("Importance")
    _save(fig, "feature_importance.png")


def create_contact_sheet() -> None:
    names = [
        "volatility_timeseries.png",
        "volatility_correlation_heatmap.png",
        "predicted_vs_actual.png",
        "model_comparison.png",
        "error_distribution.png",
        "feature_importance.png",
    ]
    images = [Image.open(FIGURES_DIR / name).convert("RGB").resize((520, 330)) for name in names]
    sheet = Image.new("RGB", (1040, 990), "white")
    for idx, image in enumerate(images):
        x = (idx % 2) * 520
        y = (idx // 2) * 330
        sheet.paste(image, (x, y))
    for directory in [FIGURES_DIR, PAPER_FIGURES_DIR]:
        sheet.save(directory / "figure_contact_sheet.png", quality=92)
