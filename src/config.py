from pathlib import Path

RANDOM_SEED = 2147

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"
PAPER_FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"

SYMBOLS = ["SPY", "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "TSLA", "META"]
TARGET_SYMBOL = "SPY"
START_DATE = "2014-01-01"
END_DATE = None

TRAIN_END = "2021-12-31"
VALIDATION_END = "2022-12-31"

TARGET_WINDOW = 5
ROLLING_WINDOWS = [5, 10, 20]
MAX_LAG = 30
VOLUME_WINDOWS = [5, 20]


def ensure_directories() -> None:
    for path in [RAW_DIR, PROCESSED_DIR, FIGURES_DIR, RESULTS_DIR, PAPER_FIGURES_DIR]:
        path.mkdir(parents=True, exist_ok=True)
