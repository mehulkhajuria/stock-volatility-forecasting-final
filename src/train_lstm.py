from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_SEED
from .train_ml import feature_columns


def train_lstm_or_fallback(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.Series, float, str]:
    """Attempt a small LSTM; fall back to a deterministic neural baseline if DL stacks are unavailable."""
    start = time.perf_counter()
    columns = feature_columns(train)
    seq_len = 20

    def make_sequences(frame: pd.DataFrame):
        x_raw = frame[columns].to_numpy(dtype=np.float32)
        y_raw = frame["target_future_vol"].to_numpy(dtype=np.float32)
        xs, ys, dates = [], [], []
        for i in range(seq_len - 1, len(frame)):
            xs.append(x_raw[i - seq_len + 1 : i + 1])
            ys.append(y_raw[i])
            dates.append(frame.index[i])
        return np.asarray(xs), np.asarray(ys), pd.Index(dates)

    combined = pd.concat([train, validation])
    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(RANDOM_SEED)
        scaler = StandardScaler()
        scaler.fit(combined[columns])

        def scaled(frame):
            temp = frame.copy()
            temp.loc[:, columns] = scaler.transform(temp[columns])
            return temp

        x_train, y_train, _ = make_sequences(scaled(combined))
        x_test, _, dates = make_sequences(scaled(pd.concat([combined.tail(seq_len - 1), test])))
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(seq_len, len(columns))),
                tf.keras.layers.LSTM(32, dropout=0.1),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(1),
            ]
        )
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
        model.fit(x_train, y_train, epochs=20, batch_size=32, verbose=0, validation_split=0.1)
        pred = model.predict(x_test, verbose=0).reshape(-1)
        note = "TensorFlow/Keras LSTM trained on 20-day sequences."
    except Exception as tf_exc:
        try:
            import torch
            from torch import nn

            torch.manual_seed(RANDOM_SEED)
            scaler = StandardScaler()
            scaler.fit(combined[columns])

            def scaled(frame):
                temp = frame.copy()
                temp.loc[:, columns] = scaler.transform(temp[columns])
                return temp

            x_train, y_train, _ = make_sequences(scaled(combined))
            x_test, _, dates = make_sequences(scaled(pd.concat([combined.tail(seq_len - 1), test])))
            x_tensor = torch.tensor(x_train)
            y_tensor = torch.tensor(y_train).view(-1, 1)

            class TinyLSTM(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.lstm = nn.LSTM(len(columns), 32, batch_first=True)
                    self.head = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

                def forward(self, x):
                    out, _ = self.lstm(x)
                    return self.head(out[:, -1, :])

            model = TinyLSTM()
            opt = torch.optim.Adam(model.parameters(), lr=0.001)
            loss_fn = nn.MSELoss()
            for _ in range(25):
                opt.zero_grad()
                loss = loss_fn(model(x_tensor), y_tensor)
                loss.backward()
                opt.step()
            with torch.no_grad():
                pred = model(torch.tensor(x_test)).numpy().reshape(-1)
            note = f"PyTorch LSTM trained after TensorFlow was unavailable ({tf_exc})."
        except Exception as torch_exc:
            flat_train = combined.copy()
            flat_test = test.copy()
            model = make_pipeline(
                StandardScaler(),
                MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    alpha=0.001,
                    max_iter=500,
                    random_state=RANDOM_SEED,
                    early_stopping=True,
                ),
            )
            model.fit(flat_train[columns], flat_train["target_future_vol"])
            pred = model.predict(flat_test[columns])
            dates = flat_test.index
            note = (
                "TensorFlow and PyTorch LSTM stacks unavailable or unreliable; "
                f"reported deterministic MLP neural fallback. TF error: {tf_exc}; Torch error: {torch_exc}"
            )

    pred_series = pd.Series(np.maximum(pred, 0), index=dates, name="MLP neural fallback")
    return pred_series.loc[test.index.intersection(pred_series.index)], time.perf_counter() - start, note
