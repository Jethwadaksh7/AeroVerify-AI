import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import StandardScaler
import joblib


# ===============================
# Config
# ===============================

TRAIN_FILE = "training_data.csv"

SEQ_LEN = 5          # Reduced for small dataset

MODEL_FILE = "lstm_model.h5"
SCALER_FILE = "lstm_scaler.save"


# ===============================
# Load Data
# ===============================

df = pd.read_csv(TRAIN_FILE)

features = [
    "fuel_err",
    "speed_err",
    "fuel_trend",
    "speed_var",
    "drift_rate",
    "alert_rate"
]

X = df[features].values
y = df["label"].values


# ===============================
# Normalize
# ===============================

scaler = StandardScaler()
X = scaler.fit_transform(X)

joblib.dump(scaler, SCALER_FILE)


# ===============================
# Build Sequences
# ===============================

def build_sequences(X, y, seq_len):

    Xs, ys = [], []

    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])

    return np.array(Xs), np.array(ys)


X_seq, y_seq = build_sequences(X, y, SEQ_LEN)


# ===============================
# Safety Check
# ===============================

if len(X_seq) == 0:
    print("❌ Not enough data for LSTM training.")
    print("Add more rows to training_data.csv or reduce SEQ_LEN.")
    exit()


print("Sequences shape:", X_seq.shape)


# ===============================
# Build LSTM Model
# ===============================

model = Sequential()

model.add(
    LSTM(
        64,
        return_sequences=True,
        input_shape=(SEQ_LEN, X_seq.shape[2])
    )
)

model.add(Dropout(0.3))

model.add(LSTM(32))

model.add(Dense(3, activation="softmax"))


model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=Adam(0.001),
    metrics=["accuracy"]
)


model.summary()


# ===============================
# Train
# ===============================

model.fit(
    X_seq,
    y_seq,
    epochs=25,
    batch_size=16,
    validation_split=0.2
)


# ===============================
# Save Model
# ===============================

model.save(MODEL_FILE)

print("\n✅ LSTM Model Saved Successfully")