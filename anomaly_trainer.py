import pandas as pd
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import StandardScaler
import joblib


TRAIN_FILE = "training_data.csv"

MODEL_FILE = "anomaly_model.h5"
SCALER_FILE = "anomaly_scaler.save"


# Load Data
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


# Normalize
scaler = StandardScaler()
X = scaler.fit_transform(X)

joblib.dump(scaler, SCALER_FILE)


# Build Autoencoder
inp = Input(shape=(6,))

encoded = Dense(4, activation="relu")(inp)
encoded = Dense(2, activation="relu")(encoded)

decoded = Dense(4, activation="relu")(encoded)
decoded = Dense(6, activation="linear")(decoded)

autoencoder = Model(inp, decoded)

autoencoder.compile(
    optimizer=Adam(0.001),
    loss="mse"
)

autoencoder.summary()


# Train
autoencoder.fit(
    X, X,
    epochs=50,
    batch_size=16,
    validation_split=0.2
)


# Save
autoencoder.save(MODEL_FILE, save_format="h5")
print("✅ Anomaly Model Saved")