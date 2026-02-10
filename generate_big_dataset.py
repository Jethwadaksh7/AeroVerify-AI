import csv
import random
import numpy as np

# ===============================
# CONFIG
# ===============================

FILE = "big_training_data.csv"
ROWS = 500000      # Increase later: 300k / 500k

# ===============================
# CSV HEADER
# ===============================

HEADER = [
    "fuel_err",
    "speed_err",
    "fuel_trend",
    "speed_var",
    "drift_rate",
    "alert_rate",
    "label"
]

# ===============================
# SYSTEM STATE (Simulates Aging)
# ===============================

fuel_health = 1.0     # 1 = perfect, 0 = broken
speed_health = 1.0
sensor_health = 1.0

alert_counter = 0


# ===============================
# MAIN GENERATOR
# ===============================

def generate_row():

    global fuel_health
    global speed_health
    global sensor_health
    global alert_counter


    # -------------------------------
    # System degradation
    # -------------------------------

    fuel_health -= random.uniform(0, 0.00002)
    speed_health -= random.uniform(0, 0.000015)
    sensor_health -= random.uniform(0, 0.00001)

    fuel_health = max(0.3, fuel_health)
    speed_health = max(0.3, speed_health)
    sensor_health = max(0.4, sensor_health)


    # -------------------------------
    # Base errors
    # -------------------------------

    fuel_err = random.uniform(0.5, 2.5) / fuel_health
    speed_err = random.uniform(0.3, 1.8) / speed_health


    # -------------------------------
    # Trends (leaks cause negative)
    # -------------------------------

    fuel_trend = random.uniform(-0.05, 0.05)

    if fuel_health < 0.7:
        fuel_trend -= random.uniform(0.05, 0.3)


    # -------------------------------
    # Variance (instability)
    # -------------------------------

    speed_var = random.uniform(3, 10)

    if speed_health < 0.7:
        speed_var += random.uniform(10, 30)


    # -------------------------------
    # Drift
    # -------------------------------

    drift_rate = random.uniform(0.01, 0.08) / sensor_health


    # -------------------------------
    # Alert Rate
    # -------------------------------

    if fuel_err > 6 or speed_err > 6:
        alert_counter += 1

    alert_rate = min(1.0, alert_counter / 1000)


    # -------------------------------
    # Add Sensor Noise
    # -------------------------------

    fuel_err += np.random.normal(0, 0.3)
    speed_err += np.random.normal(0, 0.2)
    speed_var += np.random.normal(0, 0.8)


    # -------------------------------
    # Clamp Values
    # -------------------------------

    fuel_err = max(0, fuel_err)
    speed_err = max(0, speed_err)
    speed_var = max(1, speed_var)
    drift_rate = max(0.001, drift_rate)


    # -------------------------------
    # Label Logic (Ground Truth)
    # -------------------------------

    if fuel_err > 14 or speed_err > 12 or speed_var > 35:
        label = 2   # FAULT

    elif fuel_err > 6 or speed_err > 6 or speed_var > 18:
        label = 1   # NOISE

    else:
        label = 0   # NORMAL


    return [
        round(fuel_err, 3),
        round(speed_err, 3),
        round(fuel_trend, 4),
        round(speed_var, 3),
        round(drift_rate, 4),
        round(alert_rate, 4),
        label
    ]


# ===============================
# GENERATE FILE
# ===============================

def main():

    print("🚀 Generating realistic dataset...")
    print(f"Rows: {ROWS}")
    print(f"File: {FILE}\n")


    with open(FILE, "w", newline="") as f:

        writer = csv.writer(f)
        writer.writerow(HEADER)

        for i in range(ROWS):

            row = generate_row()
            writer.writerow(row)

            if i % 10000 == 0 and i > 0:
                print(f"Generated {i} rows...")


    print("\n✅ DONE!")
    print(f"Saved to: {FILE}")


# ===============================
# RUN
# ===============================

if __name__ == "__main__":
    main()