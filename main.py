import json
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import time
import random
import threading
import csv
import numpy as np
import pandas as pd
import joblib
Q_FILE = "q_table.pkl"
from collections import deque

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import load_model


# ==================================================
# CONFIG
# ==================================================

TRAIN_FILE = "training_data.csv"

BUFFER_SIZE = 60
SEQ_LEN = 5


# ==================================================
# LABELS
# ==================================================

LABELS = {
    "NORMAL": 0,
    "NOISE": 1,
    "FAULT": 2
}


# ==================================================
# GLOBAL STATE
# ==================================================

risk_history = deque(maxlen=10)

# ==================================================
# LEVEL 5-C: LEARNING BRAIN (Q-LEARNING)
# ==================================================

Q_TABLE = {}

# Load previous brain if exists
if os.path.exists(Q_FILE):
    Q_TABLE = joblib.load(Q_FILE)
    print("🧠 Loaded existing Q-Table")
else:
    print("🆕 Starting fresh Q-Table")

LEARNING_RATE = 0.2
DISCOUNT = 0.9
EXPLORATION = 0.3   # 10% random decisions

# ==================================================
# INIT TRAIN FILE
# ==================================================

def init_training_file():

    if not os.path.exists(TRAIN_FILE):

        with open(TRAIN_FILE, "w", newline="") as f:

            writer = csv.writer(f)

            writer.writerow([
                "fuel_err",
                "speed_err",
                "fuel_trend",
                "speed_var",
                "drift_rate",
                "alert_rate",
                "label"
            ])


init_training_file()


# ==================================================
# SAFE CSV WRITER
# ==================================================

def save_experience(row):

    clean = row[:7]

    if len(clean) != 7:
        print("❌ BAD ROW BLOCKED:", row)
        return

    clean = [
        float(clean[0]),
        float(clean[1]),
        float(clean[2]),
        float(clean[3]),
        float(clean[4]),
        float(clean[5]),
        int(clean[6])
    ]

    with open(TRAIN_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(clean)


# ==================================================
# SAFE TREND
# ==================================================

def safe_trend(values):

    n = len(values)

    if n < 3:
        return 0.0

    if np.std(values) < 1e-6:
        return 0.0

    try:
        x = np.arange(n)
        y = np.array(values)

        coef = np.polyfit(x, y, 1)[0]

        return float(coef)

    except:
        return 0.0


# ==================================================
# LOAD MODELS
# ==================================================

# LSTM
try:
    lstm_model = load_model("lstm_model.h5")
    lstm_scaler = joblib.load("lstm_scaler.save")
    lstm_ready = True
    print("✅ LSTM LOADED")
except Exception as e:
    lstm_ready = False
    print("❌ LSTM FAILED:", e)


# Anomaly AI
try:
    anomaly_model = load_model("anomaly_model.h5", compile=False)
    anomaly_scaler = joblib.load("anomaly_scaler.save")
    anomaly_ready = True
    print("✅ Anomaly AI Loaded")
except Exception as e:
    anomaly_ready = False
    print("❌ Anomaly AI Load Failed:", e)


lstm_buffer = deque(maxlen=SEQ_LEN)


# ==================================================
# TRAIN CLASSICAL ML
# ==================================================

def train_from_memory():

    if not os.path.exists(TRAIN_FILE):
        return None, None

    if os.path.getsize(TRAIN_FILE) == 0:
        return None, None


    try:
        df = pd.read_csv(TRAIN_FILE)
    except:
        return None, None


    if len(df) < 25:
        return None, None


    if df["label"].nunique() < 2:
        return None, None


    X = df[
        [
            "fuel_err",
            "speed_err",
            "fuel_trend",
            "speed_var",
            "drift_rate",
            "alert_rate"
        ]
    ].values

    y = df["label"].values


    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)


    model = LogisticRegression(max_iter=1500)
    model.fit(Xs, y)


    return model, scaler


# ==================================================
# PHYSICS ENGINE
# ==================================================

def physics_check(fuel_err, speed_err):

    if fuel_err > 15 or speed_err > 20:
        return "FAULT"

    if fuel_err > 6 or speed_err > 8:
        return "NOISE"

    return "NORMAL"


# ==================================================
# SENSOR SIMULATOR
# ==================================================

def simulate_sensor():
    # Slowly drifting baseline
    simulate_sensor.fuel_base = getattr(simulate_sensor, "fuel_base", random.uniform(4, 6))
    simulate_sensor.speed_base = getattr(simulate_sensor, "speed_base", random.uniform(2, 4))

    simulate_sensor.fuel_base += random.uniform(-0.3, 0.3)
    simulate_sensor.speed_base += random.uniform(-0.2, 0.2)

    # Rare but strong fault
    fault = 0
    if random.random() < 0.1:   # 10% chance
        fault = random.uniform(5, 20)

    fuel = simulate_sensor.fuel_base + fault
    speed = simulate_sensor.speed_base + fault * 0.6

    return fuel, speed


# ==================================================
# ROOT CAUSE ENGINE (LEVEL 5-A)
# ==================================================

def infer_root_cause(
        fuel_err,
        speed_err,
        fuel_trend,
        speed_var,
        drift_rate,
        anomaly_score,
        risk_prob
):

    scores = {
        "Sensor Drift": 0.0,
        "Fuel Leak": 0.0,
        "Blockage": 0.0,
        "Software Fault": 0.0
    }


    # Sensor Drift
    if drift_rate > 0.5:
        scores["Sensor Drift"] += 0.4

    if abs(fuel_trend) > 0.3:
        scores["Sensor Drift"] += 0.3

    if anomaly_score > 0.01:
        scores["Sensor Drift"] += 0.3


    # Fuel Leak
    if fuel_err > 12:
        scores["Fuel Leak"] += 0.5

    if fuel_trend < -0.5:
        scores["Fuel Leak"] += 0.3

    if risk_prob > 40:
        scores["Fuel Leak"] += 0.2


    # Blockage
    if speed_var > 30:
        scores["Blockage"] += 0.4

    if speed_err > 10:
        scores["Blockage"] += 0.3

    if anomaly_score > 0.02:
        scores["Blockage"] += 0.3


    # Software Fault
    if anomaly_score > 0.03:
        scores["Software Fault"] += 0.5

    if fuel_err < 2 and speed_err < 2 and risk_prob > 20:
        scores["Software Fault"] += 0.5


    # Normalize
    total = sum(scores.values())

    if total == 0:
        return {k: 0.0 for k in scores}

    for k in scores:
        scores[k] /= total


    return scores

# ==================================================
# WHAT-IF SIMULATOR (LEVEL 5-B)
# ==================================================

def simulate_future(
        fuel_err,
        speed_err,
        drift_rate,
        risk_prob,
        action="CONTINUE",
        runs=300
):

    failures = 0
    safe = 0
    times = []

    for _ in range(runs):

        # More realistic initial state
        fuel = max(40, 120 - fuel_err * 2)
        speed = max(180, 260 - speed_err * 3)

        risk = max(0.05, risk_prob / 100)


        for minute in range(1, 61):

            # ---------- ACTION EFFECT ----------

            if action == "DESCEND":
                fuel -= random.uniform(0.8, 1.3)
                risk *= 0.95

            elif action == "DIVERT":
                fuel -= random.uniform(1.0, 1.6)
                risk *= 0.92

            else:  # CONTINUE
                fuel -= random.uniform(1.4, 2.0)
                risk *= 1.01

            # ---------- DRIFT + NOISE ----------

            risk += drift_rate * 0.002
            risk += random.uniform(-0.003, 0.006)

            # ---------- RECOVERY EFFECT ----------

            if fuel > 35 and speed > 210:
                risk *= 0.96

            # ---------- CLAMP ----------

            risk = min(max(risk, 0.02), 0.95)

            # ---------- PROBABILISTIC FAILURE ----------

            fail_chance = 0.0

            # Fuel stress
            if fuel < 25:
                fail_chance += (25 - fuel) * 0.02

            # Risk stress
            fail_chance += risk * 0.45

            # External disturbance
            fail_chance += random.uniform(0, 0.08)

            # Normalize
            fail_chance = min(fail_chance, 0.9)

            # Failure draw
            if random.random() < fail_chance:
                failures += 1
                times.append(minute)
                break


        else:
            safe += 1
            times.append(60)


    avg_time = np.mean(times)

    fail_prob = failures / runs * 100
    safe_prob = safe / runs * 100


    return {
        "fail_prob": fail_prob,
        "safe_prob": safe_prob,
        "avg_time": avg_time
    }

# ==================================================
# LEARNING BRAIN FUNCTIONS
# ==================================================

def get_state(fuel_err, drift, anomaly, risk):
    """
    Convert continuous values into discrete state
    """

    f = int(fuel_err // 3)
    d = int(drift // 0.5)
    a = int(anomaly // 0.5)
    r = int(risk // 20)

    return (f, d, a, r)


def get_q(state, action):

    if state not in Q_TABLE:
        Q_TABLE[state] = {
            "CONTINUE": 0.0,
            "DESCEND": 0.0,
            "DIVERT": 0.0
        }

    return Q_TABLE[state][action]


def update_q(state, action, reward, next_state):

    if state not in Q_TABLE:
        get_q(state, action)

    max_future = max(Q_TABLE.get(next_state, {
        "CONTINUE": 0,
        "DESCEND": 0,
        "DIVERT": 0
    }).values())

    old = Q_TABLE[state][action]

    new = old + LEARNING_RATE * (
        reward + DISCOUNT * max_future - old
    )

    Q_TABLE[state][action] = new


def choose_action(state):

    # Exploration
    if random.random() < EXPLORATION:
        return random.choice(["CONTINUE", "DESCEND", "DIVERT"])

    if state not in Q_TABLE:
        return "CONTINUE"

    return max(Q_TABLE[state], key=Q_TABLE[state].get)

# ==================================================
# MAIN
# ==================================================
# ================= LIVE DATA LOGGER =================

def safe_float(x):
    if isinstance(x, (np.float32, np.float64)):
        return float(x)
    return float(x)


def save_live_data(
    fuel_err,
    speed_err,
    risk_prob,
    anomaly_score,
    final,
    ai_status
):
    live_data = {
        "timestamp": time.time(),

        "fuel_err": round(float(fuel_err), 2),
        "speed_err": round(float(speed_err), 2),

        "risk": round(float(risk_prob), 2),
        "anomaly": round(float(anomaly_score), 4),

        "status": str(final),
        "ai_state": str(ai_status)
    }

    try:
        with open("live_data.json.tmp", "w") as f:
            json.dump(live_data, f, indent=2)

        os.replace("live_data.json.tmp", "live_data.json")

    except Exception as e:
        print("❌ Live data save failed:", e)

def main():

    print("\n=== AeroVerify-AI : Level 5 Diagnostic System ===\n")


    buffer = deque(maxlen=BUFFER_SIZE)

    alert_count = 0


    model, scaler = train_from_memory()

    t = 0
    max_alerts = 30

    # ---------- INIT LIVE VALUES ----------
    fuel_err = 0.0
    speed_err = 0.0
    risk_prob = 0.0
    anomaly_score = 0.0
    final = "NORMAL"
    ai_status = "BOOTING"


    while alert_count < max_alerts:

        time.sleep(0.2)
        t += 1

        fuel, speed = simulate_sensor()
        buffer.append((fuel, speed))

        # ===== CONTINUOUS FEATURE UPDATE =====
        fuels = np.nan_to_num([b[0] for b in buffer])
        speeds = np.nan_to_num([b[1] for b in buffer])

        fuel_err = abs(fuels[-1] - np.mean(fuels))
        speed_err = abs(speeds[-1] - np.mean(speeds))

        fuel_trend = safe_trend(fuels)
        speed_var = np.var(speeds)

        drift_rate = abs(fuels[-1] - fuels[0]) / max(1, len(fuels))

        alert_rate = alert_count / max(1, t)

        # ---- LIVE TERMINAL STATUS ----
        print(
            f"[LIVE] FuelErr={fuel_err:.2f} | "
            f"SpeedErr={speed_err:.2f} | "
            f"Risk={risk_prob:.2f}% | "
            f"Anomaly={anomaly_score:.4f} | "
            f"Final={final} | "
            f"AI={ai_status}"
        )

        if random.random() < 0.3:

            alert_count += 1  # REQUIRED
            print(f"⚠️ ALERT @ t={t}s")



            # ================= FEATURES =================

            fuels = np.nan_to_num([b[0] for b in buffer])
            speeds = np.nan_to_num([b[1] for b in buffer])


            fuel_err = abs(fuels[-1] - np.mean(fuels))
            speed_err = abs(speeds[-1] - np.mean(speeds))

            fuel_trend = safe_trend(fuels)
            speed_var = np.var(speeds)

            drift_rate = abs(fuels[-1] - fuels[0]) / max(1, len(fuels))

            alert_rate = alert_count / max(1, t)


            # ================= PHYSICS =================

            physics = physics_check(fuel_err, speed_err)


            # ================= ML =================

            ai_status = "LEARNING"

            if model is not None:

                X = np.array([
                    fuel_err,
                    speed_err,
                    fuel_trend,
                    speed_var,
                    drift_rate,
                    alert_rate
                ]).reshape(1, -1)

                Xs = scaler.transform(X)

                pred = model.predict(Xs)[0]

                ai_status = list(LABELS.keys())[pred]


            # ================= LSTM =================

            lstm_status = "OFF"
            risk_prob = risk_prob * 0.95

            lstm_buffer.append([
                fuel_err,
                speed_err,
                fuel_trend,
                speed_var,
                drift_rate,
                alert_rate
            ])


            if lstm_ready and len(lstm_buffer) == SEQ_LEN:

                seq = np.array(lstm_buffer)

                seq = lstm_scaler.transform(seq)

                seq = seq.reshape(1, SEQ_LEN, 6)

                pred = lstm_model.predict(seq, verbose=0)[0]

                risk_prob = pred[2] * 100

                cls = np.argmax(pred)

                if cls == 0:
                    lstm_status = "NORMAL"
                elif cls == 1:
                    lstm_status = "NOISE"
                else:
                    lstm_status = "FAULT"

                risk_history.append(risk_prob)


            # ================= ANOMALY =================

            anomaly_score = 0.0
            anomaly_status = "OFF"


            if anomaly_ready:

                vec = np.array([
                    fuel_err,
                    speed_err,
                    fuel_trend,
                    speed_var,
                    drift_rate,
                    alert_rate
                ]).reshape(1, -1)

                vec = anomaly_scaler.transform(vec)

                recon = anomaly_model.predict(vec, verbose=0)

                anomaly_score = np.mean(np.square(vec - recon))


                if not hasattr(main, "anom_history"):
                    main.anom_history = deque(maxlen=50)

                main.anom_history.append(anomaly_score)

                mean_err = np.mean(main.anom_history)
                std_err = np.std(main.anom_history)

                if anomaly_score > mean_err + 3 * std_err:
                    anomaly_status = "HIGH"
                elif anomaly_score > mean_err + 1.5 * std_err:
                    anomaly_status = "MEDIUM"
                else:
                    anomaly_status = "LOW"


            # ================= ROOT CAUSE =================

            causes = infer_root_cause(
                fuel_err,
                speed_err,
                fuel_trend,
                speed_var,
                drift_rate,
                anomaly_score,
                risk_prob
            )

            main_cause = max(causes, key=causes.get)
            main_prob = causes[main_cause] * 100


            # ================= WHAT-IF SIMULATION =================

            sim_results = {}

            sim_results["continue"] = simulate_future(
                fuel_err,
                speed_err,
                drift_rate,
                risk_prob,
                action="CONTINUE"
            )

            sim_results["descend"] = simulate_future(
                fuel_err,
                speed_err,
                drift_rate,
                risk_prob,
                action="DESCEND"
            )

            sim_results["divert"] = simulate_future(
                fuel_err,
                speed_err,
                drift_rate,
                risk_prob,
                action="DIVERT"
            )
            # ================= LEARNING STATE =================

            state = get_state(
                fuel_err,
                drift_rate,
                anomaly_score,
                risk_prob
            )

            chosen_action = choose_action(state)

            # ================= DECISION =================

            statuses = [physics, ai_status, lstm_status]

            if "FAULT" in statuses and risk_prob > 40:
                final = "FAULT"
            elif statuses.count("NOISE") >= 2:
                final = "NOISE"
            else:
                final = "NORMAL"

            # ================= REWARD =================

            # Use simulation outcome
            best_safe = max(
                sim_results.get("continue", {"safe_prob": 50})["safe_prob"],
                sim_results.get("descend", {"safe_prob": 50})["safe_prob"],
                sim_results.get("divert", {"safe_prob": 50})["safe_prob"]
            )

            chosen_sim = sim_results.get(
                chosen_action.lower(),
                {"fail_prob": 50, "safe_prob": 50, "avg_time": 60}
            )

            reward = 0

            # Encourage safety
            if final == "NORMAL":
                reward += 2

            elif final == "NOISE":
                reward += 0.5

            else:  # FAULT
                reward -= 2

            # Encourage lower risk
            reward += max(0, (100 - risk_prob) / 50)

            # Penalize bad outcomes
            if risk_prob > 50:
                reward -= 1

            # Reward choosing safest option
            if chosen_sim["safe_prob"] == best_safe:
                reward += 3

            # Penalize risky choice
            if chosen_sim["fail_prob"] > 70:
                reward -= 2

            # Reward good system agreement
            if final == "NORMAL" and risk_prob < 30:
                reward += 1

            if final == "FAULT" and risk_prob > 50:
                reward += 2

            # Penalize contradiction
            if final == "NORMAL" and risk_prob > 50:
                reward -= 2

            if final == "FAULT" and risk_prob < 20:
                reward -= 2

            # Penalize anomalies
            reward -= anomaly_score * 0.05

            # ================= CONFIDENCE =================

            confidence = max(
                statuses.count("FAULT"),
                statuses.count("NOISE"),
                statuses.count("NORMAL")
            ) / 3 * 100


            # ================= SAVE =================

            save_experience([
                fuel_err,
                speed_err,
                fuel_trend,
                speed_var,
                drift_rate,
                alert_rate,
                LABELS[final]
            ])

            model, scaler = train_from_memory()

            # ================= REPORT =================

            print(f"FuelErr={fuel_err:.2f} | SpeedErr={speed_err:.2f}")
            print(f"Trend={fuel_trend:.3f} | Var={speed_var:.3f} | Drift={drift_rate:.3f}")

            print(f"Physics     : {physics}")
            print(f"ML AI       : {ai_status}")
            print(f"LSTM AI     : {lstm_status}")
            print(f"Anomaly AI  : {anomaly_status} ({anomaly_score:.4f})")

            print(f"Risk Prob   : {risk_prob:.2f}%")
            print(f"Confidence  : {confidence:.2f}%")

            # ---------- SAVE LIVE DATA ----------
            ##save_live_data(
               # fuel_err,
               # speed_err,
              #  risk_prob,
              #  anomaly_score,
              #  final,
              #  ai_status
            #)


            # ---------- ROOT CAUSE REPORT ----------

            print("\nRoot Cause Analysis:")

            for k, v in causes.items():
                print(f"  {k:15s}: {v * 100:.1f}%")

            print(f"Likely Cause: {main_cause} ({main_prob:.1f}%)")

            # ---------- WHAT-IF SIMULATION REPORT ----------

            print("\nWhat-If Simulation:")

            sim_c = sim_results.get(
                "continue",
                {"fail_prob": 50, "safe_prob": 50, "avg_time": 60}
            )

            print(
                f"  Continue → Fail {sim_c['fail_prob']:.1f}% | "
                f"Safe {sim_c['safe_prob']:.1f}% | "
                f"Time {sim_c['avg_time']:.1f} min"
            )

            sim_d = sim_results.get(
                "descend",
                {"fail_prob": 50, "safe_prob": 50, "avg_time": 60}
            )

            print(
                f"  Descend  → Fail {sim_d['fail_prob']:.1f}% | "
                f"Safe {sim_d['safe_prob']:.1f}% | "
                f"Time {sim_d['avg_time']:.1f} min"
            )

            sim_v = sim_results.get(
                "divert",
                {"fail_prob": 50, "safe_prob": 50, "avg_time": 60}
            )

            print(
                f"  Divert   → Fail {sim_v['fail_prob']:.1f}% | "
                f"Safe {sim_v['safe_prob']:.1f}% | "
                f"Time {sim_v['avg_time']:.1f} min"
            )

            # ---------- FINAL STATUS ----------
            print("\nLearning Brain:")

            print(f"  State        : {state}")
            print(f"  Chosen Action: {chosen_action}")
            print(f"  Reward       : {reward:.2f}")

            if state in Q_TABLE:
                print("  Q-Values:")
                for a, v in Q_TABLE[state].items():
                    print(f"    {a:8s} → {v:.2f}")

            print(f"\nFinal       : {final}")




            print("-" * 45)
            # ================= SAVE LEARNING BRAIN =================

joblib.dump(Q_TABLE, Q_FILE)
print("💾 Q-Table Saved")

# ==================================================
# RUN
# ==================================================

if __name__ == "__main__":
    main()