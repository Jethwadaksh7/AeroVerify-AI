import streamlit as st
import json
import time
import pandas as pd
import os

# ================= PAGE CONFIG =================

st.set_page_config(
    page_title="AeroVerify-AI Dashboard",
    page_icon="✈️",
    layout="wide"
)

# ================= SESSION STATE =================

if "history" not in st.session_state:
    st.session_state.history = []

# ================= STYLING =================

st.markdown("""
<style>
body {
    background-color: #0E1117;
}
</style>
""", unsafe_allow_html=True)

# ================= LOAD DATA =================

DATA_FILE = "live_data.json"


def load_data():
    try:
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, "r") as f:
                return json.load(f)
    except:
        return None

    return None


# ================= STATUS COLOR =================

def status_color(status):

    if status == "NORMAL":
        return f"🟢 {status}"

    elif status == "NOISE":
        return f"🟡 {status}"

    elif status == "FAULT":
        return f"🔴 {status}"

    return status


# ================= HEADER =================

st.markdown("# ✈️ AeroVerify-AI : Real-Time Diagnostic Dashboard")
st.markdown("### Level 5 AI-Driven Aircraft Health Monitoring")

st.markdown("---")


# ================= MAIN =================

data = load_data()

if data:

    # Save history
    st.session_state.history.append({
        "time": time.strftime("%H:%M:%S"),
        "fuel": float(data["fuel_err"]),
        "speed": float(data["speed_err"]),
        "risk": float(data["risk"]),
        "anomaly": float(data["anomaly"]),
        "status": data["status"]
    })

    # Keep last 100 records
    if len(st.session_state.history) > 100:
        st.session_state.history.pop(0)

    df = pd.DataFrame(st.session_state.history)

    # ================= METRICS =================

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Fuel Error", data["fuel_err"])
    c2.metric("Speed Error", data["speed_err"])
    c3.metric("Risk Probability (%)", data["risk"])
    c4.metric("Anomaly Score", data["anomaly"])

    st.markdown("---")

    # ================= STATUS =================

    s1, s2, s3 = st.columns(3)

    s1.success(f"AI Status: {data['ai_state']}")
    s2.info(f"Final State: {status_color(data['status'])}")
    s3.warning("📡 Live Monitoring Active")

    st.markdown("---")

    # ================= LIVE GRAPHS =================

    st.subheader("📈 Live Sensor Trends")

    g1, g2 = st.columns(2)

    with g1:

        st.line_chart(
            df.set_index("time")[["fuel"]],
            height=300,
            use_container_width=True
        )

        st.caption("Fuel Error Over Time")

    with g2:

        st.line_chart(
            df.set_index("time")[["speed"]],
            height=300,
            use_container_width=True
        )

        st.caption("Speed Error Over Time")


else:

    st.error("❌ No live data found. Run main.py first.")


# ================= AUTO REFRESH =================

time.sleep(2)
st.rerun()