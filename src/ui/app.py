import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
st.title("ML Demo")

number = st.number_input("number", value=22, step=1)
driver_number = st.number_input("driver_number", value=7, step=1)
lap_number = st.number_input("lap_number", value=12, step=1)
kph = st.number_input("kph", value=150.0, step=0.1)
top_speed = st.number_input("top_speed", value=300.0, step=0.1)
season = st.number_input("season", value=2022, step=1)
round_num = st.number_input("round", value=4, step=1)

if st.button("Predict"):
    payload = {
        "number": int(number),
        "driver_number": int(driver_number),
        "lap_number": int(lap_number),
        "kph": float(kph),
        "top_speed": float(top_speed),
        "season": int(season),
        "round": int(round_num),
    }
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        st.write("Status:", r.status_code)
        if r.status_code == 200:
            response = r.json()
            st.write("Response:", response)
            st.success(f"Predicted lap time: {response.get('lap_time_s', 'N/A')}s")
        else:
            st.error(f"Error: {r.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")
