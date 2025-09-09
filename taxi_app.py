import streamlit as st
import joblib
import pandas as pd

# -------------------------------
# Load trained model (cached)
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load("taxi_model_v2.pkl")

model = load_model()

# -------------------------------
# CaseLearn Branding
# -------------------------------
st.markdown(
    """
    <div style="display:flex; align-items:center; justify-content:flex-start; margin-left:0; padding-left:0;">
        <!-- Logo on the left -->
        <div>
            <img src="https://i.postimg.cc/gkG9vKXG/Case-Learn.png" width="200"/>
        </div>
        <!-- Text on the right -->
        <div style="margin-left:15px;">
            <div style="font-size:16px; font-weight:bold; color:#333;">CaseLearn</div>
            <div style="font-size:22px; font-weight:bold; color:#000; margin-top:3px;">
                Introduction to Machine Learning
            </div>
            <div style="font-size:14px; color:#555; margin-top:2px;">
                Supervised Learning - Regression
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# -------------------------------
# Page Title
# -------------------------------
st.title("🚖 Taxi Fare Prediction Case Study")

# -------------------------------
# Initialize session state
# -------------------------------
if "prediction" not in st.session_state:
    st.session_state.prediction = None

# -------------------------------
# Sidebar for inputs
# -------------------------------
st.sidebar.header("Trip Details")

time_of_day = st.sidebar.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
day_of_week = st.sidebar.selectbox("Day of Week", ["Weekday", "Weekend"])
traffic = st.sidebar.selectbox("Traffic Conditions", ["Low", "Medium", "High"])
weather = st.sidebar.selectbox("Weather", ["Clear", "Rain", "Snow"])

trip_distance = st.sidebar.number_input("Trip Distance (km)", min_value=0.5, max_value=100.0, value=5.0, step=0.5)
passenger_count = st.sidebar.number_input("Passenger Count", min_value=1, max_value=6, value=1, step=1)
base_fare = st.sidebar.number_input("Base Fare", min_value=1.0, max_value=20.0, value=3.0, step=0.5)
per_km_rate = st.sidebar.number_input("Per Km Rate", min_value=0.5, max_value=10.0, value=2.0, step=0.1)
per_minute_rate = st.sidebar.number_input("Per Minute Rate", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
trip_duration = st.sidebar.number_input("Trip Duration (minutes)", min_value=1, max_value=180, value=15, step=1)

# New input for range %
fare_range_pct = st.sidebar.number_input("Fare Range %", min_value=1, max_value=50, value=6, step=1)

# -------------------------------
# Prediction
# -------------------------------
if st.sidebar.button("Predict Fare"):
    input_data = pd.DataFrame([{
        "Time_of_Day": time_of_day,
        "Day_of_Week": day_of_week,
        "Traffic_Conditions": traffic,
        "Weather": weather,
        "Trip_Distance_km": trip_distance,
        "Passenger_Count": passenger_count,
        "Base_Fare": base_fare,
        "Per_Km_Rate": per_km_rate,
        "Per_Minute_Rate": per_minute_rate,
        "Trip_Duration_Minutes": trip_duration
    }])

    st.session_state.prediction = model.predict(input_data)[0]

# -------------------------------
# Display result (always visible)
# -------------------------------
if st.session_state.prediction is not None:
    pred = st.session_state.prediction
    lower = pred * (1 - fare_range_pct / 100)
    upper = pred * (1 + fare_range_pct / 100)

    prediction_text = f"Predicted Taxi Fare: ${pred:.2f}<br><br>" \
                      f"<span style='font-size:22px; color:#333;'>Expected Fare Range: " \
                      f"${lower:.2f} - ${upper:.2f}</span>"
else:
    prediction_text = "Pending Prediction..."

st.markdown(
    f"""
    <div style='text-align: center; background-color:#fff3e6; padding:20px; border-radius:15px;'>
        <h2 style='color:#ff6600;'>{prediction_text}</h2>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# About this app
# -------------------------------
st.markdown("---")
st.subheader("ℹ️ About this app")
st.markdown(
    """
    This application was created on behalf of **CaseLearn**  
    for the **Introduction to Machine Learning (Regression)** course.  

    Learn more at [CaseLearn.com](https://caselearn.com)
    """
)

