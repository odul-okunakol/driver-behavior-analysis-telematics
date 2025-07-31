import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go

# Headline
st.title("🚘 Drive Guard")
st.markdown("Enter the `insured_no` to see the driver's **predicted driving class** and some **important driving indicators**.")

# Upload the Model and the data
model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")
X_test = pd.read_csv("X_test.csv")
X_test_scaled = pd.read_csv("X_test_scaled.csv")

label_map = {0: "Normal", 1: "Reckless", 2: "Aggressive"}
color_map = {
    "Normal": "#d4edda",   # green
    "Reckless": "#fff3cd",  # yellow
    "Aggressive": "#f8d7da"    # red
}

insured_ids = X_test["insured_no"].tolist()

# === Tabs ===
tab1, tab2 = st.tabs(["🔎 Predict by Insured ID", "🧮 Manual Input for Risk Class"])

def show_prediction(prediction_label):
    st.subheader("🚦 Predicted Driving Class:")
    bg_color = color_map.get(prediction_label, "#e2e3e5")
    st.markdown(
        f"""
        <div style='padding: 20px; border-radius: 8px; background-color: {bg_color}; 
                    border: 1px solid rgba(0,0,0,0.1); text-align:center;'>
            <h2 style='margin: 0;'><strong>{prediction_label}</strong></h2>
        </div>
        """,
        unsafe_allow_html=True
    )

with tab1:
    insured_no_input = st.text_input("🔢 Please enter `insured_no`:")

    if insured_no_input:
        try:
            insured_no = int(insured_no_input)
            if insured_no in insured_ids:
                idx = insured_ids.index(insured_no)
                x_scaled = X_test_scaled.iloc[idx]
                prediction_encoded = model.predict([x_scaled])[0]
                prediction_label = label_map.get(prediction_encoded, "Unknown")

                # Results
                show_prediction(prediction_label)

                # Driver Indicators
                st.markdown("### 📊 Driver Indicators")
                driver_data = X_test.iloc[idx]
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("🔺 Avg Speed (km/h)", f"{driver_data['avg_speed']:.1f}")
                    st.metric("🌀 Avg Acceleration", f"{driver_data['avg_acc_total']:.3f}")
                with col2:
                    st.metric("🔧 Std. Deviation RPM", f"{driver_data['std_rpm']:.0f}")

                # 📈 Graphs
                st.markdown("### 📈 Driving Behavior Charts")

                # Speed Gauge
                speed_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=driver_data["avg_speed"],
                    title={'text': "Average Speed (km/h)"},
                    gauge={'axis': {'range': [0, 150]},
                           'bar': {'color': "green"}}
                ))
                st.plotly_chart(speed_gauge, use_container_width=True)

                # Acceleration Bar
                acc_bar = go.Figure(data=[
                    go.Bar(name='Avg Acceleration', x=["Acceleration"], y=[driver_data["avg_acc_total"]],
                           marker_color='orange')
                ])
                acc_bar.update_layout(title="Average Acceleration", yaxis_range=[0, 1.5])
                st.plotly_chart(acc_bar, use_container_width=True)

                # RPM Gauge
                rpm_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=driver_data["std_rpm"],
                    title={'text': "Standard Deviation RPM"},
                    gauge={'axis': {'range': [0, 4000]},
                           'bar': {'color': "red"}}
                ))
                st.plotly_chart(rpm_gauge, use_container_width=True)
            else:
                st.warning("❌ The `insured_no` you entered was not found in the test data.")
        except ValueError:
            st.warning("⚠️ Please enter a valid number.")

with tab2:
    st.markdown("### Fill in the fields below to predict the driving class :")

    # Manual input fields
    day_night = st.selectbox("🌗 Driving Time", options=["day", "night"])
    avg_speed = st.slider("🚗 Avg Speed (km/h)", min_value=0.0, max_value=150.0, step=1.0)
    avg_acc_total = st.slider("💨 Avg Acceleration", min_value=0.5, max_value=1.2, step=0.01)
    std_rpm = st.slider("🔧 RPM Standard Deviation", min_value=500, max_value=4000, step=50)

    if st.button("Predict"):
        # Model inputs
        model_features = X_test.drop(columns=["insured_no"]).columns
        input_df = pd.DataFrame([np.zeros(len(model_features))], columns=model_features)

        # Insert user inputs"
        input_df["avg_speed"] = avg_speed
        input_df["avg_acc_total"] = avg_acc_total
        input_df["std_rpm"] = std_rpm
        input_df["day_night_night"] = 1 if day_night == "night" else 0

        # Scaling
        input_scaled = scaler.transform(input_df)

        # Prediction
        prediction_encoded = model.predict(input_scaled)[0]
        prediction_label = label_map.get(prediction_encoded, "Unknown")

        # Results
        show_prediction(prediction_label)

        # Graps
        st.markdown("### 📈 Driving Behavior Charts")

        # Speed Gauge
        speed_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg_speed,
            title={'text': "Average Speed (km/h)"},
            gauge={'axis': {'range': [0, 150]},
                   'bar': {'color': "green"}}
        ))
        st.plotly_chart(speed_gauge, use_container_width=True, key="manual_speed_gauge")

        # Acceleration Bar
        acc_bar = go.Figure(data=[
            go.Bar(name='Avg Acceleration', x=["Acceleration"], y=[avg_acc_total],
                   marker_color='orange')
        ])
        acc_bar.update_layout(title="Average Acceleration", yaxis_range=[0, 1.5])
        st.plotly_chart(acc_bar, use_container_width=True, key="manual_acc_bar")

        # RPM Gauge
        rpm_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=std_rpm,
            title={'text': "Standard Deviation RPM"},
            gauge={'axis': {'range': [0, 4000]},
                   'bar': {'color': "red"}}
        ))
        st.plotly_chart(rpm_gauge, use_container_width=True, key="manual_rpm_gauge")



