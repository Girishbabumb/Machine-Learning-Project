import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Live Hybrid Solar-Wind Forecasting", layout="wide")

# Load trained artifacts from train_model.py
artifacts = joblib.load("hybrid_forecast_artifacts.pkl")
scaler = artifacts["scaler"]
model_solar = artifacts["model_solar"]
model_wind = artifacts["model_wind"]
model_hybrid = artifacts["model_hybrid"]
metrics = artifacts["metrics"]

LOCATIONS = {
    "Bengaluru / Harohalli (DSU Campus)": (12.6606, 77.4485),
    "Pavagada Solar Park, Karnataka": (14.2690, 77.4121),
    "Muppandal Wind Farm, Tamil Nadu": (8.2583, 77.5350),
    "Jaisalmer Hybrid Park, Rajasthan": (26.9157, 70.9083),
    "Kutch Renewable Park, Gujarat": (23.7337, 69.8597),
}


@st.cache_data(ttl=600)
def fetch_live_weather(lat: float, lon: float):
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&hourly=shortwave_radiation,direct_normal_irradiance,wind_speed_10m,wind_speed_80m,"
        f"wind_direction_10m,temperature_2m,relative_humidity_2m,cloud_cover,surface_pressure,rain"
        f"&wind_speed_unit=ms&timezone=Asia%2FKolkata&forecast_days=2"
    )
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()


def build_features(row_dict: dict) -> np.ndarray:
    df_feat = pd.DataFrame([row_dict])
    return scaler.transform(df_feat)


st.title("⚡ Live Hybrid Solar-Wind Energy Forecasting System")
st.caption("Department of CSE (AI&ML) | Dayananda Sagar University | Real-Time Multimodal Inference & Grid Stability")

tab_live, tab_sim = st.tabs(["🌐 Live Real-Time Weather & 24-Hr Forecast", "🎛️ Manual What-If Simulator"])

with tab_live:
    col_loc, col_btn = st.columns([3, 1])
    with col_loc:
        selected_site = st.selectbox("📍 Select Hybrid Plant / Grid Location:", list(LOCATIONS.keys()))
    with col_btn:
        st.write("")
        st.write("")
        if st.button("🔄 Refresh Live Data"):
            st.cache_data.clear()

    lat, lon = LOCATIONS[selected_site]

    try:
        raw_data = fetch_live_weather(lat, lon)
        hourly = raw_data["hourly"]

        live_df = pd.DataFrame({
            "timestamp": pd.to_datetime(hourly["time"]),
            "solar_irradiance": hourly["shortwave_radiation"],
            "dni_irradiance": hourly["direct_normal_irradiance"],
            "wind_speed": hourly["wind_speed_10m"],
            "wind_speed_80m": hourly["wind_speed_80m"],
            "wind_direction": hourly["wind_direction_10m"],
            "temperature": hourly["temperature_2m"],
            "humidity": hourly["relative_humidity_2m"],
            "cloud_cover": hourly["cloud_cover"],
            "pressure": hourly["surface_pressure"],
            "rainfall": hourly["rain"],
        })

        now_ist = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).floor("h")
        future_df = live_df[live_df["timestamp"] >= now_ist].head(25).reset_index(drop=True)

        # Autoregressive multimodal inference for next 24 hours
        prev_s = max(0.0, (future_df.loc[0, "solar_irradiance"] / 1000.0) * 50.0)
        w0 = future_df.loc[0, "wind_speed"]
        prev_w = 0.0 if w0 < 3 else (50.0 if w0 > 14 else 50.0 * ((w0 - 3) / 11) ** 3)

        pred_solar_list, pred_wind_list, pred_hybrid_list = [], [], []

        for _, r in future_df.iterrows():
            hr = r["timestamp"].hour
            mo = r["timestamp"].month
            w_dir = r["wind_direction"]

            feat_row = {
                "solar_irradiance": r["solar_irradiance"],
                "wind_speed": r["wind_speed"],
                "wind_dir_sin": np.sin(np.deg2rad(w_dir)),
                "wind_dir_cos": np.cos(np.deg2rad(w_dir)),
                "temperature": r["temperature"],
                "humidity": r["humidity"],
                "cloud_cover": r["cloud_cover"],
                "pressure": r["pressure"],
                "rainfall": r["rainfall"],
                "hour_sin": np.sin(2 * np.pi * hr / 24),
                "hour_cos": np.cos(2 * np.pi * hr / 24),
                "month": mo,
                "prev_solar_mw": prev_s,
                "prev_wind_mw": prev_w,
            }

            X_scaled = build_features(feat_row)
            p_s = 0.0 if r["solar_irradiance"] < 10 else max(0.0, float(model_solar.predict(X_scaled)[0]))
            p_w = max(0.0, float(model_wind.predict(X_scaled)[0]))
            p_h = max(0.0, float(model_hybrid.predict(X_scaled)[0]))

            pred_solar_list.append(round(p_s, 2))
            pred_wind_list.append(round(p_w, 2))
            pred_hybrid_list.append(round(p_h, 2))
            prev_s, prev_w = p_s, p_w

        future_df["Predicted Solar (MW)"] = pred_solar_list
        future_df["Predicted Wind (MW)"] = pred_wind_list
        future_df["Predicted Hybrid (MW)"] = pred_hybrid_list

        # Row 1: Live Meteorological Telemetry
        curr = future_df.iloc[0]
        st.subheader(f"📡 Live Meteorological Telemetry ({curr['timestamp'].strftime('%d %b %Y, %I:%M %p IST')})")
        w1, w2, w3, w4 = st.columns(4)
        w1.metric("☀️ GHI / DNI Irradiance", f"{curr['solar_irradiance']:.0f} / {curr['dni_irradiance']:.0f} W/m²")
        w2.metric("🌬️ Wind (10m / 80m Hub)", f"{curr['wind_speed']:.1f} / {curr['wind_speed_80m']:.1f} m/s")
        w3.metric("🌡️ Temp / Cloud Cover", f"{curr['temperature']:.1f} °C | {curr['cloud_cover']:.0f}%")
        w4.metric("💧 Humidity / Pressure", f"{curr['humidity']:.0f}% | {curr['pressure']:.0f} hPa")

        st.divider()

        # Row 2: Live ML Power Forecasts & Grid Stability
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("☀️ Forecasted Solar Output", f"{curr['Predicted Solar (MW)']:.2f} MW")
        m2.metric("🌬️ Forecasted Wind Output", f"{curr['Predicted Wind (MW)']:.2f} MW")
        m3.metric("⚡ Total Hybrid Output", f"{curr['Predicted Hybrid (MW)']:.2f} MW")

        # Calculate 24-hr stability summary
        avg_24h = future_df["Predicted Hybrid (MW)"].mean()
        grid_msg = "Balanced (Optimal)" if curr["Predicted Hybrid (MW)"] >= 18 else "Low Output (Battery Backup)"
        m4.metric("🔌 Grid Dispatch Advisory", grid_msg, delta=f"24h Avg: {avg_24h:.1f} MW")

        # Row 3: Chart + Model Benchmark Table
        chart_col, eval_col = st.columns([2, 1])
        with chart_col:
            st.subheader("📈 Next 24-Hour Hybrid Generation Schedule")
            chart_df = future_df.set_index("timestamp")[
                ["Predicted Solar (MW)", "Predicted Wind (MW)", "Predicted Hybrid (MW)"]
            ]
            st.line_chart(chart_df)

        with eval_col:
            st.subheader("📊 Model Benchmark (Test Set)")
            st.dataframe(pd.DataFrame(metrics).T, use_container_width=True)
            csv_bytes = future_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download 24-Hr Grid Schedule (CSV)",
                data=csv_bytes,
                file_name="live_hybrid_grid_forecast.csv",
                mime="text/csv",
            )

        with st.expander("📋 View Full Hourly Live Sensor Feed & Predictions"):
            st.dataframe(future_df, use_container_width=True)

    except Exception as e:
        st.error(f"Could not fetch live weather API data: {e}")

with tab_sim:
    st.subheader("🎛️ Stress-Test Grid Stability Under Custom Weather Conditions")
    c1, c2, c3 = st.columns(3)
    with c1:
        sim_irr = st.slider("Solar Irradiance (W/m²)", 0.0, 1000.0, 700.0)
        sim_wind = st.slider("Wind Speed (m/s)", 0.0, 25.0, 10.0)
        sim_wdir = st.slider("Wind Direction (°)", 0, 360, 180)
        sim_temp = st.slider("Ambient Temperature (°C)", 10.0, 45.0, 28.0)
    with c2:
        sim_hum = st.slider("Humidity (%)", 15.0, 100.0, 50.0)
        sim_cloud = st.slider("Cloud Cover (%)", 0.0, 100.0, 20.0)
        sim_press = st.number_input("Pressure (hPa)", 990.0, 1030.0, 1013.0)
        sim_rain = st.number_input("Rainfall (mm)", 0.0, 50.0, 0.0)
    with c3:
        sim_hr = st.slider("Hour of Day", 0, 23, 14)
        sim_mo = st.slider("Month", 1, 12, 5)
        sim_ps = st.number_input("Prior Solar (MW)", 0.0, 50.0, 30.0)
        sim_pw = st.number_input("Prior Wind (MW)", 0.0, 50.0, 14.0)

    sim_row = {
        "solar_irradiance": sim_irr,
        "wind_speed": sim_wind,
        "wind_dir_sin": np.sin(np.deg2rad(sim_wdir)),
        "wind_dir_cos": np.cos(np.deg2rad(sim_wdir)),
        "temperature": sim_temp,
        "humidity": sim_hum,
        "cloud_cover": sim_cloud,
        "pressure": sim_press,
        "rainfall": sim_rain,
        "hour_sin": np.sin(2 * np.pi * sim_hr / 24),
        "hour_cos": np.cos(2 * np.pi * sim_hr / 24),
        "month": sim_mo,
        "prev_solar_mw": sim_ps,
        "prev_wind_mw": sim_pw,
    }
    sim_scaled = build_features(sim_row)
    s_out = 0.0 if sim_irr < 10 else max(0.0, float(model_solar.predict(sim_scaled)[0]))
    w_out = max(0.0, float(model_wind.predict(sim_scaled)[0]))
    h_out = max(0.0, float(model_hybrid.predict(sim_scaled)[0]))

    st.divider()
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("☀️ Simulated Solar", f"{s_out:.2f} MW")
    sc2.metric("🌬️ Simulated Wind", f"{w_out:.2f} MW")
    sc3.metric("⚡ Simulated Hybrid", f"{h_out:.2f} MW")