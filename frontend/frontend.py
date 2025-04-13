import streamlit as st
import requests
import numpy as np
from PIL import Image
import pandas as pd
import os
import matplotlib.pyplot as plt

# Page settings
st.set_page_config(
    page_title="Maize Maturity Predictor",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom dark theme with blue highlights
st.markdown("""
    <style>
        body {
            background-color: #121212;
            color: #e0e0e0;
        }
        .stApp {
            background-color: #121212;
            color: #e0e0e0;
        }
        .css-1v0mbdj, .css-1v0mbdj h1, .css-1v0mbdj h2, .css-1v0mbdj h3 {
            color: #00b4d8;
        }
        .stButton>button {
            background-color: #00b4d8;
            color: white;
            border: none;
        }
        .stDownloadButton>button {
            background-color: #0077b6;
            color: white;
        }
        .stSlider .st-bo {
            color: #00b4d8;
        }
        .stRadio > div {
            color: #e0e0e0;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🌽 Maize Maturity Predictor")

HISTORY_FILE = "prediction_history.csv"

# Load history
if "history" not in st.session_state:
    if os.path.exists(HISTORY_FILE):
        st.session_state.history = pd.read_csv(HISTORY_FILE).to_dict(orient="records")
    else:
        st.session_state.history = []

# Input mode
st.subheader("🎛️ Choose Input Mode")
mode = st.radio("", ["Manual RGB Entry", "Upload Image for RGB"])

if mode == "Manual RGB Entry":
    r = st.number_input("🔴 R", min_value=0, max_value=255, value=100)
    g = st.number_input("🟢 G", min_value=0, max_value=255, value=100)
    b = st.number_input("🔵 B", min_value=0, max_value=255, value=100)
else:
    uploaded_file = st.file_uploader("📸 Upload an image of the maize kernel", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

        # Resize and extract
        resized = image.resize((100, 100))
        img_np = np.array(resized)
        avg_color = img_np.mean(axis=(0, 1)).astype(int)
        r, g, b = int(avg_color[0]), int(avg_color[1]), int(avg_color[2])

        st.success(f"🎨 Extracted RGB → R: {r}, G: {g}, B: {b}")

        # RGB heatmap
        st.subheader("🌈 RGB Heatmap")
        fig, axs = plt.subplots(1, 3, figsize=(12, 3))
        cmap_labels = ['R', 'G', 'B']
        for i, ax in enumerate(axs):
            ax.imshow(img_np[:, :, i], cmap='Reds' if i == 0 else 'Greens' if i == 1 else 'Blues')
            ax.set_title(f"{cmap_labels[i]} Channel", color="#00b4d8")
            ax.axis("off")
        st.pyplot(fig)

# Environmental data
st.subheader("🌡️ Environmental Data")
temp = st.slider("Temperature (°C)", 20.0, 35.0, 25.0)
hum = st.slider("Humidity (%)", 30.0, 80.0, 50.0)

# Predict
if st.button("🚀 Predict Maturity"):
    data = {
        "R": r, "G": g, "B": b,
        "temperature": temp,
        "humidity": hum
    }
    try:
        response = requests.post("http://127.0.0.1:5000/predict", json=data)
        result = response.json()
        if "prediction" in result:
            prediction = result["prediction"]
            st.success(f"✅ Prediction: **{prediction}**")

            entry = {
                "R": r, "G": g, "B": b,
                "Temp": temp, "Humidity": hum,
                "Prediction": prediction
            }
            st.session_state.history.append(entry)
            pd.DataFrame(st.session_state.history).to_csv(HISTORY_FILE, index=False)
        else:
            st.error(f"⚠️ Error: {result.get('error', 'Unknown error')}")
    except Exception as e:
        st.error(f"🚫 Failed to connect to server: {e}")

# History
if st.session_state.history:
    st.subheader("📜 Prediction History")
    for i, entry in enumerate(reversed(st.session_state.history), 1):
        st.markdown(f"""
        <div style="background-color:#1a1a1a; padding:10px; border-radius:10px; margin-bottom:10px">
        <strong>#{i}</strong><br>
        🌈 RGB: ({entry['R']}, {entry['G']}, {entry['B']})<br>
        🌡️ Temp: {entry['Temp']}°C<br>
        💧 Humidity: {entry['Humidity']}%<br>
        🔍 Result: <strong>{entry['Prediction']}</strong>
        </div>
        """, unsafe_allow_html=True)

    # Export CSV
    df = pd.DataFrame(st.session_state.history)
    st.download_button(
        label="📥 Download Prediction History",
        data=df.to_csv(index=False),
        file_name="maize_prediction_history.csv",
        mime="text/csv"
    )
