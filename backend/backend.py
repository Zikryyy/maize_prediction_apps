from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model with error handling
try:
    model = joblib.load("rf_model_maize_maturity.pkl")
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    model = None


# Logging setup
def log_request(data, prediction=None, error=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "timestamp": timestamp,
        "data": data,
        "prediction": prediction,
        "error": error
    }
    print(f"📝 Request log: {log_entry}")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        error_msg = "Model not loaded - service unavailable"
        log_request(None, error=error_msg)
        return jsonify({"error": error_msg}), 503

    data = request.get_json()

    # Input validation
    required_fields = ["R", "G", "B", "temperature", "humidity"]
    if not all(field in data for field in required_fields):
        error_msg = f"Missing required fields. Required: {required_fields}"
        log_request(data, error=error_msg)
        return jsonify({"error": error_msg}), 400

    try:
        # Convert and validate input values
        r = float(data["R"])
        g = float(data["G"])
        b = float(data["B"])
        temp = float(data["temperature"])
        hum = float(data["humidity"])

        # Validate ranges
        if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
            error_msg = "RGB values must be between 0-255"
            log_request(data, error=error_msg)
            return jsonify({"error": error_msg}), 400

        if not (15 <= temp <= 45):
            error_msg = "Temperature must be between 15-45°C"
            log_request(data, error=error_msg)
            return jsonify({"error": error_msg}), 400

        if not (0 <= hum <= 100):
            error_msg = "Humidity must be between 0-100%"
            log_request(data, error=error_msg)
            return jsonify({"error": error_msg}), 400

        # Make prediction
        features = np.array([[r, g, b, temp, hum]])
        prediction = model.predict(features)
        result = "Mature" if prediction[0] == 1 else "Immature"

        log_request(data, prediction=result)
        return jsonify({
            "prediction": result,
            "confidence": float(prediction[0])  # Convert numpy float to Python float
        })

    except ValueError as e:
        error_msg = f"Invalid input values: {str(e)}"
        log_request(data, error=error_msg)
        return jsonify({"error": error_msg}), 400
    except Exception as e:
        error_msg = f"Prediction error: {str(e)}"
        log_request(data, error=error_msg)
        return jsonify({"error": error_msg}), 500


@app.route("/health", methods=["GET"])
def health_check():
    status = {
        "status": "healthy" if model else "unhealthy",
        "model_loaded": bool(model),
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }
    return jsonify(status)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Maize Maturity Prediction API",
        "endpoints": {
            "/predict": "POST with RGB, temperature, humidity data",
            "/health": "GET service health status"
        },
        "documentation": "https://github.com/your-repo/docs"
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render provides PORT environment variable
    app.run(host="0.0.0.0", port=port, debug=False)