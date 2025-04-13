from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os
from datetime import datetime
import sys
import traceback

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# --- Configuration ---
PORT = int(os.environ.get("PORT", 10000))
MODEL_FILE = "rf_model_maize_maturity.pkl"

# --- Model Loading with Error Handling ---
model = None
try:
    # Verify model file exists before loading
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Model file {MODEL_FILE} not found")

    if os.path.getsize(MODEL_FILE) == 0:
        raise ValueError("Model file is empty")

    model = joblib.load(MODEL_FILE)
    print(f"✅ Model loaded successfully from {MODEL_FILE}")
    print(f"Model type: {type(model)}")

except Exception as e:
    print(f"❌ Critical error loading model: {str(e)}", file=sys.stderr)
    print(traceback.format_exc(), file=sys.stderr)
    model = None


# --- Helper Functions ---
def validate_input(data):
    """Validate prediction input data"""
    required = {"R", "G", "B", "temperature", "humidity"}
    if not all(field in data for field in required):
        return False, "Missing required fields"

    try:
        if not (0 <= float(data["R"]) <= 255):
            return False, "R value must be 0-255"
        if not (0 <= float(data["G"]) <= 255):
            return False, "G value must be 0-255"
        if not (0 <= float(data["B"]) <= 255):
            return False, "B value must be 0-255"
        if not (15 <= float(data["temperature"]) <= 45):
            return False, "Temperature must be 15-45°C"
        if not (0 <= float(data["humidity"]) <= 100):
            return False, "Humidity must be 0-100%"
    except ValueError:
        return False, "Invalid numeric values"

    return True, ""


# --- Routes ---
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded", "status": "unavailable"}), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    # Validate input
    is_valid, msg = validate_input(data)
    if not is_valid:
        return jsonify({"error": msg}), 400

    try:
        # Prepare features
        features = np.array([
            [float(data["R"]),
             float(data["G"]),
             float(data["B"]),
             float(data["temperature"]),
             float(data["humidity"])]
        ])

        # Make prediction
        prediction = model.predict(features)
        result = "Mature" if prediction[0] == 1 else "Immature"

        return jsonify({
            "prediction": result,
            "confidence": float(prediction[0]),
            "status": "success"
        })

    except Exception as e:
        print(f"Prediction error: {str(e)}", file=sys.stderr)
        return jsonify({
            "error": "Prediction failed",
            "details": str(e),
            "status": "error"
        }), 500


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy" if model else "unhealthy",
        "model_loaded": bool(model),
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "dependencies": {
            "numpy": np.__version__,
            "scikit-learn": joblib.__version__.split('.')[0] + "." + joblib.__version__.split('.')[1]
            # Get sklearn version
        }
    })


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "Maize Maturity Prediction API",
        "status": "operational" if model else "degraded",
        "endpoints": {
            "/predict": "POST with RGB and environmental data",
            "/health": "GET service status"
        },
        "documentation": "https://your-docs-url.com"
    })


# --- Main ---
if __name__ == "__main__":
    # Print environment info for debugging
    print(f"Starting server with Python {sys.version}")
    print(f"NumPy version: {np.__version__}")
    print(f"Joblib version: {joblib.__version__}")
    print(f"Model status: {'Loaded' if model else 'Not loaded'}")

    # Start the server
    app.run(host="0.0.0.0", port=PORT, debug=False)