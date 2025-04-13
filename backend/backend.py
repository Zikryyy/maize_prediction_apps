from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the model
try:
    model = joblib.load("rf_model_maize_maturity.pkl")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()
    try:
        # Validate input data
        if not all(key in data for key in ["R", "G", "B", "temperature", "humidity"]):
            return jsonify({"error": "Missing required fields"}), 400

        r = float(data["R"])
        g = float(data["G"])
        b = float(data["B"])
        temp = float(data["temperature"])
        hum = float(data["humidity"])

        features = np.array([[r, g, b, temp, hum]])
        prediction = model.predict(features)
        result = "Mature" if prediction[0] == 1 else "Immature"
        return jsonify({"prediction": result})
    except ValueError as e:
        return jsonify({"error": "Invalid input values"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "model_loaded": model is not None})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Default to 10000 if PORT not set
    app.run(host="0.0.0.0", port=port, debug=False)  # debug=False for production