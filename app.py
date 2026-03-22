from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import requests
import shap
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))

# SHAP explainer
explainer = shap.TreeExplainer(model)

# API Key
API_KEY = os.getenv("WEATHER_API_KEY")
if not API_KEY:
    raise ValueError("API key not found. Set WEATHER_API_KEY in .env file")


# Crop Data
CROP_DATA = {
    "Rice": {"price": 2200, "yield": 25},
    "Maize": {"price": 1800, "yield": 20},
    "Chickpea": {"price": 5000, "yield": 10},
    "KidneyBeans": {"price": 6000, "yield": 9},
    "PigeonPeas": {"price": 5500, "yield": 8},
    "MothBeans": {"price": 4500, "yield": 7},
    "MungBean": {"price": 4800, "yield": 9},
    "Blackgram": {"price": 5000, "yield": 8},
    "Lentil": {"price": 5200, "yield": 7},
    "Pomegranate": {"price": 8000, "yield": 12},
    "Banana": {"price": 3000, "yield": 30},
    "Mango": {"price": 6000, "yield": 15},
    "Grapes": {"price": 7000, "yield": 14},
    "Watermelon": {"price": 2000, "yield": 18},
    "Muskmelon": {"price": 2200, "yield": 16},
    "Apple": {"price": 9000, "yield": 10},
    "Orange": {"price": 4000, "yield": 12},
    "Papaya": {"price": 3500, "yield": 20},
    "Coconut": {"price": 4500, "yield": 18},
    "Cotton": {"price": 7000, "yield": 12},
    "Jute": {"price": 3000, "yield": 15},
    "Coffee": {"price": 8500, "yield": 10}
}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    if not data or "features" not in data:
        return jsonify({"error": "Invalid input"})

    if len(data["features"]) != 7:
        return jsonify({"error": "Expected 7 input features"})

    features = np.array(data["features"]).reshape(1, -1)

    # Feature unpacking
    n, p, k, temp, humidity, ph, rainfall = features[0]

    # Advice system
    advice = []

    if n < 50:
        advice.append("Increase nitrogen levels in soil")

    if humidity < 50:
        advice.append("Irrigation recommended to improve humidity")

    if ph < 5.5:
        advice.append("Soil is too acidic, consider liming")

    if rainfall > 250:
        advice.append("Ensure proper drainage to avoid waterlogging")

    if not advice:
        advice.append("Conditions are optimal for this crop")

    # Prediction
    probabilities = model.predict_proba(features)[0]

    # -------------------------
    # SHAP EXPLANATION (ROBUST)
    # -------------------------
    try:
        shap_values = explainer(features)

        top_class_index = np.argmax(probabilities)

        # Handles newer SHAP versions
        if hasattr(shap_values, "values"):
            shap_contributions = shap_values.values[0][top_class_index]
        else:
            # Fallback for older versions
            shap_values = explainer.shap_values(features)
            shap_contributions = shap_values[top_class_index][0]

        feature_names = ["N", "P", "K", "Temperature", "Humidity", "pH", "Rainfall"]

        explanation = []

        for i in range(len(feature_names)):
            value = shap_contributions[i]
            feature = feature_names[i]

            # Convert to meaningful explanation
            if feature == "Temperature":
                if value > 0:
                    text = "Temperature supports crop growth"
                else:
                    text = "Temperature is not ideal for this crop"

            elif feature == "Humidity":
                if value > 0:
                    text = "Humidity level is favorable"
                else:
                    text = "Humidity is slightly lower than ideal"

            elif feature == "pH":
                if value > 0:
                    text = "Soil pH is suitable for this crop"
                else:
                    text = "Soil pH may not be optimal"

            elif feature == "Rainfall":
                if value > 0:
                    text = "Rainfall conditions are beneficial"
                else:
                    text = "Rainfall may be insufficient"

            elif feature == "N":
                if value > 0:
                    text = "Nitrogen level supports growth"
                else:
                    text = "Nitrogen level is slightly low"

            elif feature == "P":
                if value > 0:
                    text = "Phosphorus level is adequate"
                else:
                    text = "Phosphorus level could be improved"

            elif feature == "K":
                if value > 0:
                    text = "Potassium level is good"
                else:
                    text = "Potassium level is slightly low"

            explanation.append({
                "feature": feature,
                "impact": float(round(value, 3)),
                "reason": text
            })

        # Sort by importance
        explanation = sorted(explanation, key=lambda x: abs(x["impact"]), reverse=True)

    except Exception as e:
        explanation = [{
            "feature": "Error",
            "impact": 0,
            "reason": "Unable to generate explanation"
        }]

    # -------------------------

    # Top 3 crops
    top_indices = probabilities.argsort()[-3:][::-1]

    results = []
    for idx in top_indices:
        crop = model.classes_[idx]
        prob = probabilities[idx]

        crop_info = CROP_DATA.get(crop, {"price": 0, "yield": 0})
        profit = crop_info["price"] * crop_info["yield"]

        results.append({
            "crop": crop,
            "confidence": round(prob * 100, 2),
            "profit": profit
        })

    # Risk calculation
    top_conf = results[0]["confidence"]

    if top_conf < 40:
        risk = "High"
    elif top_conf < 70:
        risk = "Medium"
    else:
        risk = "Low"

    # Best crop decision
    best_crop = max(
        results,
        key=lambda x: (x["profit"] * 0.7 + x["confidence"] * 1000 * 0.3)
    )

    reasons = []

    if best_crop["profit"] > 50000:
        reasons.append("High profit potential")

    if best_crop["confidence"] > 60:
        reasons.append("Strong model confidence")
    elif best_crop["confidence"] > 40:
        reasons.append("Moderate confidence")

    if not reasons:
        reasons.append("Balanced choice based on conditions")

    decision = {
        "crop": best_crop["crop"],
        "profit": best_crop["profit"],
        "confidence": best_crop["confidence"],
        "reason": reasons
    }

    return jsonify({
        "top_crops": results,
        "risk": risk,
        "best_crop": decision,
        "advice": advice,
        "explanation": explanation[:5]
    })


@app.route("/get_weather", methods=["POST"])
def get_weather_data():
    data = request.json
    city = data.get("city")

    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

    try:
        res = requests.get(url).json()
    except Exception:
        return jsonify({"error": "Failed to fetch weather data"})

    if "main" in res:
        return jsonify({
            "temperature": res["main"]["temp"],
            "humidity": res["main"]["humidity"]
        })
    else:
        return jsonify({"error": res.get("message", "Unknown error")})


if __name__ == "__main__":
    app.run(debug=True)