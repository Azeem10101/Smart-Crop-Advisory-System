from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

API_KEY = os.getenv("WEATHER_API_KEY")
if not API_KEY:
    raise ValueError("API key not found. Set WEATHER_API_KEY in .env file")


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

    # 🌱 ADVICE SYSTEM INPUT
    n, p, k, temp, humidity, ph, rainfall = features[0]

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

    probabilities = model.predict_proba(features)[0]

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

    top_conf = results[0]["confidence"]

    if top_conf < 40:
        risk = "High"
    elif top_conf < 70:
        risk = "Medium"
    else:
        risk = "Low"

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
        "advice": advice
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