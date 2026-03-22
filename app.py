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
    print("⚠️ WARNING: WEATHER_API_KEY not set. Weather autofill disabled.")


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

    # -------------------------
    # INPUT SANITY CHECKS
    # -------------------------
    warnings = []

    # Temperature
    if temp > 45:
        warnings.append("Temperature is too high for most crops")
    elif temp < 5:
        warnings.append("Temperature is too low — frost risk for most crops")

    # pH
    if ph < 4.5:
        warnings.append("Soil pH is critically acidic — unsuitable for most crops")
    elif ph > 9.0:
        warnings.append("Soil pH is too alkaline — outside optimal farming range")
    elif ph < 5.5 or ph > 8.0:
        warnings.append("Soil pH is outside the optimal farming range (5.5–8.0)")

    # Rainfall
    if rainfall > 500:
        warnings.append("Rainfall is extremely high — high risk of waterlogging or flooding")
    elif rainfall > 300:
        warnings.append("Rainfall is high — ensure good drainage")

    # Humidity
    if humidity > 95:
        warnings.append("Humidity is extremely high — risk of fungal diseases")
    elif humidity < 10:
        warnings.append("Humidity is extremely low — severe drought risk")

    # Nitrogen
    if n > 130:
        warnings.append("Nitrogen level is very high — risk of over-fertilisation")

    # Extreme combination check
    if len(warnings) >= 3:
        warnings.insert(0, "Current conditions may not be suitable for reliable crop recommendations")

    # -------------------------

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
        explanation = []

    # -------------------------
    # NATURAL LANGUAGE SUMMARY
    # -------------------------
    def build_summary(explanation_list, crop_name):
        """Compose 1-2 advisor-style sentences from SHAP explanation."""

        FRIENDLY_NAMES = {
            "N": "nitrogen levels",
            "P": "phosphorus levels",
            "K": "potassium levels",
            "Temperature": "temperature",
            "Humidity": "humidity",
            "pH": "soil pH",
            "Rainfall": "rainfall"
        }

        positives = [e for e in explanation_list if e["impact"] > 0]
        negatives = [e for e in explanation_list if e["impact"] < 0]

        # Take top 2 positive, top 1 negative (already sorted by |impact|)
        top_pos = positives[:2]
        top_neg = negatives[:1]

        parts = []

        # Positive sentence
        if len(top_pos) >= 2:
            a = FRIENDLY_NAMES.get(top_pos[0]["feature"], top_pos[0]["feature"])
            b = FRIENDLY_NAMES.get(top_pos[1]["feature"], top_pos[1]["feature"])
            if a == b:
                parts.append(
                    f"{a.capitalize()} creates favorable conditions for {crop_name}, supporting healthy growth."
                )
            else:
                parts.append(
                    f"{a.capitalize()} and {b} create favorable conditions for {crop_name}, supporting healthy growth."
                )
        elif len(top_pos) == 1:
            a = FRIENDLY_NAMES.get(top_pos[0]["feature"], top_pos[0]["feature"])
            parts.append(
                f"{a.capitalize()} is well-suited for growing {crop_name}."
            )
        else:
            parts.append(
                f"Current conditions are acceptable for {crop_name}, though no single factor strongly favors it."
            )

        # Negative sentence
        if top_neg:
            n = FRIENDLY_NAMES.get(top_neg[0]["feature"], top_neg[0]["feature"])
            parts.append(
                f"However, improving {n} could further increase yield potential."
            )

        return " ".join(parts)

    # -------------------------

    # Build summary from top crop's explanation
    top_crop_name = model.classes_[np.argmax(probabilities)]
    summary = build_summary(explanation, top_crop_name)

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

    # -------------------------
    # CONFIDENCE FLAG
    # -------------------------
    # Count only the real warning messages (exclude the summary line added at index 0)
    real_warning_count = sum(1 for w in warnings if "may not be suitable" not in w)

    if top_conf < 40 or real_warning_count >= 2:
        confidence_flag = "Low Reliability"
    elif top_conf < 70 or real_warning_count == 1:
        confidence_flag = "Moderate Reliability"
    else:
        confidence_flag = "High Reliability"
    # -------------------------

    # -------------------------
    # SOIL STATUS
    # -------------------------
    npk_avg = (n + p + k) / 3
    soil_score = 0

    # NPK component (0-50)
    if npk_avg >= 60:
        soil_score += 50
    elif npk_avg >= 30:
        soil_score += 30
    else:
        soil_score += 10

    # pH component (0-50): ideal = 5.5-7.5
    if 5.5 <= ph <= 7.5:
        soil_score += 50
    elif 4.5 <= ph <= 8.5:
        soil_score += 30
    else:
        soil_score += 10

    if soil_score >= 80:
        soil_status = "Healthy"
    elif soil_score >= 50:
        soil_status = "Moderate"
    else:
        soil_status = "Poor"

    # -------------------------
    # CLIMATE STATUS
    # -------------------------
    climate_score = 0

    # Temperature (0-35): ideal = 15-35
    if 15 <= temp <= 35:
        climate_score += 35
    elif 5 <= temp <= 45:
        climate_score += 20
    else:
        climate_score += 5

    # Humidity (0-35): ideal = 40-80
    if 40 <= humidity <= 80:
        climate_score += 35
    elif 20 <= humidity <= 95:
        climate_score += 20
    else:
        climate_score += 5

    # Rainfall (0-30): ideal = 50-250
    if 50 <= rainfall <= 250:
        climate_score += 30
    elif 20 <= rainfall <= 400:
        climate_score += 18
    else:
        climate_score += 5

    if climate_score >= 80:
        climate_status = "Ideal"
    elif climate_score >= 50:
        climate_status = "Moderate"
    else:
        climate_status = "Unfavorable"

    # -------------------------
    # OVERALL SCORE (0-100)
    # -------------------------
    warning_penalty = min(real_warning_count * 8, 30)
    overall_score = round(
        top_conf * 0.40 +
        soil_score * 0.30 +
        climate_score * 0.20 -
        warning_penalty
    )
    overall_score = max(0, min(100, overall_score))

    # -------------------------
    # CONDITION SUMMARY
    # -------------------------
    soil_adj = {"Healthy": "healthy", "Moderate": "moderate", "Poor": "poor"}
    climate_adj = {"Ideal": "ideal", "Moderate": "moderate", "Unfavorable": "unfavorable"}

    condition_summary = (
        f"Soil health is {soil_adj[soil_status]} and climate conditions are {climate_adj[climate_status]} for farming."
    )
    if overall_score >= 70:
        condition_summary += " Overall, conditions are favorable for crop cultivation."
    elif overall_score >= 40:
        condition_summary += " Some adjustments may improve yield potential."
    else:
        condition_summary += " Significant improvements are needed for reliable crop production."
    # -------------------------

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

    # Build a short selection note comparing best vs second-best
    def build_decision_reason(best, all_results):
        """1-2 sentences explaining why best was chosen over the runner-up."""
        others = [r for r in all_results if r["crop"] != best["crop"]]
        if not others:
            return f"{best['crop']} is the top recommendation based on current conditions."

        runner = sorted(
            [r for r in all_results if r["crop"] != best["crop"]],
            key=lambda x: (x["profit"] * 0.7 + x["confidence"] * 1000 * 0.3),
            reverse=True
        )[0]
        parts = []

        # Lead: profit or confidence edge
        if best["profit"] > runner["profit"] and best["confidence"] > runner["confidence"]:
            parts.append(
                f"{best['crop']} was selected over {runner['crop']} for both higher profit potential and stronger model confidence."
            )
        elif best["profit"] > runner["profit"]:
            parts.append(
                f"{best['crop']} was selected over {runner['crop']} for its higher estimated profit."
            )
        elif best["confidence"] > runner["confidence"]:
            parts.append(
                f"{best['crop']} was selected over {runner['crop']} due to stronger suitability for the current conditions."
            )
        else:
            parts.append(
                f"{best['crop']} offers the best overall balance of profit and suitability compared to {runner['crop']}."
            )

        return " ".join(parts)

    selection_note = build_decision_reason(best_crop, results)

    decision = {
        "crop": best_crop["crop"],
        "profit": best_crop["profit"],
        "confidence": best_crop["confidence"],
        "reason": reasons,
        "selection_note": selection_note
    }

    return jsonify({
        "top_crops": results,
        "risk": risk,
        "best_crop": decision,
        "advice": advice,
        "explanation": explanation[:5],
        "summary": summary,
        "warnings": warnings,
        "confidence_flag": confidence_flag,
        "soil_status": soil_status,
        "climate_status": climate_status,
        "overall_score": overall_score,
        "condition_summary": condition_summary
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