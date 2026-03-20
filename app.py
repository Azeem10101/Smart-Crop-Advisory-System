from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import requests

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

# OPTIONAL: Weather API (you can enable later)
API_KEY = "YOUR_API_KEY"  # put later

def get_weather(city):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        res = requests.get(url).json()

        return {
            "temperature": res["main"]["temp"],
            "humidity": res["main"]["humidity"]
        }
    except:
        return None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = np.array(data["features"]).reshape(1, -1)

    prediction = model.predict(features)[0]

    probabilities = model.predict_proba(features)[0]

    # SMART CONFIDENCE (difference-based)
    sorted_probs = sorted(probabilities, reverse=True)
    confidence = sorted_probs[0] - sorted_probs[1]

    confidence = min(confidence * 2, 1.0)  # scale + cap

    # Risk logic
    if confidence < 0.3:
        risk = "High"
    elif confidence < 0.6:
        risk = "Medium"
    else:
        risk = "Low"

    return jsonify({
        "prediction": prediction,
        "confidence": round(confidence * 100, 2),
        "risk": risk
    })

# FUTURE ROUTE (weather-based prediction)
@app.route("/predict_with_weather", methods=["POST"])
def predict_weather():
    data = request.json
    city = data["city"]

    weather = get_weather(city)

    if not weather:
        return jsonify({"error": "Weather fetch failed"})

    # You can merge weather into features later
    return jsonify({
        "weather": weather,
        "message": "Weather integration ready"
    })

if __name__ == "__main__":
    app.run(debug=True)