from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load your trained model
model = joblib.load("traffic_model.pkl")

@app.route("/", methods=["GET", "POST"])  # ✅ ADD methods
def home():
    prediction = None

    if request.method == "POST":
        try:
            # Read all 19 input features
            temp = float(request.form['temp'])
            rain = float(request.form['rain'])
            snow = float(request.form['snow'])
            hour = int(request.form['hour'])
            month = int(request.form['month'])
            day = int(request.form['day'])
            weekday = int(request.form['weekday'])
            is_weekend = int(request.form['is_weekend'])
            is_holiday = int(request.form['is_holiday'])

            weather_conditions = ['Clear', 'Clouds', 'Drizzle', 'Fog', 'Haze', 'Mist', 'Rain', 'Smoke', 'Snow', 'Thunderstorm']
            weather_features = [int(request.form[f'weather_{cond}']) for cond in weather_conditions]

            # Combine all features
            features = np.array([[temp, rain, snow, hour, month, day, weekday, is_weekend, is_holiday] + weather_features])

            prediction = int(model.predict(features)[0])
        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
