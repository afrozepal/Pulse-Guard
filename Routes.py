from flask import Flask, request, render_template
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the pre-trained model
with open("heart_disease_model.pkl", "rb") as file:
    model = pickle.load(file)


# Load the scaler
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)


# Home route
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/form")
def form():
    return render_template("form.html")


# Prediction route
@app.route("/predict", methods=["POST"])
@app.route("/predict", methods=["POST"])
def predict():
    features = [
        request.form.get(key)
        for key in [
            "Age",
            "Sex",
            "Chest_pain_type",
            "BP",
            "Cholesterol",
            "FBS_over_120",
            "EKG_results",
            "Max_HR",
            "Exercise_angina",
            "ST_depression",
            "Slope_of_ST",
            "Number_of_vessels_fluro",
            "Thallium",
        ]
    ]

    # Convert input values to appropriate data types
    features = [
        (
            float(val)
            if val.replace(".", "", 1).isdigit()
            else (
                float(val)
                if val.replace(".", "", 1).replace("-", "", 1).isdigit()
                else val
            )
        )
        for val in features
    ]

    # Scale the input features using the loaded scaler
    input_data_scaled = scaler.transform([features])

    # Make predictions
    prediction = model.predict(input_data_scaled)

    # Define classes
    classes = ["No Heart Disease", "Heart Disease"]

    # Get the predicted class
    predicted_class = classes[prediction[0]]
    return render_template("result.html", prediction=predicted_class)


if __name__ == "__main__":
    app.run(debug=True)
