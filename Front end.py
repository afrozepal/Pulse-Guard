from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load the pre-trained model
with open('heart_disease_model.pkl', 'rb') as file:
    model = pickle.load(file)

from flask import Flask, render_template

app = Flask(__name__, static_url_path='/static')

# Home route
@app.route('/')
def home():
    return render_template('Form.html')


# Prediction route
# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    features = [request.form.get(key) for key in ['Age', 'Sex', 'Chest_pain_type', 'BP', 'Cholesterol', 'FBS_over_120', 'EKG_results', 'Max_HR', 'Exercise_angina', 'ST_depression', 'Slope_of_ST', 'Number_of_vessels_fluro', 'Thallium']]
    
    # Convert string values to appropriate data types
    try:
        features = [int(val) if val.isdigit() else float(val) if val.replace('.', '', 1).isdigit() else val for val in features]
    except ValueError as e:
        print("Error converting form input to data types:", e)
        print("Form input values:", features)
        return "Error processing form input", 400

    # Make prediction
    prediction = model.predict([features])
    print("Prediction is , " , prediction)
    # Define classes
    classes = ['No Heart Disease', 'Heart Disease']
    
    # Get the predicted class
    predicted_class = classes[(prediction[0])]
    return render_template('result.html', prediction=predicted_class)



if __name__ == '__main__':
    app.run(debug=True)
