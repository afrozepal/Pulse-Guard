import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('heart_disease_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Define the Streamlit app
# Define the Streamlit app
def main():
    # Set the title of the app
    st.title('Heart Disease Prediction')

    # Add input fields for user to enter data
    age = st.number_input('Age', min_value=0, max_value=120, value=40)
    sex = st.radio('Sex', ['Female', 'Male'])
    cp = st.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
    bp = st.number_input('Blood Pressure (BP)', min_value=0, max_value=300, value=120)
    cholesterol = st.number_input('Cholesterol', min_value=0, max_value=1000, value=200)
    fbs = st.radio('Fasting Blood Sugar (FBS) > 120', ['No', 'Yes'])
    ekg = st.selectbox('Electrocardiographic Results (EKG)', ['Normal', 'Abnormality', 'Hypertrophy'])
    max_hr = st.number_input('Maximum Heart Rate (Max HR)', min_value=0, max_value=300, value=150)
    exercise_angina = st.radio('Exercise Induced Angina', ['No', 'Yes'])
    st_depression = st.number_input('ST Depression', min_value=0.0, max_value=10.0, value=0.0)
    slope_of_st = st.number_input('Slope of ST Segment', min_value=0, max_value=3, value=1)
    num_vessels_fluro = st.number_input('Number of Vessels Fluro', min_value=0, max_value=4, value=0)
    thallium = st.selectbox('Thallium Stress Test Result', ['Normal', 'Fixed Defect', 'Reversible Defect'])

    # Convert categorical features to numerical
    sex = 0 if sex == 'Female' else 1
    fbs = 1 if fbs == 'Yes' else 0
    exercise_angina = 1 if exercise_angina == 'Yes' else 0

    cp_mapping = {'Typical Angina': 1, 'Atypical Angina': 2, 'Non-anginal Pain': 3, 'Asymptomatic': 4}
    ekg_mapping = {'Normal': 0, 'Abnormality': 1, 'Hypertrophy': 2}
    thallium_mapping = {'Normal': 0, 'Fixed Defect': 3, 'Reversible Defect': 7}
    cp = cp_mapping[cp]
    ekg = ekg_mapping[ekg]
    thallium = thallium_mapping[thallium]

    # Preprocess input data
    input_data = np.array([[age, sex, cp, bp, cholesterol, fbs, ekg, max_hr, exercise_angina, st_depression,
                            slope_of_st, num_vessels_fluro, thallium]])  # Add other features

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Make predictions
    if st.button('Predict'):
        prediction = model.predict(input_data_scaled)
        if prediction[0] == 1:
            st.write('Heart Disease Present')
        else:
            st.write('Heart Disease Absent')


# Run the Streamlit app
if __name__ == '__main__':
    main()
