import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('model.pkl')

# Define selected features
selected_features = ['Age', 'WaitingDays', 'Hypertension', 'Scholarship', 'Diabetes', 'SMS_received']

# Streamlit app
st.title('Medical Appointment No-Show Prediction')

st.write('Enter patient details to predict if they will miss their appointment.')

# Input fields
age = st.number_input('Age', min_value=0, max_value=100, value=30)
waiting_days = st.number_input('Waiting Days (days between scheduling and appointment)', min_value=0, value=7)
hypertension = st.selectbox('Hypertension', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
scholarship = st.selectbox('Scholarship (Social Welfare Program)', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
diabetes = st.selectbox('Diabetes', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
sms_received = st.selectbox('SMS Received', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

# Predict button
if st.button('Predict'):
    # Create input DataFrame
    input_data = pd.DataFrame([[age, waiting_days, hypertension, scholarship, diabetes, sms_received]], 
                              columns=selected_features)
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]  # Probability of no-show
    
    # Display result
    st.write('**Prediction**: ', 'No-show' if prediction == 1 else 'Show')
    st.write('**No-show Probability**: ', f'{probability:.2%}')

# Instructions
st.write("""
### Instructions
- Enter the patient's details using the fields above.
- Click 'Predict' to see if the patient is likely to miss their appointment.
- The model uses Age, Waiting Days, Hypertension, Scholarship, Diabetes, and SMS Received to make predictions.

### Check Cases
- Sample: (Age=80, WaitingDays=15, Hypertension=0, Scholarship=0, Diabetes=0, SMS_received=0)
Output: Show, Probability: 47.00%

- Sample: (Age=18, WaitingDays=50, Hypertension=0, Scholarship=1, Diabetes=0, SMS_received=0)
Output: No-show, Probability: 75.20%
""")