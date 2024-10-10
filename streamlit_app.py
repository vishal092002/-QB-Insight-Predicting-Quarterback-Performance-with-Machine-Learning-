import streamlit as st
import numpy as np
import joblib

# Load the trained model and scaler
rf_model = joblib.load('rf_model.joblib')
scaler = joblib.load('scaler.joblib')

# Streamlit app
st.title('QB Rating Prediction')
st.write('Predict QB ratings based on performance metrics.')

# Input fields for each feature
pass_yds = st.number_input('Pass Yards', min_value=0)
yds_per_att = st.number_input('Yards Per Attempt', min_value=0.0)
att = st.number_input('Attempts', min_value=0)
cmp = st.number_input('Completions', min_value=0)
cmp_pct = st.number_input('Completion %', min_value=0.0)
td = st.number_input('Touchdowns', min_value=0)
int_ = st.number_input('Interceptions', min_value=0)
sck = st.number_input('Sacks', min_value=0)
first_pct = st.number_input('1st%', min_value=0.0)

# Make prediction using user input
if st.button('Predict'):
    user_input = np.array([[pass_yds, yds_per_att, att, cmp, cmp_pct, td, int_, sck, first_pct]])
    user_input_scaled = scaler.transform(user_input)
    prediction = rf_model.predict(user_input_scaled)
    st.write(f'Predicted QB Rating: {prediction[0]:.2f}')
