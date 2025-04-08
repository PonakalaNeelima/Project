import streamlit as st
import joblib
import pandas as pd
import numpy as np

# General possible ranges for parameters
general_ranges = {
    "ph": (0, 14),
    "hardness": (0, 1000),  # mg/L
    "turbidity": (0, 100),  # NTU
    "arsenic": (0, 1),      # mg/L
    "chloramine": (0, 10),  # mg/L
    "bacteria": (0, 1000),  # CFU/100 mL
    "lead": (0, 1),         # mg/L
    "nitrates": (0, 100),   # mg/L
    "mercury": (0, 1)       # mg/L
}

# Units for parameters
units = {
    "ph": "pH units",
    "hardness": "mg/L (as CaCO3)",
    "turbidity": "NTU",
    "arsenic": "mg/L",
    "chloramine": "mg/L",
    "bacteria": "CFU/100 mL",
    "lead": "mg/L",
    "nitrates": "mg/L",
    "mercury": "mg/L"
}


meta_model = joblib.load('Stacked_Model.joblib')  
model1=joblib.load('lr.joblib')
model2=joblib.load('svm.joblib')
model3=joblib.load('tree.joblib')
model4=joblib.load('forest.joblib')

# Function to validate inputs
def validate_inputs(inputs):
    validation_errors = []
    
    for param_name, value in inputs.items():
        if np.isnan(value):
            validation_errors.append(f"{param_name.capitalize()} contains NaN value.")
        else:
            lower, upper = general_ranges[param_name]
            if not (lower <= value <= upper):
                validation_errors.append(f"{param_name.capitalize()} is out of range ({lower} - {upper}).")
    
    return validation_errors


st.title("Safeguarding Public Health Through Water Purity Analysis")
st.write("Please enter values in the specified units:")

inputs = {}
for param_name, (lower, upper) in general_ranges.items():
    inputs[param_name] = st.number_input(
        f"Enter {param_name.capitalize()} ({units[param_name]}):", min_value=lower, max_value=upper
    )

# Predict button
if st.button("Predict"):
    errors = validate_inputs(inputs)
    
    if errors:
        for error in errors:
            st.error(error)  # Display error messages for invalid inputs
        st.stop()  # Stop execution if there are validation errors
    else:
        st.success("All inputs are valid!")
        
        # Create a DataFrame for the input data
        input_data = pd.DataFrame([list(inputs.values())], columns=list(inputs.keys()))
        
        # Make prediction
  
        predictions = [
            model1.predict(input_data)[0],
            model2.predict(input_data)[0],
            model3.predict(input_data)[0],
            model4.predict(input_data)[0]
        ]
        
        # Create a DataFrame for the predictions
        prediction_data = pd.DataFrame([predictions], columns=['Model1', 'Model2', 'Model3', 'Model4'])
        
        # Make final prediction using the meta model
        final_prediction = meta_model.predict(prediction_data)[0]
        
        # Display the final prediction
        result = "Safe" if final_prediction == 1 else "Not Safe"
        st.write(f"The water is predicted to be: **{result}**")
