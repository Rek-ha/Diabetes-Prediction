import numpy as np
import pickle
import streamlit as st

# Loading the saved model
loaded_model = pickle.load(open('trained model.sav', 'rb'))

# Creating a function for prediction
def diabetic_prediction(input_data):
    # Converting input data to float and handling potential errors
    try:
        input_data = [float(value) for value in input_data]
    except ValueError:
        return "Invalid input. Please provide numerical values for all fields."

    # Reshaping the array as we are predicting for one instance
    input_data_reshaped = np.array(input_data).reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    
    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic' 

# Building the Streamlit app
def main():
    st.title('Diabetic Prediction')
    st.write("Please enter the following information:")

    # Input fields
    pregnancies = st.text_input('No. of Pregnancies')
    glucose = st.text_input('Glucose Value')
    blood_pressure = st.text_input('Blood Pressure Range')
    skin_thickness = st.text_input('Skin Thickness')
    insulin = st.text_input('Insulin Range')
    bmi = st.text_input('BMI Value')
    diabetes_pedigree_function = st.text_input('Diabetes Pedigree Function Value')
    age = st.text_input('Age')

    # Prediction button
    if st.button('Diabetic Test Result'):
        diagnosis = diabetic_prediction([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age])
        st.success(diagnosis)

if __name__ == '__main__':
    main()








