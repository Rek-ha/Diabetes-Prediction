import numpy as np
import pickle
import streamlit as st


#loading the saved model
loaded_model = pickle.load(open('C:/Users/ADMIN/Downloads/diabetic/Deployment/trained model.sav', 'rb'))

#creating a function for prediction
def diabetic_prediction(input_data):

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic' 


#building the stream lit

def main():
    #giving the title
    st.title('Diabetic Prediction')
    #creating the input data options--gettiing the input data from the user
    #Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age
    Pregnancies=st.text_input(' No of pregnancies')
    Glucose=st.text_input('Glucose value')
    BloodPressure=st.text_input('BloodPressure range')
    SkinThickness=st.text_input('SkinThickness')
    Insulin=st.text_input('Insulin range')
    BMI=st.text_input('Bmi value')
    DiabetesPedigreeFunction=st.text_input('DiabetesPedigreeFunction value')
    Age=st.text_input('Age')

    #code for prediction
    diagnosis=''

    #creating the button
    if st.button('Diabetic test result'):
        diagnosis=diabetic_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])

    
    st.success(diagnosis)



if __name__=='__main__':
    main()









