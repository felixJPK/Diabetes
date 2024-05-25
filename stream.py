import numpy as np
import pickle
import streamlit as st
page_bg_img = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Aptos&display=swap');
[data-testid="stAppViewContainer"]{
background-color: #E4F1EE;
# linear-gradient(to left, , white);
}
h1{
    text-align: center;
    color:#153a6d;
    font-family: 'Aptos', sans-serif;  
}

h2{
    color: #F0F0F0;
    font-family: 'Aptos', sans-serif; 
}
h3{
    color: white;
    font-family: 'Aptos', sans-serif; 
}
h4{
    color: white;
    font-family: 'Aptos', sans-serif; 
}
p{
    color:#10244f;
    # font-size: 50px;
    font-family: 'Aptos', sans-serif; 
}
input {
    background-color: white;
    color: #white;
    border: 1px solid #4E65FF;
    border-radius: 5px;
    padding: 10px;
}

textarea {
    background-color: black;
    color: #000000;
    border: 1px solid #4E65FF;
    border-radius: 5px;
    padding: 10px;
}
.st-bc.st-bx.st-by.st-bz.st-c0.st-c1.st-c2.st-c3.st-c4.st-c5.st-c6.st-b9.st-c7.st-c8.st-c9.st-ca.st-cb.st-cc.st-cd.st-ce.st-ae.st-af.st-ag.st-cf.st-ai.st-aj.st-bw.st-cg.st-ch.st-ci {
    # background-color: #D9EDF8;
    background-color: white ;
    color: 	black; 
    border-color: white; 
}

.stButton button {
    background-color: #4FC0D0;
    color: white;
    border-radius: 5px;
    padding: 10px 20px;
    border: none;
    cursor: pointer;
}

.stButton button:hover {
    # background-color: #3B50CC;
    background-color: #1B6B93;
    color:white !important;
}
# .st-au {
#     background-color: blac;
# }
</style>

"""
st.markdown(page_bg_img, unsafe_allow_html=True)
#loading the saved model
loaded_model = pickle.load(open('D:/document/Binus/Semester4/ML/Assessment/latdeploy4usingstreamlit/trained_model.sav', 'rb'))

#creating a function for prediction
def diabetes_prediction(input_data):

    
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        st.markdown('<style> .st-au {background-color: #32CD32} </style>', unsafe_allow_html=True)
        return 'The person is not diabetic'

    else:
        st.markdown('<style> .st-au {background-color: #FF4500;} </style>', unsafe_allow_html=True)
        return 'The person is diabetic'
    
def main():
    #giving a title 
    st.title('Diabetes Prediction Web App')
    
    #getting the input data from the user 
    
    Pregnancies = st.text_input("Number of Pregnancies")
    Glucose = st.text_input("Glucose Level")
    BloodPressure = st.text_input("Blood Pressure Value")
    SkinThickness = st.text_input("Skin Thickness value")
    Insulin = st.text_input("Insulin Level")
    BMI = st.text_input("BMI value")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function value")
    Age = st.text_input("Age of the Person")
    
    #code for prediction 
    diagnosis = ''
    
    #Creating a button for prediction
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])

    st.success(diagnosis)
    
if __name__ == '__main__':
    main()
    
    
