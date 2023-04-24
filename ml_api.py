from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json
import pandas as pd


app = FastAPI()

class model_input(BaseModel):
    age : int
    sex : int
    chest_pain_type : float
    resting_blood_pressure : float
    cholesterol : float
    fasting_blood_sugar : float
    rest_ecg : float
    max_heart_rate_achieved : float
    exercise_induced_angina : float
    st_depression: float
    st_slope : float
    num_major_vessels : float
    thalassemia : float

#loading the saved model
def load_model():
    model = pickle.load(open('rf_model.sav','rb'))
    return model

def data_scaling(df):

    cont_data = df.values
    m = open('scaled_model.pkl', 'rb')
    model = pickle.load(m)
    cont_data = model.transform(cont_data)
    return cont_data

@app.get("/")
def hello():
    return "Welcome to heart disease prediction api"


@app.post("/predict")
def heart_predict(input_parameters : model_input):
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    age = input_dictionary["age"]
    sex = input_dictionary["sex"]
    chest_pain_type = input_dictionary["chest_pain_type"]
    resting_blood_pressure = input_dictionary["resting_blood_pressure"]
    cholesterol = input_dictionary["cholesterol"]
    fasting_blood_sugar = input_dictionary["fasting_blood_sugar"]
    rest_ecg = input_dictionary["rest_ecg"]
    max_heart_rate_achieved = input_dictionary["max_heart_rate_achieved"]
    exercise_induced_angina = input_dictionary["exercise_induced_angina"]
    st_depression = input_dictionary["st_depression"]
    st_slope = input_dictionary["st_slope"]
    num_major_vessels = input_dictionary["num_major_vessels"]
    thalassemia = input_dictionary["thalassemia"]
    data = ([[age, sex, chest_pain_type, resting_blood_pressure,cholesterol,fasting_blood_sugar,rest_ecg,max_heart_rate_achieved,exercise_induced_angina,st_depression,st_slope, num_major_vessels,thalassemia]])
    #  data = pd.DataFrame(data,columns = col)
    # data = data_scaling(data)
    model = load_model()
    prediction = model.predict(data)
    if prediction == 0:
        object = {"message" : "no"}
        return object 
    else:
        object = {"message" : "yes"}
        return object 

    
