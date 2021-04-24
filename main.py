from pywebio.input import *
from pywebio.output import *
import pickle
import numpy as np
# age = input("How old are you?", type=NUMBER)
# password=input("Enter password",type=PASSWORD)

# Load the Logistic Regression model
classifier = pickle.load(open('finalmodel.pkl', 'rb'))


data = input_group("Diabetes Prediction",[
  input('Number of pregnency', placeholder="e.g. 0", name='pregnancy',type=NUMBER),
  input('Glucose(mg/dL)', name='glucose', placeholder="e.g. 120",type=NUMBER),
  input('Blood Pressure(mm Hg)', name='bp', placeholder="e.g. 80",type=NUMBER),
  input('Insulin(IU/mL)', name='insulin', placeholder="e.g. 79",type=NUMBER),
  input('Skin Thickness(mm)', name='skin', placeholder="e.g. 20",type=NUMBER),
  input('Body Mass Index(kg/m2)', name='bmi', placeholder="e.g. 19.5",type=NUMBER),
  input('Diabetes Prediction Function', name='dpf', placeholder="e.g. 0.6", type=NUMBER),
  input('Age', name='age', placeholder="e.g. 0", type=NUMBER),
])

preg = data['pregnancy']
glucose = data['glucose']
bp = data['bp']
st = data['skin']
insulin = data['insulin']
bmi = data['bmi']
dpf = data['dpf']
age = data['age']

fetch_data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
my_prediction = classifier.predict(fetch_data)
if my_prediction == 1:
    result = "Great! You DON'T have diabetes."
    put_success(result)
    put_html('<img src="https://media0.giphy.com/media/TGQyLdzYv90FjplXkr/giphy.gif">')
if my_prediction == 0:
    result = "Oops! You Have Diabetes"
    put_warning(result)
    put_html('<img src="https://media4.giphy.com/media/1k1ytTA4AHJnp7OvUJ/giphy.gif">')

