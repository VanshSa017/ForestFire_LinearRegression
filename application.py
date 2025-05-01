from flask import Flask,request,jsonify,render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application=Flask(__name__)
app=application

#import ridge rgressor and standard scalar pickle

ridge_model=pickle.load(open('models/ridge.pkl','rb'))
stand=pickle.load(open('models/scaler.pkl','rb')) 

@app.route('/')
def index():
    return render_template('index.html') ## it will find and load the html file to the home page

@app.route('/predict_data', methods=['GET', 'POST'])
def predict_data():
    if request.method == 'POST':
        try:
            Temperature = float(request.form.get('Temperature'))
            RH = float(request.form.get('RH'))
            Ws = float(request.form.get('Ws'))
            Rain = float(request.form.get('Rain'))
            FFMC = float(request.form.get('FFMC'))
            DMC = float(request.form.get('DMC'))
            ISI = float(request.form.get('ISI'))
            Classes = float(request.form.get('Classes'))
            Region = float(request.form.get('Region'))

            # Scale and predict
            new_scaled_data = stand.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
            result = ridge_model.predict(new_scaled_data)
            print(result)

            return render_template('home.html', result=round(result[0], 2))

        except Exception as e:
            print("Error during prediction:", e)
            return render_template('home.html', result="Error: Invalid input or internal error.")
    else:
        return render_template('home.html', result=None)

if __name__ == '__main__':
    app.run(host="0.0.0.0")