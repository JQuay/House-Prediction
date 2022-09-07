from contextlib import redirect_stderr
from distutils.log import debug
from pyexpat import model
import pickle
from flask import Flask, request ,jsonify,url_for,render_template
# import sklearn 
import numpy as np
import pandas as pd

# Creating the app object
app =Flask(__name__)

#loading the model
regmodel = pickle.load(open('regmodel.pkl','rb'))
scaler = pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')




# Creating Predict API
@app.route('/predict_api', methods = ['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    new_data = scaler.transform(np.array(list(data.values())).reshape(1,-1))
    print(np.array(list(data.values())).reshape(1,-1))
    output=regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/prediction',methods=['POST','GET'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scaler.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output= regmodel.predict(final_input)[0]
    
    return render_template('predict.html', prediction = "The House price is {}".format(output))



@app.route('/dasboard',methods=['GET'])
def dashboard():
    return render_template('dashboard.html')
    
if __name__ == '__main__':
    app.run(debug=True)