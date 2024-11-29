from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():

    data = request.get_json()

    year = data['year']
    month = data['month']

    str = f"{year}-{month}-01"

    dataset = pd.read_csv('app_data.csv')

    dataset.set_index('Date', inplace=True)
    
    prediction = model.predict(dataset.loc[dataset.index == str])
    
    # Return the prediction in JSON format
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)