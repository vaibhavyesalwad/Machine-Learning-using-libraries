from flask import Flask, request
import os
import pandas as pd
import numpy as np
import joblib
import sys

sys.path.append('/home/admin1/PycharmProjects/Machine Learning using libraries')
from ipynb.fs.full.ml_library import *

os.chdir('/home/admin1/PycharmProjects/Machine Learning using libraries/Classification/Datasets & pickled objects')

app = Flask(__name__)


def data_processing():
    # requesting for data in json format (it is stored as python dictionary)
    data_fr_predict = request.json
    # creating pandas data frame from python dictionary
    data = pd.DataFrame(data_fr_predict, index=[0])

    with open('DataProcessingBank.pkl', 'rb') as file:
        features = joblib.load(file)
        label = joblib.load(file)
        oh_enc_column = joblib.load(file)
        lbl_enc_x = joblib.load(file)
        oh_enc = joblib.load(file)
        sc_x = joblib.load(file)

    x_values = data.loc[:, features].values
    new_x_values = data.loc[:, oh_enc_column].values
    y_values = data.loc[:, label].values

    x_values[:, 1] = lbl_enc_x.transform(x_values[:, 1])
    new_x_values = oh_enc.transform(new_x_values.reshape(-1, 1)).todense()
    x_values = np.append(x_values, new_x_values, 1)
    x_values = sc_x.transform(x_values)

    return x_values, y_values


@app.route("/ANN/Bank", methods=["POST"])
def predict_cust_exit():
    # getting feature matrix & label for Restaurant Review data set
    x_values, y_values = data_processing()

    # loading pickled object of random  forest classifier model Restaurant Review data set
    file = open('ANNModelBank.pkl', 'rb')
    classifier = joblib.load(file)
    file.close()

    # predicting for a given observation
    prediction = classifier.predict(x_values)
    prediction = np.where(prediction >=0.5, 1, 0)

    # returning result as python dictionary
    result = {'Prediction': prediction.tolist()[0][0], 'Actual value': y_values.tolist()[0]}
    return result


if __name__ == "__main__":
    app.run(debug=True)
