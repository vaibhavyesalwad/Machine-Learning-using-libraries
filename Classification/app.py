from flask import Flask, request
import os
import pandas as pd
import joblib
import sys

sys.path.append('/home/admin1/PycharmProjects/Machine Learning using libraries')
from ipynb.fs.full.ml_library import *

os.chdir('/home/admin1/PycharmProjects/Machine Learning using libraries/Classification/Datasets & pickled objects')

app = Flask(__name__)


def data_processing_adclick():
    # requesting for data in json format (it is stored as python dictionary)
    data_fr_predict = request.json
    # creating pandas dataframe from python dictionary
    data = pd.DataFrame(data_fr_predict, index=[0])

    # loading pickled file for preprocessing of adclick dataset
    file = open('DataProcessingAdClick.pkl', 'rb')
    features = joblib.load(file)
    label = joblib.load(file)
    sc_x = joblib.load(file)
    file.close()

    # using objects created during data pre-processing prior to model building
    x_values = data.loc[:, features].values
    x_values = sc_x.transform(x_values)
    y_values = data.loc[:, label].values

    # returning feature matrix & label array
    return x_values, y_values


@app.route("/classification/adclick/<name>", methods=["POST"])
def classify_adclick(name):
    # identifying type of classier & using it in getting classifier object pickle file name
    classifier_pickle_file = name + 'ModelAdClick.pkl'

    # getting feature matrix & label for AdClick dataset
    x_values, y_values = data_processing_adclick()

    # loading pickled object of random  forest classifier model for AdClick dataset
    file = open(classifier_pickle_file, 'rb')
    classifier = joblib.load(file)
    file.close()

    # predicting for a given observation
    prediction = classifier.predict(x_values)

    # returning result as python dictionary
    result = {'Prediction': prediction.tolist()[0], 'Actual value': y_values.tolist()[0]}
    return result


def data_processing_hiv(name):
    # requesting for data in json format (it is stored as python dictionary)
    data_fr_predict = request.json
    # creating pandas dataframe from python dictionary
    data = pd.DataFrame(data_fr_predict, index=[0])

    if name == 'GaussianNB':
        file = open('DataProcessingGaussianNBHIV.pkl', 'rb')
        feature = joblib.load(file)
        label = joblib.load(file)
        lbl_encode = joblib.load(file)
        file.close()

        # separating string of 8 characters into 8 different features
        x_values = separate_feature_column(data, feature)
        # label encoder works on 1D array only
        x_values = x_values.flatten()
        # using encoding same as encoding used while data pre-processing prior to model building
        x_values = lbl_encode.transform(x_values)

        # changing values to interger numpy int32 datatypes & reshaping feature matrix to 2D array for classifier
        x_values = x_values.astype('int32').reshape(-1, 1)

    else:
        # loading pickled object of random  forest classifier model for HIV dataset
        file = open('DataProcessingHIV.pkl', 'rb')
        feature = joblib.load(file)
        label = joblib.load(file)
        one_hot_encode = joblib.load(file)
        file.close()

        # separating string of 8 characters into 8 different features
        x_values = separate_feature_column(data, feature)

        # using encoding same as encoding used while data pre-processing prior to model building
        x_values = one_hot_encode.transform(x_values)

    y_values = data[label].values

    return x_values, y_values


@app.route("/classification/HIV/<name>", methods=["POST"])
def classify_hiv(name):
    # getting feature matrix & label for HIV dataset
    x_values, y_values = data_processing_hiv(name)

    # identifying type of classier & using it in getting classifier object pickle file name
    classifier_pickle_file = name + 'ModelHIV.pkl'

    # loading pickled object of random  forest classifier model for AdClick dataset
    file = open(classifier_pickle_file, 'rb')
    classifier = joblib.load(file)
    file.close()

    # predicting for a given observation
    prediction = classifier.predict(x_values)

    # returning result as python dictionary
    result = {'Prediction': prediction.tolist()[0], 'Actual value': y_values.tolist()[0]}
    return result


if __name__ == "__main__":
    app.run(debug=True)
