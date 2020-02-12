from flask import Flask, request
import pandas as pd
import joblib
import os
import sys
os.chdir('/home/admin1/PycharmProjects/Machine Learning using libraries/Classification/Datasets & pickled objects')
sys.path.append('/home/admin1/PycharmProjects/Machine Learning using libraries')
from ipynb.fs.full.ml_library import *

app = Flask(__name__)


def data_processing():
    # requesting for data in json format (it is stored as python dictionary)
    data_fr_predict = request.json
    # creating pandas data frame from python dictionary
    data = pd.DataFrame(data_fr_predict, index=[0])

    # loading pickled file for pre processing of Restaurant Review data set
    file = open('DataProcessingNLPRestaurantReview.pkl', 'rb')
    feature = joblib.load(file)
    label = joblib.load(file)
    cv = joblib.load(file)
    file.close()

    # removing stopwords & getting root words for remaining
    corpus = get_corpus(data, feature)
    x_values = cv.transform(corpus).toarray()
    y_values = data[label].values

    return x_values, y_values


@app.route("/NLP/RFRestaurantReview", methods=["POST"])
def predict_review():
    # getting feature matrix & label for Restaurant Review data set
    x_values, y_values = data_processing()

    # loading pickled object of random  forest classifier model Restaurant Review data set
    file = open('SVMModelNLPRestaurantReview.pkl', 'rb')
    classifier = joblib.load(file)
    file.close()

    # predicting for a given observation
    prediction = classifier.predict(x_values)

    # returning result as python dictionary
    result = {'Prediction': prediction.tolist()[0], 'Actual value': y_values.tolist()[0]}
    return result


if __name__ == "__main__":
    app.run(debug=True)