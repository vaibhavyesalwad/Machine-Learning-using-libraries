from flask import Flask, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

def data_preprocessing(data):
    # data pre-processing as before model building for accurate prediction
    x_values = data.loc[:, ['temp', 'hum', 'windspeed', 'yr', 'workingday']].values
    y_values = data.loc[:, 'cnt'].values

    # opening pickle file & loading objects from pickle file
    file = open(
        '/home/admin1/PycharmProjects/Machine Learning using libraries/Regression/Polynomial Regression/PolyRegModel2.pkl','rb')
    one_hot_encode = joblib.load(file)
    poly = joblib.load(file)
    regressor = joblib.load(file)
    file.close()

    # One-hot-encoding for categorical as before model building
    categorical_cols = ['season', 'mnth', 'hr', 'weekday', 'weathersit']
    new_columns = one_hot_encode.transform(data.loc[:, categorical_cols]).toarray()
    x_values = np.append(x_values, new_columns, axis=1)
    x_values = poly.fit_transform(x_values)

    return x_values, y_values, regressor



@app.route("/", methods=["POST"])
def poly_reg_prediction():
    # requesting for data in json format
    data_fr_predict = request.json

    data_fr_predict = pd.DataFrame(data_fr_predict, index=[0])

    x_values, y_values, regressor = data_preprocessing(data_fr_predict)
    prediction = regressor.predict(x_values)

    result={}
    result['Prediction']=prediction.tolist()[0]
    result['Actual value']=y_values.tolist()[0]
    return result

if __name__ == "__main__":
    app.run(debug=True)
