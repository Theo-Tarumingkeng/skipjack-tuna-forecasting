import numpy as np
import pickle
import joblib

from flask import Flask, request, render_template
from keras.models import load_model
from numpy import concatenate

# Create flask app
flask_app = Flask(__name__)

# Load model
linreg = pickle.load(open("../LinReg model.pkl", "rb"))
rnn = load_model("../RNN model.h5")

# Load scaler
scaler = joblib.load("../scaler.joblib")

@flask_app.route("/")
def Home():
    return render_template("index3.html")

def inverse_transform(prediction, input, scaler):

    # Concat the prediction result with features inputted
    features = concatenate((prediction, input), axis=1)

    # Inverse transform the prediction resuls + features
    features_inversed = scaler.inverse_transform(features)

    # Take the prediction result only
    rnn_prediction = features_inversed[:, 0]
    return rnn_prediction

@flask_app.route("/predict", methods = ["POST"])
def predict():

    # Acquiring the input data
    input = np.array([float(x) for x in request.form.getlist('variabel_input')]).reshape(1, -1)

    # Reshape the input data
    linreg_input_reshaped = input.reshape(1, -1).astype(float)
    rnn_input_reshaped = input.reshape(1, 1, -1).astype(float)

    # Predict the data from inputted data
    linreg_prediction = linreg.predict(linreg_input_reshaped)
    rnn_prediction = rnn.predict(rnn_input_reshaped)

    # Inverse Transform for the RNN predictions
    rnn_prediction = inverse_transform(rnn_prediction, input, scaler)

    # Ensure the prediction is displayed as a positive number
    linreg_prediction = np.abs(linreg_prediction)
    rnn_prediction = np.abs(rnn_prediction)

    # Format the number into 2 decimal numbers
    linreg_format = [f"{float(x):.2f}" for x in linreg_prediction]
    rnn_format = [f"{float(x):.2f}" for x in rnn_prediction]

    return render_template("index3.html",
                           prediction_text="Perkiraan Hasil Tangkapan Cakalang {} kg untuk Regresi Linear dan {} kg untuk RNN".format(linreg_format, rnn_format))

if __name__ == "__main__":
    flask_app.run(debug=True)