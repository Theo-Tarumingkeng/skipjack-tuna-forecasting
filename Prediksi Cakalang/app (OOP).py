import numpy as np
import pickle
import joblib

from flask import Flask, request, render_template
from keras.models import load_model
from numpy import concatenate

# Create flask app
flask_app = Flask(__name__)

# Load model
linreg = pickle.load(open("LinReg model.pkl", "rb"))
rnn = load_model("RNN model.h5")

# Load scaler
#scaler = joblib.load("scaler.joblib")
scaler = pickle.load(open("scaler.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

def reshape_input(input):
    # Reshape the input data
    linreg_input_reshaped = input.reshape(1, -1).astype(float)
    rnn_input_reshaped = input.reshape(1, 1, -1).astype(float)

    return linreg_input_reshaped, rnn_input_reshaped

def inverse_transform(prediction, input, scaler):
    # Concat the prediction result with features inputted
    features = concatenate((prediction, input), axis=1)

    # Inverse transform the prediction results + features
    features_inversed = scaler.inverse_transform(features)

    # Take the prediction result only
    rnn_prediction_inversed = features_inversed[:, 0]
    return rnn_prediction_inversed

@flask_app.route("/predict", methods = ["POST"])
def predict():
    # Acquiring the input data
    input = np.array([float(x) for x in request.form.getlist('variabel_input')]).reshape(1, -1)

    # Reshape the input data into shape expected by linreg & RNN model
    linreg_input_reshaped, rnn_input_reshaped = reshape_input(input)

    # Predict the data from inputted data
    linreg_prediction = linreg.predict(linreg_input_reshaped)
    rnn_prediction = rnn.predict(rnn_input_reshaped)

    # Inverse Transform for the RNN predictions
    rnn_prediction = inverse_transform(rnn_prediction, input, scaler)

    # Ensure the prediction is displayed as an absolute number
    linreg_prediction = np.abs(linreg_prediction)
    rnn_prediction = np.abs(rnn_prediction)

    # Format the number into 2 decimal numbers
    linreg_format = [f"{float(x):.0f}" for x in linreg_prediction]
    rnn_format = [f"{float(x):.0f}" for x in rnn_prediction]

    # Join the formatted predictions into a single string
    linreg_text = " ".join(linreg_format)
    rnn_text = " ".join(rnn_format)

    return render_template("index.html",
                           prediction_text="{} kg dari Regresi Linear dan {} kg dari RNN"
                           .format(linreg_text, rnn_text))

    #return linreg_text, rnn_text

if __name__ == "__main__":
    flask_app.run(debug=True)