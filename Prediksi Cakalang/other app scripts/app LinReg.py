import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import tensorflow as tf

# Create flask app
flask_app = Flask(__name__)

# Loading model
linreg = pickle.load(open("../LinReg model.pkl", "rb"))
rnn_model_path = "model_rnn.h5"
rnn = tf.keras.models.load_model(rnn_model_path)

@flask_app.route("/")
def Home():
    return render_template("index3.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.getlist('variabel_input')]
    features = np.array(float_features)
    features_reshaped = features.reshape(1, -1)
    print(features_reshaped.shape)

    prediction = linreg.predict(features_reshaped)

    formatted_prediction = [f"{x:.2f}" for x in prediction]
    return render_template("index3.html", prediction_text="Perkiraan Hasil Tangkapan Cakalang {} kg".format(formatted_prediction))

if __name__ == "__main__":
    flask_app.run(debug=True)