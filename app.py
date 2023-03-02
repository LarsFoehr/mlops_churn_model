"""Build a flask API.

This script builds a Flask API / App, which can be used by users, to make predictions.

The following scripts are used, to build the API:
    * All scripts in /webapp folder

The following functions and classes are written and used:

    * NotANumber - A class giving feedback, if the entered value is not a number.
    * predict - Takes input data and the latest best model and predicts the outcome.
    * validate_input - Validates the user given input by using the class NotANumber.
    * form_response - Builds the response given over the UI form in the App.
 
"""
from flask import Flask, render_template, request, jsonify
import joblib
import os
import numpy as np
import pandas as pd
import yaml
 
from src.data.load_data import read_params

# Set paths to required html and css scripts
webapp_root = "webapp"
static_dir = os.path.join(webapp_root, "static")
template_dir = os.path.join(webapp_root, "templates")

params_path = "params.yaml"

# Initiate the App
app = Flask(__name__, 
            static_folder = static_dir,
            template_folder = template_dir)

class  NotANumber(Exception):
    """Give feedback if the user given input is not numerical.

    Args:
        Exception (str): The Exception type.
    """
    def __init__(self, message="Values entered are not Numerical"):
        self.message = message
        super().__init__(self.message)

def predict(data: pd.DataFrame) -> pd.Series:
    """Take input data and the latest model to make predictions.

    Args:
        data (pd.DataFrame): _description_

    Returns:
        pd.Series: Predictions.
    """
    config = read_params(params_path)
    model_dir_path = config["model_webapp_dir"]
    model = joblib.load(model_dir_path)
    prediction = model.predict(data).tolist()[0]
    return prediction 

def validate_input(dict_request: dict) -> bool:
    """Take in the user given input and validate it.

    Args:
        dict_request (dict): User given input.

    Raises:
        NotANumber: Class which gives feedback to the user, if given value is not numerical.

    Returns:
        bool: True
    """
    for _, val in dict_request.items():
        try:
            val=float(val)
        except Exception as e:
            raise NotANumber
    return True

def form_response(dict_request: dict) -> str:
    """Build the response given over the UI form in the App.

    Args:
        dict_request (dict): User given input.

    Returns:
        str: Response the user can see.
    """
    try:
        if validate_input(dict_request):
            data = dict_request.values()
            data = [list(map(float, data))]
            response = predict(data)
            return response
    except NotANumber as e:
        response =  str(e)
        return response 

# Route the app to its URL using GET and POST requests.
@app.route("/", methods=["GET", "POST"])

def index():
    """Build the frontend, but use exceptions, if something goes wrong.

    Returns:
        html: Rendered html index template.
    """
    if request.method == "POST":
        try:
            if request.form:
                dict_req = dict(request.form)
                response = form_response(dict_req)
                return render_template("index.html", response=response)
        except Exception as e:
            print(e)
            error = {"error": "Something went wrong!! Try again later!"}
            error = {"error": e}
            return render_template("404.html", error=error)
    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)