"""Train the model

The following information is relevant:
    * Model specific information can be found in params.yaml file
    * These parameters can be changed to experiment
    * MlFlow tracks model performance using a specific dashboard

The following functions are written and used:

    * accuracymeasures - Read in project related parameters
    * load_data - Load csv dataset from given path
    * load_raw_data - Load train and test data from external location(data/external) to raw folder(data/raw)
"""

from src.data.load_data import read_params
import json
import yaml
import joblib
import mlflow
import argparse
import numpy as np
import pandas as pd
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score, confusion_matrix, classification_report

def accuracymeasures(y_test:pd.DataFrame, predictions:pd.Series, avg_method:str) -> tuple:
    """This function calculates different performance measures for the model(s).

    Args:
        y_test (pd.DataFrame): Data frame containing test data
        predictiopredictions (pd.Series): Series containing the predictions
        avg_method (str): Sets, which average method should be used to calculate the measures

    Returns:
        tuple containing performance measures accuracy, precision, recall, f1score

    """
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average=avg_method)
    recall = recall_score(y_test, predictions, average=avg_method)
    f1score = f1_score(y_test, predictions, average=avg_method)
    target_names = ['0', '1']
    
    # TODO: Replace print statements with logging feature
    
    print("Classification report")
    print("---------------------","\n")
    print(classification_report(y_test, predictions, target_names=target_names), "\n")
    print("Confusion Matrix")
    print("---------------------","\n")
    print(confusion_matrix(y_test, predictions),"\n")

    print("Accuracy Measures")
    print("---------------------","\n")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1score)

    return accuracy, precision, recall, f1score

def get_feat_and_target(df:pd.DataFrame, target:str) -> :
    """This function separates the feature dataframe and the target variable.

    Args:
        df (pd.DataFrame): DataFrame which will be separated
        target (str): Name of the targed variable
        
    Returns:
        Tuple containing feature dataframe and the target variable

    """
    x = df.drop(target, axis=1)
    y = df[[target]]
    return x, y    

def train_and_evaluate(config_path:str):
    """This function executes training of the data and the evaluation using MLFlow.

    Args:
        config_path (str): Path to params.yaml
        
    Returns:
        

    """
    # Read in all necessary paths from params.yaml, e.g. to train and test data
    config = read_params(config_path)
    train_data_path = config["processed_data_config"]["train_data_csv"]
    test_data_path = config["processed_data_config"]["test_data_csv"]
    target = config["raw_data_config"]["target"]
    max_depth=config["random_forest"]["max_depth"]
    n_estimators=config["random_forest"]["n_estimators"]

    # Read in the corresponding data
    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")
    train_x,train_y=get_feat_and_target(train,target)
    test_x,test_y=get_feat_and_target(test,target)

    # Get paths, which are used to execute MLFlow
    mlflow_config = config["mlflow_config"]
    remote_server_uri = mlflow_config["remote_server_uri"]

    # Set MLFlow tracking uri and MLFlow experiment
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(mlflow_config["experiment_name"])

    # Start MLFlow run
    with mlflow.start_run(run_name=mlflow_config["run_name"]) as mlops_run:
        
        # First fit and train the model and then make predictions
        model = RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators)
        model.fit(train_x, train_y)
        y_pred = model.predict(test_x)
        accuracy,precision,recall,f1score = accuracymeasures(test_y,y_pred,'weighted')

        # Log important model parameter metrics
        mlflow.log_param("max_depth",max_depth)
        mlflow.log_param("n_estimators", n_estimators)

        # Log performance measures
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1score)
       
       # Get path to MLFlow type store
        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme

        # Depending on tracking_url_type_store is not "file":
        # Log the new model, because it doesn't exist already
        # Otherwise load the latest model
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                model, 
                "model", 
                registered_model_name=mlflow_config["registered_model_name"])
        else:
            mlflow.sklearn.load_model(model, "model")
 
if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)