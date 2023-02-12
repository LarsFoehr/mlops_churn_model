"""Select the best possible model for production
    
- This script helps to select the best-performing model from the model registry.
- It saves the best model in the model directory. 
- The best model is selected using the model performance metric - accuracy.
"""

import joblib
import mlflow
import argparse
from pprint import pprint
from mlflow.tracking import MlflowClient
from ..data.load_data import read_params

def log_production_model(config_path:str) -> str:
    """_summary_

    Args:
        config_path (str): Path to params.yaml

    Returns:
        str: _description_
    """
    # Read in all parameters, i.e. paths from params.yaml
    config = read_params(config_path)
    
    # Set paths, which MlFlow needs to work
    mlflow_config = config["mlflow_config"] 
    model_name = mlflow_config["registered_model_name"]
    model_dir = config["model_dir"]
    remote_server_uri = mlflow_config["remote_server_uri"]

    # Search for all experiment which were run via MlFlow
    # Take the experiment and the model with the best performance metric - accuracy
    mlflow.set_tracking_uri(remote_server_uri)
    runs = mlflow.search_runs(experiment_ids=1)
    max_accuracy = max(runs["metrics.accuracy"])
    max_accuracy_run_id = list(runs[runs["metrics.accuracy"] == max_accuracy]["run_id"])[0]
    
    # Now build the MlFlow Client which is used, to bring the chosen model either into production or stage mode
    # The chosen model is put into production phase, if this model has already been staged and logged
    # Otherwise the model first has to be staged
    client = MlflowClient()
    for mv in client.search_model_versions(f"name='{model_name}'"):
        mv = dict(mv)

        if mv["run_id"] == max_accuracy_run_id:
            current_version = mv["version"]
            logged_model = mv["source"]
            pprint(mv, indent=4)
            client.transition_model_version_stage(
                name=model_name,
                version=current_version,
                stage="Production"
            )
        else:
            current_version = mv["version"]
            client.transition_model_version_stage(
                name=model_name,
                version=current_version,
                stage="Staging"
            )        

    # Finally load the logged model and save it
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    joblib.dump(loaded_model, model_dir)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    data = log_production_model(config_path=parsed_args.config)