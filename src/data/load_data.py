"""Data loader

This script loads in all necessary data.

Three functions read in the following data:

    * read_params - Read in project related parameters
    * load_data - Load csv dataset from given path
    * load_raw_data - Load train and test data from external location(data/external) to raw folder(data/raw)
"""

import yaml
import argparse
import pandas as pd 

def read_params(config_path:str) -> dict:
    """Read in project related parameters from params.yaml file.

    Args:
        config_path (str): Path to params.yaml.

    Returns:
        Project related parameters as dictionary.

    """
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def load_data(data_path:str, model_var:str) -> pd.DataFrame:
    """Load csv dataset from given path.

    Args:
        data_path (str): Path to csv file.
        model_var (str): One or more variables to be used, to filter read in csv file.

    Returns:
        Project related parameters as dictionary.

    """
    df = pd.read_csv(data_path, sep=",", encoding="utf-8")
    df = df[model_var]
    return df

def load_raw_data(config_path):
    """Read in csv data and write it as raw data to corresponding folder.
    
    This function contains three steps:
    - Read in yaml file of object related parameters
    - Load external data with information from parameters yaml
    - Write filtered data to raw data path

    Args:
        config_path (str): Path to object related parameters
        model_var (str): One or more variables to be used, to filter read in csv file

    Returns:
        None

    """
    config = read_params(config_path)
    external_data_path = config["external_data_config"]["external_data_csv"]
    raw_data_path = config["raw_data_config"]["raw_data_csv"]
    model_var = config["raw_data_config"]["model_var"]
    
    df = load_data(external_data_path, model_var)
    df.to_csv(raw_data_path, index=False)
    
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    load_raw_data(config_path=parsed_args.config)