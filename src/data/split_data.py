"""Data splitter

This script splits raw data and create new churn_train and churn_test inside the processed folder.

Three functions read in the following data:

    * split_data - Split dataframe and write train and test set to csv
    * load_data - Load csv dataset from given path
    * load_raw_data - Load train and test data from external location(data/external) to raw folder(data/raw)
"""

import argparse
import pandas as pd
from load_data import read_params
from sklearn.model_selection import train_test_split

def split_data(df:pd.DataFrame,
               train_data_path:str,
               test_data_path:str,
               split_ratio:float,
               random_state:int) -> None:
    """Split dataframe and write train and test set to csv.

    Args:
        df (pd.DataFrame): DataFrame to be splitted
        train_data_path (str): Path, to which train data will be written to
        test_data_path (str): Path, to which test data will be written to
        split_ratio (float): Ratio, which is used to split the data into train and test
        random_state (int): Random state, to make sure that results are reproducible

    Returns:
        None

    """
    train, test = train_test_split(df, test_size=split_ratio, random_state=random_state)
    train.to_csv(train_data_path, sep=",", index=False, encoding="utf-8")
    test.to_csv(test_data_path, sep=",", index=False, encoding="utf-8")    

def split_and_saved_data(config_path:str) -> None:
    """Split raw data into train and test set and save them both.

    Args:
        config_path (str): Path to object related parameters.

    Returns:
        None.

    """
    config = read_params(config_path)
    raw_data_path = config["raw_data_config"]["raw_data_csv"]
    test_data_path = config["processed_data_config"]["test_data_csv"] 
    train_data_path = config["processed_data_config"]["train_data_csv"]
    split_ratio = config["raw_data_config"]["train_test_split_ratio"]
    random_state = config["raw_data_config"]["random_state"]
    raw_df=pd.read_csv(raw_data_path)
    split_data(raw_df, train_data_path, test_data_path, split_ratio, random_state)
    
if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    split_and_saved_data(config_path=parsed_args.config)