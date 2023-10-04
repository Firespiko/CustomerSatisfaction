import logging

import pandas as pd
from zenml import step

from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .Config import ModelNameconfig

@step
def train_model(
X_train: pd.DataFrame,
Y_train: pd.DataFrame,Config: ModelNameconfig) -> RegressorMixin:
    """_summary_
    Trains the model on the ingested data

    Args:
        df (pd.DataFrame): Dataframe of the data
    """
    
    try:
        if Config.model_name == "Linear Regression":
            model = LinearRegressionModel()
            trained_model = model.train(X_train,Y_train)
            return trained_model
        else:
            raise ValueError("Model {} not supported".format(Config.model_name))
    
    except Exception as e:
        logging.error("Error in training model {}".format(Config.model_name))
        raise e        
    