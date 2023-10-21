import os
import json
import pandas as pd
import numpy as np
import pandas as pd

# from materializer.custom_materializer import cs_materializer
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.evaluation import evaluate_model
from steps.model_train import train_model


from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW, TENSORFLOW

from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)

#from .utils import get_data_for_test
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters, Output

docker_settings = DockerSettings(required_integrations=[MLFLOW])

class DeploymentTriggerConfig(BaseParameters):
    """ Deployment trigger config"""
    min_accuracy = 0.92
    
@step
def deployment_trigger(
    accuracy: float,
    config: DeploymentTriggerConfig
):
    """
    Implements a simple model deployment trigger that looks at the input model accuracy
    """
    
    return accuracy >= config.min_accuracy




@pipeline(enable_cache= True, settings= {"docker": docker_settings})
def continuous_deployment_pipeline(
    data_path: str,
    min_accuracy: float = 0.92,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT):

    df = ingest_data()
    X_train,X_test,Y_test,Y_train, = clean_data(df)
    model = train_model(X_train,Y_train)
    r2_score,rmse = evaluate_model(model,X_test,Y_test)
    deployment_decision = deployment_trigger(r2_score)
    mlflow_model_deployer_step(
        model = model,
        deploy_decision = deployment_decision,
        workers = workers,
        timeout = timeout
    )