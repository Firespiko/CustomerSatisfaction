from zenml.steps import BaseParameters


class ModelNameconfig(BaseParameters):
    """Model Configs"""
    model_name :str = "Linear Regression"