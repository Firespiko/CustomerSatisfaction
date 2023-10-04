import logging
import pandas as pd
from zenml import step
from typing import Tuple
from typing_extensions import Annotated

from src.evaluation import R2,RMSE,MSE
from sklearn.base import RegressorMixin

@step
def evaluate_model(model: RegressorMixin,
X_test: pd.DataFrame,
Y_test: pd.DataFrame) -> Tuple[Annotated[float, "r2_score"], Annotated[float, "rmse"]]:
    
    """_summary_
        
        
    """
    
    try:
        prediction = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.calculate_scores(Y_test,prediction)
        
        r2_class = R2()
        r2 = r2_class.calculate_scores(Y_test,prediction)
        
        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(Y_test,prediction)
        
        return r2,rmse

    except Exception as e:
        
        logging.error("Error in evaluating model: {}".format(e))
        raise e
        
    