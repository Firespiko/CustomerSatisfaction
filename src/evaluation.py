import logging
from abc import ABC,abstractmethod

import numpy as np
from sklearn.metrics import mean_squared_error,r2_score

class Evaluation(ABC):
    """
    Abstract class defining strategy for evaluation for our models 
    
    """
    @abstractmethod
    def calculate_scores(self,y_true: np.ndarray, y_pred: np.ndarray):
        """
            Abstract method for evaluation scores

        Args:
            y_true (np.ndarray): The given label value
            y_pred (np.ndarray): Predicted label value
            
        """
        pass

class MSE(Evaluation):
    """
    Evaluation strategy using mean squared error method
    
    """
    
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        
        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_true,y_pred)
            logging.info("MSE: {} ".format(mse))
            return mse
        
        except Exception as e:
            logging.error("Error occured in calculating MSE {}".format(e))
            raise e
        
class R2(Evaluation):
    """
    
    Evaluation strategy using R2 score
    
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        
        try:
            logging.info("Calculating R2")
            r2 = r2_score(y_true,y_pred)
            logging.info("R2 Score: {} ".format(r2))
            return r2
        
        except Exception as e:
            logging.error("Error occured in calculating R2 score {}".format(e))
            raise e
        
class RMSE(Evaluation):
    """
    
    Evaluation strategy using RMSE score
    
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        
        try:
            logging.info("Calculating RMSE")
            rmse = mean_squared_error(y_true, y_pred, squared = False)
            logging.info("RMSE Score: {} ".format(rmse))
            return rmse
        
        except Exception as e:
            logging.error("Error occured in calculating RMSE score {}".format(e))
            raise e