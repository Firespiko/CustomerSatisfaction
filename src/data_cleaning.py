import logging as log
from abc import ABC,abstractmethod
from typing import Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class Datastrategy(ABC):
    
    """Abstract class for defining strategy for handling data
    """
    
    @abstractmethod
    def handle_data(self,data: pd.DataFrame) -> Union[pd.DataFrame,pd.Series]:
        pass
    

class DataPreProcessingStrategy(Datastrategy):
    """Strategy for Preprocessing data

    Args:
        Datastrategy (_type_): _description_
        
    """
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """_summary_
        Preprocessing data
        Args:
            data (pd.DataFrame): _description_

        Returns:
            pd.DataFrame: _description_
        """
        try:
            data = data.drop([
                "order_approved_at",
                "order_delivered_carrier_date",
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
                "order_purchase_timestamp"
            ],axis = 1)
            data["product_weight_g"].fillna(data["product_weight_g"].median(), inplace = True)
            data["product_length_cm"].fillna(data["product_length_cm"].median(), inplace = True)
            data["product_height_cm"].fillna(data["product_height_cm"].median(), inplace = True)
            data["product_width_cm"].fillna(data["product_width_cm"].median(), inplace = True)
            data["review_comment_message"].fillna("No review", inplace = True)
            
            data = data.select_dtypes(include=[np.number])
            cols_to_drop = ["customer_zip_code_prefix","order_item_id"]
            data = data.drop(cols_to_drop,axis=1)
            return data

        except Exception as e:
            log.error("Error in preprocessing data: {}".format(e))
            raise e
        
class DataDivideStrategy(Datastrategy):
    """_summary_
    Divide data into test and train
    """
    
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame,pd.Series]:
        """
        Didvide Data into train and test
        
        """
        try:
            X = data.drop(["review_score"],axis=1)
            Y = data["review_score"]
            X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.2, random_state = 42)
            log.error(X_train.shape)
            log.error(Y_train.shape)
            return X_train,X_test,Y_train,Y_test
        
        except Exception as e:
            log.error("Error in dividing data: {}".format(e))
            raise e

class DataCleaning:
    """_summary_
    Class for cleaning data which processes the data and divides it into train and test

    """
    
    def __init__(self, data: pd.DataFrame, strategy: Datastrategy):
        """
        Initializes the strategy and the data
        """
        self.data = data
        self.strategy = strategy 
        
    def handle_data(self) -> Union[pd.DataFrame , pd.Series]:
        
        """
        Handle data
        """
        try:
            return self.strategy.handle_data(self.data)
        
        except Exception as e:
            log.error("Error in handling data {}".format(e))
            raise e