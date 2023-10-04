import logging
import pandas as pd
from zenml import step
from src.data_cleaning import DataCleaning,DataDivideStrategy,DataPreProcessingStrategy
from typing import Tuple
from typing_extensions import Annotated


@step
def clean_data(df: pd.DataFrame) -> Tuple[
Annotated[pd.DataFrame,"X_train"],
Annotated[pd.DataFrame,"X_test"],                                          
Annotated[pd.Series,"Y_train"],                                          
Annotated[pd.Series,"Y_test"]]:
    
    """_summary_
    Cleans the data and divides into train and test
    Args:
    df: Raw data
    
    returns:
    X_train : Training data
    X_test: Testing data
    Y_train: Training labels
    Y_test: Testing labels
    
    Raises:
        e: exception
    """
    
    try:
        process_strategy = DataPreProcessingStrategy()
        data_cleaning = DataCleaning(df,process_strategy)
        processed_data = data_cleaning.handle_data()
        
        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(processed_data,divide_strategy)
        X_test,X_train,Y_test,Y_train = data_cleaning.handle_data()
        logging.info("Data cleaning is completed")
        return X_train,X_test,Y_test,Y_train
        
    
    except Exception as e:
        logging.error("Error in cleaning data:{}".format(e))
        raise e
    
    