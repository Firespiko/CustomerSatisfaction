import logging

import pandas as pd
from zenml import step

class Ingestdata:
    
    """"
    Ingesting the data from the data path
    """
    
    def __init__(self,data_path: str):
        """_summary_
        Initializes the data_path object

        Args:
            data_path (str): 
        """
        self.data_path = data_path
        
    def get_data(self):
        """_summary_
        Ingests the data from the data path
        Returns:
            _type_: pd.Dataframe
        """
        
        logging.info(f"Ingest data from {self.data_path}")
        return pd.read_csv(self.data_path)
    
@step
def ingest_data(data_path: str) -> pd.DataFrame:
    """_summary_
    Ingesting the data from the data path
    
    Args:
        data_path (str): the data path of the data file

    Returns:
        pd.DataFrame: returns the datframe from the respective datapath
    """
    try:
        ingest_data = Ingestdata(data_path)
        df = ingest_data.get_data()
        logging.info("Ingesting data completed.")
        return df
    except Exception as e:
        logging.error("Error while ingesting data")
        raise e  
    