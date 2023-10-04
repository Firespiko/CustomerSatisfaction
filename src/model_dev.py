import logging 
from abc import ABC,abstractmethod
from sklearn.linear_model import LinearRegression 


class Model(ABC):

    """
    Abstract class for all models
    """
    
    @abstractmethod
    def train(self, X_train, Y_train):
        """_summary_
        Trains the model
        
        Args:
            X_train (pd.Dataframe): Training data
            Y_train (pd.Dataframe): Training label
            
        Return:
        None 
        
        """
        pass
    
        
class LinearRegressionModel(Model):
    
    """
    Class for linear regression model
    """
    
    def train(self, X_train, Y_train, **kwargs):
        """
                _summary_
        Trains the model
        
        Args:
            X_train (pd.Dataframe): Training data
            Y_train (pd.Dataframe): Training label
            
        Return:
        None 
        
        """
        try:
            reg = LinearRegression(**kwargs)
            print(X_train.shape)
            print(Y_train.shape)
            reg.fit(X_train,Y_train)
            logging.info("Model Training Completed")
            return reg
        except Exception as e:
            logging.info("Error occured in model training {}".format(e))
            raise e
        
        
    