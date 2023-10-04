from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.evaluation import evaluate_model
from steps.model_train import train_model


@pipeline(enable_cache= False)
def train_pipeline(data_path: str) -> None:
    df = ingest_data(data_path)
    X_train,X_test,Y_test,Y_train, = clean_data(df)
    model = train_model(X_train,Y_test)
    r2_score,rmse = evaluate_model(model,X_test,Y_test)
    