# 🚀 Customer Satisfaction Prediction with ZenML

Predict how customers will feel about a product *before* they even order it! This project demonstrates an end-to-end MLOps pipeline using ZenML to predict customer review scores for future orders.

## 🎯 Problem Statement

Leveraging the **Brazilian E-Commerce Public Dataset by Olist**, we aim to predict a customer's review score for their next order or purchase. This enables businesses to proactively address potential dissatisfaction based on order status, price, payment, and other relevant features.

## ✨ Why ZenML?

ZenML provides a robust framework to build and deploy production-ready ML pipelines. This repository showcases:

* **A Solid Framework:** A reusable template for building ML pipelines.
* **MLflow Integration:** Seamless integration with MLflow for experiment tracking, model deployment, and more.
* **Simplified Deployment:** Effortless construction and deployment of continuous ML pipelines.

## 📦 Getting Started

### 🐍 Python Requirements

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/zenml-io/zenml-projects.git](https://github.com/zenml-io/zenml-projects.git)
    cd zenml-projects/customer-satisfaction
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### 📊 ZenML Dashboard & Integrations

1.  **Install ZenML server dependencies:**
    ```bash
    pip install zenml["server"]
    ```
2.  **Launch ZenML UI:**
    ```bash
    zenml up
    ```
3.  **Install MLflow integration:**
    ```bash
    zenml integration install mlflow -y
    ```

### ⚙️ Configure ZenML Stack

This project requires an MLflow experiment tracker and model deployer.

```bash
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-deployer register mlflow --flavor=mlflow
zenml stack register mlflow_stack -a default -o default -d mlflow -e mlflow_tracker --set
```

🚀 (Deployment Pipeline)
Extends the training pipeline for continuous deployment:

* (Same as Training Pipeline)
* `deployment_trigger`: Checks if the new model meets a configurable MSE threshold for deployment.
* `model_deployer`: Deploys the model as a service using MLflow if deployment criteria are met. This automatically updates the running MLflow deployment server with the new model.

ZenML's MLflow integration handles hyperparameter logging, model artifact storage, and evaluation metrics tracking.

🏃‍♀️ Running the Pipelines

* **Training Pipeline:**

    ```bash
    python run_pipeline.py
    ```

* **Continuous Deployment Pipeline:**

    ```bash
    python run_deployment.py
    ```

📈 Demo Streamlit App
A Streamlit application consumes the deployed model to predict customer satisfaction in real-time.

* **Run the demo app:**

    ```bash
    streamlit run streamlit_app.py
    ```
    (A live demo is also available here).

❓ FAQ

* **`No Step found for the name mlflow_deployer`**: Your artifact store might be corrupted.
    * Find its location: `zenml artifact-store describe`
    * **CAUTION: DESTRUCTIVE COMMAND!** Delete it: `rm -rf PATH_TO_ARTIFACT_STORE` and rerun.
* **`No Environment component with name mlflow is currently registered`**: You forgot to install the MLflow integration.
    * Install it: `zenml integration install mlflow -y`
