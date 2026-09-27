import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("MRI_Dementia_Classification")

with mlflow.start_run(run_name="MLflow_Setup_Test"):
    mlflow.log_param("model", "test_model")
    mlflow.log_param("seed", 42)
    mlflow.log_param("batch_size", 32)

    mlflow.log_metric("accuracy", 0.85)
    mlflow.log_metric("macro_f1", 0.82)

    print("MLflow test completed!")
