import sys
import mlflow

def delete_run(run_id: str, tracking_uri: str = "https://mentis.io/mlflow/"):
    mlflow.tracking.MlflowClient(tracking_uri=tracking_uri).delete_run(run_id)

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python delete_run.py <run_id> <tracking_uri>")
        sys.exit(1)
    
    run_id = sys.argv[1]

    tracking_uri = "https://mentis.io/mlflow/" if len(sys.argv) < 3 else sys.argv[2]

    delete_run(run_id, tracking_uri)