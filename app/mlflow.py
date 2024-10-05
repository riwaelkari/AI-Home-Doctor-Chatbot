from train_models.neural_network import SymptomDiseaseModel
import mlflow

# Load the pre-trained model
model = SymptomDiseaseModel()
model.load_model('models/saved_model.keras')  # Load your saved model

# Save the model to a local directory
model_save_path = "models/saved_model.keras"
model.model.save(model_save_path)

# Set the tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Start an MLflow run and log the saved model as an artifact
with mlflow.start_run():
    mlflow.log_artifact(model_save_path, artifact_path="model")

