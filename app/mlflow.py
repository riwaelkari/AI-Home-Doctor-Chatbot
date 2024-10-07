# log_model_mlflow.py

import mlflow
import mlflow.sklearn  # Adjust if your model is not scikit-learn
from symptom_disease_model import SymptomDiseaseModel
import joblib
import os
import json

def log_symptom_disease_model():
    # Initialize the model
    symptom_model = SymptomDiseaseModel()

    # Load the model and label encoder
    symptom_model.load_model()

    # If you have additional attributes, set them here
    # For example, load 'all_symptoms' from a file
    all_symptoms_path = 'models/all_symptoms.json'
    if os.path.exists(all_symptoms_path):
        with open(all_symptoms_path, 'r') as f:
            all_symptoms = json.load(f)
        symptom_model.set_additional_attributes(all_symptoms, y_encoded=None)

    # Start an MLflow run
    with mlflow.start_run() as run:
        # Log the scikit-learn model
        mlflow.sklearn.log_model(
            sk_model=symptom_model.model,
            artifact_path="SymptomDiseaseModel",
            registered_model_name="SymptomDiseaseModel"  # Optional: Registers the model
        )

        # Log the label encoder as an artifact
        encoder_path = 'models/label_encoder.pkl'
        mlflow.log_artifact(encoder_path, artifact_path="SymptomDiseaseModel")

        # Log 'all_symptoms' if available
        if symptom_model.all_symptoms:
            symptoms_path = 'models/all_symptoms.json'
            # Ensure 'all_symptoms.json' is saved
            with open(symptoms_path, 'w') as f:
                json.dump(symptom_model.all_symptoms, f)
            mlflow.log_artifact(symptoms_path, artifact_path="SymptomDiseaseModel")

        # Log parameters (e.g., number of neighbors 'k')
        if hasattr(symptom_model.model, 'n_neighbors'):
            mlflow.log_param("n_neighbors", symptom_model.model.n_neighbors)

        # Log any metrics if available
        # mlflow.log_metric("metric_name", metric_value)

        # Print out the run ID for reference
        print(f"Model logged under run ID: {run.info.run_id}")

if __name__ == "__main__":
    log_symptom_disease_model()

'''
from train_models.symptom_disease_model import SymptomDiseaseModel
import mlflow

# Load the pre-trained model
model = SymptomDiseaseModel()
model.load_model('models/knn_model.pkl')  # Load your saved model

# Save the model to a local directory
model_save_path = 'models/knn_model.pkl'
model.model.save(model_save_path)

# Set the tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5001")

# Start an MLflow run and log the saved model as an artifact
with mlflow.start_run():
    mlflow.log_artifact(model_save_path, artifact_path="model")

'''