# train_models/knn_model.py
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
class SymptomDiseaseModel:
    """
    A class to handle disease prediction based on symptoms using a KNN model.

    This class provides methods to load a pre-trained KNN model, encode symptoms, 
    and predict diseases based on a list of provided symptoms. It also handles 
    unrecognized symptoms and returns predictions for the most likely diseases.

    Attributes:
        model (sklearn.neighbors.KNeighborsClassifier): The pre-trained KNN model.
        all_symptoms (list): A list of all possible symptoms used in training the model.
        y_encoded (list): A list of encoded disease labels corresponding to the symptoms.
        label_encoder (LabelEncoder): The encoder used to map disease names to numeric values.
  Methods:
        load_model(model_path, encoder_path): Loads the pre-trained KNN model and label encoder.
        set_additional_attributes(all_symptoms, y_encoded): Sets the list of symptoms and encoded disease labels.
        predict_disease(symptoms_list): Predicts the disease based on the provided list of symptoms.
    """
    def __init__(self):
        self.model = None
        self.all_symptoms = []
        self.y_encoded = []
        self.label_encoder = None

    def load_model(self, model_path='saved_models/knn_model.pkl', encoder_path='saved_models/label_encoder.pkl'):
        """
        Loads the pre-trained KNN model and label encoder from specified paths.

        Args:
            model_path (str): Path to the saved KNN model.
            encoder_path (str): Path to the saved label encoder.

        Returns:
            None
        """
             # Load the trained KNN model and label encoder
        self.model = joblib.load(model_path)
        self.label_encoder = joblib.load(encoder_path)

    def set_additional_attributes(self, all_symptoms, y_encoded):
        """
        Sets additional attributes like the list of symptoms and encoded disease labels.

        Args:
            all_symptoms (list): List of all possible symptoms used for training.
            y_encoded (list): List of encoded disease labels corresponding to symptoms.

        Returns:
            None
        """
        self.all_symptoms = all_symptoms
        self.y_encoded = y_encoded

    def predict_disease(self, symptoms_list):
        """
        Predicts diseases based on a list of symptoms.

        Args:
            symptoms_list (list): List of symptoms, e.g., ["itching", "skin rash"]

        Returns:
            dict: A dictionary containing either the disease confidences or an error message.
        """
        # Normalize symptom names in the dataset
        all_symptoms_normalized = [symptom.strip().lower() for symptom in self.all_symptoms]
        symptom_mapping = dict(zip(all_symptoms_normalized, self.all_symptoms))
        # Initialize input vector
        input_vector = [0] * len(self.all_symptoms)
        # Process input symptoms
        symptoms_normalized = [symptom.strip().lower() for symptom in symptoms_list]
        unrecognized_symptoms = []
         # Update the input vector for each recognized symptom
        for symptom in symptoms_normalized:
            if symptom in symptom_mapping:
                idx = self.all_symptoms.index(symptom_mapping[symptom])
                input_vector[idx] = 1
            else:
                unrecognized_symptoms.append(symptom)
        if unrecognized_symptoms:
            error_message = f"Symptom(s) not recognized: {', '.join(unrecognized_symptoms)}. Please check the symptom names and try again."
            return {"error": error_message}
        # Convert to numpy array and reshape
        input_vector = np.array(input_vector).reshape(1, -1)
        # Find the k nearest neighbors
        neighbors = self.model.kneighbors(input_vector, return_distance=True)
        distances = neighbors[0][0]
        neighbor_indices = neighbors[1][0]
        neighbor_classes = self.model._y[neighbor_indices]
        # Count the occurrences of each class among neighbors
        class_counts = np.bincount(neighbor_classes, minlength=len(self.label_encoder.classes_))
        total_neighbors = class_counts.sum()
        # Calculate confidences for all diseases
        disease_confidences = []
        for cls_idx in np.nonzero(class_counts)[0]: 
            disease_name = self.label_encoder.inverse_transform([cls_idx])[0]
            disease_confidence = class_counts[cls_idx] / total_neighbors
            disease_confidences.append((disease_name, disease_confidence))
        # Sort the diseases by confidence in descending order
        disease_confidences.sort(key=lambda x: x[1], reverse=True)
        return [disease_name for disease_name in disease_confidences]
    
    


    