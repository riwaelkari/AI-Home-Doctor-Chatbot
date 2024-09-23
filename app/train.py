# scripts/train.py

import pickle
import numpy as np
from models.neural_network import SymptomDiseaseModel
from data_processing import load_data, preprocess_data

def main():
    # Load and preprocess data
    symptom_df, description_df, precaution_df, severity_df, testing_symptoms_df = load_data()
    training_data_cleaned, testing_data_cleaned, classes, all_symptoms = preprocess_data(symptom_df, testing_symptoms_df)
    
    print(f"Training Data Shape: {training_data_cleaned.shape}")
    print(f"Testing Data Shape: {testing_data_cleaned.shape}")

    # Split features and labels
    X_train = training_data_cleaned.drop(columns=['prognosis', 'prognosis_encoded'])
    y_train = training_data_cleaned['prognosis_encoded']
    X_test = testing_data_cleaned.drop(columns=['prognosis', 'prognosis_encoded'])
    y_test = testing_data_cleaned['prognosis_encoded']

    # Initialize and train the model
    model = SymptomDiseaseModel(y_train)
    model.train(X_train, y_train)
    model.evaluate_model(X_test, y_test)
    model.save_model('models/saved_model.h5')
    
    print("Model training and saving completed successfully.")

    def getpar():
        return classes, all_symptoms
if __name__ == "__main__":
    main()
