# scripts/train.py
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from train_models.neural_network import SymptomDiseaseModel
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


    # Make sure y_train and y_test are one-hot encoded
    num_classes = len(np.unique(y_train))  # Number of unique diseases
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)

    # Initialize and train the model
    model = SymptomDiseaseModel()
    model.train(X_train, y_train)
    #model.evaluate_model(X_test, y_test)
    model.save_model('models/saved_model.keras')
    print("Model training and saving completed successfully.")
if __name__ == "__main__":
    main()