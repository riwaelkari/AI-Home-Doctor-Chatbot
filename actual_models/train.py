# scripts/train.py
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from chatbot.data_processing import load_data, preprocess_data

def main():
    # Load and preprocess data
    symptom_df, description_df, precaution_df, severity_df, testing_symptoms_df = load_data()
    training_data_cleaned, testing_data_cleaned, classes, all_symptoms, le = preprocess_data(symptom_df, testing_symptoms_df)
    
    print(f"Training Data Shape: {training_data_cleaned.shape}")
    print(f"Testing Data Shape: {testing_data_cleaned.shape}")

    training_data_cleaned = training_data_cleaned.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split features and labels
    X_train = training_data_cleaned.drop(columns=['prognosis', 'prognosis_encoded'])
    y_train = training_data_cleaned['prognosis']

    # Encode the target variable
    # Initialize and train the KNN model with k=3
    model = KNeighborsClassifier(n_neighbors=3, weights='uniform', metric='euclidean')
    model.fit(X_train, y_train)

    # Save the trained model and the label encoder
    joblib.dump(model, 'saved_models/knn_model.pkl')
    joblib.dump(le, 'saved_models/label_encoder.pkl')
    print("Model training and saving completed successfully.")

if __name__ == "__main__":
    main()
