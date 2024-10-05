# scripts/train.py
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from data_processing import load_data, preprocess_data

def main():
    # Load and preprocess data
    symptom_df, description_df, precaution_df, severity_df, testing_symptoms_df = load_data()
    training_data_cleaned, testing_data_cleaned, classes, all_symptoms = preprocess_data(symptom_df, testing_symptoms_df)
    
    print(f"Training Data Shape: {training_data_cleaned.shape}")
    print(f"Testing Data Shape: {testing_data_cleaned.shape}")

    # Split features and labels
    X_train = training_data_cleaned.drop(columns=['prognosis', 'prognosis_encoded'])
    y_train = training_data_cleaned['prognosis']

    # Encode the target variable
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)

    # Initialize and train the KNN model with k=3
    model = KNeighborsClassifier(n_neighbors=3, weights='uniform', metric='euclidean')
    model.fit(X_train, y_train_encoded)

    # Save the trained model and the label encoder
    joblib.dump(model, 'models/knn_model.pkl')
    joblib.dump(le, 'models/label_encoder.pkl')
    print("Model training and saving completed successfully.")

if __name__ == "__main__":
    main()
